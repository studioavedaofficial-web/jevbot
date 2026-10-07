"""The risk governor.

The signal layer proposes; this layer disposes. Nothing reaches a broker without
passing through :meth:`RiskGovernor.evaluate`, and the governor's job is to be
the least interesting code in the repository: no model, no probabilities, just
limits enforced in a fixed, documented order, every one of them able to explain
itself in a log line.

Enforcement order matters and is fixed:

1. **Kill switches** (daily loss, max drawdown) — flatten first, ask later.
2. **Data freshness** — never trade a symbol whose tape is stale.
3. **Per-symbol cap** — with the volatility sizer already applied.
4. **Signal floor** — forget positions too small to be worth the fees.
5. **Portfolio caps** — gross, net, position count, cash buffer.
6. **Turnover cap** — a news burst cannot turn the book over in one cycle.

Steps 5 and 6 resolve competition between symbols by *strength of signal*, not
by iteration order, because dict order is not a risk policy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .text import clip
from .types import MarketSnapshot, Signal

log = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    starting_equity: float = 100_000.0
    max_weight_per_symbol: float = 0.25
    max_gross_weight: float = 1.0
    max_net_weight: float = 0.6
    max_positions: int = 8
    max_trades_per_cycle: int = 4
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.20
    min_cash_buffer: float = 0.02
    stale_data_seconds: float = 900.0
    min_order_notional: float = 25.0
    cooldown_seconds_after_stop: float = 1800.0
    max_turnover_per_cycle: float = 0.5
    min_signal: float = 0.05
    hold_signal_ratio: float = 0.5
    stop_atr_multiple: float = 3.0
    vol_target_annual: float = 0.25
    vol_floor_annual: float = 0.08

    @classmethod
    def from_config(cls, cfg) -> RiskLimits:
        section = cfg.section("risk") if cfg else {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in section.items() if k in known})


@dataclass
class RiskVerdict:
    approved: dict[str, float] = field(default_factory=dict)   # symbol -> target weight
    dropped: dict[str, str] = field(default_factory=dict)      # symbol -> reason
    flatten: bool = False
    halted: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": {k: round(v, 5) for k, v in self.approved.items()},
            "dropped": self.dropped,
            "flatten": self.flatten,
            "halted": self.halted,
            "notes": self.notes,
        }


class RiskGovernor:
    def __init__(self, limits: RiskLimits, starting_equity: float | None = None) -> None:
        self.limits = limits
        self.day = self._utc_day()
        self.day_start_equity = float(starting_equity if starting_equity is not None else limits.starting_equity)
        # The high-water mark starts at the account's starting equity, not at
        # the first equity this process happens to see. A governor that only
        # remembers peaks it personally witnessed cannot fire on its first
        # cycle — so a bot restarted into a crash would trade straight through
        # a drawdown it should have killed.
        self._peak = self.day_start_equity
        self.killed = False
        self.kill_reason = ""
        self.halted_today = False
        self.cooldowns: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []

    # ── bookkeeping ────────────────────────────────────────────────────────

    @staticmethod
    def _utc_day(ts: float | None = None) -> str:
        return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).strftime("%Y-%m-%d")

    def roll_day(self, equity: float, now: float | None = None) -> None:
        """Reset daily counters at the UTC boundary."""
        today = self._utc_day(now)
        if today != self.day:
            self.day = today
            self.day_start_equity = equity
            self.halted_today = False
            self.events.append({"ts": now or time.time(), "event": "day_roll", "equity": equity})

    def daily_pnl_pct(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return equity / self.day_start_equity - 1.0

    def cooldown_active(self, symbol: str, now: float) -> bool:
        until = self.cooldowns.get(symbol, 0.0)
        return until > now

    def note_stop(self, symbol: str, now: float) -> None:
        """A stop-out puts the symbol in the penalty box."""
        self.cooldowns[symbol] = now + self.limits.cooldown_seconds_after_stop

    def reset_kill(self) -> None:
        self.killed = False
        self.kill_reason = ""
        self.events.append({"ts": time.time(), "event": "kill_reset"})

    # ── the gate ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        proposed: dict[str, Signal],
        snapshots: dict[str, MarketSnapshot],
        equity: float,
        current_weights: dict[str, float],
        now: float,
    ) -> RiskVerdict:
        L = self.limits
        v = RiskVerdict()
        self.roll_day(equity, now)

        # 1 ── kill switches
        dd = 0.0
        peak = max(equity, getattr(self, "_peak", equity))
        self._peak = peak
        if peak > 0:
            dd = max(0.0, 1.0 - equity / peak)
        if dd >= L.max_drawdown_pct and not self.killed:
            self.killed = True
            self.kill_reason = f"max drawdown {dd:.1%} >= {L.max_drawdown_pct:.1%}"
            self.events.append({"ts": now, "event": "kill", "reason": self.kill_reason})
            log.error("RISK KILL SWITCH: %s", self.kill_reason)
        if self.killed:
            v.halted = True
            v.flatten = True
            v.notes.append(f"kill switch active: {self.kill_reason}")
            return v

        pnl_today = self.daily_pnl_pct(equity)
        if pnl_today <= -abs(L.daily_loss_limit_pct):
            if not self.halted_today:
                self.halted_today = True
                self.events.append({"ts": now, "event": "daily_halt", "pnl": pnl_today})
                log.warning("daily loss limit hit (%.2f%%) — flattening and standing down", pnl_today * 100)
            v.halted = True
            v.flatten = True
            v.notes.append(f"daily loss limit: {pnl_today:.2%}")
            return v

        # 2 ── data freshness, and the cooldown after a stop-out
        ranked: list[tuple[float, str, float]] = []
        for symbol, sig in proposed.items():
            snap = snapshots.get(symbol)
            if snap is None:
                v.dropped[symbol] = "no snapshot"
                continue
            age = now - snap.ts
            if age > L.stale_data_seconds:
                v.dropped[symbol] = f"stale data ({age:.0f}s)"
                continue
            if self.cooldown_active(symbol, now):
                v.dropped[symbol] = "cooldown after stop"
                continue
            # Hysteresis: the floor to *open* a position is higher than the
            # floor to *keep* one. A single threshold makes a signal hovering
            # around it oscillate between in and out, paying the round trip
            # every time to express no change of opinion.
            held = abs(current_weights.get(symbol, 0.0)) > 1e-6
            floor = L.min_signal * (L.hold_signal_ratio if held else 1.0)
            if abs(sig.score) < floor:
                v.dropped[symbol] = f"signal below floor ({sig.score:+.3f} < {floor:.3f})"
                continue
            ranked.append((abs(sig.score), symbol, sig.target_weight))

        # 3/4 ── per-symbol cap, strongest signals first
        ranked.sort(key=lambda r: -r[0])
        for _strength, symbol, weight in ranked:
            v.approved[symbol] = clip(weight, -L.max_weight_per_symbol, L.max_weight_per_symbol)

        # 5 ── portfolio caps: keep the strongest, drop the rest
        kept: list[tuple[float, str, float]] = []
        gross = 0.0
        net = 0.0
        for strength, symbol, _w in ranked:
            weight = v.approved[symbol]
            if len(kept) >= L.max_positions:
                v.dropped[symbol] = "position count limit"
                v.approved.pop(symbol, None)
                continue
            if gross + abs(weight) > L.max_gross_weight + 1e-9:
                v.dropped[symbol] = "gross exposure limit"
                v.approved.pop(symbol, None)
                continue
            new_net = net + weight
            if abs(new_net) > L.max_net_weight + 1e-9:
                # trim toward the cap rather than dropping a good signal outright
                room = (L.max_net_weight if new_net > 0 else -L.max_net_weight) - net
                weight = clip(room, -L.max_weight_per_symbol, L.max_weight_per_symbol)
                v.notes.append(f"{symbol} trimmed to {weight:+.2%} by net exposure limit")
                if abs(weight) < 1e-6:
                    v.dropped[symbol] = "net exposure limit"
                    v.approved.pop(symbol, None)
                    continue
            gross += abs(weight)
            net += weight
            v.approved[symbol] = weight
            kept.append((strength, symbol, weight))

        # cash buffer: clip longs so the book never spends the buffer
        long_total = sum(w for w in v.approved.values() if w > 0)
        max_long = max(0.0, 1.0 - L.min_cash_buffer)
        if long_total > max_long:
            scale = max_long / long_total
            for symbol, w in list(v.approved.items()):
                if w > 0:
                    v.approved[symbol] = w * scale
            v.notes.append(f"long exposure scaled {scale:.2f}x to hold the cash buffer")

        # 6 ── turnover cap: scale the whole change down, preserving direction
        turnover = sum(abs(v.approved.get(s, 0.0) - current_weights.get(s, 0.0))
                       for s in set(v.approved) | set(current_weights))
        if turnover > L.max_turnover_per_cycle > 0:
            scale = L.max_turnover_per_cycle / turnover
            for symbol in set(v.approved) | set(current_weights):
                target = v.approved.get(symbol, 0.0)
                current = current_weights.get(symbol, 0.0)
                blended = current + (target - current) * scale
                if abs(blended) < 1e-6:
                    v.approved.pop(symbol, None)
                else:
                    v.approved[symbol] = blended
            v.notes.append(f"turnover {turnover:.2f}x capped to {L.max_turnover_per_cycle:.2f}x "
                           f"(moves scaled {scale:.2f})")

        # symbols currently held but with no proposal are exits by omission
        for symbol in current_weights:
            if symbol not in v.approved and abs(current_weights[symbol]) > 1e-6:
                if symbol not in v.dropped:
                    v.dropped[symbol] = "no signal, position to be closed"
        return v

    # ── reporting ──────────────────────────────────────────────────────────

    def status(self, equity: float, now: float | None = None) -> dict[str, Any]:
        now = now or time.time()
        return {
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "halted_today": self.halted_today,
            "day": self.day,
            "day_start_equity": round(self.day_start_equity, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct(equity), 5),
            "daily_loss_limit_pct": self.limits.daily_loss_limit_pct,
            "max_drawdown_pct": self.limits.max_drawdown_pct,
            "cooldowns": {k: round(v - now, 1) for k, v in self.cooldowns.items() if v > now},
            "events": self.events[-10:],
            "limits": {f: getattr(self.limits, f) for f in self.limits.__dataclass_fields__},
        }
