"""Fusing typed decisions into a signal, and sizing that signal.

Everything between "the engine answered seven questions" and "the risk governor
sees a target weight" happens here, and it is written as a formula you can read
rather than a fitted model you cannot.

The fusion is deliberately *not* a learned combination. A learned combiner would
be fitted on the same data the engine is evaluated on, and then the evaluation
would be measuring the combiner. Instead the weights below are policy — declared
in config, overridable, and small enough to reason about:

    core      = base + w_materiality·P(tradeable) + w_conviction·conviction
                     + w_context·(P(aligned) - ½)
    magnitude = core / Σ(positive terms)                     → 0..1
    discount  = (1 - w_priced_in·(1 - priced_in/3)) · (1 - w_risk_event·P(tail))
    score     = sign(direction) · magnitude · discount · horizon_scale

Read the middle line out loud and it says what a good trader says: *how much of
this is new, and how much of it is a landmine.* Those two factors can only ever
reduce a signal. The engine never gets to argue its way past "this was already
priced in".
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .text import clip, safe_div
from .types import Decision, Instrument, MarketSnapshot, NewsItem, Signal

DECISION_KEYS = ("direction", "materiality", "conviction", "priced_in",
                 "context_aligned", "horizon", "risk_event")


class SignalBuilder:
    """Aggregates a symbol's live decisions into one signed score."""

    def __init__(self, cfg=None) -> None:
        s = (cfg.section("signals") if cfg else {})
        self.w_materiality = float(s.get("w_materiality", 0.75))
        self.w_conviction = float(s.get("w_conviction", 0.65))
        self.w_context = float(s.get("w_context", 0.45))
        self.w_priced_in = float(s.get("w_priced_in", 0.55))
        self.w_risk_event = float(s.get("w_risk_event", 0.70))
        self.base = float(s.get("base", 0.20))
        self.aggregation = str(s.get("aggregation", "decay_sum"))
        self.clip_value = float(s.get("clip", 1.0))
        self.horizon_scale: dict[str, float] = dict(
            s.get("horizon", {"intraday": 0.35, "swing": 0.75, "positional": 1.0})
        )
        self.half_life: dict[str, float] = dict(
            s.get("decay_half_life", {"intraday": 1800, "swing": 21600, "positional": 86400})
        )
        self.min_materiality = float(s.get("min_materiality", 0.30))

    # ── one decision ───────────────────────────────────────────────────────

    @property
    def positive_denominator(self) -> float:
        return self.base + self.w_materiality + self.w_conviction + self.w_context * 0.5

    def fuse(self, d: Decision) -> float:
        """The signed score a single decision contributes, in -1..1.

        An abstention is *not* a short. It is nothing at all: the answer was not
        confident enough to act on, and turning that into a directional bet
        would invert the meaning of the gate. Materiality below the configured
        floor works the same way.
        """
        if d.abstained or d.materiality < self.min_materiality:
            return 0.0
        sign = d.direction_sign
        if sign == 0:
            return 0.0

        core = (
            self.base
            + self.w_materiality * d.materiality
            + self.w_conviction * d.conviction_norm
            + self.w_context * (d.context_aligned - 0.5)
        )
        magnitude = clip(core / self.positive_denominator, 0.0, 1.0)

        magnitude *= 1.0 - self.w_priced_in * (1.0 - d.priced_in_factor)
        magnitude *= 1.0 - self.w_risk_event * d.risk_event
        magnitude *= self.horizon_scale.get(d.horizon, 0.75)

        return clip(sign * magnitude, -self.clip_value, self.clip_value)

    def decay_weight(self, d: Decision, now: float) -> float:
        hl = max(1.0, self.half_life.get(d.horizon, 21600.0))
        age = max(0.0, now - d.ts)
        return float(0.5 ** (age / hl))

    # ── many decisions ─────────────────────────────────────────────────────

    def aggregate(
        self,
        decisions: Iterable[Decision],
        now: float,
        instruments: Sequence[Instrument] = (),
    ) -> dict[str, Signal]:
        """One signal per symbol that has something to say."""
        buckets: dict[str, list[tuple[Decision, float, float]]] = {}
        for d in decisions:
            score = self.fuse(d)
            if score == 0.0:
                continue
            buckets.setdefault(d.symbol, []).append((d, score, self.decay_weight(d, now)))

        out: dict[str, Signal] = {}
        for symbol, rows in buckets.items():
            if self.aggregation == "max":
                top = max(rows, key=lambda r: abs(r[1]))
                score = top[1]
            elif self.aggregation == "mean":
                score = sum(r[1] * r[2] for r in rows) / safe_div(sum(r[2] for r in rows), 1.0, 1.0)
            else:  # decay_sum
                score = sum(r[1] * r[2] for r in rows)
            score = clip(score, -self.clip_value, self.clip_value)

            weights = [r[2] for r in rows]
            horizon = "swing"
            if weights:
                horizon = rows[max(range(len(rows)), key=lambda i: weights[i])][0].horizon
            out[symbol] = Signal(
                symbol=symbol,
                ts=now,
                score=score,
                target_weight=0.0,             # filled in by the sizer
                direction="long" if score > 0 else "short" if score < 0 else "flat",
                horizon=horizon,
                contributors=tuple(sorted({c for d, _, _ in rows for c in self._contributors(d)})),
                decisions=tuple(d.news_id for d, _, _ in rows),
                notes=(
                    f"{len(rows)} decision(s), "
                    f"newest {min(d.ts for d, _, _ in rows):.0f}..{max(d.ts for d, _, _ in rows):.0f}"
                ),
            )
        return out

    @staticmethod
    def _contributors(d: Decision) -> list[str]:
        out = []
        if d.materiality >= 0.5:
            out.append("material")
        if d.conviction_norm >= 0.5:
            out.append("high-conviction")
        if d.priced_in <= 1.0:
            out.append("priced-in-discount")
        if d.risk_event >= 0.5:
            out.append("risk-event")
        if d.context_aligned >= 0.5:
            out.append("with-trend")
        return out


class VolatilitySizer:
    """Volatility targeting: same conviction, same risk, different notional.

    1% of equity in Bitcoin and 1% of equity in a utilities ETF are not the same
    trade. The sizer scales the signal's target weight so that a symbol's
    contribution lands near the configured annualised volatility target, and it
    only ever *reduces* the position — a quiet instrument does not get leveraged
    up to the target, it just gets its full signal weight.
    """

    def __init__(self, cfg=None) -> None:
        r = (cfg.section("risk") if cfg else {})
        self.vol_target = float(r.get("vol_target_annual", 0.25))
        self.vol_floor = float(r.get("vol_floor_annual", 0.08))

    @staticmethod
    def annual_vol(snap: MarketSnapshot) -> float:
        """Hourly realised vol -> annualised, assuming ~24 traded hours/day."""
        return max(0.0, snap.vol_24h) * math.sqrt(24.0 * 365.0)

    def scalar(self, snap: MarketSnapshot) -> float:
        vol = self.annual_vol(snap)
        if vol <= 0:
            return 0.0
        return clip(self.vol_target / max(vol, self.vol_floor), 0.0, 1.0)


def size_signals(
    signals: dict[str, Signal],
    snapshots: dict[str, MarketSnapshot],
    cfg=None,
    max_weight: float | None = None,
) -> dict[str, Signal]:
    """Attach a target weight to every signal, in place, and return them."""
    sizer = VolatilitySizer(cfg)
    cap = float(max_weight if max_weight is not None else (
        cfg.get("risk", "max_weight_per_symbol", default=0.25) if cfg else 0.25))
    for symbol, sig in signals.items():
        snap = snapshots.get(symbol)
        if snap is None:
            sig.target_weight = 0.0
            sig.notes += " | no market snapshot, not sized"
            continue
        sig.target_weight = clip(sig.score, -1.0, 1.0) * cap * sizer.scalar(snap)
    return signals


def relevant_news(
    news: list[NewsItem], symbol: str, window_seconds: float, now: float
) -> list[NewsItem]:
    """News tagged to a symbol (or market-wide) inside the recency window."""
    out = []
    for item in news:
        if now - item.ts > window_seconds:
            continue
        if not item.symbols or symbol in item.symbols:
            out.append(item)
    return out
