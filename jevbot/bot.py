"""The trading loop.

One cycle, in order, and the order is the design:

1. **Read the tape** for every instrument in the universe.
2. **Read the text** — new headlines only — and assign each one a subject.
3. **Decide.** Each (headline, symbol) pair becomes one rendered state: the
   market context, the recent news, and the headline under consideration. The
   engine answers the seven typed questions for all of them in one batched call.
4. **Fuse and size.** Typed answers become one signed score per symbol, decayed
   by horizon, and the volatility sizer turns the score into a target weight.
5. **Check stops.** A protective stop removes the symbol from proposals before
   the risk governor ever sees it, and puts it in a cooldown.
6. **Govern.** The risk engine approves, trims or rejects weights, and can halt
   the whole book.
7. **Execute.** Weights become orders (exits first), orders become fills.
8. **Record.** Everything above lands in SQLite, keyed so a fill can be traced
   back to the headline that caused it.

The loop holds no strategy logic of its own — that lives in the engine, the
signal builder and the risk governor. What it holds is the *sequence*, because
sequencing is where a trading system usually loses money.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .brokers import build_broker
from .config import Config
from .engine import build_engine, engine_summary
from .engine.base import DecisionEngine
from .execution import (close_orders, flatten_orders, one_order_per_symbol, plan_orders,
                        update_stop_price)
from .feeds import build_news_feed, build_price_feed
from .feeds.base import NewsFeed, PriceFeed
from .portfolio import Portfolio
from .risk import RiskGovernor, RiskLimits
from .routing import NewsRouter
from .signals import SignalBuilder, relevant_news, size_signals
from .store import Store
from .text import render_state
from .types import Decision, Instrument, MarketSnapshot, NewsItem, Order, Signal

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    ts: float
    cycle: int
    equity: float
    decisions: int = 0
    signals: int = 0
    orders: int = 0
    fills: int = 0
    news: int = 0
    halted: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "cycle": self.cycle,
            "equity": round(self.equity, 2),
            "decisions": self.decisions,
            "signals": self.signals,
            "orders": self.orders,
            "fills": self.fills,
            "news": self.news,
            "halted": self.halted,
            "notes": self.notes,
        }


class TradingBot:
    def __init__(
        self,
        cfg: Config,
        *,
        engine: DecisionEngine | None = None,
        price_feed: PriceFeed | None = None,
        news_feed: NewsFeed | None = None,
        broker=None,
        store: Store | None = None,
        speed: float = 1.0,
        portfolio: Portfolio | None = None,
    ) -> None:
        self.cfg = cfg
        self.speed = speed
        self.instruments: list[Instrument] = cfg.instruments
        self.symbols = [i.symbol for i in self.instruments]
        self.instrument_by_symbol = {i.symbol: i for i in self.instruments}

        risk_limits = RiskLimits.from_config(cfg)
        self.portfolio = portfolio if portfolio is not None else Portfolio.fresh(risk_limits.starting_equity)
        self.risk = RiskGovernor(risk_limits, starting_equity=self.portfolio.starting_equity
                                 or risk_limits.starting_equity)

        # Feeds share one simulated clock when they come from the demo market.
        self.price_feed = price_feed or build_price_feed(cfg, speed=speed)
        self.news_feed = news_feed or build_news_feed(cfg, price_feed=self.price_feed)

        self.engine = engine or build_engine(
            cfg.engine_name,
            repo=str(cfg.get("engine", "laya_repo", default="convaiinnovations/laya")),
            device=str(cfg.get("engine", "device", default="") or "") or None,
            preload=bool(cfg.get("engine", "preload", default=False)),
            min_confidence=float(cfg.get("engine", "min_confidence", default=0.0) or 0.0),
            cache=bool(cfg.get("engine", "cache", default=True)),
            max_len=None,
        )
        self.router = NewsRouter(
            self.instruments,
            mode=str(cfg.get("feeds", "news_routing", default="hybrid")),
            engine=self.engine,
        )
        self.builder = SignalBuilder(cfg)
        if broker is not None:
            # A broker holding a different Portfolio fills into a book nothing
            # else can read: the bot would report a flat account while orders
            # were being booked somewhere else, and every fill would be
            # invisible to the risk governor and the dashboard. Fail loudly.
            bound = getattr(broker, "portfolio", None)
            if bound is not None and bound is not self.portfolio:
                raise ValueError(
                    "the supplied broker is bound to a different Portfolio than the bot's; "
                    "pass `portfolio=` as well so both share one book"
                )
            self.broker = broker
        else:
            self.broker = build_broker(cfg, self.portfolio)
        self.store = store

        # loop state
        self.cycle_count = 0
        self.news_buffer: list[NewsItem] = []
        self.news_window = float(cfg.get("signals", "news_window_seconds", default=43200))
        self.refresh_seconds = float(cfg.get("signals", "refresh_seconds", default=900))
        self._decisions: dict[tuple[str, str], Decision] = {}
        self._decided_at: dict[tuple[str, str], float] = {}
        self.clock_fn: Callable[[], float] | None = None
        self._last_cycle_ts = 0.0
        # control surface, written by the dashboard thread and read by the loop
        self.paused = False
        self.stop_requested = False
        self.flatten_requested = False
        self._risk_events_recorded = 0
        self.events: list[dict[str, Any]] = []
        self.latest_snapshots: dict[str, MarketSnapshot] = {}
        self.latest_signals: dict[str, Signal] = {}
        self.last_verdict: dict[str, Any] = {}

    # ── clock ──────────────────────────────────────────────────────────────

    def now(self) -> float:
        """Current time: the backtester's clock if one is installed, else the feeds'.

        The backtester drives the *same* ``cycle()`` with time handed to it, which
        is what makes a backtest a statement about the live code rather than about
        a reimplementation of it.
        """
        if self.clock_fn is not None:
            return float(self.clock_fn())
        for feed in (self.news_feed, self.price_feed):
            sim = getattr(feed, "sim_now", None)
            if callable(sim):
                return float(sim())
        return time.time()

    def _log(self, kind: str, message: str, **extra: Any) -> None:
        entry = {"ts": self.now(), "kind": kind, "message": message, **extra}
        self.events.append(entry)
        self.events = self.events[-400:]
        log.info("[%s] %s", kind, message)

    # ── the cycle ──────────────────────────────────────────────────────────

    def cycle(self) -> CycleResult:
        cfg = self.cfg
        now = self.now()
        result = CycleResult(ts=now, cycle=self.cycle_count + 1, equity=self.portfolio.equity())

        # 1 ── tape
        snapshots = self.price_feed.snapshots(self.symbols, now)
        if not snapshots:
            self._log("warn", "no market snapshots this cycle; skipping")
            return result
        self.latest_snapshots = snapshots

        # 2 ── text
        fresh = self.news_feed.poll(now)
        routed = self.router.route_many(fresh)
        for item, symbols, reason in routed:
            item.symbols = tuple(symbols)
            self.news_buffer.append(item)
            if item.symbols:
                self._log("news", f"{reason} → {', '.join(symbols)}", text=item.text[:180],
                          source=item.source, news_id=item.news_id)
        cutoff = now - self.news_window
        self.news_buffer = [n for n in self.news_buffer if n.ts >= cutoff]
        result.news = len(routed)
        if self.store and routed:
            self.store.record_news(routed)

        # 3 ── decide
        requests = self._pending_requests(snapshots, now)
        decisions: list[Decision] = []
        if requests:
            t0 = time.perf_counter()
            produced = self.engine.decide_batch(requests)
            elapsed = (time.perf_counter() - t0) * 1000.0
            for decision, (state, symbol, news_id, ts) in zip(produced, requests):
                key = (news_id, symbol)
                self._decisions[key] = decision
                self._decided_at[key] = now
                decisions.append(decision)
            if decisions:
                models = {d.model for d in decisions if d.model}
                self._log(
                    "decide",
                    f"{len(decisions)} decision(s) in {elapsed:.0f} ms"
                    + (f" via {', '.join(sorted(models))}" if models else ""),
                    engine=self.engine.name,
                )
        result.decisions = len(decisions)
        if self.store and decisions:
            self.store.record_decisions(decisions)

        # live decisions still inside the news window
        live = self._live_decisions(now)

        # 4 ── fuse and size
        signals = self.builder.aggregate(live, now, self.instruments)
        size_signals(signals, snapshots, cfg, max_weight=self.risk.limits.max_weight_per_symbol)
        self.latest_signals = signals
        result.signals = len(signals)
        if self.store and signals:
            self.store.record_signals(signals.values())

        # 5 ── stops before approval
        stopped = self._stops_hit(snapshots)
        for symbol, why in stopped:
            signals.pop(symbol, None)
            self.risk.note_stop(symbol, now)
            self._log("stop", f"{symbol}: {why}")

        # 6 ── risk governor
        verdict = self.risk.evaluate(
            signals, snapshots, self.portfolio.equity(), self.portfolio.weights(), now
        )
        self.last_verdict = verdict.to_dict()
        result.halted = verdict.halted
        result.notes = verdict.notes
        for note in verdict.notes:
            self._log("risk", note)
        for symbol, reason in verdict.dropped.items():
            if reason not in {"no signal, position to be closed"}:
                self._log("risk-drop", f"{symbol}: {reason}")

        # 7 ── execute
        if verdict.flatten:
            orders = flatten_orders(self.portfolio, ts=now)
            orders += close_orders(self.portfolio, [s for s, _ in stopped], ts=now)
        else:
            orders = close_orders(self.portfolio, [s for s, _ in stopped], ts=now)
            approved = verdict.approved
            orders += plan_orders(
                self.portfolio,
                approved,
                snapshots,
                min_order_notional=self.risk.limits.min_order_notional,
                max_trades=self.risk.limits.max_trades_per_cycle,
                lot_step=float(cfg.get("broker", "lot_step", default=1e-6)),
                allow_fractional=bool(cfg.get("broker", "allow_fractional", default=True)),
                allow_short=self.broker.can_short(),
                half_spread_bps=float(cfg.get("broker", "half_spread_bps", default=1.5)),
                fee_bps=float(cfg.get("broker", "fee_bps", default=5.0)),
                rebalance_tolerance=float(cfg.get("broker", "rebalance_tolerance", default=0.01)),
                rebalance_tolerance_relative=float(
                    cfg.get("broker", "rebalance_tolerance_relative", default=0.25)
                ),
                reason="rebalance",
                ts=now,
            )
        planned = len(orders)
        orders = one_order_per_symbol(orders)
        if len(orders) < planned:
            self._log("order", f"dropped {planned - len(orders)} duplicate order(s) this cycle "
                               f"— a stop and a rebalance would have reversed the position")
        fills = self.submit(orders, snapshots, now)
        result.orders = len(orders)
        result.fills = len(fills)
        for order in orders:
            self._log("order", f"{order.side.value} {order.qty:g} {order.symbol} ({order.reason})")

        # 8 ── record
        self._record(now, signals, fills)
        self.cycle_count += 1
        self._last_cycle_ts = now
        result.equity = self.portfolio.equity()
        return result

    # ── helpers ────────────────────────────────────────────────────────────

    def _pending_requests(self, snapshots: dict[str, MarketSnapshot],
                          now: float) -> list[tuple[str, str, str, float]]:
        """(state, symbol, news_id, ts) for every pair that needs a fresh answer."""
        requests: list[tuple[str, str, str, float]] = []
        for item in self.news_buffer:
            if not item.symbols:
                continue
            for symbol in item.symbols:
                snap = snapshots.get(symbol)
                if snap is None:
                    continue
                key = (item.news_id, symbol)
                decided = self._decided_at.get(key)
                if decided is not None and (now - decided) < self.refresh_seconds:
                    continue
                # the headline under consideration sits inside a context window
                # of the other recent headlines for the same instrument
                context = [
                    n for n in relevant_news(self.news_buffer, symbol, self.news_window, now)
                    if n.news_id != item.news_id
                ][-6:]
                state = render_state(
                    snap,
                    [*context, item],
                    self.instrument_by_symbol.get(symbol),
                    focus=item,
                )
                requests.append((state, symbol, item.news_id, item.ts))
        return requests

    def _live_decisions(self, now: float) -> list[Decision]:
        cutoff = now - self.news_window
        out = [
            d for key, d in self._decisions.items()
            if d.ts >= cutoff
        ]
        return out

    def _stops_hit(self, snapshots: dict[str, MarketSnapshot]) -> list[tuple[str, str]]:
        L = self.risk.limits
        out: list[tuple[str, str]] = []
        for symbol, pos in self.portfolio.positions.items():
            if pos.is_flat:
                continue
            snap = snapshots.get(symbol)
            if snap is None:
                continue
            if pos.stop_price is None:
                pos.stop_price = update_stop_price(
                    pos.avg_price, snap, pos.qty > 0, L.stop_atr_multiple
                )
                continue
            if pos.qty > 0 and snap.last <= pos.stop_price:
                out.append((symbol, f"long stop {pos.stop_price:.4f} vs {snap.last:.4f}"))
            elif pos.qty < 0 and snap.last >= pos.stop_price:
                out.append((symbol, f"short stop {pos.stop_price:.4f} vs {snap.last:.4f}"))
        return out

    def submit(self, orders: Iterable[Order], snapshots: dict[str, MarketSnapshot],
               ts: float | None = None) -> list[Any]:
        fills: list[Any] = []
        statuses: list[tuple[Order, str, str]] = []
        for order in orders:
            try:
                fill = self.broker.submit(order, snapshots.get(order.symbol), ts)
            except Exception as exc:
                self._log("error", f"order failed for {order.symbol}: {exc}")
                statuses.append((order, "rejected", str(exc)))
                continue
            if fill is not None:
                fills.append(fill)
                statuses.append((order, "filled", f"@ {fill.price:.4f} fee {fill.fee:.4f}"))
            else:
                statuses.append((order, "accepted", "no immediate fill"))
        if self.store and statuses:
            self.store.record_orders(statuses)
            if fills:
                self.store.record_fills(fills)
        return fills

    def _record(self, now: float, signals: dict[str, Signal], fills: list[Any]) -> None:
        prices = {s: snap.last for s, snap in self.latest_snapshots.items()}
        self.portfolio.mark(prices, now)
        if self.store is None:
            return
        snap = self.portfolio.snapshot(now)
        model = ""
        if self._decisions:
            model = next(iter(self._decisions.values())).model or ""
        self.store.record_cycle(
            ts=now,
            equity=snap.equity,
            cash=snap.cash,
            gross=snap.gross_weight,
            net=snap.net_weight,
            realized=snap.realized_pnl,
            unrealized=snap.unrealized_pnl,
            engine=self.engine.name,
            model=model,
            halted=self.risk.halted_today or self.risk.killed,
            notes="; ".join(self.last_verdict.get("notes", []) or []),
        )
        self.store.record_positions(now, self.portfolio.open_positions())
        self.store.record_equity(
            now, snap.equity, snap.cash, snap.gross_weight, snap.net_weight,
            self.portfolio.drawdown(), snap.positions,
        )
        # Risk events are append-only on the governor; record each one exactly
        # once, or the dashboard shows the same halt three hundred times.
        pending = self.risk.events[self._risk_events_recorded:]
        for event in pending:
            if event.get("ts") and event["ts"] <= now:
                self.store.record_risk_event(str(event.get("event")), event)
                self._risk_events_recorded += 1

    # ── control (dashboard and CLI) ────────────────────────────────────────

    def pause(self, paused: bool = True) -> None:
        self.paused = paused
        self._log("control", "paused" if paused else "resumed")

    def request_stop(self) -> None:
        self.stop_requested = True
        self._log("control", "stop requested")

    def request_flatten(self) -> None:
        """Close everything at the next cycle, whatever the signals say."""
        self.flatten_requested = True
        self._log("control", "flatten requested")

    def flatten_and_stop_trading(self) -> list[Any]:
        orders = flatten_orders(self.portfolio, ts=self.now())
        fills = self.submit(orders, self.latest_snapshots, self.now())
        self.risk.killed = True
        self.risk.kill_reason = "flattened by operator"
        self.store and self.store.record_risk_event("operator_flatten", {"orders": len(orders)})
        return fills

    def reset_kill_switch(self) -> None:
        self.risk.reset_kill()
        self._log("control", "kill switch reset")

    # ── running ────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Prime the feeds and say, once, what the bot is about to do."""
        prime = getattr(self.price_feed, "prime", None)
        if callable(prime):
            prime()
        if self.store:
            self.store.record_engine_info(engine_summary(self.engine))
        self._log(
            "boot",
            f"engine={self.engine.name} broker={self.broker.name} "
            f"symbols={','.join(self.symbols)} equity={self.portfolio.equity():,.0f}",
        )

    def run(
        self,
        *,
        cycles: int | None = None,
        duration: float | None = None,
        interval: float | None = None,
        on_cycle: Callable[[CycleResult], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[CycleResult]:
        interval = float(interval if interval is not None else
                         self.cfg.get("loop", "cycle_seconds", default=5.0))
        started = time.time()
        out: list[CycleResult] = []
        n = 0
        while True:
            if self.stop_requested or (should_stop and should_stop()):
                break
            if cycles is not None and n >= cycles:
                break
            if duration is not None and (time.time() - started) >= duration:
                break
            if self.flatten_requested:
                self.flatten_requested = False
                self.flatten_and_stop_trading()
                if cycles is None and duration is None:
                    break
            if self.paused:
                time.sleep(max(0.05, interval))
                continue
            try:
                result = self.cycle()
                out.append(result)
                if on_cycle:
                    on_cycle(result)
            except KeyboardInterrupt:  # pragma: no cover
                break
            except Exception as exc:  # pragma: no cover - keep the loop alive
                log.exception("cycle failed: %s", exc)
                self._log("error", f"cycle failed: {exc}")
                time.sleep(min(5.0, interval))
            n += 1
            if cycles is not None and n >= cycles:
                break
            if duration is not None and (time.time() - started) >= duration:
                break
            time.sleep(max(0.0, interval))
        return out

    # ── status ─────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        pw = self.portfolio.weights()
        return {
            "cycle": self.cycle_count,
            # The operator's flags belong in the payload the dashboard reads:
            # a page that tracks "paused" in its own memory shows the wrong
            # state after a refresh, and offers the wrong button.
            "paused": self.paused,
            "stop_requested": self.stop_requested,
            "flatten_requested": self.flatten_requested,
            "now": self.now(),
            "engine": engine_summary(self.engine),
            "broker": self.broker.info() if hasattr(self.broker, "info") else {},
            "price_feed": self.price_feed.info(),
            "news_feed": self.news_feed.info(),
            "routing": self.router.info(),
            "portfolio": self.portfolio.to_dict(self.instruments),
            "risk": self.risk.status(self.portfolio.equity(), self.now()),
            "signals": [
                {**s.to_dict(), "current_weight": round(pw.get(s.symbol, 0.0), 5)}
                for s in sorted(self.latest_signals.values(), key=lambda x: -abs(x.score))
            ],
            "positions": [
                {**p.to_dict(), "weight": round(pw.get(p.symbol, 0.0), 5)}
                for p in self.portfolio.open_positions()
            ],
            "events": self.events[-60:],
            "news": [n.to_dict() for n in self.news_buffer[-25:]],
        }
