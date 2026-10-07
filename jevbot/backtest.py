"""Event-driven backtester.

Not a separate simulation. This drives :meth:`TradingBot.cycle` itself, with the
clock handed in and the feeds backed by recorded or generated data — so what is
measured is the live code path, including the risk governor, the order planner
and the fee model. A backtester that reimplements the strategy measures the
reimplementation.

What it also ships is a falsification control, because a backtest without one is
a sales document. ``shuffle=True`` keeps every generated headline but permutes
which one lands when, so the texts and the tape stop being related. Any strategy
whose edge survives that run did not have a news edge to begin with, and the
number is printed next to the real one.

The honest reading of a demo-market result: *plumbing verified, edge not
established*. The generative process is a toy and it was written by the same
person who wrote the strategy. Real claims need real data — ``--price-feed ccxt``
and a recorded headline file.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .bot import TradingBot
from .config import Config
from .engine import build_engine
from .feeds.demo import DemoMarket, DemoNewsFeed, DemoPriceFeed, Headline, shuffle_headlines
from .metrics import EngineScore, Performance, performance, round_trips, score_decisions
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    engine: str
    seed: int
    days: float
    symbols: list[str]
    cycles: int = 0
    decisions: int = 0
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    perf: Performance = field(default_factory=Performance)
    benchmark: Performance = field(default_factory=Performance)
    score: EngineScore = field(default_factory=EngineScore)
    oracle: EngineScore = field(default_factory=EngineScore)
    trades: list[dict[str, Any]] = field(default_factory=list)
    halted: bool = False
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    shuffled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "seed": self.seed,
            "days": self.days,
            "symbols": self.symbols,
            "cycles": self.cycles,
            "decisions": self.decisions,
            "shuffled": self.shuffled,
            "halted": self.halted,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "performance": self.perf.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "engine_score": self.score.to_dict(),
            "oracle_ceiling": self.oracle.to_dict(),
            "notes": self.notes,
        }

    def summary(self) -> str:
        # A news strategy that is 6% invested and a buy-and-hold index are not
        # the same trade, and comparing their raw returns flatters whichever one
        # happened to be more exposed. The matched line scales the benchmark to
        # the exposure the strategy actually ran.
        p, b, s = self.perf.to_dict(), self.benchmark.to_dict(), self.score.to_dict()
        matched = b["total_return"] * p["exposure"]
        tag = "SHUFFLED CONTROL (headlines decoupled from the tape)" if self.shuffled else "backtest"
        lines = [
            f"Jevbot {tag} — engine={self.engine} seed={self.seed} days={self.days:g}",
            "-" * 68,
            f"  cycles                  {self.cycles}",
            f"  decisions               {self.decisions}",
            f"  ended equity            {p['ending_equity']:,.2f}  (from {p['starting_equity']:,.2f})",
            f"  total return            {p['total_return'] * 100:+,.2f}%",
            f"  buy-and-hold benchmark  {b['total_return'] * 100:+,.2f}%  (fully invested)",
            f"    at matched exposure   {matched * 100:+,.2f}%  (benchmark x mean exposure)",
            f"  excess vs matched B&H   {(p['total_return'] - matched) * 100:+,.2f}%",
            f"  annual vol              {p['vol_annual'] * 100:,.2f}%",
            f"  Sharpe / Sortino        {p['sharpe']:,.2f} / {p['sortino']:,.2f}",
            f"  max drawdown            {p['max_drawdown'] * 100:,.2f}%",
            f"  trades / hit rate       {p['trades']} / {p['hit_rate'] * 100:,.1f}%",
            f"  profit factor           {p['profit_factor']:,.2f}",
            f"  fees / turnover         {p['fees_paid']:,.2f} / {p['turnover']:,.2f}x",
            f"  mean exposure           {p['exposure'] * 100:,.1f}%",
            "",
            "  engine skill (against held-out forward returns)",
            f"    labelled decisions    {s.get('decisions', 0)}",
            f"    direction accuracy    {_pct(s.get('direction_accuracy'))} on {s.get('direction_n', 0)} directional calls",
            f"    coverage / flat share {_pct(1 - (s.get('flat_share') or 0))} / {_pct(s.get('flat_share'))}",
            f"    materiality Brier     {_num(s.get('materiality_brier'))}  (base rate {_pct(s.get('materiality_base_rate'))})",
            f"    context Brier         {_num(s.get('context_brier'))}",
            f"    risk-event Brier      {_num(s.get('risk_brier'))}",
            f"    confidence ECE        {_num(s.get('answer_confidence_ece'))}",
            f"    signal IC (Spearman)  {_num(s.get('signal_ic'))}",
            "",
            "  oracle reference, same rows (knows the headline's hidden truth",
            "  and nothing about the tape — see the note in the README)",
            f"    direction accuracy    {_pct(self.oracle.direction_accuracy)}"
            f" on {self.oracle.direction_n} calls",
            f"    signal IC (Spearman)  {_num(self.oracle.signal_ic)}",
        ]
        if self.notes:
            lines += ["", "  notes"] + [f"    {n}" for n in self.notes]
        if not self.shuffled:
            lines += [
                "",
                "  Synthetic market: this validates the pipeline, it does not establish an edge.",
                "  Run scripts/backtest.py with the same seed and --shuffle to see the control.",
            ]
        return "\n".join(lines)


def _horizon_sigma(market: DemoMarket, symbol: str, ts: float,
                   horizon_seconds: float) -> float:
    """Volatility the instrument was expected to realise over the horizon."""
    returns = market._returns.get(symbol) or market.bars(symbol) and market._returns[symbol]
    idx = market.index_at(ts)
    window = max(12, int(86400 / market.bar_seconds))
    recent = returns[max(0, idx - window):max(1, idx + 1)]
    if len(recent) < 3:
        return 0.0
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
    per_bar = math.sqrt(max(0.0, var))
    per_hour = per_bar * math.sqrt(3600.0 / market.bar_seconds)
    return per_hour * math.sqrt(max(0.0, horizon_seconds) / 3600.0)


def _pct(v: Any) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) and math.isfinite(float(v)) else "n/a"


def _num(v: Any) -> str:
    return f"{v:,.4f}" if isinstance(v, (int, float)) and math.isfinite(float(v)) else "n/a"


def build_demo_market(cfg: Config, *, days: float | None = None,
                      seed: int | None = None) -> DemoMarket:
    d = cfg.section("demo")
    bar_seconds = int(cfg.get("loop", "bar_seconds", default=300))
    return DemoMarket(
        cfg.instruments,
        seed=int(seed if seed is not None else d.get("seed", 7)),
        days=float(days if days is not None else d.get("days", 30)),
        bar_seconds=bar_seconds,
        headlines_per_day=int(d.get("headlines_per_day", 24)),
        impact_scale=float(d.get("impact_scale", 0.012)),
        noise_share=float(d.get("noise_share", 0.30)),
    )


def _benchmark(market: DemoMarket, cfg: Config, t0: float, t1: float,
               step: float) -> Performance:
    """Equal-weight buy-and-hold over the same window, same universe, same steps."""
    if t1 <= t0:
        return Performance()
    stamps: list[float] = []
    ts = t0
    while ts <= t1:
        stamps.append(ts)
        ts += step
    if not stamps or stamps[-1] < t1:
        stamps.append(t1)

    series: list[list[float]] = []
    for inst in cfg.instruments:
        bars = market.bars(inst.symbol)
        base = bars[max(0, min(len(bars) - 1, market.index_at(t0)))] or 1.0
        series.append([
            (bars[max(0, min(len(bars) - 1, market.index_at(t)))] / base)
            for t in stamps
        ])
    curve = [
        (t, 100.0 * sum(s[i] for s in series) / len(series))
        for i, t in enumerate(stamps)
    ]
    return performance(curve, starting_equity=100.0, period_seconds=step)


def run_backtest(
    cfg: Config,
    *,
    engine_name: str | None = None,
    days: float | None = None,
    seed: int | None = None,
    step_seconds: float | None = None,
    warmup_days: float = 8.0,
    shuffle: bool = False,
    horizon_seconds: float = 6 * 3600.0,
    store: Store | None = None,
    progress: Any | None = None,
) -> BacktestResult:
    """Replay the demo market through the live loop and measure what happened."""
    d = cfg.section("demo")
    bar_seconds = int(cfg.get("loop", "bar_seconds", default=300))
    market = build_demo_market(cfg, days=days, seed=seed)
    if shuffle:
        market.use_headlines(shuffle_headlines(market.headlines(), seed=int(seed or market.seed) + 1))

    price_feed = DemoPriceFeed(market, replay_from_start=True)
    news_feed = DemoNewsFeed(market, price_feed)
    # Every run records into SQLite, in memory by default: the recorded
    # decisions are what the skill metrics are computed from, and "the eval
    # reads what the bot actually recorded" is worth more than saving a table.
    store = store if store is not None else Store(":memory:")

    engine = build_engine(
        engine_name or cfg.engine_name,
        repo=str(cfg.get("engine", "laya_repo", default="convaiinnovations/laya")),
        device=str(cfg.get("engine", "device", default="") or "") or None,
        preload=False,
        min_confidence=float(cfg.get("engine", "min_confidence", default=0.0) or 0.0),
        cache=bool(cfg.get("engine", "cache", default=True)),
    )

    bot = TradingBot(cfg, engine=engine, price_feed=price_feed, news_feed=news_feed, store=store)
    clock = {"now": market.start_ts + warmup_days * 86400.0}
    bot.clock_fn = lambda: clock["now"]
    step = float(step_seconds or bar_seconds)
    # the warmup window has to leave a testable period behind it, even for a
    # short run: never spend more than half the generated history on warmup
    warmup_days = min(float(warmup_days), max(0.25, market.days * 0.5))
    start = market.start_ts + warmup_days * 86400.0
    end = market.end_ts
    result = BacktestResult(
        engine=engine.name,
        seed=market.seed,
        days=market.days,
        symbols=[i.symbol for i in cfg.instruments],
        shuffled=shuffle,
    )
    t0 = time.perf_counter()
    ts = start
    last_report = 0.0
    while ts <= end:
        clock["now"] = ts
        res = bot.cycle()
        result.cycles += 1
        result.equity_curve.append((ts, bot.portfolio.equity()))
        result.decisions += res.decisions
        if progress is not None and (ts - last_report) > (end - start) / 20.0:
            progress((ts - start) / max(1e-9, end - start), result)
            last_report = ts
        ts += step
    clock["now"] = end
    result.elapsed_seconds = time.perf_counter() - t0
    result.halted = bot.risk.killed or bot.risk.halted_today
    result.notes = [
        f"engine fallback: {engine.name}" if engine.name != (engine_name or cfg.engine_name) else "",
        f"risk halt: {bot.risk.kill_reason}" if bot.risk.killed else "",
    ]
    result.notes = [n for n in result.notes if n]

    # ── performance
    fills = [f.to_dict() for f in bot.portfolio.fills]
    trades = round_trips(bot.portfolio)
    # exposure sampled from the equity rows the bot recorded
    exposure_values: list[float] = []
    if store is not None:
        exposure_values = [float(r["gross_weight"] or 0.0) for r in store.latest_equity(100000)]
    result.perf = performance(
        result.equity_curve,
        starting_equity=bot.portfolio.starting_equity,
        fills=fills,
        trades=trades,
        exposure_values=exposure_values,
        period_seconds=step,
    )
    result.trades = trades
    result.benchmark = _benchmark(market, cfg, start, end, step)

    # ── engine skill against labels the engine never saw
    #
    # The label is measured from the moment the *decision* was made, not from
    # the moment the headline was published. Those differ: a headline is
    # re-read against an updated market once an hour, and a decision taken two
    # hours late already has two hours of the move in the price it was shown.
    # Scoring it against the headline's own window credits the engine with a
    # move it could not have captured — which flatters late decisions, and
    # flatters exactly the refresh behaviour this system was built around.
    rows = bot.store.query("SELECT * FROM decisions") if bot.store else []
    if not rows and store is not None:
        rows = store.query("SELECT * FROM decisions")
    forward: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    for row in rows:
        key = f"{row.get('news_id')}|{row.get('symbol')}"
        if key in forward:
            continue
        ts = float(row.get("ts") or 0.0)
        forward[key] = market.forward_return(str(row.get("symbol")), ts, horizon_seconds)
        sigmas[key] = _horizon_sigma(market, str(row.get("symbol")), ts, horizon_seconds)
    result.score = score_decisions(rows, forward, sigmas=sigmas)

    # The ceiling. Score an oracle that is handed the generator's hidden truth
    # on the *same* labels: any engine scoring near it is reading the news, an
    # engine scoring above it is reading the future. Note what this does to the
    # shuffled control — the oracle's edge survives the shuffle, because the
    # tape is still built from the same hidden impacts. The text is the only
    # thing that lost its meaning, which is precisely what the control tests.
    head_map: dict[str, Headline] = {}
    for headline in market.labelled_headlines(horizon_seconds):
        head_map[headline.item.news_id] = headline
        head_map[headline.item.news_id + "-shuffled"] = headline
    #
    # It answers on exactly the rows the engine answered on — one oracle row per
    # decision row, not one per headline. Scoring the engine over a thousand
    # re-decisions of two hundred headlines while scoring the oracle over the
    # two hundred is not a comparison: the re-decisions are easier (the price
    # has already moved and the position is already on) and they swamp the
    # average. Same rows, or no conclusion.
    oracle_rows: list[dict[str, Any]] = []
    for row in rows:
        news_id = str(row.get("news_id"))
        headline = head_map.get(news_id)
        if headline is None:
            continue
        valence = float(headline.valence)
        magnitude = min(1.0, abs(float(headline.magnitude)))
        oracle_rows.append({
            "news_id": news_id,
            "symbol": row.get("symbol"),
            "direction": "long" if valence > 0 else "short" if valence < 0 else "flat",
            "direction_p": min(1.0, abs(float(headline.impact)) * 40.0) or 0.5,
            "materiality": magnitude,
            "conviction_norm": magnitude,
            "priced_in": 0.0,
            "context_aligned": 0.5,
            "risk_event": 0.0,
            "engine": "oracle",
            "model": "oracle",
        })
    result.oracle = score_decisions(oracle_rows, forward, sigmas=sigmas)

    if store is not None:
        store.record_engine_info({"backtest": result.to_dict()})
        store.record_risk_event("backtest_complete", result.to_dict()["performance"])
    return result


def compare_engines(
    cfg: Config,
    *,
    engines: Iterable[str] = ("heuristic",),
    **kwargs: Any,
) -> dict[str, BacktestResult]:
    """Run the same replay for each engine. Identical seed, identical headlines."""
    out: dict[str, BacktestResult] = {}
    for name in engines:
        log.info("backtest: %s", name)
        out[name] = run_backtest(cfg, engine_name=name, **kwargs)
    return out


def write_result(result: BacktestResult, directory: str | Path = "data/runs") -> Path:
    """Persist a run: JSON summary plus the equity curve as CSV."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base = out / f"{stamp}-{result.engine}{'-shuffled' if result.shuffled else ''}"
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
    with open(base.with_suffix(".csv"), "w", encoding="utf-8") as fh:
        fh.write("ts,equity\n")
        for ts, eq in result.equity_curve:
            fh.write(f"{ts:.0f},{eq:.6f}\n")
    return base


def compare_markdown(results: dict[str, BacktestResult], control: BacktestResult | None = None) -> str:
    """A small comparison table, for a README or a pull request body."""
    header = "| engine | total return | excess vs B&H | Sharpe | max DD | trades | hit rate | direction acc | signal IC |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for name, r in results.items():
        p, b, s = r.perf.to_dict(), r.benchmark.to_dict(), r.score.to_dict()
        rows.append(
            f"| `{name}` | {p['total_return'] * 100:+.2f}% | "
            f"{(p['total_return'] - b['total_return']) * 100:+.2f}% | {p['sharpe']:.2f} | "
            f"{p['max_drawdown'] * 100:.2f}% | {p['trades']} | {p['hit_rate'] * 100:.1f}% | "
            f"{_pct(s.get('direction_accuracy'))} | {_num(s.get('signal_ic'))} |"
        )
    if control is not None:
        p, s = control.perf.to_dict(), control.score.to_dict()
        rows.append(
            f"| `{control.engine}` + shuffled headlines | {p['total_return'] * 100:+.2f}% | "
            f" | {p['sharpe']:.2f} | {p['max_drawdown'] * 100:.2f}% | {p['trades']} | "
            f"{p['hit_rate'] * 100:.1f}% | {_pct(s.get('direction_accuracy'))} | "
            f"{_num(s.get('signal_ic'))} |"
        )
    return "\n".join(rows)
