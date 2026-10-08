"""Command line interface.

    jevbot run          start trading (paper by default) with the dashboard
    jevbot backtest     replay the demo market and measure what happened
    jevbot eval         score the engine, compare engines, and run the control
    jevbot gen-data     write the demo headlines to disk
    jevbot route TEXT   show what the router decides and what the engine answers
    jevbot doctor       check the environment before trusting anything

Written with argparse and nothing else, so it works on a bare interpreter in the
same way the bot does.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from . import __version__
from .backtest import compare_engines, compare_markdown, run_backtest, write_result
from .bot import TradingBot
from .brokers.base import LiveTradingRefused
from .config import ROOT, ConfigError, load_config
from .engine import build_engine
from .engine.base import EngineUnavailable
from .feeds import build_news_feed, build_price_feed
from .feeds.base import FeedError
from .metrics import score_decisions
from .risk import RiskLimits
from .store import Store

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load(args) -> tuple[Config, list[str]]:
    notes: list[str] = []
    cfg = load_config(*getattr(args, "config", []) or [])
    if getattr(args, "engine", None):
        cfg.raw.setdefault("engine", {})["name"] = args.engine
    if getattr(args, "mode", None):
        cfg.raw.setdefault("mode", {})["live"] = args.mode == "live"
        if args.mode == "paper":
            cfg.raw.setdefault("broker", {})["kind"] = "paper"
    if getattr(args, "price_feed", None):
        cfg.raw.setdefault("feeds", {})["price"] = args.price_feed
    if getattr(args, "news_feed", None):
        cfg.raw.setdefault("feeds", {})["news"] = args.news_feed
    if getattr(args, "impact_scale", None) is not None:
        cfg.raw.setdefault("demo", {})["impact_scale"] = args.impact_scale
    if getattr(args, "noise_share", None) is not None:
        cfg.raw.setdefault("demo", {})["noise_share"] = args.noise_share
    if getattr(args, "set", None):
        for pair in args.set:
            if "=" not in pair:
                raise ConfigError(f"--set expects key.path=value, got {pair!r}")
            key, value = pair.split("=", 1)
            node = cfg.raw
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = _coerce(value)
    return cfg, notes


def _coerce(value: str):
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# ── doctor ──────────────────────────────────────────────────────────────────


def cmd_doctor(args) -> int:
    cfg, _ = _load(args)
    print(f"jevbot {__version__}  (python {sys.version.split()[0]})")
    print(f"config: {cfg.path}")

    optional = {
        "laya": "the decision engine (pip install 'jevbot[engine]')",
        "ccxt": "crypto live data and orders (pip install 'jevbot[crypto]')",
        "alpaca": "US equities (pip install 'jevbot[equities]')",
    }
    for module, why in optional.items():
        try:
            __import__(module)
            print(f"  [ok]   {module}")
        except ImportError:
            print(f"  [--]   {module} — {why}")

    print(f"\nuniverse: {', '.join(cfg.symbols)}")
    engine = None
    try:
        engine = build_engine(
            cfg.engine_name,
            repo=str(cfg.get("engine", "laya_repo", default="convaiinnovations/laya")),
            device=str(cfg.get("engine", "device", default="") or "") or None,
            preload=False,
            min_confidence=float(cfg.get("engine", "min_confidence", default=0.0) or 0.0),
        )
        info = engine.info()
        print(f"engine:   {info.get('engine')} (requested {cfg.engine_name})")
        for key in ("repo", "device", "cache_hit_rate", "checkpoints", "note"):
            if info.get(key) is not None:
                print(f"          {key}: {info[key]}")
    except EngineUnavailable as exc:
        print(f"engine:   UNAVAILABLE — {exc}")
        print("          fix the checkpoint download, or run with --engine heuristic")
        return 2
    finally:
        if engine is not None:
            engine.close()

    feed_failed = False
    try:
        price = build_price_feed(cfg, speed=1.0)
        news = build_news_feed(cfg, price_feed=price)
        info = price.info()
        print(f"\nfeeds:    price={info.get('feed')} news={news.info().get('feed')}")
        if info.get("exchange"):
            where = "testnet" if info.get("testnet") else "mainnet"
            print(f"          exchange: {info['exchange']} ({where})")
            print(f"          endpoint: {info.get('endpoint') or 'unknown'}")
        snaps = price.snapshots(cfg.symbols)
        for symbol, snap in list(snaps.items())[:5]:
            print(f"          {symbol:10s} {snap.last:>12,.2f}  24h {snap.ret_24h * 100:+6.2f}%  "
                  f"vol {snap.vol_24h * 100:5.2f}%  trend {snap.trend}")

        # The independent check: candles answer "what happened", a ticker
        # answers "what is it now". If the two disagree, the feed is pointed at
        # the wrong venue or market — and that is exactly the failure a
        # reasonable-looking candle chart cannot show you.
        if hasattr(price, "ticker"):
            print("          cross-check against the venue's own ticker:")
            for symbol in cfg.symbols:
                snap = snaps.get(symbol)
                if snap is None:
                    continue
                try:
                    tick = price.ticker(symbol)
                except Exception as exc:
                    print(f"            {symbol:10s} ticker unavailable: {str(exc)[:70]}")
                    continue
                if not tick:
                    print(f"            {symbol:10s} ticker returned nothing")
                    continue
                delta = (snap.last - tick) / tick if tick else 0.0
                flag = "ok" if abs(delta) <= 0.005 else "MISMATCH"
                print(f"            {symbol:10s} feed {snap.last:>12,.2f} vs ticker {tick:>12,.2f}"
                      f"  ({delta * 100:+6.2f}% {flag})")
    except Exception as exc:
        # Report and keep going: the broker block below is exactly what an
        # operator needs next, and it does not depend on the feed being up.
        print(f"feeds:    FAILED — {exc}")
        feed_failed = True

    from .routing import NewsRouter

    r = NewsRouter(cfg.instruments, mode="keywords")
    for text in ("Bitcoin ETF inflows hit a record as BTC rallies",
                 "Apple beats earnings as iPhone sales jump",
                 "Global markets sell off on rate fears"):
        syms, why = r.route(type("N", (), {"text": text, "symbols": (), "url": ""})())
        print(f"  route: {text[:48]:<50s} -> {syms or '—'}  ({why})")

    if cfg.broker_kind != "paper" or cfg.testnet:
        from .brokers import build_broker
        from .portfolio import Portfolio

        try:
            # Built for real, not forced to paper: the question this block
            # answers is "where would an order actually go", and a paper broker
            # cannot answer it. Constructing one places nothing.
            broker = build_broker(cfg, Portfolio.fresh(
                float(cfg.get("risk", "starting_equity", default=1000.0))))
            binfo = broker.info()
            print(f"\nbroker:   {binfo.get('broker')} ({binfo.get('exchange', '—')})")
            print(f"          endpoint: {binfo.get('endpoint') or 'unknown'}")
            print(f"          testnet:  {binfo.get('testnet')}   live: {binfo.get('live')}")
            if not binfo.get("testnet"):
                print("          WARNING: this endpoint can move real money; orders need")
                print("                   JEVBOT_I_UNDERSTAND_LIVE_RISK=yes as well.")
            else:
                print("          orders would be placed on the sandbox matching engine")
                equity_fn = getattr(broker, "equity", None)
                if callable(equity_fn):
                    try:
                        cash, holdings = equity_fn()
                        held = ", ".join(f"{v:g} {k}" for k, v in holdings.items() if v)
                        print(f"          sandbox balance: {cash:,.2f} USDT"
                              + (f" · holding {held}" if held else ""))
                        if cash <= 0 and not held:
                            print("                           (use the testnet faucet before trading)")
                    except Exception as exc:
                        print(f"          sandbox balance: unavailable — {str(exc)[:70]}")
            limits = RiskLimits.from_config(cfg)
            print("          safety:   max order "
                  + (f"{limits.max_order_notional:,.0f}" if limits.max_order_notional else "uncapped")
                  + f" · stop after {limits.max_consecutive_rejections} rejections"
                  + f" · daily loss {limits.daily_loss_limit_pct:.0%}"
                  + f" · kill at {limits.max_drawdown_pct:.0%} drawdown")
        except Exception as exc:
            print(f"\nbroker:   unavailable — {exc}")

    if cfg.testnet:
        print("\nnote: testnet endpoints — real market data, fake money. Orders routed to")
        print("      the venue cannot leave the sandbox.")
    elif not cfg.live and cfg.broker_kind != "paper":
        print("\nnote: mode is paper, so live orders are impossible in this run.")

    if feed_failed:
        print("\nAt least one check failed — the notes above name what.")
        return 3
    print("\nAll good. Next: `jevbot run --serve` or `jevbot backtest --days 30`.")
    return 0


def cmd_testnet_order(args) -> int:
    """Place one small order on the sandbox venue and report what came back.

    This is the answer to "did it really go to Binance?", and it is deliberately
    a separate command from `run`: one order, on request, with the endpoint and
    the order id printed. A bot loop is a slow and confusing way to find out that
    a key is wrong.
    """
    cfg, _ = _load(args)
    from .brokers import build_broker
    from .portfolio import Portfolio
    from .types import MarketSnapshot, Order, Side
    from .venues import is_local_endpoint, is_testnet_endpoint

    if cfg.broker_kind != "ccxt":
        print(f"this command only places orders through the ccxt broker "
              f"(broker.kind is {cfg.broker_kind!r})")
        return 2
    try:
        broker = build_broker(cfg, Portfolio.fresh(
            float(cfg.get("risk", "starting_equity", default=1000.0))))
    except Exception as exc:
        # Constructing a mainnet broker without the live acknowledgement raises
        # here. That is the refusal working, not a crash: say so plainly.
        print(f"refusing: {exc}")
        return 2
    endpoint = getattr(broker, "rest_endpoint", "")
    # Two independent conditions, and the first is not a string match: the
    # broker only reports testnet when set_sandbox_mode was actually applied.
    # The endpoint check then catches a config that points a "sandbox" broker at
    # a live host anyway (sandbox mode rewrites every family, so this is a
    # belt-and-braces check against a future adapter change).
    sandbox = is_testnet_endpoint(endpoint) or is_local_endpoint(endpoint)
    if not getattr(broker, "testnet", False) or not sandbox:
        print(f"refusing: {endpoint or 'the configured endpoint'} is not a sandbox.")
        print("this command exists to smoke-test a testnet key; it will not place")
        print("an order anywhere that can move real money.")
        return 2

    symbol = args.symbol or cfg.symbols[0]
    side = Side.BUY if (args.side or "buy").lower() == "buy" else Side.SELL
    limits = RiskLimits.from_config(cfg)
    notional = float(args.notional)
    if limits.max_order_notional and notional > limits.max_order_notional:
        print(f"clamping the order to risk.max_order_notional = {limits.max_order_notional:,.2f}")
        notional = limits.max_order_notional

    try:
        broker.exchange.load_markets()
        ticker = broker.exchange.fetch_ticker(symbol)
    except Exception as exc:
        print(f"could not read a price for {symbol} from {endpoint}: {exc}")
        return 3
    price = float(ticker.get("last") or ticker.get("close") or 0.0)
    if not price:
        print(f"the venue returned no price for {symbol}")
        return 3
    qty = float(broker.exchange.amount_to_precision(symbol, notional / price))
    if qty <= 0:
        print(f"{notional:,.2f} is below one lot of {symbol} at {price:,.2f}")
        return 3

    print(f"venue:    {endpoint}  (testnet — fake money)")
    print(f"order:    {side.value} {qty:g} {symbol} ≈ {qty * price:,.2f} USDT at {price:,.2f}")
    if not args.yes:
        print("\nrefusing to place it without --yes.")
        return 2

    snap = MarketSnapshot(symbol=symbol, ts=time.time(), last=price)
    try:
        fill = broker.submit(Order(symbol=symbol, side=side, qty=qty, reason="testnet smoke order"),
                             snap, ts=time.time())
    except Exception as exc:
        print(f"\nREJECTED by the venue: {exc}")
        print("nothing was filled. Check the key's permissions and the account balance:")
        print(f"  {endpoint}/account")
        return 3
    if fill is None:
        print("\nthe venue accepted the order but has not printed a fill yet;"
              " check open orders on the dashboard")
        return 0
    print(f"\nFILLED by the venue: {fill.qty:g} {symbol} at {fill.price:,.2f} · "
          f"fee {fill.fee:,.4f} · order id {fill.order_id}")
    print("\nConfirm it on Binance testnet → Wallet → Order history")
    print("  https://testnet.binance.vision/")
    print("If it is not there, the order did not reach the venue.")
    return 0


# ── run ─────────────────────────────────────────────────────────────────────


def cmd_run(args) -> int:
    cfg, _ = _load(args)
    speed = float(args.speed or (200.0 if str(cfg.get("feeds", "price", default="")).lower()
                                in {"synthetic", "demo", ""} else 1.0))
    store = Store(cfg.get("server", "db", default="data/store/jevbot.db"))
    bot = TradingBot(cfg, store=store, speed=speed)
    try:
        bot.warmup()
    except FeedError as exc:
        # A venue that is down, or a key that has been revoked, must not take
        # the dashboard with it: the process that says *why* nothing is trading
        # is the one worth keeping up, and the cycle loop is already written to
        # retry every interval. Headless runs still fail fast — a script that
        # asked for a live feed should not exit 0 having read nothing.
        if args.no_serve:
            print(f"could not start: {exc}")
            return 3
        print(f"warning: {exc}")
        print("         the dashboard is up and each cycle will retry; the feed's")
        print("         endpoint is named above, and the failure is logged in the UI.")

    if cfg.live and not os.environ.get("JEVBOT_I_UNDERSTAND_LIVE_RISK"):
        print("refusing to start: live mode needs JEVBOT_I_UNDERSTAND_LIVE_RISK=yes")
        return 2

    server = None
    if not args.no_serve:
        from .server import DashboardServer

        server = DashboardServer(
            bot, store,
            host=str(args.host or cfg.get("server", "host", default="0.0.0.0")),
            port=int(args.port or cfg.get("server", "port", default=8000)),
        )
        try:
            url = server.start(background=True)
        except OSError as exc:
            # Overwhelmingly this is "address already in use", and the port is
            # almost always held by the dashboard the operator forgot about.
            print(f"could not start the dashboard on {server.host}:{server.port}: {exc}")
            print("pass --port to use another one, or stop the process holding it:")
            print(f"  ss -ltnp | grep {server.port}")
            return 4
        print(f"dashboard: {url}  (bind {server.host}:{server.port})")
    print(f"engine={bot.engine.name} broker={bot.broker.name} symbols={len(bot.symbols)} "
          f"speed={speed:g}x — Ctrl-C to stop")

    # A demo tape is finite: at speed it runs out, and a bot with a clock
    # pinned to the last bar stops being a demo of anything. Loop the tape
    # instead, so the dashboard keeps showing a live market.
    loop_tape = bool(args.loop if args.loop is not None
                     else getattr(bot.price_feed, "resettable", False))

    def _maybe_loop():
        if not loop_tape or not getattr(bot.price_feed, "resettable", False):
            return
        market = getattr(bot.price_feed, "market", None)
        end = getattr(market, "end_ts", None)
        if end is None or bot.price_feed.sim_now() < end:
            return
        for feed in (bot.price_feed, bot.news_feed):
            if callable(getattr(feed, "reset", None)):
                feed.reset()
        bot._log("tape", "demo tape exhausted — rewinding to the start")

    def _on_cycle(r) -> None:
        _maybe_loop()
        if args.verbose:
            print(
                f"  cycle {r.cycle:>4}  eq {r.equity:>12,.2f}  news {r.news:>3}  "
                f"decisions {r.decisions:>3}  orders {r.orders:>2}  fills {r.fills:>2}"
                + ("  [HALTED]" if r.halted else "")
            )

    stop = threading.Event()

    def _sigterm(_signum, _frame):  # pragma: no cover - signal path
        print("\nstopping…")
        stop.set()
        bot.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sigterm)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass

    try:
        results = bot.run(
            cycles=args.cycles,
            duration=args.duration,
            interval=float(args.interval or cfg.get("loop", "cycle_seconds", default=5.0)),
            should_stop=stop.is_set,
            on_cycle=_on_cycle if (args.verbose or loop_tape) else None,
        )
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        total = bot.portfolio.equity() - bot.portfolio.starting_equity
        print(f"\nstopped after {len(results)} cycles · equity {bot.portfolio.equity():,.2f} "
              f"({total:+,.2f}) · fills {len(bot.portfolio.fills)} · fees {bot.portfolio.fees_paid:,.2f}")
        if server:
            server.stop()
        store.close()
    return 0


# ── backtest ────────────────────────────────────────────────────────────────


def cmd_backtest(args) -> int:
    cfg, _ = _load(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [None]
    engines = [e.strip() for e in (args.engines or cfg.engine_name).split(",") if e.strip()]
    # A backtest is a measurement, not trading: by default it keeps its rows in
    # memory and leaves `data/store/jevbot.db` to the live book. Sharing one
    # database means the dashboard charts a backtest's equity next to the live
    # account — two different questions drawn as one line.
    store = Store(args.db) if args.db else Store(":memory:")

    all_results: dict[str, list] = {}
    for engine_name in engines:
        all_results[engine_name] = []
        for seed in seeds:
            result = run_backtest(
                cfg,
                engine_name=engine_name,
                days=args.days,
                seed=seed,
                step_seconds=args.step,
                horizon_seconds=args.horizon,
                shuffle=args.shuffle,
                store=store,
            )
            all_results[engine_name].append(result)
            if not args.quiet and len(seeds) == 1 and len(engines) == 1:
                print(result.summary())
            if args.out:
                path = write_result(result, cfg.resolve_path(args.out))
                print(f"wrote {path}.json")

    if len(seeds) > 1 or len(engines) > 1:
        print(_aggregate_table(all_results, shuffled=args.shuffle))
    store.close()
    return 0


def _aggregate_table(results: dict[str, list], shuffled: bool = False) -> str:
    import statistics

    tag = " (shuffled control)" if shuffled else ""
    lines = [
        f"Multi-seed summary{tag} — mean across seeds, with the range",
        "=" * 78,
        f"{'engine':<12}{'return':>12}{'excess':>12}{'sharpe':>10}{'maxDD':>10}{'trades':>8}{'dir acc':>10}{'IC':>9}",
    ]
    for name, runs in results.items():
        if not runs:
            continue
        rets = [r.perf.total_return for r in runs]
        exc = [r.perf.total_return - r.benchmark.total_return * r.perf.exposure for r in runs]
        sharpes = [r.perf.sharpe for r in runs]
        dds = [r.perf.max_drawdown for r in runs]
        trades = [r.perf.trades for r in runs]
        accs = [r.score.direction_accuracy for r in runs if r.score.direction_accuracy == r.score.direction_accuracy]
        ics = [r.score.signal_ic for r in runs if r.score.signal_ic == r.score.signal_ic]
        lines.append(
            f"{name:<12}{statistics.mean(rets) * 100:>11.2f}%{statistics.mean(exc) * 100:>11.2f}%"
            f"{statistics.mean(sharpes):>10.2f}{statistics.mean(dds) * 100:>9.2f}%"
            f"{statistics.mean(trades):>8.0f}"
            f"{(statistics.mean(accs) * 100 if accs else float('nan')):>9.1f}%"
            f"{(statistics.mean(ics) if ics else float('nan')):>9.3f}"
        )
        lines.append(
            f"{'  range':<12}{min(rets) * 100:>11.2f}%{min(exc) * 100:>11.2f}%"
            f"{min(sharpes):>10.2f}{max(dds) * 100:>9.2f}%"
            f"{min(trades):>8.0f}   .. {max(trades) if trades else 0}"
        )
    lines += [
        "",
        "Read the range, not just the mean: with a handful of seeds the spread is the story.",
    ]
    return "\n".join(lines)


# ── eval ────────────────────────────────────────────────────────────────────


def cmd_eval(args) -> int:
    cfg, _ = _load(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [7, 11, 23]
    engines = [e.strip() for e in (args.engines or "heuristic").split(",") if e.strip()]
    print(f"scoring {', '.join(engines)} over {len(seeds)} seed(s), {args.days:g} days each, "
          f"horizon {args.horizon / 3600:g}h\n")

    results: dict[str, list] = {}
    for engine_name in engines:
        results[engine_name] = [
            run_backtest(cfg, engine_name=engine_name, days=args.days, seed=seed,
                         step_seconds=args.step, horizon_seconds=args.horizon)
            for seed in seeds
        ]
    print(_aggregate_table(results))

    if args.control:
        control = run_backtest(cfg, engine_name=engines[0], days=args.days, seed=seeds[0],
                               step_seconds=args.step, horizon_seconds=args.horizon, shuffle=True)
        real = results[engines[0]][0]
        print("\nFalsification control (same tape, headlines permuted, seed "
              f"{seeds[0]})\n" + "=" * 78)
        for label, r in (("real", real), ("shuffled", control)):
            print(f"  {label:<9} return {r.perf.total_return * 100:+6.2f}%   "
                  f"excess {100 * (r.perf.total_return - r.benchmark.total_return * r.perf.exposure):+6.2f}%   "
                  f"dir acc {r.score.direction_accuracy * 100:5.1f}%   IC {r.score.signal_ic:+.3f}")
        print("\n  If the shuffled run keeps the edge, the edge was not the news.")

    if args.compare and len(engines) > 1:
        print()
        print(compare_markdown({name: runs[0] for name, runs in results.items()}))
    return 0


# ── gen-data ────────────────────────────────────────────────────────────────


def cmd_gen_data(args) -> int:
    cfg, _ = _load(args)
    from .backtest import build_demo_market

    market = build_demo_market(cfg, days=args.days, seed=args.seed)
    out = cfg.resolve_path(args.out or "data/demo")
    out.mkdir(parents=True, exist_ok=True)
    heads = market.labelled_headlines(args.horizon)
    news_path = out / "headlines.jsonl"
    with open(news_path, "w", encoding="utf-8") as fh:
        for h in heads:
            fh.write(json.dumps(h.to_dict()) + "\n")

    prices_path = out / "prices.jsonl"
    with open(prices_path, "w", encoding="utf-8") as fh:
        for inst in cfg.instruments:
            bars = market.bars(inst.symbol)
            for i, price in enumerate(bars):
                if i % max(1, int(args.sample)) != 0:
                    continue
                fh.write(json.dumps({
                    "ts": market.start_ts + i * market.bar_seconds,
                    "symbol": inst.symbol,
                    "price": round(price, 6),
                    "volume": 1.0,
                }) + "\n")
    print(f"wrote {len(heads)} labelled headlines -> {news_path}")
    print(f"wrote bars -> {prices_path}")
    print(f"seed {market.seed}, {market.days:g} days, bar {market.bar_seconds}s, "
          f"impact scale {market.impact_scale}, noise share {market.noise_share}")
    print("\nReplay them with:  jevbot run --price-feed jsonl --news-feed jsonl")
    return 0


# ── route (inspection) ──────────────────────────────────────────────────────


def cmd_route(args) -> int:
    cfg, _ = _load(args)
    engine = build_engine(
        cfg.engine_name,
        repo=str(cfg.get("engine", "laya_repo", default="convaiinnovations/laya")),
        device=str(cfg.get("engine", "device", default="") or "") or None,
        min_confidence=0.0,
    )
    from .questions import decision_questions
    from .routing import NewsRouter
    from .signals import SignalBuilder, size_signals
    from .text import render_state
    from .types import NewsItem

    price = build_price_feed(cfg, speed=1.0)
    symbol = args.symbol or cfg.symbols[0]
    snap = price.snapshot(symbol)
    if snap is None:
        print(f"no market snapshot for {symbol}")
        return 2

    item = NewsItem(text=args.text, symbols=(), source="cli")
    router = NewsRouter(cfg.instruments, mode=str(cfg.get("feeds", "news_routing", default="hybrid")),
                        engine=engine)
    symbols, reason = router.route(item)
    print(f"routing: {symbols or '(none)'}  —  {reason}")

    for sym in (symbols or (symbol,)):
        state = render_state(price.snapshot(sym), [item], cfg.instrument(sym), focus=item)
        print("\n" + "─" * 72)
        print(state)
        print("─" * 72)
        decision = engine.decide(state, sym, item.news_id, snap.ts)
        print(json.dumps({
            "engine": decision.engine, "model": decision.model,
            "routing_reason": decision.routing_reason,
            "direction": decision.direction, "direction_p": decision.direction_p,
            "direction_probs": decision.direction_probs,
            "materiality": decision.materiality, "conviction": decision.conviction,
            "conviction_norm": decision.conviction_norm, "priced_in": decision.priced_in,
            "context_aligned": decision.context_aligned, "horizon": decision.horizon,
            "risk_event": decision.risk_event, "abstained": decision.abstained,
            "latency_ms": decision.latency_ms,
        }, indent=2))
        builder = SignalBuilder(cfg)
        print(f"\nfused score: {builder.fuse(decision):+.4f}")
        if engine.name == "heuristic":
            print("note: using the offline fallback engine; --engine laya needs the checkpoint")
    return 0


# ── parser ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("jevbot", description="News-driven algo trading bot on Laya")
    p.add_argument("--version", action="version", version=f"jevbot {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", action="append", metavar="FILE", help="extra TOML config")
        sp.add_argument("--engine", choices=["auto", "laya", "heuristic"])
        sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override any config value, e.g. --set risk.max_positions=3")

    run = sub.add_parser("run", help="trade (paper by default) with the dashboard")
    common(run)
    run.add_argument("--mode", choices=["paper", "live"], default=None)
    run.add_argument("--price-feed", dest="price_feed")
    run.add_argument("--news-feed", dest="news_feed")
    run.add_argument("--speed", type=float, help="demo-market playback speed (real seconds per sim)")
    run.add_argument("--cycles", type=int, help="stop after N cycles")
    run.add_argument("--duration", type=float, help="stop after N seconds")
    run.add_argument("--interval", type=float, help="seconds between cycles")
    run.add_argument("--host", default=None)
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--no-serve", action="store_true", help="do not start the dashboard")
    run.add_argument("--loop", dest="loop", action="store_true", default=None,
                     help="restart the demo tape when it runs out (default: on for demo feeds)")
    run.add_argument("--no-loop", dest="loop", action="store_false",
                     help="let the demo tape finish and hold the last bar")
    run.set_defaults(func=cmd_run)

    bt = sub.add_parser("backtest", help="replay the demo market")
    common(bt)
    bt.add_argument("--days", type=float, default=30.0)
    bt.add_argument("--seed", type=int, default=None)
    bt.add_argument("--seeds", help="comma-separated seeds for a multi-seed summary")
    bt.add_argument("--engines", help="comma-separated engine names")
    bt.add_argument("--step", type=float, default=900.0, help="replay step in seconds")
    bt.add_argument("--horizon", type=float, default=4 * 3600.0, help="label horizon in seconds")
    bt.add_argument("--shuffle", action="store_true", help="falsification control")
    bt.add_argument("--impact-scale", dest="impact_scale", type=float,
                    help="demo market news impact; lower it and the edge should vanish")
    bt.add_argument("--noise-share", dest="noise_share", type=float)
    bt.add_argument("--out", default="data/runs", help="write JSON/CSV here ('' to skip)")
    bt.add_argument("--db", default="", metavar="PATH",
                    help="persist this run's rows to a database (default: in-memory)")
    bt.add_argument("--quiet", action="store_true", help="skip the run summary")
    bt.set_defaults(func=cmd_backtest)

    ev = sub.add_parser("eval", help="score the engine against realised forward returns")
    common(ev)
    ev.add_argument("--seeds", default="7,11,23")
    ev.add_argument("--engines", default="heuristic")
    ev.add_argument("--days", type=float, default=20.0)
    ev.add_argument("--step", type=float, default=900.0)
    ev.add_argument("--horizon", type=float, default=4 * 3600.0)
    ev.add_argument("--control", action="store_true", help="also run the shuffled control")
    ev.add_argument("--compare", action="store_true")
    ev.set_defaults(func=cmd_eval)

    gen = sub.add_parser("gen-data", help="write the demo market to JSONL for replay")
    common(gen)
    gen.add_argument("--days", type=float, default=None)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--out", default="data/demo")
    gen.add_argument("--horizon", type=float, default=4 * 3600.0)
    gen.add_argument("--sample", type=int, default=1, help="write every Nth bar")
    gen.set_defaults(func=cmd_gen_data)

    route = sub.add_parser("route", help="explain one headline end to end")
    common(route)
    route.add_argument("text")
    route.add_argument("--symbol", default=None)
    route.set_defaults(func=cmd_route)

    smoke = sub.add_parser("testnet-order",
                           help="place one small order on the sandbox venue and report the result")
    common(smoke)
    smoke.add_argument("--symbol", default=None, help="defaults to the first symbol in the universe")
    smoke.add_argument("--notional", type=float, default=12.0, help="order size in quote currency")
    smoke.add_argument("--side", default="buy", choices=["buy", "sell"])
    smoke.add_argument("--yes", action="store_true", help="actually place it")
    smoke.set_defaults(func=cmd_testnet_order)

    doc = sub.add_parser("doctor", help="check the environment")
    common(doc)
    doc.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except LiveTradingRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except EngineUnavailable as exc:
        print(f"engine unavailable: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
