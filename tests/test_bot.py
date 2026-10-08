"""The loop, driven by a hand-held clock.

Everything here is about the seams: the bot must use the clock it is given
(never the wall clock), must record what it decided, and must stop when told.
"""

from __future__ import annotations

import time

import pytest

from jevbot.feeds.base import FeedError, PriceFeed
from jevbot.types import MarketSnapshot


def test_the_bot_uses_the_clock_it_was_given(rig):
    bot, clock, market = rig
    assert bot.now() == pytest.approx(clock["now"])
    result = bot.cycle()
    assert result.ts == pytest.approx(clock["now"])


def test_a_cycle_records_everything_the_dashboard_reads(rig):
    bot, clock, market = rig
    result = bot.cycle()
    assert result.cycle == 1
    counts = bot.store.counts()
    assert counts["equity"] == 1, "the cycle has to leave an equity point"
    assert counts["news"] >= 1, "the demo market owes us headlines within six hours"
    assert counts["decisions"] >= 1
    assert bot.status()["cycle"] == 1


def test_run_stops_after_the_requested_number_of_cycles(rig):
    bot, clock, market = rig
    results = []
    for i in range(5):
        clock["now"] += 300.0
        results.append(bot.cycle())
    assert [r.cycle for r in results] == [1, 2, 3, 4, 5]


def test_pausing_leaves_the_book_alone(rig):
    bot, clock, market = rig
    bot.cycle()
    equity = bot.portfolio.equity()
    bot.pause(True)
    assert bot.paused
    bot.pause(False)
    assert not bot.paused
    assert bot.portfolio.equity() == pytest.approx(equity)


def test_request_stop_is_visible_to_the_run_loop(rig):
    bot, clock, market = rig
    bot.request_stop()
    assert bot.stop_requested
    out = bot.run(cycles=3, interval=0.0)
    assert out == [], "a stop request must beat the cycle count"


def test_a_cycle_without_a_snapshot_does_not_crash(rig, monkeypatch):
    bot, clock, market = rig
    monkeypatch.setattr(bot.price_feed, "snapshots", lambda *a, **k: {})
    result = bot.cycle()
    assert result.decisions == 0 and result.orders == 0


def test_a_broker_with_a_foreign_portfolio_is_refused(cfg, market):
    """The failure mode: fills booked into a book nothing else can see.

    A bot wired to a mismatched broker keeps trading, reports a flat account,
    and never books a fill — no exception, no warning, just a strategy that
    appears to do nothing.
    """
    from jevbot.bot import TradingBot
    from jevbot.brokers.paper import PaperBroker
    from jevbot.engine import HeuristicEngine
    from jevbot.feeds.demo import DemoNewsFeed, DemoPriceFeed
    from jevbot.portfolio import Portfolio

    price = DemoPriceFeed(market)
    with pytest.raises(ValueError, match="different Portfolio"):
        TradingBot(cfg, engine=HeuristicEngine(), price_feed=price,
                   news_feed=DemoNewsFeed(market, price),
                   broker=PaperBroker(Portfolio.fresh(1_000.0)))


def test_a_stop_out_does_not_get_reversed_by_the_rebalance(rig, monkeypatch):
    """The live bug: two orders for one symbol in one cycle."""
    from jevbot.types import Fill, Side

    bot, clock, market = rig
    snap = bot.price_feed.snapshot(bot.symbols[0], clock["now"])
    bot.portfolio.apply_fill(Fill(symbol=snap.symbol, side=Side.BUY, qty=2.0,
                                  price=snap.last, fee=0.0, ts=clock["now"]))
    monkeypatch.setattr(bot, "_stops_hit", lambda snapshots: [(snap.symbol, "test stop")])
    emitted: list = []
    original = bot.submit

    def capture(orders, snapshots, ts=None):
        emitted.extend(list(orders))
        return original(orders, snapshots, ts)

    monkeypatch.setattr(bot, "submit", capture)
    result = bot.cycle()
    symbols = [o.symbol for o in emitted]
    assert len(symbols) == len(set(symbols)), f"duplicate orders: {symbols}"
    assert result.orders == len(emitted)


def test_events_are_kept_for_the_operator(rig):
    bot, clock, market = rig
    bot.cycle()
    kinds = {e["kind"] for e in bot.events}
    assert "boot" in kinds
    assert any(k in kinds for k in {"news", "decide", "risk", "order"})


# ── a dead feed is reported once, not once per cycle ─────────────────────────
# The dashboard's event list holds 400 entries. A feed that is down for an hour
# would fill all of them with the same sentence and push out everything else.


class _FlakyFeed(PriceFeed):
    """Serves a snapshot until told to fail, then raises the same error."""

    name = "flaky"

    def __init__(self, price: float = 100.0):
        self.price = price
        self.fail = False
        self.polls = 0

    def info(self):
        return {"feed": self.name, "synthetic": False}

    def snapshot(self, symbol, now=None):
        self.polls += 1
        if self.fail:
            raise FeedError("could not fetch markets from binance testnet (https://testnet.binance.vision/api/v3): refused")
        return MarketSnapshot(symbol=symbol, ts=now or 1_700_000_000.0, last=self.price)


def _flaky_bot(cfg, feed, market):
    """A bot around a stand-in price feed, with the demo news feed it needs."""
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine
    from jevbot.feeds.demo import DemoNewsFeed

    return TradingBot(cfg, engine=HeuristicEngine(), price_feed=feed,
                      news_feed=DemoNewsFeed(market, None))


def test_a_repeated_feed_failure_is_logged_once_and_counted(cfg, market):
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine
    from jevbot.feeds.base import FeedError

    feed = _FlakyFeed()
    bot = _flaky_bot(cfg, feed, market)
    bot.cycle()
    feed.fail = True
    results = bot.run(cycles=5, interval=0)
    errors = [e for e in bot.events if e["kind"] == "error"]
    assert len(errors) == 1, f"expected one error row, got {len(errors)}: {errors}"
    assert "testnet.binance.vision" in errors[0]["message"]
    assert results == [], "a cycle that raised must not be reported as a cycle"
    assert bot._cycles_since_error == 5  # noqa: SLF001 - the count is the point


def test_a_recovered_feed_says_so(cfg, market):
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine

    feed = _FlakyFeed()
    bot = _flaky_bot(cfg, feed, market)
    feed.fail = True
    bot.run(cycles=2, interval=0)
    feed.fail = False
    bot.run(cycles=1, interval=0)
    kinds = [e["kind"] for e in bot.events]
    assert kinds.count("error") == 1, kinds
    assert "ok" in kinds, kinds
    recovered = [e for e in bot.events if e["kind"] == "ok"][0]
    assert "recovered after" in recovered["message"]
    assert bot._cycle_error == ""  # noqa: SLF001


# ── the news feed must be visible even when prices are not ──────────────────
# A price outage used to short-circuit the cycle before the news poll, which
# made the news feed's health unobservable: "0 headlines" could mean "dead
# source" or "never asked", and the panel showed the same 0 either way.


class _DeadNews:
    name = "dead-rss"

    def info(self):
        return {"feed": self.name, "ok": False, "served": 0, "total": 0,
                "errors": ["https://feeds.example/x: URLError: timed out"],
                "status": "0/1 sources reachable — no headlines are arriving; "
                          "first failure: https://feeds.example/x: URLError: timed out"}

    def poll(self, now=None):
        return []


def test_a_dead_news_feed_is_reported_once_not_once_per_cycle(cfg, market):
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine

    feed = _FlakyFeed()
    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=feed, news_feed=_DeadNews())
    bot.cycle()
    errors = [e for e in bot.events if e["kind"] == "error"
              and "not delivering headlines" in e["message"]]
    assert len(errors) == 1, [e["message"] for e in errors]
    assert "feeds.example" in errors[0]["message"]
    assert bot.status()["news_ok"] is False

    # still dead, still one row
    for _ in range(5):
        bot.cycle()
    again = [e for e in bot.events if e["kind"] == "error"
             and "not delivering headlines" in e["message"]]
    assert len(again) == 1, "a repeated failure is not five failures"


def test_news_is_polled_even_when_the_price_feed_returns_nothing(cfg, market):
    """Text arrives on its own schedule; a price outage must not blind us to it."""
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine

    class Silent(_FlakyFeed):
        def snapshot(self, symbol, now=None):
            return None

    class CountingNews:
        name = "counting"

        def __init__(self):
            self.polls = 0

        def info(self):
            return {"feed": self.name, "ok": True}

        def poll(self, now=None):
            self.polls += 1
            return []

    price, news = Silent(), CountingNews()
    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=price, news_feed=news)
    result = bot.cycle()
    assert news.polls == 1, "the news feed was skipped because prices were missing"
    assert result.decisions == 0 and result.orders == 0
    warns = [e for e in bot.events if e["kind"] == "warn"]
    assert len(warns) == 1, "the missing tape is reported once, not per cycle"


def test_a_recovered_news_feed_says_so(cfg, market):
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine

    class Flapping:
        name = "flap"

        def __init__(self):
            self.ok = False

        def info(self):
            return {"feed": self.name, "ok": self.ok,
                    "status": "1/1 sources ok" if self.ok else "0/1 sources reachable"}

        def poll(self, now=None):
            return []

    news = Flapping()
    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=_FlakyFeed(), news_feed=news)
    bot.cycle()
    news.ok = True
    bot.cycle()
    kinds = [e["kind"] for e in bot.events]
    assert kinds.count("error") == 1 and "ok" in kinds, kinds
    assert bot.status()["news_ok"] is True


def test_a_repeated_risk_drop_is_reported_once(cfg, market):
    """A stale tape drops every symbol every cycle. The panel must survive it."""
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine
    from jevbot.types import MarketSnapshot, Signal

    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=_FlakyFeed(),
                     news_feed=_DeadNews())
    # the test config starts at 100k; anchor to the equity we are going to pass,
    # or the drawdown kill switch fires on the first cycle and there are no drops
    bot.risk.anchor_equity(1_000.0)
    stale = MarketSnapshot(symbol="BTC/USDT", ts=0.0, last=100.0)   # ancient
    old = time.time() - 10_000.0
    signal = Signal(symbol="BTC/USDT", ts=time.time(), score=0.9, target_weight=0.1)

    for i in range(6):
        # the age differs every cycle, which is what used to make each row unique
        verdict = bot.risk.evaluate({"BTC/USDT": signal}, {"BTC/USDT": stale},
                                    1_000.0, {}, old + i)
        bot._log_risk_drops(verdict.dropped)  # noqa: SLF001
    rows = [e for e in bot.events if e["kind"] == "risk-drop"]
    assert len(rows) == 1, [r["message"] for r in rows]
    assert "stale data" in rows[0]["message"]
