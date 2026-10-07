"""The loop, driven by a hand-held clock.

Everything here is about the seams: the bot must use the clock it is given
(never the wall clock), must record what it decided, and must stop when told.
"""

from __future__ import annotations

import pytest

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
