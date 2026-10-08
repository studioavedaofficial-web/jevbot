"""Brokers: the seam between an intention and a book.

The paper broker is not a stub. It is the thing that decides the price a fill
gets, and therefore the only place where fees and slippage are real. Its
arithmetic has to match the live adapters' or a backtest cannot say anything
about a live run.
"""

from __future__ import annotations

import pytest

from jevbot.brokers import build_broker
from jevbot.brokers.base import LiveTradingRefused
from jevbot.brokers.paper import PaperBroker
from jevbot.portfolio import Portfolio
from jevbot.types import Fill, MarketSnapshot, Order, Side

TS = 1_700_000_000.0


def snap(price: float = 100.0, spread_bps: float = 0.0) -> MarketSnapshot:
    return MarketSnapshot(symbol="X", ts=TS, last=price, half_spread_bps=spread_bps)


def test_a_paper_fill_books_into_the_portfolio_and_the_fee_shows_up():
    p = Portfolio.fresh(10_000.0)
    b = PaperBroker(p, fee_bps=10.0, slippage_bps=0.0, half_spread_bps=0.0)
    fill = b.submit(Order(symbol="X", side=Side.BUY, qty=10.0), snap(100.0))
    assert fill is not None
    assert fill.fee == pytest.approx(1.0)          # 10 bps of a $1,000 ticket
    assert p.cash == pytest.approx(10_000.0 - 1_000.0 - 1.0)
    assert p.positions["X"].qty == pytest.approx(10.0)
    assert b.fills and b.info()["fills"] == 1


def test_buys_pay_the_spread_and_sells_receive_it():
    """Slippage is the difference between the mid and the price you got."""
    p = Portfolio.fresh(1_000_000.0)
    b = PaperBroker(p, fee_bps=0.0, slippage_bps=0.0, half_spread_bps=10.0)
    buy = b.submit(Order(symbol="X", side=Side.BUY, qty=1.0), snap(100.0, spread_bps=10.0))
    sell = b.submit(Order(symbol="X", side=Side.SELL, qty=1.0), snap(100.0, spread_bps=10.0))
    assert buy.price > 100.0 > sell.price
    assert buy.price == pytest.approx(100.1)
    assert sell.price == pytest.approx(99.9)


def test_an_order_without_a_price_is_rejected_not_filled():
    p = Portfolio.fresh(1_000.0)
    b = PaperBroker(p)
    assert b.submit(Order(symbol="X", side=Side.BUY, qty=1.0), None) is None
    assert b.orders_rejected == 1
    assert p.cash == pytest.approx(1_000.0), "a rejected order must not move cash"


def test_an_oversized_buy_is_rejected_rather_than_quietly_shrunk():
    """The clamp used to hide an oversized order behind a "filled" status.

    A venue rejects a ticket it cannot honour. A simulator that trims it to fit
    makes a sizing bug indistinguishable from a normal fill.
    """
    p = Portfolio.fresh(1_000.0)
    b = PaperBroker(p, fee_bps=0.0, slippage_bps=0.0)
    order = Order(symbol="X", side=Side.BUY, qty=100.0)
    assert b.submit(order, snap(100.0)) is None
    assert order.qty == pytest.approx(100.0), "the order must not be mutated"
    assert p.cash == pytest.approx(1_000.0)
    assert "insufficient cash" in b.last_error


def test_a_fill_that_fails_to_book_is_reported_as_a_rejection():
    """A venue rejects; it does not raise a Python exception at the caller."""
    p = Portfolio.fresh(1_000.0)
    b = PaperBroker(p)

    def exploding(fill):
        raise RuntimeError("booking failed")

    p.apply_fill = exploding  # type: ignore[assignment]
    assert b.submit(Order(symbol="X", side=Side.BUY, qty=1.0), snap(10.0)) is None
    assert b.orders_rejected == 1
    assert not b.fills, "a fill that was not booked must not be listed as one"
    assert p.cash == pytest.approx(1_000.0)


def test_live_adapters_refuse_to_start_without_credentials():
    from jevbot.config import load_config

    cfg = load_config()
    cfg.raw.setdefault("broker", {})["kind"] = "ccxt"
    cfg.raw.setdefault("mode", {})["live"] = False
    with pytest.raises(LiveTradingRefused):
        build_broker(cfg, Portfolio.fresh(1_000.0))


def test_paper_is_the_default_broker(cfg):
    b = build_broker(cfg, Portfolio.fresh(1_000.0))
    assert b.name == "paper"
    assert b.info()["live"] is False


def test_rows_fill_round_trip_through_the_book():
    p = Portfolio.fresh(10_000.0)
    f = Fill(symbol="X", side=Side.BUY, qty=2.0, price=50.0, fee=0.05, ts=TS)
    p.apply_fill(f)
    assert f.notional == pytest.approx(100.0)
    assert p.to_dict()["positions"][0]["symbol"] == "X"


# ── a plain-string side must not invert an order ─────────────────────────────
# `Side` is a str enum, so "buy" == Side.BUY is True while `"buy" is Side.BUY`
# is False. Every broker picks direction with `is`. A side that arrives as a
# string — from JSON, a script, or a test — used to pass every equality check
# and then fill the wrong way round: a buy below the mid, a stop that adds to
# the position it was meant to close. Found by the venue integration test.


def test_order_normalises_a_string_side():
    assert Order(symbol="BTC/USDT", side="buy", qty=1.0).side is Side.BUY
    assert Order(symbol="BTC/USDT", side="sell", qty=1.0).side is Side.SELL


def test_fill_normalises_a_string_side():
    assert Fill(symbol="BTC/USDT", side="buy", qty=1.0, price=100.0, fee=0.0, ts=0.0).side is Side.BUY


def test_a_buy_stated_as_a_string_still_fills_above_the_mid():
    from jevbot.brokers import PaperBroker
    from jevbot.types import MarketSnapshot, Side

    broker = PaperBroker(Portfolio.fresh(100_000.0), fee_bps=0.0, slippage_bps=10.0)
    snap = MarketSnapshot(symbol="BTC/USDT", ts=1.0, last=30_000.0, half_spread_bps=5.0)
    fill = broker.submit(Order(symbol="BTC/USDT", side="buy", qty=0.1), snap, ts=1.0)
    assert fill is not None
    assert fill.price > snap.last, "a buy must cross the spread, not collect it"
    assert fill.side is Side.BUY


def test_a_sell_stated_as_a_string_still_fills_below_the_mid():
    from jevbot.brokers import PaperBroker
    from jevbot.types import MarketSnapshot

    broker = PaperBroker(Portfolio.fresh(100_000.0), fee_bps=0.0, slippage_bps=10.0)
    snap = MarketSnapshot(symbol="BTC/USDT", ts=1.0, last=30_000.0, half_spread_bps=5.0)
    fill = broker.submit(Order(symbol="BTC/USDT", side="sell", qty=0.1), snap, ts=1.0)
    assert fill is not None
    assert fill.price < snap.last
