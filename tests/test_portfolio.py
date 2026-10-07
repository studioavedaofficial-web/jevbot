"""The trading book, and the arithmetic that has to be exact.

A cash ledger in a trading system is one of the few places where a rounding
mistake is indistinguishable from a strategy that works: you see a position and
a P&L either way. These tests pin the algebra down.
"""

from __future__ import annotations

import pytest

from jevbot.portfolio import Portfolio
from jevbot.types import DUST_QUANTITY, Fill, Position, Side

TS = 1_700_000_000.0


def fill(symbol: str, qty: float, price: float, side: Side, fee: float = 0.0) -> Fill:
    return Fill(symbol=symbol, side=side, qty=qty, price=price, fee=fee, ts=TS)


def test_a_buy_moves_exactly_the_notional_out_of_cash():
    p = Portfolio.fresh(10_000.0)
    p.apply_fill(fill("BTC/USDT", 1.0, 100.0, Side.BUY, fee=0.5))
    assert p.cash == pytest.approx(10_000.0 - 100.0 - 0.5)
    assert p.positions["BTC/USDT"].qty == pytest.approx(1.0)
    assert p.fees_paid == pytest.approx(0.5)


def test_buy_then_sell_returns_cash_to_start_less_fees():
    p = Portfolio.fresh(10_000.0)
    p.apply_fill(fill("BTC/USDT", 1.0, 100.0, Side.BUY, fee=0.5))
    p.mark({"BTC/USDT": 100.0})
    p.apply_fill(fill("BTC/USDT", 1.0, 100.0, Side.SELL, fee=0.5))
    assert p.positions["BTC/USDT"].is_flat
    assert p.cash == pytest.approx(10_000.0 - 1.0)
    assert p.realized_pnl == pytest.approx(-1.0)


def test_every_fill_debits_cash_by_exactly_what_was_ordered():
    """The invariant the double-division bug broke.

    It is not enough for a fill to appear: the cash movement has to equal
    ``qty * price + fee``. When it does not, equity drifts upward out of nothing
    and every downstream number is fiction.
    """
    p = Portfolio.fresh(50_000.0)
    for qty, price in ((3.0, 250.0), (0.25, 1_000.0), (7.0, 40.0)):
        before = p.cash
        held = p.positions.get("X").qty if "X" in p.positions else 0.0
        p.apply_fill(fill("X", qty, price, Side.BUY))
        assert p.cash == pytest.approx(before - qty * price), "cash moved by something else"
        assert p.positions["X"].qty == pytest.approx(held + qty)


def test_apply_fill_is_atomic():
    """Nothing may be written until everything has been computed.

    ``apply_fill`` computes the new average price, the realised delta, the stop
    flag and the cash delta before touching the object, so a failure in the
    middle cannot leave cash debited with no position to show for it.
    """

    class BadQty(float):
        """Adds like an object that fails; multiplied like one that works."""

        def __radd__(self, other):  # type: ignore[override]
            raise RuntimeError("boom")

    p = Portfolio.fresh(1_000.0)
    p.apply_fill(fill("X", 2.0, 100.0, Side.BUY))
    cash, qty = p.cash, p.positions["X"].qty

    bad = fill("X", 0.0, 100.0, Side.BUY)
    bad.qty = BadQty(1.0)
    with pytest.raises(RuntimeError):
        p.apply_fill(bad)
    assert p.cash == cash, "cash was debited by a fill that never completed"
    assert p.positions["X"].qty == qty


def test_average_price_on_add():
    p = Portfolio.fresh(10_000.0)
    p.apply_fill(fill("X", 1.0, 100.0, Side.BUY))
    p.apply_fill(fill("X", 1.0, 200.0, Side.BUY))
    assert p.positions["X"].avg_price == pytest.approx(150.0)


def test_partial_close_keeps_average_and_realises_the_difference():
    p = Portfolio.fresh(10_000.0)
    p.apply_fill(fill("X", 2.0, 100.0, Side.BUY))
    p.mark({"X": 150.0})
    p.apply_fill(fill("X", 1.0, 150.0, Side.SELL))
    assert p.positions["X"].qty == pytest.approx(1.0)
    assert p.positions["X"].avg_price == pytest.approx(100.0)
    assert p.realized_pnl == pytest.approx(50.0)


def test_flip_through_zero_starts_a_fresh_average():
    p = Portfolio.fresh(10_000.0)
    p.apply_fill(fill("X", 1.0, 100.0, Side.BUY))
    p.apply_fill(fill("X", 3.0, 120.0, Side.SELL))
    pos = p.positions["X"]
    assert pos.qty == pytest.approx(-2.0)
    assert pos.avg_price == pytest.approx(120.0)
    assert p.realized_pnl == pytest.approx(20.0)


def test_dust_is_clamped_flat_not_carried_forever():
    """A position of 1e-12 units is not a position.

    Left alone it re-arms a close on every cycle, and each "close" pays a
    minimum ticket at the venue: an edge handed to the exchange one fee at a
    time.
    """
    p = Portfolio.fresh(1_000.0)
    p.positions["X"] = Position(symbol="X", qty=1e-12, avg_price=100.0, opened_ts=TS)
    p.apply_fill(fill("X", 1e-12, 100.0, Side.SELL))
    assert p.positions["X"].is_flat
    assert p.positions["X"].qty == 0.0
    assert DUST_QUANTITY > 0


def test_equity_is_cash_plus_marks():
    p = Portfolio.fresh(1_000.0)
    p.apply_fill(fill("X", 1.0, 400.0, Side.BUY))
    p.mark({"X": 500.0})
    assert p.equity() == pytest.approx(1_100.0)
    assert p.gross_weight() == pytest.approx(500.0 / 1_100.0)
    assert p.net_weight() == pytest.approx(500.0 / 1_100.0)


def test_short_position_has_negative_net_weight():
    p = Portfolio.fresh(1_000.0)
    p.apply_fill(fill("X", 1.0, 400.0, Side.SELL))
    p.mark({"X": 300.0})
    assert p.net_weight() < 0
    assert p.equity() == pytest.approx(1_100.0)


def test_drawdown_is_measured_from_the_peak_not_the_start():
    p = Portfolio.fresh(1_000.0)
    p.mark({"X": 0.0, "Y": 0.0})
    p.cash = 1_500.0
    p.mark({})
    p.cash = 1_200.0
    p.mark({})
    assert p.drawdown() == pytest.approx(0.2, abs=1e-6)
