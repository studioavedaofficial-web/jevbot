"""Order planning, and the quantity/notional boundary.

``quote_to_base(notional, price, step, fractional)`` turns money into an amount
of an asset. ``lot_round(quantity, step, fractional)`` snaps an amount. They are
not interchangeable, and swapping them produces a number that is small, wrong
and perfectly fillable — which is what happened once, silently.
"""

from __future__ import annotations

import pytest

from jevbot.execution import (
    close_orders,
    flatten_orders,
    lot_round,
    one_order_per_symbol,
    plan_orders,
    quote_to_base,
)
from jevbot.portfolio import Portfolio
from jevbot.types import Fill, MarketSnapshot, Order, Side

TS = 1_700_000_000.0


def snap(symbol: str, price: float, ts: float = TS) -> MarketSnapshot:
    return MarketSnapshot(symbol=symbol, ts=ts, last=price, half_spread_bps=1.0,
                          ret_24h=0.0, vol_24h=0.01)


def hold(p: Portfolio, symbol: str, qty: float, price: float, side: Side = Side.BUY) -> None:
    p.apply_fill(Fill(symbol=symbol, side=side, qty=qty, price=price, fee=0.0, ts=TS))


def test_quote_to_base_returns_a_quantity():
    assert quote_to_base(1_000.0, 100.0, 1e-6, True) == pytest.approx(10.0)


def test_a_buy_never_spends_more_than_cash():
    p = Portfolio.fresh(1_000.0)
    orders = plan_orders(p, {"X": 1.0}, {"X": snap("X", 100.0)}, min_order_notional=10.0)
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].qty * 100.0 <= 1_000.0 + 1e-9


def test_a_fully_invested_buy_moves_roughly_the_whole_book():
    """The regression, as an assertion.

    With a target weight of 1.0 and no fees the planner should buy about
    ``equity / price`` units. The double-division bug bought ``cash / price²``:
    at a price of 100 that is 1/100th of the intended size — and it still filled.
    """
    p = Portfolio.fresh(1_000.0)
    orders = plan_orders(p, {"X": 1.0}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, fee_bps=0.0, max_trades=1)
    assert orders
    assert orders[0].qty == pytest.approx(10.0, rel=0.02), f"bought {orders[0].qty}, expected ~10"


def test_buy_affordability_accounts_for_the_fee():
    p = Portfolio.fresh(100.0)
    orders = plan_orders(p, {"X": 1.0}, {"X": snap("X", 100.0)},
                         min_order_notional=1.0, fee_bps=100.0, max_trades=1)  # 1 % fee
    assert orders
    assert orders[0].qty <= 0.9902
    assert orders[0].qty * 100.0 * 1.01 <= 100.0 + 1e-6


def test_rebalance_band_suppresses_tiny_changes():
    p = Portfolio.fresh(10_000.0)
    hold(p, "X", 10.0, 100.0)
    p.mark({"X": 100.0})
    # 10 % held, 10.05 % approved — far inside the band
    assert plan_orders(p, {"X": 0.1005}, {"X": snap("X", 100.0)},
                       rebalance_tolerance=0.015, rebalance_tolerance_relative=0.25) == []


def test_closes_send_the_exact_held_quantity():
    """Rounding a close onto the lot grid leaves dust behind, and dust re-arms."""
    p = Portfolio.fresh(1_000.0)
    hold(p, "AAPL", 3.0, 100.0)
    p.positions["AAPL"].qty = 2.9999999  # an awkward amount, as fills produce
    orders = close_orders(p, ["AAPL"])
    assert len(orders) == 1
    assert orders[0].qty == pytest.approx(2.9999999)
    assert orders[0].side is Side.SELL


def test_flatten_covers_every_open_position_and_ignores_flat_ones():
    p = Portfolio.fresh(10_000.0)
    hold(p, "X", 1.0, 100.0)
    hold(p, "Y", 2.0, 50.0, side=Side.SELL)
    p.positions["Z"] = type(p.position("Z"))(symbol="Z", qty=0.0)
    orders = flatten_orders(p)
    assert {o.symbol for o in orders} == {"X", "Y"}
    assert {o.side for o in orders} == {Side.BUY, Side.SELL}


def test_lot_round_respects_fractional_permission():
    assert lot_round(1.37, 1.0, False) == pytest.approx(1.0)
    assert lot_round(1.37, 1e-6, True) == pytest.approx(1.37)


def test_no_short_is_opened_when_shorts_are_disallowed():
    p = Portfolio.fresh(1_000.0)
    orders = plan_orders(p, {"X": -0.5}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, allow_short=False)
    assert orders == []


def test_a_symbol_gets_one_order_per_cycle_and_the_stop_wins():
    """A stop-out and the rebalance behind it must not both fire.

    Both were planned against the same pre-stop position, so the second one
    fills on top of the first and reverses it — a short the bot never intended,
    plus a second round trip to escape it.
    """
    p = Portfolio.fresh(10_000.0)
    hold(p, "AAPL", 55.0, 180.0)
    stop = close_orders(p, ["AAPL"], reason="protective stop")
    approved = {"AAPL": 0.0, "BTC/USDT": 0.2}
    rebalance = plan_orders(p, approved,
                            {"AAPL": snap("AAPL", 180.0), "BTC/USDT": snap("BTC/USDT", 100.0)},
                            min_order_notional=10.0, fee_bps=0.0)
    orders = one_order_per_symbol(stop + rebalance)
    symbols = [o.symbol for o in orders]
    assert len(symbols) == len(set(symbols)), "two orders for one symbol"
    aapl = [o for o in orders if o.symbol == "AAPL"]
    assert len(aapl) == 1 and aapl[0].reason == "protective stop"


def test_order_sign_follows_side():
    assert Order(symbol="X", side=Side.SELL, qty=2.0).side.sign == -1


# ── the per-order size cap ──────────────────────────────────────────────────
# A weight bug, a price feed that returns 1.0, a fat-fingered config: any of
# them turns into one enormous ticket. The cap is the difference between a bad
# cycle and a bad day — and it must never be able to block an exit.


def test_the_order_cap_clips_a_large_buy():
    p = Portfolio.fresh(1_000.0)
    orders = plan_orders(p, {"X": 1.0}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, max_order_notional=250.0, fee_bps=0.0)
    assert len(orders) == 1
    assert orders[0].qty == pytest.approx(2.5), f"bought {orders[0].qty}, cap allows 2.5"
    assert orders[0].side is Side.BUY
    assert "capped by" in orders[0].notes


def test_the_order_cap_does_not_block_a_close():
    """A close is exact, or dust re-arms it every cycle forever."""
    p = Portfolio.fresh(1_000.0)
    hold(p, "X", 10.0, 100.0)                       # $1,000 position
    orders = plan_orders(p, {"X": 0.0}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, max_order_notional=50.0, fee_bps=0.0)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].qty == pytest.approx(10.0), "the exit must be the full position"
    assert p.positions["X"].qty == pytest.approx(10.0), "nothing was touched"


def test_the_order_cap_does_not_nibble_a_reduction_into_nothing():
    """Trimming a position is a reduction; the cap is about adding exposure."""
    p = Portfolio.fresh(1_000.0)
    hold(p, "X", 10.0, 100.0)                       # $1,000 held
    orders = plan_orders(p, {"X": 0.2}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, max_order_notional=50.0, fee_bps=0.0)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].qty == pytest.approx(8.0), "the reduction is not clipped by the entry cap"


def test_a_capped_ticket_below_the_minimum_is_dropped_entirely():
    """Clipping to $2 is worse than not trading: it pays a round trip to be dust."""
    p = Portfolio.fresh(1_000.0)
    orders = plan_orders(p, {"X": 1.0}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, max_order_notional=5.0, fee_bps=0.0)
    assert orders == []


def test_zero_means_no_cap():
    p = Portfolio.fresh(1_000.0)
    orders = plan_orders(p, {"X": 1.0}, {"X": snap("X", 100.0)},
                         min_order_notional=10.0, max_order_notional=0.0, fee_bps=0.0)
    assert orders[0].qty == pytest.approx(10.0, rel=0.02)
