"""Turning approved target weights into orders.

Separation of concerns, stated plainly: the risk governor approves *weights*,
this module decides *shares*. It is the only place that knows about lot sizes,
minimum notionals, cash availability, protective stops and trade-count budgets.

Two things here are less obvious than they look:

* **Exits before entries.** Every cycle is ordered so that risk-reducing trades
  are priced and sent first. If the trade budget or the data window runs out
  mid-cycle, the book is left smaller, not larger.
* **Stops are recomputed, not remembered.** A stop is derived from current
  volatility and the position's entry, so a position that has moved in your
  favour and a position that has been sitting still do not share a stop distance
  that was appropriate for neither.
"""

from __future__ import annotations

import math
import time
from typing import Iterable

from .portfolio import Portfolio
from .text import clip
from .types import MarketSnapshot, Order, Side


def lot_round(qty: float, step: float, allow_fractional: bool = True) -> float:
    """Round a quantity onto the venue's lot grid, always toward zero."""
    if not allow_fractional:
        step = 1.0
    step = step or 1e-6
    steps = math.floor(abs(qty) / step + 1e-9)
    return math.copysign(steps * step, qty)


def quote_to_base(notional: float, price: float, step: float, fractional: bool) -> float:
    if price <= 0:
        return 0.0
    return lot_round(notional / price, step, fractional)


def plan_orders(
    portfolio: Portfolio,
    approved: dict[str, float],
    snapshots: dict[str, MarketSnapshot],
    *,
    min_order_notional: float = 25.0,
    max_order_notional: float = 0.0,
    max_trades: int = 4,
    lot_step: float = 1e-6,
    allow_fractional: bool = True,
    allow_short: bool = True,
    half_spread_bps: float = 1.5,
    fee_bps: float = 5.0,
    rebalance_tolerance: float = 0.01,
    rebalance_tolerance_relative: float = 0.25,
    reason: str = "",
    ts: float | None = None,
) -> list[Order]:
    """Diff the book against the approved weights and emit orders.

    Sells that shrink or close a position come first; among those, the largest
    reduction goes first. Buys are ordered by size. In both cases the trade
    budget can stop the cycle early, and it will have stopped it on the safe
    side.
    """
    equity = portfolio.equity()
    if equity <= 0:
        return []

    # available cash, net of the fee the trade itself will cost
    fee_rate = max(0.0, fee_bps) / 10_000.0
    spendable = max(0.0, portfolio.cash)

    candidates: list[tuple[float, Order, bool]] = []
    symbols = set(approved) | {s for s, p in portfolio.positions.items() if not p.is_flat}

    for symbol in symbols:
        snap = snapshots.get(symbol)
        price = (snap.last if snap else None) or portfolio.marks.get(symbol) or 0.0
        if not price:
            continue
        step = 1.0 if not allow_fractional else lot_step
        pos = portfolio.positions.get(symbol)
        current_qty = pos.qty if pos else 0.0
        current_notional = current_qty * price
        target_notional = float(approved.get(symbol, 0.0)) * equity

        # never let a short approval open an unsupported short
        if target_notional < 0 and not allow_short:
            target_notional = 0.0

        delta = target_notional - current_notional
        closing = target_notional == 0 and abs(current_qty) > 0
        # Two gates, and both matter. The notional minimum keeps small tickets
        # away from the venue; the *weight* tolerance keeps the bot from paying
        # the spread to express a change of opinion it barely has. Without the
        # second one, a signal that drifts a few basis points per cycle
        # rebalances every cycle and hands the whole edge to the exchange.
        if abs(delta) < min_order_notional and not closing:
            continue
        band = max(
            rebalance_tolerance * equity,
            rebalance_tolerance_relative * max(abs(target_notional), abs(current_notional)),
        )
        if abs(delta) < band and not closing:
            continue

        # A close is exact. Rounding a close onto the lot grid rounds it *down*,
        # which leaves a dust position behind — and a dust position is not
        # harmless: it re-triggers a close on the next cycle, forever, filling
        # the log and the fee column with one-dollar orders.
        qty = abs(current_qty) if closing else quote_to_base(abs(delta), price, step, allow_fractional)
        if qty <= 0:
            continue
        if delta > 0:                       # buying
            # `affordable` is already a *quantity* (cash / price), so it goes
            # straight onto the lot grid. Running it through quote_to_base as
            # well would divide by the price a second time and clip every buy to
            # cash/price² — which looks like a position, reports like a fill, and
            # is wrong by a factor of the price.
            affordable_qty = spendable / (price * (1.0 + fee_rate))
            qty = min(qty, lot_round(affordable_qty, step, allow_fractional))
            if qty <= 0:
                continue
            spendable -= qty * price * (1.0 + fee_rate)
            side = Side.BUY
        else:
            side = Side.SELL
            if allow_short is False and current_qty - qty < 0:
                qty = abs(min(current_qty, 0.0))
                if qty <= 0:
                    continue
        # Per-order size cap. Exits are exempt by design: clipping a close
        # leaves dust, and dust re-arms a close on every cycle forever. A ticket
        # that adds exposure is the one a fat finger or a bad weight can inflate.
        capped_by = 0.0
        increases = (current_qty >= 0) if side is Side.BUY else (current_qty < 0)
        if max_order_notional > 0 and not closing and increases:
            cap_qty = quote_to_base(max_order_notional, price, step, allow_fractional)
            if qty > cap_qty:
                capped_by = (qty - cap_qty) * price
                qty = cap_qty
                # A capped ticket that no longer clears the minimum is not worth
                # the venue's time; the next cycle can decide again.
                if qty * price < min_order_notional:
                    continue

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            reason=reason or ("close" if closing else "rebalance"),
            ts=ts if ts is not None else time.time(),
        )
        order.notes = f"capped by {capped_by:,.2f} at {max_order_notional:,.2f}/order" if capped_by else ""
        is_exit = abs(target_notional) < abs(current_notional)
        candidates.append((abs(delta), order, is_exit))

    exits = sorted([c for c in candidates if c[2]], key=lambda c: -c[0])
    entries = sorted([c for c in candidates if not c[2]], key=lambda c: -c[0])
    ordered = [c[1] for c in (exits + entries)][: max(0, max_trades)]
    return ordered


# ── protective stops ────────────────────────────────────────────────────────


def daily_vol(snap: MarketSnapshot) -> float:
    """Hourly realised vol scaled to a daily figure."""
    return max(0.0, snap.vol_24h) * math.sqrt(24.0)


def stop_distance(snap: MarketSnapshot, multiple: float = 3.0, floor: float = 0.005) -> float:
    """Volatility-scaled distance, with a floor so a flat tape still has a stop."""
    return max(floor, multiple * daily_vol(snap))


def one_order_per_symbol(orders: Iterable[Order]) -> list[Order]:
    """Keep the first order for each symbol; drop the rest.

    Every caller builds the list with the risk-reducing orders first —
    protective stops, then a flatten, then the rebalance. But they were all
    planned against the position as it stood *before* any of them filled, so a
    rebalance order that follows a stop-out is sized against a position the
    stop is already closing. Both fill, and the second one reverses the
    position the first just closed: an unintended short, plus a second round
    trip paid to get out of it. One order per symbol per cycle, and the
    risk-reducing one wins because it comes first.
    """
    out: list[Order] = []
    seen: set[str] = set()
    for order in orders:
        if order.symbol in seen:
            continue
        seen.add(order.symbol)
        out.append(order)
    return out


def update_stop_price(entry: float, snap: MarketSnapshot, long: bool,
                      multiple: float = 3.0) -> float:
    d = stop_distance(snap, multiple)
    return entry * (1.0 - d) if long else entry * (1.0 + d)


def stops_hit(portfolio: Portfolio, snapshots: dict[str, MarketSnapshot],
              multiple: float = 3.0) -> list[tuple[str, str]]:
    """Positions whose protective stop the tape has crossed this cycle."""
    out: list[tuple[str, str]] = []
    for symbol, pos in portfolio.positions.items():
        if pos.is_flat or not pos.avg_price:
            continue
        snap = snapshots.get(symbol)
        price = (snap.last if snap else None) or portfolio.marks.get(symbol) or pos.last_price
        if not price:
            continue
        stop = pos.stop_price or update_stop_price(pos.avg_price, snap, pos.qty > 0, multiple) \
            if snap else None
        if stop is None:
            continue
        if pos.qty > 0 and price <= stop:
            out.append((symbol, f"long stop {stop:.4f} vs {price:.4f}"))
        elif pos.qty < 0 and price >= stop:
            out.append((symbol, f"short stop {stop:.4f} vs {price:.4f}"))
    return out


def close_orders(portfolio: Portfolio, symbols: Iterable[str],
                 ts: float | None = None, reason: str = "protective stop") -> list[Order]:
    out = []
    for symbol in symbols:
        pos = portfolio.positions.get(symbol)
        if not pos or pos.is_flat:
            continue
        out.append(
            Order(
                symbol=symbol,
                side=Side.SELL if pos.qty > 0 else Side.BUY,
                qty=abs(pos.qty),
                reason=reason,
                ts=ts if ts is not None else time.time(),
            )
        )
    return out


def flatten_orders(portfolio: Portfolio, ts: float | None = None,
                   reason: str = "risk flatten") -> list[Order]:
    return close_orders(portfolio, [p.symbol for p in portfolio.open_positions()],
                        ts=ts, reason=reason)
