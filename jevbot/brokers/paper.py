"""Paper broker: immediate simulated fills against the current quote.

The simulation is deliberately pessimistic about the things that are usually
hand-waved:

* **Half-spread plus slippage, always paid on the side you are taking.** A buy
  pays the offer, a sell hits the bid; market orders never fill at the mid.
* **Fees on notional**, charged on entry *and* exit, booked to the portfolio.
* **Lot-size rounding toward zero**, so a fractional venue cannot fill an order
  it could not actually fill.

It is still a simulation: no queue position, no partial fills, no gap risk
between the decision and the print. The honest use of a paper broker is to
detect plumbing errors and over-trading, not to estimate live expectancy.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..portfolio import Portfolio
from ..types import Fill, MarketSnapshot, Order, Side
from .base import Broker

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    name = "paper"

    def __init__(
        self,
        portfolio: Portfolio,
        fee_bps: float = 5.0,
        slippage_bps: float = 3.0,
        half_spread_bps: float | None = None,
        allow_short: bool = True,
    ) -> None:
        super().__init__(portfolio)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.default_half_spread_bps = float(half_spread_bps if half_spread_bps is not None else 0.0)
        self.allow_short = allow_short
        self.fills: list[Fill] = []

    def can_short(self) -> bool:
        return self.allow_short

    def submit(self, order: Order, snapshot: MarketSnapshot | None,
               ts: float | None = None) -> Fill | None:
        if snapshot is None or snapshot.last <= 0:
            self.orders_rejected += 1
            self.last_error = f"no price for {order.symbol}"
            return None
        mid = snapshot.last if order.order_type == "market" else float(order.limit_price or snapshot.last)
        half_spread = (snapshot.half_spread_bps or self.default_half_spread_bps) / 10_000.0
        slip = self.slippage_bps / 10_000.0
        direction = 1 if order.side is Side.BUY else -1
        price = mid * (1.0 + direction * (half_spread + slip))

        # Sizing belongs to the risk layer, which sized this order against the
        # cash it could see. A venue does not quietly shrink your ticket to fit
        # your balance — and a simulator that does hides exactly the bug this
        # check exists to catch: an oversized order that "filled" anyway. A
        # hair of tolerance covers the price having moved between sizing and
        # submission; beyond that it is a rejection.
        notional = order.qty * price
        if order.side is Side.BUY and notional > self.portfolio.cash * 1.001 + 1e-9:
            self.orders_rejected += 1
            self.last_error = (
                f"insufficient cash for {order.symbol}: needed {notional:,.2f}, "
                f"have {self.portfolio.cash:,.2f}"
            )
            return None

        fee = abs(notional) * self.fee_bps / 10_000.0
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            fee=fee,
            ts=ts if ts is not None else time.time(),
            slippage=abs(price - mid) * order.qty,
            order_id=order.client_id,
        )
        try:
            self.portfolio.apply_fill(fill)
        except Exception as exc:  # noqa: BLE001 - a venue rejects; it does not raise
            self.orders_rejected += 1
            self.last_error = f"booking failed for {order.symbol}: {exc}"
            log.exception("paper fill could not be booked")
            return None
        self.fills.append(fill)
        self.orders_sent += 1
        return fill

    def info(self) -> dict[str, Any]:
        return {
            "broker": self.name,
            "live": False,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "half_spread_bps": self.default_half_spread_bps,
            "allow_short": self.allow_short,
            "orders_sent": self.orders_sent,
            "orders_rejected": self.orders_rejected,
            "fills": len(self.fills),
            "last_error": self.last_error,
        }
