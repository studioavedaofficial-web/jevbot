"""Live US equity broker via Alpaca.

Same two locks as the crypto adapter (config kind + explicit acknowledgement
env var). Alpaca's own paper endpoint is the sane way to rehearse: point
``ALPACA_PAPER=1`` at it and set the acknowledgement, and orders are sent to a
real broker API with real order-state reconciliation but no real money.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..portfolio import Portfolio
from ..types import Fill, MarketSnapshot, Order, Side
from .base import Broker, BrokerError, LiveTradingRefused


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(
        self,
        portfolio: Portfolio,
        *,
        api_key: str = "",
        api_secret: str = "",
        paper: bool = True,
        acknowledged: bool | None = None,
    ) -> None:
        super().__init__(portfolio)
        ack = (
            os.environ.get("JEVBOT_I_UNDERSTAND_LIVE_RISK", "").strip().lower() == "yes"
            if acknowledged is None else acknowledged
        )
        if not ack:
            raise LiveTradingRefused(
                "live equity trading refused: set JEVBOT_I_UNDERSTAND_LIVE_RISK=yes to confirm, "
                "or run with --mode paper"
            )
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BrokerError("alpaca-py is not installed (pip install 'jevbot[equities]')") from exc

        self._MarketOrderRequest = MarketOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self.paper = bool(paper)
        self.client = TradingClient(api_key, api_secret, paper=self.paper)
        try:
            account = self.client.get_account()
            self.portfolio.cash = float(getattr(account, "cash", portfolio.cash))
        except Exception as exc:  # pragma: no cover - network
            self.last_error = f"account fetch failed: {exc}"

    @property
    def is_live(self) -> bool:
        return not self.paper

    def can_short(self) -> bool:
        return True

    def submit(self, order: Order, snapshot: MarketSnapshot | None,
               ts: float | None = None) -> Fill | None:
        try:
            req = self._MarketOrderRequest(
                symbol=order.symbol,
                qty=abs(order.qty),
                side=self._OrderSide.BUY if order.side is Side.BUY else self._OrderSide.SELL,
                time_in_force=self._TimeInForce.DAY,
            )
            submitted = self.client.submit_order(order_data=req)
            self.orders_sent += 1
            filled_qty = float(getattr(submitted, "filled_qty", 0) or 0)
            price = float(getattr(submitted, "filled_avg_price", 0) or 0)
            if not filled_qty or not price:
                return None                     # accepted, not yet filled
            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                qty=filled_qty,
                price=price,
                fee=0.0,                        # commission-free equities
                ts=ts if ts is not None else time.time(),
                order_id=str(getattr(submitted, "id", order.client_id)),
            )
            self.portfolio.apply_fill(fill)
            return fill
        except Exception as exc:  # pragma: no cover - network
            self.orders_rejected += 1
            self.last_error = str(exc)
            raise BrokerError(f"order rejected for {order.symbol}: {exc}") from exc

    def reconcile(self) -> None:  # pragma: no cover - network
        try:
            account = self.client.get_account()
            self.portfolio.cash = float(getattr(account, "cash", self.portfolio.cash))
            held = {p.symbol: p for p in self.client.get_all_positions()}
            for symbol, pos in self.portfolio.positions.items():
                venue = held.get(symbol)
                venue_qty = float(getattr(venue, "qty", 0) or 0)
                if abs(venue_qty - pos.qty) > 1e-9:
                    self.last_error = (
                        f"position drift on {symbol}: local {pos.qty} vs venue {venue_qty}"
                    )
                    pos.qty = venue_qty
        except Exception as exc:  # pragma: no cover - network
            self.last_error = f"reconcile failed: {exc}"

    def info(self) -> dict[str, Any]:
        return {
            "broker": self.name,
            "live": self.is_live,
            "paper_endpoint": self.paper,
            "orders_sent": self.orders_sent,
            "orders_rejected": self.orders_rejected,
            "last_error": self.last_error,
        }
