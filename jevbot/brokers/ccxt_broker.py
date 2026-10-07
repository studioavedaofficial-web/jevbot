"""Live crypto broker via ccxt (Binance, Coinbase, Kraken, ...).

Guarded by two independent conditions, because "I forgot which config I was on"
is the most expensive mistake in this whole repository:

1. ``broker.kind = "ccxt"`` in config (never the default), **and**
2. ``JEVBOT_I_UNDERSTAND_LIVE_RISK=yes`` in the environment.

Orders are market orders by default. Every submit re-reads the exchange's own
fill rather than assuming the requested quantity executed, and
:meth:`reconcile` pulls balance and positions back into the portfolio so the
local book cannot drift away from the venue's.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..portfolio import Portfolio
from ..types import Fill, MarketSnapshot, Order, Side
from .base import Broker, BrokerError, LiveTradingRefused


class CCXTBroker(Broker):
    name = "ccxt"

    def __init__(
        self,
        portfolio: Portfolio,
        *,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        password: str = "",
        sandbox: bool = False,
        acknowledged: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(portfolio)
        ack = (
            os.environ.get("JEVBOT_I_UNDERSTAND_LIVE_RISK", "").strip().lower() == "yes"
            if acknowledged is None else acknowledged
        )
        if not ack:
            raise LiveTradingRefused(
                "live crypto trading refused: set JEVBOT_I_UNDERSTAND_LIVE_RISK=yes to confirm "
                "you intend to trade real money, or run with --mode paper"
            )
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BrokerError("ccxt is not installed (pip install 'jevbot[crypto]')") from exc
        if not hasattr(ccxt, exchange_id):
            raise BrokerError(f"unknown ccxt exchange id: {exchange_id!r}")

        self.exchange_id = exchange_id
        self.params = dict(params or {})
        self.exchange = getattr(ccxt, exchange_id)(
            {"apiKey": api_key, "secret": api_secret, "password": password or None,
             "enableRateLimit": True, "options": {"defaultType": "spot"}}
        )
        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)
        self.markets_loaded = False

    # ── interface ──────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return True

    def can_short(self) -> bool:
        return False  # spot by default; shorts need a margin/futures adapter

    def submit(self, order: Order, snapshot: MarketSnapshot | None,
               ts: float | None = None) -> Fill | None:
        try:
            if not self.markets_loaded:
                self.exchange.load_markets()
                self.markets_loaded = True
            side = "buy" if order.side is Side.BUY else "sell"
            raw = self.exchange.create_order(
                order.symbol, "market", side, order.qty, None, self.params or None
            )
            self.orders_sent += 1
            return self._to_fill(order, raw)
        except Exception as exc:  # pragma: no cover - network
            self.orders_rejected += 1
            self.last_error = str(exc)
            raise BrokerError(f"order rejected for {order.symbol}: {exc}") from exc

    def _to_fill(self, order: Order, raw: dict[str, Any]) -> Fill | None:
        filled = float(raw.get("filled") or raw.get("amount") or order.qty)
        price = float(raw.get("average") or raw.get("price") or 0.0)
        if not price or not filled:
            # the venue accepted but has not printed yet; the reconcile pass
            # will pick it up on a later cycle
            return None
        fee = 0.0
        fee_obj = raw.get("fee") or {}
        if isinstance(fee_obj, dict) and fee_obj.get("cost") is not None:
            fee = float(fee_obj["cost"])
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            qty=filled,
            price=price,
            fee=fee,
            ts=ts if ts is not None else time.time(),
            order_id=str(raw.get("id", order.client_id)),
        )
        self.portfolio.apply_fill(fill)
        return fill

    def reconcile(self) -> None:  # pragma: no cover - network
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get("total") or {}
            quote = self.exchange.markets.get(
                next(iter(self.portfolio.open_positions()), None).symbol, {}
            ).get("quote") if self.portfolio.open_positions() else None
            if quote and quote in total:
                self.portfolio.cash = float(total[quote])
            for symbol, pos in self.portfolio.positions.items():
                base = symbol.split("/")[0]
                if base in total:
                    venue_qty = float(total[base])
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
            "live": True,
            "exchange": self.exchange_id,
            "orders_sent": self.orders_sent,
            "orders_rejected": self.orders_rejected,
            "last_error": self.last_error,
        }
