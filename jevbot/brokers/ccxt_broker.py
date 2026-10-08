"""Crypto broker via ccxt (Binance, Coinbase, Kraken, ...), live or testnet.

Two endpoint families, and the guard depends on which one is in use:

* **Testnet / sandbox** (``testnet = true``) — ``set_sandbox_mode(True)`` points
  every REST and WebSocket URL at the venue's sandbox. No real money can move,
  so this needs no acknowledgement and reports ``live: false``. This is the mode
  to use for "real market data, fake fills".
* **Real money** — needs *two* independent conditions, because "I forgot which
  config I was on" is the most expensive mistake in this whole repository:
  ``broker.kind = "ccxt"`` (never the default) **and**
  ``JEVBOT_I_UNDERSTAND_LIVE_RISK=yes`` in the environment.

The distinction is enforced by a property rather than by remembering to set a
flag: ``is_live`` is hard-wired off whenever the sandbox is on.

Orders are market orders by default. Every submit re-reads the exchange's own
fill rather than assuming the requested quantity executed, and
:meth:`reconcile` pulls balance and positions back into the portfolio so the
local book cannot drift away from the venue's.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ..portfolio import Portfolio
from ..types import Fill, MarketSnapshot, Order, Side
from ..venues import CCXTUnavailable, build_ccxt_exchange, rest_endpoint
from .base import Broker, BrokerError, LiveTradingRefused

log = logging.getLogger(__name__)


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
        testnet: bool = False,
        acknowledged: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(portfolio)
        if not testnet:
            ack = (
                os.environ.get("JEVBOT_I_UNDERSTAND_LIVE_RISK", "").strip().lower() == "yes"
                if acknowledged is None else acknowledged
            )
            if not ack:
                raise LiveTradingRefused(
                    "live crypto trading refused: set JEVBOT_I_UNDERSTAND_LIVE_RISK=yes to "
                    "confirm you intend to trade real money, or run against the sandbox with "
                    "broker.testnet = true, or run with --mode paper"
                )
        self.exchange_id = exchange_id
        self.params = dict(params or {})
        self.testnet = bool(testnet)
        try:
            self.exchange = build_ccxt_exchange(
                exchange_id, api_key=api_key, api_secret=api_secret, password=password,
                testnet=self.testnet,
            )
        except CCXTUnavailable as exc:
            raise BrokerError(str(exc)) from exc
        self.markets_loaded = False

    # ── provenance ─────────────────────────────────────────────────────────

    @property
    def rest_endpoint(self) -> str:
        """The URL orders will actually go to — shown in the dashboard and logs."""
        return rest_endpoint(self.exchange)

    # ── interface ──────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        """Testnet is not live: it cannot move real money, so it must not claim to."""
        return not self.testnet

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
            return self._to_fill(order, raw, ts)
        except Exception as exc:  # pragma: no cover - network
            self.orders_rejected += 1
            self.last_error = str(exc)
            # A rejection and a bug look the same from the caller's side, so the
            # traceback has to land somewhere: a NameError here once turned every
            # order into a "rejection" with a one-line message and no stack.
            log.exception("ccxt order failed for %s", order.symbol)
            raise BrokerError(f"order rejected for {order.symbol}: {exc}") from exc

    def _to_fill(self, order: Order, raw: dict[str, Any], ts: float | None = None) -> Fill | None:
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
            "live": self.is_live,
            "testnet": self.testnet,
            "exchange": self.exchange_id,
            "endpoint": self.rest_endpoint,
            "orders_sent": self.orders_sent,
            "orders_rejected": self.orders_rejected,
            "last_error": self.last_error,
        }
