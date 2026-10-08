"""A stand-in for the ccxt package, precise enough to test the wiring.

The properties worth testing about a venue adapter are not "does ccxt work" —
they are "did we point it at the endpoint we think we did, and does that matter
to the guards". A fake that records calls answers both, offline, which is the
only way this can be tested in CI.
"""

from __future__ import annotations

import types
from typing import Any

MAINNET = "https://api.binance.com/api/v3"
TESTNET = "https://testnet.binance.vision/api/v3"
LISTED = ("BTC/USDT", "ETH/USDT", "SOL/USDT")


class FakeExchange:
    """Records every call; overridable failure modes for the error paths."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.calls: list[tuple[str, Any]] = []
        self.sandbox = False
        self.urls = {"api": {"public": MAINNET, "private": MAINNET}}
        self.markets: dict[str, Any] = {s: {"symbol": s} for s in LISTED}
        self.has = {"fetchOHLCV": True, "fetchTicker": True, "fetchFundingRate": False}
        # failure switches
        self.fail_markets: Exception | None = None
        self.fail_ohlcv: Exception | None = None
        self.fail_ticker: Exception | None = None
        self.fail_order: Exception | None = None
        self.base_price = 30_000.0
        self.bars = 60

    # ── endpoints ──────────────────────────────────────────────────────────
    def set_sandbox_mode(self, enabled: bool) -> None:
        self.sandbox = bool(enabled)
        self.calls.append(("set_sandbox_mode", self.sandbox))
        url = TESTNET if self.sandbox else MAINNET
        self.urls["api"] = {"public": url, "private": url}

    def load_markets(self) -> dict[str, Any]:
        self.calls.append(("load_markets", None))
        if self.fail_markets:
            raise self.fail_markets
        return self.markets

    # ── market data ────────────────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100,
                    **_: Any) -> list[list[float]]:
        self.calls.append(("fetch_ohlcv", symbol))
        if self.fail_ohlcv:
            raise self.fail_ohlcv
        step = 60_000
        start = 1_700_000_000_000
        return [
            [start + i * step, self.base_price + i, self.base_price + i + 1,
             self.base_price + i - 1, self.base_price + i + 0.5, 10.0]
            for i in range(min(limit, self.bars))
        ]

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetch_ticker", symbol))
        if self.fail_ticker:
            raise self.fail_ticker
        return {"symbol": symbol, "last": self.base_price + 1_000.0}

    def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetch_funding_rate", symbol))
        return {"fundingRate": 0.0001}

    # ── trading ────────────────────────────────────────────────────────────
    def create_order(self, symbol: str, order_type: str, side: str, amount: float,
                     price: float | None = None, params: Any = None) -> dict[str, Any]:
        self.calls.append(("create_order", (symbol, order_type, side, amount)))
        if self.fail_order:
            raise self.fail_order
        return {"id": "venue-1", "filled": amount, "average": 30_000.0,
                "fee": {"cost": 1.5, "currency": "USDT"}}

    def fetch_balance(self) -> dict[str, Any]:
        self.calls.append(("fetch_balance", None))
        return {"total": {"USDT": 1_000.0, "BTC": 0.0}}


def install(monkeypatch) -> types.ModuleType:
    """Put a fake `ccxt` in sys.modules and return it."""
    module = types.ModuleType("ccxt")
    module.__version__ = "0.0-fake"
    module.binance = FakeExchange
    module.kraken = FakeExchange
    module.__dict__["_instances"] = []

    original_init = FakeExchange.__init__

    def recording_init(self, config=None):  # type: ignore[no-untyped-def]
        original_init(self, config)
        module._instances.append(self)  # noqa: SLF001

    FakeExchange.__init__ = recording_init  # type: ignore[method-assign]
    monkeypatch.setitem(__import__("sys").modules, "ccxt", module)
    return module
