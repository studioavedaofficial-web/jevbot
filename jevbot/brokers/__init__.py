"""Broker construction. Paper is the default and live requires a second key."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..portfolio import Portfolio
from .base import Broker, BrokerError, LiveTradingRefused
from .paper import PaperBroker

log = logging.getLogger(__name__)

__all__ = ["Broker", "BrokerError", "LiveTradingRefused", "PaperBroker", "build_broker"]


def build_broker(cfg, portfolio: Portfolio, *, force_paper: bool = False) -> Broker:
    """Build the configured broker; anything but ``paper`` needs acknowledgement."""
    kind = "paper" if force_paper else cfg.broker_kind
    if force_paper and cfg.testnet:
        # `--mode paper` with broker.testnet: keep the venue connection but do
        # not route orders to it. The price feed is unaffected either way.
        log.info("--mode paper: order routing stays local; the testnet endpoint is not used")
    if kind == "paper":
        b = cfg.section("broker")
        return PaperBroker(
            portfolio,
            fee_bps=float(b.get("fee_bps", 5.0)),
            slippage_bps=float(b.get("slippage_bps", 3.0)),
            half_spread_bps=float(b.get("half_spread_bps", 1.5)),
            allow_short=bool(cfg.get("risk", "allow_short", default=True)),
        )
    if kind == "ccxt":
        from .ccxt_broker import CCXTBroker

        env = os.environ
        return CCXTBroker(
            portfolio,
            exchange_id=(cfg.get("feeds", "ccxt_exchange", default=None)
                         or env.get("JEVBOT_CCXT_EXCHANGE", "binance")),
            api_key=env.get("BINANCE_API_KEY", env.get("JEVBOT_CCXT_KEY", "")),
            api_secret=env.get("BINANCE_API_SECRET", env.get("JEVBOT_CCXT_SECRET", "")),
            testnet=cfg.testnet,
        )
    if kind == "alpaca":
        from .alpaca_broker import AlpacaBroker

        env = os.environ
        return AlpacaBroker(
            portfolio,
            api_key=env.get("ALPACA_API_KEY_ID", ""),
            api_secret=env.get("ALPACA_API_SECRET_KEY", ""),
            paper=env.get("ALPACA_PAPER", "1").strip().lower() not in {"0", "false", "no"},
        )
    raise BrokerError(f"unknown broker kind: {kind!r} (expected paper | ccxt | alpaca)")


def broker_summary(broker: Broker) -> dict[str, Any]:
    info = broker.info()
    info["is_live"] = broker.is_live
    return info
