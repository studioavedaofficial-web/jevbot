"""Broker interface.

A broker's only job is to turn an :class:`~jevbot.types.Order` into a
:class:`~jevbot.types.Fill` and own the resulting book. The paper broker does it
by simulating; the live adapters do it by calling an exchange and then
reconciling what came back into the same :class:`~jevbot.portfolio.Portfolio`.
"""

from __future__ import annotations

import abc
from typing import Any

from ..portfolio import Portfolio
from ..types import Fill, MarketSnapshot, Order


class BrokerError(RuntimeError):
    pass


class LiveTradingRefused(BrokerError):
    """Raised when live trading is requested without the explicit acknowledgement."""


class Broker(abc.ABC):
    name = "broker"

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio
        self.orders_sent = 0
        self.orders_rejected = 0
        self.last_error = ""

    @abc.abstractmethod
    def submit(self, order: Order, snapshot: MarketSnapshot | None,
               ts: float | None = None) -> Fill | None:
        """Send one order; return the fill if it executed immediately.

        ``ts`` is the caller's clock. In a backtest that is the simulated time,
        not the wall clock — a fill stamped with the wrong time lands on the
        wrong equity point and makes every holding-period number meaningless.
        """

    @property
    def is_live(self) -> bool:
        return False

    def can_short(self) -> bool:
        return True

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Status for logs and the dashboard."""

    def reconcile(self) -> None:  # pragma: no cover - no-op for paper
        """Pull broker-side state back into the portfolio (live adapters)."""
        return None
