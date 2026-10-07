"""Feed interfaces: where prices and text come from.

A price feed answers "what is the state of this instrument now"; a news feed
answers "what has been said since I last asked". Both are pull-based, because
the bot's loop is a pull loop, and both are allowed to be backed by a synthetic
market, a CSV file, an exchange or an RSS reader without the strategy noticing.
"""

from __future__ import annotations

import abc
from typing import Any, Iterable

from ..types import Instrument, MarketSnapshot, NewsItem


class FeedError(RuntimeError):
    pass


class PriceFeed(abc.ABC):
    name = "prices"

    @abc.abstractmethod
    def snapshot(self, symbol: str, now: float | None = None) -> MarketSnapshot | None:
        """The instrument's current numeric state, or None if unavailable."""

    def snapshots(self, symbols: Iterable[str], now: float | None = None) -> dict[str, MarketSnapshot]:
        out = {}
        for symbol in symbols:
            snap = self.snapshot(symbol, now)
            if snap is not None:
                out[symbol] = snap
        return out

    @property
    def resettable(self) -> bool:
        return False

    def reset(self) -> None:  # pragma: no cover - optional
        return None

    def info(self) -> dict[str, Any]:
        return {"feed": self.name}

    def close(self) -> None:  # pragma: no cover - optional
        return None


class NewsFeed(abc.ABC):
    name = "news"

    @abc.abstractmethod
    def poll(self, now: float | None = None) -> list[NewsItem]:
        """News items that have not been returned before, oldest first."""

    def info(self) -> dict[str, Any]:
        return {"feed": self.name}

    def close(self) -> None:  # pragma: no cover - optional
        return None
