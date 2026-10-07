"""Shared fixtures.

The suite runs against the *offline* engine and the synthetic market on purpose:
`pytest` must pass on a machine with no network, no GPU and no API keys, which
is also the environment a contributor is most likely to have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbot.config import load_config  # noqa: E402
from jevbot.engine import HeuristicEngine  # noqa: E402
from jevbot.feeds.demo import DemoMarket  # noqa: E402
from jevbot.types import Instrument, MarketSnapshot  # noqa: E402


@pytest.fixture
def cfg():
    """A small, fast config: two symbols, a four-day market, no network."""
    c = load_config()
    c.raw["engine"]["name"] = "heuristic"
    c.raw["universe"]["instruments"] = [
        {"symbol": "BTC/USDT", "market": "crypto", "name": "Bitcoin"},
        {"symbol": "AAPL", "market": "equity", "name": "Apple"},
    ]
    c.raw["demo"]["days"] = 4
    c.raw["demo"]["headlines_per_day"] = 40
    c.raw["loop"]["bar_seconds"] = 300
    return c


@pytest.fixture
def market(cfg):
    return DemoMarket(cfg.instruments, seed=3, days=4, bar_seconds=300, headlines_per_day=40)


@pytest.fixture
def engine():
    return HeuristicEngine()


def snapshot(price: float, ts: float = 1_700_000_000.0, symbol: str = "BTC/USDT",
             spread_bps: float = 1.5) -> MarketSnapshot:
    return MarketSnapshot(symbol=symbol, ts=ts, last=price, half_spread_bps=spread_bps,
                          ret_24h=0.0, vol_24h=0.01)


@pytest.fixture
def snapshot_factory():
    return snapshot


@pytest.fixture
def instruments():
    return [
        Instrument("BTC/USDT", "crypto", "Bitcoin"),
        Instrument("AAPL", "equity", "Apple"),
    ]


@pytest.fixture
def rig(cfg, market):
    """A fully offline bot with a hand-held clock, plus the store it writes to."""
    from jevbot.bot import TradingBot
    from jevbot.brokers.paper import PaperBroker
    from jevbot.feeds.demo import DemoNewsFeed, DemoPriceFeed
    from jevbot.portfolio import Portfolio
    from jevbot.store import Store

    price = DemoPriceFeed(market, replay_from_start=True)
    news = DemoNewsFeed(market, price)
    store = Store(":memory:")
    portfolio = Portfolio.fresh(100_000.0)
    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=price, news_feed=news,
                     broker=PaperBroker(portfolio), portfolio=portfolio, store=store)
    clock = {"now": market.start_ts + 6 * 3600.0}
    bot.clock_fn = lambda: clock["now"]
    bot.warmup()
    yield bot, clock, market
    store.close()
