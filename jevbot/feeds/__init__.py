"""Feed construction: one place that maps config to concrete feeds.

Everything downstream sees only :class:`~jevbot.feeds.base.PriceFeed` and
:class:`~jevbot.feeds.base.NewsFeed`, so adding a venue means adding a class
here and a branch below — never touching the strategy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..types import Instrument
from .base import FeedError, NewsFeed, PriceFeed

log = logging.getLogger(__name__)

__all__ = [
    "FeedError",
    "NewsFeed",
    "PriceFeed",
    "build_price_feed",
    "build_news_feed",
    "symbol_map_from_instruments",
]


def symbol_map_from_instruments(instruments: list[Instrument]) -> dict[str, tuple[str, ...]]:
    """Keyword -> symbols, for tagless feeds (RSS and webhooks).

    Deliberately crude and deliberately explicit: an unmatched headline is
    market-wide, which means the engine may read it for every instrument rather
    than the bot inventing a subject it cannot verify.
    """
    mapping: dict[str, tuple[str, ...]] = {}
    for inst in instruments:
        keys = {inst.symbol, inst.symbol.split("/")[0], inst.name}
        for key in keys:
            if key:
                mapping[key.lower()] = mapping.get(key.lower(), ()) + (inst.symbol,)
    return {k: tuple(sorted(set(v))) for k, v in mapping.items()}


def build_price_feed(cfg, *, speed: float = 1.0, clock: Any | None = None,
                     market: Any | None = None) -> PriceFeed:
    kind = str(cfg.get("feeds", "price", default="synthetic")).lower()
    instruments = cfg.instruments
    bar_seconds = int(cfg.get("loop", "bar_seconds", default=300))

    if kind in {"synthetic", "demo"}:
        from .demo import DemoMarket, DemoPriceFeed

        if market is None:
            d = cfg.section("demo")
            market = DemoMarket(
                instruments,
                seed=int(d.get("seed", 7)),
                days=float(d.get("days", 30)),
                bar_seconds=bar_seconds,
                headlines_per_day=int(d.get("headlines_per_day", 24)),
                impact_scale=float(d.get("impact_scale", 0.012)),
                noise_share=float(d.get("noise_share", 0.30)),
            )
        return DemoPriceFeed(market, speed=speed)

    if kind == "csv":
        from .live import CSVPriceFeed

        path = str(cfg.get("feeds", "csv_path", default=os.environ.get("JEVBOT_CSV_PATH", "")))
        if not path:
            raise FeedError("feeds.csv_path (or JEVBOT_CSV_PATH) is required for the csv feed")
        return CSVPriceFeed(path, clock=clock, bar_seconds=bar_seconds)

    if kind == "ccxt":
        from .live import CCXTPriceFeed

        feed = CCXTPriceFeed(
            instruments,
            exchange_id=(cfg.get("feeds", "ccxt_exchange", default=None)
                         or os.environ.get("JEVBOT_CCXT_EXCHANGE", "binance")),
            bar_seconds=bar_seconds,
            # Default to the broker's endpoint family so a testnet run cannot
            # end up reading mainnet prices (or the reverse) by omission.
            testnet=bool(cfg.get("feeds", "testnet", default=cfg.testnet)),
            api_key=os.environ.get("BINANCE_API_KEY", ""),
            api_secret=os.environ.get("BINANCE_API_SECRET", ""),
        )
        # Deliberately not primed here. Priming is a network call, and a builder
        # that does I/O makes every caller fail at construction — including the
        # dashboard process that exists to *report* the outage. TradingBot.warmup
        # primes; snapshot() primes on first use for anything that skips warmup.
        return feed

    if kind == "alpaca":
        from .live import AlpacaPriceFeed

        return AlpacaPriceFeed(instruments, bar_seconds=bar_seconds)

    if kind in {"jsonl", "replay"}:
        from .jsonl import JsonlPriceFeed

        path = str(cfg.get("feeds", "price_path", default="data/demo/prices.jsonl"))
        return JsonlPriceFeed(cfg.resolve_path(path), bar_seconds=bar_seconds, clock=clock)

    raise FeedError(f"unknown price feed: {kind!r}")


def build_news_feed(cfg, *, price_feed: PriceFeed | None = None, market: Any | None = None) -> NewsFeed:
    kind = str(cfg.get("feeds", "news", default="demo")).lower()
    instruments = cfg.instruments

    if kind in {"demo", "synthetic"}:
        from .demo import DemoMarket, DemoNewsFeed, DemoPriceFeed

        if market is None:
            d = cfg.section("demo")
            market = DemoMarket(
                instruments,
                seed=int(d.get("seed", 7)),
                days=float(d.get("days", 30)),
                bar_seconds=int(cfg.get("loop", "bar_seconds", default=300)),
                headlines_per_day=int(d.get("headlines_per_day", 24)),
                impact_scale=float(d.get("impact_scale", 0.012)),
                noise_share=float(d.get("noise_share", 0.30)),
            )
        if not isinstance(price_feed, DemoPriceFeed):
            raise FeedError("the demo news feed needs the demo price feed (same simulated clock)")
        return DemoNewsFeed(market, price_feed)

    if kind == "jsonl":
        from .jsonl import JsonlNewsFeed

        path = str(cfg.get("feeds", "news_path", default="data/demo/headlines.jsonl"))
        return JsonlNewsFeed(cfg.resolve_path(path), speed=1.0)

    if kind == "rss":
        from .live import RSSNewsFeed

        # config first, then the environment: JEVBOT_RSS_URLS already lands in
        # feeds.rss_urls via ENV_MAP, so one lookup covers both.
        raw = cfg.get("feeds", "rss_urls", default=[]) or []
        urls = [str(u).strip() for u in (raw if isinstance(raw, (list, tuple)) else str(raw).split(","))]
        return RSSNewsFeed([u for u in urls if u],
                           symbol_map=symbol_map_from_instruments(instruments))

    if kind == "webhook":
        from .live import WebhookNewsFeed

        path = cfg.resolve_path(str(cfg.get("feeds", "queue_dir", default="data/queue")))
        return WebhookNewsFeed(str(path), symbol_map=symbol_map_from_instruments(instruments))

    raise FeedError(f"unknown news feed: {kind!r}")
