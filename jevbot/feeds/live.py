"""Live feeds: exchange candles for prices, RSS and webhooks for text.

These are the only modules in the bot that touch the network, and they are
imported lazily so nothing else pays for their dependencies.

Price feeds poll OHLCV candles and push them through :class:`RollingFeatures`,
which is the same code path the synthetic market uses — feature definitions stay
in one place regardless of where the prices came from.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Iterable

from ..types import Instrument, MarketSnapshot, NewsItem
from ..venues import CCXTUnavailable, build_ccxt_exchange, rest_endpoint
from .base import FeedError, NewsFeed, PriceFeed
from .features import RollingFeatures


class CCXTPriceFeed(PriceFeed):
    """Crypto candles via ccxt, mainnet or testnet. Public data: no keys needed.

    ``testnet=True`` calls ``set_sandbox_mode(True)``, which is the only correct
    way to point ccxt at a sandbox — it rewrites the REST *and* WebSocket URLs
    together. Overriding ``base_url`` by hand leaves the other half aimed at the
    live venue, which is the kind of mistake that only shows up as a puzzling
    404 at 3am.
    """

    name = "ccxt"

    def __init__(
        self,
        instruments: Iterable[Instrument],
        *,
        exchange_id: str = "binance",
        bar_seconds: int = 300,
        history_bars: int = 400,
        funding: bool = True,
        testnet: bool = False,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        try:
            self.exchange = build_ccxt_exchange(
                exchange_id, api_key=api_key, api_secret=api_secret, testnet=testnet
            )
        except CCXTUnavailable as exc:
            raise FeedError(str(exc)) from exc
        self.exchange_id = exchange_id
        self.testnet = bool(testnet)
        self.symbols = [i.symbol for i in instruments if i.market == "crypto"]
        self.bar_seconds = bar_seconds
        self.history_bars = history_bars
        self.use_funding = funding
        self.features = RollingFeatures(bar_seconds=bar_seconds)
        self._last_bar: dict[str, float] = {}
        self._markets_loaded = False
        self._funding_rate: dict[str, float] = {}

    # ── provenance ─────────────────────────────────────────────────────────

    @property
    def endpoint(self) -> str:
        return rest_endpoint(self.exchange)

    def _network_error(self, exc: Exception, symbol: str, what: str) -> FeedError:
        where = f"{self.exchange_id}{' testnet' if self.testnet else ''}"
        subject = f"{what} for {symbol}" if symbol and symbol != "-" else what
        return FeedError(
            f"could not fetch {subject} from {where} "
            f"({self.endpoint or 'endpoint unknown'}): {exc}"
        )

    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            try:
                self.exchange.load_markets()
            except Exception as exc:  # pragma: no cover - network
                raise self._network_error(exc, "-", "markets") from exc
            self._markets_loaded = True
        missing = [s for s in self.symbols if s not in (self.exchange.markets or {})]
        if missing:
            available = sorted(
                s for s in (self.exchange.markets or {})
                if s.endswith("/USDT") and ":" not in s
            )[:12]
            raise FeedError(
                f"{self.exchange_id}{' testnet' if self.testnet else ''} does not list "
                f"{', '.join(missing)}. Check universe.instruments against the venue; "
                f"some USDT pairs it does list: {', '.join(available) or 'none'}"
            )

    def _candle_limit(self) -> int:
        seconds = int(self.bar_seconds)
        return seconds // 60 if seconds >= 60 else 1

    def prime(self, ts: float | None = None) -> None:
        """Load enough history for the vol and trend windows to mean something."""
        self._ensure_markets()
        for symbol in self.symbols:
            self._poll_symbol(symbol)
        if self.use_funding:
            self._refresh_funding()

    def _poll_symbol(self, symbol: str) -> None:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self._timeframe(), limit=self.history_bars)
        except Exception as exc:  # pragma: no cover - network
            raise self._network_error(exc, symbol, "candles") from exc
        if not ohlcv:
            raise FeedError(
                f"{self.exchange_id}{' testnet' if self.testnet else ''} returned no candles "
                f"for {symbol} at {self._timeframe()}"
            )
        # Each poll returns the whole window, so every poll but the first
        # replays bars the feature engine has already ingested. Those must be
        # skipped: re-feeding an older bar is what the engine rejects as
        # out-of-order, and re-feeding it politely would be worse — the window
        # would be counted several times and the volatility estimate would
        # quietly collapse. The newest bar is deliberately *not* skipped: it is
        # still forming, and the engine knows how to update a bar it has seen.
        seen = self._last_bar.get(symbol)
        newest = seen or 0.0
        for ts_ms, _o, _h, _l, close, volume in ohlcv:
            ts = ts_ms / 1000.0
            if seen is not None and ts < seen:
                continue
            self.features.update(symbol, ts, float(close), float(volume or 0.0))
            newest = max(newest, ts)
        if ohlcv:
            self._last_bar[symbol] = newest

    def _timeframe(self) -> str:
        minutes = max(1, self.bar_seconds // 60)
        return {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h"}.get(
            minutes, "5m"
        )

    def _refresh_funding(self) -> None:
        for symbol in self.symbols:
            try:
                if self.exchange.has.get("fetchFundingRate"):
                    rate = self.exchange.fetch_funding_rate(symbol)
                    value = rate.get("fundingRate")
                    if value is not None:
                        self._funding_rate[symbol] = float(value)
            except Exception:  # pragma: no cover - network
                continue

    def snapshot(self, symbol: str, now: float | None = None) -> MarketSnapshot | None:
        if symbol in self.symbols and not self.features.has(symbol):
            self.prime()
        else:
            self._poll_symbol(symbol)
        return self.features.snapshot(symbol, funding=self._funding_rate.get(symbol))

    def ticker(self, symbol: str) -> float | None:
        """The venue's latest traded price — used to verify the feed independently.

        Candles answer "what happened", a ticker answers "what is it now", and
        comparing the two catches the case this class cannot see on its own: a
        feed pointed at the right venue but the wrong market.
        """
        self._ensure_markets()
        try:
            tick = self.exchange.fetch_ticker(symbol)
        except Exception as exc:  # pragma: no cover - network
            raise self._network_error(exc, symbol, "ticker") from exc
        for key in ("last", "close", "bid", "ask"):
            value = tick.get(key)
            if value:
                return float(value)
        return None

    def info(self) -> dict[str, Any]:
        return {
            "feed": self.name,
            "exchange": self.exchange_id,
            "testnet": self.testnet,
            "endpoint": self.endpoint,
            "symbols": self.symbols,
            "bar_seconds": self.bar_seconds,
            "synthetic": False,
        }


class AlpacaPriceFeed(PriceFeed):
    """US equity bars via Alpaca. Public data needs keys; market data is free."""

    name = "alpaca"

    def __init__(
        self,
        instruments: Iterable[Instrument],
        *,
        bar_seconds: int = 300,
        history_bars: int = 400,
    ) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise FeedError(
                "alpaca-py is not installed (pip install 'jevbot[equities]')"
            ) from exc
        import os

        self._StockBarsRequest = StockBarsRequest
        self.client = StockHistoricalDataClient(
            os.environ.get("ALPACA_API_KEY_ID", ""), os.environ.get("ALPACA_API_SECRET_KEY", "")
        )
        unit = TimeFrameUnit.Minute if bar_seconds < 3600 else TimeFrameUnit.Hour
        self._timeframe = TimeFrame(max(1, bar_seconds // (60 if unit is TimeFrameUnit.Minute else 3600)), unit)
        self.symbols = [i.symbol for i in instruments if i.market == "equity"]
        self.bar_seconds = bar_seconds
        self.history_bars = history_bars
        self.features = RollingFeatures(bar_seconds=bar_seconds)
        self._primed = False

    def prime(self) -> None:
        from datetime import datetime, timedelta, timezone

        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(seconds=self.bar_seconds * self.history_bars * 2)
        req = self._StockBarsRequest(
            symbol_or_symbols=self.symbols, timeframe=self._timeframe, start=start, end=end
        )
        bars = self.client.get_stock_bars(req)
        for symbol in self.symbols:
            for bar in bars.data.get(symbol, []):
                self.features.update(symbol, bar.timestamp.timestamp(), float(bar.close), float(bar.volume or 0))
        self._primed = True

    def snapshot(self, symbol: str, now: float | None = None) -> MarketSnapshot | None:
        if not self._primed:
            self.prime()
        return self.features.snapshot(symbol)

    def info(self) -> dict[str, Any]:
        return {"feed": self.name, "symbols": self.symbols, "bar_seconds": self.bar_seconds}


class CSVPriceFeed(PriceFeed):
    """Bars from a local CSV: ``ts,symbol,price[,volume]`` (header optional)."""

    name = "csv"

    def __init__(self, path: str, clock: Any | None = None, bar_seconds: int = 300) -> None:
        import csv as _csv
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FeedError(f"csv not found: {p}")
        self.features = RollingFeatures(bar_seconds=bar_seconds)
        self.clock = clock
        self.rows = 0
        with open(p, newline="", encoding="utf-8") as fh:
            for row in _csv.reader(fh):
                if not row or not row[0] or row[0].lower() in {"ts", "timestamp", "time"}:
                    continue
                try:
                    ts = float(row[0])
                    symbol = str(row[1])
                    price = float(row[2])
                    volume = float(row[3]) if len(row) > 3 and row[3] else 0.0
                except (ValueError, IndexError):
                    continue
                self.features.update(symbol, ts, price, volume)
                self.rows += 1

    def snapshot(self, symbol: str, now: float | None = None) -> MarketSnapshot | None:
        return self.features.snapshot(symbol)

    def info(self) -> dict[str, Any]:
        return {"feed": self.name, "rows": self.rows}


# ── news ────────────────────────────────────────────────────────────────────


class RSSNewsFeed(NewsFeed):
    """Headlines from one or more RSS/Atom feeds, deduplicated by title hash."""

    name = "rss"

    def __init__(self, urls: Iterable[str], symbol_map: dict[str, tuple[str, ...]] | None = None,
                 timeout: float = 10.0) -> None:
        self.urls = [u for u in urls if u]
        if not self.urls:
            raise FeedError("no RSS urls configured (JEVBOT_RSS_URLS)")
        self.symbol_map = symbol_map or {}
        self.timeout = timeout
        self.seen: set[str] = set()

    def _tag(self, xml: str, name: str) -> list[str]:
        import re

        pattern = rf"<{name}(?:\s[^>]*)?>(.*?)</{name}>"
        return [re.sub(r"<[^>]+>", "", m).strip() for m in re.findall(pattern, xml, re.S)]

    def poll(self, now: float | None = None) -> list[NewsItem]:
        out: list[NewsItem] = []
        for url in self.urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "jevbot/0.1"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    xml = resp.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - network
                continue
            titles = self._tag(xml, "title")
            descriptions = self._tag(xml, "description")
            dates = self._tag(xml, "pubDate") or self._tag(xml, "updated")
            for i, title in enumerate(titles[1:], start=1):     # [0] is the channel title
                key = title.lower()[:120]
                if not title or key in self.seen:
                    continue
                self.seen.add(key)
                ts = time.time()
                if i - 1 < len(dates):
                    ts = _parse_rfc822(dates[i - 1]) or ts
                body = descriptions[i - 1] if i - 1 < len(descriptions) else ""
                text = title if not body else f"{title}. {body[:400]}"
                symbols: tuple[str, ...] = ()
                for token, syms in self.symbol_map.items():
                    if token.lower() in text.lower():
                        symbols = tuple(syms)
                        break
                out.append(
                    NewsItem(text=text, ts=ts, symbols=symbols, source="rss", url=url,
                             news_id=f"rss-{abs(hash(key)) % 10**10:010d}")
                )
        out.sort(key=lambda n: n.ts)
        return out

    def info(self) -> dict[str, Any]:
        return {"feed": self.name, "urls": self.urls, "seen": len(self.seen)}


class WebhookNewsFeed(NewsFeed):
    """News pushed in by an external process.

    Writes land in a directory as one JSON object per file (or per line in a
    ``queue.jsonl``), and the loop picks them up on its next poll. It is the
    integration point for whatever actually produces your text.
    """

    name = "webhook"

    def __init__(self, path: str, symbol_map: dict[str, tuple[str, ...]] | None = None) -> None:
        from pathlib import Path

        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.queue = self.dir / "queue.jsonl"
        self.symbol_map = symbol_map or {}
        self._offset = self.queue.stat().st_size if self.queue.exists() else 0

    def poll(self, now: float | None = None) -> list[NewsItem]:
        out: list[NewsItem] = []
        if not self.queue.exists():
            return out
        with open(self.queue, "r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(row.get("text") or row.get("headline") or "")
                if not text:
                    continue
                symbols = row.get("symbols") or ()
                if isinstance(symbols, str):
                    symbols = (symbols,)
                if not symbols:
                    for token, syms in self.symbol_map.items():
                        if token.lower() in text.lower():
                            symbols = tuple(syms)
                            break
                out.append(
                    NewsItem(
                        text=text,
                        ts=float(row.get("ts", time.time())),
                        symbols=tuple(symbols),
                        source=str(row.get("source", "webhook")),
                        url=str(row.get("url", "")),
                    )
                )
            self._offset = fh.tell()
        return out

    def info(self) -> dict[str, Any]:
        return {"feed": self.name, "queue": str(self.queue), "offset": self._offset}


def _parse_rfc822(value: str) -> float | None:
    from email.utils import parsedate_to_datetime

    try:
        dt = parsedate_to_datetime(value)
        return dt.timestamp()
    except Exception:  # pragma: no cover
        return None
