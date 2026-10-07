"""Replay feeds: a recorded headline file, and a recorded price file.

The files written by ``scripts/gen_data.py`` are line-delimited JSON, one object
per line, sorted by timestamp. Replay is the bridge between "a backtest" and
"the live loop": the same loop code runs, fired by the same clock, reading news
from a file instead of a socket — which is how a strategy's behaviour under
latency and partial information gets tested at all.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from ..types import Instrument, MarketSnapshot, NewsItem
from .base import FeedError, NewsFeed, PriceFeed
from .features import RollingFeatures


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FeedError(f"file not found: {p}")
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise FeedError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
    rows.sort(key=lambda r: float(r.get("ts", 0.0)))
    return rows


class JsonlNewsFeed(NewsFeed):
    """Serves headlines whose timestamp has arrived, in play order."""

    name = "jsonl"

    def __init__(self, path: str | Path, speed: float = 1.0, loop: bool = False) -> None:
        self.rows = load_jsonl(path)
        self.speed = max(0.001, float(speed))
        self.loop = loop
        self._t0_wall = time.time()
        self._t0_sim = float(self.rows[0]["ts"]) if self.rows else time.time()
        self._cursor = 0

    def sim_now(self, wall_now: float | None = None) -> float:
        wall_now = wall_now if wall_now is not None else time.time()
        return self._t0_sim + (wall_now - self._t0_wall) * self.speed

    def poll(self, now: float | None = None) -> list[NewsItem]:
        ts = self.sim_now() if now is None else now
        out: list[NewsItem] = []
        while self._cursor < len(self.rows) and float(self.rows[self._cursor]["ts"]) <= ts:
            row = self.rows[self._cursor]
            self._cursor += 1
            symbols = row.get("symbols") or ()
            if isinstance(symbols, str):
                symbols = (symbols,)
            out.append(
                NewsItem(
                    text=str(row.get("text", "")),
                    ts=float(row.get("ts", ts)),
                    symbols=tuple(symbols),
                    source=str(row.get("source", "replay")),
                    url=str(row.get("url", "")),
                    news_id=str(row.get("news_id", f"jsonl-{self._cursor:05d}")),
                )
            )
        return out

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self.rows)

    def info(self) -> dict[str, Any]:
        return {
            "feed": self.name,
            "rows": len(self.rows),
            "served": self._cursor,
            "speed": self.speed,
            "exhausted": self.exhausted,
        }


class JsonlPriceFeed(PriceFeed):
    """Bars from a recorded file, advanced by the shared replay clock.

    Expects rows shaped ``{"ts": ..., "symbol": ..., "price": ..., "volume": ...}``;
    features are computed through the same :class:`RollingFeatures` as every
    other source.
    """

    name = "jsonl-prices"

    def __init__(self, path: str | Path, bar_seconds: int = 300,
                 clock: Any | None = None) -> None:
        self.rows = load_jsonl(path)
        self.bar_seconds = bar_seconds
        self.clock = clock
        self.features = RollingFeatures(bar_seconds=bar_seconds)
        self._cursor = 0

    def _advance(self, ts: float) -> None:
        while self._cursor < len(self.rows) and float(self.rows[self._cursor]["ts"]) <= ts:
            row = self.rows[self._cursor]
            self._cursor += 1
            self.features.update(
                str(row["symbol"]), float(row["ts"]), float(row["price"]),
                float(row.get("volume", 0.0) or 0.0),
            )

    def snapshot(self, symbol: str, now: float | None = None) -> MarketSnapshot | None:
        ts = now
        if ts is None and self.clock is not None:
            ts = self.clock()
        if ts is None:
            ts = time.time()
        self._advance(ts)
        return self.features.snapshot(symbol)

    def info(self) -> dict[str, Any]:
        return {"feed": self.name, "rows": len(self.rows), "served": self._cursor}
