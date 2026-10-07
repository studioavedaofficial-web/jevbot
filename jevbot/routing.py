"""Which instruments is this text actually about?

Every news item gets a subject before the trading questions are asked of it, and
the subject is kept strictly separate from the direction. Two reasons:

* **Cost, honestly.** Asking all seven trading questions once per (headline,
  symbol) pair multiplies the engine's work by the size of the universe. Half of
  those forward passes are spent telling the model about NVIDIA while it reads a
  story about Ethereum.
* **Correctness, less obviously.** A model asked "what does this Ethereum story
  mean for Apple" will answer. It will not answer "nothing"; it will find a
  plausible second-order channel and report it confidently. Filtering the subject
  *before* the judgement is what stops a confident non-answer from reaching the
  book.

Three strategies, in the config as ``feeds.news_routing``:

``keywords``
    Alias matching against the instrument list. Cheap, explainable, no model.
``laya``
    The ``subject`` question from :func:`jevbot.questions.routing_questions`,
    answered by the same Router that will answer the trading questions. Slower
    per headline, far better on paraphrase and on text that never names a ticker.
``hybrid`` (default)
    Keywords first; anything unmatched, or matched as market-wide, gets one Laya
    subject question. Cheap where it can be, smart where it has to be.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from .types import Instrument, NewsItem

log = logging.getLogger(__name__)

# Words that mean "this is about the whole market", not one name.
MARKET_WIDE = (
    "the market", "markets", "stocks", "equities", "crypto market", "risk assets",
    "all assets", "broad", "sector-wide", "fed ", "federal reserve", "central bank",
    "interest rate", "inflation", "recession", "geopolitic", "tariff", "global",
)

ALIASES: dict[str, tuple[str, ...]] = {
    "BTC/USDT": ("btc", "bitcoin", "xbt"),
    "ETH/USDT": ("eth", "ethereum", "ether", "vitalik"),
    "SOL/USDT": ("sol", "solana"),
    "AAPL": ("aapl", "apple", "iphone", "ipad", "macbook", "cupertino"),
    "NVDA": ("nvda", "nvidia", "geforce", "jensen huang", "blackwell"),
    "SPY": ("spy", "s&p", "s&p 500", "sp500", "index futures"),
    "TSLA": ("tsla", "tesla", "elon musk", "model y"),
    "MSFT": ("msft", "microsoft", "azure", "windows"),
}


class NewsRouter:
    """Assigns instruments to news items. Pure, deterministic, no side effects."""

    def __init__(
        self,
        instruments: Iterable[Instrument],
        mode: str = "hybrid",
        engine: Any | None = None,
        threshold: float = 0.35,
        market_wide_policy: str = "all",
    ) -> None:
        self.instruments = list(instruments)
        self.symbols = [i.symbol for i in self.instruments]
        self.mode = mode if mode in {"keywords", "laya", "hybrid"} else "hybrid"
        self.engine = engine
        self.threshold = float(threshold)
        self.market_wide_policy = market_wide_policy
        self.stats = {"keyword_hits": 0, "market_wide": 0, "laya_hits": 0, "unmatched": 0}
        self._questions = None

    # ── keyword stage ──────────────────────────────────────────────────────

    def keyword_subject(self, text: str) -> tuple[tuple[str, ...], bool]:
        """(symbols, market_wide) from alias matching."""
        low = " " + re.sub(r"\s+", " ", text.lower()) + " "
        hits: list[str] = []
        for inst in self.instruments:
            tokens = ALIASES.get(inst.symbol, ())
            if not tokens and inst.name:
                tokens = (inst.name.lower(),)
            candidates = set(tokens) | {inst.symbol.lower(), inst.symbol.split("/")[0].lower()}
            for token in candidates:
                if not token:
                    continue
                pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
                if re.search(pattern, low):
                    hits.append(inst.symbol)
                    break
        if hits:
            return tuple(sorted(set(hits))), False
        wide = any(term in low for term in MARKET_WIDE)
        return (), wide

    # ── engine stage ───────────────────────────────────────────────────────

    def _laya_questions(self):
        if self._questions is None:
            from .questions import routing_questions

            self._questions = routing_questions(self.symbols)
        return self._questions

    def laya_subject(self, text: str) -> tuple[tuple[str, ...], str]:
        """Ask the engine which instrument the text is about. Returns (symbols, reason)."""
        if self.engine is None or not getattr(self.engine, "available", False):
            return (), "engine unavailable"
        route = getattr(self.engine, "subject_route", None)
        if route is None:
            return (), f"{getattr(self.engine, 'name', 'engine')} has no subject router"
        try:
            return route(text, self.symbols)
        except Exception as exc:  # pragma: no cover - engine dependent
            log.debug("engine subject routing failed: %s", exc)
            return (), f"routing error: {exc}"

    # ── public ─────────────────────────────────────────────────────────────

    def route(self, item: NewsItem) -> tuple[tuple[str, ...], str]:
        """Symbols for one item, plus the reason, which is logged with the decision."""
        if item.symbols:
            return tuple(item.symbols), "tagged by the feed"
        if item.url and item.source.endswith("routing"):  # pragma: no cover
            pass
        symbols, wide = self.keyword_subject(item.text)
        if symbols:
            self.stats["keyword_hits"] += 1
            return symbols, "keyword match on the headline"
        if wide and self.market_wide_policy == "all":
            self.stats["market_wide"] += 1
            return tuple(self.symbols), "market-wide language, applied to all instruments"
        if self.mode in {"laya", "hybrid"} and self.engine is not None:
            picked, reason = self.laya_subject(item.text)
            if picked:
                self.stats["laya_hits"] += 1
                return picked, reason
            self.stats["unmatched"] += 1
            return (), reason
        self.stats["unmatched"] += 1
        return (), "no instrument matched"

    def route_many(self, items: Iterable[NewsItem]) -> list[tuple[NewsItem, tuple[str, ...], str]]:
        out = []
        for item in items:
            symbols, reason = self.route(item)
            out.append((item, symbols, reason))
        return out

    def info(self) -> dict[str, Any]:
        return {"mode": self.mode, "stats": dict(self.stats), "threshold": self.threshold}
