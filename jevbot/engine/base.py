"""The decision-engine interface.

Two engines implement it:

``LayaEngine``
    The real thing. Sends the rendered state through Laya's ``Router``, which
    picks ``laya`` (English) or ``laya-multilingual`` per request and reports
    the reason, and returns typed answers.

``HeuristicEngine``
    A deterministic, dependency-free stand-in that answers the *same* seven
    typed questions from the *same* rendered text. It exists for three reasons:
    the pipeline must be runnable and testable with no network and no 2.4 GB
    checkpoint download; CI must not depend on a model; and it gives the Laya
    answers something to be compared against, which is the only way "the engine
    is better than a lexicon" becomes a measurable claim rather than a belief.

Both engines are text-only by construction — the heuristic parses the rendered
state rather than receiving the numbers, so a comparison between them is a fair
one. If the heuristic were handed the raw features it would have an advantage
the language model does not have, and any accuracy difference would be
uninterpretable.
"""

from __future__ import annotations

import abc
from typing import Any

from ..types import Decision


class EngineUnavailable(RuntimeError):
    """Raised when a forced engine cannot be built (e.g. ``--engine laya`` offline)."""


class DecisionEngine(abc.ABC):
    """Answers typed questions about a rendered state."""

    #: short name reported on every :class:`~jevbot.types.Decision`
    name: str = "engine"

    @abc.abstractmethod
    def decide(self, state: str, symbol: str, news_id: str, ts: float) -> Decision:
        """Answer the trading question set for one (state, symbol) pair."""

    def subject_route(self, text: str, symbols: list[str]) -> tuple[tuple[str, ...], str]:
        """Which instruments is this text about? (symbols, reason).

        A separate question set from the trading questions on purpose — see
        :mod:`jevbot.routing`. Engines that cannot answer it return no symbols
        and a reason saying so, and the caller falls back to keywords.
        """
        return (), f"{self.name} has no subject router"

    def decide_batch(
        self, requests: list[tuple[str, str, str, float]]
    ) -> list[Decision]:
        """Answer many states at once.

        The default is a loop. Laya overrides this with a genuine batched call,
        which is the throughput path (several states share one forward pass).
        """
        return [self.decide(state, sym, nid, ts) for state, sym, nid, ts in requests]

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Human-readable status for logs and the dashboard."""

    @property
    def available(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class DecisionCache:
    """A tiny LRU keyed by (state, question fingerprint).

    Backtests replay the same state more than once (cross-validation folds, a
    second risk configuration), and the live loop re-reads overlapping news
    windows. Caching the model call is what keeps the second pass cheap; the
    state text is already deterministic, so a cache hit is not an approximation.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._data: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._data:
            self.hits += 1
            value = self._data.pop(key)
            self._data[key] = value          # move to MRU
            return value
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.pop(next(iter(self._data)))

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
