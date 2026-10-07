"""Engine selection.

``build_engine`` is the only place that decides which brain the bot runs on, and
it is deliberately loud about it. ``engine = "auto"`` tries Laya and falls back
to the offline engine with a warning that names the reason; ``engine = "laya"``
refuses to fall back, because a bot that silently degrades to a lexicon engine
after a network blip is worse than one that stops.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import DecisionCache, DecisionEngine, EngineUnavailable
from .heuristic import HeuristicEngine
from .laya_engine import LayaEngine
from .resilient import ResilientEngine

log = logging.getLogger(__name__)

__all__ = [
    "DecisionCache",
    "DecisionEngine",
    "EngineUnavailable",
    "HeuristicEngine",
    "LayaEngine",
    "ResilientEngine",
    "build_engine",
]


def build_engine(
    name: str = "auto",
    *,
    repo: str = "convaiinnovations/laya",
    device: str | None = None,
    preload: bool = False,
    min_confidence: float = 0.0,
    cache: bool = True,
    batch_size: int | None = None,
    max_len: int | None = None,
) -> DecisionEngine:
    """Build the configured engine, honouring the fallback policy of ``name``."""
    name = (name or "auto").lower()
    if name == "heuristic":
        return HeuristicEngine(min_confidence=min_confidence)

    try:
        engine = LayaEngine(
            repo=repo,
            device=device,
            preload=preload,
            min_confidence=min_confidence,
            cache=cache,
            batch_size=batch_size,
            max_len=max_len,
        )
    except EngineUnavailable as exc:
        if name == "laya":
            raise
        log.warning(
            "Laya unavailable (%s) — falling back to the offline heuristic engine. "
            "Signals are produced by lexicons, not by Laya. Install the engine extra "
            "(`pip install 'jevbot[engine]'`) and make sure the checkpoint is "
            "downloadable to use the real thing.",
            exc,
        )
        return HeuristicEngine(min_confidence=min_confidence)

    if name == "laya":
        return engine
    # `auto`: prefer Laya, but never let a mid-session engine failure leave the
    # bot deciding on stale signals in silence.
    return ResilientEngine(engine, HeuristicEngine(min_confidence=min_confidence))


def engine_summary(engine: DecisionEngine) -> dict[str, Any]:
    """A status dict for the dashboard, kept free of engine internals."""
    try:
        info = engine.info()
    except Exception as exc:  # pragma: no cover - defensive
        info = {"engine": engine.name, "error": str(exc)}
    info["is_laya"] = engine.name == "laya"
    return info
