"""Keep trading when the model does not.

A checkpoint on a network filesystem, a GPU that a neighbour just filled, an
OOM after six hours of uptime — the engine is the least reliable component in
the system and the only one whose failure mode is *silence*. A bot whose engine
starts throwing does not stop trading; it stops deciding, and then does whatever
its last signal said, indefinitely.

``ResilientEngine`` makes that failure loud and bounded: the first hard failure
is logged with the underlying exception, and the engine that answers after it is
named on every decision so no downstream report can attribute a lexicon's answer
to Laya.
"""

from __future__ import annotations

import logging
from typing import Any

from ..types import Decision
from .base import DecisionEngine, EngineUnavailable
from .heuristic import HeuristicEngine

log = logging.getLogger(__name__)


class ResilientEngine(DecisionEngine):
    """Wraps a preferred engine, degrading to a fallback that keeps working."""

    def __init__(self, primary: DecisionEngine, fallback: DecisionEngine | None = None,
                 max_failures: int = 2) -> None:
        self.primary = primary
        self.fallback = fallback or HeuristicEngine()
        self.max_failures = max(1, max_failures)
        self.failures = 0
        self.degraded = False
        self.last_error = ""

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.fallback.name if self.degraded else self.primary.name

    @property
    def available(self) -> bool:
        return True

    @property
    def active(self) -> DecisionEngine:
        return self.fallback if self.degraded else self.primary

    def _degrade(self, exc: Exception) -> None:
        self.failures += 1
        self.last_error = str(exc)
        if not self.degraded and self.failures >= self.max_failures:
            self.degraded = True
            log.error(
                "engine %s failed %d times (%s) — switching to the offline %s engine. "
                "Signals after this point are NOT from %s.",
                self.primary.name, self.failures, exc, self.fallback.name, self.primary.name,
            )

    def decide(self, state: str, symbol: str, news_id: str, ts: float) -> Decision:
        if self.degraded:
            return self.fallback.decide(state, symbol, news_id, ts)
        try:
            return self.primary.decide(state, symbol, news_id, ts)
        except Exception as exc:  # noqa: BLE001 - any engine failure is handled the same way
            self._degrade(exc)
            return self.fallback.decide(state, symbol, news_id, ts)

    def decide_batch(self, requests: list[tuple[str, str, str, float]]) -> list[Decision]:
        if self.degraded:
            return self.fallback.decide_batch(requests)
        try:
            out = self.primary.decide_batch(requests)
            if len(out) != len(requests):
                raise RuntimeError(
                    f"{self.primary.name} returned {len(out)} answers for {len(requests)} requests"
                )
            return out
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            return self.fallback.decide_batch(requests)

    def subject_route(self, text: str, symbols: list[str]) -> tuple[tuple[str, ...], str]:
        if self.degraded:
            return (), f"degraded to {self.fallback.name}"
        try:
            return self.primary.subject_route(text, symbols)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            return (), f"engine error: {exc}"

    def info(self) -> dict[str, Any]:
        base = self.active.info()
        base["wrapped"] = f"{self.primary.name}->{self.fallback.name}"
        base["degraded"] = self.degraded
        base["failures"] = self.failures
        base["last_error"] = self.last_error
        base["active"] = self.active.name
        if self.degraded:
            base["note"] = (
                f"{self.primary.name} failed ({self.last_error}); answers are coming from the "
                f"offline {self.fallback.name} engine"
            )
        return base

    def close(self) -> None:
        for engine in (self.primary, self.fallback):
            try:
                engine.close()
            except Exception:  # pragma: no cover
                pass
