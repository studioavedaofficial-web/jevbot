"""The Laya decision engine.

This is the integration that matters, so it is kept thin and explicit: build a
``Router`` once, render a state, ask the seven typed questions, translate the
answers into a :class:`~jevbot.types.Decision` without losing a single field.

Three properties of Laya are used on purpose:

* **``Router``, not ``Agent``.** News is multilingual; the router detects script
  and language in well under a millisecond and sends the request to the
  checkpoint that can actually read it, and it reports *why*. That routing
  decision is logged on every :class:`Decision`, because "the signal came from a
  model that could not read the text" is exactly the failure mode that must be
  visible after the fact.
* **``predict_batch``.** A news cycle is inherently a batch: one headline,
  several instruments; or several headlines, one instrument. Batching groups
  requests that share a checkpoint and a question schema into shared forward
  passes.
* **``answer_confidence``.** It is the quantity Laya's temperature calibration
  fits, and therefore the only one the risk engine is allowed to gate on.

Nothing here reads ``confidence`` (normalised entropy) for gating, and the
abstention gate is off by default, because Laya's shipped checkpoints are
over-confident as shipped and an untuned threshold silently discards good
answers. Fit one with ``scripts/eval_engine.py`` before enabling it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

from ..questions import decision_questions, question_fingerprint
from ..types import Decision
from .base import DecisionCache, DecisionEngine, EngineUnavailable

log = logging.getLogger(__name__)

HORIZON_NAMES = ("intraday", "swing", "positional")
CONVICTION_MAX = 4.0   # top of the `conviction` score scale
PRICED_IN_MAX = 3.0    # top of the `priced_in` score scale


class LayaEngine(DecisionEngine):
    name = "laya"

    def __init__(
        self,
        repo: str = "convaiinnovations/laya",
        device: str | None = None,
        preload: bool = False,
        min_confidence: float = 0.0,
        cache: bool = True,
        batch_size: int | None = None,
        max_len: int | None = None,
        probe: bool = True,
    ) -> None:
        try:
            import laya  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on env
            raise EngineUnavailable(
                "the `laya` package is not installed (pip install 'jevbot[engine]')"
            ) from exc
        self._laya = laya
        self.repo = repo
        self.device = device or None
        self.min_confidence = float(min_confidence or 0.0)
        self.batch_size = batch_size
        self.max_len = max_len

        t0 = time.perf_counter()
        try:
            self.router = laya.Router(
                models={"english": repo, "multilingual": f"{repo}-multilingual"},
                device=self.device,
                preload=preload,
            )
        except Exception as exc:  # pragma: no cover - network / disk
            raise EngineUnavailable(f"could not build a Laya Router: {exc}") from exc
        if probe:
            # `Router()` is lazy: it builds in microseconds and defers the
            # checkpoint to the first forward pass. Constructing one therefore
            # proves nothing, and a bot that discovers the model is missing on
            # its first live decision has already been running without it.
            self._probe()
        self.load_seconds = time.perf_counter() - t0

        self.questions = decision_questions()
        self.qkey = question_fingerprint(self.questions)
        self.cache = DecisionCache() if cache else None
        self.calls = 0
        self.batched_calls = 0
        self.cache_replays = 0
        self.last_routing: dict[str, Any] = {}
        log.info(
            "Laya engine ready: repo=%s device=%s preload=%s in %.2fs",
            repo, self.device or "auto", preload, self.load_seconds,
        )

    def _probe(self) -> None:
        """One tiny forward pass, so a missing checkpoint fails here and not live."""
        questions = {
            "smoke": {
                "type": "noul",
                "instructions": "Is this text readable?",
            }
        }
        try:
            result = self.router.predict("smoke test", questions)
        except Exception as exc:  # pragma: no cover - network / disk
            raise EngineUnavailable(
                f"the Laya checkpoint could not answer a smoke test: {exc}"
            ) from exc
        if not (result or {}).get("answers"):
            raise EngineUnavailable("the Laya checkpoint returned no answers to a smoke test")

    # ── single ─────────────────────────────────────────────────────────────

    def decide(self, state: str, symbol: str, news_id: str, ts: float) -> Decision:
        return self.decide_batch([(state, symbol, news_id, ts)])[0]

    # ── batched ────────────────────────────────────────────────────────────

    def decide_batch(self, requests: Sequence[tuple[str, str, str, float]]) -> list[Decision]:
        requests = list(requests)
        if not requests:
            return []
        out: list[Decision | None] = [None] * len(requests)
        pending: list[int] = []
        for i, (state, symbol, news_id, ts) in enumerate(requests):
            cached = self._cache_get(state, symbol, news_id, ts)
            if cached is not None:
                out[i] = cached
                self.cache_replays += 1
            else:
                pending.append(i)

        if pending:
            states = [requests[i][0] for i in pending]
            t0 = time.perf_counter()
            try:
                results = self.router.predict_batch(
                    states,
                    self.questions,
                    batch_size=self.batch_size,
                    max_len=self.max_len,
                    min_confidence=self.min_confidence or None,
                )
            except TypeError:
                # Older Laya builds without the batch keyword path.
                results = [
                    self.router.predict(
                        s, self.questions, max_len=self.max_len,
                        min_confidence=self.min_confidence or None,
                    )
                    for s in states
                ]
            elapsed = (time.perf_counter() - t0) * 1000.0
            per_item = elapsed / max(1, len(states))
            self.batched_calls += 1
            for slot, i in enumerate(pending):
                state, symbol, news_id, ts = requests[i]
                dec = self._to_decision(results[slot], state, symbol, news_id, ts, per_item)
                self._cache_put(state, symbol, dec)
                out[i] = dec
        self.calls += len(requests)
        return [d for d in out if d is not None]

    # ── translation ────────────────────────────────────────────────────────

    def _to_decision(
        self,
        result: dict[str, Any],
        state: str,
        symbol: str,
        news_id: str,
        ts: float,
        latency_ms: float,
    ) -> Decision:
        answers = (result or {}).get("answers") or {}
        routing = (result or {}).get("routing") or {}
        if routing:
            self.last_routing = dict(routing)

        def ans(key: str) -> dict[str, Any]:
            return answers.get(key) or {}

        d_ans = ans("direction")
        direction = str(d_ans.get("choice", "flat"))
        if direction not in ("long", "flat", "short"):
            direction = "flat"

        conv = float(ans("conviction").get("score", 0.0) or 0.0)
        priced = float(ans("priced_in").get("score", 0.0) or 0.0)
        horizon = str(ans("horizon").get("choice", "swing"))

        answer_conf = float(d_ans.get("answer_confidence", d_ans.get("confidence", 0.0)) or 0.0)
        abstained = bool(self.min_confidence and answer_conf < self.min_confidence)

        raw_ok = bool(answers)
        if not raw_ok:
            log.warning("Laya returned no answers for %s/%s", news_id, symbol)

        return Decision(
            news_id=news_id,
            symbol=symbol,
            ts=ts,
            engine=self.name,
            model=str(routing.get("model", result.get("model", "")) or ""),
            routing_reason=str(routing.get("reason", "") or ""),
            direction=direction,
            direction_p=float(d_ans.get("probabilities", {}).get(direction, 0.0) or 0.0),
            direction_probs={k: float(v) for k, v in (d_ans.get("probabilities") or {}).items()},
            materiality=float(ans("materiality").get("noul", 0.0) or 0.0),
            conviction=conv,
            conviction_norm=max(0.0, min(1.0, conv / CONVICTION_MAX)),
            priced_in=max(0.0, min(PRICED_IN_MAX, priced)),
            context_aligned=float(ans("context_aligned").get("noul", 0.0) or 0.0),
            horizon=horizon if horizon in HORIZON_NAMES else "swing",
            risk_event=float(ans("risk_event").get("noul", 0.0) or 0.0),
            abstained=abstained,
            latency_ms=round(latency_ms, 3),
            raw={
                "answers": answers,
                "routing": routing,
                "usage": (result or {}).get("usage"),
                "abstention": {
                    a.get("abstention") for a in answers.values() if isinstance(a, dict) and "abstention" in a
                } or None,
            },
        )

    # ── cache ──────────────────────────────────────────────────────────────

    def _key(self, state: str, symbol: str) -> str:
        import hashlib

        payload = f"{self.repo}\x00{self.qkey}\x00{symbol}\x00{state}"
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=24).hexdigest()

    def _cache_get(self, state: str, symbol: str, news_id: str, ts: float) -> Decision | None:
        if not self.cache:
            return None
        hit = self.cache.get(self._key(state, symbol))
        if hit is None:
            return None
        # Same answers, new provenance: the replay is attributed to this
        # headline rather than silently reporting the first caller's ids.
        return Decision(**{**hit.to_dict(), "news_id": news_id, "ts": ts}, raw=hit.raw)

    def _cache_put(self, state: str, symbol: str, dec: Decision) -> None:
        if self.cache:
            self.cache.put(self._key(state, symbol), dec)

    # ── status ─────────────────────────────────────────────────────────────

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "available": True,
            "repo": self.repo,
            "device": self.device or "auto",
            "checkpoints": list(self.router.models.values()) if hasattr(self.router, "models") else [],
            "preloaded": len(getattr(self.router, "_loaded", {}) or {}),
            "min_confidence_gate": self.min_confidence or None,
            "calls": self.calls,
            "batches": self.batched_calls,
            "cache_replays": self.cache_replays,
            "cache_size": len(self.cache) if self.cache else 0,
            "cache_hit_rate": round(self.cache.hit_rate, 3) if self.cache else 0.0,
            "load_seconds": round(self.load_seconds, 2),
            "last_routing": self.last_routing,
            "questions": list(self.questions.keys()),
        }

    def subject_route(self, text: str, symbols: list[str]) -> tuple[tuple[str, ...], str]:
        """One cheap forward pass asking which instrument a headline is about."""
        from ..questions import routing_questions

        try:
            result = self.router.predict(text, routing_questions(symbols))
        except Exception as exc:  # pragma: no cover - network / engine
            return (), f"routing question failed: {exc}"
        answer = ((result or {}).get("answers") or {}).get("subject") or {}
        choice = answer.get("choice")
        probs = answer.get("probabilities") or {}
        if choice in symbols:
            return (str(choice),), f"engine routing: subject is {choice}"
        if choice == "market_wide":
            return tuple(symbols), "engine routing: the text is market-wide"
        picked = tuple(s for s in symbols if float(probs.get(s, 0.0)) >= 0.25)
        if picked:
            return picked, f"engine routing: spread over {len(picked)} symbols"
        return (), f"engine routing: unrelated ({choice})"

    def route_preview(self, texts: Iterable[str]) -> list[dict[str, str]]:
        """Which checkpoint would read each text, and why. No forward pass."""
        out = []
        for text in texts:
            try:
                d = self.router.route({"body": text})
                out.append({"model": d.model, "reason": getattr(d, "reason", "")})
            except Exception as exc:  # pragma: no cover
                out.append({"model": "error", "reason": str(exc)})
        return out
