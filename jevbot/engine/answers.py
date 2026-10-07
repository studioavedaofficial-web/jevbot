"""Builders for Laya-shaped answer dicts.

The offline engine could have returned whatever shape was convenient — but then
the two engines would be judged by two different scorers, and every comparison
between them would be an artifact of the comparison code. So the heuristic
engine emits answers in *exactly* Laya's schema: same keys, same rounding,
``confidence`` as normalised entropy for ``choice``/``score`` and ``max(p)`` for
``noul``, ``answer_confidence`` as ``max(p)`` throughout, ``probabilities``
keyed by the option labels, and ``legend`` on every score.

That is the whole reason a calibration number computed by
:mod:`jevbot.metrics` means the same thing for both engines.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


def _norm_entropy_confidence(probs: Sequence[float]) -> float:
    """1 - H(p)/log(k): Laya's `confidence` for choice and score questions."""
    k = len(probs)
    if k <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return round(max(0.0, 1.0 - entropy / math.log(k)), 4)


def _empty(n: int) -> list[float]:
    return [1.0 / n] * n


def softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def choice_answer(labels: Sequence[str], probabilities: Sequence[float],
                  act_probability: float = 1.0) -> dict[str, Any]:
    probs = list(probabilities) or _empty(len(labels))
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "type": "choice",
        "choice": labels[best],
        "probabilities": {label: round(p, 4) for label, p in zip(labels, probs)},
        "confidence": _norm_entropy_confidence(probs),
        "answer_confidence": round(max(probs), 4),
        "action": {"act_probability": round(float(act_probability), 4)},
    }


def noul_answer(p_true: float, act_probability: float = 1.0) -> dict[str, Any]:
    p = max(0.0, min(1.0, float(p_true)))
    return {
        "type": "noul",
        "noul": round(p, 4),
        "confidence": round(max(p, 1.0 - p), 4),
        "answer_confidence": round(max(p, 1.0 - p), 4),
        "action": {"act_probability": round(float(act_probability), 4)},
    }


def score_answer(levels: Sequence[str], probabilities: Sequence[float],
                 act_probability: float = 1.0) -> dict[str, Any]:
    probs = list(probabilities) or _empty(len(levels))
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    expected = sum(i * p for i, p in enumerate(probs))
    return {
        "type": "score",
        "score": round(float(expected), 4),
        "legend": {str(i): str(level) for i, level in enumerate(levels)},
        "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
        "confidence": _norm_entropy_confidence(probs),
        "answer_confidence": round(max(probs), 4),
        "action": {"act_probability": round(float(act_probability), 4)},
    }


def answer_confidence_of(answer: dict[str, Any]) -> float:
    """``answer_confidence``, falling back to ``confidence`` on older payloads."""
    return float(answer.get("answer_confidence", answer.get("confidence", 0.0)) or 0.0)
