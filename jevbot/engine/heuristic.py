"""A deterministic, dependency-free stand-in for the Laya engine.

It parses the *same rendered state text* the Router would be given — never the
raw numbers — and answers the *same* seven typed questions, emitting answers in
the *same* schema (see :mod:`jevbot.engine.answers`).

Why this exists:

1. **Runnability.** The full pipeline, backtester, dashboard and test suite work
   on a machine with no network and no checkpoint download.
2. **A baseline with teeth.** "Laya adds value" is only a claim if something
   weaker was measured against it over identical inputs and identical scoring.
   ``scripts/eval_engine.py --compare`` does exactly that.
3. **A canary.** When the two engines agree, the signal path is probably fine.
   When they disagree wildly on plain text, that is worth looking at before it
   reaches a broker.

It is not a good model. It is a *stable* one, and it is honest about being a
bag of lexicons with a parser attached.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from ..questions import (
    CONVICTION_SCALE,
    DIRECTION_CRITERIA,
    HORIZONS,
    PRICED_IN_SCALE,
)
from ..types import Decision
from .answers import choice_answer, noul_answer, score_answer, softmax
from .base import DecisionEngine

# ── lexicons ────────────────────────────────────────────────────────────────
# Weights are "how much this phrase means the price goes up", roughly calibrated
# so that a single strong word moves the signal about a third of the way.
BULLISH: dict[str, float] = {
    "surge": 0.9, "surges": 0.9, "soar": 0.9, "soars": 0.9, "rally": 0.75, "rallies": 0.75,
    "jumps": 0.7, "climbs": 0.6, "rises": 0.55, "gains": 0.55, "rebounds": 0.6,
    "record high": 1.0, "all-time high": 1.0, "all time high": 1.0, "new high": 0.8,
    "beats": 0.7, "beat expectations": 0.85, "tops estimates": 0.85, "outperform": 0.7,
    "upgrade": 0.75, "upgrades": 0.75, "raises guidance": 1.0, "raises outlook": 0.9,
    "buyback": 0.7, "share repurchase": 0.7, "dividend increase": 0.6,
    "approval": 0.85, "approves": 0.85, "approved": 0.85, "green light": 0.8, "cleared": 0.6,
    "adoption": 0.7, "partnership": 0.55, "integration": 0.45, "launches": 0.45,
    "inflows": 0.75, "inflow": 0.75, "net inflows": 0.8, "accumulation": 0.6,
    "bullish": 0.85, "optimism": 0.5, "confidence": 0.35, "demand": 0.45, "expands": 0.45,
    "wins": 0.6, "win": 0.5, "settlement": 0.45, "halving": 0.6, "etf": 0.4,
    "institutional": 0.45, "treasury": 0.4, "buyback program": 0.7, "guidance raised": 1.0,
}

BEARISH: dict[str, float] = {
    "plunge": -1.0, "plunges": -1.0, "crash": -1.0, "crashes": -1.0, "tumbles": -0.9,
    "slumps": -0.85, "sell-off": -0.85, "selloff": -0.85, "dumps": -0.75, "falls": -0.5,
    "drops": -0.5, "declines": -0.5, "slides": -0.55, "retreats": -0.5,
    "record low": -0.95, "all-time low": -0.95, "new low": -0.8, "bear market": -1.0,
    "misses": -0.7, "missed estimates": -0.8, "disappoints": -0.75, "shortfall": -0.6,
    "cuts guidance": -1.0, "lowers outlook": -0.9, "warns": -0.7, "warning": -0.6,
    "downgrade": -0.75, "downgrades": -0.75, "underperform": -0.7,
    "lawsuit": -0.65, "sues": -0.65, "sued": -0.65, "probe": -0.55, "investigation": -0.55,
    "fine": -0.5, "penalty": -0.5, "settles allegations": -0.6, "class action": -0.7,
    "hack": -1.0, "hacked": -1.0, "exploit": -1.0, "exploited": -1.0, "stolen": -0.9,
    "breach": -0.8, "fraud": -1.0, "bankruptcy": -1.0, "insolvency": -1.0, "insolvent": -1.0,
    "default": -0.9, "defaults": -0.9, "liquidation": -0.7, "delisting": -1.0,
    "delist": -1.0, "halted": -0.85, "suspends": -0.75, "outflows": -0.75, "outflow": -0.75,
    "bearish": -0.85, "ban": -0.8, "banned": -0.8, "crackdown": -0.75, "tariff": -0.55,
    "layoffs": -0.55, "restructuring": -0.4, "recall": -0.7, "resigns": -0.5,
    "steps down": -0.55, "exits": -0.45, "recession": -0.85, "contagion": -0.85,
    "depeg": -1.0, "rug": -1.0, "stagnant": -0.4, "pressure": -0.35,
}

UNCERTAINTY = (
    "may ", "might ", "could ", "reportedly", "rumour", "rumor", "speculation",
    "considering", "weighs", "exploring", "said to", "sources say", "unconfirmed",
    "possibly", "reportedly considering", "is said to be", "mulls",
)

STALENESS = ("yesterday", "last week", "last month", "earlier this year", "recap")

RISK_TERMS = (
    "hack", "exploit", "breach", "fraud", "bankruptcy", "insolvency", "default",
    "delisting", "delist", "halted", "sanctions", "war", "exchange failure",
    "depeg", "contagion", "circuit breaker", "resigns", "steps down", "emergency",
)

HORIZON_KEYWORDS: dict[str, tuple[str, ...]] = {
    "intraday": (
        "earnings", "quarterly", "cpi", "inflation print", "fomc", "rate decision",
        "jobless", "payrolls", "guidance", "today", "this morning", "upgrade",
        "downgrade", "beats", "misses", "price target", "halted", "flash",
    ),
    "positional": (
        "regulation", "regulatory", "law", "legislation", "rule", "framework",
        "structural", "supply", "emission", "tokenomics", "halving", "tariff",
        "treaty", "ban", "long-term", "multi-year", "reserve", "adoption curve",
    ),
}

PRICED_IN_STRONG = ("already priced in", "priced in", "fully reflected", "already rallied on",
                    "already surged on", "already jumped on", "was already known")
PRICED_IN_MILD = ("as expected", "widely expected", "widely anticipated", "telegraphed",
                  "in line with expectations", "long anticipated", "expected to")
SURPRISE = ("unexpected", "unexpectedly", "surprise", "surprises", "shocked", "shock",
            "unprecedented", "first time", "stuns", "unforeseen", "blindsided")

# ── parsing ─────────────────────────────────────────────────────────────────

_NUM = r"[-+]?\d+(?:\.\d+)?"
_RE_PERF = re.compile(rf"({_NUM})% over (?:the last hour|24 hours|7 days)")
_RE_NEWS = re.compile(r"^\s*\d+\.\s*\[(.*?)\]\s*(.+)$")
_SECTION_FOCUS = "HEADLINE UNDER CONSIDERATION"
_SECTION_CONTEXT = "EARLIER HEADLINES"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_state(state: str) -> dict[str, Any]:
    """Pull the few numbers the heuristic uses back out of the rendered text.

    Deliberately regex-over-the-rendered-string rather than a side channel: the
    point of the exercise is to answer from the text that a language model also
    receives, so reading the text is not a handicap, it is the rules of the game.
    """
    out: dict[str, Any] = {
        "rets": [], "trend": "unknown", "news": [], "focus": None, "context": [],
        "vol": None, "volume_z": None, "range_pos": None, "funding": None,
    }
    news: list[tuple[float, str, str]] = []
    focus: list[tuple[float, str, str]] = []
    context: list[tuple[float, str, str]] = []
    section = "context"
    for raw in state.splitlines():
        line = raw.strip()
        if _SECTION_FOCUS in line:
            section = "focus"
            continue
        if _SECTION_CONTEXT in line:
            section = "context"
            continue
        if line.startswith("PERFORMANCE:"):
            out["rets"] = [float(m) / 100.0 for m in _RE_PERF.findall(line)]
        elif line.startswith("TREND:"):
            if "above the slower" in line:
                out["trend"] = "up"
            elif "below the slower" in line:
                out["trend"] = "down"
            elif "level with" in line:
                out["trend"] = "sideways"
        elif line.startswith("VOLATILITY:"):
            m = re.search(rf"realised ({_NUM})% of price", line)
            if m:
                out["vol"] = abs(float(m.group(1))) / 100.0
        elif line.startswith("VOLUME:"):
            if "far above" in line:
                out["volume_z"] = 2.5
            elif "above" in line:
                out["volume_z"] = 1.2
            elif "far below" in line:
                out["volume_z"] = -2.5
            elif "below" in line:
                out["volume_z"] = -1.2
            else:
                out["volume_z"] = 0.0
        elif line.startswith("RANGE POSITION:"):
            out["range_pos"] = (
                0.95 if "at the top" in line
                else 0.75 if "upper half" in line
                else 0.5 if "near the middle" in line
                else 0.25 if "lower half" in line
                else 0.05 if "at the bottom" in line
                else 0.5
            )
        elif line.startswith("FUNDING:"):
            m = re.search(rf"funding is ({_NUM})% per 8 hours", line)
            if m:
                out["funding"] = float(m.group(1)) / 100.0
        else:
            m = _RE_NEWS.match(line)
            if m:
                stamp, text = m.group(1), m.group(2)
                hours = 0.0
                if "minutes ago" in stamp:
                    nums = re.findall(rf"({_NUM})\s*(seconds|minutes|hours|days) ago", stamp)
                    if nums:
                        value, unit = nums[0]
                        hours = float(value) * {
                            "seconds": 1 / 3600, "minutes": 1 / 60, "hours": 1.0, "days": 24.0,
                        }[unit]
                row = (hours, stamp.split(",")[-1].strip() if "," in stamp else "", text)
                news.append(row)
                (focus if section == "focus" else context).append(row)
    out["news"] = news
    out["focus"] = focus
    out["context"] = context
    out["primary"] = focus or news       # what the answer is *about*
    return out


# ── scoring helpers ─────────────────────────────────────────────────────────


def _negated(text: str, pos: int, span: int = 46) -> bool:
    window = text[max(0, pos - span):pos]
    return any(cue in window for cue in ("not ", "no ", "never ", "unlikely", "denies",
                                         "denied", "without ", "rules out", "fails to",
                                         "unable to", "delays ", "halted "))


def lexicon_score(text: str) -> tuple[float, list[str]]:
    """Signed headline sentiment in roughly -1..1 plus the matched terms."""
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    raw = 0.0
    hits: list[str] = []
    for table in (BULLISH, BEARISH):
        for term, weight in table.items():
            start = 0
            while True:
                pos = low.find(term, start)
                if pos < 0:
                    break
                w = weight
                if _negated(low, pos):
                    w = -w * 0.55
                raw += w
                hits.append(term)
                start = pos + len(term)
    # magnitude and hedging modifiers
    if any(m in low for m in ("record ", "massive", "huge", "unprecedented", "billion")):
        raw *= 1.35
    if any(m in low for m in ("slight", "modest", "minor", "marginally", "tiny", "small")):
        raw *= 0.6
    if any(u in low for u in UNCERTAINTY):
        raw *= 0.55
    if any(s in low for s in STALENESS):
        raw *= 0.7
    # squashing keeps a burst of synonyms from saturating the scale
    return float(_clip(raw / 2.4, -1.0, 1.0)), hits


def _specificity(text: str) -> float:
    """0..1: how much verifiable detail the sentence carries."""
    digits = len(re.findall(r"\d", text))
    numbers = len(re.findall(rf"{_NUM}%|\$\s?{_NUM}|\b{_NUM}\b", text))
    names = len(re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text))
    score = 0.35 * min(1.0, digits / 6.0) + 0.4 * min(1.0, numbers / 3.0) + 0.25 * min(1.0, names / 4.0)
    if any(u in text.lower() for u in UNCERTAINTY):
        score *= 0.6
    return float(_clip(score, 0.0, 1.0))


def _recency(hours: float) -> float:
    return float(0.5 ** (max(0.0, hours) / 12.0))


class HeuristicEngine(DecisionEngine):
    """Lexicon + context engine, deterministic and offline."""

    name = "heuristic"

    def __init__(self, min_confidence: float = 0.0, half_life_hours: float = 12.0) -> None:
        self.min_confidence = float(min_confidence or 0.0)
        self.half_life = half_life_hours

    # ── public API ─────────────────────────────────────────────────────────

    def decide(self, state: str, symbol: str, news_id: str, ts: float) -> Decision:
        parsed = parse_state(state)
        # The headline under consideration carries the decision; the earlier
        # items only shade it. Averaging the two equally would answer "what do
        # these seven headlines mean", which is not the question asked.
        focus_text = [t for _, _, t in parsed["focus"]]
        context_text = [t for _, _, t in parsed["context"]]
        sentiment_focus, hits_focus, age_focus = self._sentiment(parsed["focus"] or parsed["news"])
        sentiment_context, hits_context, age_context = self._sentiment(parsed["context"])
        if parsed["focus"]:
            sentiment = 0.72 * sentiment_focus + 0.28 * sentiment_context if parsed["context"] else sentiment_focus
            hits = hits_focus + hits_context
            best_age = age_focus
        else:
            sentiment, hits, best_age = sentiment_context, hits_context, age_context
        specificity = _specificity(" ".join(focus_text) or " ".join(context_text))
        ret_24h = parsed["rets"][1] if len(parsed["rets"]) > 1 else 0.0
        vol = parsed["vol"] or 0.02
        trend = parsed["trend"]

        answers = self.answer_set(sentiment, specificity, parsed, ret_24h, vol, best_age)

        d = answers["direction"]
        direction = str(d["choice"])
        conv = float(answers["conviction"]["score"])
        priced = float(answers["priced_in"]["score"])
        answer_conf = float(d["answer_confidence"])
        return Decision(
            news_id=news_id,
            symbol=symbol,
            ts=ts,
            engine=self.name,
            model="heuristic",
            routing_reason="offline lexicon engine (no checkpoint loaded)",
            direction=direction,
            direction_p=float(d["probabilities"].get(direction, 0.0)),
            direction_probs={k: float(v) for k, v in d["probabilities"].items()},
            materiality=float(answers["materiality"]["noul"]),
            conviction=conv,
            conviction_norm=_clip(conv / 4.0, 0.0, 1.0),
            priced_in=priced,
            context_aligned=float(answers["context_aligned"]["noul"]),
            horizon=str(answers["horizon"]["choice"]),
            risk_event=float(answers["risk_event"]["noul"]),
            abstained=bool(self.min_confidence and answer_conf < self.min_confidence),
            latency_ms=0.0,
            raw={"answers": answers, "hits": hits, "sentiment": round(sentiment, 4)},
        )

    def answer_set(
        self,
        sentiment: float,
        specificity: float,
        parsed: dict[str, Any],
        ret_24h: float,
        vol: float,
        age_hours: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        """The seven typed answers, in Laya's schema. Kept pure for testing."""
        labels = list(DIRECTION_CRITERIA.keys())          # long, flat, short
        s = sentiment
        strength = abs(s)

        # direction: a three-way softmax with a flat option that wins when the
        # text says nothing. Symmetric in s by construction. The flat bias grows
        # as the sentiment weakens, so silence stays silent and a clear story
        # gets a clear call.
        logits = [3.0 * s, 0.15 + 0.55 * (1.0 - strength), -3.0 * s]
        direction = choice_answer(labels, softmax(logits), act_probability=0.5 + 0.5 * strength)

        # materiality: is there anything to trade?
        materiality = _logistic(2.3 * strength + 1.5 * specificity - 1.5)
        materiality *= 0.55 + 0.45 * _recency(age_hours)

        # conviction: how strong, on the 0-4 scale
        conv = _clip(3.0 * strength + 1.1 * specificity - 0.15, 0.0, 4.0)
        conv_probs = _spread(conv, len(CONVICTION_SCALE))

        # priced in: 0 = fully reflected, 3 = not yet reflected
        priced = self._priced_in(parsed, s, ret_24h, vol)
        priced_probs = _spread(priced, len(PRICED_IN_SCALE))

        # context alignment
        trend = parsed["trend"]
        trend_sign = {"up": 1.0, "down": -1.0}.get(trend, 0.0)
        if trend_sign == 0.0:
            aligned = 0.5 + 0.15 * strength
        else:
            agree = trend_sign * s
            aligned = _logistic(2.2 * agree * max(strength, 0.15) + 0.35)
            if abs(ret_24h) > 1.5 * vol:
                aligned = _clip(aligned + 0.08 * (1 if ret_24h > 0 else -1) * trend_sign, 0.02, 0.98)

        # horizon from news type
        joined = " ".join(t for _, _, t in parsed["primary"]).lower()
        horizon = "swing"
        if any(k in joined for k in HORIZON_KEYWORDS["positional"]):
            horizon = "positional"
        elif any(k in joined for k in HORIZON_KEYWORDS["intraday"]):
            horizon = "intraday"
        horizon_answer = choice_answer(
            list(HORIZONS.keys()),
            [0.2 if h == horizon else 0.8 / (len(HORIZONS) - 1) for h in HORIZONS],
        )

        # tail risk
        risk_hits = [t for t in RISK_TERMS if t in joined]
        risk = _clip(0.15 + 0.28 * len(risk_hits) + 0.25 * max(strength if s < 0 else 0.0, 0.0), 0.02, 0.98)

        return {
            "direction": direction,
            "materiality": noul_answer(materiality, act_probability=materiality),
            "conviction": score_answer(CONVICTION_SCALE, conv_probs, act_probability=0.5 + 0.5 * strength),
            "priced_in": score_answer(PRICED_IN_SCALE, priced_probs, act_probability=0.6),
            "context_aligned": noul_answer(aligned, act_probability=aligned),
            "horizon": horizon_answer,
            "risk_event": noul_answer(risk, act_probability=risk),
        }

    def _sentiment(self, news: list[tuple[float, str, str]]) -> tuple[float, list[str], float]:
        """Recency-weighted sentiment across the headline list."""
        if not news:
            return 0.0, [], 0.0
        total = 0.0
        weight = 0.0
        hits: list[str] = []
        best_age = min(h for h, _, _ in news)
        for hours, _src, text in news:
            sc, terms = lexicon_score(text)
            w = 0.5 ** (max(0.0, hours) / self.half_life)
            total += sc * w
            weight += w
            hits.extend(terms)
        return (total / weight if weight else 0.0), hits, best_age

    def _priced_in(self, parsed: dict[str, Any], sentiment: float, ret_24h: float,
                   vol: float) -> float:
        joined = " ".join(t for _, _, t in parsed["primary"]).lower()
        value = 1.9                                       # baseline "partly reflected"
        if any(p in joined for p in PRICED_IN_STRONG):
            value = 0.25
        elif any(p in joined for p in PRICED_IN_MILD):
            value = 1.0
        if any(p in joined for p in SURPRISE):
            value = max(value, 2.8)
        # the tape's own reaction: a move already in the news' direction is
        # evidence the market got there first
        if vol > 0 and sentiment != 0:
            move_in_vols = ret_24h / vol
            if move_in_vols * sentiment > 0:
                value -= _clip(abs(move_in_vols) * 0.45, 0.0, 1.5)
            else:
                value += 0.35
        return _clip(value, 0.0, 3.0)

    # ── status ─────────────────────────────────────────────────────────────

    def info(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "available": True,
            "model": "heuristic",
            "note": (
                "deterministic lexicon engine, no checkpoint required; reads the same "
                "rendered state as Laya and answers the same typed questions"
            ),
            "min_confidence_gate": self.min_confidence or None,
            "lexicon_terms": len(BULLISH) + len(BEARISH),
        }


def _logistic(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _spread(value: float, levels: int) -> list[float]:
    """A unimodal distribution over levels whose mean is ``value``."""
    weights = []
    for i in range(levels):
        d = abs(i - value)
        weights.append(max(1e-6, 1.0 - d / max(1.0, levels - 1)) ** 2)
    total = sum(weights)
    return [w / total for w in weights]


def _iter_terms() -> Iterable[str]:  # pragma: no cover - convenience for tooling
    yield from BULLISH
    yield from BEARISH
