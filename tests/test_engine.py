"""The decision engine interface, and the offline engine that stands in for Laya.

The heuristic engine exists so the pipeline can be tested, replayed and shipped
without a checkpoint. It is not a substitute for Laya and these tests only pin
the things that must hold for *any* engine: bounded answers, a decision that is
about the headline it was asked about, and abstention behaving as abstention.
"""

from __future__ import annotations

import pytest

from jevbot.engine import HeuristicEngine, build_engine
from jevbot.engine.answers import answer_confidence_of, choice_answer, noul_answer, score_answer
from jevbot.engine.base import DecisionCache, EngineUnavailable
from jevbot.engine.resilient import ResilientEngine
from jevbot.types import Instrument, NewsItem

TS = 1_700_000_000.0


def state_for(text: str, price: float = 100.0) -> str:
    from jevbot.text import render_state
    from tests.conftest import snapshot

    item = NewsItem(text=text, ts=TS, source="test")
    inst = Instrument("BTC/USDT", "crypto", "Bitcoin")
    return render_state(snapshot(price), [item], inst, focus=item)


def test_answers_are_probability_distributions():
    a = choice_answer(["long", "flat", "short"], [0.2, 0.3, 0.5])
    assert sum(a["probabilities"].values()) == pytest.approx(1.0)
    assert a["choice"] == "short"
    assert 0.0 <= a["answer_confidence"] <= 1.0
    assert a["type"] == "choice"


def test_noul_and_score_shapes_match_layas_schema():
    n = noul_answer(0.8)
    assert n["type"] == "noul" and n["noul"] == pytest.approx(0.8)
    s = score_answer(["a", "b", "c"], [0.1, 0.6, 0.3])
    assert s["type"] == "score" and 0.0 <= s["score"] <= 2.0
    assert set(s["probabilities"]) == {"0", "1", "2"}


def test_answer_confidence_of_handles_every_type():
    assert answer_confidence_of(choice_answer(["a", "b"], [0.9, 0.1])) >= 0.0
    assert answer_confidence_of(noul_answer(0.9)) >= 0.0
    assert answer_confidence_of(score_answer(["a", "b"], [0.5, 0.5])) >= 0.0


def test_heuristic_decision_is_bounded_and_typed():
    e = HeuristicEngine()
    d = e.decide(state_for("Bitcoin ETF inflows hit a record as BTC rallies"), "BTC/USDT", "n1", TS)
    assert d.engine == "heuristic"
    assert d.direction in {"long", "flat", "short"}
    for value in (d.direction_p, d.materiality, d.conviction_norm, d.context_aligned, d.risk_event):
        assert 0.0 <= value <= 1.0, f"{value} out of range"
    assert 0.0 <= d.priced_in <= 3.0
    assert d.horizon in {"intraday", "swing", "positional"}
    assert d.news_id == "n1" and d.symbol == "BTC/USDT"
    assert d.raw["answers"], "the raw answers have to survive for the dashboard"


def test_a_decision_is_about_the_headline_it_was_asked_about():
    """Two headlines in the same state must not produce the same answer.

    The state names the subject (``focus``) precisely so this holds: if an
    engine averages the whole list, every headline in a cycle gets one shared
    answer and the accuracy metric measures nothing.
    """
    from jevbot.text import render_state
    from tests.conftest import snapshot

    good = NewsItem(text="Bitcoin ETF inflows hit a record as BTC rallies", ts=TS, source="t")
    bad = NewsItem(text="Exchange halts withdrawals after a security breach", ts=TS, source="t")
    inst = Instrument("BTC/USDT", "crypto", "Bitcoin")
    snap = snapshot(100.0)
    e = HeuristicEngine()
    d_good = e.decide(render_state(snap, [good, bad], inst, focus=good), "BTC/USDT", good.news_id, TS)
    d_bad = e.decide(render_state(snap, [good, bad], inst, focus=bad), "BTC/USDT", bad.news_id, TS)
    assert d_good.direction != d_bad.direction, (
        f"both headlines answered {d_good.direction}: the state does not name its subject"
    )


def test_a_confidence_gate_turns_answers_into_abstentions():
    e = HeuristicEngine(min_confidence=0.999)
    d = e.decide(state_for("Mildly interesting market chatter"), "BTC/USDT", "n", TS)
    assert d.abstained


def test_build_engine_refuses_laya_when_asked_for_it_explicitly():
    """Asking for Laya and silently getting a lexicon would be a lie."""
    try:
        engine = build_engine("laya", preload=False)
    except EngineUnavailable:
        return
    # a checkpoint is present in this environment: the name must reflect it
    assert engine.name == "laya"


def test_auto_falls_back_without_pretending_to_be_laya():
    engine = build_engine("auto", preload=False)
    assert engine.name in {"laya", "heuristic"}
    assert engine.info().get("engine") == engine.name or "note" in engine.info()


def test_resilient_engine_degrades_loudly_and_keeps_answering():
    class Broken:
        name = "broken"

        def decide(self, *a, **kw):
            raise RuntimeError("checkpoint evicted")

        def decide_batch(self, requests):
            raise RuntimeError("checkpoint evicted")

        def info(self):
            return {"engine": "broken"}

        def close(self):
            pass

    e = ResilientEngine(Broken(), HeuristicEngine(), max_failures=1)
    d1 = e.decide(state_for("Bitcoin ETF inflows hit a record"), "BTC/USDT", "n", TS)
    assert e.degraded
    assert "checkpoint evicted" in e.last_error
    assert d1.engine == "heuristic"
    assert e.info()["active"] == "heuristic"
    assert "offline" in e.info()["note"]
    # and from here on it does not even try the broken engine
    d2 = e.decide(state_for("Bitcoin ETF inflows hit a record"), "BTC/USDT", "n", TS)
    assert d2.engine == "heuristic"


def test_decision_cache_is_keyed_and_bounded():
    c = DecisionCache(capacity=2)
    assert c.get("a") is None
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert c.get("a") is None, "the oldest entry should have been evicted"
    assert c.get("c") == 3
    assert 0.0 <= c.hit_rate <= 1.0
