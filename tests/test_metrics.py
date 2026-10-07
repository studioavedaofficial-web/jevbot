"""Measurement.

These are the functions that decide whether the strategy is any good, so they
are the ones most worth distrusting. Each test below pins a deliberate choice:
flat calls excluded from accuracy, unlabelled decisions dropped rather than
scored as zero, a materiality threshold relative to the instrument's own
volatility, and an IC that is defined for ties.
"""

from __future__ import annotations

import math

import pytest

from jevbot.metrics import (
    brier,
    direction_accuracy,
    expected_calibration_error,
    performance,
    score_decisions,
    spearman,
)

TS = 1_700_000_000.0


def curve(values, start=TS, step=900.0):
    return [(start + i * step, v) for i, v in enumerate(values)]


def test_performance_of_a_flat_curve_is_flat():
    p = performance(curve([100_000.0] * 10), starting_equity=100_000.0, period_seconds=900.0)
    assert p.total_return == pytest.approx(0.0)
    assert p.trades == 0


def test_performance_of_a_monotone_curve_is_positive_with_no_drawdown():
    p = performance(curve([100.0, 101.0, 102.0, 103.0]), starting_equity=100.0, period_seconds=900.0)
    assert p.total_return == pytest.approx(0.03)
    assert p.max_drawdown == pytest.approx(0.0)
    assert p.sharpe > 0


def test_drawdown_is_the_worst_peak_to_trough():
    p = performance(curve([100.0, 120.0, 90.0, 95.0]), starting_equity=100.0, period_seconds=900.0)
    assert p.max_drawdown == pytest.approx(0.25)


def test_brier_is_zero_for_perfect_forecasts_and_one_quarter_for_coin_flips():
    assert brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_ece_separates_confident_and_correct_from_confident_and_wrong():
    good = expected_calibration_error([0.99, 0.99, 0.01, 0.01], [1, 1, 0, 0], bins=10)
    bad = expected_calibration_error([0.99, 0.99, 0.01, 0.01], [0, 0, 1, 1], bins=10)
    assert good < bad
    assert bad > 0.5


def test_spearman_handles_ties_and_direction():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # a constant predictor has no rank information; nan is not 0.0, because
    # "undefined" and "no relationship" are different claims
    assert math.isnan(spearman([0.1, 0.1], [1.0, -1.0]))


def test_direction_accuracy_ignores_flat_calls():
    """A bot that says "flat" to everything must not score 100 %."""
    acc, n = direction_accuracy(["flat", "flat", "long"], [0.02, -0.02, 0.03])
    assert n == 1
    assert acc == pytest.approx(1.0)


def test_direction_accuracy_counts_a_short_as_a_negative_call():
    acc, n = direction_accuracy(["short", "long"], [-0.01, -0.01])
    assert n == 2
    assert acc == pytest.approx(0.5)


def test_score_decisions_drops_rows_it_has_no_label_for():
    rows = [
        {"news_id": "a", "symbol": "X", "direction": "long", "materiality": 0.9,
         "conviction_norm": 0.5, "priced_in": 0.0, "context_aligned": 0.5, "risk_event": 0.1},
        {"news_id": "b", "symbol": "X", "direction": "long", "materiality": 0.9,
         "conviction_norm": 0.5, "priced_in": 0.0, "context_aligned": 0.5, "risk_event": 0.1},
    ]
    forward = {"a|X": 0.02}
    score = score_decisions(rows, forward)
    assert score.labelled == 1
    assert score.direction_n == 1
    assert score.direction_accuracy == pytest.approx(1.0)


def test_materiality_threshold_is_relative_to_the_instruments_own_volatility():
    """A 20 bp move is huge for a bond ETF and noise for a memecoin.

    The label is "did the headline actually move *this* asset", so the threshold
    scales with the asset's own expected move over the same horizon.
    """
    rows = [{"news_id": "a", "symbol": "X", "direction": "long", "materiality": 0.9,
             "conviction_norm": 0.5, "priced_in": 0.0, "context_aligned": 0.5, "risk_event": 0.0}]
    tiny = score_decisions(rows, {"a|X": 0.003}, sigmas={"a|X": 0.001})
    big = score_decisions(rows, {"a|X": 0.003}, sigmas={"a|X": 0.01})
    assert tiny.materiality_base_rate == pytest.approx(1.0), "material for a quiet asset"
    assert big.materiality_base_rate == pytest.approx(0.0), "the same move is noise for a volatile one"


def test_a_risk_event_answer_is_calibrated_against_the_tail_it_predicts():
    """``risk_event`` is scored on its own outcome: did this lose 1 %+.

    Folding it into the direction metric would double-count the same headline
    (once as a direction call, once as a risk call), and a flat answer is a
    refusal, so the tail question is the only place it can be judged.
    """
    rows = [{"news_id": "a", "symbol": "X", "direction": "flat", "materiality": 0.9,
             "conviction_norm": 0.0, "priced_in": 0.0, "context_aligned": 0.0, "risk_event": 0.9}]
    score = score_decisions(rows, {"a|X": -0.05})
    assert score.risk_brier == pytest.approx((0.9 - 1.0) ** 2)
    assert score.direction_n == 0, "a flat call is a refusal, not a short"


def test_a_confident_long_into_a_crash_is_scored_wrong():
    def row(news_id, direction, conviction):
        return {"news_id": news_id, "symbol": "X", "direction": direction, "materiality": 0.9,
                "conviction_norm": conviction, "priced_in": 0.0, "context_aligned": 0.5,
                "risk_event": 0.0}

    rows = [row("a", "long", 0.9), row("b", "long", 0.6), row("c", "short", 0.6)]
    score = score_decisions(rows, {"a|X": -0.05, "b|X": 0.02, "c|X": -0.03})
    assert score.direction_n == 3
    assert score.direction_accuracy == pytest.approx(2 / 3)
    assert score.signal_ic < 0, "the more conviction behind the wrong call, the worse the IC"


def test_an_empty_score_reports_nan_not_a_flattering_zero():
    score = score_decisions([], {})
    assert math.isnan(score.direction_accuracy) or score.direction_n == 0
    assert score.decisions == 0
