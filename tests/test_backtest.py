"""The backtester, and the honesty of the demo market.

Two things have to be true for any number this prints to mean anything: the run
has to actually trade (a warmup longer than the history prints a full report of
zeros), and the same seed has to produce the same run. Both have gone wrong here
before.
"""

from __future__ import annotations

import json

import pytest

from jevbot.backtest import build_demo_market, compare_markdown, run_backtest, write_result


@pytest.fixture
def run(cfg):
    return run_backtest(cfg, engine_name="heuristic", days=4, seed=11,
                        step_seconds=1800, warmup_days=1.0, horizon_seconds=4 * 3600)


def test_a_short_run_still_trades(cfg):
    """A run that measures nothing must not print a report that looks like one."""
    result = run_backtest(cfg, engine_name="heuristic", days=4, seed=11,
                          step_seconds=1800, warmup_days=1.0, horizon_seconds=4 * 3600)
    assert result.cycles > 0
    assert result.decisions > 0, "the engine was never asked anything"
    assert len(result.equity_curve) == result.cycles
    assert result.perf.starting_equity == pytest.approx(100_000.0)


def test_the_same_seed_replays_to_the_same_result(cfg):
    a = run_backtest(cfg, engine_name="heuristic", days=4, seed=5,
                     step_seconds=1800, warmup_days=1.0, horizon_seconds=4 * 3600)
    b = run_backtest(cfg, engine_name="heuristic", days=4, seed=5,
                     step_seconds=1800, warmup_days=1.0, horizon_seconds=4 * 3600)
    assert a.perf.total_return == pytest.approx(b.perf.total_return)
    assert a.decisions == b.decisions
    assert [p[1] for p in a.equity_curve] == [p[1] for p in b.equity_curve]


def test_the_benchmark_uses_the_same_window_and_universe(run, cfg):
    assert run.benchmark.total_return != 0.0
    assert run.benchmark.periods > 1
    assert run.perf.exposure >= 0.0


def test_the_shuffled_control_shares_the_tape_but_not_the_timing(cfg):
    real = run_backtest(cfg, engine_name="heuristic", days=4, seed=7,
                        step_seconds=1800, warmup_days=1.0, horizon_seconds=4 * 3600)
    control = run_backtest(cfg, engine_name="heuristic", days=4, seed=7,
                           step_seconds=1800, warmup_days=1.0, horizon_seconds=4 * 3600,
                           shuffle=True)
    assert control.shuffled
    assert [p[1] for p in control.equity_curve] != [p[1] for p in real.equity_curve]
    # the tape is generated from the seed, so it is identical in both
    market = build_demo_market(cfg, days=4, seed=7)
    assert market.bars("BTC/USDT") == build_demo_market(cfg, days=4, seed=7).bars("BTC/USDT")


def test_summary_is_printable_and_says_it_is_synthetic(run):
    text = run.summary()
    assert "equity" in text.lower()
    assert "synthetic" in text.lower()
    assert "not" in text.lower()


def test_results_are_written_reopenably(run, tmp_path):
    path = write_result(run, tmp_path)
    payload = json.loads(path.with_suffix(".json").read_text())
    assert payload["engine"] == "heuristic"
    assert "performance" in payload and "engine_score" in payload
    csv_text = path.with_suffix(".csv").read_text().strip().splitlines()
    assert csv_text[0].startswith("ts,equity")
    assert len(csv_text) == len(run.equity_curve) + 1


def test_compare_markdown_names_both_engines(run):
    md = compare_markdown({"first": run, "second": run})
    assert "first" in md and "second" in md
    assert "|" in md
