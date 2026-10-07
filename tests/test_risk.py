"""The governor.

Every limit here exists because some version of this bot lost money without it,
and the tests are written as "the thing that would have happened, cannot".
"""

from __future__ import annotations

import pytest

from jevbot.risk import RiskGovernor, RiskLimits
from jevbot.types import MarketSnapshot, Signal

TS = 1_700_000_000.0


def limits(**kw) -> RiskLimits:
    base = dict(starting_equity=100_000.0, min_order_notional=0.0,
                max_weight_per_symbol=1.0, max_gross_weight=1.0, max_net_weight=1.0)
    base.update(kw)
    return RiskLimits(**base)


def snap(symbol: str = "X", ts: float = TS, price: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot(symbol=symbol, ts=ts, last=price, half_spread_bps=1.0)


def sig(symbol: str = "X", score: float = 0.5, weight: float = 0.5,
        ts: float = TS) -> Signal:
    return Signal(symbol=symbol, ts=ts, score=score, target_weight=weight)


def test_a_normal_signal_passes_through():
    g = RiskGovernor(limits())
    v = g.evaluate({"X": sig()}, {"X": snap()}, 100_000.0, {}, TS)
    assert v.approved == {"X": 0.5}
    assert not v.halted


def test_weak_signal_is_dropped():
    g = RiskGovernor(limits(min_signal=0.3))
    v = g.evaluate({"X": sig(score=0.1)}, {"X": snap()}, 100_000.0, {}, TS)
    assert v.approved == {}
    assert "X" in v.dropped


def test_hysteresis_lets_a_held_position_stay_on_a_weaker_signal():
    """One threshold oscillates around itself and pays the round trip to do it.

    The floor to open is higher than the floor to hold; a signal at 0.4 is not
    enough to open (floor 0.5) but is enough to keep a position already on.
    """
    g = RiskGovernor(limits(min_signal=0.5, hold_signal_ratio=0.5))
    opened = g.evaluate({"X": sig(score=0.4)}, {"X": snap()}, 100_000.0, {}, TS)
    assert opened.approved == {}
    kept = g.evaluate({"X": sig(score=0.4)}, {"X": snap()}, 100_000.0, {"X": 0.5}, TS)
    assert kept.approved == {"X": 0.5}


def test_stale_snapshot_is_not_traded():
    g = RiskGovernor(limits(stale_data_seconds=900))
    v = g.evaluate({"X": sig()}, {"X": snap(ts=TS - 3_600)}, 100_000.0, {}, TS)
    assert v.approved == {}
    assert "stale" in v.dropped["X"]


def test_per_symbol_cap_clips_instead_of_dropping():
    g = RiskGovernor(limits(max_weight_per_symbol=0.2))
    v = g.evaluate({"X": sig(weight=0.9)}, {"X": snap()}, 100_000.0, {}, TS)
    assert v.approved["X"] == pytest.approx(0.2)


def test_position_count_keeps_the_strongest_signals():
    g = RiskGovernor(limits(max_positions=1))
    v = g.evaluate(
        {"X": sig("X", score=0.9), "Y": sig("Y", score=0.4)},
        {"X": snap("X"), "Y": snap("Y")},
        100_000.0, {}, TS,
    )
    assert set(v.approved) == {"X"}
    assert v.dropped["Y"] == "position count limit"


def test_daily_loss_limit_halts_the_day():
    g = RiskGovernor(limits(daily_loss_limit_pct=0.03))
    g.roll_day(100_000.0, TS)
    v = g.evaluate({"X": sig()}, {"X": snap()}, 96_000.0, {}, TS)
    assert v.halted and v.flatten
    assert g.halted_today


def test_max_drawdown_kills_the_bot_and_stays_killed():
    g = RiskGovernor(limits(max_drawdown_pct=0.10))
    v = g.evaluate({"X": sig()}, {"X": snap()}, 80_000.0, {}, TS)
    assert g.killed and v.flatten
    assert "drawdown" in g.kill_reason
    # and it does not un-kill itself on the next healthy cycle
    v2 = g.evaluate({"X": sig()}, {"X": snap()}, 100_000.0, {}, TS + 60)
    assert v2.halted


def test_kill_switch_can_be_reset_deliberately():
    g = RiskGovernor(limits(max_drawdown_pct=0.10))
    g.evaluate({"X": sig()}, {"X": snap()}, 80_000.0, {}, TS)
    g.reset_kill()
    assert not g.killed


def test_cooldown_after_a_stop_blocks_re_entry_then_expires():
    g = RiskGovernor(limits(cooldown_seconds_after_stop=1800))
    g.note_stop("X", TS)
    blocked = g.evaluate({"X": sig()}, {"X": snap()}, 100_000.0, {}, TS + 60)
    assert blocked.approved == {}
    assert blocked.dropped["X"] == "cooldown after stop"
    later = g.evaluate({"X": sig()}, {"X": snap(ts=TS + 3600)}, 100_000.0, {}, TS + 3600)
    assert later.approved == {"X": 0.5}


def test_turnover_cap_scales_a_sudden_all_in():
    g = RiskGovernor(limits(max_turnover_per_cycle=0.25, max_weight_per_symbol=1.0))
    v = g.evaluate({"X": sig(weight=1.0)}, {"X": snap()}, 100_000.0, {}, TS)
    assert v.approved["X"] <= 0.25 + 1e-9
    assert any("turnover" in n for n in v.notes)


def test_cash_buffer_keeps_some_cash():
    g = RiskGovernor(limits(min_cash_buffer=0.10, max_gross_weight=1.0))
    v = g.evaluate({"X": sig(weight=1.0)}, {"X": snap()}, 100_000.0, {}, TS)
    assert v.approved["X"] <= 0.9 + 1e-9


def test_a_held_symbol_with_no_signal_is_an_exit():
    g = RiskGovernor(limits(min_signal=0.3))
    v = g.evaluate({}, {"X": snap()}, 100_000.0, {"X": 0.4}, TS)
    assert v.approved == {}
    assert v.dropped["X"] == "no signal, position to be closed"


def test_status_reports_the_limits_it_enforced():
    g = RiskGovernor(limits(max_positions=3))
    st = g.status(100_000.0, TS)
    assert st["limits"]["max_positions"] == 3
    assert "killed" in st and "daily_pnl_pct" in st
