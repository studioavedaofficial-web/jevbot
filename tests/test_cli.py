"""The command line: the surface a stranger meets first.

A CLI that silently ignores an override is worse than one that crashes: the run
finishes, prints plausible numbers, and answers a different question than the one
asked. ``--set`` therefore raises on anything it cannot parse.
"""

from __future__ import annotations

import pytest

from jevbot.cli import _coerce, build_parser, main
from jevbot.config import ConfigError, load_config


def test_every_documented_command_exists():
    parser = build_parser()
    for command in ("run", "backtest", "eval", "gen-data", "route", "doctor"):
        assert command in parser._subparsers._group_actions[0].choices  # noqa: SLF001


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_defaults_are_paper_and_served():
    args = build_parser().parse_args(["run"])
    assert args.no_serve is False
    assert args.mode is None  # config decides; paper is the default there
    assert args.func.__name__ == "cmd_run"


def test_backtest_defaults_are_the_honest_ones():
    args = build_parser().parse_args(["backtest"])
    assert args.shuffle is False
    assert args.step == 900.0
    assert args.days == 30.0


def test_coerce_reads_the_scalar_types_a_config_needs():
    assert _coerce("true") is True
    assert _coerce("false") is False
    assert _coerce("42") == 42
    assert _coerce("0.015") == pytest.approx(0.015)
    assert _coerce("BTC/USDT") == "BTC/USDT"


def test_set_overrides_reach_the_config_object(tmp_path):
    args = build_parser().parse_args(
        ["backtest", "--set", "risk.max_positions=3", "--set", "broker.fee_bps=12.5"]
    )
    cfg = load_config()
    from jevbot.cli import _load

    cfg, _notes = _load(args)
    assert cfg.get("risk", "max_positions") == 3
    assert cfg.get("broker", "fee_bps") == pytest.approx(12.5)


def test_a_malformed_set_is_refused():
    args = build_parser().parse_args(["backtest", "--set", "risk.max_positions"])
    from jevbot.cli import _load

    with pytest.raises(ConfigError):
        _load(args)


def test_engine_flag_reaches_the_config():
    args = build_parser().parse_args(["backtest", "--engine", "heuristic"])
    from jevbot.cli import _load

    cfg, _ = _load(args)
    assert cfg.engine_name == "heuristic"


def test_doctor_reports_a_working_environment_without_network(capsys):
    code = main(["doctor", "--engine", "heuristic"])
    out = capsys.readouterr().out
    assert code == 0
    assert "engine:" in out
    assert "heuristic" in out
    assert "route:" in out, "doctor should show that routing works"


def test_route_explains_one_headline_end_to_end(capsys):
    code = main(["route", "Apple beats earnings as iPhone sales jump", "--engine", "heuristic"])
    out = capsys.readouterr().out
    assert code == 0
    assert "routing:" in out
    assert "MARKET CONTEXT" in out
    assert "HEADLINE UNDER CONSIDERATION" in out
    assert '"direction"' in out


def test_an_empty_route_does_not_pretend_to_have_answered(capsys):
    code = main(["route", "Local bakery wins regional award", "--engine", "heuristic"])
    out = capsys.readouterr().out
    assert code == 0
    assert "(none)" in out
