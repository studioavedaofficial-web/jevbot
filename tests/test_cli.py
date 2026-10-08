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


# ── a broken feed must not take the dashboard with it ────────────────────────
# The live path can fail at startup for reasons that are not the operator's
# fault: a venue outage, a revoked key, a sandbox with no egress. The one
# process worth keeping alive is the one that says so.


def _feed_error():
    from jevbot.feeds.base import FeedError

    return FeedError(
        "could not fetch markets from binance testnet "
        "(https://testnet.binance.vision/api/v3): binance GET /exchangeInfo"
    )


def test_a_headless_run_fails_fast_when_the_venue_is_unreachable(monkeypatch, capsys):
    from jevbot import cli

    def boom(self):
        raise _feed_error()

    monkeypatch.setattr(cli.TradingBot, "warmup", boom)
    code = main(["run", "--no-serve", "--cycles", "1", "--set", "server.db=/tmp/jevbot-test.db"])
    out = capsys.readouterr().out
    assert code == 3, "a script that asked for a live feed must not exit 0 having read nothing"
    assert "testnet.binance.vision" in out, out


def test_a_served_run_keeps_the_dashboard_up_and_names_the_endpoint(monkeypatch, capsys):
    from jevbot import cli, server
    from jevbot.bot import CycleResult

    def boom(self):
        raise _feed_error()

    started = {}

    class FakeServer:
        def __init__(self, bot, store, host=None, port=None):
            started["host"], started["port"] = host, port
            self.host, self.port = host, 8000

        def start(self, background=False):
            started["started"] = True
            return "http://127.0.0.1:8000"

        def stop(self):
            started["stopped"] = True

    monkeypatch.setattr(cli.TradingBot, "warmup", boom)
    monkeypatch.setattr(server, "DashboardServer", FakeServer)
    # One cycle, which will also fail: the loop must survive it and return.
    monkeypatch.setattr(cli.TradingBot, "cycle", lambda self: CycleResult(ts=0.0, cycle=1, equity=0.0))
    code = main(["run", "--cycles", "1", "--interval", "0",
                 "--set", "server.db=/tmp/jevbot-test-served.db"])
    out = capsys.readouterr().out
    assert started.get("started"), "the dashboard must start even when the feed does not"
    assert code == 0
    assert "testnet.binance.vision" in out, out
    assert "warning" in out.lower()


def test_a_port_already_in_use_is_explained_not_raised(monkeypatch, capsys):
    from jevbot import cli, server

    class Busy:
        host, port = "0.0.0.0", 8000

        def __init__(self, *_a, **_kw):
            pass

        def start(self, background=False):
            raise OSError(98, "Address already in use")

        def stop(self):
            pass

    monkeypatch.setattr(cli.TradingBot, "warmup", lambda self: None)
    monkeypatch.setattr(server, "DashboardServer", Busy)
    code = main(["run", "--cycles", "0", "--set", "server.db=/tmp/jevbot-test-busy.db"])
    out = capsys.readouterr().out
    assert code == 4
    assert "Address already in use" in out or "already" in out
    assert "--port" in out
