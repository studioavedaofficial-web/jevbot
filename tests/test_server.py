"""The dashboard's API, over a real socket.

The dashboard is the only part of this project a human watches, and it is the
part most likely to be quietly wrong: a broken endpoint shows an empty panel,
and an empty panel looks like "no trades yet". So the test drives a real HTTP
server on a real port and asserts on the JSON contract the page consumes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from jevbot.server import DashboardServer


@pytest.fixture
def served(rig):
    bot, clock, market = rig
    bot.cycle()
    clock["now"] += 300.0
    bot.cycle()
    server = DashboardServer(bot, bot.store, host="127.0.0.1", port=0)
    server.start(background=True)
    port = server.httpd.server_address[1]  # type: ignore[union-attr]
    yield f"http://127.0.0.1:{port}", bot
    server.stop()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def post(url):
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def test_the_page_and_its_assets_are_served(served):
    base, _ = served
    for path, needle in (("/", "<html"), ("/static/app.js", "function"),
                         ("/static/app.css", "{")):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            body = r.read().decode()
        assert needle in body, f"{path} did not look like the expected asset"
        assert r.headers["Content-Type"].startswith(("text/html", "application/javascript",
                                                     "text/css"))


def test_state_carries_everything_the_kpis_need(served):
    base, bot = served
    payload = get(base + "/api/state")
    assert payload["live"] is True
    inner = payload["bot"]
    for key in ("cycle", "now", "engine", "broker", "portfolio", "risk", "signals",
                "positions", "events", "news"):
        assert key in inner, f"the dashboard reads bot.{key}"
    assert inner["cycle"] == 2
    assert "equity" in inner["portfolio"]
    assert "daily_pnl_pct" in inner["risk"]
    assert "limits" in inner["risk"]


def test_equity_points_are_arrays_the_chart_can_plot(served):
    base, _ = served
    points = get(base + "/api/equity?limit=100")["points"]
    assert len(points) == 2
    ts, equity, gross, net, dd, positions = points[-1]
    assert ts > 0 and equity > 0
    assert 0.0 <= gross <= 2.0 and -1.0 <= net <= 1.0


def test_decisions_keep_the_headline_and_the_answers(served):
    base, _ = served
    decisions = get(base + "/api/decisions?limit=10")["decisions"]
    assert decisions, "two cycles of a demo market should have produced decisions"
    d = decisions[0]
    for key in ("news_id", "symbol", "engine", "direction", "materiality", "conviction_norm",
                "priced_in", "context_aligned", "risk_event", "horizon"):
        assert key in d
    assert d["news_text"], "a decision without its headline is unreadable"


def test_orders_endpoint_returns_both_orders_and_fills(served):
    base, _ = served
    payload = get(base + "/api/orders?limit=10")
    assert "orders" in payload and "fills" in payload


def test_risk_endpoint_reports_the_governor(served):
    base, _ = served
    payload = get(base + "/api/risk")
    assert "status" in payload and "events" in payload
    assert payload["status"]["killed"] is False


def test_control_endpoints_actually_move_the_bot(served):
    base, bot = served
    assert post(base + "/api/control/pause")["ok"] is True
    assert bot.paused is True
    assert post(base + "/api/control/resume")["ok"] is True
    assert bot.paused is False
    assert post(base + "/api/control/stop")["ok"] is True
    assert bot.stop_requested is True
    # a stopped bot refuses to run more cycles, which is the whole point
    assert bot.run(cycles=1, interval=0.0) == []


def test_flatten_button_closes_the_book(served):
    base, bot = served
    if not bot.portfolio.open_positions():
        # the demo tape may not have put a position on in two cycles; open one
        # the way a fill does, so the button is exercised either way
        from jevbot.types import Fill, Side

        snap = bot.latest_snapshots[bot.symbols[0]]
        bot.portfolio.apply_fill(Fill(symbol=snap.symbol, side=Side.BUY, qty=1.0,
                                      price=snap.last, fee=0.0, ts=bot.now()))
    assert bot.portfolio.open_positions()
    post(base + "/api/control/flatten_now")
    assert bot.portfolio.open_positions() == []


def test_an_unknown_action_is_refused_not_ignored(served):
    base, _ = served
    payload = post(base + "/api/control/launch_missiles")
    assert payload["ok"] is False
    assert "launch_missiles" in payload["error"]


def test_health_is_cheap_and_always_answers(served):
    base, _ = served
    assert get(base + "/api/health")["ok"] is True


def test_a_missing_route_is_a_404(served):
    base, _ = served
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/api/nonsense", timeout=10)
    assert exc.value.code == 404
