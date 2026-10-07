"""The record.

The store is not a logging convenience: the engine's accuracy is measured from
the decisions written here, so a dropped column or a mangled timestamp changes
the verdict on the strategy.
"""

from __future__ import annotations

import json

import pytest

from jevbot.store import Store
from jevbot.types import Decision, Fill, NewsItem, Order, Position, Side, Signal

TS = 1_700_000_000.0


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_a_cycle_and_its_equity_point_round_trip(store):
    store.record_cycle(TS, equity=100_500.0, cash=20_000.0, gross=0.8, net=0.6,
                       realized=500.0, unrealized=0.0, engine="heuristic", model="heuristic")
    store.record_equity(TS, 100_500.0, 20_000.0, 0.8, 0.6, 0.001, 3)
    rows = store.latest_equity(10)
    assert len(rows) == 1
    assert rows[0]["equity"] == pytest.approx(100_500.0)
    assert rows[0]["gross_weight"] == pytest.approx(0.8)
    assert store.counts()["equity"] == 1


def test_decisions_join_back_to_the_headline_they_were_about(store):
    d = Decision(news_id="n1", symbol="BTC/USDT", ts=TS, engine="heuristic", model="heuristic",
                 direction="long", direction_p=0.6, materiality=0.8, conviction_norm=0.5,
                 priced_in=1.0, context_aligned=0.7, horizon="swing", risk_event=0.1,
                 raw={"answers": {"materiality": {"noul": 0.8}}})
    store.record_decisions([d])
    text = "Bitcoin ETF inflows hit a record"
    store.record_news([(NewsItem(text=text, news_id="n1", ts=TS, source="demo"),
                        ("BTC/USDT",), "keyword match")])

    rows = store.recent_decisions(5)
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "BTC/USDT" and r["direction"] == "long"
    assert r["news_id"] == "n1" and r["horizon"] == "swing"
    assert r["news_text"] == text, "a decision has to be readable without a second query"
    assert json.loads(r["answers"])["materiality"]["noul"] == pytest.approx(0.8)

    news = store.recent_news(5)
    assert json.loads(news[0]["symbols"]) == ["BTC/USDT"]


def test_orders_and_fills_keep_the_reason_that_produced_them(store):
    o = Order(symbol="AAPL", side=Side.BUY, qty=3.0, ts=TS, reason="rebalance")
    store.record_orders([(o, "submitted", "paper")])
    rows = store.recent_orders(5)
    assert rows[0]["reason"] == "rebalance"
    assert rows[0]["status"] == "submitted"

    f = Fill(symbol="AAPL", side=Side.BUY, qty=3.0, price=180.0, fee=0.27, ts=TS,
             order_id=o.client_id)
    store.record_fills([f])
    fills = store.recent_fills(5)
    assert fills[0]["notional"] == pytest.approx(540.0)
    assert fills[0]["fee"] == pytest.approx(0.27)
    assert fills[0]["order_client_id"] == o.client_id, "a fill has to link to its order"


def test_positions_are_snapshots_not_live_rows(store):
    pos = Position(symbol="AAPL", qty=3.0, avg_price=180.0, last_price=181.0, realized_pnl=3.0,
                   opened_ts=TS)
    store.record_positions(TS, [pos])
    pos.qty = 99.0  # the stored row must not move with the live object
    rows = store.query("SELECT * FROM positions")
    assert len(rows) == 1
    assert rows[0]["qty"] == pytest.approx(3.0)
    assert rows[0]["unrealized_pnl"] == pytest.approx(3.0)


def test_signals_survive_with_their_contributors(store):
    s = Signal(symbol="BTC/USDT", ts=TS, score=0.42, target_weight=0.15, direction="long",
               horizon="swing", contributors=("materiality", "conviction"), decisions=("n1",))
    store.record_signals([s])
    row = store.recent_signals(5)[0]
    assert row["score"] == pytest.approx(0.42)
    assert json.loads(row["contributors"]) == ["materiality", "conviction"]
    assert json.loads(row["decisions"]) == ["n1"]


def test_risk_events_are_recorded_with_detail(store):
    store.record_risk_event("kill", {"reason": "max drawdown 21% >= 20%"})
    events = store.risk_events(5)
    assert events[0]["event"] == "kill"
    assert "max drawdown" in events[0]["detail"]


def test_counts_answers_for_every_table_the_dashboard_reads(store):
    counts = store.counts()
    for table in ("news", "decisions", "signals", "orders", "fills", "equity"):
        assert table in counts, f"{table} missing from counts()"
    assert all(v == 0 for v in counts.values())


def test_a_file_backed_store_survives_reopening(tmp_path):
    path = tmp_path / "jevbot.db"
    s = Store(str(path))
    s.record_equity(TS, 101_000.0, 10_000.0, 0.9, 0.9, 0.0, 2)
    s.close()

    again = Store(str(path))
    assert again.counts()["equity"] == 1
    assert again.latest_equity(1)[0]["equity"] == pytest.approx(101_000.0)
    again.close()


def test_latest_equity_is_chronological_whatever_order_it_was_written(store):
    for i in range(5):
        store.record_equity(TS + i * 60, 100_000.0 + i, 0.0, 0.0, 0.0, 0.0, 0)
    points = store.latest_equity(3)
    assert [p["ts"] for p in points] == [TS + 2 * 60, TS + 3 * 60, TS + 4 * 60]
