"""SQLite persistence for everything the bot did.

The dashboard reads from here, the eval script reads from here, and a post-mortem
reads from here — so the schema is designed around one question: *given a trade,
what did the bot know and what did it decide?* A fill links to the orders that
produced it, an order links to the signal that motivated it, and a signal links
to the individual typed decisions behind it, each with the engine that answered
and the checkpoint the Router chose.

Nothing is pruned automatically. Storage is cheap; a missing row is not.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .types import Decision, Fill, NewsItem, Order, Position, Signal

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    equity REAL, cash REAL, gross_weight REAL, net_weight REAL,
    realized_pnl REAL, unrealized_pnl REAL,
    engine TEXT, model TEXT, halted INTEGER DEFAULT 0, notes TEXT
);
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY, ts REAL, source TEXT, text TEXT, symbols TEXT,
    routing_reason TEXT, url TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, news_id TEXT, symbol TEXT, engine TEXT, model TEXT, routing_reason TEXT,
    direction TEXT, direction_p REAL, materiality REAL, conviction REAL, conviction_norm REAL,
    priced_in REAL, context_aligned REAL, horizon TEXT, risk_event REAL,
    abstained INTEGER, latency_ms REAL, answers TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, symbol TEXT, score REAL, target_weight REAL, direction TEXT, horizon TEXT,
    decisions TEXT, contributors TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, client_id TEXT, symbol TEXT, side TEXT, qty REAL, order_type TEXT,
    reason TEXT, status TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, order_client_id TEXT, symbol TEXT, side TEXT, qty REAL, price REAL,
    fee REAL, slippage REAL, notional REAL
);
CREATE TABLE IF NOT EXISTS positions (
    ts REAL, symbol TEXT, qty REAL, avg_price REAL, last_price REAL,
    realized_pnl REAL, unrealized_pnl REAL, notional REAL
);
CREATE TABLE IF NOT EXISTS equity (
    ts REAL PRIMARY KEY, equity REAL, cash REAL, gross_weight REAL, net_weight REAL,
    drawdown REAL, positions INTEGER
);
CREATE TABLE IF NOT EXISTS risk_events (
    ts REAL, event TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS engine_info (
    ts REAL, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_news ON decisions(news_id, symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_news_ts ON news(ts);
"""


class Store:
    def __init__(self, path: str | Path = "data/store/jevbot.db") -> None:
        if str(path) == ":memory:":
            self.path = Path(":memory:")
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.cycle_id = 0

    # ── writes ─────────────────────────────────────────────────────────────

    def record_cycle(self, ts: float, equity: float, cash: float, gross: float, net: float,
                     realized: float, unrealized: float, engine: str, model: str,
                     halted: bool = False, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO cycles (ts, equity, cash, gross_weight, net_weight, realized_pnl,"
            " unrealized_pnl, engine, model, halted, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ts, equity, cash, gross, net, realized, unrealized, engine, model, int(halted), notes),
        )
        self.cycle_id = int(cur.lastrowid or 0)
        self.conn.commit()
        return self.cycle_id

    def record_news(self, items: Iterable[tuple[NewsItem, Sequence[str], str]]) -> None:
        rows = [
            (item.news_id, item.ts, item.source, item.text, json.dumps(list(syms)),
             reason, item.url)
            for item, syms, reason in items
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO news (id, ts, source, text, symbols, routing_reason, url)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_decisions(self, decisions: Iterable[Decision]) -> None:
        rows = [
            (
                d.ts, d.news_id, d.symbol, d.engine, d.model, d.routing_reason,
                d.direction, d.direction_p, d.materiality, d.conviction, d.conviction_norm,
                d.priced_in, d.context_aligned, d.horizon, d.risk_event,
                int(d.abstained), d.latency_ms,
                json.dumps((d.raw or {}).get("answers", {}), default=str),
            )
            for d in decisions
        ]
        self.conn.executemany(
            "INSERT INTO decisions (ts, news_id, symbol, engine, model, routing_reason,"
            " direction, direction_p, materiality, conviction, conviction_norm, priced_in,"
            " context_aligned, horizon, risk_event, abstained, latency_ms, answers)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_signals(self, signals: Iterable[Signal]) -> None:
        rows = [
            (s.ts, s.symbol, s.score, s.target_weight, s.direction, s.horizon,
             json.dumps(list(s.decisions)), json.dumps(list(s.contributors)), s.notes)
            for s in signals
        ]
        self.conn.executemany(
            "INSERT INTO signals (ts, symbol, score, target_weight, direction, horizon,"
            " decisions, contributors, notes) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_orders(self, orders: Iterable[tuple[Order, str, str]]) -> None:
        rows = [
            (o.ts, o.client_id, o.symbol, o.side.value, o.qty, o.order_type,
             o.reason, status, detail)
            for o, status, detail in orders
        ]
        self.conn.executemany(
            "INSERT INTO orders (ts, client_id, symbol, side, qty, order_type, reason,"
            " status, detail) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_fills(self, fills: Iterable[Fill]) -> None:
        rows = [
            (f.ts, f.order_id, f.symbol, f.side.value, f.qty, f.price, f.fee,
             f.slippage, f.notional)
            for f in fills
        ]
        self.conn.executemany(
            "INSERT INTO fills (ts, order_client_id, symbol, side, qty, price, fee,"
            " slippage, notional) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_positions(self, ts: float, positions: Iterable[Position]) -> None:
        rows = [
            (ts, p.symbol, p.qty, p.avg_price, p.last_price, p.realized_pnl,
             p.unrealized(), p.notional)
            for p in positions
        ]
        self.conn.executemany(
            "INSERT INTO positions (ts, symbol, qty, avg_price, last_price, realized_pnl,"
            " unrealized_pnl, notional) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_equity(self, ts: float, equity: float, cash: float, gross: float, net: float,
                      drawdown: float, positions: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO equity (ts, equity, cash, gross_weight, net_weight,"
            " drawdown, positions) VALUES (?,?,?,?,?,?,?)",
            (ts, equity, cash, gross, net, drawdown, positions),
        )
        self.conn.commit()

    def record_risk_event(self, event: str, detail: Any) -> None:
        self.conn.execute(
            "INSERT INTO risk_events (ts, event, detail) VALUES (?,?,?)",
            (time.time(), event, json.dumps(detail, default=str)),
        )
        self.conn.commit()

    def record_engine_info(self, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO engine_info (ts, payload) VALUES (?,?)",
            (time.time(), json.dumps(payload, default=str)),
        )
        self.conn.commit()

    # ── reads (dashboard, reports, evals) ──────────────────────────────────

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def latest_equity(self, limit: int = 2000) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM equity ORDER BY ts DESC LIMIT ?", (limit,))
        return list(reversed(rows))

    def recent_decisions(self, limit: int = 60) -> list[dict[str, Any]]:
        return self.query(
            "SELECT d.*, n.text AS news_text, n.symbols AS news_symbols, n.source AS news_source"
            " FROM decisions d LEFT JOIN news n ON n.id = d.news_id"
            " ORDER BY d.id DESC LIMIT ?",
            (limit,),
        )

    def recent_signals(self, limit: int = 40) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))

    def recent_orders(self, limit: int = 60) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))

    def recent_fills(self, limit: int = 60) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,))

    def recent_news(self, limit: int = 40) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM news ORDER BY ts DESC LIMIT ?", (limit,))

    def risk_events(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM risk_events ORDER BY ts DESC LIMIT ?", (limit,))

    def counts(self) -> dict[str, int]:
        out = {}
        for table in ("news", "decisions", "signals", "orders", "fills", "equity"):
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[table] = int(row["n"] if row else 0)
        return out

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:  # pragma: no cover
            pass
