"""Portfolio accounting: cash, positions, weights, P&L.

One implementation serves the paper broker, the backtester and (as the mirror of
broker-reported state) the live adapters, so a fill booked live and a fill
booked in a replay follow identical arithmetic. Average-cost basis, realised P&L
on the closing quantity, fees always charged, and a weight definition that is
always *notional / equity* — never notional over something else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .types import DUST_QUANTITY, Fill, Instrument, Position, Side


@dataclass
class PortfolioSnapshot:
    ts: float
    equity: float
    cash: float
    gross_weight: float
    net_weight: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    positions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "equity": round(self.equity, 4),
            "cash": round(self.cash, 4),
            "gross_weight": round(self.gross_weight, 6),
            "net_weight": round(self.net_weight, 6),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "fees_paid": round(self.fees_paid, 4),
            "positions": self.positions,
        }


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    fees_paid: float = 0.0
    realized_pnl: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    starting_equity: float = 0.0
    peak_equity: float = 0.0
    equity_curve: list[tuple[float, float]] = field(default_factory=list)

    # ── lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    def fresh(cls, starting_equity: float) -> Portfolio:
        p = cls(cash=float(starting_equity), starting_equity=float(starting_equity))
        p.peak_equity = float(starting_equity)
        return p

    def position(self, symbol: str) -> Position:
        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(symbol=symbol)
            self.positions[symbol] = pos
        return pos

    # ── marks and equity ───────────────────────────────────────────────────

    def mark(self, prices: dict[str, float], ts: float | None = None, record: bool = True) -> float:
        self.marks.update({k: v for k, v in prices.items() if v})
        for symbol, pos in self.positions.items():
            if symbol in prices and prices[symbol]:
                pos.last_price = prices[symbol]
        equity = self.equity()
        self.peak_equity = max(self.peak_equity, equity)
        if record:
            self.equity_curve.append((ts if ts is not None else time.time(), equity))
        return equity

    def equity(self) -> float:
        total = self.cash
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            price = self.marks.get(symbol) or pos.last_price or pos.avg_price
            total += pos.qty * price
        return total

    def unrealized(self) -> float:
        return sum(p.unrealized() for p in self.positions.values())

    def gross_weight(self) -> float:
        eq = self.equity()
        if eq <= 0:
            return 0.0
        return sum(abs(p.qty * (self.marks.get(s) or p.last_price or p.avg_price))
                   for s, p in self.positions.items()) / eq

    def net_weight(self) -> float:
        eq = self.equity()
        if eq <= 0:
            return 0.0
        return sum(p.qty * (self.marks.get(s) or p.last_price or p.avg_price)
                   for s, p in self.positions.items()) / eq

    def weights(self) -> dict[str, float]:
        eq = self.equity()
        if eq <= 0:
            return {}
        out = {}
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            price = self.marks.get(symbol) or pos.last_price or pos.avg_price
            out[symbol] = pos.qty * price / eq
        return out

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_flat]

    def drawdown(self) -> float:
        eq = self.equity()
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - eq / self.peak_equity)

    # ── fills ──────────────────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> Position:
        """Book a fill with average-cost basis and immediate fee accounting.

        Two phases, deliberately: everything that can raise happens first, and
        only then is state written. A booking that fails halfway leaves the cash
        debited and no position to show for it — a corrupt book that no
        subsequent account reconciliation can repair, and one this code has
        already produced once.
        """
        pos = self.position(fill.symbol)
        cash_delta = -fill.qty * fill.price * (1 if fill.side is Side.BUY else -1)
        signed_qty = fill.qty if fill.side is Side.BUY else -fill.qty

        # ── phase 1: decide everything
        opening = pos.is_flat or (pos.qty > 0) == (signed_qty > 0)
        new_qty = pos.qty + signed_qty
        new_avg = pos.avg_price
        realized_delta = 0.0
        new_opened_ts = pos.opened_ts
        clear_stop = False

        if opening:
            if abs(new_qty) > DUST_QUANTITY:
                new_avg = (pos.avg_price * abs(pos.qty) + fill.price * abs(signed_qty)) / abs(new_qty)
            else:
                new_avg = fill.price
            if not new_opened_ts:
                new_opened_ts = fill.ts
            # A fee is realised the moment it is paid. Charging only the exit
            # fee to this position's P&L would make a flat book report a
            # smaller loss than the cash says it lost — and the entry fee is
            # the half of the round trip that is easy to forget.
            realized_delta = -fill.fee
        else:
            closing = min(abs(signed_qty), abs(pos.qty))
            direction = 1 if pos.qty > 0 else -1
            realized_delta = (fill.price - pos.avg_price) * closing * direction - fill.fee
            if abs(new_qty) < DUST_QUANTITY * 1e3:
                new_qty, new_avg, new_opened_ts, clear_stop = 0.0, 0.0, 0.0, True
            elif (new_qty > 0) != (direction > 0):
                new_avg, new_opened_ts = fill.price, fill.ts    # flipped through zero

        # ── phase 2: commit
        self.cash += cash_delta - fill.fee
        self.fees_paid += fill.fee
        self.fills.append(fill)
        self.marks[fill.symbol] = fill.price

        pos.qty = new_qty
        pos.avg_price = new_avg
        pos.opened_ts = new_opened_ts
        pos.last_price = fill.price
        pos.realized_pnl += realized_delta
        self.realized_pnl += realized_delta
        if clear_stop:
            pos.stop_price = None
        return pos

    # ── reporting ──────────────────────────────────────────────────────────

    def snapshot(self, ts: float | None = None) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            ts=ts if ts is not None else time.time(),
            equity=self.equity(),
            cash=self.cash,
            gross_weight=self.gross_weight(),
            net_weight=self.net_weight(),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized(),
            fees_paid=self.fees_paid,
            positions=len(self.open_positions()),
        )

    def to_dict(self, instruments: Iterable[Instrument] = ()) -> dict[str, Any]:
        names = {i.symbol: i.name for i in instruments}
        return {
            "cash": round(self.cash, 2),
            "equity": round(self.equity(), 2),
            "starting_equity": round(self.starting_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized(), 2),
            "fees_paid": round(self.fees_paid, 2),
            "gross_weight": round(self.gross_weight(), 4),
            "net_weight": round(self.net_weight(), 4),
            "drawdown": round(self.drawdown(), 4),
            "positions": [
                {**p.to_dict(), "name": names.get(p.symbol, p.symbol)}
                for p in sorted(self.open_positions(), key=lambda x: -abs(x.notional))
            ],
        }
