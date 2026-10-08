"""Core value types shared by every layer of the bot.

These are deliberately plain dataclasses: they serialise to JSON with
``dataclasses.asdict`` and they carry no behaviour beyond a couple of derived
properties, so the same objects flow through the live loop, the backtester and
the dashboard without adapters.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


@dataclass(frozen=True)
class Instrument:
    """One tradable symbol, in the form the strategy sees it."""

    symbol: str
    market: Literal["crypto", "equity"]
    name: str = ""

    @property
    def asset_class(self) -> str:
        return self.market


@dataclass
class NewsItem:
    """One piece of text the engine will read.

    ``symbols`` is the *routing* decision — which instruments this text is even
    about — and is made upstream of the engine (by an explicit tagging phase or
    a Laya shortlist call, see :mod:`jevbot.signals`). It is never a hint about
    direction: direction is the engine's job.
    """

    text: str
    ts: float = field(default_factory=time.time)
    symbols: tuple[str, ...] = ()
    source: str = "unknown"
    url: str = ""
    news_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketSnapshot:
    """A candle-free numeric view of one instrument at one instant.

    The bot never feeds raw numeric series to the engine. :func:`jevbot.text.render_state`
    turns this into prose, because the text *is* the model input — a table of
    floats would read as noise to a language-shaped decision head.
    """

    symbol: str
    ts: float
    last: float
    ret_1h: float = 0.0
    ret_24h: float = 0.0
    ret_7d: float = 0.0
    vol_24h: float = 0.0          # realised vol, fraction of price
    volume_z: float = 0.0         # volume z-score vs its 24h mean
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    range_pos: float = 0.5        # where `last` sits in the last 24h range, 0..1
    half_spread_bps: float = 1.5
    funding: float | None = None  # perps only
    quote_volume: float = 0.0

    @property
    def trend(self) -> str:
        if not self.ema_fast or not self.ema_slow:
            return "unknown"
        gap = (self.ema_fast - self.ema_slow) / self.ema_slow
        if gap > 0.004:
            return "up"
        if gap < -0.004:
            return "down"
        return "sideways"

    @property
    def market_move_today(self) -> float:
        """How much of the day's move is already banked (signed)."""
        return self.ret_24h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """The engine's typed read of one (news, symbol) pair.

    Every field is a probability or a bounded score, exactly as Laya returns
    them, plus the routing metadata that says which checkpoint answered and why.
    ``abstained`` records that the answer fell below the configured confidence
    gate — the signal layer treats an abstention as "no opinion", never as "no".
    """

    news_id: str
    symbol: str
    ts: float
    engine: str
    model: str = ""                     # english | multilingual | heuristic
    routing_reason: str = ""

    direction: str = "flat"             # long | flat | short
    direction_p: float = 0.0            # P(chosen direction)
    direction_probs: dict[str, float] = field(default_factory=dict)

    materiality: float = 0.0            # P(the news is tradeable)
    conviction: float = 0.0             # expected value over the conviction scale
    conviction_norm: float = 0.0        # that value mapped to 0..1
    priced_in: float = 0.0              # expected value 0..3, 0 = not yet reflected
    context_aligned: float = 0.0        # P(with-trend reading)
    horizon: str = "swing"              # intraday | swing | positional
    risk_event: float = 0.0             # P(tail / unmodellable event)

    abstained: bool = False
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    # ── derived ────────────────────────────────────────────────────────────
    @property
    def direction_sign(self) -> int:
        if self.direction == "long":
            return 1
        if self.direction == "short":
            return -1
        return 0

    @property
    def priced_in_factor(self) -> float:
        """1.0 for text the market has not reflected, 0.25 for stale news."""
        return 0.25 + 0.25 * max(0.0, min(3.0, self.priced_in))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    """A sized, risk-aware intention: how much of ``symbol`` to hold."""

    symbol: str
    ts: float
    score: float                # -1..1 fused engine output
    target_weight: float        # -max_weight..+max_weight after risk governor
    direction: str = "flat"
    horizon: str = "swing"
    contributors: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()   # news ids behind it
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    ts: float = field(default_factory=time.time)
    order_type: str = "market"
    limit_price: float | None = None
    reason: str = ""
    notes: str = ""          # why the risk layer changed this ticket, if it did
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        """Normalise `side` at the boundary.

        `Side` is a `str` enum, so ``"buy"`` and ``Side.BUY`` hash and compare
        equal — but the brokers test direction with ``order.side is Side.BUY``,
        and a plain string fails that test while passing every equality check.
        The result is not an error: it is an order that fills below the mid on
        the way in and a stop that closes the wrong way. Coercing here makes the
        `is` comparisons safe for callers that never imported the enum.
        """
        if not isinstance(self.side, Side):
            self.side = Side(self.side)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        return d


@dataclass
class Fill:
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    ts: float
    slippage: float = 0.0
    order_id: str = ""

    def __post_init__(self) -> None:
        """Normalise `side` at the boundary.

        `Side` is a `str` enum, so ``"buy"`` and ``Side.BUY`` hash and compare
        equal — but the brokers test direction with ``order.side is Side.BUY``,
        and a plain string fails that test while passing every equality check.
        The result is not an error: it is an order that fills below the mid on
        the way in and a stop that closes the wrong way. Coercing here makes the
        `is` comparisons safe for callers that never imported the enum.
        """
        if not isinstance(self.side, Side):
            self.side = Side(self.side)

    @property
    def notional(self) -> float:
        return self.qty * self.price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["notional"] = self.notional
        return d


#: Quantities below this are rounding residue, not positions. A portfolio that
#: treats a stray 1e-12 units as a holding will re-close it forever, so "flat"
#: has to be a tolerance rather than an exact zero.
DUST_QUANTITY = 1e-9


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    opened_ts: float = 0.0
    last_price: float = 0.0
    stop_price: float | None = None

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < DUST_QUANTITY

    @property
    def notional(self) -> float:
        return self.qty * (self.last_price or self.avg_price)

    def unrealized(self) -> float:
        if self.is_flat:
            return 0.0
        return (self.last_price - self.avg_price) * self.qty

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_flat"] = self.is_flat
        d["notional"] = self.notional
        d["unrealized_pnl"] = self.unrealized()
        return d
