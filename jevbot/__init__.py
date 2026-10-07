"""Jevbot — a news-driven algorithmic trading bot whose signal engine is Laya.

The pipeline, in one line: text in, typed decisions out, risk-gated orders after.

    news + market context -> state text -> Laya typed questions -> fused signal
        -> vol-targeted size -> risk governor -> broker (paper by default)

Nothing in the core package imports a third-party library at module scope. Laya,
ccxt and alpaca are imported only when the corresponding component is actually
built, so the bot, the backtester and the dashboard run on a bare interpreter.
"""

from .config import Config, load_config
from .types import (
    Decision,
    Fill,
    Instrument,
    MarketSnapshot,
    NewsItem,
    Order,
    Position,
    Side,
    Signal,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "Decision",
    "Fill",
    "Instrument",
    "MarketSnapshot",
    "NewsItem",
    "Order",
    "Position",
    "Side",
    "Signal",
    "__version__",
]
