"""Turning numbers into the text the decision engine actually reads.

This module is the reason the bot is not a pile of indicators. Laya is a
*text* decision engine operating over a state; a matrix of floats is not a state
it can read, and bolting a numeric head onto it would throw away the thing being
tested. So every numeric input is rendered, in a fixed order and with fixed
wording, into a short factual brief.

Two rules make that renderer safe to re-use everywhere:

* **Deterministic wording.** The same numbers always produce the same string, so
  a cached decision is a valid decision, and a backtest replay is exact.
* **Facts, not conclusions.** The brief says "the fast average is above the slow
  average", never "uptrend confirmed". Judgement belongs to the engine; that is
  the division of labour the whole project rests on.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .types import Instrument, MarketSnapshot, NewsItem

# ── number formatting ───────────────────────────────────────────────────────


def fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def fmt_pct(value: float, digits: int = 2) -> str:
    """Signed percentage from a fraction, e.g. 0.018 -> '+1.80%'."""
    return f"{value * 100:+.{digits}f}%"


def _magnitude(value: float) -> str:
    a = abs(value)
    if a < 0.002:
        return "flat"
    if a < 0.01:
        return "small"
    if a < 0.03:
        return "moderate"
    if a < 0.08:
        return "large"
    return "extreme"


def _vol_label(vol: float) -> str:
    a = abs(vol)
    if a < 0.005:
        return "calm"
    if a < 0.015:
        return "normal"
    if a < 0.035:
        return "elevated"
    return "turbulent"


def _z_label(z: float) -> str:
    if z > 2:
        return "far above"
    if z > 0.75:
        return "above"
    if z > -0.75:
        return "in line with"
    if z > -2:
        return "below"
    return "far below"


def _range_label(pos: float) -> str:
    if pos >= 0.85:
        return "at the top of the 24h range"
    if pos >= 0.6:
        return "in the upper half of the 24h range"
    if pos > 0.4:
        return "near the middle of the 24h range"
    if pos > 0.15:
        return "in the lower half of the 24h range"
    return "at the bottom of the 24h range"


def _ago(ts: float, now: float) -> str:
    delta = max(0.0, now - ts)
    if delta < 90:
        return f"{int(delta)} seconds ago"
    if delta < 5400:
        return f"{int(delta // 60)} minutes ago"
    if delta < 172800:
        return f"{int(delta // 3600)} hours ago"
    return f"{int(delta // 86400)} days ago"


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def trend_phrase(snap: MarketSnapshot) -> str:
    """The same trend read as :attr:`MarketSnapshot.trend`, in words."""
    return {
        "up": "the faster moving average sits above the slower one",
        "down": "the faster moving average sits below the slower one",
        "sideways": "the two moving averages are level with each other",
        "unknown": "not enough history for moving averages",
    }[snap.trend]


# ── renderers ───────────────────────────────────────────────────────────────


def render_market(snap: MarketSnapshot, inst: Instrument | None = None) -> str:
    """One instrument's numeric context as a fixed-format prose block."""
    label = f"{inst.name} ({snap.symbol})" if inst and inst.name else snap.symbol
    lines = [
        f"INSTRUMENT: {label}",
        f"PRICE: {fmt_price(snap.last)}",
        (
            "PERFORMANCE: "
            f"{fmt_pct(snap.ret_1h)} over the last hour, "
            f"{fmt_pct(snap.ret_24h)} over 24 hours, "
            f"{fmt_pct(snap.ret_7d)} over 7 days"
        ),
        f"TREND: {trend_phrase(snap)}",
        (
            f"VOLATILITY: 24h realised {fmt_pct(snap.vol_24h)} of price, a "
            f"{_vol_label(snap.vol_24h)} regime; the last 24 hours moved "
            f"{_magnitude(snap.ret_24h)} for this instrument"
        ),
        f"VOLUME: trading volume is {_z_label(snap.volume_z)} its 24h average",
        f"RANGE POSITION: price is {_range_label(snap.range_pos)}",
    ]
    if snap.funding is not None:
        side = "longs are paying shorts" if snap.funding > 0 else "shorts are paying longs"
        lines.append(
            "FUNDING: perpetual funding is "
            f"{fmt_pct(snap.funding, 4)} per 8 hours, so {side} "
            f"({_magnitude(snap.funding)})"
        )
    if snap.half_spread_bps:
        lines.append(f"SPREAD: {snap.half_spread_bps:.1f} basis points each side")
    return "\n".join(lines)


def render_news(items: list[NewsItem], now: float | None = None) -> str:
    """A numbered list of headlines, oldest first, with source and age."""
    if not items:
        return "No news in the last few hours."
    now = now if now is not None else max(i.ts for i in items)
    lines = []
    for i, item in enumerate(sorted(items, key=lambda x: x.ts), 1):
        src = f", {item.source}" if item.source else ""
        lines.append(f"{i}. [{_clock(item.ts)} — {_ago(item.ts, now)}{src}] {item.text.strip()}")
    return "\n".join(lines)


def render_state(
    snap: MarketSnapshot,
    headlines: list[NewsItem] | None = None,
    inst: Instrument | None = None,
    focus: NewsItem | None = None,
    extra: str = "",
) -> str:
    """The full state for one instrument: market context, then the news.

    The ordering is fixed and load-bearing. Context comes first so a headline is
    always read against "where is price and what is it doing"; then the headline
    under consideration, then the earlier items as backdrop.

    ``focus`` is not decoration. A decision is attributed to *one* headline —
    that is what makes it scoreable against that headline's forward return — so
    the state has to say which headline the question is about. Without it, a
    model reading a list answers about the list, and every downstream metric
    measures something other than what it claims to.

    ``extra`` is the escape hatch for an operator note (earnings date, a
    scheduled macro print) that should be visible without being silently part of
    the numbers.
    """
    now = snap.ts
    others = [h for h in (headlines or []) if focus is None or h.news_id != focus.news_id]
    parts = ["MARKET CONTEXT", render_market(snap, inst)]
    if focus is not None:
        parts += [
            "",
            "HEADLINE UNDER CONSIDERATION",
            render_news([focus], now=now),
            "",
            "EARLIER HEADLINES (context only, do not answer about these)",
            render_news(others[-6:], now=now),
        ]
    else:
        parts += ["", "RECENT NEWS", render_news(others, now=now)]
    if extra.strip():
        parts += ["", "OPERATOR NOTE", extra.strip()]
    return "\n".join(parts)


def render_portfolio(equity: float, gross: float, net: float, pnl_today: float, positions: int) -> str:
    """A short account brief, used for the daily risk memo (not per-trade)."""
    return (
        f"PORTFOLIO: equity {fmt_price(equity)}, gross exposure {gross * 100:.1f}% of equity, "
        f"net exposure {net * 100:+.1f}%, {positions} open positions, "
        f"today's P&L {fmt_pct(pnl_today)} of equity."
    )


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not math.isfinite(b):
        return default
    out = a / b
    return out if math.isfinite(out) else default
