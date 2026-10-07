"""A deterministic synthetic market that produces *both* prices and headlines.

This exists so the whole bot — engine, fusion, risk, execution, dashboard,
metrics — can be run and tested with no network, no API keys and no checkpoint
download, and so that the answer to "did the plumbing work?" is a number rather
than a shrug.

**What it is.** A seeded random walk per instrument, plus a stream of generated
headlines. A minority of headlines carry a hidden impact: they push the price in
a direction, with the push decaying over a few hours. Prices are simulated
*after* the headlines, so news genuinely precedes and causes the move.

**What it is not.** It is not evidence of edge, and no result on it should be
presented as one. The generative process is a toy: one headline, one linear
impulse, no microstructure, no regime shifts invented from nothing. A strategy
that reads these headlines well will look good here, which is exactly why the
harness ships a falsification control (:func:`shuffle_headlines` — permute the
labels and the edge must disappear). A real edge claim needs real data; the same
strategy core runs against ccxt or Alpaca bars with ``--price-feed ccxt``.

Determinism is the design constraint that shapes everything: the market is a
pure function of ``(seed, config)``, so prices are never stored — only the
generated headlines are written to disk, and the price path is recomputed
identically on demand.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from ..types import Instrument, MarketSnapshot, NewsItem
from .base import NewsFeed, PriceFeed
from .features import RollingFeatures

# ── headline templates ──────────────────────────────────────────────────────
# Each entry: (template, category, valence, typical magnitude on 0..1).
# `valence` is the *hidden* truth: -1 bearish, 0 neutral, +1 bullish. The engine
# never sees it; it is what the eval script scores the engine against.
# Decay half-life per category, in hours. The tape and the question the engine
# is asked about horizon have to agree, or the bot is being graded on a market
# that does not exist: an "intraday" reading of a story whose effect takes two
# days to play out is wrong in a way no amount of reading skill can fix.
CATEGORY_HALF_LIFE_HOURS: dict[str, float] = {
    "upgrade": 6.0, "downgrade": 6.0, "earnings": 8.0, "buyback": 8.0,
    "flows": 4.0, "technical": 2.0, "macro": 3.0, "neutral": 1.0,
    "partnership": 18.0, "product": 24.0, "regulation": 72.0,
    "security": 12.0, "legal": 30.0, "holder": 10.0, "demand": 16.0, "outage": 2.0,
}

TEMPLATES: list[tuple[str, str, int, float]] = [
    ("{name} jumps {pct}% after {partner} announces a multi-year partnership", "partnership", 1, 0.75),
    ("{name} slips {pct}% as analysts flag slowing demand in {region}", "demand", -1, 0.55),
    ("Regulators in {region} approve a long-awaited framework for {asset} trading", "regulation", 1, 0.7),
    ("{name} falls {pct}% after exchange reports {region} trading outage", "outage", -1, 0.6),
    ("{name} rises {pct}% as institutional inflows hit a record for the quarter", "flows", 1, 0.8),
    ("{name} drops {pct}% on reports of a large wallet moving to exchanges", "flows", -1, 0.45),
    ("{name} trades flat as traders await tomorrow's {macro} decision", "macro", 0, 0.1),
    ("Analysts raise their price target for {name} to {target}, citing {driver}", "upgrade", 1, 0.6),
    ("{name} downgraded to neutral as valuation looks stretched after the recent run", "downgrade", -1, 0.5),
    ("{name} confirms a {pct}% increase in quarterly revenue, beating expectations", "earnings", 1, 0.85),
    ("{name} misses revenue estimates, guides below consensus for next quarter", "earnings", -1, 0.85),
    ("{name} announces a ${bn}B share repurchase programme", "buyback", 1, 0.65),
    ("A security breach at a {asset} infrastructure provider rattles {name} holders", "security", -1, 0.8),
    ("{name} is reportedly considering a tokenised product for {region} clients", "product", 1, 0.35),
    ("Unconfirmed reports say {name} may face enforcement action in {region}", "regulation", -1, 0.5),
    ("{name} unchanged after the {macro} print came in line with expectations", "macro", 0, 0.05),
    ("{name} rallies {pct}% as short positioning unwinds across {region} venues", "flows", 1, 0.55),
    ("{name} slides {pct}% after a large holder signals it will reduce its stake", "holder", -1, 0.6),
    ("{name} gains {pct}% on news that {partner} will accept it as collateral", "partnership", 1, 0.6),
    ("{name} extends losses as {region} regulators tighten leverage limits", "regulation", -1, 0.55),
    ("{name} holds steady; commentary notes thin liquidity over the holiday", "neutral", 0, 0.05),
    ("{name} climbs {pct}% after a technical breakout triggers momentum buying", "technical", 1, 0.35),
    ("{name} plunges {pct}% as a major venue halts withdrawals", "security", -1, 0.9),
    ("{name} up {pct}% after {region} pension fund discloses a position", "flows", 1, 0.5),
    ("{name} faces a class action over disclosures during the last quarter", "legal", -1, 0.6),
    ("{name} down {pct}% on a broader risk-off move across markets", "macro", -1, 0.4),
    ("{name} launches a fee waiver programme, pressuring competitors' margins", "product", -1, 0.3),
    ("{name} adds {pct}% as an ETF issuer refiles with the regulator", "product", 1, 0.5),
    ("Traders debate whether {name} has already priced in the {macro} outcome", "neutral", 0, 0.15),
    ("{name} little changed after a {macro} headline that was widely telegraphed", "macro", 0, 0.05),
]

REGIONS = ["the United States", "Europe", "Asia", "the UK", "Singapore", "the UAE"]
MACROS = ["inflation print", "rate decision", "jobs report", "GDP release", "policy meeting"]
DRIVERS = ["margin expansion", "the new product cycle", "cost discipline", "share gains"]
PARTNERS = ["a major payments network", "a global bank", "a sovereign fund",
            "a top-five exchange", "a large retailer", "a cloud provider"]


def _hash_seed(*parts: Any) -> int:
    payload = "|".join(str(p) for p in parts)
    return int(hashlib.blake2b(payload.encode(), digest_size=8).hexdigest(), 16)


@dataclass
class Headline:
    """A generated headline plus the truth the engine is never shown."""

    item: NewsItem
    category: str
    valence: int
    magnitude: float          # 0..1 generative strength
    impact: float             # signed price impact applied to the tape
    symbol: str
    forward_return: float = 0.0   # realised return over the horizon after it

    def to_dict(self) -> dict[str, Any]:
        d = self.item.to_dict()
        d.update(
            {
                "category": self.category,
                "valence": self.valence,
                "magnitude": round(self.magnitude, 4),
                "impact": round(self.impact, 6),
                "truth_direction": "long" if self.valence > 0 else "short" if self.valence < 0 else "flat",
                "symbol": self.symbol,
                "forward_return": round(self.forward_return, 6),
            }
        )
        return d


class DemoMarket:
    """The seeded generative market: bars for every instrument, plus headlines.

    The price path is a pure function of the seed. Bars are generated lazily per
    instrument on first access and memoised, because generating 30 days of
    five-minute bars for five instruments is a few hundred thousand floats.
    """

    def __init__(
        self,
        instruments: Iterable[Instrument],
        *,
        seed: int = 7,
        days: float = 30.0,
        bar_seconds: int = 300,
        headlines_per_day: int = 24,
        impact_scale: float = 0.012,
        noise_share: float = 0.30,
        end_ts: float | None = None,
    ) -> None:
        self.instruments = list(instruments)
        self.seed = int(seed)
        self.days = float(days)
        self.bar_seconds = int(bar_seconds)
        self.headlines_per_day = int(headlines_per_day)
        self.impact_scale = float(impact_scale)
        self.noise_share = float(noise_share)
        self.end_ts = float(end_ts if end_ts is not None else time.time())
        self.start_ts = self.end_ts - self.days * 86400.0
        self.n_bars = int(self.days * 86400 / self.bar_seconds)
        self._bars: dict[str, list[float]] = {}
        self._returns: dict[str, list[float]] = {}
        self._volumes: dict[str, list[float]] = {}
        self._headlines: list[Headline] | None = None

    # ── generative core ────────────────────────────────────────────────────

    def _base_vol(self, symbol: str) -> float:
        """Per-bar volatility, per instrument, stable across runs."""
        rng = random.Random(_hash_seed(self.seed, "vol", symbol))
        if "/" in symbol:
            return 0.0022 * rng.uniform(0.8, 1.3)       # crypto: lively
        return 0.0009 * rng.uniform(0.8, 1.3)           # equities: calmer

    def _base_price(self, symbol: str) -> float:
        rng = random.Random(_hash_seed(self.seed, "price", symbol))
        if "/" in symbol:
            return 30000.0 if symbol.startswith("BTC") else 1800.0
        return rng.uniform(80.0, 420.0)

    def headlines(self) -> list[Headline]:
        if self._headlines is not None:
            return self._headlines
        rng = random.Random(_hash_seed(self.seed, "news"))
        total = int(self.days * self.headlines_per_day)
        window = max(300.0, 86400.0 / max(1, self.headlines_per_day))
        out: list[Headline] = []
        for i in range(total):
            ts = self.start_ts + (i + rng.random() * 0.8) * window
            inst = rng.choice(self.instruments)
            template, category, valence, mag = rng.choice(TEMPLATES)
            magnitude = max(0.05, min(1.0, mag * rng.uniform(0.7, 1.3)))
            # `noise_share` of items keep their text but get no effect on the
            # tape — the honest hard case, and the reason a lexicon cannot
            # simply score keywords: a story that reads bullish and does
            # nothing is indistinguishable from a real one until afterwards.
            effective = valence
            if rng.random() < self.noise_share:
                effective = 0
            signed = effective * magnitude * rng.uniform(0.6, 1.4)
            impact = signed * self.impact_scale
            text = template.format(
                name=inst.name or inst.symbol,
                asset="crypto" if inst.market == "crypto" else "equities",
                pct=f"{max(0.4, abs(signed) * 4.5):.1f}",
                region=rng.choice(REGIONS),
                macro=rng.choice(MACROS),
                driver=rng.choice(DRIVERS),
                partner=rng.choice(PARTNERS),
                target=f"${rng.uniform(120, 900):.0f}",
                bn=f"{rng.uniform(1, 40):.1f}",
            )
            item = NewsItem(
                text=text,
                ts=ts,
                symbols=(inst.symbol,),
                source=rng.choice(["Wire", "MarketDesk", "Newswire", "Terminal", "Blog"]),
                news_id=f"demo-{i:05d}",
            )
            out.append(
                Headline(
                    item=item,
                    category=category,
                    valence=effective,
                    magnitude=magnitude,
                    impact=impact,
                    symbol=inst.symbol,
                )
            )
        out.sort(key=lambda h: h.item.ts)
        self._headlines = out
        return out

    def _impulse(self, symbol: str) -> "list[float]":
        """Per-bar drift contribution from active headlines, with decay.

        A headline's push is spread over the following bars and decays with a
        six-hour half-life, so a "priced in by the close" story is priced in by
        the close and a slow story keeps working — which is the structure the
        bot's own horizon question is supposed to notice.
        """
        impulse = [0.0] * (self.n_bars + 1)
        for h in self.headlines():
            if h.symbol != symbol or h.impact == 0:
                continue
            start = int((h.item.ts - self.start_ts) / self.bar_seconds)
            if start < 0 or start >= self.n_bars:
                continue
            hours = CATEGORY_HALF_LIFE_HOURS.get(h.category, 6.0)
            half_life_bars = max(1.0, hours * 3600.0 / self.bar_seconds)
            decay = 2 ** (-1.0 / half_life_bars)
            # `impact` is the headline's *total* effect on the price, so the
            # per-bar drift is scaled by (1 - decay) to make the decayed sum
            # converge on it. Without this the impulse compounds to an
            # absurd multiple of its intended size.
            amp = h.impact * (1.0 - decay)
            for k in range(start, min(self.n_bars, start + int(half_life_bars * 8))):
                impulse[k] += amp
                amp *= decay
        return impulse

    def bars(self, symbol: str) -> list[float]:
        if symbol in self._bars:
            return self._bars[symbol]
        rng = random.Random(_hash_seed(self.seed, "bars", symbol))
        vol = self._base_vol(symbol)
        price = self._base_price(symbol)
        impulse = self._impulse(symbol)
        prices: list[float] = []
        returns: list[float] = []
        volumes: list[float] = []
        for i in range(self.n_bars):
            shock = rng.gauss(0.0, vol)
            # gentle mean reversion keeps a 30-day walk from drifting absurdly
            ret = shock + impulse[i] + (-0.02 * rng.gauss(0.0, 1.0) * vol)
            price = max(0.01, price * (1.0 + ret))
            prices.append(price)
            returns.append(ret)
            volumes.append(abs(rng.gauss(1.0, 0.35)) * (1.0 + 4.0 * abs(impulse[i])))
        self._bars[symbol] = prices
        self._returns[symbol] = returns
        self._volumes[symbol] = volumes
        return prices

    def forward_return(self, symbol: str, ts: float, horizon_seconds: float = 6 * 3600.0) -> float:
        bars = self.bars(symbol)
        i = int((ts - self.start_ts) / self.bar_seconds)
        j = int((ts + horizon_seconds - self.start_ts) / self.bar_seconds)
        if i < 0 or j >= len(bars) or i >= len(bars):
            return 0.0
        return bars[j] / bars[i] - 1.0

    # ── labelling ──────────────────────────────────────────────────────────

    def labelled_headlines(self, horizon_seconds: float = 6 * 3600.0) -> list[Headline]:
        for h in self.headlines():
            if not h.forward_return:
                h.forward_return = self.forward_return(h.symbol, h.item.ts, horizon_seconds)
        return self.headlines()

    def index_at(self, ts: float) -> int:
        return int((ts - self.start_ts) / self.bar_seconds)

    def use_headlines(self, headlines: list[Headline]) -> None:
        """Swap the headline set (used by the falsification control).

        The price path is rebuilt from the new set on next access, so a shuffle
        changes both what the engine reads *and* what the tape did — which is the
        point: if the edge survives the mismatch, it was never a news edge.
        """
        self._headlines = sorted(headlines, key=lambda h: h.item.ts)
        self._bars.clear()
        self._returns.clear()
        self._volumes.clear()


class DemoPriceFeed(PriceFeed):
    name = "synthetic"

    def __init__(self, market: DemoMarket, speed: float = 1.0, replay_from_start: bool = True) -> None:
        self.market = market
        self.speed = max(0.001, float(speed))
        self._t0_wall = time.time()
        self._t0_sim = market.start_ts if replay_from_start else market.end_ts
        self._cache: dict[str, MarketSnapshot] = {}

    # simulated "now"
    def sim_now(self, wall_now: float | None = None) -> float:
        wall_now = wall_now if wall_now is not None else time.time()
        return min(self.market.end_ts, self._t0_sim + (wall_now - self._t0_wall) * self.speed)

    def snapshot(self, symbol: str, now: float | None = None) -> MarketSnapshot | None:
        ts = self.sim_now() if now is None else now
        return self.snapshot_at(symbol, ts)

    def snapshot_at(self, symbol: str, ts: float) -> MarketSnapshot | None:
        bars = self.market.bars(symbol)
        idx = self.market.index_at(ts)
        if idx < 2 or idx >= len(bars):
            idx = min(max(idx, 2), len(bars) - 1)
        price = bars[idx]
        returns = self.market._returns[symbol]
        volumes = self.market._volumes[symbol]
        bar = self.market.bar_seconds

        def ret_over(seconds: float) -> float:
            k = max(1, int(seconds / bar))
            j = max(0, idx - k)
            base = bars[j]
            return price / base - 1.0 if base else 0.0

        window = max(12, int(86400 / bar))
        recent = returns[max(0, idx - window):idx + 1]
        vol = (sum((r - sum(recent) / len(recent)) ** 2 for r in recent) / max(1, len(recent))) ** 0.5
        vol_series = volumes[max(0, idx - window):idx + 1]
        vmean = sum(vol_series) / max(1, len(vol_series))
        vstd = (sum((v - vmean) ** 2 for v in vol_series) / max(1, len(vol_series))) ** 0.5 or 1.0
        day_hi = max(bars[max(0, idx - window):idx + 1])
        day_lo = min(bars[max(0, idx - window):idx + 1])
        rng_span = (day_hi - day_lo) or 1e-9

        fast_bars = max(2, int(7200 / bar))
        slow_bars = max(4, int(86400 / bar))
        ema_fast = sum(bars[max(0, idx - fast_bars):idx + 1]) / len(bars[max(0, idx - fast_bars):idx + 1])
        ema_slow = sum(bars[max(0, idx - slow_bars):idx + 1]) / len(bars[max(0, idx - slow_bars):idx + 1])

        is_crypto = "/" in symbol
        funding = None
        if is_crypto:
            funding = 0.0001 + 0.00035 * max(-1.0, min(1.0, ret_over(86400) * 60))

        return MarketSnapshot(
            symbol=symbol,
            ts=ts,
            last=price,
            ret_1h=ret_over(3600),
            ret_24h=ret_over(86400),
            ret_7d=ret_over(7 * 86400),
            vol_24h=vol,
            volume_z=(volumes[idx] - vmean) / vstd,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            range_pos=(price - day_lo) / rng_span,
            half_spread_bps=1.5 if is_crypto else 1.0,
            funding=funding,
            quote_volume=volumes[idx] * price,
        )

    @property
    def resettable(self) -> bool:
        return True

    def reset(self) -> None:
        self._t0_wall = time.time()

    def info(self) -> dict[str, Any]:
        return {
            "feed": self.name,
            "seed": self.market.seed,
            "days": self.market.days,
            "bar_seconds": self.market.bar_seconds,
            "speed": self.speed,
            "sim_now": round(self.sim_now(), 1),
            "sim_now_utc": datetime.fromtimestamp(self.sim_now(), tz=timezone.utc).isoformat(timespec="seconds"),
            "synthetic": True,
            "warning": "Synthetic market. Useful for plumbing; not evidence of live edge.",
        }


class DemoNewsFeed(NewsFeed):
    name = "demo"

    def __init__(self, market: DemoMarket, price_feed: DemoPriceFeed, max_items: int = 400) -> None:
        self.market = market
        self.price_feed = price_feed
        self.max_items = max_items
        self._cursor = 0
        self._t0_wall = time.time()

    @property
    def resettable(self) -> bool:
        return True

    def reset(self) -> None:
        """Rewind to the first headline, for a looping demo run."""
        self._cursor = 0
        self._t0_wall = time.time()

    def poll(self, now: float | None = None) -> list[NewsItem]:
        ts = self.price_feed.sim_now() if now is None else now
        items: list[NewsItem] = []
        heads = self.market.headlines()
        while self._cursor < len(heads) and heads[self._cursor].item.ts <= ts:
            item = heads[self._cursor].item
            # keep the age honest when a headline is served late
            items.append(item)
            self._cursor += 1
        # No synthetic backlog. An earlier version re-served the last few
        # headlines while the cursor was young "to give the first cycles some
        # context" — which put the same story on the dashboard several times
        # over and woke the engine for news it had already read. Context is the
        # caller's job: the bot keeps its own rolling window of everything it
        # has been served, and that window is what the state renderer shows.
        return items[-self.max_items:]

    def info(self) -> dict[str, Any]:
        return {
            "feed": self.name,
            "served": self._cursor,
            "total": len(self.market.headlines()),
            "synthetic": True,
        }


def shuffle_headlines(headlines: list[Headline], seed: int = 11) -> list[Headline]:
    """Falsification control: keep every text, permute which one hits when.

    If a strategy's measured edge survives this, the edge was not coming from
    the news-to-price relationship, and the harness is lying to you.
    """
    rng = random.Random(seed)
    texts = [h.item.text for h in headlines]
    rng.shuffle(texts)
    out: list[Headline] = []
    for h, text in zip(headlines, texts):
        item = NewsItem(
            text=text,
            ts=h.item.ts,
            symbols=h.item.symbols,
            source=h.item.source,
            news_id=h.item.news_id + "-shuffled",
        )
        out.append(
            Headline(
                item=item,
                category=h.category,
                valence=h.valence,
                magnitude=h.magnitude,
                impact=h.impact,
                symbol=h.symbol,
            )
        )
    return out
