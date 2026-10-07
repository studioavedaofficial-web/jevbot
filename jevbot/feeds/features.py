"""Turning a price series into the numeric snapshot the state renderer reads.

One implementation for every source: the synthetic market, a CSV file, ccxt
candles or Alpaca bars all funnel through :class:`RollingFeatures`, so a signal
computed live and the same signal computed in a replay are computed from
identically defined inputs. Indicator definitions live here and nowhere else —
if the definition of "trend" moved between the backtest and the bot, every
comparison between them would be meaningless.

Definitions, stated once so they cannot drift:

``ret_1h/24h/7d``   simple return over the trailing window
``vol_24h``         standard deviation of per-bar returns over the trailing 24h
``volume_z``        z-score of the latest bar's volume against that window
``ema_fast/slow``   exponential moving average over trailing 2h / 24h
``range_pos``       (last - low) / (high - low) over the trailing 24h
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..types import MarketSnapshot


@dataclass
class _SymbolState:
    prices: deque = field(default_factory=lambda: deque(maxlen=4096))
    stamps: deque = field(default_factory=lambda: deque(maxlen=4096))
    returns: deque = field(default_factory=lambda: deque(maxlen=4096))
    volumes: deque = field(default_factory=lambda: deque(maxlen=4096))
    ema_fast: float | None = None
    ema_slow: float | None = None
    last_ts: float = 0.0
    last_price: float = 0.0


class RollingFeatures:
    """Incremental feature state per symbol.

    Bars must be pushed oldest first. Pushing an out-of-order bar is a bug in the
    feed, not something to paper over, so it raises.
    """

    def __init__(
        self,
        bar_seconds: int = 300,
        fast_seconds: int = 2 * 3600,
        slow_seconds: int = 24 * 3600,
        vol_seconds: int = 24 * 3600,
        max_bars: int = 4096,
    ) -> None:
        self.bar_seconds = int(bar_seconds)
        self.fast_seconds = int(fast_seconds)
        self.slow_seconds = int(slow_seconds)
        self.vol_seconds = int(vol_seconds)
        self.max_bars = max_bars
        self._state: dict[str, _SymbolState] = {}

    # ── ingestion ──────────────────────────────────────────────────────────

    def update(self, symbol: str, ts: float, price: float, volume: float = 0.0) -> None:
        if price <= 0:
            return
        st = self._state.setdefault(symbol, _SymbolState(
            prices=deque(maxlen=self.max_bars),
            stamps=deque(maxlen=self.max_bars),
            returns=deque(maxlen=self.max_bars),
            volumes=deque(maxlen=self.max_bars),
        ))
        if st.stamps and ts < st.stamps[-1]:
            raise ValueError(f"out-of-order bar for {symbol}: {ts} < {st.stamps[-1]}")
        if st.stamps and ts == st.stamps[-1]:
            # Same bar seen again: overwrite rather than double count. The live
            # case is the *current* bar, which is re-polled every few seconds and
            # is still moving, so the bar's return has to be recomputed against
            # the previous bar — not against its own earlier value, which would
            # score every re-poll as a zero-return bar and quietly deflate the
            # volatility the sizer depends on.
            previous = st.prices[-2] if len(st.prices) > 1 else 0.0
            delta = price - st.prices[-1]
            st.prices[-1] = price
            st.volumes[-1] = volume
            if st.returns:
                st.returns[-1] = math.log(price / previous) if previous > 0 else 0.0
            # The EMAs are linear in each sample, so moving one sample by delta
            # moves them by (weight x delta) exactly.
            a_fast = 2.0 / (max(1.0, self.fast_seconds / self.bar_seconds) + 1.0)
            a_slow = 2.0 / (max(1.0, self.slow_seconds / self.bar_seconds) + 1.0)
            if st.ema_fast is not None:
                st.ema_fast += a_fast * delta
            if st.ema_slow is not None:
                st.ema_slow += a_slow * delta
            st.last_price = price
            return
        ret = math.log(price / st.prices[-1]) if st.prices else 0.0
        st.prices.append(price)
        st.stamps.append(ts)
        st.returns.append(ret)
        st.volumes.append(volume)
        a_fast = 2.0 / (max(1.0, self.fast_seconds / self.bar_seconds) + 1.0)
        a_slow = 2.0 / (max(1.0, self.slow_seconds / self.bar_seconds) + 1.0)
        st.ema_fast = price if st.ema_fast is None else st.ema_fast + a_fast * (price - st.ema_fast)
        st.ema_slow = price if st.ema_slow is None else st.ema_slow + a_slow * (price - st.ema_slow)
        st.last_ts = ts
        st.last_price = price

    def update_many(self, symbol: str, bars: list[tuple[float, float, float]]) -> None:
        for ts, price, volume in bars:
            self.update(symbol, ts, price, volume)

    def has(self, symbol: str) -> bool:
        st = self._state.get(symbol)
        return bool(st and len(st.prices) >= 3)

    # ── output ─────────────────────────────────────────────────────────────

    def snapshot(
        self,
        symbol: str,
        half_spread_bps: float | None = None,
        funding: float | None = None,
    ) -> MarketSnapshot | None:
        st = self._state.get(symbol)
        if not st or len(st.prices) < 3:
            return None
        prices = st.prices
        window = max(3, int(self.vol_seconds / self.bar_seconds))
        recent = list(st.returns)[-window:]
        mean = sum(recent) / len(recent)
        var = sum((r - mean) ** 2 for r in recent) / max(1, len(recent) - 1)
        vol = math.sqrt(max(0.0, var))

        vols = list(st.volumes)[-window:]
        vmean = sum(vols) / max(1, len(vols))
        vvar = sum((v - vmean) ** 2 for v in vols) / max(1, len(vols) - 1)
        vstd = math.sqrt(max(0.0, vvar)) or 1.0

        def ret_over(seconds: float) -> float:
            k = max(1, int(seconds / self.bar_seconds))
            idx = max(0, len(prices) - 1 - k)
            base = prices[idx]
            return prices[-1] / base - 1.0 if base else 0.0

        window_prices = list(prices)[-window:]
        hi, lo = max(window_prices), min(window_prices)
        span = (hi - lo) or 1e-12
        is_crypto = "/" in symbol
        return MarketSnapshot(
            symbol=symbol,
            ts=st.last_ts,
            last=st.last_price,
            ret_1h=ret_over(3600),
            ret_24h=ret_over(86400),
            ret_7d=ret_over(7 * 86400),
            vol_24h=vol,
            volume_z=(vols[-1] - vmean) / vstd if vols else 0.0,
            ema_fast=st.ema_fast or st.last_price,
            ema_slow=st.ema_slow or st.last_price,
            range_pos=max(0.0, min(1.0, (st.last_price - lo) / span)),
            half_spread_bps=float(
                half_spread_bps if half_spread_bps is not None else (1.5 if is_crypto else 1.0)
            ),
            funding=funding,
            quote_volume=st.last_price * (vols[-1] if vols else 0.0),
        )
