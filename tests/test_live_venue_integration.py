"""End-to-end test of the live path, against a local Binance-shaped venue.

This sandbox cannot reach ``testnet.binance.vision``, so the venue is replaced
rather than the client: a real ``ccxt.binance`` instance is constructed, put
into sandbox mode exactly as production does, and only then pointed at a local
HTTP server that answers the same endpoints with the same JSON shape.

That distinction is the whole point. Mocking ccxt at the Python level (as
``tests/fake_ccxt.py`` does) proves the bot calls the right methods; it cannot
catch a wrong request path, a mis-parsed ``exchangeInfo``, a URL that sandbox
mode rewrote somewhere unexpected, or a candle whose timestamp is in the wrong
unit. Those all fail here, over a real socket, through ccxt's real parsing code.

What this still cannot tell you: whether Binance testnet accepts these keys and
returns these numbers. Nothing run in this sandbox can.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevbot.config import load_config
from jevbot.feeds import build_price_feed
from jevbot.portfolio import Portfolio
from jevbot.types import Instrument

# Deterministic tape: a gentle up-trend so the trend features are non-zero.
START_MS = 1_700_000_000_000  # 2023-11-14, arbitrary but fixed
BAR_MS = 60_000
N_BARS = 600

_PRICES = {
    "BTCUSDT": 30_000.0,
    "ETHUSDT": 1_800.0,
}
_DRIFT = {"BTCUSDT": 1.5, "ETHUSDT": 0.08}  # per-bar close increase


def _closes(symbol: str) -> list[float]:
    base, drift = _PRICES[symbol], _DRIFT[symbol]
    return [base + drift * i for i in range(N_BARS)]


def _klines(symbol: str, limit: int) -> list[list]:
    out = []
    for i, close in enumerate(_closes(symbol)[-limit:]):
        idx = N_BARS - min(limit, N_BARS) + i
        ts = START_MS + idx * BAR_MS
        out.append([ts, close - 0.4, close + 0.6, close - 0.9, close, 7.5 + (idx % 3),
                    ts + BAR_MS - 1, 100.0, 12, 4.0, 40.0, "0"])
    return out


class _Venue(BaseHTTPRequestHandler):
    """The subset of the Binance spot REST API that jevbot touches."""

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server's naming
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        self.server.seen.append((path, params))

        if path.endswith("/exchangeInfo"):
            return self._send({"timezone": "UTC", "serverTime": START_MS + N_BARS * BAR_MS,
                               "rateLimits": [], "exchangeFilters": [],
                               "symbols": [self._symbol(s) for s in _PRICES]})
        if path.endswith("/klines"):
            symbol = params.get("symbol", "")
            if symbol not in _PRICES:
                return self._send({"code": -1121, "msg": "Invalid symbol."}, 400)
            return self._send(_klines(symbol, int(params.get("limit", 500))))
        if path.endswith("/ticker/24hr") or path.endswith("/ticker/price"):
            symbol = params.get("symbol", "")
            price = _closes(symbol)[-1]
            if path.endswith("/ticker/price"):
                return self._send({"symbol": symbol, "price": f"{price:.2f}"})
            return self._send({"symbol": symbol, "lastPrice": f"{price:.2f}",
                               "bidPrice": f"{price - 0.5:.2f}", "askPrice": f"{price + 0.5:.2f}",
                               "closeTime": START_MS + N_BARS * BAR_MS})
        return self._send({"code": -1121, "msg": f"unhandled {path}"}, 404)

    @staticmethod
    def _symbol(symbol: str) -> dict:
        base = symbol[:-4]
        return {
            "symbol": symbol, "status": "TRADING", "baseAsset": base, "quoteAsset": "USDT",
            "baseAssetPrecision": 8, "quoteAssetPrecision": 8, "quotePrecision": 8,
            "baseCommissionPrecision": 8, "quoteCommissionPrecision": 8,
            "orderTypes": ["LIMIT", "MARKET"], "icebergAllowed": True, "ocoAllowed": True,
            "quoteOrderQtyMarketAllowed": True, "isSpotTradingAllowed": True,
            "isMarginTradingAllowed": False, "permissions": ["SPOT"],
            "filters": [
                {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000",
                 "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000",
                 "stepSize": "0.00001"},
                {"filterType": "NOTIONAL", "minNotional": "5", "applyMinToMarket": True},
            ],
        }


@pytest.fixture
def venue(monkeypatch):
    """A local venue plus a real ccxt client aimed at it."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Venue)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/v3"

    from jevbot import venues

    real_build = venues.build_ccxt_exchange

    def build(exchange_id, **kwargs):
        exchange = real_build(exchange_id, **kwargs)
        # Everything above this line is production: sandbox mode applied by the
        # shared builder, real headers, real parsing. Only the host is swapped.
        exchange.urls["api"]["public"] = base
        exchange.urls["api"]["private"] = base
        return exchange

    monkeypatch.setattr("jevbot.feeds.live.build_ccxt_exchange", build)
    monkeypatch.setattr(venues, "build_ccxt_exchange", build)
    yield server
    server.shutdown()
    server.server_close()


def _feed(*, history_bars: int = 300, testnet: bool = True):
    """A real feed, built directly so the test controls the history window.

    `build_price_feed` primes with production defaults; building the feed here
    and priming exactly once avoids re-polling the same candles with a different
    limit, which the feature engine rightly rejects as out-of-order.
    """
    from jevbot.feeds.live import CCXTPriceFeed

    return CCXTPriceFeed(
        [Instrument("BTC/USDT", "crypto", "Bitcoin"), Instrument("ETH/USDT", "crypto", "Ethereum")],
        exchange_id="binance", bar_seconds=60, history_bars=history_bars, funding=False,
        testnet=testnet,
    )


def _crypto_cfg(tmp_path, *, testnet=True, symbols=("BTC/USDT", "ETH/USDT")):
    cfg = load_config(dotenv=False)
    cfg.raw["feeds"] = {
        "price": "ccxt", "news": "demo", "ccxt_exchange": "binance",
        "testnet": testnet, "rss_urls": [],
    }
    cfg.raw["universe"]["instruments"] = [
        {"symbol": s, "market": "crypto", "name": s.split("/")[0]} for s in symbols
    ]
    cfg.raw["loop"] = {"bar_seconds": 60}
    cfg.raw["feeds"]["funding"] = False
    return cfg


def test_real_ccxt_reads_the_venue_through_a_socket(venue):
    """Prime a real ccxt client and check the numbers are the venue's own."""
    feed = _feed()
    feed.prime()

    paths = {p for p, _ in venue.seen}
    assert any(p.endswith("/exchangeInfo") for p in paths), "markets were never loaded"
    assert any(p.endswith("/klines") for p in paths), "candles were never fetched"

    snap = feed.snapshot("BTC/USDT")
    expected = _closes("BTCUSDT")[-1]
    assert snap is not None
    assert snap.last == pytest.approx(expected, rel=1e-6), (
        f"feed says {snap.last}, venue says {expected} — the candle was misparsed"
    )
    # The bar timestamp must be a real epoch second, not milliseconds or a counter.
    assert 1_600_000_000 < snap.ts < 1_800_000_000, f"suspicious bar ts: {snap.ts}"

    info = feed.info()
    assert info["synthetic"] is False
    assert info["testnet"] is True
    assert info["exchange"] == "binance"
    assert "127.0.0.1" in info["endpoint"], info["endpoint"]


def test_the_venues_ticker_is_reachable_independently_of_the_candles(venue):
    """`ticker()` is the cross-check `doctor` relies on; it must be a real fetch."""
    feed = _feed(history_bars=120)
    price = feed.ticker("ETH/USDT")
    assert price == pytest.approx(_closes("ETHUSDT")[-1], rel=1e-9)
    assert any("/ticker" in p for p, _ in venue.seen), venue.seen


def test_a_paper_fill_happens_at_the_price_the_venue_quoted(venue):
    """The full path: venue candles → snapshot → order → fill, over a socket."""
    from jevbot.brokers import build_broker

    cfg = _crypto_cfg(None)
    feed = _feed()
    feed.prime()
    snap = feed.snapshot("BTC/USDT")

    broker = build_broker(cfg, Portfolio.fresh(100_000.0))
    assert broker.name == "paper", "the shipped testnet profile must not route orders anywhere"
    from jevbot.types import Order

    order = Order(symbol="BTC/USDT", side="buy", qty=0.1, reason="test", ts=snap.ts)
    fill = broker.submit(order, snap, ts=snap.ts)
    assert fill is not None and fill.qty == pytest.approx(0.1)
    # Slippage and half-spread are charged against us, never for us: a buy fills
    # at or above the venue price.
    assert fill.price >= snap.last
    assert fill.price <= snap.last * 1.01, f"implausible fill {fill.price} vs {snap.last}"
    assert broker.is_live is False


def test_a_bot_cycle_on_real_candles_records_decisions_and_orders(venue, tmp_path):
    """One full cycle through the real feed, into the store the dashboard reads."""
    from jevbot.bot import TradingBot
    from jevbot.engine import HeuristicEngine
    from jevbot.feeds.jsonl import JsonlNewsFeed
    from jevbot.store import Store

    cfg = _crypto_cfg(tmp_path)
    cfg.raw["engine"]["name"] = "heuristic"
    cfg.raw["loop"] = {"bar_seconds": 60, "cycle_seconds": 0.01}
    feed = _feed()
    feed.prime()

    # Real headlines are what the engine will read in production; the jsonl feed
    # is the offline stand-in that exercises the same parsing and clock.
    rows = [
        # Both already in the past, so they are served on the first poll rather
        # than waiting out real wall-clock time.
        {"ts": time.time() - 120, "text": "Bitcoin ETF inflows hit a record as BTC rallies",
         "symbols": ["BTC/USDT"], "source": "test", "news_id": "n1"},
        {"ts": time.time() - 120, "text": "Ethereum upgrade ships; ETH developers cheer",
         "symbols": ["ETH/USDT"], "source": "test", "news_id": "n2"},
    ]
    news_path = tmp_path / "news.jsonl"
    news_path.write_text("\n".join(json.dumps(r) for r in rows))
    store = Store(tmp_path / "live.db")
    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=feed,
                     news_feed=JsonlNewsFeed(news_path), store=store)
    bot.warmup()
    results = bot.run(cycles=4, interval=0.05)
    counts = store.counts()
    fills = store.recent_fills(10)
    store.close()

    assert len(results) == 4, f"expected 4 cycles, got {len(results)}"
    assert bot.price_feed.info()["synthetic"] is False
    # Whatever it decided, it was reading the venue's prices, not a generator's.
    assert any(p.endswith("/klines") for p, _ in venue.seen)
    assert counts["news"] >= 2, counts
    assert counts["decisions"] >= 2, counts
    # Anything it traded filled against the venue's own quote, and the row is in
    # the store the dashboard reads.
    for fill in fills:
        assert 25_000 < float(fill["price"]) < 35_000, fill


def test_the_shipped_testnet_profile_asks_for_the_sandbox_venue(monkeypatch):
    """No network: assert the profile itself selects testnet pricing."""
    from jevbot import venues

    captured = {}
    real_build = venues.build_ccxt_exchange

    def build(exchange_id, **kwargs):
        captured.update({"exchange_id": exchange_id, **kwargs})
        return real_build(exchange_id, **kwargs)

    from jevbot.feeds.live import CCXTPriceFeed

    monkeypatch.setattr("jevbot.feeds.live.build_ccxt_exchange", build)
    # Constructing the feed must not touch the network (build_price_feed primes
    # it, so priming is stubbed): this test is about *configuration*.
    monkeypatch.setattr(CCXTPriceFeed, "prime", lambda self, ts=None: None)
    cfg = load_config("config/binance_testnet.toml", dotenv=False)
    feed = build_price_feed(cfg)
    assert captured.get("testnet") is True, "the profile must ask for testnet"
    assert feed.info()["testnet"] is True
    assert "testnet.binance.vision" in feed.info()["endpoint"], feed.info()["endpoint"]
    assert cfg.live is False


# ── a venue that is down must not stop the bot from being built ──────────────
# The dashboard is the thing that explains an outage, so it has to survive one.
# Priming therefore belongs to warmup(), not to the feed builder.


def test_a_dead_venue_still_lets_a_bot_be_constructed(venue, monkeypatch, tmp_path):
    from jevbot.bot import TradingBot
    from jevbot.feeds.base import FeedError
    from jevbot.feeds.jsonl import JsonlNewsFeed
    from jevbot.engine import HeuristicEngine

    cfg = _crypto_cfg(None)
    news_path = tmp_path / "news.jsonl"
    news_path.write_text(json.dumps({"ts": time.time() - 60, "text": "BTC steady",
                                     "symbols": ["BTC/USDT"], "news_id": "n1"}))
    bot = TradingBot(cfg, engine=HeuristicEngine(), price_feed=_feed(),
                     news_feed=JsonlNewsFeed(news_path))  # must not raise
    assert bot.price_feed.info()["synthetic"] is False

    monkeypatch.setattr("jevbot.feeds.live.CCXTPriceFeed._ensure_markets",
                        lambda self: (_ for _ in ()).throw(
                            FeedError("could not fetch markets from binance testnet "
                                      "(https://testnet.binance.vision/api/v3): refused")))
    with pytest.raises(FeedError) as exc:
        bot.warmup()
    assert "testnet.binance.vision" in str(exc.value)
