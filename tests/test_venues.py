"""Venue wiring: Binance testnet for market data, paper fills, real guards.

The distinction this file exists to protect:

* a **testnet** endpoint cannot move real money, so it must not demand the
  live-trading acknowledgement, must not report itself as live, and must never
  be reachable from a code path that has not been told it is a sandbox;
* a **mainnet** endpoint is the opposite on every count.

Getting that backwards is the expensive direction, so most of these tests assert
the *negative*: that the dangerous thing cannot happen.
"""

from __future__ import annotations

import pytest

from jevbot.brokers import build_broker
from jevbot.brokers.base import LiveTradingRefused
from jevbot.brokers.ccxt_broker import CCXTBroker
from jevbot.feeds import build_news_feed, build_price_feed
from jevbot.feeds.base import FeedError
from jevbot.feeds.live import CCXTPriceFeed, RSSNewsFeed
from jevbot.portfolio import Portfolio
from jevbot.types import Instrument, Order, Side
from jevbot.venues import build_ccxt_exchange, is_testnet_endpoint, rest_endpoint
from tests import fake_ccxt

INSTRUMENTS = [Instrument("BTC/USDT", "crypto", "Bitcoin"),
               Instrument("ETH/USDT", "crypto", "Ethereum")]


@pytest.fixture
def fake(monkeypatch):
    return fake_ccxt.install(monkeypatch)


# ── endpoints ───────────────────────────────────────────────────────────────


def test_testnet_switches_the_public_endpoint(fake):
    feed = CCXTPriceFeed(INSTRUMENTS, testnet=True)
    assert feed.info()["endpoint"] == fake_ccxt.TESTNET
    assert feed.info()["testnet"] is True
    assert feed.info()["synthetic"] is False


def test_mainnet_is_the_default(fake):
    feed = CCXTPriceFeed(INSTRUMENTS)
    assert feed.info()["endpoint"] == fake_ccxt.MAINNET
    assert feed.info()["testnet"] is False


def test_endpoint_reporting_picks_the_spot_url_not_a_futures_one():
    """``urls['api']`` is a dict of API families; the first key is not the spot one.

    Reporting ``dapi`` (coin-margined futures) because it sorted first is how a
    spot testnet run ends up printing a futures URL in its own logs — and then
    someone debugs the wrong endpoint.
    """

    class Exchange:
        urls = {"api": {"dapiPublic": "https://testnet.binancefuture.com/dapi/v1",
                        "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                        "public": "https://testnet.binance.vision/api/v3"}}

    assert rest_endpoint(Exchange()) == "https://testnet.binance.vision/api/v3"


def test_is_testnet_endpoint_recognises_sandboxes():
    assert is_testnet_endpoint(fake_ccxt.TESTNET)
    assert is_testnet_endpoint("https://paper-api.alpaca.markets")
    assert not is_testnet_endpoint(fake_ccxt.MAINNET)


def test_sandbox_mode_is_applied_by_the_shared_builder(fake):
    exchange = build_ccxt_exchange("binance", testnet=True)
    assert exchange.sandbox is True
    assert ("set_sandbox_mode", True) in exchange.calls
    assert not is_testnet_endpoint(rest_endpoint(build_ccxt_exchange("binance")))


def test_an_exchange_without_a_sandbox_is_refused(fake, monkeypatch):
    class NoSandbox(fake_ccxt.FakeExchange):
        set_sandbox_mode = None  # type: ignore[assignment]

    monkeypatch.setattr(fake, "binance", NoSandbox)
    with pytest.raises(Exception, match="sandbox"):
        build_ccxt_exchange("binance", testnet=True)


# ── the broker ──────────────────────────────────────────────────────────────


def test_a_testnet_broker_needs_no_real_money_acknowledgement(fake, monkeypatch):
    monkeypatch.delenv("JEVBOT_I_UNDERSTAND_LIVE_RISK", raising=False)
    broker = CCXTBroker(Portfolio.fresh(100_000.0), testnet=True)
    assert broker.is_live is False
    info = broker.info()
    assert info["testnet"] is True and info["live"] is False
    assert "testnet" in info["endpoint"]


def test_a_mainnet_broker_still_demands_the_acknowledgement(fake, monkeypatch):
    monkeypatch.delenv("JEVBOT_I_UNDERSTAND_LIVE_RISK", raising=False)
    with pytest.raises(LiveTradingRefused, match="real money"):
        CCXTBroker(Portfolio.fresh(100_000.0))
    monkeypatch.setenv("JEVBOT_I_UNDERSTAND_LIVE_RISK", "yes")
    broker = CCXTBroker(Portfolio.fresh(100_000.0))
    assert broker.is_live is True
    assert broker.info()["testnet"] is False


def test_testnet_fills_land_in_the_local_book(fake):
    """Paper fills from testnet prices: the book is ours, the prices are theirs."""
    portfolio = Portfolio.fresh(100_000.0)
    broker = CCXTBroker(portfolio, testnet=True)
    fill = broker.submit(Order(symbol="BTC/USDT", side=Side.BUY, qty=0.1), None)
    assert fill is not None
    assert fill.price == pytest.approx(30_000.0)
    assert fill.fee == pytest.approx(1.5)
    assert portfolio.positions["BTC/USDT"].qty == pytest.approx(0.1)
    assert portfolio.cash == pytest.approx(100_000.0 - 3_000.0 - 1.5)


def test_a_rejected_order_raises_and_leaves_the_book_alone(fake):
    from jevbot.brokers.base import BrokerError

    portfolio = Portfolio.fresh(100_000.0)
    broker = CCXTBroker(portfolio, testnet=True)
    exchange = [o for o in fake._instances][-1]  # noqa: SLF001
    exchange.fail_order = RuntimeError("insufficient balance")
    with pytest.raises(BrokerError, match="insufficient balance"):
        broker.submit(Order(symbol="BTC/USDT", side=Side.BUY, qty=1.0), None)
    assert portfolio.cash == pytest.approx(100_000.0)
    assert broker.orders_rejected == 1


def test_a_spot_broker_does_not_short(fake):
    broker = CCXTBroker(Portfolio.fresh(1_000.0), testnet=True)
    assert broker.can_short() is False


# ── config drives all of it ─────────────────────────────────────────────────


def test_config_selects_testnet_for_both_the_feed_and_the_broker(fake):
    from jevbot.config import load_config

    cfg = load_config("config/binance_testnet.toml")
    assert cfg.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    feed = build_price_feed(cfg, speed=1.0)
    assert feed.info()["testnet"] is True
    assert feed.info()["endpoint"] == fake_ccxt.TESTNET

    # the shipped profile routes orders nowhere: paper fills, testnet prices
    paper = build_broker(cfg, Portfolio.fresh(1_000.0))
    assert paper.name == "paper" and paper.is_live is False

    cfg.raw["broker"]["kind"] = "ccxt"
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    assert broker.info()["testnet"] is True
    assert broker.is_live is False
    assert broker.info()["endpoint"] == fake_ccxt.TESTNET


def test_a_testnet_run_can_never_be_live(fake):
    """Belt and braces: ``mode.live = true`` plus testnet must still be paper."""
    from jevbot.config import load_config

    cfg = load_config()
    cfg.raw["broker"]["kind"] = "ccxt"
    cfg.raw["broker"]["testnet"] = True
    cfg.raw["mode"] = {"live": True}
    assert cfg.live is False
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    assert broker.is_live is False


def test_force_paper_keeps_order_routing_local(fake):
    from jevbot.config import load_config

    cfg = load_config("config/binance_testnet.toml")
    cfg.raw["broker"]["kind"] = "ccxt"
    broker = build_broker(cfg, Portfolio.fresh(1_000.0), force_paper=True)
    assert broker.name == "paper"
    assert broker.is_live is False


def test_the_feed_follows_the_broker_when_the_feed_does_not_say(fake):
    """A testnet broker with an unset feed setting must not read mainnet prices."""
    from jevbot.config import load_config

    cfg = load_config()
    cfg.raw["broker"]["testnet"] = True
    cfg.raw["feeds"] = {"price": "ccxt", "news": "demo"}     # feeds.testnet unset
    feed = build_price_feed(cfg, speed=1.0)
    assert feed.info()["testnet"] is True


# ── failure modes are loud and specific ─────────────────────────────────────


def test_an_unreachable_venue_names_the_endpoint(fake, monkeypatch):
    def refused(*_a, **_kw):
        raise RuntimeError("Connection refused")

    monkeypatch.setattr(fake_ccxt.FakeExchange, "load_markets", refused)
    feed = build_price_feed(_ccxt_config(), speed=1.0)
    with pytest.raises(FeedError) as exc:
        feed.prime()
    message = str(exc.value)
    assert fake_ccxt.TESTNET in message
    assert "testnet" in message


def _ccxt_config():
    from jevbot.config import load_config

    cfg = load_config("config/binance_testnet.toml")
    return cfg


def test_a_pair_the_venue_does_not_list_is_named_with_alternatives(fake):
    cfg = _ccxt_config()
    cfg.raw["universe"]["instruments"] = [
        {"symbol": "DOGE/USDT", "market": "crypto", "name": "Dogecoin"}]
    feed = build_price_feed(cfg, speed=1.0)
    with pytest.raises(FeedError, match="does not list DOGE/USDT"):
        feed.prime()


def test_a_feed_that_stops_answering_says_so_rather_than_freezing(fake):
    """A silent freeze is worse than an error: the bot keeps trading the old price."""
    feed = CCXTPriceFeed(INSTRUMENTS, testnet=True)
    feed._ensure_markets()  # noqa: SLF001
    exchange = feed.exchange
    exchange.fail_ohlcv = RuntimeError("502 Bad Gateway")
    with pytest.raises(FeedError, match="502 Bad Gateway"):
        feed.snapshot("BTC/USDT")


def test_the_ticker_cross_check_uses_the_venues_own_price(fake):
    feed = CCXTPriceFeed(INSTRUMENTS, testnet=True)
    assert feed.ticker("BTC/USDT") == pytest.approx(31_000.0)
    assert ("fetch_ticker", "BTC/USDT") in feed.exchange.calls
    feed.exchange.fail_ticker = RuntimeError("timeout")
    with pytest.raises(FeedError, match="ticker"):
        feed.ticker("BTC/USDT")


def test_a_snapshot_carries_a_real_timestamp(fake):
    """Staleness checks downstream are only meaningful if ts is the bar's own."""
    feed = CCXTPriceFeed(INSTRUMENTS, testnet=True, bar_seconds=60)
    snap = feed.snapshot("BTC/USDT")
    assert snap is not None
    assert snap.ts > 1_600_000_000          # a real epoch, not 0 or "now-ish"
    assert snap.last > 0
    assert 0.0 <= snap.range_pos <= 1.0


# ── news ────────────────────────────────────────────────────────────────────


def test_rss_urls_come_from_config_when_the_environment_is_silent(fake):
    from jevbot.config import load_config

    cfg = load_config()
    cfg.raw["feeds"] = {"news": "rss", "rss_urls": ["https://example.test/feed.xml"]}
    feed = build_news_feed(cfg, price_feed=None)
    assert isinstance(feed, RSSNewsFeed)
    assert feed.info()["urls"] == ["https://example.test/feed.xml"]


def test_rss_requires_at_least_one_url(fake, monkeypatch):
    from jevbot.config import load_config

    cfg = load_config()
    cfg.raw["feeds"] = {"news": "rss", "rss_urls": []}
    monkeypatch.delenv("JEVBOT_RSS_URLS", raising=False)
    with pytest.raises(FeedError, match="no RSS urls"):
        build_news_feed(cfg, price_feed=None)


def test_a_dead_rss_source_does_not_take_the_bot_down(fake):
    """News is a nicety; a news outage must not stop a position being closed."""
    feed = RSSNewsFeed(["https://unreachable.invalid/feed.xml"], timeout=0.5)
    assert feed.poll() == []
