"""Orders that leave the process: does the venue actually fill them?

Step 1 asks for real testnet orders, so the thing to prove is that a submitted
order becomes a signed request to the exchange's matching engine and that the
resulting fill is the *venue's* number — not the local simulator's.

The sandbox cannot reach ``testnet.binance.vision``, so :mod:`tests.local_venue`
stands in. Everything that decides whether the real order would be accepted is
still exercised: ccxt's request construction, HMAC-SHA256 signing with the key
we hold, form encoding, the order-response schema, and error mapping. The venue
verifies the signature on the way in, so a passing test means the client signed
correctly — a signature nobody checks proves nothing.

What these tests cannot prove, and no test in this sandbox can: that the real
venue accepts these keys and returns these numbers. That check has to happen on
a machine with egress, and `jevbot doctor` and `jevbot testnet-order` exist to
perform it.
"""

from __future__ import annotations

import pytest

from jevbot.brokers import build_broker
from jevbot.brokers.base import BrokerError
from jevbot.bot import TradingBot
from jevbot.config import load_config
from jevbot.engine import HeuristicEngine
from jevbot.portfolio import Portfolio
from jevbot.types import MarketSnapshot, Order, Side
from tests.local_venue import LocalVenue


@pytest.fixture
def venue(monkeypatch):
    """A local venue with real ccxt clients (feed and broker) aimed at it."""
    with LocalVenue() as v:
        monkeypatch.setenv("BINANCE_API_KEY", v.api_key)
        monkeypatch.setenv("BINANCE_API_SECRET", v.api_secret)
        v.attach(monkeypatch)
        yield v


def _cfg(**overrides):
    cfg = load_config("config/binance_testnet.toml", dotenv=False)
    cfg.raw["feeds"]["news"] = "jsonl"      # no demo market needed
    cfg.raw["loop"] = {"bar_seconds": 60, "cycle_seconds": 0.01}
    for path, value in overrides.items():
        section, _, key = path.partition(".")
        cfg.raw.setdefault(section, {})[key] = value
    return cfg


def _snapshot(symbol="BTC/USDT", price=30_000.0, ts=1_700_000_000.0):
    return MarketSnapshot(symbol=symbol, ts=ts, last=price)


# ── the order reaches the venue ─────────────────────────────────────────────


def test_the_shipped_profile_routes_orders_to_the_testnet_venue(venue):
    cfg = _cfg()
    assert cfg.broker_kind == "ccxt"
    portfolio = Portfolio.fresh(1_000.0)
    broker = build_broker(cfg, portfolio)
    assert broker.name == "ccxt"
    assert broker.is_live is False, "testnet can never be live"
    assert "127.0.0.1" in broker.info()["endpoint"] or "testnet" in broker.info()["endpoint"]


def test_an_order_is_signed_and_filled_by_the_venue(venue):
    """The whole point: a POST to /order, signed, filled at the venue's price."""
    cfg = _cfg()
    portfolio = Portfolio.fresh(1_000.0)
    broker = build_broker(cfg, portfolio)
    snap = _snapshot()

    fill = broker.submit(Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005), snap, ts=snap.ts)

    # 1. the venue saw exactly one order, over a real socket
    assert [p for p in venue.paths if p.endswith("/order")], venue.paths
    sent = venue.orders[0]
    assert sent["symbol"] == "BTCUSDT", "the venue wants its own symbol format"
    assert sent["side"] == "BUY"
    assert sent["type"] == "MARKET"
    assert float(sent["quantity"]) == pytest.approx(0.0005)

    # 2. and it was signed with the key we supplied
    assert venue.signatures == [True], "the venue rejected the signature"

    # 3. the fill is the venue's number, not a local simulation
    assert fill is not None
    venue_price = 30_000.0 + 1.5 * 599          # last close in the local tape
    assert fill.price == pytest.approx(venue_price * 1.002, rel=1e-6), (
        f"fill price {fill.price} is not the venue's quote for {venue_price}"
    )
    assert fill.qty == pytest.approx(0.0005)
    assert fill.fee > 0, "the venue charged commission and the fill should carry it"
    assert broker.orders_sent == 1 and broker.orders_rejected == 0
    # 4. and it landed in the local book, so the dashboard and risk layer see it
    assert portfolio.positions["BTC/USDT"].qty == pytest.approx(0.0005)


def test_a_wrong_secret_is_refused_by_the_venue(venue, monkeypatch):
    """Proves the signature is real: a bad one is rejected, not ignored."""
    cfg = _cfg()
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    broker.exchange.secret = "not-the-secret"
    with pytest.raises(BrokerError) as exc:
        broker.submit(Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005), _snapshot())
    assert venue.orders == [], "the venue must not accept an unsigned order"
    assert venue.signatures == [False]
    assert broker.orders_rejected == 1
    assert "3075" in str(exc.value) or "signature" in str(exc.value).lower() or "-2015" in str(exc.value)


# ── the account, not our idea of it ─────────────────────────────────────────


def test_the_venue_balance_is_adopted_on_startup(venue):
    """Sizing follows the account that will actually fill the orders."""
    cfg = _cfg()
    cfg.raw["risk"]["starting_equity"] = 100_000.0     # wildly wrong on purpose
    venue.balances["USDT"] = 1_500.0
    broker = build_broker(cfg, Portfolio.fresh(cfg.get("risk", "starting_equity")))
    bot = TradingBot(cfg, engine=HeuristicEngine(), broker=broker,
                     portfolio=broker.portfolio, price_feed=_QuietFeed(), news_feed=_NoNews())
    bot.warmup()

    assert broker.portfolio.cash == pytest.approx(1_500.0)
    assert bot.risk.day_start_equity == pytest.approx(1_500.0), "the anchor must follow"
    assert bot.risk._peak == pytest.approx(1_500.0)  # noqa: SLF001
    # the trap this guards: a $100k peak against a $1.5k account reads as a
    # 98.5% drawdown and trips the kill switch on cycle one
    assert bot.risk.killed is False
    assert any("venue account adopted" in e["message"] for e in bot.events)


def test_an_empty_venue_account_is_reported_not_traded(venue):
    cfg = _cfg()
    venue.balances.clear()
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    bot = TradingBot(cfg, engine=HeuristicEngine(), broker=broker,
                     portfolio=broker.portfolio, price_feed=_QuietFeed(), news_feed=_NoNews())
    bot.warmup()
    assert any("empty" in e["message"] for e in bot.events), bot.events
    assert broker.portfolio.cash == pytest.approx(1_000.0), "the local book is left alone"


# ── safety: rejections stop the bot ─────────────────────────────────────────


def test_rejections_in_a_row_trip_the_breaker(venue):
    cfg = _cfg(**{"risk.max_consecutive_rejections": 3})
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    bot = TradingBot(cfg, engine=HeuristicEngine(), broker=broker, portfolio=broker.portfolio,
                     price_feed=_QuietFeed(), news_feed=_NoNews())
    venue.reject_orders()

    for _ in range(3):
        bot.submit([Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005)], {"BTC/USDT": _snapshot()}, 1.0)

    assert bot.venue_stopped is True
    assert "consecutive venue rejections" in bot.venue_stop_reason
    assert bot.status()["venue_stopped"] is True

    # and no further orders leave the process — not even a rejected one
    before = len(venue.attempts)
    bot.submit([Order(symbol="ETH/USDT", side=Side.BUY, qty=0.01)],
               {"ETH/USDT": _snapshot("ETH/USDT", 1_800.0)}, 2.0)
    assert len(venue.attempts) == before, "the breaker must stop the traffic, not just log it"


def test_the_breaker_is_not_tripped_by_a_success(venue):
    cfg = _cfg(**{"risk.max_consecutive_rejections": 2})
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    bot = TradingBot(cfg, engine=HeuristicEngine(), broker=broker, portfolio=broker.portfolio,
                     price_feed=_QuietFeed(), news_feed=_NoNews())
    venue.reject_orders()
    bot.submit([Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005)], {"BTC/USDT": _snapshot()}, 1.0)
    venue.accept_orders()
    bot.submit([Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005)], {"BTC/USDT": _snapshot()}, 2.0)
    venue.reject_orders()
    bot.submit([Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005)], {"BTC/USDT": _snapshot()}, 3.0)
    assert bot.venue_stopped is False, "a success resets the streak"


def test_the_breaker_is_cleared_by_an_operator_only(venue):
    cfg = _cfg(**{"risk.max_consecutive_rejections": 1})
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    bot = TradingBot(cfg, engine=HeuristicEngine(), broker=broker, portfolio=broker.portfolio,
                     price_feed=_QuietFeed(), news_feed=_NoNews())
    venue.reject_orders()
    bot.submit([Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005)], {"BTC/USDT": _snapshot()}, 1.0)
    assert bot.venue_stopped is True

    venue.accept_orders()
    bot.reset_venue_breaker()
    assert bot.venue_stopped is False
    fill = bot.submit([Order(symbol="BTC/USDT", side=Side.BUY, qty=0.0005)],
                      {"BTC/USDT": _snapshot()}, 2.0)
    assert len(fill) == 1, "after a reset the bot trades again"


def test_a_cycle_while_the_breaker_is_on_does_not_send_orders(venue):
    """The cycle must still record equity, and must not 'flatten' into a venue
    that is refusing orders — that would report a flat book it does not have."""
    cfg = _cfg()
    broker = build_broker(cfg, Portfolio.fresh(1_000.0))
    feed = _QuietFeed()
    bot = TradingBot(cfg, engine=HeuristicEngine(), broker=broker, portfolio=broker.portfolio,
                     price_feed=feed, news_feed=_NoNews())
    bot.warmup()
    bot.venue_stopped = True
    bot.venue_stop_reason = "test"
    before = len(venue.attempts)
    result = bot.cycle()
    assert len(venue.attempts) == before
    assert result.equity > 0, "the equity point still has to be recorded"
    assert any("venue breaker" in n for n in result.notes)


# ── helpers: quiet feeds so a cycle needs no market data ────────────────────


class _QuietFeed:
    """A price feed that answers with one fixed candle and never touches the net."""

    name = "quiet"

    def __init__(self, price: float = 30_000.0):
        self.price = price

    def info(self):
        return {"feed": self.name, "synthetic": False}

    def snapshot(self, symbol, now=None):
        return MarketSnapshot(symbol=symbol, ts=now or 1_700_000_000.0, last=self.price,
                              ema_fast=self.price, ema_slow=self.price, vol_24h=0.01)

    def snapshots(self, symbols, now=None):
        return {s: self.snapshot(s, now) for s in symbols}


class _NoNews:
    name = "none"

    def info(self):
        return {"feed": self.name}

    def poll(self, now=None):
        return []


# ── the smoke-order command ─────────────────────────────────────────────────
# `jevbot testnet-order` is how an operator answers "does my key work, and did
# the order really reach the venue" without starting the whole bot.


def test_the_smoke_order_reaches_the_venue_and_reports_the_order_id(venue, capsys):
    from jevbot.cli import main

    code = main(["testnet-order", "--config", "config/binance_testnet.toml",
                 "--symbol", "BTC/USDT", "--notional", "30", "--yes"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "127.0.0.1" in out or "testnet" in out
    assert "FILLED by the venue" in out, out
    assert "testnet.binance.vision" in out, "it must say where to look for the order"
    assert len(venue.orders) == 1, venue.orders
    assert venue.signatures == [True]
    sent = venue.orders[0]
    assert sent["side"] == "BUY" and sent["symbol"] == "BTCUSDT"


def test_the_smoke_order_refuses_without_yes(venue, capsys):
    from jevbot.cli import main

    code = main(["testnet-order", "--config", "config/binance_testnet.toml"])
    out = capsys.readouterr().out
    assert code == 2
    assert venue.attempts == [], "nothing may reach the venue without --yes"
    assert "--yes" in out


def test_the_smoke_order_refuses_a_mainnet_endpoint(monkeypatch, capsys):
    """The one command that places orders must not be a way to reach real money."""
    from jevbot import cli
    from jevbot.venues import build_ccxt_exchange as real_build

    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    # Even with the live acknowledgement given, the endpoint itself must stop it.
    monkeypatch.setenv("JEVBOT_I_UNDERSTAND_LIVE_RISK", "yes")

    def mainnet(exchange_id, **kwargs):
        exchange = real_build(exchange_id, **kwargs)
        exchange.urls["api"]["public"] = "https://api.binance.com/api/v3"
        exchange.urls["api"]["private"] = "https://api.binance.com/api/v3"
        return exchange

    monkeypatch.setattr("jevbot.venues.build_ccxt_exchange", mainnet)
    monkeypatch.setattr("jevbot.brokers.ccxt_broker.build_ccxt_exchange", mainnet)
    cfg_override = ["--set", "broker.testnet=false", "--set", "feeds.testnet=false",
                    "--set", "mode.live=true"]
    code = cli.main(["testnet-order", "--config", "config/binance_testnet.toml",
                     "--yes", *cfg_override])
    out = capsys.readouterr().out
    assert code == 2, out
    assert "refusing" in out and ("is not a sandbox" in out or "live crypto" in out), out


def test_the_smoke_order_refuses_a_mainnet_endpoint_before_the_ack(monkeypatch, capsys):
    """Without the acknowledgement the broker refuses even to be constructed."""
    from jevbot import cli

    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.delenv("JEVBOT_I_UNDERSTAND_LIVE_RISK", raising=False)
    code = cli.main(["testnet-order", "--config", "config/binance_testnet.toml",
                     "--yes", "--set", "broker.testnet=false", "--set", "feeds.testnet=false"])
    out = capsys.readouterr().out
    assert code == 2, out
    assert "refusing" in out, out


def test_the_smoke_order_respects_the_order_size_cap(venue, capsys):
    from jevbot.cli import main

    code = main(["testnet-order", "--config", "config/binance_testnet.toml",
                 "--notional", "5000", "--yes"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "clamping" in out, out
    qty = float(venue.orders[0]["quantity"])
    assert qty * 30_898.5 <= 50.0 * 1.01, f"{qty} BTC is more than the cap allows"
