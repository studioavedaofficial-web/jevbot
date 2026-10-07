"""Routing: which instrument is this headline even about?

Routing is upstream of the engine and is deliberately dumb and explainable.
A model asked "which of these five assets is this about" is the wrong tool:
the failure mode is quiet and it happens on the one headline that mattered.
"""

from __future__ import annotations

import pytest

from jevbot.routing import NewsRouter
from jevbot.types import Instrument, NewsItem

INSTRUMENTS = [
    Instrument("BTC/USDT", "crypto", "Bitcoin"),
    Instrument("ETH/USDT", "crypto", "Ethereum"),
    Instrument("AAPL", "equity", "Apple"),
    Instrument("SPY", "equity", "S&P 500 ETF"),
]


@pytest.fixture
def router():
    return NewsRouter(INSTRUMENTS, mode="keywords")


def route(router, text):
    return router.route(NewsItem(text=text, source="test"))


def test_a_named_company_maps_to_its_symbol(router):
    symbols, reason = route(router, "Apple beats earnings as iPhone sales jump")
    assert symbols == ("AAPL",)
    assert "keyword" in reason


def test_aliases_are_case_insensitive_and_word_bounded(router):
    assert route(router, "BITCOIN ETF inflows hit a record")[0] == ("BTC/USDT",)
    # "btc" inside another word must not count
    assert route(router, "the btcd protocol launched")[0] != ("BTC/USDT",)


def test_a_headline_can_be_about_two_instruments(router):
    symbols, _ = route(router, "Bitcoin and Ethereum fall together after the Fed decision")
    assert "BTC/USDT" in symbols and "ETH/USDT" in symbols


def test_market_wide_language_goes_to_everything_or_nowhere():
    all_in = NewsRouter(INSTRUMENTS, mode="keywords", market_wide_policy="all")
    symbols, reason = route(all_in, "Global markets sell off on rate fears")
    assert len(symbols) == len(INSTRUMENTS)
    assert "market-wide" in reason

    none = NewsRouter(INSTRUMENTS, mode="keywords", market_wide_policy="none")
    assert route(none, "Global markets sell off on rate fears")[0] == ()


def test_an_unmatched_headline_routes_to_nothing(router):
    symbols, reason = route(router, "Local bakery wins a regional award")
    assert symbols == ()
    assert "no instrument" in reason


def test_explicitly_tagged_news_keeps_its_tags(router):
    item = NewsItem(text="anything at all", symbols=("NVDA",), source="feed")
    assert router.route(item) == (("NVDA",), "tagged by the feed")


def test_routing_is_deterministic(router):
    texts = ["Apple ships a new chip", "Ethereum upgrade ships", "Fed holds rates steady"]
    first = [route(router, t) for t in texts]
    second = [route(NewsRouter(INSTRUMENTS, mode="keywords"), t) for t in texts]
    assert first == second


def test_engine_routing_is_used_only_when_keywords_find_nothing():
    class FakeEngine:
        name = "fake"
        available = True

        def __init__(self):
            self.calls = 0

        def subject_route(self, text, symbols):
            self.calls += 1
            return (symbols[-1],), "engine picked it"

    engine = FakeEngine()
    router = NewsRouter(INSTRUMENTS, mode="hybrid", engine=engine)
    assert route(router, "Apple ships a new chip")[0] == ("AAPL",)
    assert engine.calls == 0, "keywords matched; the engine should not have been consulted"
    assert route(router, "A cryptic announcement baffles analysts")[0] == ("SPY",)
    assert engine.calls == 1
