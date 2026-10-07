"""Feeds: reproducible markets, shared indicators, and honest replay.

The demo market is the measuring instrument. If it is not reproducible, not
seeded, or leaks the future into a snapshot, every number measured on it is
noise — and it is exactly the kind of noise that looks like skill.
"""

from __future__ import annotations

import json

import pytest

from jevbot.feeds import build_news_feed, build_price_feed
from jevbot.feeds.demo import DemoMarket, DemoNewsFeed, DemoPriceFeed, shuffle_headlines
from jevbot.feeds.features import RollingFeatures


def test_the_same_seed_generates_the_same_tape(cfg):
    a = DemoMarket(cfg.instruments, seed=5, days=2, bar_seconds=300, headlines_per_day=20)
    b = DemoMarket(cfg.instruments, seed=5, days=2, bar_seconds=300, headlines_per_day=20)
    assert a.bars("BTC/USDT") == b.bars("BTC/USDT")
    assert [h.item.text for h in a.headlines()] == [h.item.text for h in b.headlines()]

    c = DemoMarket(cfg.instruments, seed=6, days=2, bar_seconds=300, headlines_per_day=20)
    assert a.bars("BTC/USDT") != c.bars("BTC/USDT")


def test_rolling_features_agree_with_the_bars_they_were_fed():
    f = RollingFeatures(bar_seconds=60)
    for i in range(200):
        f.update("X", 1_000.0 + 60 * i, 100.0 + i * 0.1)
    snap = f.snapshot("X")
    assert snap is not None
    assert snap.last == pytest.approx(100.0 + 199 * 0.1)
    assert snap.ret_24h != 0.0
    assert 0.0 <= snap.range_pos <= 1.0
    assert snap.ts == pytest.approx(1_000.0 + 60 * 199)


def test_rolling_features_are_bounded_in_memory():
    """A live feed runs for months; the feature state must not grow with it."""
    f = RollingFeatures(bar_seconds=60, max_bars=64)
    for i in range(500):
        f.update("X", 1_000.0 + 60 * i, 100.0)
    assert len(f._state["X"].prices) <= 64  # noqa: SLF001 - the bound is the point


def test_out_of_order_bars_are_an_error_not_a_silent_rewrite():
    f = RollingFeatures(bar_seconds=60)
    f.update("X", 1_000.0, 100.0)
    with pytest.raises(ValueError):
        f.update("X", 940.0, 99.0)


def test_a_revised_bar_recomputes_its_return_instead_of_zeroing_it():
    """The live case: the current bar is re-polled while it is still moving.

    If the overwrite path scored the bar against its own previous value, every
    re-poll would record a zero-return sample — and the volatility the sizer
    reads would collapse toward zero as the cycle rate rose.
    """
    import math

    f = RollingFeatures(bar_seconds=60)
    for i in range(4):
        f.update("X", 1_000.0 + 60 * i, 100.0 + i)
    f.update("X", 1_180.0, 104.0)          # the same bar, refined
    f.update("X", 1_180.0, 104.0)          # and again, unchanged
    assert len(f._state["X"].prices) == 4  # noqa: SLF001
    assert list(f._state["X"].returns)[-1] == pytest.approx(math.log(104.0 / 102.0))  # noqa: SLF001

    f.update("X", 1_180.0, 105.0)          # a genuine revision upward
    assert f.snapshot("X").last == pytest.approx(105.0)
    assert list(f._state["X"].returns)[-1] == pytest.approx(math.log(105.0 / 102.0))  # noqa: SLF001


def test_demo_price_feed_never_shows_the_future(cfg):
    market = DemoMarket(cfg.instruments, seed=4, days=2, bar_seconds=300, headlines_per_day=10)
    feed = DemoPriceFeed(market, replay_from_start=True)
    ts = market.start_ts + 3600.0
    snap = feed.snapshot_at("BTC/USDT", ts)
    assert snap is not None and snap.ts == pytest.approx(ts)
    later = feed.snapshot_at("BTC/USDT", ts + 1800.0)
    assert later.ts > snap.ts
    # a snapshot taken 30 minutes later may differ, but never in the past
    first_again = feed.snapshot_at("BTC/USDT", ts)
    assert first_again.ts == pytest.approx(ts)


def test_news_feed_only_serves_headlines_that_have_already_happened(cfg):
    market = DemoMarket(cfg.instruments, seed=9, days=2, bar_seconds=300, headlines_per_day=30)
    price = DemoPriceFeed(market, replay_from_start=True)
    news = DemoNewsFeed(market, price)
    cutoff = market.start_ts + 86400.0  # a day into the tape
    items = news.poll(cutoff)
    assert items, "a day of headlines should produce some items"
    assert all(i.ts <= cutoff + 1.0 for i in items)
    assert all(i.text for i in items)
    # and asking again does not serve the same headline twice
    assert news.poll(cutoff) == []


def test_each_headline_is_served_exactly_once(cfg):
    """A feed that repeats itself is a feed nobody can read.

    An earlier version re-served the last few headlines while the cursor was
    young, which put the same story on the dashboard several times over and
    woke the engine for news it had already answered.
    """
    market = DemoMarket(cfg.instruments, seed=5, days=2, bar_seconds=300, headlines_per_day=30)
    price = DemoPriceFeed(market, replay_from_start=True)
    news = DemoNewsFeed(market, price)

    served: list[str] = []
    for i in range(60):                      # two hours of cycles
        now = market.start_ts + 120.0 * i
        batch = news.poll(now)
        assert news.poll(now) == [], f"the feed repeated itself at {i}"
        served += [item.news_id for item in batch]
    assert served, "two hours should have produced headlines"
    assert len(served) == len(set(served)), "a headline was served twice"


def test_shuffle_permutes_which_headline_lands_when(cfg):
    """The control keeps the tape and the timing, and breaks only the pairing.

    If the shuffle also moved the *impulses*, the control would be a different
    market and it would prove nothing about the news.
    """
    market = DemoMarket(cfg.instruments, seed=8, days=3, bar_seconds=300, headlines_per_day=20)
    original = market.headlines()
    shuffled = shuffle_headlines(original, seed=8)
    assert sorted(h.item.text for h in shuffled) == sorted(h.item.text for h in original)
    assert [h.item.ts for h in shuffled] == [h.item.ts for h in original], "timing is unchanged"
    assert [h.impact for h in shuffled] == [h.impact for h in original], "tape is unchanged"
    assert [h.item.text for h in shuffled] != [h.item.text for h in original], "text is permuted"


def test_builders_return_the_configured_kinds(cfg):
    cfg.raw["feeds"] = {"price": "synthetic", "news": "demo"}
    price = build_price_feed(cfg, speed=1.0)
    news = build_news_feed(cfg, price_feed=price)
    assert price.info()["feed"]
    assert news.info()["feed"]
    assert price.snapshot(cfg.symbols[0]) is not None


def test_jsonl_round_trip(tmp_path, cfg, market):
    from jevbot.feeds.jsonl import JsonlNewsFeed, JsonlPriceFeed

    price_path = tmp_path / "prices.jsonl"
    news_path = tmp_path / "news.jsonl"
    with open(price_path, "w") as fh:
        for i in range(300):
            fh.write(json.dumps({"ts": market.start_ts + 300 * i, "symbol": "BTC/USDT",
                                 "price": 100.0 + i}) + "\n")
    heads = market.labelled_headlines()
    with open(news_path, "w") as fh:
        for h in heads[:20]:
            fh.write(json.dumps(h.to_dict()) + "\n")

    pf = JsonlPriceFeed(str(price_path))
    assert pf.snapshot("BTC/USDT") is not None
    nf = JsonlNewsFeed(str(news_path))
    assert len(nf.poll(market.start_ts + 10 * 86400)) >= 1
