"""`.env` loading and the environment override surface.

A `.env` that nothing reads is worse than no `.env`: the keys look configured,
the process runs without them, and the bot quietly trades a different venue than
the operator believes. These tests exist so that cannot happen again — the
loader is asserted to work, and to lose to a real environment variable.
"""

from __future__ import annotations

import os

import pytest

from jevbot.config import ENV_MAP, load_config, load_env_file


@pytest.fixture(autouse=True)
def _restore_environment():
    """load_env_file() writes to os.environ directly, so tests must undo it.

    Without this, a value one test leaves behind becomes ambient state for every
    test that runs after it — which showed up as a ``price feed`` of
    ``"ccxt   # trailing comment..."`` in an unrelated file.
    """
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("JEVBOT_") or key in {"BINANCE_API_KEY", "BINANCE_API_SECRET",
                                                "BINANCE_TESTNET", "ALPACA_PAPER"}:
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def write_env(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body)
    return path


def test_env_file_values_reach_the_environment(tmp_path, clean_env):
    path = write_env(tmp_path, """
# a comment is a whole line
export BINANCE_API_SECRET='shh secret'
BINANCE_API_KEY=abc123
JEVBOT_PRICE_FEED=ccxt
""")
    applied = load_env_file(path)
    assert set(applied) == {"BINANCE_API_KEY", "BINANCE_API_SECRET", "JEVBOT_PRICE_FEED"}
    assert os.environ["BINANCE_API_KEY"] == "abc123"
    assert os.environ["BINANCE_API_SECRET"] == "shh secret"


def test_a_real_environment_variable_wins(tmp_path, clean_env):
    clean_env.setenv("BINANCE_API_KEY", "from-the-shell")
    path = write_env(tmp_path, "BINANCE_API_KEY=from-the-file\n")
    load_env_file(path)
    assert os.environ["BINANCE_API_KEY"] == "from-the-shell", "the file must not clobber a real export"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_env_file_can_configure_the_run(tmp_path, clean_env):
    write_env(tmp_path, """
JEVBOT_BROKER_KIND=ccxt
JEVBOT_BROKER_TESTNET=true
JEVBOT_PRICE_FEED=ccxt
JEVBOT_FEED_TESTNET=true
JEVBOT_NEWS_FEED=rss
JEVBOT_RSS_URLS=https://a.test/f.xml,https://b.test/f.xml
JEVBOT_CCXT_EXCHANGE=binance
""")
    load_env_file(tmp_path / ".env")
    cfg = load_config("config/binance_testnet.toml")
    assert cfg.broker_kind == "ccxt"
    assert cfg.testnet is True
    assert cfg.get("feeds", "price") == "ccxt"
    assert cfg.get("feeds", "news") == "rss"
    assert cfg.get("feeds", "rss_urls") == ["https://a.test/f.xml", "https://b.test/f.xml"]


def test_testnet_via_environment_still_cannot_be_live(tmp_path, clean_env):
    clean_env.setenv("JEVBOT_MODE", "live")
    clean_env.setenv("JEVBOT_BROKER_KIND", "ccxt")
    clean_env.setenv("JEVBOT_BROKER_TESTNET", "true")
    cfg = load_config()
    assert cfg.live is False, "a sandbox endpoint must never satisfy the live check"


def test_every_mapped_env_key_has_a_config_home():
    for name, (path, _kind) in ENV_MAP.items():
        assert path and len(path) >= 2, f"{name} maps to a bare top-level key"
        assert name.startswith("JEVBOT_"), f"{name} should be namespaced"
    for expected in ("JEVBOT_PRICE_FEED", "JEVBOT_NEWS_FEED", "JEVBOT_RSS_URLS",
                     "JEVBOT_BROKER_TESTNET", "JEVBOT_FEED_TESTNET"):
        assert expected in ENV_MAP


def test_boolean_casting_is_forgiving(tmp_path, clean_env):
    for raw, expected in (("true", True), ("1", True), ("yes", True), ("on", True),
                          ("false", False), ("0", False), ("No", False)):
        write_env(tmp_path, f"JEVBOT_BROKER_TESTNET={raw}\n")
        clean_env.delenv("JEVBOT_BROKER_TESTNET", raising=False)
        load_env_file(tmp_path / ".env")
        assert load_config().testnet is expected, f"{raw!r} should cast to {expected}"
        clean_env.delenv("JEVBOT_BROKER_TESTNET", raising=False)
