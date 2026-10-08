"""ccxt plumbing shared by the price feed and the broker.

Two things live here because both adapters need them and getting either wrong is
silent:

1. **Pointing a ccxt exchange at its sandbox.** ``set_sandbox_mode(True)``
   rewrites the REST *and* WebSocket URLs together. Setting ``base_url`` by hand
   is how you end up with the public half on testnet and the private half still
   aimed at the live venue — a mistake that shows up as a puzzling 404 on the
   one order you actually cared about.
2. **Saying which endpoint is in use.** ``urls['api']`` is a dictionary of
   *families* (spot ``public``/``private``, plus futures ``dapi``/``fapi`` and
   ``sapi``). "The first URL in there" is whichever key sorts first, which is
   how a spot testnet run reports a coin-margined futures URL in its own logs.
   The spot public endpoint is the one that answers for this bot.
3. **Loading only the markets this bot trades.** ccxt's binance client loads
   spot *and* both futures families by default, so a spot-only run issues
   requests to ``fapi``/``dapi`` as well — and fails outright if either is
   unreachable, taking the spot prices down with it. The demo tape is a spot
   venue; ask for spot.
"""

from __future__ import annotations

from typing import Any


class CCXTUnavailable(RuntimeError):
    """ccxt is missing or the requested exchange id is unknown."""


def build_ccxt_exchange(
    exchange_id: str = "binance",
    *,
    api_key: str = "",
    api_secret: str = "",
    password: str = "",
    testnet: bool = False,
    default_type: str = "spot",
    options: dict[str, Any] | None = None,
) -> Any:
    """Construct a ccxt exchange, optionally aimed at its sandbox."""
    try:
        import ccxt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise CCXTUnavailable("ccxt is not installed (pip install 'jevbot[crypto]')") from exc
    if not hasattr(ccxt, exchange_id):
        raise CCXTUnavailable(f"unknown ccxt exchange id: {exchange_id!r}")

    opts: dict[str, Any] = {"defaultType": default_type}
    if default_type == "spot":
        # Restrict market loading to the family we asked for. Without this,
        # ccxt fetches exchangeInfo for linear and inverse futures too, and any
        # hiccup there fails the whole call — the bot would lose spot prices
        # because a derivatives endpoint it never uses was slow.
        opts["fetchMarkets"] = {"types": ["spot"]}
    opts.update(options or {})
    exchange = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
        "apiKey": api_key or None,
        "secret": api_secret or None,
        "password": password or None,
        "options": opts,
    })
    if testnet:
        # callable(), not hasattr(): an attribute that exists but is None passes
        # hasattr and then explodes as "NoneType is not callable" at the call.
        if not callable(getattr(exchange, "set_sandbox_mode", None)):
            raise CCXTUnavailable(
                f"ccxt exchange {exchange_id!r} has no sandbox to point at "
                f"(set_sandbox_mode missing)"
            )
        exchange.set_sandbox_mode(True)
    return exchange


def is_local_endpoint(url: str) -> bool:
    """True for a loopback address — a venue that cannot be a real exchange.

    Used by the guard on the one command that places orders: `api.binance.com`
    must be refused while a server on 127.0.0.1 (the test venue) is obviously
    not a real account. Without this the guard would be a string match on
    "testnet", which is fine for Binance and useless for anything else.
    """
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except ValueError:  # pragma: no cover - defensive
        return False
    # "localhost" and "127.x" and "[::1]" — the shapes a local venue takes.
    return host in {"localhost", "::1"} or host.startswith("127.")


def rest_endpoint(exchange: Any, *keys: str) -> str:
    """The URL this exchange will actually use for the given API family.

    Defaults to the spot public endpoint, which is what market data and spot
    orders both go through.
    """
    urls = getattr(exchange, "urls", None) or {}
    api = urls.get("api")
    if not isinstance(api, dict):
        return ""
    for key in keys or ("public", "private"):
        value = api.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return ""


def is_testnet_endpoint(url: str) -> bool:
    """True when a REST URL points at a sandbox rather than a live venue."""
    lowered = (url or "").lower()
    return any(marker in lowered for marker in ("testnet", "sandbox", "paper-api", "demo"))
