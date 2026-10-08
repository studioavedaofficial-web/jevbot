"""A local server that speaks the Binance spot REST API.

This exists because the sandbox cannot reach ``testnet.binance.vision``. The
venue is replaced, not the client: a real ``ccxt.binance`` instance is built
exactly as production builds it — sandbox mode, real signing, real HTTP — and
only then pointed at this server. So a test here exercises ccxt's own request
building, HMAC signing, JSON parsing and error mapping, over a real socket.

The server verifies the HMAC-SHA256 signature on private requests. That is the
point of the exercise: a signature is checked, not assumed, so a test that
passes proves the client actually signed with the key it was given, in the
exact shape Binance demands.

What none of this can tell you: whether the real venue accepts these keys, or
returns these numbers. Nothing runnable in this sandbox can.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Deterministic tape: a gentle up-trend so the trend features are non-zero.
START_MS = 1_700_000_000_000  # 2023-11-14, arbitrary but fixed
BAR_MS = 60_000
N_BARS = 600

PRICES = {"BTCUSDT": 30_000.0, "ETHUSDT": 1_800.0}
DRIFT = {"BTCUSDT": 1.5, "ETHUSDT": 0.08}  # per-bar close increase


def closes(symbol: str) -> list[float]:
    base, drift = PRICES[symbol], DRIFT[symbol]
    return [base + drift * i for i in range(N_BARS)]


def klines(symbol: str, limit: int) -> list[list]:
    out = []
    start = N_BARS - min(limit, N_BARS)
    for i, close in enumerate(closes(symbol)[start:]):
        idx = start + i
        ts = START_MS + idx * BAR_MS
        out.append([ts, close - 0.4, close + 0.6, close - 0.9, close, 7.5 + (idx % 3),
                    ts + BAR_MS - 1, 100.0, 12, 4.0, 40.0, "0"])
    return out


class _Handler(BaseHTTPRequestHandler):
    """The subset of the Binance spot API that jevbot touches."""

    def log_message(self, *args):  # keep pytest output clean
        pass

    # ── plumbing ───────────────────────────────────────────────────────────

    def _params(self) -> tuple[str, dict[str, str]]:
        """The signed parameter string and its parsed form.

        Binance puts GET parameters in the query string and POST parameters in
        the body; ccxt appends `signature` last in both cases, which is what
        makes the raw string reusable as the signing input.
        """
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else ""
        raw = body or urllib.parse.urlparse(self.path).query
        return raw, dict(urllib.parse.parse_qsl(raw))

    def _authorised(self, raw: str, params: dict[str, str]) -> tuple[bool, str]:
        """Verify the API key header and the HMAC signature, like the venue does."""
        key = self.headers.get("X-MBX-APIKEY") or ""
        if key != self.server.api_key:
            return False, "bad API key"
        signature = params.pop("signature", "")
        if not signature:
            return False, "missing signature"
        expected = hmac.new(self.server.api_secret.encode(),
                            (raw[: raw.rfind("signature=")]).rstrip("&").encode(),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False, "bad signature"
        return True, ""

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── endpoints ──────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802 - http.server's naming
        path = urllib.parse.urlparse(self.path).path
        raw, params = self._params()
        self.server.seen.append((path, params))
        self.server.requests.append({"method": "GET", "path": path, "params": params})

        if path.endswith("/exchangeInfo"):
            return self._send({"timezone": "UTC", "serverTime": START_MS + N_BARS * BAR_MS,
                               "rateLimits": [], "exchangeFilters": [],
                               "symbols": [self._symbol(s) for s in PRICES]})
        if path.endswith("/klines"):
            symbol = params.get("symbol", "")
            if symbol not in PRICES:
                return self._send({"code": -1121, "msg": "Invalid symbol."}, 400)
            return self._send(klines(symbol, int(params.get("limit", 500))))
        if path.endswith("/ticker/24hr") or path.endswith("/ticker/price"):
            symbol = params.get("symbol", "")
            price = closes(symbol)[-1]
            if path.endswith("/ticker/price"):
                return self._send({"symbol": symbol, "price": f"{price:.2f}"})
            return self._send({"symbol": symbol, "lastPrice": f"{price:.2f}",
                               "bidPrice": f"{price - 0.5:.2f}", "askPrice": f"{price + 0.5:.2f}",
                               "closeTime": START_MS + N_BARS * BAR_MS})
        if path.endswith("/account"):
            ok, why = self._authorised(raw, params)
            self.server.signatures.append(ok)
            if not ok:
                return self._send({"code": -2015, "msg": f"Invalid API-key, IP, or "
                                                         f"permissions for action: {why}"}, 401)
            balances = [{"asset": asset, "free": f"{amount:.8f}", "locked": "0.00000000"}
                        for asset, amount in self.server.balances.items()]
            return self._send({"makerCommission": 10, "takerCommission": 10,
                               "canTrade": True, "canWithdraw": False, "canDeposit": False,
                               "accountType": "SPOT", "balances": balances,
                               "permissions": ["SPOT"]})
        return self._send({"code": -1121, "msg": f"unhandled {path}"}, 404)

    def do_POST(self):  # noqa: N802 - http.server's naming
        path = urllib.parse.urlparse(self.path).path
        raw, params = self._params()
        self.server.seen.append((path, params))
        self.server.requests.append({"method": "POST", "path": path, "params": params})

        if not path.endswith("/order"):
            return self._send({"code": -1121, "msg": f"unhandled {path}"}, 404)

        ok, why = self._authorised(raw, params)
        self.server.signatures.append(ok)
        if not ok:
            return self._send({"code": -2015, "msg": f"Invalid API-key, IP, or "
                                                     f"permissions for action: {why}"}, 401)
        self.server.attempts.append(params)

        if self.server.reject:
            code, message = self.server.reject
            return self._send({"code": code, "msg": message}, 400)

        self.server.orders.append(params)
        symbol = params.get("symbol", "")
        qty = float(params.get("quantity") or 0.0)
        if symbol not in PRICES:
            return self._send({"code": -1121, "msg": "Invalid symbol."}, 400)
        # The fill price is the venue's own: last close plus the venue's own
        # slippage. Deliberately *not* the client's snapshot price, so a test
        # can tell a venue fill from a locally simulated one.
        price = closes(symbol)[-1]
        if params.get("side") == "BUY":
            price *= 1.0 + self.server.slippage
        else:
            price *= 1.0 - self.server.slippage
        commission = qty * price * 0.001
        balance_asset = "USDT" if params.get("side") == "BUY" else symbol[:-4]
        self.server.balances[balance_asset] = max(
            0.0, self.server.balances.get(balance_asset, 0.0)
            + (-qty * price - commission if params.get("side") == "BUY" else qty * price - commission)
        )
        return self._send({
            "symbol": symbol, "orderId": 100 + len(self.server.orders),
            "clientOrderId": params.get("newClientOrderId", ""),
            "transactTime": START_MS + N_BARS * BAR_MS, "price": "0.00000000",
            "origQty": f"{qty:.8f}", "executedQty": f"{qty:.8f}", "status": "FILLED",
            "type": params.get("type", "MARKET"), "side": params.get("side", "BUY"),
            "fills": [{"price": f"{price:.2f}", "qty": f"{qty:.8f}",
                       "commission": f"{commission:.8f}",
                       "commissionAsset": balance_asset}],
        })

    # ── market metadata ────────────────────────────────────────────────────

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


class LocalVenue:
    """A running venue plus the bookkeeping a test wants to assert on."""

    api_key = "local-venue-key"
    api_secret = "local-venue-secret"

    def __init__(self, *, balances: dict[str, float] | None = None,
                 slippage: float = 0.002, reject: tuple[int, str] | None = None) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.seen = []          # (path, params) — order preserved
        self._server.requests = []      # richer records for new tests
        self._server.orders = []        # orders the venue accepted and filled
        self._server.attempts = []      # every authenticated order request, rejections included
        self._server.signatures = []    # True/False per signed request
        self._server.api_key = self.api_key
        self._server.api_secret = self.api_secret
        self._server.balances = dict(balances or {"USDT": 1_000.0, "BTC": 0.0, "ETH": 0.0})
        self._server.slippage = slippage
        self._server.reject = reject
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    # ── what a test reads ──────────────────────────────────────────────────
    @property
    def seen(self) -> list[tuple[str, dict]]:
        return self._server.seen

    @property
    def requests(self) -> list[dict]:
        return self._server.requests

    @property
    def orders(self) -> list[dict]:
        return self._server.orders

    @property
    def attempts(self) -> list[dict]:
        """Every order request that reached the venue, rejections included."""
        return self._server.attempts

    @property
    def balances(self) -> dict[str, float]:
        return self._server.balances

    @property
    def signatures(self) -> list[bool]:
        return self._server.signatures

    @property
    def paths(self) -> list[str]:
        return [p for p, _ in self._server.seen]

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/api/v3"

    def reject_orders(self, code: int = -2010,
                      message: str = "Account has insufficient balance for requested action."):
        self._server.reject = (code, message)

    def accept_orders(self):
        self._server.reject = None

    # ── wiring ─────────────────────────────────────────────────────────────
    def attach(self, monkeypatch, *, patch_broker: bool = True, patch_feed: bool = True) -> None:
        """Point real ccxt clients built by jevbot at this server."""
        from jevbot import venues

        real_build = venues.build_ccxt_exchange

        def build(exchange_id, **kwargs):
            exchange = real_build(exchange_id, **kwargs)
            # Everything above is production: sandbox mode applied by the shared
            # builder, real headers, real signing. Only the host is swapped.
            exchange.urls["api"]["public"] = self.base
            exchange.urls["api"]["private"] = self.base
            return exchange

        if patch_feed:
            monkeypatch.setattr("jevbot.feeds.live.build_ccxt_exchange", build)
        if patch_broker:
            monkeypatch.setattr("jevbot.brokers.ccxt_broker.build_ccxt_exchange", build)
            monkeypatch.setattr("jevbot.venues.build_ccxt_exchange", build)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
