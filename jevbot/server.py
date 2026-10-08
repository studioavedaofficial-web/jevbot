"""Dashboard server.

Standard library only — ``http.server`` and ``json``. No web framework, no
build step, no CDN: the page loads as two static files and one JSON endpoint,
which matters because a dashboard you cannot start without a toolchain is a
dashboard that is not running when you need it.

The dashboard reads the *recorded* state from SQLite plus the live snapshot from
the running bot, so it says the same thing whether it is watching a live loop or
replayed from a finished backtest.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .store import Store

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class DashboardServer:
    """Serves the single-page dashboard and its JSON API."""

    def __init__(self, bot: Any, store: Store, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.bot = bot
        self.store = store
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.started_at = time.time()

    # ── payloads ───────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "server": {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "now": time.time(),
            },
            "store": self.store.counts(),
        }
        if self.bot is not None:
            payload["bot"] = self.bot.status()
            payload["live"] = True
        else:
            payload["live"] = False
        return payload

    def equity(self, limit: int) -> dict[str, Any]:
        rows = self.store.latest_equity(limit)
        return {
            "points": [[r["ts"], r["equity"], r["gross_weight"], r["net_weight"],
                        r["drawdown"], r["positions"]] for r in rows]
        }

    def decisions(self, limit: int) -> dict[str, Any]:
        return {"decisions": self.store.recent_decisions(limit)}

    def news(self, limit: int) -> dict[str, Any]:
        return {"news": self.store.recent_news(limit)}

    def signals(self, limit: int) -> dict[str, Any]:
        return {"signals": self.store.recent_signals(limit)}

    def orders(self, limit: int) -> dict[str, Any]:
        return {"orders": self.store.recent_orders(limit), "fills": self.store.recent_fills(limit)}

    def risk(self) -> dict[str, Any]:
        events = []
        for row in self.store.risk_events(20):
            detail = row.get("detail")
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except ValueError:
                    pass
            events.append({**row, "detail": detail})
        out: dict[str, Any] = {"events": events}
        if self.bot is not None:
            out["status"] = self.bot.risk.status(self.bot.portfolio.equity(), self.bot.now())
        return out

    # ── control ────────────────────────────────────────────────────────────

    def control(self, action: str) -> dict[str, Any]:
        if self.bot is None:
            return {"ok": False, "error": "no running bot attached"}
        try:
            if action == "pause":
                self.bot.pause(True)
            elif action == "resume":
                self.bot.pause(False)
            elif action == "flatten":
                self.bot.request_flatten()
            elif action == "flatten_now":
                self.bot.flatten_and_stop_trading()
            elif action == "reset_kill":
                self.bot.reset_kill_switch()
            elif action == "reset_breaker":
                self.bot.reset_venue_breaker()
            elif action == "stop":
                self.bot.request_stop()
            else:
                return {"ok": False, "error": f"unknown action {action!r}"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": str(exc)}
        if self.store is not None:
            self.store.record_risk_event("operator_action", {"action": action})
        return {"ok": True, "action": action}

    # ── server plumbing ────────────────────────────────────────────────────

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
                log.debug("%s - %s", self.address_string(), fmt % args)

            # helpers ------------------------------------------------------
            def _send_json(self, payload: Any, status: int = 200) -> None:
                body = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not getattr(self, "_head_only", False):
                    self.wfile.write(body)

            def _send_file(self, path: Path) -> None:
                if not path.exists():
                    self._send_json({"error": "not found", "path": str(path)}, 404)
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not getattr(self, "_head_only", False):
                    self.wfile.write(body)

            def _query(self) -> tuple[str, dict[str, list[str]]]:
                parsed = urlparse(self.path)
                return parsed.path, parse_qs(parsed.query)

            def _limit(self, qs: dict[str, list[str]], default: int = 100) -> int:
                try:
                    return max(1, min(5000, int(qs.get("limit", [default])[0])))
                except (TypeError, ValueError):
                    return default

            # routes -------------------------------------------------------

            def _dispatch(self, verb: str) -> None:
                """Route one request; never let the socket die mid-response.

                An exception inside a payload builder (SQLite busy while the bot
                is mid-cycle, a half-written row) would otherwise drop the
                connection with no reply at all — which a browser reports the
                same way it reports a dead server. A 500 with the reason keeps
                the page alive and puts the fault where it can be read.
                """
                try:
                    return self._route(verb)
                except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                    raise
                except Exception as exc:
                    log.exception("dashboard %s %s failed", verb, self.path)
                    try:
                        self._send_json({"error": str(exc), "path": self.path}, 500)
                    except Exception:  # pragma: no cover - headers already sent
                        pass

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch("GET")

            def do_HEAD(self) -> None:  # noqa: N802
                """HEAD gets the same status and headers, minus the body.

                Proxies and preview panels probe with HEAD before framing the
                page. Answering 501 to that made a perfectly healthy dashboard
                look like a broken one.
                """
                self._head_only = True
                try:
                    self._dispatch("GET")
                finally:
                    self._head_only = False

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch("POST")

            def _route(self, verb: str) -> None:
                path, qs = self._query()
                if verb == "POST":
                    if path.startswith("/api/control/"):
                        action = path.rsplit("/", 1)[-1]
                        return self._send_json(server.control(action))
                    return self._send_json({"error": "not found", "path": path}, 404)
                if path in ("/", "/index.html"):
                    return self._send_file(WEB_DIR / "index.html")
                if path.startswith("/static/"):
                    return self._send_file(WEB_DIR / path.split("/", 2)[-1])
                if path == "/api/state":
                    return self._send_json(server.state())
                if path == "/api/equity":
                    return self._send_json(server.equity(self._limit(qs, 2000)))
                if path == "/api/decisions":
                    return self._send_json(server.decisions(self._limit(qs, 60)))
                if path == "/api/news":
                    return self._send_json(server.news(self._limit(qs, 40)))
                if path == "/api/signals":
                    return self._send_json(server.signals(self._limit(qs, 40)))
                if path == "/api/orders":
                    return self._send_json(server.orders(self._limit(qs, 60)))
                if path == "/api/risk":
                    return self._send_json(server.risk())
                if path == "/api/health":
                    return self._send_json({"ok": True, **server.state()["store"]})
                self._send_json({"error": "not found", "path": path}, 404)

        return Handler

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self, background: bool = True) -> str:
        self.httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.httpd.daemon_threads = True
        url = f"http://{self.host}:{self.port}/"
        if background:
            self.thread = threading.Thread(target=self.httpd.serve_forever, name="jevbot-http",
                                           daemon=True)
            self.thread.start()
        return url

    def serve_forever(self) -> None:
        if self.httpd is None:
            self.start(background=False)
        assert self.httpd is not None
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover
            pass

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
