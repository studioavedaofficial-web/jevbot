"""Configuration loading: TOML files + environment overrides.

Precedence, lowest to highest: ``config/default.toml`` → any ``--config`` files
in the order given → ``JEVBOT_*`` environment variables. The environment only
overrides the small set of keys listed in :data:`ENV_MAP`; everything else is
intentionally file/CLI-only so a stray export cannot silently retune the bot.
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Instrument

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.toml"


class ConfigError(RuntimeError):
    """Raised for a configuration that parses but cannot be run."""


# ── deep merge ──────────────────────────────────────────────────────────────


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ── env overrides ───────────────────────────────────────────────────────────
# name -> (config path, cast)
ENV_MAP: dict[str, tuple[tuple[str, ...], str]] = {
    "JEVBOT_ENGINE": (("engine", "name"), "str"),
    "JEVBOT_LAYA_REPO": (("engine", "laya_repo"), "str"),
    "JEVBOT_LAYA_DEVICE": (("engine", "device"), "str"),
    "JEVBOT_LAYA_PRELOAD": (("engine", "preload"), "bool"),
    "JEVBOT_LAYA_MIN_CONFIDENCE": (("engine", "min_confidence"), "float"),
    "JEVBOT_MODE": (("broker", "kind"), "mode"),           # paper|live -> broker kind stays paper unless live
    "JEVBOT_PRICE_FEED": (("feeds", "price"), "str"),
    "JEVBOT_NEWS_FEED": (("feeds", "news"), "str"),
    "JEVBOT_HOST": (("server", "host"), "str"),
    "JEVBOT_PORT": (("server", "port"), "int"),
    "JEVBOT_DB": (("server", "db"), "str"),
}


def _cast(raw: str, kind: str) -> Any:
    if kind == "bool":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    return raw


def _apply_env(cfg: dict[str, Any], env: dict[str, str]) -> None:
    for name, (path, kind) in ENV_MAP.items():
        if name not in env or env[name] == "":
            continue
        value = _cast(env[name], kind) if kind != "mode" else env[name].strip().lower()
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        if kind == "mode":
            # JEVBOT_MODE=live is a *mode*, not a broker kind: it flips paper
            # trading off but leaves the broker adapter choice alone.
            cfg.setdefault("mode", {})["live"] = value == "live"
            continue
        node[path[-1]] = value


# ── typed view ──────────────────────────────────────────────────────────────


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # ── access helpers ─────────────────────────────────────────────────────
    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, *path: str) -> dict[str, Any]:
        return dict(self.get(*path, default={}) or {})

    # ── named views ────────────────────────────────────────────────────────
    @property
    def engine_name(self) -> str:
        name = str(self.get("engine", "name", default="auto")).lower()
        if name not in {"auto", "laya", "heuristic"}:
            raise ConfigError(f"engine.name must be auto|laya|heuristic, got {name!r}")
        return name

    @property
    def instruments(self) -> list[Instrument]:
        rows = self.get("universe", "instruments", default=[]) or []
        if not rows:
            raise ConfigError("universe.instruments is empty — nothing to trade")
        out = []
        for row in rows:
            out.append(
                Instrument(
                    symbol=str(row["symbol"]),
                    market=str(row.get("market", "crypto")),
                    name=str(row.get("name", row["symbol"])),
                )
            )
        return out

    def instrument(self, symbol: str) -> Instrument:
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst
        return Instrument(symbol=symbol, market="crypto" if "/" in symbol else "equity")

    @property
    def symbols(self) -> list[str]:
        return [i.symbol for i in self.instruments]

    @property
    def live(self) -> bool:
        return bool(self.get("mode", "live", default=False)) and self.broker_kind != "paper"

    @property
    def broker_kind(self) -> str:
        return str(self.get("broker", "kind", default="paper")).lower()

    @property
    def data_dir(self) -> Path:
        return Path(self.get("storage", "data_dir", default=str(ROOT / "data")))

    def resolve_path(self, value: str | Path) -> Path:
        """Relative paths in config resolve against the repository root."""
        p = Path(value)
        return p if p.is_absolute() else (ROOT / p)

    def to_dict(self) -> dict[str, Any]:
        d = copy.deepcopy(self.raw)
        d.setdefault("_meta", {})["path"] = str(self.path) if self.path else None
        return d


def load_config(*paths: str | Path, env: dict[str, str] | None = None) -> Config:
    """Load, merge and validate a configuration."""
    env = dict(os.environ) if env is None else env
    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG.exists():
        raw = _load_toml(DEFAULT_CONFIG)
    last: Path | None = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        raw = _merge(raw, _load_toml(path))
        last = path
    _apply_env(raw, env)
    cfg = Config(raw=raw, path=last)
    cfg.instruments  # noqa: B018 — validate the universe at load time
    return cfg
