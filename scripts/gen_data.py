#!/usr/bin/env python
"""Write a reproducible demo market to disk for replay.

    python scripts/gen_data.py --days 30 --seed 7 --out data/demo

Emits ``headlines.jsonl`` (every generated headline with its routed symbols and
its realised forward return, so an engine can be scored offline) and
``prices.jsonl`` (the bar series). Both are ignored by git: they are outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbot.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["gen-data", *sys.argv[1:]]))
