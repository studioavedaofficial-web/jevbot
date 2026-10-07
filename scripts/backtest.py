#!/usr/bin/env python
"""Replay a market through the live loop and print what happened.

    python scripts/backtest.py --days 30 --seed 7
    python scripts/backtest.py --days 30 --seed 7 --shuffle

The second command is the one that matters: it permutes the headlines in time
while keeping the tape identical. If the edge survives the shuffle it was never
a news edge. Run the pair before believing the first number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbot.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["backtest", *sys.argv[1:]]))
