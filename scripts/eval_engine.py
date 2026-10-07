#!/usr/bin/env python
"""Score a decision engine against forward returns it never saw.

    python scripts/eval_engine.py --engines heuristic --seeds 7,11,23 --control
    python scripts/eval_engine.py --engines heuristic,laya --compare

Two numbers decide whether an engine is worth wiring to an order path:

* direction accuracy on material headlines, against the realised forward return;
* the shuffled control — same tape, same headlines, uncorrelated in time. If the
  accuracy does not fall back toward chance there, the measurement is leaking.

An engine that cannot beat the control is a filter, not an edge, and should be
run with ``--set risk.min_signal=`` high enough that it mostly abstains.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbot.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--control" not in argv and "-h" not in argv and "--help" not in argv:
        argv.append("--control")
    raise SystemExit(main(["eval", *argv]))
