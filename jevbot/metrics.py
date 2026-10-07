"""Performance and calibration metrics.

Two families, and keeping them apart is the point.

**Performance** — what the account did: return, volatility, Sharpe, drawdown,
hit rate, profit factor, costs. Necessary, and insufficient: a strategy can be
profitable and wrong.

**Calibration and skill** — whether the engine's numbers mean anything: the
Brier score of its ``noul`` probabilities, the expected calibration error of its
``answer_confidence``, how often its direction call matched the forward return,
and the rank correlation between its signal and what the price did next. A bot
whose Brier score is worse than always saying "0.5" does not have an opinion,
and the P&L that says otherwise is luck.

Both are computed from the same recorded artefacts (decisions, signals, equity)
so live and backtest results are described by identical code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ── performance ─────────────────────────────────────────────────────────────


@dataclass
class Performance:
    starting_equity: float = 0.0
    ending_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    vol_annual: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    trades: int = 0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    fees_paid: float = 0.0
    turnover: float = 0.0
    exposure: float = 0.0
    avg_holding_seconds: float = 0.0
    periods: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.ending_equity, 2),
            "total_return": round(self.total_return, 5),
            "cagr": round(self.cagr, 5),
            "vol_annual": round(self.vol_annual, 5),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "max_drawdown": round(self.max_drawdown, 5),
            "calmar": round(self.calmar, 3),
            "trades": self.trades,
            "hit_rate": round(self.hit_rate, 4),
            "profit_factor": round(self.profit_factor, 3),
            "fees_paid": round(self.fees_paid, 2),
            "turnover": round(self.turnover, 4),
            "exposure": round(self.exposure, 4),
            "avg_holding_seconds": round(self.avg_holding_seconds, 1),
            "periods": self.periods,
        }
        d.update(self.extra)
        return d


def returns_from_curve(curve: Sequence[tuple[float, float]]) -> list[float]:
    """Point-to-point returns of an equity curve."""
    out = []
    for (_, prev), (_, cur) in zip(curve, curve[1:]):
        if prev > 0:
            out.append(cur / prev - 1.0)
    return out


def performance(
    curve: Sequence[tuple[float, float]],
    *,
    starting_equity: float | None = None,
    fills: Iterable[dict[str, Any]] = (),
    trades: Sequence[dict[str, Any]] = (),
    exposure_values: Sequence[float] = (),
    period_seconds: float | None = None,
) -> Performance:
    p = Performance()
    if not curve:
        return p
    p.periods = len(curve)
    p.starting_equity = _f(starting_equity if starting_equity is not None else curve[0][1])
    p.ending_equity = _f(curve[-1][1])
    p.total_return = (p.ending_equity / p.starting_equity - 1.0) if p.starting_equity else 0.0

    rets = returns_from_curve(curve)
    if period_seconds is None and len(curve) > 1:
        period_seconds = max(1e-6, (curve[-1][0] - curve[0][0]) / max(1, len(curve) - 1))
    period_seconds = period_seconds or 1.0
    periods_per_year = (365.25 * 86400.0) / period_seconds

    if rets:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        sd = math.sqrt(max(0.0, var))
        p.vol_annual = sd * math.sqrt(periods_per_year)
        downside = [r for r in rets if r < 0]
        dsd = math.sqrt(sum(r * r for r in downside) / len(downside)) if downside else 0.0
        p.sharpe = (mean / sd) * math.sqrt(periods_per_year) if sd > 0 else 0.0
        p.sortino = (mean / dsd) * math.sqrt(periods_per_year) if dsd > 0 else 0.0
        elapsed_days = max(1e-9, (curve[-1][0] - curve[0][0]) / 86400.0)
        if p.starting_equity > 0 and p.ending_equity > 0:
            growth = p.ending_equity / p.starting_equity
            # CAGR on a two-day window is an extrapolation, not a measurement:
            # compute it, clamp it, and say that it was clamped
            raw = math.log(growth) * (365.25 / elapsed_days)
            capped = max(-10.0, min(10.0, raw))
            p.cagr = math.expm1(capped)
            if abs(raw - capped) > 1e-9:
                p.extra["cagr_clamped"] = True
                p.extra["cagr_annualisation_days"] = round(elapsed_days, 2)

    peak = curve[0][1]
    mdd = 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, 1.0 - eq / peak)
    p.max_drawdown = mdd
    p.calmar = (p.cagr / mdd) if mdd > 0 else 0.0

    fills = list(fills)
    p.fees_paid = sum(_f(f.get("fee")) for f in fills)
    p.turnover = sum(abs(_f(f.get("notional"))) for f in fills) / p.starting_equity if p.starting_equity else 0.0

    trades = list(trades)
    p.trades = len(trades)
    if trades:
        pnls = [_f(t.get("pnl")) for t in trades]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        p.hit_rate = len(wins) / len(pnls)
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        p.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if wins else 0.0
        holds = [_f(t.get("holding_seconds")) for t in trades if t.get("holding_seconds")]
        p.avg_holding_seconds = sum(holds) / len(holds) if holds else 0.0

    if exposure_values:
        p.exposure = sum(exposure_values) / len(exposure_values)
    return p


def round_trips(portfolio) -> list[dict[str, Any]]:
    """Reconstruct closed trades from a fill log (average-cost basis)."""
    open_trades: dict[str, dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    for fill in portfolio.fills:
        symbol = fill.symbol
        sign = 1 if fill.side.value == "buy" else -1
        state = open_trades.setdefault(
            symbol, {"symbol": symbol, "qty": 0.0, "avg": 0.0, "opened": fill.ts, "fees": 0.0}
        )
        state["fees"] += fill.fee
        if state["qty"] == 0 or (state["qty"] > 0) == (sign > 0):
            new_qty = state["qty"] + sign * fill.qty
            if new_qty != 0:
                state["avg"] = (
                    state["avg"] * abs(state["qty"]) + fill.price * fill.qty
                ) / abs(new_qty)
            state["qty"] = new_qty
            continue
        closing = min(abs(sign * fill.qty), abs(state["qty"]))
        direction = 1 if state["qty"] > 0 else -1
        pnl = (fill.price - state["avg"]) * closing * direction - state["fees"]
        closed.append(
            {
                "symbol": symbol,
                "pnl": pnl,
                "qty": closing,
                "entry": state["avg"],
                "exit": fill.price,
                "opened": state["opened"],
                "closed": fill.ts,
                "holding_seconds": fill.ts - state["opened"],
            }
        )
        state["qty"] += sign * fill.qty
        state["fees"] = 0.0
        if abs(state["qty"]) < 1e-12:
            state["qty"] = 0.0
            state["avg"] = 0.0
            state["opened"] = fill.ts
        else:
            state["avg"] = fill.price
            state["opened"] = fill.ts
    return closed


# ── calibration and skill ───────────────────────────────────────────────────


def brier(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if not probabilities:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(probabilities)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[int], bins: int = 10
) -> float:
    """ECE over equal-width confidence bins: |accuracy - confidence|, weighted."""
    if not confidences:
        return float("nan")
    total = len(confidences)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += (len(idx) / total) * abs(acc - conf)
    return ece


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, no scipy needed."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def direction_accuracy(predicted: Sequence[str], realised: Sequence[float],
                       deadband: float = 0.0) -> tuple[float, int]:
    """Accuracy of the calls the engine actually made.

    ``flat`` is a refusal, not a wrong answer, so it is excluded from both the
    numerator and the denominator and reported separately as coverage. Folding a
    refusal into the error rate punishes abstention and rewards a model that
    guesses; excluding it without saying so hides how often there was no call at
    all. Both numbers, always together.
    """
    hits = 0
    counted = 0
    for pred, ret in zip(predicted, realised):
        if pred == "flat" or abs(ret) <= deadband:
            continue
        truth = "long" if ret > 0 else "short"
        counted += 1
        if pred == truth:
            hits += 1
    return (hits / counted if counted else float("nan")), counted


@dataclass
class EngineScore:
    """Everything we can honestly say about one engine's decisions."""

    engine: str = ""
    model: str = ""
    decisions: int = 0
    # noul calibration
    materiality_brier: float = float("nan")
    materiality_base_rate: float = float("nan")
    context_brier: float = float("nan")
    risk_brier: float = float("nan")
    answer_confidence_ece: float = float("nan")
    # direction skill
    direction_accuracy: float = float("nan")
    direction_n: int = 0
    flat_share: float = float("nan")
    labelled: int = 0
    # signal skill
    signal_ic: float = float("nan")
    mean_conviction: float = float("nan")
    mean_priced_in: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = round(v, 4) if isinstance(v, float) and math.isfinite(v) else v
        return out


def score_decisions(
    rows: Sequence[dict[str, Any]],
    forward_returns: dict[str, float],
    sigmas: dict[str, float] | None = None,
    materiality_multiple: float = 0.75,
) -> EngineScore:
    """Score recorded decisions against realised forward returns.

    ``forward_returns`` maps ``news_id|symbol`` to the return of that symbol over
    the horizon after the headline — the label the engine never saw. Rows whose
    label is missing are dropped rather than counted as neutral, because a
    missing label is an absence of evidence, not evidence of no move.

    ``sigmas`` is the volatility each symbol was expected to realise over the
    horizon. "Material" is then *relative*: a 0.5% move in a flat instrument is
    news, the same move in Bitcoin is Tuesday. Scoring materiality against an
    absolute threshold with a 90%+ base rate does not measure the engine, it
    measures the threshold — and it flatters a model that says yes to
    everything.
    """
    s = EngineScore()
    if not rows:
        return s
    s.engine = str(rows[0].get("engine", ""))
    s.model = str(rows[0].get("model", ""))
    s.decisions = len(rows)

    # only labelled rows count; an unlabelled headline is missing evidence, not
    # evidence of no move, so it must not enter any average
    labelled: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        fwd = forward_returns.get(f"{row.get('news_id')}|{row.get('symbol')}")
        if fwd is not None:
            labelled.append((row, float(fwd)))
    if not labelled:
        return s

    mat_p, mat_y, ctx_p, ctx_y, risk_p, risk_y = [], [], [], [], [], []
    conf, correct, preds, rets, signed = [], [], [], [], []
    flats = 0
    for row, fwd in labelled:
        direction = str(row.get("direction", "flat"))
        aligned = 1 if (direction == "long" and fwd > 0) or (direction == "short" and fwd < 0) else 0
        sigma = (sigmas or {}).get(f"{row.get('news_id')}|{row.get('symbol')}")
        threshold = max(0.0005, materiality_multiple * sigma) if sigma else 0.002
        mat_p.append(_f(row.get("materiality"), 0.5))
        mat_y.append(1 if abs(fwd) >= threshold else 0)
        ctx_p.append(_f(row.get("context_aligned"), 0.5))
        ctx_y.append(aligned)
        risk_p.append(_f(row.get("risk_event"), 0.5))
        risk_y.append(1 if fwd <= -0.01 else 0)
        conf.append(_f(row.get("direction_p")))
        correct.append(1 if direction != "flat" and aligned else 0)
        preds.append(direction)
        rets.append(fwd)
        flats += 1 if direction == "flat" else 0
        sign = 1.0 if direction == "long" else -1.0 if direction == "short" else 0.0
        signed.append(sign * _f(row.get("conviction_norm")))

    s.materiality_brier = brier(mat_p, mat_y)
    s.materiality_base_rate = sum(mat_y) / len(mat_y)
    s.context_brier = brier(ctx_p, ctx_y)
    s.risk_brier = brier(risk_p, risk_y)
    s.answer_confidence_ece = expected_calibration_error(conf, correct)
    s.direction_accuracy, s.direction_n = direction_accuracy(preds, rets)
    s.flat_share = flats / len(preds)
    s.labelled = len(labelled)
    s.signal_ic = spearman(signed, rets)
    s.mean_conviction = sum(_f(r.get("conviction_norm")) for r in rows) / len(rows)
    s.mean_priced_in = sum(_f(r.get("priced_in")) for r in rows) / len(rows)
    return s


def format_report(title: str, perf: Performance, extra_lines: Sequence[str] = ()) -> str:
    d = perf.to_dict()
    lines = [
        title,
        "=" * len(title),
        f"  period            {d['periods']} cycles",
        f"  starting equity   {d['starting_equity']:,.2f}",
        f"  ending equity     {d['ending_equity']:,.2f}",
        f"  total return      {d['total_return'] * 100:,.2f}%",
        f"  CAGR              {d['cagr'] * 100:,.2f}%",
        f"  annual vol        {d['vol_annual'] * 100:,.2f}%",
        f"  Sharpe            {d['sharpe']:,.2f}",
        f"  Sortino           {d['sortino']:,.2f}",
        f"  max drawdown      {d['max_drawdown'] * 100:,.2f}%",
        f"  Calmar            {d['calmar']:,.2f}",
        f"  trades            {d['trades']}",
        f"  hit rate          {d['hit_rate'] * 100:,.1f}%",
        f"  profit factor     {d['profit_factor']:,.2f}",
        f"  fees paid         {d['fees_paid']:,.2f}",
        f"  turnover (x eq)   {d['turnover']:,.2f}",
        f"  mean exposure     {d['exposure'] * 100:,.1f}%",
    ]
    lines += [f"  {line}" for line in extra_lines]
    return "\n".join(lines)
