# jevbot

A news-driven trading bot whose signal engine is **Laya**'s typed System 1
decisions — and a harness honest enough to say when those decisions are worth
nothing.

The bot reads headlines and a text-rendered market snapshot, asks one language
model a set of *typed* questions (direction, materiality, conviction, how much
is already priced in, whether the context agrees, whether this is a tail event),
fuses the answers into a signed score, sizes it against volatility, runs the
result past a risk governor, and sends the difference to a broker. Paper by
default. A dashboard shows every step of that chain, including the raw answers
behind each trade. The same loop runs live, in paper, and in a backtest — the
backtester drives `TradingBot.cycle()` with a clock handed in, so what a
backtest measures is the live code path and not a reimplementation of it.

```
headlines ─┐
           ├─► router ──► rendered state ──► Laya ──► typed answers ──► fusion
market  ───┘   (which        (prose, not      (one     (7 questions)      │
              instrument?)    numbers)      forward                        ▼
                                            pass)                  sizing + risk
                                                                        │
                                       dashboard ◄── store ◄── orders ──┘
                                                                        │
                                                              paper / ccxt / alpaca
```

## Why Laya, and why typed questions

A single forward pass that returns a *choice with probabilities*, a *P(true)*,
and an *expected value with a legend* is a different object from a chatbot. Every
answer is a number you can put in a ledger, gate on, and score against what
actually happened:

| question | type | what it answers |
| --- | --- | --- |
| `direction` | choice | long / flat / short, with probabilities |
| `materiality` | noul | P(this is tradeable at all) |
| `conviction` | score | expected value over a 5-point conviction scale |
| `priced_in` | score | 0–3, how much the market already reflects |
| `context_aligned` | noul | P(the tape agrees with the text) |
| `horizon` | choice | intraday / swing / positional |
| `risk_event` | noul | P(a tail event, i.e. do not size this normally) |

The router runs a *separate* question set for routing ("which instrument is this
about?"), and the keyword router runs first because it is free and explainable —
the model is consulted only when the text names nothing recognisable. Every
decision records which checkpoint answered and why (`routing_reason`), and the
dashboard shows it.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[engine]"     # laya; or -e ".[all]" for ccxt + alpaca too
jevbot doctor                   # checks the environment before you trust it
```

The core bot, the risk engine, the paper broker, the backtester and the
dashboard need **nothing but the standard library**. `laya` is the point of the
project, but its absence is handled honestly: `engine.name = "auto"` tries Laya,
and if the checkpoint cannot answer a smoke test it falls back to a deterministic
offline engine that answers the *same seven questions* from lexicons — and every
decision from then on says `model="heuristic"`, so nothing downstream can
mistake a lexicon's answer for a model's.

## Quickstart

```bash
jevbot run                      # paper trading + dashboard on 0.0.0.0:8000
jevbot backtest --days 30       # replay the demo market through the live loop
jevbot backtest --days 30 --shuffle   # the control (do this one too)
jevbot eval --seeds 7,11,23 --control # multi-seed scoring vs the control
jevbot route "Apple beats earnings as iPhone sales jump"   # one headline, end to end
jevbot gen-data --days 30       # write a replayable market to data/demo/
```

`jevbot run` serves the dashboard at `http://localhost:8000` (see
`docs`-in-code: `jevbot/server.py`). It shows equity, exposure, P&L, open
positions, live signals, every headline the bot read, every decision with its
raw answers, the order/fill log, and the risk governor's state. Buttons:
pause, resume, flatten, flatten-and-stop, reset kill switch, stop.

## What the numbers actually say

Replaying the built-in demo market — a **synthetic** generator, 5 symbols,
30 headlines/day, 10 days, 900 s step, 4 h label horizon, `laya` unavailable in
this environment so the answers come from the offline engine:

| seed | return | excess vs matched B&H | Sharpe | max DD | trades | direction acc | signal IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | +5.11 % | +3.24 % | 20.9 | 1.07 % | 163 | 66.2 % (n=720) | +0.217 |
| 11 | +2.61 % | +3.60 % | 17.5 | 0.78 % | 137 | 65.6 % (n=721) | +0.169 |
| 23 | +7.36 % | +6.67 % | 36.2 | 0.60 % | 140 | 77.6 % (n=751) | +0.436 |
| mean | **+5.03 %** | **+4.50 %** | 24.8 | 0.82 % | 147 | 69.8 % | +0.274 |

And the control that makes those numbers mean something — same tape, same
headlines, permuted in time so text and price stop being related:

| run (seed 7) | return | excess | trades | direction acc | IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | +5.11 % | +3.24 % | 163 | 66.2 % | +0.217 |
| shuffled headlines | −1.01 % | −2.55 % | 148 | 52.4 % | +0.024 |

The edge collapses when the pairing is broken. That is the evidence that the
edge is the *pairing* and not the generator's up-drift or the fee model.

### The reference line that keeps this honest

Every report also scores an **oracle** on exactly the rows the engine answered:
an answerer handed the generator's hidden truth (`valence × magnitude`) and
nothing else — no tape, no text, no context. On the same labels and the same
rows:

| | direction acc | signal IC | rows |
| --- | ---: | ---: | ---: |
| engine (seed 7) | 66.2 % | +0.217 | 1813 |
| oracle (seed 7) | 52.7 % | +0.006 | 1813 |
| engine (seed 11) | 65.6 % | +0.169 | 1815 |
| oracle (seed 11) | 67.1 % | +0.322 | 1815 |
| oracle, shuffled tape | 52.7 % | +0.006 | 1813 |

The oracle scores *identically* on the shuffled control, and that is the
cleanest possible statement of what the shuffle does: it moves the words, not
the tape. The hidden impacts stay where they were, so the answerer that reads
them is unaffected; the answerer that reads the words collapses.

Read that carefully, because it is the most informative table in this file. The
oracle's edge swings from nothing (seed 7) to large (seed 11), which is a
statement about the *testbed*: over a 4 h horizon the demo's noise is larger
than a typical headline's impulse, so knowing the truth is not enough to call
the direction. And the engine beats the oracle on seed 7 — not because it is
clairvoyant, but because it is not the same kind of answerer: it reads the text
*and* the tape, and in this market the impulse is still unfolding, so the price
action it is shown carries information the headline-truth oracle never sees.
That is a legitimate edge for a trade to have, and an illegitimate one for a
*news* claim to take credit for.

**Read this before quoting any of it.** The demo market is a toy written by the
same person who wrote the strategy, and an up-drift makes long books look
clever. What the table demonstrates is that the pipeline carries information
from text to P&L under controlled conditions; it is not evidence of live profit.
Two knobs make that testable rather than rhetorical:

```bash
jevbot backtest --days 10 --impact-scale 0.0   # news stops moving price
jevbot backtest --days 10 --shuffle            # news stops matching the move
```

Both should take the edge away. If a run keeps making money with them, the
money is coming from somewhere else and the harness is lying.

Costs are the honest part of the story. Turnover runs ~8.6× equity per ten days,
and the fees alone come to roughly $430 per run (~0.43 % of a $100k account)
against a net ~+5 %; slippage and spread sit inside the fill prices and are not
broken out. Mean exposure is only ~14 %, so what the table really compares is a
lightly-invested book against a buy-and-hold index scaled to the same exposure.
Turnover, not signal quality, is the binding constraint — see *Known
limitations*.

## Configuration

Everything lives in `config/default.toml` and every key is overridable:
`--config my.toml` (merged), environment variables (`JEVBOT_*`), or
`--set risk.max_positions=3`. The demo market is reproducible from `demo.seed`;
the same seed replays to the same tape, fill for fill.

Points worth knowing:

* `risk.min_order_notional` — the smallest ticket worth managing. Below it, the
  round trip costs more than the position can earn.
* `broker.rebalance_tolerance` (+ `_relative`) — two bands, whichever is larger.
  Without them a signal that drifts a few basis points per cycle rebalances
  every cycle and hands the edge to the exchange.
* `signals.refresh_seconds` — how often an already-decided headline is re-read
  in an updated market context. Re-reading everything every cycle makes the
  target jitter with no new information, and the jitter is paid for in spread.
* `signals.min_materiality` — headlines the engine finds untradeable are not
  aggregated into a position.
* `risk.max_drawdown_pct` — kill switch. It latches; only an operator
  (`POST /api/control/reset_kill`, or `reset_kill_switch()`) clears it.
* `demo.impact_scale` — the size of a headline's price impact. Lower it toward
  zero and the strategy should stop making money.
* `feeds.testnet` / `broker.testnet` — run against sandbox endpoints. Either end
  being a sandbox marks the whole run as one: `cfg.live` is then hard-wired
  `false`, so a testnet process can never demand a live-money acknowledgement or
  put a LIVE badge on the dashboard. The two default to each other, so a
  half-sandboxed run (testnet prices, mainnet orders) has to be asked for.

## Real prices, fake money (Binance testnet)

`config/binance_testnet.toml` is the whole setup — Binance testnet candles, live
RSS headlines, the paper broker, and **unchanged risk limits**:

```bash
export BINANCE_API_KEY=...      # https://testnet.binance.vision — no real money
export BINANCE_API_SECRET=...
export BINANCE_TESTNET=true

jevbot doctor --config config/binance_testnet.toml   # fetches a ticker, cross-checks it
jevbot run    --config config/binance_testnet.toml   # paper fills on real prices
```

`doctor` names the endpoint it is about to use (`https://testnet.binance.vision/api/v3`)
and compares a freshly built snapshot against the venue's own ticker, so a feed
that silently fell back to a generator — or to the wrong network — is caught
before an order is sized off it. Any of `JEVBOT_BROKER_TESTNET`,
`JEVBOT_FEED_TESTNET`, or `BINANCE_TESTNET` sets the same flags, and `.env` is
read at startup (see `.env.example`); a real environment variable wins over the
file.

### Orders on the testnet matching engine

The profile ships with `[broker] kind = "ccxt"`, so orders are **placed on the
sandbox venue and filled by Binance**, not by the local simulator. Same fake
money, real order flow: real signing, real rejections, real lot sizes, real
fees, and a real order history you can open on
[testnet.binance.vision](https://testnet.binance.vision).

Before letting the loop trade, place one order by hand:

```bash
jevbot testnet-order --config config/binance_testnet.toml --notional 12 --yes
```

It prints the endpoint, the quantity it will send, the venue's order id and the
fill, then tells you where to look for it. It refuses to run against anything
that is not a sandbox endpoint, and it clamps the ticket to
`risk.max_order_notional`. Nothing else in the project places an order on
demand like this.

Going back to local fills is one line — `[broker] kind = "paper"` — or one flag:
`jevbot run --mode paper`, which keeps the testnet prices and routes nothing.

### Safety rails

The bot sizes positions as a fraction of equity, so a wrong price or a wrong
weight becomes a wrong ticket. These are the limits that stand between that and
the venue (`[risk]` in `config/binance_testnet.toml`):

| limit | default | what it does |
|---|---|---|
| `max_order_notional` | 50 | no single order may *add* more exposure than this. Exits are exempt — clipping a close leaves dust, and dust re-arms a close every cycle forever |
| `max_consecutive_rejections` | 5 | stop sending orders entirely after this many refusals in a row. A venue that says no five times is not unlucky; the account, symbol or key is wrong, and the next order will not be the one that works |
| `daily_loss_limit_pct` | 3% | flatten and stand down for the rest of the UTC day |
| `max_drawdown_pct` | 10% | latching kill switch; only an operator clears it (`Reset kill` on the dashboard) |
| `min_order_notional` | 25 | the smallest ticket worth paying a round trip for |

Two behaviours worth knowing before you trust any of it:

* **The breaker is a choke point.** Once tripped, `submit()` refuses orders no
  matter who calls it, the cycle stops planning them, and the dashboard shows
  *venue breaker* with a `Reset breaker` button. It does **not** try to flatten:
  a venue refusing orders would refuse the exit too, and the bot would report a
  flat account it does not have.
* **Sizing follows the account, not the config.** On startup a venue broker
  reads the sandbox balance and re-bases the portfolio and the risk anchor on
  it — otherwise a $100,000 config against a $1,500 sandbox reads as a 98.5%
  drawdown on cycle one and trips the kill switch.

### Real money is a separate, deliberate step

Testnet is a sandbox that says yes. Real money is the venue saying no for
reasons nobody documented. When you get there:

1. Create the keys **with withdrawals disabled** (Binance asks for a separate
   permission for that — leave it off; a bot only ever needs spot trade).
2. Restrict the key to your IP if the exchange allows it.
3. Fund it with an amount you would shrug at losing, and set
   `testnet = false` for **both** `[feeds]` and `[broker]`.
4. Tighten the rails first: `max_order_notional`, `daily_loss_limit_pct`,
   `max_drawdown_pct`. Then run with `[broker] kind = "paper"` against live
   prices for a day and watch what it *would* have done.
5. Live orders additionally need `JEVBOT_I_UNDERSTAND_LIVE_RISK=yes`. The
   dashboard then shows `LIVE ORDERS — REAL MONEY` and `broker.live` is true.

Two caveats that matter more than the setup:

* **Testnet is not a preview of mainnet.** Its book is thin and its prices can
  diverge from the real venue; a fill you would never get on mainnet looks
  perfectly ordinary here. That is why this profile quotes more slippage than
  the default, and why nothing in this README's performance table was produced
  from it.
* **A paper fill is still a paper fill.** Real prices do not make simulated
  fills real: no queue position, no partial fills, no funding, no borrow, no
  rejected order that would have been rejected. The engine's edge here has still
  never been tested against a matching engine that can say no.

## Live mode

Live trading needs a broker adapter, API keys, and an explicit acknowledgement:

```bash
pip install -e ".[crypto]"       # ccxt
export JEVBOT_MODE=live JEVBOT_BROKER_KIND=ccxt BINANCE_API_KEY=... BINANCE_API_SECRET=...
export JEVBOT_I_UNDERSTAND_LIVE_RISK=yes
jevbot run
```

The adapters refuse to construct without credentials, the bot refuses to start
in live mode without the acknowledgement variable, and `broker.live` is shown on
the dashboard so a paper run can never be mistaken for a live one. The
acknowledgement is required for mainnet only, and it is unreachable from a
testnet config: sandbox runs are `live = false` by construction. Live orders
are also gated by the same risk governor, the same turnover cap, and the same
kill switch as the backtest.

## Layout

```
jevbot/
  types.py        value types: Decision, Signal, Order, Fill, Position, MarketSnapshot
  text.py         the model's input: market + headline rendered as prose
  questions.py    the typed question sets (trading, routing, quality)
  engine/         base + Laya engine (batched, cached, provenance) + resilient wrapper
                  + the offline fallback, which answers the same seven questions
  routing.py      which instrument is this headline about (keywords first, then the model)
  signals.py      fuse typed answers into a score, decay across headlines, size by vol
  risk.py         the governor: kill switches, caps, cooldowns, hysteresis, turnover
  execution.py    target weights -> orders (bands, lot rounding, fee-aware sizing)
  portfolio.py    the book: average cost, exact cash, two-phase fills
  brokers/        paper (default), ccxt, alpaca — one interface
  feeds/          synthetic demo, jsonl replay, ccxt/alpaca/csv prices, rss/webhook news
  venues.py       shared ccxt plumbing: exchange construction, endpoint naming, testnet detection
  store.py        SQLite: every decision, order and fill, joinable back to its headline
  metrics.py      performance, calibration (Brier/ECE), rank IC, direction accuracy
  backtest.py     drives the live loop on a handed-in clock + the falsification control
  server.py       the dashboard's JSON API and static files
  web/            the dashboard (no framework, no build step)
scripts/          gen_data.py, backtest.py, eval_engine.py
tests/            203 tests: the book, the planner, the governor, the metrics, the API
```

## Testing

```bash
pip install -e ".[dev]" && pytest -q
```

The suite runs entirely offline: no network, no GPU, no API keys — the handlers
in `tests/test_live_venue_integration.py` and `tests/test_venue_orders.py` are
the one exception, and they talk to a local HTTP server instead of a venue,
through a real `ccxt` client so the request paths, the HMAC signing and the
response parsing are the production ones. That server **verifies the signature
on the way in**, so a passing order test means the client really signed with the
key it was given. It is
written around the failures that actually cost time here — a quantity passed where a
notional was expected (a fill that is small, wrong, and perfectly plausible), a
fill that debits cash without writing a position, dust that re-arms a close
forever, wall-clock timestamps leaking into a replay, a warmup longer than the
history producing a full report of zeros, a broker bound to a different
portfolio than the bot, and a side passed as a plain string (equal to `Side.BUY`
but not identical to it, so a buy filled below the mid).

Order routing is verified the same way, and it is worth being precise about
what that does and does not prove. `tests/test_venue_orders.py` covers the POST
to `/order`, the signature, the venue's own fill price and commission, a
rejected order, the rejection breaker, and the startup balance adoption — all
over a real socket, through ccxt's real signing and parsing. It cannot tell you
that Binance testnet accepts these keys: that check requires egress, and is what
`jevbot doctor --config config/binance_testnet.toml` and
`jevbot testnet-order` are for.

Bugs this caught before any of them reached a venue: `create_order(...,
params=None)` — ccxt mutates the params dict it is handed and `None` is not a
dict, so the very first real order died inside ccxt's own plumbing; and the
breaker's first implementation only blocked the cycle path, so a direct
`submit()` still put orders on the wire after it tripped.

Two more were found by the venue test above rather than by reading the code:
the live candle feed re-fed its whole window on every poll — which the feature
engine correctly rejects as out-of-order, so the second cycle of any real run
failed — and the spot exchange client was loading futures markets too, so an
unreachable derivatives endpoint took spot pricing down with it.

## Known limitations

* **The demo market is synthetic, and noise-dominated at the label horizon.**
  Over 4 h the generator's per-bar noise exceeds a typical headline's impact, so
  even an oracle holding the hidden truth scores 53–67 % direction accuracy and
  IC 0.006–0.322 depending on seed. Treat the oracle as a *reference*, not a
  ceiling: the engine also reads the tape, and where an impulse is still
  unfolding, momentum is real information. Any engine claiming a large edge
  above the oracle deserves the interrogation, not the benefit of the doubt.
* **Turnover is high** (~8.6× equity per 10 days) and fees take a visible bite
  out of the gross edge.
  The next honest improvement is not a better signal, it is fewer round trips:
  a minimum holding period, or sizing positions so the expected move dominates
  the round trip by a comfortable multiple.
* **`laya` checkpoints are unverified here** — this sandbox cannot reach
  HuggingFace, so the Laya path is exercised only through its interface (the
  batched call, the cache, the confidence gate, the provenance) and the offline
  engine stands in for everything measured. Run `jevbot doctor` on a machine
  with network access and check `engine.repo` before trusting a Laya number.
* **Equity/fx**: US equities are handled as if $1 of notional is $1 of USDT.
  A multi-currency book would need an fx layer the repo does not have.
* **Shorting** is enabled by default in the paper broker and the demo market
  allows it. Real venues differ in borrow availability; `broker.allow_short`
  restricts the book to longs.

## Licence

Apache-2.0.
