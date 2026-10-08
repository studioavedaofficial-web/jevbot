/* Jevbot dashboard.
 *
 * Plain ES2020, no framework, no build step. Polls a handful of JSON endpoints
 * and redraws. Everything the page shows comes from the bot's own records —
 * there is no separate model of the world here that could disagree with the
 * order log.
 */

const POLL_MS = 2000;
const state = { paused: false, equity: [], decisions: [], news: [], signals: [], positions: [] };

/* ── formatting ──────────────────────────────────────────────────────── */
const fmt = {
  money(v, digits = 2) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    const sign = v < 0 ? '-' : '';
    return `${sign}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
  },
  pct(v, digits = 2, signed = true) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    const s = signed && v > 0 ? '+' : '';
    return `${s}${(v * 100).toFixed(digits)}%`;
  },
  num(v, digits = 2) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return Number(v).toFixed(digits);
  },
  int(v) { return v === null || v === undefined ? '—' : Number(v).toLocaleString(); },
  time(ts) {
    if (!ts) return '—';
    const d = new Date(ts * 1000);
    return d.toISOString().slice(5, 16).replace('T', ' ');
  },
  ago(ts) {
    if (!ts) return '';
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 90) return `${s.toFixed(0)}s ago`;
    if (s < 5400) return `${(s / 60).toFixed(0)}m ago`;
    if (s < 172800) return `${(s / 3600).toFixed(0)}h ago`;
    return `${(s / 86400).toFixed(0)}d ago`;
  },
  cls(v) { return v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral'; },
  esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  },
};

const el = id => document.getElementById(id);
const bar = (p, cls = '') => `<span class="bar ${cls}"><i style="width:${Math.max(0, Math.min(1, p)) * 100}%"></i></span>`;

/* ── data ────────────────────────────────────────────────────────────── */
async function get(path) {
  const r = await fetch(path, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function post(path) {
  const r = await fetch(path, { method: 'POST' });
  return r.json();
}

async function poll() {
  try {
    const [stateRes, equityRes, decRes, newsRes, sigRes, ordRes] = await Promise.all([
      get('/api/state'), get('/api/equity?limit=1500'), get('/api/decisions?limit=40'),
      get('/api/news?limit=25'), get('/api/signals?limit=25'), get('/api/orders?limit=40'),
    ]);
    state.api = stateRes;
    state.equity = equityRes.points || [];
    state.decisions = decRes.decisions || [];
    state.news = newsRes.news || [];
    state.signals = sigRes.signals || [];
    state.orders = ordRes.orders || [];
    state.fills = ordRes.fills || [];
    render();
  } catch (e) {
    el('badges').innerHTML = `<span class="badge killed">disconnected</span>`;
    console.error(e);
  }
}

/* ── header ──────────────────────────────────────────────────────────── */
function renderHeader() {
  const a = state.api || {};
  const bot = a.bot || {};
  const engine = bot.engine || {};
  const broker = bot.broker || {};
  const feed = bot.price_feed || {};
  const risk = bot.risk || {};
  const news = bot.news_feed || {};

  const badges = [];
  // The single most important thing for an operator to see: where orders go.
  if (broker.live) {
    badges.push(`<span class="badge live">LIVE ORDERS — REAL MONEY</span>`);
  } else if (broker.testnet) {
    badges.push(`<span class="badge testnet">TESTNET ORDERS</span>`);
  } else {
    badges.push(`<span class="badge paper">paper — fills simulated locally</span>`);
  }
  if (broker.testnet && broker.endpoint) {
    badges.push(`<span class="badge venue">${fmt.esc(broker.endpoint.replace(/^https?:\/\//, '').split('/')[0])}</span>`);
  }
  badges.push(engine.is_laya
    ? `<span class="badge laya">laya</span>`
    : `<span class="badge fallback">offline fallback engine</span>`);
  if (feed.synthetic) badges.push(`<span class="badge synth">synthetic market</span>`);
  if (risk.killed) badges.push(`<span class="badge killed">kill switch</span>`);
  if (risk.halted_today) badges.push(`<span class="badge killed">halted today</span>`);
  if (bot.paused) badges.push(`<span class="badge fallback">paused</span>`);
  if (bot.venue_stopped) badges.push(`<span class="badge killed">venue breaker</span>`);
  if (bot.stop_requested) badges.push(`<span class="badge killed">stopping</span>`);
  el('badges').innerHTML = badges.join('');

  const pauseBtn = document.querySelector('[data-action="pause"], [data-action="resume"]');
  if (pauseBtn) {
    pauseBtn.dataset.action = bot.paused ? 'resume' : 'pause';
    pauseBtn.textContent = bot.paused ? 'Resume' : 'Pause';
  }

  const simClock = feed.sim_now_utc ? `sim ${fmt.time(feed.sim_now)}` : fmt.time(bot.now);
  el('clock').innerHTML = `${fmt.esc(simClock)} UTC<br/>cycle ${fmt.int(bot.cycle)} · up ${fmt.int(a.server?.uptime_seconds)}s`;

  const banner = [];
  if (broker.live) {
    banner.push(`<b>LIVE ORDERS.</b> Every order below leaves for ${fmt.esc(broker.endpoint || 'the venue')} ` +
      `and spends real money.`);
  } else if (broker.testnet) {
    banner.push(`<b>Testnet orders.</b> Orders are placed on ${fmt.esc(broker.endpoint || 'the sandbox venue')} ` +
      `and filled by the venue's matching engine — fake money, real order flow. ` +
      `Sent ${fmt.int(broker.orders_sent || 0)} · rejected ${fmt.int(broker.orders_rejected || 0)}.`);
  }
  if (feed.synthetic) {
    banner.push(`<b>Synthetic market.</b> ${fmt.esc(feed.warning || '')} ` +
      `Numbers here validate the pipeline end to end; they are not a claim about live edge.`);
  }
  if (bot.venue_stopped) {
    banner.push(`<b>Venue breaker tripped.</b> ${fmt.esc(bot.venue_stop_reason || '')} — ` +
      `no orders are being sent. Fix the cause, then press Reset breaker.`);
  }
  if (news.ok === false && news.status) {
    banner.push(`<b>News feed is not delivering.</b> ${fmt.esc(news.status)}. ` +
      `Nothing can be decided without headlines, so no orders will be sent.`);
  } else if (news.ok === true && (news.served || 0) === 0 && (news.total || 0) === 0
             && (news.sources || []).some(s => s.ok)) {
    banner.push(`<b>News feed is connected but has delivered no headlines yet.</b> ` +
      `Sources reachable, nothing parsed — the next poll is in a few seconds.`);
  }
  if (banner.length) {
    el('warnbar').classList.remove('hidden');
    el('warnbar').innerHTML = banner.join('<br/>');
  } else {
    el('warnbar').classList.add('hidden');
  }
}

/* ── kpis ────────────────────────────────────────────────────────────── */
function renderKpis() {
  const bot = (state.api || {}).bot || {};
  const p = bot.portfolio || {};
  const risk = bot.risk || {};
  const curve = state.equity;
  const start = p.starting_equity || 0;
  const equity = p.equity || 0;
  const total = start ? equity / start - 1 : 0;
  const last = curve.length ? curve[curve.length - 1] : null;

  let sharpe = null;
  if (curve.length > 10) {
    const rets = [];
    for (let i = 1; i < curve.length; i++) {
      const prev = curve[i - 1][1];
      if (prev > 0) rets.push(curve[i][1] / prev - 1);
    }
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, rets.length - 1));
    const dt = (curve[curve.length - 1][0] - curve[0][0]) / Math.max(1, curve.length - 1);
    if (sd > 0 && dt > 0) sharpe = (mean / sd) * Math.sqrt(365.25 * 86400 / dt);
  }

  const kpis = [
    ['equity', fmt.money(equity), `${fmt.int((p.positions || []).length)} open`],
    ['total return', `<span class="${fmt.cls(total)}">${fmt.pct(total)}</span>`, `from ${fmt.money(start, 0)}`],
    ['today', `<span class="${fmt.cls(risk.daily_pnl_pct)}">${fmt.pct(risk.daily_pnl_pct)}</span>`, `limit ${fmt.pct(-(risk.daily_loss_limit_pct || 0), 1, false)}`],
    ['gross / net', `${fmt.pct(p.gross_weight, 1, false)} / ${fmt.pct(p.net_weight, 1)}`, 'exposure of equity'],
    ['realised', `<span class="${fmt.cls(p.realized_pnl)}">${fmt.money(p.realized_pnl)}</span>`, `unreal ${fmt.money(p.unrealized_pnl)}`],
    ['drawdown', `${fmt.pct(p.drawdown, 2, false)}`, `kills at ${fmt.pct(risk.max_drawdown_pct, 0, false)}`],
    ['fees', fmt.money(p.fees_paid), 'cumulative'],
    ['sharpe (live)', sharpe === null ? '—' : fmt.num(sharpe, 2), `${curve.length} points`],
  ];
  el('kpis').innerHTML = kpis.map(([k, v, s]) =>
    `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`
  ).join('');

  el('equity-hint').textContent = last
    ? `gross ${fmt.pct(last[2], 1, false)} · net ${fmt.pct(last[3], 1)} · dd ${fmt.pct(last[4], 1, false)}`
    : '';
}

/* ── chart ───────────────────────────────────────────────────────────── */
function renderChart() {
  const host = el('chart');
  const pts = state.equity;
  if (!pts || pts.length < 2) {
    host.innerHTML = `<div class="empty">Waiting for the first equity points…</div>`;
    return;
  }
  const W = 900, H = 240, PAD = { l: 56, r: 12, t: 12, b: 22 };
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const span = (yMax - yMin) || 1;
  const lo = yMin - span * 0.08, hi = yMax + span * 0.08;
  const X = t => PAD.l + ((t - x0) / ((x1 - x0) || 1)) * (W - PAD.l - PAD.r);
  const Y = v => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);

  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
  const area = `${line} L${X(x1).toFixed(1)},${Y(lo).toFixed(1)} L${X(x0).toFixed(1)},${Y(lo).toFixed(1)} Z`;

  // exposure ribbon, drawn under the price line, scaled to its own max
  const maxGross = Math.max(0.05, ...pts.map(p => p[2] || 0));
  const expo = pts.map(p => {
    const h = (p[2] / maxGross) * 34;
    return `<rect x="${X(p[0]).toFixed(1)}" y="${(H - PAD.b - h).toFixed(1)}" width="1.4" height="${h.toFixed(1)}" fill="#1f4f78" opacity="0.55" />`;
  }).join('');

  const gridY = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const v = lo + (hi - lo) * f;
    return `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="#1c2530" />
            <text x="6" y="${(Y(v) + 3).toFixed(1)}">${fmt.money(v, 0)}</text>`;
  }).join('');

  const start = state.api?.bot?.portfolio?.starting_equity;
  const startLine = start && start > lo && start < hi
    ? `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${Y(start).toFixed(1)}" y2="${Y(start).toFixed(1)}" stroke="#3a4a5c" stroke-dasharray="4 4" />`
    : '';

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${gridY}${startLine}${expo}
    <path d="${area}" fill="url(#g)" opacity="0.35" />
    <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#58a6ff" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#58a6ff" stop-opacity="0"/>
    </linearGradient></defs>
    <path d="${line}" fill="none" stroke="#58a6ff" stroke-width="1.6" />
    <text x="${PAD.l}" y="${H - 6}">${fmt.time(x0)}</text>
    <text x="${W - PAD.r}" y="${H - 6}" text-anchor="end">${fmt.time(x1)}</text>
  </svg>`;
}

/* ── panels ──────────────────────────────────────────────────────────── */
function renderRisk() {
  const bot = (state.api || {}).bot || {};
  const risk = bot.risk || {};
  const broker = bot.broker || {};
  const L = risk.limits || {};
  const rows = [
    ['state', risk.killed ? `<span class="pill short">killed</span>`
      : risk.halted_today ? `<span class="pill warn">halted today</span>`
      : `<span class="pill long">trading</span>`],
    ['kill reason', fmt.esc(risk.kill_reason || '—')],
    ['day', `${fmt.esc(risk.day || '—')} · start ${fmt.money(risk.day_start_equity, 0)}`],
    ['daily pnl', `<span class="${fmt.cls(risk.daily_pnl_pct)}">${fmt.pct(risk.daily_pnl_pct)}</span>`],
    ['per symbol cap', fmt.pct(L.max_weight_per_symbol, 0, false)],
    ['gross / net cap', `${fmt.pct(L.max_gross_weight, 0, false)} / ${fmt.pct(L.max_net_weight, 0, false)}`],
    ['max positions', fmt.int(L.max_positions)],
    ['signal floor', `${fmt.num(L.min_signal, 3)} (hold ${fmt.num((L.min_signal || 0) * (L.hold_signal_ratio ?? 0.5), 3)})`],
    ['round trip cost', `${fmt.num((broker.fee_bps || 0) + (broker.slippage_bps || 0) + (broker.half_spread_bps || 0), 1)} bps/side`],
    ['max order', L.max_order_notional ? fmt.money(L.max_order_notional, 0) : 'uncapped'],
    ['venue breaker', `${fmt.int(L.max_consecutive_rejections)} rejections in a row`
      + (broker.orders_rejected ? ` (${fmt.int(broker.orders_rejected)} so far)` : '')],
    ['stale data', `${fmt.int(L.stale_data_seconds)}s`],
  ];
  const cool = Object.entries(risk.cooldowns || {});
  const events = (risk.events || []).slice(-6).reverse();
  el('risk').innerHTML = `<div class="kv">${rows.map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join('')}</div>`
    + (cool.length ? `<div class="empty">cooldowns: ${cool.map(([s, t]) => `${fmt.esc(s)} ${t}s`).join(', ')}</div>` : '')
    + (events.length ? `<div class="scroll" style="max-height:120px;margin-top:6px">${events.map(e =>
        `<div class="event-item"><span class="meta">${fmt.time(e.ts)}</span> ${fmt.esc(e.event || e.kind || '')} ${fmt.esc(JSON.stringify(e.reason ?? e.detail ?? e.message ?? '').slice(0, 90))}</div>`).join('')}</div>` : '');
}

function renderSignals() {
  const live = ((state.api || {}).bot || {}).signals || [];
  const rows = live.length ? live : state.signals.map(s => ({ ...s, decisions: JSON.parse(s.decisions || '[]'), contributors: JSON.parse(s.contributors || '[]') }));
  if (!rows.length) return el('signals').innerHTML = `<div class="empty">No signals yet.</div>`;
  el('signals').innerHTML = `<table><thead><tr>
      <th>symbol</th><th>score</th><th class="num">target</th><th class="num">now</th><th>horizon</th><th>because</th>
    </tr></thead><tbody>${rows.slice(0, 12).map(s => {
      const score = s.score || 0;
      const cls = score > 0 ? 'long' : score < 0 ? 'short' : 'ghost';
      return `<tr>
        <td>${fmt.esc(s.symbol)}</td>
        <td><div class="split"><span style="width:44px" class="${fmt.cls(score)}">${fmt.num(score, 3)}</span>${bar(Math.abs(score), cls)}</div></td>
        <td class="num">${fmt.pct(s.target_weight, 2)}</td>
        <td class="num">${fmt.pct(s.current_weight ?? 0, 2)}</td>
        <td>${fmt.esc(s.horizon || '')}</td>
        <td>${fmt.esc((s.contributors || []).join(', '))}</td>
      </tr>`;
    }).join('')}</tbody></table>`;
}

function renderPositions() {
  const positions = ((state.api || {}).bot || {}).positions || [];
  if (!positions.length) return el('positions').innerHTML = `<div class="empty">Flat. No open positions.</div>`;
  el('positions').innerHTML = `<table><thead><tr>
      <th>symbol</th><th class="num">qty</th><th class="num">avg</th><th class="num">last</th>
      <th class="num">unreal</th><th class="num">weight</th>
    </tr></thead><tbody>${positions.map(p => `<tr>
      <td>${fmt.esc(p.symbol)}</td>
      <td class="num">${fmt.num(p.qty, 4)}</td>
      <td class="num">${fmt.num(p.avg_price, 2)}</td>
      <td class="num">${fmt.num(p.last_price, 2)}</td>
      <td class="num ${fmt.cls(p.unrealized_pnl)}">${fmt.money(p.unrealized_pnl)}</td>
      <td class="num">${fmt.pct(p.weight ?? 0, 2)}</td>
    </tr>`).join('')}</tbody></table>`;
}

function answerBits(d) {
  const bits = [];
  bits.push(`<span>dir <b>${fmt.esc(d.direction)}</b> p=${fmt.num(d.direction_p, 2)}</span>`);
  bits.push(`<span>material <b>${fmt.num(d.materiality, 2)}</b></span>`);
  bits.push(`<span>conviction <b>${fmt.num(d.conviction_norm, 2)}</b></span>`);
  bits.push(`<span>priced-in <b>${fmt.num(d.priced_in, 1)}</b>/3</span>`);
  bits.push(`<span>aligned <b>${fmt.num(d.context_aligned, 2)}</b></span>`);
  bits.push(`<span>risk-event <b>${fmt.num(d.risk_event, 2)}</b></span>`);
  bits.push(`<span class="pill">${fmt.esc(d.horizon)}</span>`);
  return bits.join('');
}

function renderDecisions() {
  const rows = state.decisions;
  if (!rows.length) return el('decisions').innerHTML = `<div class="empty">No decisions recorded yet.</div>`;
  el('decisions').innerHTML = `<div class="scroll" style="max-height:340px">` + rows.slice(0, 25).map(d => {
    const model = d.model || d.engine;
    const badgeCls = d.engine === 'laya' ? 'laya' : 'flat';
    return `<div class="decision-item">
      <div class="meta">
        <span>${fmt.time(d.ts)}</span>
        <span class="pill ${d.direction === 'long' ? 'long' : d.direction === 'short' ? 'short' : 'flat'}">${fmt.esc(d.symbol)} ${fmt.esc(d.direction)}</span>
        <span class="pill ${badgeCls}">${fmt.esc(model)}</span>
        ${d.materiality > 0.65 ? '<span class="pill warn">material</span>' : ''}
        ${d.priced_in < 1 ? '<span class="pill warn">already priced in</span>' : ''}
        ${d.risk_event > 0.5 ? '<span class="pill short">tail risk</span>' : ''}
        ${d.abstained ? '<span class="pill short">abstained</span>' : ''}
        <span>${fmt.num(d.latency_ms, 1)} ms</span>
      </div>
      ${d.news_text ? `<div class="text">${fmt.esc(d.news_text).slice(0, 220)}</div>` : ''}
      <div class="answers">${answerBits(d)}</div>
      ${d.routing_reason ? `<div class="meta">routing: ${fmt.esc(d.routing_reason)}</div>` : ''}
    </div>`;
  }).join('') + `</div>`;
}

function renderEngine() {
  const engine = ((state.api || {}).bot || {}).engine || {};
  const feed = ((state.api || {}).bot || {}).price_feed || {};
  const news = ((state.api || {}).bot || {}).news_feed || {};
  const routing = ((state.api || {}).bot || {}).routing || {};
  el('engine-hint').textContent = engine.is_laya ? 'Laya checkpoints' : 'offline fallback';
  const rows = [
    ['engine', fmt.esc(engine.engine || '—')],
    ['repo', fmt.esc(engine.repo || engine.model || '—')],
    ['device', fmt.esc(engine.device || '—')],
    ['calls', fmt.int(engine.calls ?? '—')],
    ['batches / cache', `${fmt.int(engine.batches ?? '—')} / ${fmt.int(engine.cache_replays ?? '—')}`],
    ['cache hit rate', engine.cache_hit_rate === undefined ? '—' : fmt.pct(engine.cache_hit_rate, 0, false)],
    ['confidence gate', engine.min_confidence_gate ?? 'off'],
    ['price feed', `${fmt.esc(feed.feed || '—')} ${feed.synthetic ? '(synthetic)' : ''}`],
    ['news feed', `${fmt.esc(news.feed || '—')} · served ${fmt.int(news.served ?? 0)}/${fmt.int(news.total ?? 0)}`
      + (news.status ? `<div class="meta" style="color:${news.ok ? 'var(--muted)' : 'var(--warn)'};font-size:10px">${fmt.esc(news.status)}</div>` : '')],
    ['news routing', fmt.esc(routing.mode || '—')],
  ];
  const stats = routing.stats || {};
  el('engine').innerHTML = `<div class="kv">${rows.map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join('')}</div>`
    + (Object.keys(stats).length ? `<div class="meta" style="color:var(--muted);font-size:10px;margin-top:6px">routing: ${Object.entries(stats).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>` : '')
    + (engine.note ? `<div class="empty" style="font-size:11px">${fmt.esc(engine.note)}</div>` : '');
}

function renderNews() {
  const rows = state.news;
  if (!rows.length) return el('news').innerHTML = `<div class="empty">No headlines read yet.</div>`;
  el('news').innerHTML = `<div class="scroll" style="max-height:320px">` + rows.slice(0, 18).map(n => `
    <div class="news-item">
      <div class="meta">
        <span>${fmt.time(n.ts)}</span><span>${fmt.esc(n.source)}</span>
        <span class="pill">${fmt.esc((JSON.parse(n.symbols || '[]') || []).join(', ') || 'unrouted')}</span>
      </div>
      <div class="text" style="color:#cbd6e2">${fmt.esc(n.text).slice(0, 240)}</div>
      <div class="meta">${fmt.esc(n.routing_reason || '')}</div>
    </div>`).join('') + `</div>`;
}

function renderOrders() {
  const orders = state.orders || [];
  if (!orders.length) return el('orders').innerHTML = `<div class="empty">No orders yet.</div>`;
  el('orders').innerHTML = `<div class="scroll" style="max-height:300px"><table><thead><tr>
      <th>time</th><th>symbol</th><th>side</th><th class="num">qty</th><th>reason</th><th>status</th>
    </tr></thead><tbody>${orders.slice(0, 22).map(o => `<tr>
      <td>${fmt.time(o.ts)}</td>
      <td>${fmt.esc(o.symbol)}</td>
      <td class="${o.side === 'buy' ? 'pos' : 'neg'}">${fmt.esc(o.side)}</td>
      <td class="num">${fmt.num(o.qty, 4)}</td>
      <td>${fmt.esc(o.reason)}</td>
      <td>${fmt.esc(o.status)} <span style="color:var(--muted)">${fmt.esc((o.detail || '').slice(0, 120))}</span></td>
    </tr>`).join('')}</tbody></table></div>`;
}

function renderEvents() {
  const events = (((state.api || {}).bot || {}).events || []).slice(-30).reverse();
  if (!events.length) return el('events').innerHTML = `<div class="empty">Nothing logged yet.</div>`;
  el('events').innerHTML = `<div class="scroll" style="max-height:220px">` + events.map(e => `
    <div class="event-item">
      <div class="meta"><span>${fmt.time(e.ts)}</span><span class="pill">${fmt.esc(e.kind)}</span></div>
      <div>${fmt.esc(e.message || e.text || '')}</div>
      ${e.text && e.message !== e.text ? `<div style="color:var(--muted)">${fmt.esc(e.text).slice(0, 180)}</div>` : ''}
    </div>`).join('') + `</div>`;
}

/* ── wiring ──────────────────────────────────────────────────────────── */
function render() {
  renderHeader(); renderKpis(); renderChart(); renderRisk(); renderSignals();
  renderPositions(); renderDecisions(); renderEngine(); renderNews(); renderOrders(); renderEvents();
}

document.querySelectorAll('[data-action]').forEach(b => {
  b.addEventListener('click', async () => {
    const action = b.dataset.action;
    if (action === 'stop' && !confirm('Stop the trading loop after the current cycle?')) return;
    if (action === 'flatten' && !confirm('Close every position at the next cycle and stop trading?')) return;
    if (action === 'reset_breaker' && !confirm('Allow the bot to send orders to the venue again?')) return;
    const res = await post(`/api/control/${action}`);
    if (!res.ok) alert(res.error || 'action failed');
    poll();
  });
});

poll();
setInterval(poll, POLL_MS);
