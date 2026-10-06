'use strict';
/*
 * Results page for the keypoint selection benchmark.
 *
 * Every number on the page is computed here, in the browser, from the static files in data/
 * (format: docs/site_data_spec.md). All fetches use relative paths ("data/...") so the page
 * works under any sub-path (GitHub Pages) and under `python -m http.server` run inside docs/.
 *
 * Vocabulary used in the comments:
 *   query  = one test frame of one flight, identified by (flight index, frame number q)
 *   key    = a single number for (flight, q), used to match the same query across runs
 *   fails  = the query failed at threshold T: no pose, or position error > T cm, or rotation error > rot_fail_deg
 */

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Escape text before putting it into HTML. */
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
/** Return the value if it is a finite number, otherwise null. */
function num(v) { return (typeof v === 'number' && Number.isFinite(v)) ? v : null; }
/** Format a number with d decimals; missing values become a dash. */
function fmt(v, d = 1) { return (v == null || !Number.isFinite(v)) ? '–' : v.toFixed(d); }
/** Mean of the non-null values (null when there are none). */
function mean(values) {
  let s = 0, n = 0;
  for (const v of values) if (v != null) { s += v; n++; }
  return n ? s / n : null;
}
/** Median of the non-null values. */
function median(values) {
  const a = values.filter(v => v != null).sort((x, y) => x - y);
  if (!a.length) return null;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
/** One number per query: flight index and frame number packed together. */
const keyOf = (flight, q) => flight * 10000000 + q;
const pad6 = n => String(n).padStart(6, '0');

async function fetchJSON(path) {
  const r = await fetch(path, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

// Fixed colour per run, by its position in index.json (so a run keeps its colour when others are toggled).
const PALETTE = ['#2b6cb0', '#c53030', '#2f855a', '#dd6b20', '#6b46c1', '#975a16', '#d53f8c', '#0987a0', '#4a5568', '#9c9a1c'];

// ---------------------------------------------------------------------------
// Random numbers and statistics
// ---------------------------------------------------------------------------

/** Hash a string to a 32-bit integer (used to seed the bootstrap so results are repeatable). */
function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
/** Small seeded random generator (mulberry32): returns numbers in [0, 1). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 95 % bootstrap interval of a failure rate.
 * fails: array of 0/1, one entry per query. We draw 1000 new samples of the same size
 * with replacement, compute the fail % of each, and take the 2.5 % and 97.5 % percentiles.
 * Example: 10 queries with 2 fails -> 20 %; resamples give e.g. 0 % ... 50 %; the interval is the middle 95 %.
 */
const N_BOOT = 1000;
function bootstrapCI(fails, seedText) {
  const n = fails.length;
  if (n === 0) return null;
  const rand = mulberry32(hashStr(seedText));
  const pcts = new Float64Array(N_BOOT);
  for (let b = 0; b < N_BOOT; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += fails[(rand() * n) | 0];
    pcts[b] = 100 * s / n;
  }
  pcts.sort();
  const lo = pcts[Math.round(0.025 * (N_BOOT - 1))];
  const hi = pcts[Math.round(0.975 * (N_BOOT - 1))];
  return { lo, hi, half: (hi - lo) / 2 };
}

/** Error function (Abramowitz and Stegun 7.1.26, error < 1.5e-7). */
function erf(x) {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const normCdf = z => 0.5 * (1 + erf(z / Math.SQRT2));

/**
 * Two-sided sign test for paired results.
 * saves = queries the baseline fails and the run solves; breaks = the opposite.
 * If the run were no different, each changed query would be a save or a break with probability 1/2,
 * so saves ~ Binomial(saves + breaks, 0.5). p = 2 * P(X <= min(saves, breaks)), capped at 1.
 * Exact sum up to 500 changed queries, normal approximation (with continuity correction) above.
 */
function signTestP(saves, breaks) {
  const n = saves + breaks;
  if (n === 0) return 1;
  const k = Math.min(saves, breaks);
  if (n > 500) {
    const z = Math.max(0, (Math.abs(saves - breaks) - 1) / Math.sqrt(n));
    return Math.min(1, 2 * (1 - normCdf(z)));
  }
  // P(X <= k) = sum_i C(n, i) / 2^n, computed with logs to avoid overflow.
  let logC = 0, cum = 0;
  const logTotal = n * Math.LN2;
  for (let i = 0; i <= k; i++) {
    if (i > 0) logC += Math.log((n - i + 1) / i);
    cum += Math.exp(logC - logTotal);
  }
  return Math.min(1, 2 * cum);
}
function fmtP(p) {
  if (p == null) return '–';
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

const S = {
  index: null,           // index.json
  labels: [],            // per flight index: the labels object of that flight (arrays), or null
  labelPos: [],          // per flight index: Map q -> position in the label arrays
  runInfo: new Map(),    // run id -> entry of index.json runs
  runColor: new Map(),   // run id -> colour
  runLoad: new Map(),    // run id -> Promise of the parsed run file
  runData: new Map(),    // run id -> parsed run file (only after it loaded)
  runError: new Map(),   // run id -> error text when the file could not be loaded
  picks: new Map(),      // "run|slug" -> Promise of the picks object (or null when missing)
  ciCache: new Map(),    // bootstrap results, keyed by run|bin|threshold|query set
  ciGen: 0,              // increases on every render so stale bootstrap work stops
  renderGen: 0,

  // Persisted in the URL hash
  tab: 'summary',
  sel: [],               // selected run ids (index.json order)
  base: null,            // baseline run id
  thr: 25,               // threshold in cm for Summary / Bins and flights
  shared: true,          // compute on the shared queries only
  curveBin: 'all',
  fail: { run: null, bin: 'all', flight: 'all', thr: 25, mode: 'fails' },

  // Not persisted
  sumSort: { col: null, dir: 1 },
  failSort: { col: 'flight', dir: 1 },
  failList: [],          // filtered and sorted rows of the Failures tab
  failCur: -1,           // position of the open query in failList
  failOpenKey: null,     // key of the open query (to keep it open after re-sorting)
  showPicks: true,
};

const rotFail = () => S.index.rot_fail_deg ?? 5;
const binById = id => S.index.bins.find(b => b.id === id) || S.index.bins[0];

/** Does a row fail at threshold T (cm)? */
function isFail(r, T) {
  return r.err == null || r.err > T || (r.rot != null && r.rot > rotFail());
}
/** Does the query belong to the bin? Bins without bounds ("all") hold every query. */
function inBin(bin, r) {
  if (bin.lo == null && bin.hi == null) return true;
  const pf = r.lab.plane_frac;
  if (pf == null) return false;
  return pf >= (bin.lo ?? -Infinity) && pf < (bin.hi ?? Infinity);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

const EMPTY_LABEL = { plane_frac: null, med_depth: null, missing_depth: null, coverage: null };

/** Labels of one query (nulls when not available). */
function labelOf(flight, q) {
  const pos = S.labelPos[flight];
  const i = pos ? pos.get(q) : undefined;
  if (i === undefined) return EMPTY_LABEL;
  const L = S.labels[flight];
  const pick = name => (L[name] ? num(L[name][i]) : null);
  return { plane_frac: pick('plane_frac'), med_depth: pick('med_depth'), missing_depth: pick('missing_depth'), coverage: pick('coverage') };
}

/** Turn a run file into rows with named fields, attach the query labels, and index rows by key. */
function parseRun(raw) {
  const col = {};
  (raw.cols || []).forEach((c, i) => { col[c] = i; });
  const get = (row, name) => (name in col ? num(row[col[name]]) : null);
  const rows = [];
  const byKey = new Map();
  for (const row of raw.rows || []) {
    const r = {
      flight: row[col.flight], q: row[col.q],
      err: get(row, 'err_cm'), rot: get(row, 'rot_deg'),
      n_sel: get(row, 'n_sel'), n_match: get(row, 'n_match'), n_inl: get(row, 'n_inl'),
      rep1: get(row, 'rep1'), rep5: get(row, 'rep5'), rep10: get(row, 'rep10'),
    };
    if (r.err == null) r.rot = null; // the contract says rot is null together with err
    r.key = keyOf(r.flight, r.q);
    r.lab = labelOf(r.flight, r.q);
    rows.push(r);
    byKey.set(r.key, r);
  }
  return { id: raw.id, config: raw.config || {}, rows, byKey };
}

/** Load one run file once; later calls reuse the same promise. */
function loadRun(id) {
  if (!S.runLoad.has(id)) {
    const p = fetchJSON(`data/runs/${encodeURIComponent(id)}.json`)
      .then(raw => { const d = parseRun(raw); S.runData.set(id, d); return d; })
      .catch(e => { S.runError.set(id, String(e.message || e)); return null; });
    S.runLoad.set(id, p);
  }
  return S.runLoad.get(id);
}
/** Load several runs; failures are recorded in S.runError instead of stopping the page. */
async function ensureLoaded(ids) {
  const todo = [...new Set(ids.filter(id => id && S.runInfo.has(id)))];
  if (todo.some(id => !S.runLoad.has(id))) setStatus('loading runs ...');
  await Promise.all(todo.map(loadRun));
  setStatus(S.runError.size ? `could not load: ${[...S.runError.keys()].join(', ')}` : '');
}
/** Of the given ids, keep those whose file is loaded. */
const loaded = ids => ids.filter(id => S.runData.has(id));

/** Picks of one run on one flight, fetched lazily and cached; null when missing (404 is fine). */
function loadPicks(runId, slug) {
  const k = `${runId}|${slug}`;
  if (!S.picks.has(k)) {
    S.picks.set(k, fetch(`data/picks/${encodeURIComponent(runId)}/${encodeURIComponent(slug)}.json`)
      .then(r => (r.ok ? r.json() : null)).catch(() => null));
  }
  return S.picks.get(k);
}

function setStatus(text) { $('#status').textContent = text; }

// ---------------------------------------------------------------------------
// URL hash: keeps the selection and the active tab, so a link reopens the same view
// ---------------------------------------------------------------------------

function readHash() {
  const p = new URLSearchParams(location.hash.replace(/^#/, ''));
  const ids = S.index.runs.map(r => r.id);
  if (p.has('tab')) S.tab = p.get('tab');
  if (p.has('runs')) S.sel = p.get('runs').split(',').filter(id => S.runInfo.has(id));
  if (p.has('base') && S.runInfo.has(p.get('base'))) S.base = p.get('base');
  if (p.has('thr') && S.index.thresholds_cm.includes(+p.get('thr'))) S.thr = +p.get('thr');
  if (p.has('shared')) S.shared = p.get('shared') !== '0';
  if (p.has('cbin')) S.curveBin = p.get('cbin');
  if (p.has('frun') && S.runInfo.has(p.get('frun'))) S.fail.run = p.get('frun');
  if (p.has('fbin')) S.fail.bin = p.get('fbin');
  if (p.has('ffl')) S.fail.flight = p.get('ffl');
  if (p.has('fthr') && S.index.thresholds_cm.includes(+p.get('fthr'))) S.fail.thr = +p.get('fthr');
  if (p.has('fmode')) S.fail.mode = p.get('fmode');
  if (!['summary', 'bins', 'curves', 'failures', 'runs'].includes(S.tab)) S.tab = 'summary';
  // Keep the selection in index.json order.
  S.sel = ids.filter(id => S.sel.includes(id));
}

function writeHash() {
  const p = new URLSearchParams();
  p.set('tab', S.tab);
  p.set('runs', S.sel.join(','));
  if (S.base) p.set('base', S.base);
  p.set('thr', S.thr);
  p.set('shared', S.shared ? '1' : '0');
  p.set('cbin', S.curveBin);
  if (S.fail.run) p.set('frun', S.fail.run);
  p.set('fbin', S.fail.bin);
  p.set('ffl', S.fail.flight);
  p.set('fthr', S.fail.thr);
  p.set('fmode', S.fail.mode);
  history.replaceState(null, '', '#' + p.toString());
}

// ---------------------------------------------------------------------------
// Query sets: which queries enter a comparison
// ---------------------------------------------------------------------------

/**
 * For the selected runs: do they cover different queries, and which queries do they all share?
 * Returns { set, sig, differ, nShared }. set is null when each run should use all of its own rows
 * (toggle off, or every run has the same queries anyway). sig names the query set in cache keys.
 */
function scopeOf(ids) {
  let common = null;
  for (const id of ids) {
    const m = S.runData.get(id).byKey;
    if (common === null) common = new Set(m.keys());
    else for (const k of common) if (!m.has(k)) common.delete(k);
  }
  common = common || new Set();
  const differ = ids.some(id => S.runData.get(id).rows.length !== common.size);
  const use = S.shared && differ;
  return { set: use ? common : null, sig: use ? 'shared:' + [...ids].sort().join(',') : 'own', differ, nShared: common.size };
}
/** Rows of a run that enter the comparison. */
function rowsIn(id, set) {
  const rows = S.runData.get(id).rows;
  return set ? rows.filter(r => set.has(r.key)) : rows;
}

/**
 * Area under the "share of queries with error <= x" curve for x from 0 to cap (cm), divided by cap.
 * A failed query (no pose or rotation too large) counts as error = cap, so it adds nothing.
 * Per query the area is (cap - min(err, cap)) / cap, so the AUC is the mean of that.
 * Example with cap 25: errors 5 cm and "no pose" -> (0.8 + 0) / 2 = 0.4.
 */
function aucOf(rows, cap = 25) {
  if (!rows.length) return null;
  let s = 0;
  for (const r of rows) {
    const bad = r.err == null || (r.rot != null && r.rot > rotFail());
    const e = bad ? cap : Math.min(r.err, cap);
    s += (cap - e) / cap;
  }
  return s / rows.length;
}

/** Fail % and count of a list of rows at threshold T. */
function failPct(rows, T) {
  let k = 0;
  for (const r of rows) if (isFail(r, T)) k++;
  return { pct: rows.length ? 100 * k / rows.length : null, n: rows.length, k };
}

// ---------------------------------------------------------------------------
// Shared bits of HTML
// ---------------------------------------------------------------------------

function swatch(id) { return `<span class="sw" style="background:${S.runColor.get(id) || '#999'}"></span>`; }
function runLabel(id, opts = {}) {
  let h = `${swatch(id)}<span class="rid">${esc(id)}</span>`;
  if (id === S.base) h += ' <span class="tag base">BASELINE</span>';
  if (opts.badge) h += ` <span class="tag warn" title="${esc(opts.badgeTitle || '')}">${esc(opts.badge)}</span>`;
  return h;
}
function flightsBadge(id) {
  // Badge for a run whose flights differ from the union of the selected runs' flights.
  const all = new Set();
  for (const s of S.sel) for (const f of (S.runInfo.get(s).flights || [])) all.add(f);
  const mine = S.runInfo.get(id).flights || [];
  if (mine.length < all.size) return { badge: `${mine.length}/${all.size} flights`, badgeTitle: 'This run covers fewer flights than the selection: ' + mine.join(', ') };
  return {};
}
/** Clickable table header cell used by the sortable tables. */
function th(colId, label, sort, cls = '') {
  const arrow = sort.col === colId ? (sort.dir > 0 ? ' ▲' : ' ▼') : '';
  return `<th class="sortable ${cls}" data-col="${esc(colId)}">${label}${arrow}</th>`;
}
/** Compare for sorting: numbers and strings, missing values always last. */
function cmpVals(a, b, dir) {
  const na = a == null || (typeof a === 'number' && Number.isNaN(a));
  const nb = b == null || (typeof b === 'number' && Number.isNaN(b));
  if (na && nb) return 0;
  if (na) return 1;
  if (nb) return -1;
  if (typeof a === 'string') return dir * a.localeCompare(b);
  return dir * (a - b);
}
function noRuns(el) { el.innerHTML = '<p class="muted pad">Select at least one run above.</p>'; }

// ---------------------------------------------------------------------------
// Header: dataset line, run selector, controls, tabs
// ---------------------------------------------------------------------------

function renderHeader() {
  const I = S.index;
  $('#dataset').textContent = I.dataset || '';
  $('#generated').textContent = I.generated ? `generated ${I.generated.replace('T', ' ')}` : '';

  // Run selector: one table row per run with "show" checkbox and "baseline" radio.
  const rows = I.runs.map(r => {
    const on = S.sel.includes(r.id);
    return `<tr class="${on ? 'on' : ''}">
      <td class="c"><input type="checkbox" data-sel="${esc(r.id)}" ${on ? 'checked' : ''}></td>
      <td class="c"><input type="radio" name="base" data-base="${esc(r.id)}" ${r.id === S.base ? 'checked' : ''}></td>
      <td class="l">${runLabel(r.id)}</td>
      <td class="l">${esc(r.det)}</td><td class="l">${esc(r.picker)}</td><td>${esc(r.K)}</td>
      <td class="l">${esc(r.map)}</td><td class="l">${esc(r.matcher)}</td>
      <td class="l nowrap">${esc((r.date || '').replace('T', ' '))}</td><td>${esc(r.n_queries)}</td>
      <td class="l">${esc((r.flights || []).length)}</td><td>${fmt(num(r.fail25))}</td>
      <td class="l muted">${esc(r.notes || '')}</td></tr>`;
  }).join('');
  $('#runsel').innerHTML = `<thead><tr><th class="c">show</th><th class="c">base</th><th class="l">id</th>
    <th class="l">det</th><th class="l">picker</th><th>K</th><th class="l">map</th><th class="l">matcher</th>
    <th class="l">date</th><th>n_queries</th><th class="l">flights</th><th>fail25 (index)</th><th class="l">notes</th></tr></thead>
    <tbody>${rows}</tbody>`;
  $('#runcount').textContent = `(${S.sel.length} of ${I.runs.length} shown, baseline ${S.base || 'none'})`;

  $$('#runsel [data-sel]').forEach(cb => cb.addEventListener('change', () => {
    const id = cb.dataset.sel;
    const set = new Set(S.sel);
    cb.checked ? set.add(id) : set.delete(id);
    S.sel = I.runs.map(r => r.id).filter(x => set.has(x));
    S.sumSort = { col: null, dir: 1 };
    render();
  }));
  $$('#runsel [data-base]').forEach(rb => rb.addEventListener('change', () => { S.base = rb.dataset.base; render(); }));

  $('#thr').value = String(S.thr);
  $('#shared').checked = S.shared;
  $$('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === S.tab));
}

function initControls() {
  $('#thr').innerHTML = S.index.thresholds_cm.map(t => `<option value="${t}">${t} cm</option>`).join('');
  $('#thr').addEventListener('change', e => { S.thr = +e.target.value; render(); });
  $('#shared').addEventListener('change', e => { S.shared = e.target.checked; render(); });
  $$('#tabs button').forEach(b => b.addEventListener('click', () => { S.tab = b.dataset.tab; render(); }));
  window.addEventListener('hashchange', () => { readHash(); render(); });
  document.addEventListener('keydown', e => {
    // Left / right arrow keys walk the failure table when the detail panel is open.
    if (S.tab !== 'failures' || S.failCur < 0) return;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    if (e.key === 'ArrowLeft') { stepDetail(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { stepDetail(1); e.preventDefault(); }
  });
}

/** Which run files the active tab needs. */
function neededRuns() {
  if (S.tab === 'runs') return S.index.runs.map(r => r.id);
  if (S.tab === 'failures') return [S.fail.run, S.base];
  if (S.tab === 'bins') return [...S.sel, S.base];
  return S.sel;
}

/** Redraw the header and the active tab. */
async function render() {
  const gen = ++S.renderGen;
  if (S.tab === 'failures' && !S.fail.run) {
    S.fail.run = S.sel.find(id => id !== S.base) || S.sel[0] || S.index.runs[0]?.id || null;
  }
  writeHash();
  renderHeader();
  await ensureLoaded(neededRuns());
  if (gen !== S.renderGen) return; // a newer render started while we were loading

  // Note on the query sets, shown next to the "shared queries only" toggle.
  const ids = loaded(S.sel);
  const note = $('#scopeNote');
  note.innerHTML = '';
  if (ids.length > 1) {
    const sc = scopeOf(ids);
    if (sc.differ) {
      note.innerHTML = S.shared
        ? `<span class="tag warn">different query sets</span> numbers use the ${sc.nShared} queries all selected runs share`
        : `<span class="tag warn">different query sets</span> each run uses its own queries; numbers are not directly comparable`;
    }
  }

  const main = $('#main');
  S.ciGen++; // stop any bootstrap work of the previous view
  ({ summary: renderSummary, bins: renderBins, curves: renderCurves, failures: renderFailures, runs: renderRuns })[S.tab](main);
}

// ---------------------------------------------------------------------------
// Tab: Summary
// ---------------------------------------------------------------------------

function renderSummary(el) {
  const ids = loaded(S.sel);
  if (!ids.length) return noRuns(el);
  const T = S.thr;
  const bins = S.index.bins;
  const sc = scopeOf(ids);

  // One record per run with every column value.
  const recs = ids.map(id => {
    const rows = rowsIn(id, sc.set);
    const c = { run: id, n: rows.length };
    for (const b of bins) {
      const br = rows.filter(r => inBin(b, r));
      const fails = new Uint8Array(br.length);
      br.forEach((r, i) => { fails[i] = isFail(r, T) ? 1 : 0; });
      const f = failPct(br, T);
      c['bin:' + b.id] = f.pct;
      c['_bin:' + b.id] = { ...f, fails, ciKey: `${id}|${b.id}|${T}|${sc.sig}` };
    }
    c.auc = aucOf(rows, 25);
    c.med = median(rows.filter(r => !isFail(r, T)).map(r => r.err));
    c.n_match = mean(rows.map(r => r.n_match));
    c.n_inl = mean(rows.map(r => r.n_inl));
    c.rep1 = mean(rows.map(r => r.rep1));
    c.rep5 = mean(rows.map(r => r.rep5));
    c.rep10 = mean(rows.map(r => r.rep10));
    return c;
  });

  const cols = [
    { id: 'run', label: 'run', cls: 'l' },
    { id: 'n', label: 'n' },
    ...bins.map(b => ({ id: 'bin:' + b.id, label: `fail % ${esc(b.label)}`, bin: b })),
    { id: 'auc', label: 'AUC ≤25 cm' },
    { id: 'med', label: `median err (cm), ok@${T}` },
    { id: 'n_match', label: 'n_match' },
    { id: 'n_inl', label: 'n_inl' },
    { id: 'rep1', label: 'rep1' },
    { id: 'rep5', label: 'rep5' },
    { id: 'rep10', label: 'rep10' },
  ];
  if (S.sumSort.col) recs.sort((a, b) => cmpVals(a[S.sumSort.col], b[S.sumSort.col], S.sumSort.dir));

  const tasks = []; // bootstrap cells still to compute
  const body = recs.map((c, ri) => {
    const cells = cols.map(col => {
      if (col.id === 'run') {
        const badge = sc.differ ? flightsBadge(c.run) : {};
        return `<td class="l nowrap">${runLabel(c.run, badge)}</td>`;
      }
      if (col.bin) {
        const d = c['_bin:' + col.bin.id];
        const elId = `ci-${ri}-${col.bin.id}`;
        const hit = S.ciCache.get(d.ciKey);
        if (!hit && d.n) tasks.push({ key: d.ciKey, fails: d.fails, elId });
        return `<td class="nowrap">${fmt(d.pct)} <span class="muted">(${d.n})</span> <span class="ci" id="${elId}" ${hit ? `title="95 % CI ${fmt(hit.lo)} – ${fmt(hit.hi)}"` : ''}>${hit ? '±' + fmt(hit.half) : (d.n ? '…' : '')}</span></td>`;
      }
      const d = { auc: 3, med: 1, n_match: 1, n_inl: 1, rep1: 3, rep5: 3, rep10: 3 }[col.id] ?? 0;
      return `<td>${fmt(c[col.id], d)}</td>`;
    }).join('');
    return `<tr>${cells}</tr>`;
  }).join('');

  el.innerHTML = `
    <div class="tablewrap"><table class="data" id="sumtab">
      <thead><tr>${cols.map(c => th(c.id, c.label, S.sumSort, c.cls)).join('')}</tr></thead>
      <tbody>${body}</tbody></table></div>
    <p class="note">Cells: fail % at ${T} cm (n queries in the bin) and the half-width of the 95 % bootstrap interval
      (${N_BOOT} resamples over queries, fixed seed; hover for the interval). A query fails when it has no pose, its position error is above
      ${T} cm, or its rotation error is above ${rotFail()}°. AUC: area under the share-of-queries-with-error ≤ x curve for x in 0–25 cm,
      failures counted as 25 cm, scaled to 0–1. Median error over the queries that pass at ${T} cm. Means ignore missing values.
      Click a header to sort.</p>`;

  $$('#sumtab th.sortable').forEach(h => h.addEventListener('click', () => {
    const col = h.dataset.col;
    S.sumSort = { col, dir: S.sumSort.col === col ? -S.sumSort.dir : (col === 'run' ? 1 : -1) };
    renderSummary(el);
  }));
  runBootstrapQueue(tasks);
}

/** Compute the bootstrap intervals a few at a time so the page stays responsive. */
function runBootstrapQueue(tasks) {
  const gen = S.ciGen;
  let i = 0;
  function step() {
    if (gen !== S.ciGen) return; // the view changed; stop
    const t0 = performance.now();
    while (i < tasks.length && performance.now() - t0 < 25) {
      const t = tasks[i++];
      let res = S.ciCache.get(t.key);
      if (!res) { res = bootstrapCI(t.fails, t.key); S.ciCache.set(t.key, res); }
      const span = document.getElementById(t.elId);
      if (span && res) { span.textContent = '±' + fmt(res.half); span.title = `95 % CI ${fmt(res.lo)} – ${fmt(res.hi)}`; }
    }
    if (i < tasks.length) setTimeout(step, 0);
  }
  if (tasks.length) setTimeout(step, 0);
}

// ---------------------------------------------------------------------------
// Tab: Bins and flights
// ---------------------------------------------------------------------------

/** Round a maximum up to a tidy axis end (1, 2, 2.5, 5 times a power of ten). */
function niceCeil(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/** Grouped bar chart as an SVG string. groups: labels on x; series: [{id, values: [{v, n}]}]. */
function barChartSVG(groups, series, yLabel) {
  const W = 1180, H = 320, m = { l: 52, r: 12, t: 14, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  let maxV = 0;
  for (const s of series) for (const d of s.values) if (d.v != null) maxV = Math.max(maxV, d.v);
  const yMax = Math.min(100, niceCeil(maxV * 1.1 || 1));
  const y = v => m.t + ih - (v / yMax) * ih;
  const gw = iw / groups.length;
  const bw = Math.min(34, (gw * 0.8) / Math.max(1, series.length));
  let out = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${esc(yLabel)}">`;
  for (let i = 0; i <= 5; i++) { // horizontal grid
    const v = (yMax * i) / 5;
    out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" class="grid"/>`;
    out += `<text x="${m.l - 6}" y="${y(v) + 4}" class="tick" text-anchor="end">${+v.toFixed(1)}</text>`;
  }
  out += `<text x="14" y="${m.t + ih / 2}" class="axlabel" transform="rotate(-90 14 ${m.t + ih / 2})" text-anchor="middle">${esc(yLabel)}</text>`;
  groups.forEach((g, gi) => {
    const x0 = m.l + gi * gw + (gw - bw * series.length) / 2;
    series.forEach((s, si) => {
      const d = s.values[gi];
      if (d.v == null) return;
      const x = x0 + si * bw;
      out += `<rect x="${x + 1}" y="${y(d.v)}" width="${bw - 2}" height="${m.t + ih - y(d.v)}" fill="${s.color}"><title>${esc(s.id)}\n${esc(g)}: ${fmt(d.v)} % (n=${d.n})</title></rect>`;
      if (series.length <= 8) out += `<text x="${x + bw / 2}" y="${y(d.v) - 3}" class="val" text-anchor="middle">${fmt(d.v)}</text>`;
    });
    out += `<text x="${m.l + gi * gw + gw / 2}" y="${H - m.b + 18}" class="tick" text-anchor="middle">${esc(g)}</text>`;
  });
  out += `<line x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}" class="axis"/></svg>`;
  return out;
}
function legendHTML(ids) {
  return `<div class="legend">${ids.map(id => `<span>${runLabel(id)}</span>`).join('')}</div>`;
}

function renderBins(el) {
  const ids = loaded(S.sel);
  if (!ids.length) return noRuns(el);
  const T = S.thr;
  const bins = S.index.bins;
  const sc = scopeOf(ids);
  const flights = S.index.flights;

  // 1) Grouped bar chart: fail % per bin, one bar per run.
  const series = ids.map(id => {
    const rows = rowsIn(id, sc.set);
    return { id, color: S.runColor.get(id), values: bins.map(b => { const f = failPct(rows.filter(r => inBin(b, r)), T); return { v: f.pct, n: f.n }; }) };
  });
  const chart = barChartSVG(bins.map(b => b.label), series, `fail % at ${T} cm`);

  // 2) Per-flight table: rows = flights, columns = runs.
  const usedFlights = flights.map((f, fi) => ({ f, fi })).filter(({ fi }) => ids.some(id => S.runData.get(id).rows.some(r => r.flight === fi)));
  const flRows = usedFlights.map(({ f, fi }) => {
    const cells = ids.map(id => {
      const rows = rowsIn(id, sc.set).filter(r => r.flight === fi);
      if (!rows.length) return '<td class="muted">–</td>';
      const p = failPct(rows, T);
      return `<td>${fmt(p.pct)} <span class="muted">(${p.n})</span></td>`;
    }).join('');
    return `<tr><td class="l nowrap">${esc(f.id)}</td>${cells}</tr>`;
  }).join('');
  const flTable = `<div class="tablewrap"><table class="data">
    <thead><tr><th class="l">flight</th>${ids.map(id => `<th class="nowrap">${runLabel(id)}</th>`).join('')}</tr></thead>
    <tbody>${flRows}</tbody></table></div>
    ${sc.set ? '<p class="note">Shared queries only: a dash means the run has no shared queries on that flight.</p>' : ''}`;

  // 3) Paired comparison with the baseline, per bin, on the queries both runs have.
  let paired;
  const base = S.base && S.runData.get(S.base);
  if (!base) {
    paired = '<p class="muted">Pick a baseline (radio button in the run list) for paired comparisons.</p>';
  } else {
    const others = ids.filter(id => id !== S.base);
    const prow = others.map(id => {
      const run = S.runData.get(id);
      const pairs = run.rows.filter(r => base.byKey.has(r.key)).map(r => [r, base.byKey.get(r.key)]);
      return bins.map((b, bi) => {
        let saves = 0, breaks = 0, n = 0, fr = 0, fb = 0;
        for (const [r, q] of pairs) {
          if (!inBin(b, r)) continue;
          n++;
          const a = isFail(r, T), c = isFail(q, T);
          if (a) fr++;
          if (c) fb++;
          if (c && !a) saves++;
          if (a && !c) breaks++;
        }
        const p = signTestP(saves, breaks);
        const cls = p < 0.05 ? (saves > breaks ? 'good' : 'bad') : '';
        const first = bi === 0 ? `<td class="l nowrap" rowspan="${bins.length}">${runLabel(id)}</td>` : '';
        return `<tr class="${bi === 0 ? 'grp' : ''}">${first}<td class="l">${esc(b.label)}</td><td>${n}</td>
          <td>${n ? fmt(100 * fb / n) : '–'}</td><td>${n ? fmt(100 * fr / n) : '–'}</td>
          <td>${saves}</td><td>${breaks}</td><td>${saves - breaks > 0 ? '+' : ''}${saves - breaks}</td>
          <td class="${cls}">${fmtP(p)}${saves + breaks > 500 ? ' <span class="muted">(normal)</span>' : ''}</td></tr>`;
      }).join('');
    }).join('');
    paired = others.length ? `<div class="tablewrap"><table class="data">
      <thead><tr><th class="l">run</th><th class="l">bin</th><th>n paired</th><th>baseline fail %</th><th>run fail %</th>
      <th>saves</th><th>breaks</th><th>net</th><th>p (two-sided)</th></tr></thead><tbody>${prow}</tbody></table></div>
      <p class="note">Paired on the queries the run and the baseline (${esc(S.base)}) both have, at ${T} cm. Saves: baseline fails, run passes.
      Breaks: run fails, baseline passes. p: exact two-sided binomial (sign) test on saves vs breaks with probability 1/2;
      normal approximation when saves + breaks &gt; 500. Green / red: p &lt; 0.05 in favour of / against the run.</p>`
      : '<p class="muted">Select at least one run besides the baseline.</p>';
  }

  el.innerHTML = `<h2>Fail % per bin at ${T} cm</h2>${legendHTML(ids)}${chart}
    <h2>Fail % per flight at ${T} cm (all frames)</h2>${flTable}
    <h2>Paired comparison against the baseline</h2>${paired}`;
}

// ---------------------------------------------------------------------------
// Tab: Error curves
// ---------------------------------------------------------------------------

const X_MIN = 0.5, X_MAX = 100;

/** Cumulative error curves (x log scale) as an SVG string. curves: [{id, color, errs (sorted), n}] */
function curveChartSVG(curves) {
  const W = 1180, H = 420, m = { l: 52, r: 16, t: 14, b: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lx0 = Math.log(X_MIN), lx1 = Math.log(X_MAX);
  const x = v => m.l + ((Math.log(v) - lx0) / (lx1 - lx0)) * iw;
  const y = v => m.t + ih - v * ih;
  let out = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="cumulative error curves">`;
  for (let i = 0; i <= 10; i++) {
    const v = i / 10;
    out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" class="grid"/>`;
    if (i % 2 === 0) out += `<text x="${m.l - 6}" y="${y(v) + 4}" class="tick" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  for (const t of [0.5, 1, 2, 5, 10, 20, 50, 100]) {
    out += `<line x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${m.t + ih}" class="grid"/>`;
    out += `<text x="${x(t)}" y="${m.t + ih + 16}" class="tick" text-anchor="middle">${t}</text>`;
  }
  for (const t of S.index.thresholds_cm) {
    if (t < X_MIN || t > X_MAX) continue;
    out += `<line x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${m.t + ih}" class="thr"/>`;
    out += `<text x="${x(t) + 3}" y="${m.t + 10}" class="tick">${t} cm</text>`;
  }
  out += `<text x="${m.l + iw / 2}" y="${H - 6}" class="axlabel" text-anchor="middle">camera position error (cm), log scale</text>`;
  out += `<text x="14" y="${m.t + ih / 2}" class="axlabel" transform="rotate(-90 14 ${m.t + ih / 2})" text-anchor="middle">share of queries with error ≤ x and rotation ok</text>`;
  for (const c of curves) {
    if (!c.n) continue;
    // Step curve: count the errors up to X_MIN first, then step up at each error.
    let i = 0;
    while (i < c.errs.length && c.errs[i] <= X_MIN) i++;
    let d = `M${x(X_MIN).toFixed(1)},${y(i / c.n).toFixed(1)}`;
    for (; i < c.errs.length && c.errs[i] <= X_MAX; i++) {
      d += `H${x(c.errs[i]).toFixed(1)}V${y((i + 1) / c.n).toFixed(1)}`;
    }
    d += `H${x(X_MAX).toFixed(1)}`;
    out += `<path d="${d}" fill="none" stroke="${c.color}" stroke-width="1.8"><title>${esc(c.id)}</title></path>`;
  }
  out += `<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" class="frame"/></svg>`;
  return out;
}

function renderCurves(el) {
  const ids = loaded(S.sel);
  if (!ids.length) return noRuns(el);
  const sc = scopeOf(ids);
  const bin = binById(S.curveBin);
  const curves = ids.map(id => {
    const rows = rowsIn(id, sc.set).filter(r => inBin(bin, r));
    // Only queries with a pose and rotation within the limit count; failures never reach the curve.
    const errs = rows.filter(r => r.err != null && !(r.rot != null && r.rot > rotFail())).map(r => r.err).sort((a, b) => a - b);
    return { id, color: S.runColor.get(id), errs, n: rows.length, rows };
  });
  const thr = S.index.thresholds_cm;
  const tab = curves.map(c => {
    const shares = thr.map(t => {
      let k = 0;
      for (const e of c.errs) if (e <= t) k++;
      return `<td>${c.n ? fmt(k / c.n, 3) : '–'}</td>`;
    }).join('');
    return `<tr><td class="l nowrap">${runLabel(c.id)}</td><td>${c.n}</td>${shares}<td>${fmt(aucOf(c.rows, 25), 3)}</td></tr>`;
  }).join('');

  el.innerHTML = `<div class="controls inline"><label>Bin <select id="cbin">${S.index.bins.map(b => `<option value="${esc(b.id)}" ${b.id === bin.id ? 'selected' : ''}>${esc(b.label)}</option>`).join('')}</select></label></div>
    ${legendHTML(ids)}${curveChartSVG(curves)}
    <div class="tablewrap"><table class="data"><thead><tr><th class="l">run</th><th>n</th>${thr.map(t => `<th>share ≤ ${t} cm</th>`).join('')}<th>AUC ≤25 cm</th></tr></thead>
    <tbody>${tab}</tbody></table></div>
    <p class="note">y = share of the queries in the bin whose position error is ≤ x and whose rotation error is ≤ ${rotFail()}°; queries without a pose never count.</p>`;
  $('#cbin').addEventListener('change', e => { S.curveBin = e.target.value; render(); });
}

// ---------------------------------------------------------------------------
// Tab: Failures
// ---------------------------------------------------------------------------

const FAIL_COLS = [
  { id: 'flight', label: 'flight', cls: 'l', val: r => S.index.flights[r.flight]?.id ?? String(r.flight) },
  { id: 'q', label: 'q', val: r => r.q },
  { id: 'plane_frac', label: 'plane_frac', val: r => r.lab.plane_frac, d: 3 },
  { id: 'coverage', label: 'coverage', val: r => r.lab.coverage, d: 3 },
  { id: 'med_depth', label: 'med_depth (m)', val: r => r.lab.med_depth, d: 1 },
  // A missing error means "no pose", the worst case: it sorts above every number.
  { id: 'err_cm', label: 'err_cm', val: r => r.err, sortVal: r => r.err ?? Infinity, d: 1 },
  { id: 'rot_deg', label: 'rot_deg', val: r => r.rot, sortVal: r => r.rot ?? Infinity, d: 2 },
  { id: 'n_match', label: 'n_match', val: r => r.n_match, d: 0 },
  { id: 'n_inl', label: 'n_inl', val: r => r.n_inl, d: 0 },
];
const MAX_ROWS = 500;

/** Build the filtered and sorted list of the Failures tab. */
function computeFailList() {
  const F = S.fail;
  const run = S.runData.get(F.run);
  if (!run) return [];
  const base = S.base ? S.runData.get(S.base) : null;
  const bin = binById(F.bin);
  const T = F.thr;
  const out = [];
  for (const r of run.rows) {
    if (F.flight !== 'all' && String(r.flight) !== F.flight) continue;
    if (!inBin(bin, r)) continue;
    const a = isFail(r, T);
    if (F.mode === 'fails') { if (a) out.push(r); continue; }
    const b = base && base.byKey.get(r.key);
    if (!b) continue; // paired modes need the same query in the baseline
    const c = isFail(b, T);
    if (F.mode === 'not_base' && a && !c) out.push(r);
    if (F.mode === 'base_only' && !a && c) out.push(r);
  }
  const col = FAIL_COLS.find(c => c.id === S.failSort.col) || FAIL_COLS[0];
  const sv = col.sortVal || col.val;
  const dir = S.failSort.dir;
  out.sort((x, y) => {
    let c = col.id === 'flight' ? dir * (x.flight - y.flight) : cmpVals(sv(x), sv(y), dir);
    if (c === 0) c = (x.flight - y.flight) || (x.q - y.q);
    return c;
  });
  return out;
}

function renderFailures(el) {
  const F = S.fail;
  const I = S.index;
  if (!F.run || !S.runData.has(F.run)) { el.innerHTML = `<p class="muted pad">Run ${esc(F.run || '')} could not be loaded.</p>`; return; }
  const opt = (v, label, cur) => `<option value="${esc(v)}" ${String(v) === String(cur) ? 'selected' : ''}>${esc(label)}</option>`;
  const runOpts = I.runs.map(r => opt(r.id, r.id + (S.sel.includes(r.id) ? '' : ' (not shown)'), F.run)).join('');
  const binOpts = I.bins.map(b => opt(b.id, b.label, F.bin)).join('');
  const flOpts = opt('all', 'all flights', F.flight) + I.flights.map((f, i) => opt(String(i), f.id, F.flight)).join('');
  const thrOpts = I.thresholds_cm.map(t => opt(t, t + ' cm', F.thr)).join('');
  const modeOpts = [['fails', 'fails in run'], ['not_base', 'fails in run but not in baseline'], ['base_only', 'fails in baseline but not in run']]
    .map(([v, l]) => opt(v, l, F.mode)).join('');

  S.failList = computeFailList();
  const list = S.failList;
  const needBase = F.mode !== 'fails';
  const warn = needBase && !S.base ? '<span class="tag warn">pick a baseline for this mode</span>' : '';

  el.innerHTML = `
    <div class="controls inline">
      <label>Run <select id="frun">${runOpts}</select></label>
      <label>Bin <select id="fbin">${binOpts}</select></label>
      <label>Flight <select id="ffl">${flOpts}</select></label>
      <label>Threshold <select id="fthr">${thrOpts}</select></label>
      <label>Mode <select id="fmode">${modeOpts}</select></label>
      ${warn}
      <span class="muted">${list.length} queries${list.length > MAX_ROWS ? `, first ${MAX_ROWS} shown` : ''}${needBase && S.base ? `; baseline ${esc(S.base)}` : ''}</span>
    </div>
    <div class="failgrid">
      <div class="tablewrap tall"><table class="data clickable" id="failtab"></table></div>
      <aside id="detail" class="detail"><p class="muted">Click a row to see the query.</p></aside>
    </div>`;

  const bind = (id, field, conv = v => v) => $('#' + id).addEventListener('change', e => { F[field] = conv(e.target.value); S.failCur = -1; render(); });
  bind('frun', 'run'); bind('fbin', 'bin'); bind('ffl', 'flight'); bind('fthr', 'thr', Number); bind('fmode', 'mode');

  // Keep the open query open if it is still in the list.
  if (S.failCur >= 0 && S.failOpenKey != null) {
    S.failCur = list.findIndex(r => r.key === S.failOpenKey);
  }
  drawFailTable();
  if (S.failCur >= 0) renderDetail();
}

function drawFailTable() {
  const tab = $('#failtab');
  if (!tab) return;
  const shown = S.failList.slice(0, MAX_ROWS);
  const body = shown.map((r, i) => `<tr data-i="${i}" class="${i === S.failCur ? 'cur' : ''}">${FAIL_COLS.map(c => {
    const v = c.val(r);
    return `<td class="${c.cls || ''}">${typeof v === 'number' ? fmt(v, c.d ?? 0) : esc(v ?? '–')}</td>`;
  }).join('')}</tr>`).join('');
  tab.innerHTML = `<thead><tr>${FAIL_COLS.map(c => th(c.id, c.label, S.failSort, c.cls)).join('')}</tr></thead><tbody>${body}</tbody>`;
  $$('th.sortable', tab).forEach(h => h.addEventListener('click', () => {
    const col = h.dataset.col;
    S.failSort = { col, dir: S.failSort.col === col ? -S.failSort.dir : (['err_cm', 'rot_deg'].includes(col) ? -1 : 1) };
    S.failList = computeFailList();
    if (S.failOpenKey != null) S.failCur = S.failList.findIndex(r => r.key === S.failOpenKey);
    drawFailTable();
    if (S.failCur >= 0) renderDetail();
  }));
  $$('tbody tr', tab).forEach(tr => tr.addEventListener('click', () => { openDetail(+tr.dataset.i); }));
}

function openDetail(i) {
  if (i < 0 || i >= S.failList.length) return;
  S.failCur = i;
  S.failOpenKey = S.failList[i].key;
  $$('#failtab tbody tr').forEach(tr => tr.classList.toggle('cur', +tr.dataset.i === i));
  const tr = $(`#failtab tbody tr[data-i="${i}"]`);
  if (tr) tr.scrollIntoView({ block: 'nearest' });
  renderDetail();
}
function stepDetail(d) { openDetail(Math.min(S.failList.length - 1, Math.max(0, S.failCur + d))); }

const thumbURL = (slug, frame) => `data/thumbs/${encodeURIComponent(slug)}/${pad6(frame)}.jpg`;

/** Detail panel of the open query: large thumbnail with picks, map keyframes, run vs baseline numbers. */
function renderDetail() {
  const box = $('#detail');
  const r = S.failList[S.failCur];
  if (!box || !r) return;
  const F = S.fail;
  const fl = S.index.flights[r.flight] || { id: String(r.flight), slug: String(r.flight) };
  const base = S.base ? S.runData.get(S.base) : null;
  const b = base ? base.byKey.get(r.key) : null;
  const T = F.thr;

  // Map keyframes: k = 10*m with |k - q| <= 20, inside the flight.
  const kfs = [];
  for (let k = Math.ceil((r.q - 20) / 10) * 10; k <= r.q + 20; k += 10) {
    if (k >= 0 && (fl.n_frames == null || k < fl.n_frames)) kfs.push(k);
  }

  const metric = (label, f, d) => `<tr><th class="l">${label}</th><td>${fmt(f(r), d)}</td><td>${b ? fmt(f(b), d) : '–'}</td></tr>`;
  const status = x => (x ? (isFail(x, T) ? '<span class="bad">fail</span>' : '<span class="good">ok</span>') : '–');
  const hasPicks = !!S.runInfo.get(F.run)?.has_picks;

  box.innerHTML = `
    <div class="dhead">
      <button id="dprev" ${S.failCur <= 0 ? 'disabled' : ''}>◀ prev</button>
      <button id="dnext" ${S.failCur >= S.failList.length - 1 ? 'disabled' : ''}>next ▶</button>
      <span class="muted">${S.failCur + 1} / ${S.failList.length}</span>
      <b>${esc(fl.id)} &nbsp;q=${r.q}</b>
    </div>
    <div class="qimg">
      <img src="${thumbURL(fl.slug, r.q)}" width="480" height="360" alt="query ${r.q}" onerror="this.classList.add('missing')">
      <svg viewBox="0 0 640 480" id="pickov" class="picks"></svg>
    </div>
    <div class="pickline">${hasPicks
      ? `<label><input type="checkbox" id="showPicks" ${S.showPicks ? 'checked' : ''}> show picks of ${esc(F.run)}</label> <span id="pickStatus" class="muted">loading ...</span>`
      : '<span class="muted">no picks exported for this run</span>'}</div>
    <h4>Map keyframes (|k − q| ≤ 20)</h4>
    <div class="kfs">${kfs.map(k => `<figure><img src="${thumbURL(fl.slug, k)}" width="118" height="88" alt="keyframe ${k}" onerror="this.classList.add('missing')"><figcaption>k=${k}</figcaption></figure>`).join('')}</div>
    <table class="data cmp">
      <thead><tr><th class="l"></th><th>run</th><th>baseline</th></tr>
      <tr class="sub"><th class="l"></th><th class="nowrap">${runLabel(F.run)}</th><th class="nowrap">${S.base ? runLabel(S.base) : 'none'}</th></tr></thead>
      <tbody>
        <tr><th class="l">at ${T} cm</th><td>${status(r)}</td><td>${base ? (b ? status(b) : 'not in baseline') : '–'}</td></tr>
        ${metric('err_cm', x => x.err, 1)}${metric('rot_deg', x => x.rot, 2)}
        ${metric('n_sel', x => x.n_sel, 0)}${metric('n_match', x => x.n_match, 0)}${metric('n_inl', x => x.n_inl, 0)}
        ${metric('rep1', x => x.rep1, 3)}${metric('rep5', x => x.rep5, 3)}${metric('rep10', x => x.rep10, 3)}
      </tbody></table>
    <table class="data cmp"><tbody>
      <tr><th class="l">plane_frac</th><td>${fmt(r.lab.plane_frac, 3)}</td></tr>
      <tr><th class="l">coverage</th><td>${fmt(r.lab.coverage, 3)}</td></tr>
      <tr><th class="l">med_depth (m)</th><td>${fmt(r.lab.med_depth, 1)}</td></tr>
      <tr><th class="l">missing_depth</th><td>${fmt(r.lab.missing_depth, 3)}</td></tr>
    </tbody></table>`;

  $('#dprev').addEventListener('click', () => stepDetail(-1));
  $('#dnext').addEventListener('click', () => stepDetail(1));
  if (hasPicks) {
    $('#showPicks').addEventListener('change', e => { S.showPicks = e.target.checked; drawPicks(r, fl); });
    drawPicks(r, fl);
  }
}

/** Fetch (once) and draw the run's picks for this query as dots over the large thumbnail. */
async function drawPicks(r, fl) {
  const runId = S.fail.run;
  const picks = await loadPicks(runId, fl.slug);
  const cur = S.failList[S.failCur];
  if (!cur || cur.key !== r.key || S.fail.run !== runId) return; // the user moved on
  const ov = $('#pickov');
  const st = $('#pickStatus');
  if (!ov) return;
  const pts = picks ? picks[String(r.q)] : null;
  if (st) st.textContent = !picks ? '(picks file not found)' : (pts ? `(${pts.length} points)` : '(no picks for this frame)');
  // Pick coordinates are in the 640x480 original image; the SVG viewBox uses the same units.
  ov.innerHTML = S.showPicks && pts
    ? pts.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="5"/>`).join('')
    : '';
}

// ---------------------------------------------------------------------------
// Tab: Runs
// ---------------------------------------------------------------------------

function renderRuns(el) {
  const blocks = S.index.runs.map(info => {
    const d = S.runData.get(info.id);
    const cfg = d ? d.config : null;
    const dl = obj => Object.entries(obj).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(typeof v === 'object' && v !== null ? JSON.stringify(v) : v)}</dd>`).join('');
    const indexPart = { commit: info.commit, date: (info.date || '').replace('T', ' '), n_queries: info.n_queries, flights: (info.flights || []).join(', '), has_picks: info.has_picks, notes: info.notes };
    return `<section class="runcard ${S.sel.includes(info.id) ? '' : 'dim'}">
      <h3>${runLabel(info.id)}</h3>
      <div class="cols">
        <div><h4>config</h4>${cfg ? `<dl>${dl(cfg)}</dl>` : `<p class="muted">${esc(S.runError.get(info.id) || 'not loaded')}</p>`}</div>
        <div><h4>index entry</h4><dl>${dl(indexPart)}</dl></div>
      </div></section>`;
  }).join('');
  el.innerHTML = `<p class="note">All runs in index.json; runs not selected above are dimmed.</p>${blocks}`;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function init() {
  try {
    const [index, labels] = await Promise.all([fetchJSON('data/index.json'), fetchJSON('data/labels.json').catch(() => ({}))]);
    S.index = index;
    index.bins = index.bins || [{ id: 'all', label: 'all frames' }];
    index.thresholds_cm = index.thresholds_cm || [25];
    index.flights = index.flights || [];
    index.runs = index.runs || [];
    index.flights.forEach((f, i) => {
      const L = labels[f.id] || null;
      S.labels[i] = L;
      S.labelPos[i] = L && L.q ? new Map(L.q.map((q, j) => [q, j])) : null;
    });
    index.runs.forEach((r, i) => { S.runInfo.set(r.id, r); S.runColor.set(r.id, PALETTE[i % PALETTE.length]); });

    // Defaults before reading the hash: up to 6 runs shown, the first one as baseline, 25 cm.
    S.sel = index.runs.slice(0, 6).map(r => r.id);
    S.base = index.runs[0]?.id ?? null;
    S.thr = index.thresholds_cm.includes(25) ? 25 : index.thresholds_cm[0];
    S.fail.thr = S.thr;
    readHash();
    initControls();
    await render();
  } catch (e) {
    $('#main').innerHTML = `<p class="bad pad">Could not load the data: ${esc(e.message || e)}</p>`;
    console.error(e);
  }
}

init();
