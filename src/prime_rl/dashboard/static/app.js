/* prime-rl dashboard frontend: metrics, configs, merged logs, traces, and cited reports.
   This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests. */

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const api = async (path) => {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
};

/* accent-first line palette, then the prime-context chart palette */
const PALETTE = ["#b6ff3c", "#b7a6fa", "#78f8a5", "#fcdaa4", "#4a9eff", "#ff6b4a", "#bcbcbc"];
const SINGLE_SERIES = "#b6ff3c";
const POLL_MS = 5000;
const prefs = JSON.parse(localStorage.getItem("prl-dash") || "{}");

const SORT_OPTIONS = new Set([
  "arrival:desc", "arrival:asc", "group:asc",
  "duration:desc", "duration:asc", "reward:desc", "reward:asc", "output_tokens:desc",
]);

const state = {
  runs: [],
  run: null,
  meta: null,
  tab: "metrics",
  live: true,
  metrics: {
    loaded: false, offset: 0, byKey: new Map(),
    charts: [], renderedKeys: -1, timeKeys: new Set(), timeZero: null, maxStep: null,
    collapsedSections: new Set(prefs.collapsedSections ?? []),
    mode: prefs.metricsMode ?? "overview", search: prefs.metricsSearch ?? "",
    smooth: prefs.smooth ?? 1, paneMin: prefs.paneMin ?? 260, paneH: prefs.paneH ?? 150,
    allLayout: prefs.allLayout ?? "flat",
    paneOrder: prefs.paneOrder ?? {},
  },
  compare: { runs: [], data: new Map() },
  config: {
    loaded: false, attempt: "latest", latestAttempt: null, attempts: [],
    files: [], file: null, fmt: "toml", commandText: "", cache: new Map(),
  },
  logs: {
    loaded: false, attempt: "latest", attempts: [], files: [], paneFile: {},
    components: prefs.logComponents ? new Set(prefs.logComponents) : null,
    view: prefs.logView ?? "merge", level: "DEBUG", maximized: null, buffers: new Map(), gseq: 0,
  },
  traces: {
    loaded: false, steps: [], step: null, env: "",
    mode: prefs.traceMode ?? "stream",
    kinds: { train: prefs.traceKinds?.train ?? true, eval: prefs.traceKinds?.eval ?? true },
    bin: null,
    episodes: [],
    total: 0,
    offset: 0,
    paging: false,
    errorsOnly: prefs.traceErrorsOnly ?? false,
    sorts: {
      stream: SORT_OPTIONS.has(prefs.traceSortStream) ? prefs.traceSortStream : "arrival:desc",
      step: SORT_OPTIONS.has(prefs.traceSortStep) ? prefs.traceSortStep : "group:asc",
    },
    viewMode: prefs.tokenSignal === "rendered" ? "rendered" : (prefs.traceViewMode ?? "messages"),
  },
  report: { loaded: false, files: [], file: null, wanted: null, text: null, mtime: null, citations: {}, order: [], verify: new Map() },
  follow: prefs.follow ?? true,
};

function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return "n/a";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e6 || abs < 1e-3) return v.toExponential(2);
  if (abs >= 100) return v.toFixed(1);
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}
const fmtReward = (v) => (v == null || Number.isNaN(v) ? "n/a" : v.toFixed(3));
/* compact human counts that never overflow: 1.1K, 2.2M, 3.3B */
function fmtCompact(n) {
  if (n == null || Number.isNaN(n)) return "n/a";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${+(n / 1e9).toFixed(abs >= 1e10 ? 0 : 1)}B`;
  if (abs >= 1e6) return `${+(n / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (abs >= 1000) return `${+(n / 1000).toFixed(abs >= 10000 ? 0 : 1)}K`;
  return String(n);
}
function fmtCost(v) {
  if (v == null || Number.isNaN(v)) return "n/a";
  return `$${v >= 1 ? v.toFixed(2) : v.toFixed(4)}`;
}

const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/* regex filter with a case-insensitive substring fallback for invalid patterns */
function makeFilter(query) {
  if (!query) return null;
  try {
    return new RegExp(query, "i");
  } catch {
    const needle = query.toLowerCase();
    return { test: (s) => s.toLowerCase().includes(needle) };
  }
}

function debounce(fn, ms = 200) {
  let t = 0;
  return () => {
    clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

const rewardClass = (v) => (v == null ? "" : v > 0 ? "r-pos" : v < 0 ? "r-neg" : "r-zero");

const preview = (text, n) => esc(text.replace(/\s+/g, " ").slice(0, n));

const setActive = (sel, attr, value) =>
  document.querySelectorAll(`${sel} button`).forEach((b) => b.classList.toggle("active", b.dataset[attr] === value));

const emptyState = (title, detail = "") =>
  `<div class="empty-box"><div class="empty-title">${esc(title)}</div>` +
  (detail ? `<div class="empty-detail">${esc(detail)}</div>` : "") +
  `</div>`;

/* ------------------------------------------------------------------- runs */

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  state.outputDir = data.output_dir;
  const sel = $("#run-select");
  const current = state.run;
  sel.disabled = !state.runs.length;
  sel.innerHTML = state.runs.length
    ? state.runs.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("")
    : `<option>no runs found</option>`;
  if (current && state.runs.some((r) => r.name === current)) sel.value = current;
  syncDressedSelects();
  const fresh = state.runs.find((r) => r.name === current);
  if (fresh && state.meta) {
    Object.assign(state.meta, fresh);
    renderOverview();
  }
}

function renderCompareMenu() {
  const menu = $("#compare-menu");
  const others = state.runs.filter((r) => r.name !== state.run);
  menu.innerHTML = others.length
    ? others
        .map(
          (r) =>
            `<label class="file-item"><input type="checkbox" data-compare="${esc(r.name)}"` +
            `${state.compare.runs.includes(r.name) ? " checked" : ""}><span>${esc(r.name)}</span></label>`
        )
        .join("")
    : `<div class="muted" style="padding:6px 8px">no other runs</div>`;
  const btn = $("#compare-btn");
  const n = state.compare.runs.length;
  btn.textContent = n ? `compare (${n})` : "compare";
  btn.classList.toggle("active", n > 0);
}

async function toggleCompare(name, on) {
  const runs = state.compare.runs;
  if (on && !runs.includes(name)) runs.push(name);
  if (!on) {
    state.compare.runs = runs.filter((r) => r !== name);
    state.compare.data.delete(name);
  }
  renderCompareMenu();
  await fetchCompares();
  renderMetricsBody();
}

/* eval runs have one env and no steps: the step bar, kind/subset toggles, chart
   mode, and smoothing make no sense there */
function applyRunTypeControls() {
  const isEval = state.meta?.type === "eval";
  $("#metrics-mode").hidden = isEval;
  $("#smooth-range").closest(".ctl").hidden = isEval;
  $("#step-bar").hidden = isEval;
  // an eval run has no steps to switch between, so it is stream-only
  $("#trace-mode").hidden = isEval;
  $("#tm-mode-row").hidden = isEval;
  $("#tm-step-prev").hidden = isEval;
  $("#tm-step-next").hidden = isEval;
  if (isEval) state.traces.mode = "stream";
}

async function selectRun(name, deferTab = false) {
  if (!name) return;
  state.run = name;
  state.compare = { runs: [], data: new Map() };
  $("#run-select").value = name;
  syncDressedSelects();
  state.meta = state.runs.find((r) => r.name === name) ?? (await api(`/api/runs/${encodeURIComponent(name)}`));
  state.metrics = {
    ...state.metrics,
    loaded: false, fetching: false, offset: 0, byKey: new Map(), charts: [], renderedKeys: -1,
    timeKeys: new Set(), timeZero: null, maxStep: null,
    evalEtag: null, evalCount: 0, evalCost: null,
  };
  if (state.meta?.type === "eval") fetchEvalSeries(); // populates the overview cost early
  state.config = {
    loaded: false, attempt: "latest", latestAttempt: null, attempts: [],
    files: [], file: null, fmt: state.config.fmt, commandText: "", cache: new Map(),
  };
  state.logs = {
    ...state.logs, loaded: false, attempt: "latest", latestAttempt: null,
    files: [], paneFile: {}, maximized: null, buffers: new Map(),
  };
  state.traces = {
    ...state.traces,
    loaded: false, fetching: false, steps: [], step: null, env: "", episodes: [], etag: null,
    key: null, total: 0, bin: null, hist: null,
  };
  state.report = {
    ...state.report,
    loaded: false, files: [], file: null, text: null, mtime: null, citations: {}, order: [], verify: new Map(),
  };
  applyRunTypeControls();
  renderOverview();
  renderCompareMenu();
  updateHash();
  if (!deferTab) await activateTab(state.tab, true);
}

function fmtDuration(secs) {
  if (secs == null || !isFinite(secs) || secs < 0) return "n/a";
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  const parts = [];
  if (d) parts.push(`${d}d`);
  if (d || h) parts.push(`${h}h`);
  if (d || h || m) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(" ");
}

function fmtAgo(ts) {
  if (!ts) return "n/a";
  const secs = Date.now() / 1000 - ts;
  if (secs < 90) return "just now";
  if (secs < 3600) return `${Math.round(secs / 60)} min ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)} h ago`;
  const days = Math.floor(secs / 86400);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function currentStep() {
  return state.metrics.maxStep ?? state.meta?.last_step ?? null;
}

function runStatus(step) {
  const meta = state.meta;
  if (meta.updated && Date.now() / 1000 - meta.updated < 180) return "running";
  if (step != null && meta.max_steps && step >= meta.max_steps) return "completed";
  return "stopped";
}

/* unbounded env lists: show the first two, fold the rest into "+N" (full list
   in the tooltip) */
function envListField(envs, empty = "n/a") {
  if (!envs?.length) return `<span class="val">${empty}</span>`;
  const display = envs.length > 2 ? `${envs.slice(0, 2).join(", ")} +${envs.length - 2}` : envs.join(", ");
  return `<span class="val" title="${esc(envs.join(", "))}">${esc(display)}</span>`;
}

function renderOverview() {
  const el = $("#run-overview");
  const meta = state.meta;
  if (!meta) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const step = currentStep();
  const status = runStatus(step);
  const durationEnd = status === "running" ? Date.now() / 1000 : meta.updated;
  const duration = meta.started && durationEnd ? fmtDuration(durationEnd - meta.started) : "n/a";
  const field = ([label, value]) => `<div class="ov-field"><span class="lbl">${label}</span>${value}</div>`;
  // rollout dirs can run one step past max_steps (the final ship drains late
  // arrivals), so the headline caps at the configured horizon
  const shownStep = step != null && meta.max_steps ? Math.min(step, meta.max_steps) : step;
  const stepText = `${shownStep != null ? shownStep.toLocaleString() : "–"}/${meta.max_steps ? meta.max_steps.toLocaleString() : "∞"}`;
  const left = [
    ["status", `<span class="badge st-${status}">${status}</span>`],
    ["type", `<span class="val">${esc((meta.type ?? "n/a").toUpperCase())}</span>`],
    meta.type === "eval"
      ? ["episodes", `<span class="val">${step != null ? step.toLocaleString() : "n/a"}</span>`]
      : ["step", `<span class="val">${stepText}</span>`],
    ["model", `<span class="val" title="${esc(meta.model ?? "")}">${esc(meta.model ?? "n/a")}</span>`],
    ...(meta.type === "eval"
      ? [["env", `<span class="val" title="${esc(meta.env ?? "")}">${esc(meta.env ?? "n/a")}</span>`]]
      : [
          meta.type === "sft"
            ? ["dataset", `<span class="val" title="${esc(meta.dataset ?? "")}">${esc(meta.dataset ?? "n/a")}</span>`]
            : ["train envs", envListField(meta.train_envs)],
          // an empty eval env list is a known "none", not missing data
          ["eval envs", envListField(meta.eval_envs, "–")],
        ]),
  ];
  const right = [
    ...(meta.type === "eval" && state.metrics.evalCost != null
      ? [["cost", `<span class="val">${fmtCost(state.metrics.evalCost)}</span>`]]
      : []),
    ["duration", `<span class="val">${duration}</span>`],
    ["created", `<span class="val">${fmtAgo(meta.created)}</span>`],
  ];
  el.innerHTML =
    `<div class="ov-top">` +
    left.map(field).join("") +
    `<div class="spacer"></div>` +
    right.map(field).join("") +
    `</div>`;
}

function updateHash() {
  const parts = [`run=${encodeURIComponent(state.run || "")}`, `tab=${state.tab}`];
  if (state.tab === "report" && state.report.file) parts.push(`report=${encodeURIComponent(state.report.file)}`);
  location.hash = `#${parts.join("&")}`;
}

async function activateTab(tab, force = false) {
  if (tab === state.tab && !force) return;
  state.tab = tab;
  setActive("#tabs", "tab", tab);
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${tab}`));
  updateHash();
  if (tab === "metrics") {
    if (!state.metrics.loaded) await initMetrics();
    else if (state.live) await fetchMetrics();
  }
  if (tab === "config" && !state.config.loaded) await initConfig();
  if (tab === "logs" && !state.logs.loaded) await initLogs();
  if (tab === "traces") {
    if (!state.traces.loaded) await initTraces();
    else if (state.live) await refreshTraces();
  }
  if (tab === "report") {
    if (!state.report.loaded) await initReport();
    else if (state.live) await refreshReport();
  }
}

/* ---------------------------------------------------------------- metrics */

const COMMON_METRICS = ["effective/num_turns/mean", "effective/num_total_tokens/mean", "effective/num_branches/mean"];
const COMMON_REGEXES = ["effective/[^/]+/is_truncated/mean", "all/[^/]+/has_error/mean"];
const STABILITY_METRICS = ["optim/grad_norm", "entropy/all/mean", "mismatch_kl/all/mean", "kl_ent_ratio/mean"];
const PERFORMANCE_METRICS = ["perf/mfu", "time/step", "time/wait_for_batch", "time/wait_for_policy"];
const SFT_TRAIN_METRICS = ["loss/mean", "loss/perplexity", "val/loss", "val/perplexity", "progress/epoch"];
const SFT_STABILITY_METRICS = ["optim/grad_norm", "optim/lr", "loss/nan_count"];
const SFT_PERFORMANCE_METRICS = ["perf/mfu", "perf/throughput", "perf/peak_memory", "time/step", "time/forward_backward", "time/save_ckpt"];

// Multi-series inference panels (overview.py INFERENCE_PANELS): fleet aggregate
// paired with the cross-engine tail that flags a single sick engine.
const INFERENCE_PANELS = [
  ["inference/agg/kv_cache_usage_perc/mean", "inference/agg/kv_cache_usage_perc/min", "inference/agg/kv_cache_usage_perc/max"],
  ["inference/agg/num_preemptions_total:rate/sum", "inference/agg/num_preemptions_total:rate/max"],
  ["inference/agg/num_requests_running/mean", "inference/agg/num_requests_running/min", "inference/agg/num_requests_running/max"],
  ["inference/agg/num_requests_waiting/mean", "inference/agg/num_requests_waiting/min", "inference/agg/num_requests_waiting/max"],
  ["inference/agg/prefix_cache_hit_rate/pooled", "inference/agg/prefix_cache_hit_rate/min"],
  ["inference/agg/generation_tokens_total:rate/sum", "inference/agg/generation_tokens_total:rate/min"],
  ["inference/agg/prompt_tokens_total:rate/sum", "inference/agg/prompt_tokens_total:rate/max"],
];

const TRAINER_KEY_RE = /^(perf|optim|loss|entropy|system|mismatch_kl|kl_ent_ratio|is_masked|masked_|unmasked_|max_vio|routing_|ref_kl|val)[/_]?/;
const ORCH_KEY_RE = /^(train|batch|off_policy|curriculum|eval)\//;

function rowProducer(row, meta) {
  if (typeof row.producer === "string") return row.producer; // stamped by FileMonitor
  // legacy files predate the producer stamp: infer from key namespaces
  if (meta?.type === "sft") return "trainer";
  for (const key of Object.keys(row)) {
    if (TRAINER_KEY_RE.test(key)) return "trainer";
    if (ORCH_KEY_RE.test(key)) return "orch";
  }
  return Object.keys(row).some((k) => k.startsWith("progress/")) ? "orch" : "trainer";
}

/* store = {byKey, timeKeys, timeZero} — the primary run's is state.metrics,
   compared runs get their own */
function ingestInto(store, rows, meta) {
  const touched = new Set();
  for (const row of rows) {
    // step=None rows are time-keyed (inference metrics): x = seconds since run start
    const isTime = row.step == null;
    let x;
    if (isTime) {
      const t = row.time ?? row._timestamp;
      if (typeof t !== "number") continue;
      store.timeZero ??= meta?.started ?? t;
      x = Math.max(0, t - store.timeZero);
    } else {
      if (typeof row.step !== "number") continue;
      x = row.step;
      if (store.maxStep == null || x > store.maxStep) store.maxStep = x;
    }
    const producer = isTime ? "infer" : rowProducer(row, meta);
    for (const [key, value] of Object.entries(row)) {
      if (key === "step" || key === "time" || key === "_timestamp" || key === "producer" || typeof value !== "number") continue;
      let producers = store.byKey.get(key);
      if (!producers) store.byKey.set(key, (producers = new Map()));
      let series = producers.get(producer);
      if (!series) producers.set(producer, (series = new Map()));
      series.set(x, value);
      touched.add(key);
      if (isTime) store.timeKeys.add(key);
    }
  }
  return touched;
}

/* the server caps each /metrics response, so a huge run streams in chunks: the
   first charts paint immediately and a progress readout ticks up while the rest
   loads, with the main thread yielding between chunks */
async function fetchMetrics() {
  if (state.meta?.type === "eval") return fetchEvalSeries();
  const m = state.metrics;
  if (m.fetching) return 0;
  m.fetching = true;
  let total = 0;
  let showedProgress = false;
  try {
    for (let first = true; ; first = false) {
      const requestedOffset = m.offset;
      const [data, compared] = await Promise.all([
        api(`/api/runs/${encodeURIComponent(state.run)}/metrics?offset=${m.offset}`),
        first ? fetchCompares() : false,
      ]);
      if (state.metrics !== m) return total; // the run changed mid-load
      m.offset = data.offset;
      total += data.rows.length;
      let touched = null;
      if (data.rows.length) {
        touched = ingestInto(m, data.rows, state.meta);
        renderOverview();
      }
      if (data.rows.length || compared) {
        if (m.byKey.size !== m.renderedKeys) renderMetricsBody();
        else updateCharts(compared ? null : touched); // compares may touch any panel
      }
      // A writer can leave one incomplete JSONL record at EOF. Wait for the
      // next poll instead of repeatedly requesting the same partial record.
      if (data.offset >= (data.size ?? data.offset) || data.offset === requestedOffset) break;
      showedProgress = true;
      $("#metrics-status").textContent = `loading metrics · ${Math.round((data.offset / data.size) * 100)}%`;
    }
    if (showedProgress && m.mode === "overview") $("#metrics-status").textContent = "";
  } finally {
    m.fetching = false;
  }
  return total;
}

/* eval runs have no metrics.jsonl and no step axis — their metrics view is a
   grid of stat cards showing the running average over the episodes so far */
async function fetchEvalSeries() {
  const m = state.metrics;
  let data;
  try {
    const qs = new URLSearchParams({ after: m.evalCount || 0 });
    if (m.evalEtag) qs.set("etag", m.evalEtag);
    data = await api(`/api/runs/${encodeURIComponent(state.run)}/episodes/series?${qs}`);
  } catch {
    return 0;
  }
  if (data.unchanged) return 0;
  m.evalEtag = data.etag;
  // merge the increment: keys new to this batch backfill nulls for earlier episodes
  m.evalSeries ??= {};
  for (const [key, values] of Object.entries(data.series)) {
    const existing = m.evalSeries[key] ?? new Array(data.after).fill(null);
    existing.length = data.after;
    existing.push(...values);
    m.evalSeries[key] = existing;
  }
  m.evalCount = data.count;
  m.maxStep = data.count; // the overview's episode count
  const costs = (m.evalSeries.cost || []).filter((v) => v != null); // merged, not just the increment
  m.evalCost = costs.length ? costs.reduce((a, b) => a + b, 0) : null;
  renderOverview();
  if (m.loaded) renderMetricsBody();
  return data.count;
}

const EVAL_CARD_GROUPS = [
  ["rewards", (k) => k === "reward" || k === "advantage" || k.startsWith("rewards/")],
  ["metrics", (k) => k.startsWith("metrics/")],
  ["usage", (k) => ["cost", "input_tokens", "output_tokens", "turns", "branches"].includes(k)],
  ["timing", (k) => k.startsWith("timing/")],
];

function renderEvalCards(body) {
  const m = state.metrics;
  const series = m.evalSeries || {};
  const filter = makeFilter(m.search.trim());
  const total = state.meta?.total_episodes;
  const done = m.evalCount || 0;
  if (total) {
    const pct = Math.min(100, (done / total) * 100);
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="eval-progress"><div class="ep-bar"><div class="ep-fill" style="width:${pct}%"></div></div>` +
        `<span class="ep-label">${done}/${total} episodes · ${Math.round(pct)}%</span></div>`
    );
  }
  const fmtVal = (key, v) => {
    if (v == null) return "n/a";
    if (key === "cost") return fmtCost(v);
    if (key.startsWith("timing/")) return fmtDuration(v);
    if (key.endsWith("tokens")) return fmtCompact(Math.round(v));
    return fmtNum(v);
  };
  let shown = 0;
  for (const [name, match] of EVAL_CARD_GROUPS) {
    const keys = Object.keys(series)
      .filter(match)
      .filter((k) => !filter || filter.test(k))
      .sort();
    if (!keys.length) continue;
    shown += keys.length;
    const { grid } = addSection(body, name);
    grid.className = "stat-grid";
    grid.innerHTML = keys
      .map((key) => {
        const values = series[key].filter((v) => v != null);
        const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
        const label = /^(rewards|metrics|timing)\//.test(key) ? key.split("/").slice(1).join("/") : key;
        return (
          `<div class="stat-card"><div class="stat-label" title="${esc(key)}">${esc(label)}</div>` +
          `<div class="stat-value">${fmtVal(key, avg)}</div></div>`
        );
      })
      .join("");
  }
  $("#metrics-status").textContent = total || !m.evalCount ? "" : `running avg over ${m.evalCount} episodes`;
  if (!shown) body.innerHTML = emptyState("no episodes yet", "metrics appear as episodes land");
}

async function fetchCompares() {
  const results = await Promise.all(
    state.compare.runs.map(async (name) => {
      let store = state.compare.data.get(name);
      if (!store) {
        store = { offset: 0, byKey: new Map(), timeKeys: new Set(), timeZero: null, maxStep: null, meta: null };
        state.compare.data.set(name, store);
      }
      try {
        store.meta ??= await api(`/api/runs/${encodeURIComponent(name)}`);
        const data = await api(`/api/runs/${encodeURIComponent(name)}/metrics?offset=${store.offset}`);
        store.offset = data.offset;
        if (data.rows.length) {
          ingestInto(store, data.rows, store.meta);
          return true;
        }
      } catch (err) {
        console.warn(`compare fetch failed for ${name}`, err);
      }
      return false;
    })
  );
  return results.some(Boolean);
}

function buildSections(meta) {
  // panel order: reward (effective, then all) -> turns/tokens/branches -> truncation/error
  const trainSection = (name, scope) => ({
    name,
    panels: [
      // one banded plot per agent, not a multi-color overlay
      { regex: `${escRe(scope)}/effective/[^/]+/reward/mean`, split: true },
      { regex: `${escRe(scope)}/all/[^/]+/reward/mean`, split: true },
      ...COMMON_METRICS.map((m) => ({ metric: `${scope}/${m}` })),
      ...COMMON_REGEXES.map((r) => ({ regex: `${escRe(scope)}/${r}` })),
    ],
  });
  const evalSection = (name, envPattern, configured = false) => ({
    name,
    configured,
    panels: [
      { regex: `eval/${envPattern}/all/[^/]+/avg@.*` },
      { regex: `eval/${envPattern}/effective/[^/]+/avg@.*` },
      { regex: `eval/${envPattern}/effective/[^/]+/reward/mean`, split: true },
      { regex: `eval/${envPattern}/all/[^/]+/reward/mean`, split: true },
      { regex: `eval/${envPattern}/all/cancelled/mean` },
      ...COMMON_METRICS.map((m) => ({ regex: `eval/${envPattern}/${m}` })),
      ...COMMON_REGEXES.map((r) => ({ regex: `eval/${envPattern}/${r}` })),
    ],
  });
  const sections = [];
  const evalEnvs = meta.eval_envs || [];
  if (meta.type === "sft") {
    const trainMetrics = meta.has_validation
      ? SFT_TRAIN_METRICS
      : SFT_TRAIN_METRICS.filter((m) => !m.startsWith("val/"));
    sections.push({ name: "train", panels: trainMetrics.map((m) => ({ metric: m })) });
    if (evalEnvs.length) sections.push(...evalEnvs.map((e) => evalSection(`eval/${e}`, escRe(e), true)));
    else sections.push(evalSection("eval", ".*"));
    sections.push({ name: "stability", panels: SFT_STABILITY_METRICS.map((m) => ({ metric: m })) });
    sections.push({ name: "performance", panels: SFT_PERFORMANCE_METRICS.map((m) => ({ metric: m })) });
    return sections;
  }
  const trainEnvs = meta.train_envs || [];
  if (trainEnvs.length === 1) sections.push(trainSection(`train/${trainEnvs[0]}`, `train/${trainEnvs[0]}`));
  else if (trainEnvs.length > 1) {
    sections.push(trainSection("train/agg", "train/agg"));
    sections.push(...trainEnvs.map((e) => trainSection(`train/${e}`, `train/${e}`)));
  } else sections.push(trainSection("train", "train/agg"));
  if (evalEnvs.length) sections.push(...evalEnvs.map((e) => evalSection(`eval/${e}`, escRe(e), true)));
  else sections.push(evalSection("eval", ".*"));
  sections.push({ name: "stability", panels: STABILITY_METRICS.map((m) => ({ metric: m })) });
  sections.push({ name: "inference", panels: INFERENCE_PANELS.map((metrics) => ({ metrics })) });
  sections.push({ name: "performance", panels: PERFORMANCE_METRICS.map((m) => ({ metric: m })) });
  return sections;
}

let activeFilter = null;

function compareStores() {
  const stores = [{ run: state.run, store: state.metrics }];
  for (const name of state.compare.runs) {
    const store = state.compare.data.get(name);
    if (store) stores.push({ run: name, store });
  }
  return stores;
}

/* split panels fan a regex out into one card per matched key */
function splitPanelKeys(panel) {
  const re = new RegExp(`^(?:${panel.regex})$`);
  const keys = new Set();
  for (const { store } of compareStores())
    for (const key of store.byKey.keys()) if (re.test(key) && (!activeFilter || activeFilter.test(key))) keys.add(key);
  return [...keys].sort();
}

function resolvePanel(panel) {
  const series = [];
  for (const { run, store } of compareStores()) {
    let keys;
    if (panel.metric) keys = store.byKey.has(panel.metric) ? [panel.metric] : [];
    else if (panel.metrics) keys = panel.metrics.filter((k) => store.byKey.has(k));
    else {
      const re = new RegExp(`^(?:${panel.regex})$`);
      keys = [...store.byKey.keys()].filter((k) => re.test(k)).sort();
    }
    if (activeFilter) keys = keys.filter((k) => activeFilter.test(k));
    for (const key of keys)
      for (const [producer, points] of store.byKey.get(key))
        series.push({ key, producer, points, run, time: store.timeKeys.has(key) });
  }
  return series;
}

/* series in one panel group by (run, key-minus-stat) into a single color: the
   mean draws solid, p10/p90 (or min/max) draw dashed with a shaded band, other
   stats draw dashed, and a second producer of the same key draws dashed too */
const STAT_SUFFIXES = new Set(["mean", "median", "min", "max", "p10", "p50", "p90", "p99", "sum", "pooled", "std"]);
const MAIN_STAT_PRIORITY = ["mean", "sum", "pooled"];

function statOf(key) {
  const suffix = key.split("/").at(-1);
  return STAT_SUFFIXES.has(suffix) ? suffix : null;
}

function familyOf(key) {
  return statOf(key) ? key.split("/").slice(0, -1).join("/") : key;
}

function panelGroups(seriesList) {
  const groups = new Map();
  for (const s of seriesList) {
    const groupKey = `${s.run}|${familyOf(s.key)}`;
    if (!groups.has(groupKey)) groups.set(groupKey, { run: s.run, strands: new Map() });
    const group = groups.get(groupKey);
    const producer = s.producer ?? "";
    if (!group.strands.has(producer)) group.strands.set(producer, []);
    group.strands.get(producer).push(s);
  }
  return [...groups.values()].map((group) => ({
    run: group.run,
    strands: [...group.strands.values()].map((members) => {
      const byStat = new Map(members.map((s) => [statOf(s.key), s]));
      const main = MAIN_STAT_PRIORITY.map((p) => byStat.get(p)).find(Boolean) ?? members[0];
      let lo = null;
      let hi = null;
      if (byStat.has("p10") && byStat.has("p90")) [lo, hi] = [byStat.get("p10"), byStat.get("p90")];
      else if (byStat.has("min") && byStat.has("max") && !["min", "max"].includes(statOf(main.key)))
        [lo, hi] = [byStat.get("min"), byStat.get("max")];
      return { main, lo, hi, overlays: members.filter((s) => s !== main && s !== lo && s !== hi) };
    }),
  }));
}

function groupColors(groups) {
  if (state.compare.runs.length) {
    const runs = [state.run, ...state.compare.runs];
    return groups.map((g) => PALETTE[Math.max(0, runs.indexOf(g.run)) % PALETTE.length]);
  }
  return groups.length > 1 ? groups.map((_, i) => PALETTE[i % PALETTE.length]) : [SINGLE_SERIES];
}

function singletonPoints(series, color) {
  return {
    show: () => series.points.size === 1,
    size: 6,
    width: 0,
    fill: color,
  };
}

/* flatten groups into uPlot series defs + data columns + tooltip meta */
function buildChartLayout(entry, timeAxis) {
  const groups = panelGroups(entry.series);
  const colors = groupColors(groups);
  const mains = [];
  groups.forEach((g, gi) => g.strands.forEach((strand) => mains.push({ strand, color: colors[gi] })));
  const labels = seriesLabels(mains.map((m) => m.strand.main));
  const cols = []; // parallel to uPlot series[1..]: {s, role: ghost|main|aux}
  const uSeries = [{ label: timeAxis ? "time" : "step" }];
  const bands = [];
  const meta = [];
  let mainIdx = 0;
  for (const [gi, group] of groups.entries()) {
    const color = colors[gi];
    for (const [si, strand] of group.strands.entries()) {
      cols.push({ s: strand.main, role: "ghost" });
      uSeries.push({ stroke: hexToRgba(color, 0.25), width: 1, spanGaps: true, points: { show: false } });
      cols.push({ s: strand.main, role: "main" });
      uSeries.push({
        label: labels[mainIdx] || "value",
        stroke: color,
        width: 1.25,
        dash: si > 0 ? [6, 4] : undefined,
        spanGaps: true,
        points: singletonPoints(strand.main, color),
      });
      const m = { label: labels[mainIdx] || "value", stat: statOf(strand.main.key) ?? "value", color, dataIdx: cols.length };
      meta.push(m);
      mainIdx++;
      const aux = [strand.lo, strand.hi, ...strand.overlays].filter(Boolean);
      const bandIdx = {};
      for (const s of aux) {
        cols.push({ s, role: "aux" });
        const auxColor = hexToRgba(color, 0.55);
        uSeries.push({
          stroke: auxColor,
          width: 1,
          dash: [3, 3],
          spanGaps: true,
          points: singletonPoints(s, auxColor),
        });
        if (s === strand.lo) bandIdx.lo = cols.length;
        if (s === strand.hi) bandIdx.hi = cols.length;
      }
      if (bandIdx.lo && bandIdx.hi) {
        bands.push({ series: [bandIdx.hi, bandIdx.lo], fill: hexToRgba(color, 0.09) });
        // the tooltip spells out lo/mean/hi so the band is interpretable
        m.lo = { dataIdx: bandIdx.lo, stat: statOf(strand.lo.key) };
        m.hi = { dataIdx: bandIdx.hi, stat: statOf(strand.hi.key) };
      }
    }
  }
  return { cols, uSeries, bands, meta };
}

function layoutData(layout) {
  const stepSet = new Set();
  for (const c of layout.cols) for (const step of c.s.points.keys()) stepSet.add(step);
  const steps = [...stepSet].sort((a, b) => a - b);
  const window = state.metrics.smooth;
  const out = [steps];
  for (const c of layout.cols) {
    const raw = steps.map((st) => c.s.points.get(st) ?? null);
    if (c.role === "ghost") out.push(window > 1 ? raw : raw.map(() => null));
    else out.push(rollingMean(raw, window));
  }
  return out;
}

function seriesLabels(series) {
  const comparing = state.compare.runs.length > 0;
  if (series.length === 1 && !comparing) return [""];
  const keyParts = series.map((s) => s.key.split("/"));
  let start = 0;
  while (keyParts.every((p) => p.length > start + 1 && p[start] === keyParts[0][start])) start++;
  return series.map((s, i) => {
    const tail = keyParts[i].slice(start).join("/");
    const dupKey = series.some((o, j) => j !== i && o.key === s.key && o.run === s.run);
    const label = dupKey ? `${tail} (${s.producer})` : tail;
    if (!comparing) return label;
    const runName = s.run.length > 24 ? `${s.run.slice(0, 22)}…` : s.run;
    const multiKey = series.some((o) => o.key !== s.key);
    return multiKey ? `${runName} · ${label}` : runName;
  });
}


function rollingMean(values, window) {
  if (window <= 1) return values;
  const out = new Array(values.length);
  const inWindow = []; // non-null {i, v} pairs, head-trimmed as the window slides
  let head = 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v != null) {
      inWindow.push({ i, v });
      sum += v;
    }
    while (head < inWindow.length && inWindow[head].i <= i - window) {
      sum -= inWindow[head].v;
      head++;
    }
    out[i] = v == null ? null : sum / (inWindow.length - head);
  }
  return out;
}


function chartHeight() {
  return state.metrics.paneH;
}

function chartWidth(card) {
  return card.clientWidth - 22;
}

/* axis ticks, at most ~5 chars: 1e5 · 1e4 · 1000 · 100 · 10 · 1 · 0.3 · 0.01 ·
   0.001 · 1e-4 — decimals only below 10, exponent form outside [0.001, 10000) */
function fmtAxis(v) {
  if (v == null) return "";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e4 || abs < 1e-3) {
    const exp = Math.floor(Math.log10(abs));
    const mant = +(v / 10 ** exp).toFixed(1);
    return `${mant}e${exp}`;
  }
  if (abs >= 10) return String(Math.round(v));
  return String(+v.toPrecision(2));
}

function fmtTickDur(secs) {
  if (secs == null) return "";
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${+(secs / 60).toFixed(secs < 600 ? 1 : 0)}m`;
  if (secs < 86400) return `${+(secs / 3600).toFixed(secs < 36000 ? 1 : 0)}h`;
  return `${+(secs / 86400).toFixed(1)}d`;
}

function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/* a plain click (no drag) resets a drag-zoomed chart */
function unzoomPlugin() {
  let downX = 0;
  let downY = 0;
  return {
    hooks: {
      ready: (u) => {
        u.over.addEventListener("mousedown", (e) => {
          downX = e.clientX;
          downY = e.clientY;
        });
        u.over.addEventListener("click", (e) => {
          if (Math.abs(e.clientX - downX) > 3 || Math.abs(e.clientY - downY) > 3) return;
          u.setData(u.data); // re-autoscales
        });
      },
    },
  };
}

/* hover popover with the x value and every series' y value */
function tooltipPlugin(meta, timeAxis) {
  let tip;
  let dots;
  return {
    hooks: {
      init: (u) => {
        tip = document.createElement("div");
        tip.className = "u-tip";
        tip.style.display = "none";
        u.over.appendChild(tip);
        dots = meta.map((m) => {
          const dot = document.createElement("div");
          dot.className = "u-dot";
          dot.style.background = m.color;
          dot.style.display = "none";
          u.over.appendChild(dot);
          return dot;
        });
        u.over.addEventListener("mouseleave", () => {
          tip.style.display = "none";
          for (const dot of dots) dot.style.display = "none";
        });
      },
      setCursor: (u) => {
        const { left, idx } = u.cursor;
        if (idx == null || left == null || left < 0) {
          tip.style.display = "none";
          for (const dot of dots) dot.style.display = "none";
          return;
        }
        const x = u.data[0][idx];
        let rows = `<div class="u-tip-x">${timeAxis ? fmtTickDur(x) : `step ${x}`}</div>`;
        let any = false;
        meta.forEach((m) => {
          const v = u.data[m.dataIdx][idx]; // the strand's main (smoothed) series
          if (v == null) return;
          any = true;
          const lo = m.lo ? u.data[m.lo.dataIdx][idx] : null;
          const hi = m.hi ? u.data[m.hi.dataIdx][idx] : null;
          const row = (swatch, label, value) =>
            `<div class="u-tip-row"><span class="sw${swatch ? "" : " sw-band"}" style="${
              swatch ? `background:${m.color}` : `border-color:${m.color}`
            }"></span>` +
            `${label ? `<span class="u-tip-l">${esc(label)}</span>` : ""}<span class="u-tip-v">${fmtNum(value)}</span></div>`;
          if (lo != null && hi != null) {
            // banded: three lines, hi over mean over lo
            rows += row(false, m.hi.stat, hi);
            rows += row(true, meta.length > 1 ? m.label : m.stat, v);
            rows += row(false, m.lo.stat, lo);
          } else {
            rows += row(true, meta.length > 1 ? m.label : "", v);
          }
        });
        if (!any) {
          tip.style.display = "none";
          return;
        }
        tip.innerHTML = rows;
        tip.style.display = "block";
        // anchor to the snapped data point (nearest x), not the mouse cursor,
        // with a highlight dot on each series' point
        const xPos = u.valToPos(x, "x");
        let yPos = null;
        meta.forEach((m, i) => {
          const v = u.data[m.dataIdx][idx];
          if (v == null) {
            dots[i].style.display = "none";
            return;
          }
          const py = u.valToPos(v, "y");
          if (yPos == null) yPos = py;
          dots[i].style.display = "block";
          dots[i].style.transform = `translate(${Math.round(xPos - 3)}px, ${Math.round(py - 3)}px)`;
        });
        if (yPos == null) yPos = u.over.clientHeight / 2;
        let tx = xPos + 12;
        if (tx + tip.offsetWidth > u.over.clientWidth) tx = xPos - tip.offsetWidth - 12;
        const ty = Math.max(0, Math.min(u.over.clientHeight - tip.offsetHeight, yPos - tip.offsetHeight / 2));
        tip.style.transform = `translate(${Math.round(tx)}px, ${Math.round(ty)}px)`;
      },
    },
  };
}

function makeChart(el, layout, width, timeAxis = false) {
  const axis = {
    stroke: "#767676",
    grid: { stroke: "rgba(255,255,255,0.06)", width: 1 },
    ticks: { stroke: "rgba(255,255,255,0.10)" },
    font: "10px 'ABC Favorit Mono', 'JetBrains Mono', ui-monospace, monospace",
  };
  const xAxis = timeAxis
    ? {
        ...axis,
        size: 28,
        values: (u, vals) => vals.map(fmtTickDur),
        incrs: [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 43200, 86400, 172800],
      }
    : // step axis: integer ticks only
      { ...axis, size: 28, incrs: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] };
  return new uPlot(
    {
      width,
      height: chartHeight(),
      padding: [10, 12, 0, 0],
      cursor: { points: { show: false }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [xAxis, { ...axis, size: 54, values: (u, vals) => vals.map(fmtAxis) }],
      legend: { show: false },
      bands: layout.bands,
      plugins: [tooltipPlugin(layout.meta, timeAxis), unzoomPlugin()],
      series: layout.uSeries,
    },
    [[], ...layout.cols.map(() => [])],
    el
  );
}

/* Charts below the fold mount only when scrolled into view — the all-metrics
   mode renders hundreds of panels. */
let lazyObserver = null;

function mountChart(entry) {
  if (entry.u) return;
  const plotEl = document.createElement("div");
  entry.card.appendChild(plotEl);
  const timeAxis = entry.series.every((s) => s.time);
  entry.layout = buildChartLayout(entry, timeAxis);
  entry.u = makeChart(plotEl, entry.layout, chartWidth(entry.card), timeAxis);
  updateChart(entry);
}

/* panel titles show the matched key(s), never the regex; the section's scope
   prefix is dropped since the section header already carries it */
function panelTitle(panel, series, sectionName) {
  const keys = [...new Set(series.map((s) => s.key))];
  let title;
  if (!keys.length) title = panel.metric || panel.metrics?.[0] || panel.regex || "";
  else if (keys.length === 1) title = keys[0];
  else {
    const parts = keys[0].split("/");
    while (parts.length && !keys.every((k) => `${k}/`.startsWith(`${parts.join("/")}/`))) parts.pop();
    title = parts.join("/") || keys[0];
  }
  if (sectionName && title.startsWith(`${sectionName}/`)) title = title.slice(sectionName.length + 1);
  // overview: a lone mean is implied - all mode keeps the stat next to its /min//p10 siblings
  if (state.metrics.mode === "overview" && title.endsWith("/mean")) title = title.slice(0, -"/mean".length);
  return title;
}

let dragCard = null;

function renderPanelCard(grid, panel, lazy = false) {
  const series = resolvePanel(panel);
  if (!series.length && (panel.regex || panel.metrics || activeFilter)) return; // no matches (yet)
  const card = document.createElement("div");
  card.className = "chart-card";
  const sectionName = grid.parentElement?.dataset?.name;
  const title = panelTitle(panel, series, sectionName);
  card.dataset.title = title;
  card.innerHTML =
    `<div class="chart-head" draggable="true"><div class="chart-title" title="${esc(title)}">${esc(title)}</div><div class="chart-last"></div></div>` +
    `<div class="rz rz-e" data-rz="x"></div><div class="rz rz-s" data-rz="y"></div>` +
    `<div class="rz rz-se" data-rz="xy" title="drag to resize all panes"></div>`;
  grid.appendChild(card);
  const head = card.querySelector(".chart-head");
  head.addEventListener("dragstart", (e) => {
    dragCard = card;
    e.dataTransfer.effectAllowed = "move";
    card.classList.add("dragging");
  });
  head.addEventListener("dragend", () => {
    card.classList.remove("dragging");
    persistPaneOrder(card.parentElement);
    dragCard = null;
  });
  const entry = { card, u: null, series };
  state.metrics.charts.push(entry);
  if (!series.length) {
    card.insertAdjacentHTML("beforeend", `<div class="chart-empty" style="height:${chartHeight()}px">no data yet</div>`);
    return;
  }
  if (lazy) {
    card.style.minHeight = `${chartHeight() + 40}px`;
    card.__entry = entry;
    lazyObserver.observe(card);
  } else {
    mountChart(entry);
  }
}

function paneOrderKey(sectionName) {
  return `${state.metrics.mode}:${sectionName ?? ""}`;
}

function persistPaneOrder(grid) {
  const sectionName = grid?.closest("details.section")?.dataset?.name;
  if (!sectionName) return;
  state.metrics.paneOrder[paneOrderKey(sectionName)] = [...grid.querySelectorAll(".chart-card")].map(
    (c) => c.dataset.title
  );
  savePrefs();
}

function applyPaneOrder(grid) {
  const sectionName = grid.closest("details.section")?.dataset?.name;
  const saved = sectionName && state.metrics.paneOrder[paneOrderKey(sectionName)];
  if (!saved) return;
  const rank = new Map(saved.map((t, i) => [t, i]));
  [...grid.children]
    .sort((a, b) => (rank.get(a.dataset.title) ?? 1e9) - (rank.get(b.dataset.title) ?? 1e9))
    .forEach((c) => grid.appendChild(c));
}

function updateChart(entry) {
  if (!entry.u) return;
  entry.u.setData(layoutData(entry.layout));
  const last = entry.series[0]?.points;
  if (last?.size) {
    let maxX = -Infinity;
    for (const x of last.keys()) if (x > maxX) maxX = x;
    entry.card.querySelector(".chart-last").textContent = fmtNum(last.get(maxX));
  }
}

function updateCharts(touched = null) {
  for (const entry of state.metrics.charts) {
    if (touched && !entry.series.some((s) => touched.has(s.key))) continue;
    updateChart(entry);
  }
}

function addSection(body, name, count, display = name) {
  const div = document.createElement("details");
  div.className = "section";
  div.dataset.name = name;
  // an active search auto-expands sections so hits are visible; the persisted
  // collapse state comes back when the query clears
  div.open = activeFilter ? true : !state.metrics.collapsedSections.has(name);
  div.innerHTML =
    `<summary>${esc(display)}${count != null ? ` <span class="muted">${count}</span>` : ""}` +
    `<span class="sec-chev">›</span></summary>`;
  const grid = document.createElement("div");
  grid.className = "chart-grid";
  div.appendChild(grid);
  body.appendChild(div);
  return { div, grid };
}

/* all-mode: fully recursive sections along family path segments (train → agg →
   all → agent → …). Every logged key gets its own pane — stats are never
   overlaid, so min/p10/median/... show as raw separate plots. */
function renderKeyTree(parent, name, families, depth) {
  const display = depth === 1 ? name : name.split("/").pop();
  const { div, grid } = addSection(parent, name, families.length, display);
  const children = new Map();
  const leaves = [];
  for (const f of families) {
    const segments = f.family.split("/");
    if (segments.length <= depth + 1) leaves.push(f);
    else {
      const segment = segments[depth];
      if (!children.has(segment)) children.set(segment, []);
      children.get(segment).push(f);
    }
  }
  for (const f of leaves) for (const key of f.keys) renderPanelCard(grid, { metric: key }, true);
  if (grid.children.length) applyPaneOrder(grid);
  else grid.remove();
  for (const [segment, childFamilies] of children) renderKeyTree(div, `${name}/${segment}`, childFamilies, depth + 1);
}

function renderMetricsBody() {
  const m = state.metrics;
  for (const entry of m.charts) entry.u?.destroy();
  m.charts = [];
  m.renderedKeys = m.byKey.size;
  const body = $("#metrics-body");
  body.innerHTML = "";
  lazyObserver?.disconnect();
  lazyObserver = new IntersectionObserver(
    (entries) => {
      for (const en of entries)
        if (en.isIntersecting) {
          lazyObserver.unobserve(en.target);
          mountChart(en.target.__entry);
        }
    },
    { root: body, rootMargin: "400px" }
  );
  if (state.meta?.type === "eval") return renderEvalCards(body);
  activeFilter = makeFilter(state.metrics.search.trim());
  if (!state.meta?.has_metrics && !m.byKey.size) {
    body.innerHTML = emptyState("no metrics yet");
    $("#metrics-status").textContent = "";
    return;
  }
  if (m.mode === "overview") {
    $("#metrics-status").textContent = "";
    for (const section of buildSections(state.meta)) {
      const { div, grid } = addSection(body, section.name);
      for (const panel of section.panels) {
        if (panel.split) for (const key of splitPanelKeys(panel)) renderPanelCard(grid, { metric: key });
        else renderPanelCard(grid, panel);
      }
      if (!grid.children.length) {
        // a configured eval env stays visible before its first eval fires
        if (section.configured && !activeFilter) grid.innerHTML = `<div class="chart-empty">no data yet</div>`;
        else div.remove();
      } else applyPaneOrder(grid);
    }
    if (activeFilter && !body.children.length)
      body.innerHTML = emptyState("no keys match", "no overview panels match the filter");
    return;
  }
  // all: one card per metric family - flat lists a section per family with a
  // pane per stat, nested groups families recursively along path segments
  const familyKeys = new Map();
  let shown = 0;
  for (const key of [...m.byKey.keys()].sort()) {
    if (activeFilter && !activeFilter.test(key)) continue;
    shown++;
    const family = familyOf(key);
    if (!familyKeys.has(family)) familyKeys.set(family, []);
    familyKeys.get(family).push(key);
  }
  $("#metrics-status").textContent = activeFilter ? `${shown} / ${m.byKey.size} keys` : "";
  if (!familyKeys.size) {
    body.innerHTML = emptyState("no keys match", `0 of ${m.byKey.size} keys match the filter`);
    return;
  }
  if (m.allLayout === "flat") {
    for (const [family, keys] of familyKeys) {
      const { div, grid } = addSection(body, family, keys.length);
      for (const key of keys) renderPanelCard(grid, { metric: key }, true);
      if (!grid.children.length) div.remove();
      else applyPaneOrder(grid);
    }
    return;
  }
  const groups = new Map();
  for (const [family, keys] of familyKeys) {
    const group = family.split("/")[0];
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ family, keys });
  }
  for (const [group, families] of groups) renderKeyTree(body, group, families, 1);
}

async function initMetrics() {
  state.metrics.loaded = true;
  await fetchMetrics();
  renderMetricsBody();
}

/* ----------------------------------------------------------------- config */


/* config artifacts are fetched once per attempt, so format changes never wait
   on the network */
async function fetchConfigText(file) {
  const cache = state.config.cache;
  const key = `${state.config.attempt}:${file}`;
  if (cache.has(key)) return cache.get(key);
  const data = await api(
    `/api/runs/${encodeURIComponent(state.run)}/config?file=${encodeURIComponent(file)}` +
    `&attempt=${encodeURIComponent(state.config.attempt)}`
  );
  let text = data.content;
  try {
    text = JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    /* show raw content if not valid JSON */
  }
  cache.set(key, text);
  return text;
}

async function loadConfig() {
  state.config.text = await fetchConfigText(state.config.file);
  applyConfigSearch();
}

/* prune the config to matching subtrees: a matching key keeps its whole
   subtree, a nested match keeps its ancestors for context */
function pruneConfig(node, re) {
  if (typeof node !== "object" || node === null) {
    return re.test(String(node)) ? node : undefined;
  }
  if (Array.isArray(node)) {
    const kept = node.map((v) => pruneConfig(v, re)).filter((v) => v !== undefined);
    return kept.length ? kept : undefined;
  }
  const out = {};
  for (const [key, value] of Object.entries(node)) {
    if (re.test(key)) {
      out[key] = value;
      continue;
    }
    const kept = pruneConfig(value, re);
    if (kept !== undefined) out[key] = kept;
  }
  return Object.keys(out).length ? out : undefined;
}

/* collapsible JSON tree: object/array lines fold on click */
function jsonLeafHtml(value) {
  if (typeof value === "string") return `<span class="j-str">${esc(JSON.stringify(value))}</span>`;
  if (typeof value === "number") return `<span class="j-num">${value}</span>`;
  return `<span class="j-lit">${value === null ? "null" : value}</span>`;
}

function renderJsonNode(key, value, indent, isLast) {
  const pad = "  ".repeat(indent);
  const keyHtml = key != null ? `<span class="j-key">${esc(JSON.stringify(key))}</span><span class="j-punc">: </span>` : "";
  const comma = isLast ? "" : ",";
  if (value !== null && typeof value === "object") {
    const isArray = Array.isArray(value);
    const entries = isArray ? value.map((v) => [null, v]) : Object.entries(value);
    const close = isArray ? "]" : "}";
    if (!entries.length)
      return `<div class="j-line">${pad}${keyHtml}<span class="j-punc">${isArray ? "[]" : "{}"}${comma}</span></div>`;
    const children = entries
      .map(([k, v], i) => renderJsonNode(k, v, indent + 1, i === entries.length - 1))
      .join("");
    return (
      `<details class="j-fold" open><summary class="j-line">${pad}${keyHtml}<span class="j-punc">${isArray ? "[" : "{"}</span>` +
      `<span class="j-ellip"> … ${entries.length} <span class="j-punc">${close}${comma}</span></span></summary>` +
      children +
      `<div class="j-line">${pad}<span class="j-punc">${close}${comma}</span></div></details>`
    );
  }
  return `<div class="j-line">${pad}${keyHtml}${jsonLeafHtml(value)}<span class="j-punc">${comma}</span></div>`;
}

/* minimal TOML highlighting in the same palette as the JSON tree */
function tomlValueHtml(value) {
  const re = /("(?:[^"\\]|\\.)*"|'[^']*')|(#.*$)|(\btrue\b|\bfalse\b)|(-?\b\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?\b)/g;
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(value))) {
    out += esc(value.slice(last, m.index));
    const cls = m[1] ? "j-str" : m[2] ? "j-punc" : m[3] ? "j-lit" : "j-num";
    out += `<span class="${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + esc(value.slice(last));
}

function tomlLineHtml(line) {
  if (/^\s*#/.test(line)) return `<span class="j-punc">${esc(line)}</span>`;
  if (/^\s*\[/.test(line)) return `<span class="t-section">${esc(line)}</span>`;
  const kv = line.match(/^(\s*[\w."'-]+\s*)=([\s\S]*)$/);
  if (kv) return `<span class="j-key">${esc(kv[1])}</span><span class="j-punc">=</span>${tomlValueHtml(kv[2])}`;
  return tomlValueHtml(line);
}

function renderToml(text) {
  return text
    .split("\n")
    .map((line) => `<div class="t-line">${tomlLineHtml(line)}</div>`)
    .join("");
}

/* filter the config to matched subtrees, render the collapsible tree, then
   mark every hit by walking text nodes (keeps syntax spans intact) */
function applyConfigSearch() {
  const view = $("#config-view");
  const query = $("#config-search").value.trim();
  const hitsEl = $("#config-hits");
  hitsEl.textContent = "";
  let parsed;
  try {
    parsed = JSON.parse(state.config.text ?? "");
  } catch {
    parsed = undefined; // not valid JSON: plain highlighted text, no folding
  }
  if (!query) {
    if (parsed !== undefined) view.innerHTML = renderJsonNode(null, parsed, 0, true);
    else view.innerHTML = renderToml(state.config.text ?? "");
    return;
  }
  let re;
  try {
    re = new RegExp(query, "gi");
  } catch {
    re = new RegExp(escRe(query), "gi");
  }
  const test = (line) => {
    re.lastIndex = 0;
    return re.test(line);
  };
  const noHits = () => {
    view.innerHTML = emptyState("no hits", "nothing in this config matches the filter");
    hitsEl.textContent = "no hits";
  };
  if (parsed !== undefined) {
    const pruned = pruneConfig(parsed, new RegExp(re.source, "i"));
    if (pruned === undefined) return noHits();
    view.innerHTML = renderJsonNode(null, pruned, 0, true);
  } else {
    // TOML (launch config): a matching line keeps itself (plus its [section]
    // header for context); a matching [section] header keeps the whole section
    const lines = (state.config.text ?? "").split("\n");
    const kept = [];
    let header = null;
    let headerKept = false;
    let sectionMatched = false;
    for (const line of lines) {
      if (/^\s*\[/.test(line)) {
        header = line;
        headerKept = false;
        sectionMatched = test(line);
        if (sectionMatched) {
          kept.push(line);
          headerKept = true;
        }
        continue;
      }
      if (sectionMatched || test(line)) {
        if (header !== null && !headerKept) {
          kept.push(header);
          headerKept = true;
        }
        kept.push(line);
      }
    }
    if (!kept.length) return noHits();
    view.innerHTML = renderToml(kept.join("\n"));
  }
  const walker = document.createTreeWalker(view, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  let hits = 0;
  for (const node of nodes) {
    const text = node.nodeValue;
    re.lastIndex = 0;
    if (!re.test(text)) continue;
    re.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let last = 0;
    let match;
    while ((match = re.exec(text))) {
      if (match[0] === "") {
        re.lastIndex++;
        continue;
      }
      frag.append(text.slice(last, match.index));
      const mark = document.createElement("mark");
      mark.className = "hit";
      mark.textContent = match[0];
      frag.append(mark);
      hits++;
      last = match.index + match[0].length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
  hitsEl.textContent = hits ? `${hits} hit${hits === 1 ? "" : "s"}` : "no hits";
  view.querySelector("mark.hit")?.scrollIntoView({ block: "center" });
}

/* TOML = launch config, JSON = resolved dumps */
function configFileFor(fmt) {
  const files = state.config.files || [];
  if (fmt === "toml") return files.find((f) => f.endsWith(".toml"));
  if (fmt === "json") return files.find((f) => f === "resolved");
  return files.find((f) => f === "command.txt");
}

function renderConfigFormat() {
  for (const btn of document.querySelectorAll("#config-format button")) {
    btn.disabled = !configFileFor(btn.dataset.fmt);
    btn.classList.toggle("active", btn.dataset.fmt === state.config.fmt);
  }
}

function renderConfigAttempts() {
  const config = state.config;
  const latest = config.latestAttempt == null ? "latest" : `latest (attempt ${config.latestAttempt})`;
  $("#config-attempt-select").innerHTML =
    `<option value="latest" ${config.attempt === "latest" ? "selected" : ""}>${latest}</option>` +
    config.attempts
      .map((a) => `<option value="${a}" ${String(a) === String(config.attempt) ? "selected" : ""}>attempt ${a}</option>`)
      .join("");
  syncDressedSelects();
}

function renderConfigCommand() {
  const command = state.config.commandText.trimEnd();
  $("#config-command").classList.toggle("empty", !command);
  $("#config-command-text").textContent = command || "command unavailable for this attempt";
  $("#config-command-copy").disabled = !command;
}

async function loadConfigAttempt() {
  const data = await api(
    `/api/runs/${encodeURIComponent(state.run)}/configs?attempt=${encodeURIComponent(state.config.attempt)}`
  );
  state.config.latestAttempt = state.config.attempt === "latest" ? data.attempt : state.config.latestAttempt;
  state.config.attempts = data.attempts;
  state.config.files = data.files;
  renderConfigAttempts();
  const commandFile = configFileFor("command");
  state.config.commandText = commandFile ? await fetchConfigText(commandFile) : "";
  renderConfigCommand();
  if (!configFileFor("toml") && !configFileFor("json")) {
    renderConfigFormat();
    $("#config-view").innerHTML = emptyState("no configs", "this attempt has no config files");
    return;
  }
  if (!configFileFor(state.config.fmt)) {
    state.config.fmt = ["toml", "json"].find((fmt) => configFileFor(fmt));
  }
  state.config.file = configFileFor(state.config.fmt);
  renderConfigFormat();
  await loadConfig();
  for (const fmt of ["toml", "json"]) {
    const file = configFileFor(fmt);
    if (file && file !== state.config.file) fetchConfigText(file);
  }
}

async function initConfig() {
  state.config.loaded = true;
  await loadConfigAttempt();
}

/* ------------------------------------------------------------------- logs */

const ANSI_RE = /\x1b\[[0-9;]*m/g;
const TEE_RE = /^\[[A-Za-z]+\d*\]:\s?/;
const LEVEL_RANK = { DEBUG: 0, INFO: 1, SUCCESS: 1, WARNING: 2, ERROR: 3, CRITICAL: 3 };
const TAIL_BYTES = 262144;

function ansiToHtml(raw) {
  let out = "", bold = false, dim = false, fg = null, idx = 0, match;
  const re = /\x1b\[([0-9;]*)m/g;
  const flush = (text) => {
    if (!text) return;
    const cls = [bold && "a-bold", dim && "a-dim", fg && `fg${fg}`].filter(Boolean).join(" ");
    out += cls ? `<span class="${cls}">${esc(text)}</span>` : esc(text);
  };
  while ((match = re.exec(raw))) {
    flush(raw.slice(idx, match.index));
    idx = re.lastIndex;
    for (const code of (match[1] || "0").split(";").map(Number)) {
      if (code === 0) { bold = dim = false; fg = null; }
      else if (code === 1) bold = true;
      else if (code === 2) dim = true;
      else if (code === 22) bold = dim = false;
      else if ((code >= 30 && code <= 37) || (code >= 90 && code <= 97)) fg = code;
      else if (code === 39) fg = null;
    }
  }
  flush(raw.slice(idx));
  return out;
}

function parseLines(text, file) {
  const entries = [];
  for (const rawLine of text.split("\n")) {
    if (rawLine === "") continue;
    const raw = rawLine.replace(TEE_RE, "");
    const plain = raw.replace(ANSI_RE, "");
    const timeMatch = plain.match(/^(?:\d{4}-\d{2}-\d{2} )?(\d\d):(\d\d):(\d\d)\b/);
    const levelMatch = plain.match(/^(?:\d{4}-\d{2}-\d{2} )?\d\d:\d\d:\d\d\s+(DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\b/);
    entries.push({
      rawTime: timeMatch ? +timeMatch[1] * 3600 + +timeMatch[2] * 60 + +timeMatch[3] : null,
      ownLevel: levelMatch ? levelMatch[1] : null,
      router: plain.includes("vllm_router_rs"),
      raw,
      plain,
      html: null,
      component: file.component,
      gseq: state.logs.gseq++,
      t: 0,
      level: null,
    });
  }
  return entries;
}

function retime(buffer, from = 0) {
  const s = from > 0 && buffer.tstate ? buffer.tstate : { lastSec: null, dayOffset: 0, lastLevel: null };
  for (let i = from; i < buffer.lines.length; i++) {
    const line = buffer.lines[i];
    if (line.rawTime != null) {
      if (s.lastSec != null && line.rawTime < s.lastSec - 12 * 3600) s.dayOffset++;
      s.lastSec = line.rawTime;
    }
    line.t = s.dayOffset * 86400 + (line.rawTime ?? s.lastSec ?? 0);
    if (line.ownLevel) s.lastLevel = line.ownLevel;
    line.level = line.ownLevel ?? s.lastLevel; // continuation lines (tracebacks, noise) inherit
  }
  buffer.tstate = s;
}

async function fetchLogChunk(file, params) {
  const qs = new URLSearchParams({ file: file.id, ...params });
  return api(`/api/runs/${encodeURIComponent(state.run)}/log?${qs}`);
}

const LOG_PANES = [
  { comp: "trainer", title: "trainer", match: (f) => f.component === "trainer" },
  { comp: "orch", title: "orchestrator", match: (f) => f.component === "orch" },
  { comp: "infer", title: "inference", match: (f) => f.component === "infer" },
  { comp: "evals", title: "evals", match: (f) => f.component === "evals" },
  { comp: "envs", title: "envs", match: (f) => f.component.startsWith("env:"), merged: true },
];

function paneFiles(pane) {
  return state.logs.files.filter(pane.match);
}

function paneSelectedIds(pane) {
  if (pane.merged) return paneFiles(pane).map((f) => f.id);
  const id = state.logs.paneFile[pane.comp];
  if (pane.comp === "infer" && id === "__router__") {
    // virtual selection: router lines live inside the single-node inference.log
    const master = paneFiles(pane).find((f) => f.master);
    return master ? [master.id] : [];
  }
  return id ? [id] : [];
}

function allSelectedIds() {
  const ids = new Set();
  for (const pane of enabledPanes()) for (const id of paneSelectedIds(pane)) ids.add(id);
  return ids;
}

function enabledPanes() {
  return LOG_PANES.filter((p) => paneFiles(p).length && (state.logs.components?.has(p.comp) ?? true));
}

function renderLogCompMenu() {
  const available = LOG_PANES.filter((p) => paneFiles(p).length);
  const menu = $("#log-comp-menu");
  menu.innerHTML = available
    .map(
      (p) =>
        `<label class="file-item"><input type="checkbox" data-comp="${p.comp}"` +
        `${state.logs.components?.has(p.comp) ?? true ? " checked" : ""}><span>${p.title}</span></label>`
    )
    .join("");
  const enabled = enabledPanes().length;
  $("#log-comp-btn").textContent = enabled === available.length ? "components" : `components (${enabled})`;
  $("#log-comp-btn").classList.toggle("active", enabled !== available.length);
}

function dressLogPaneSelects() {
  for (const select of dressedSelects) if (!select.isConnected) dressedSelects.delete(select);
  document.querySelectorAll("#log-panes select.lp-file").forEach(dressSelect);
}

function renderLogPanes() {
  const logs = state.logs;
  const latest = logs.latestAttempt == null ? "latest" : `latest (attempt ${logs.latestAttempt})`;
  $("#attempt-select").innerHTML =
    `<option value="latest" ${logs.attempt === "latest" ? "selected" : ""}>${latest}</option>` +
    logs.attempts
      .map((a) => `<option value="${a}" ${String(a) === String(logs.attempt) ? "selected" : ""}>attempt ${a}</option>`)
      .join("");
  renderLogCompMenu();
  const container = $("#log-panes");
  container.innerHTML = "";
  const panes = enabledPanes();
  // pick default files for every enabled component (used by both views)
  for (const pane of panes) {
    const files = paneFiles(pane);
    const isVirtual = pane.comp === "infer" && logs.paneFile[pane.comp] === "__router__";
    if (!pane.merged && !isVirtual && !files.some((f) => f.id === logs.paneFile[pane.comp]))
      logs.paneFile[pane.comp] = (files.find((f) => f.master) ?? files[0]).id;
  }
  if (!panes.length) {
    container.classList.remove("maxed");
    container.innerHTML = emptyState("no log files", "this run has no logs or every component is toggled off");
    return;
  }
  if (logs.view === "merge") {
    container.classList.remove("maxed");
    const el = document.createElement("div");
    el.className = "log-pane merged";
    el.dataset.comp = "__merged__";
    el.innerHTML =
      `<div class="log-pane-head"><span class="lp-title">${panes.map((p) => p.title).join(" \u00b7 ")}</span>` +
      `<span class="lp-count"></span><div class="spacer"></div></div>` +
      `<div class="log-pane-stream"></div>`;
    container.appendChild(el);
    return;
  }
  container.classList.toggle("maxed", !!logs.maximized);
  for (const pane of panes) {
    const files = paneFiles(pane);
    const el = document.createElement("div");
    el.className = `log-pane${logs.maximized === pane.comp ? " maximized" : ""}`;
    el.dataset.comp = pane.comp;
    el.innerHTML =
      `<div class="log-pane-head"><span class="lp-title">${pane.title}</span>` +
      (pane.merged
        ? `<span class="lp-count muted">${files.length} file${files.length === 1 ? "" : "s"} merged</span>`
        : `<select class="lp-file">${files
            .map((f) => `<option value="${esc(f.id)}"${f.id === logs.paneFile[pane.comp] ? " selected" : ""}>${esc(f.label)}</option>`)
            .join("")}${
            pane.comp === "infer" && files.some((f) => f.master)
              ? `<option value="__router__"${logs.paneFile.infer === "__router__" ? " selected" : ""}>router</option>`
              : ""
          }</select>`) +
      `<span class="lp-count"></span><div class="spacer"></div>` +
      `<button class="btn lp-max" title="${logs.maximized === pane.comp ? "restore" : "maximize"}">${logs.maximized === pane.comp ? "\u2921" : "\u2922"}</button></div>` +
      `<div class="log-pane-stream"></div>`;
    container.appendChild(el);
  }
}

/* filtered lines for one component, honoring its file selection and the
   engine/router split of the single-node inference.log */
function paneLines(pane, minRank, filter) {
  const ids = paneSelectedIds(pane);
  const selected = state.logs.paneFile[pane.comp];
  const routerOnly = pane.comp === "infer" && selected === "__router__";
  const engineOnly = pane.comp === "infer" && !routerOnly && /(^|\/)inference\.log$/.test(selected ?? "");
  const lines = [];
  for (const id of ids) {
    const buffer = state.logs.buffers.get(id);
    if (!buffer) continue;
    for (const line of buffer.lines) {
      if (engineOnly && line.router) continue;
      if (routerOnly && !line.router) continue;
      if (minRank && (LEVEL_RANK[line.level] ?? minRank) < minRank) continue;
      if (filter && !filter.test(line.plain)) continue;
      lines.push(line);
    }
  }
  return lines;
}

function compName(component) {
  if (component.startsWith("env:")) return component.slice(4);
  return LOG_PANES.find((p) => p.comp === component)?.title ?? component;
}

function renderLogPane(el) {
  const stream = el.querySelector(".log-pane-stream");
  const minRank = LEVEL_RANK[state.logs.level] ?? 0;
  const filter = makeFilter($("#log-search").value.trim());
  let lines;
  let multi;
  if (el.dataset.comp === "__merged__") {
    lines = enabledPanes().flatMap((pane) => paneLines(pane, minRank, filter));
    multi = true;
  } else {
    const pane = LOG_PANES.find((p) => p.comp === el.dataset.comp);
    if (!pane) return;
    lines = paneLines(pane, minRank, filter);
    multi = paneSelectedIds(pane).length > 1;
  }
  if (multi) lines.sort((a, b) => a.t - b.t || a.gseq - b.gseq);
  const shown = lines.slice(-3000);
  const badge = el.dataset.comp === "__merged__"; // merge view: minimal component prefix
  // always follow: stick to the bottom unless the user scrolled up to read
  const pinned = !stream.childElementCount || stream.scrollTop + stream.clientHeight >= stream.scrollHeight - 40;
  stream.innerHTML = shown
    .map(
      (line) =>
        `<div class="ll">${badge ? `<span class="lgb"><span class="lgb-badge">${esc(compName(line.component))}</span></span>` : ""}` +
        `<span class="ltext">${(line.html ??= ansiToHtml(line.raw))}</span></div>`
    )
    .join("");
  el.querySelector(".lp-count").textContent = `${fmtCompact(lines.length)} lines`;
  if (pinned) stream.scrollTop = stream.scrollHeight;
}

function renderAllLogPanes() {
  document.querySelectorAll("#log-panes .log-pane").forEach(renderLogPane);
}

async function pollLogs(render = true) {
  const logs = state.logs;
  const changed = new Set();
  await Promise.all(
    [...allSelectedIds()].map(async (id) => {
      const file = logs.files.find((f) => f.id === id);
      if (!file) return;
      let buffer = logs.buffers.get(id);
      if (!buffer) {
        const chunk = await fetchLogChunk(file, { tail: TAIL_BYTES });
        buffer = { file, lines: parseLines(chunk.text, file), headStart: chunk.start, end: chunk.end };
        retime(buffer);
        logs.buffers.set(id, buffer);
        changed.add(id);
      } else {
        const chunk = await fetchLogChunk(file, { start: buffer.end });
        if (chunk.end > buffer.end) {
          const appended = parseLines(chunk.text, file);
          buffer.lines.push(...appended);
          buffer.end = chunk.end;
          if (buffer.lines.length > 20000) {
            buffer.lines.splice(0, buffer.lines.length - 20000);
            buffer.headStart = Math.max(buffer.headStart, 1); // older data no longer contiguous
          }
          retime(buffer, buffer.lines.length - appended.length);
          changed.add(id);
        }
      }
    })
  );
  if (changed.size && render) renderChangedLogPanes(changed);
}

/* only re-render the panes whose files actually grew */
function renderChangedLogPanes(ids) {
  document.querySelectorAll("#log-panes .log-pane").forEach((el) => {
    const pane = LOG_PANES.find((p) => p.comp === el.dataset.comp);
    const paneIds = el.dataset.comp === "__merged__" ? [...allSelectedIds()] : pane ? paneSelectedIds(pane) : [];
    if (paneIds.some((id) => ids.has(id))) renderLogPane(el);
  });
}

async function loadOlder() {
  await Promise.all(
    [...state.logs.buffers.values()].map(async (buffer) => {
      if (buffer.headStart <= 0) return;
      const start = Math.max(0, buffer.headStart - TAIL_BYTES);
      const chunk = await fetchLogChunk(buffer.file, { start, end: buffer.headStart });
      buffer.lines.unshift(...parseLines(chunk.text, buffer.file));
      buffer.headStart = start;
      retime(buffer);
    })
  );
  renderAllLogPanes();
}

async function loadLogfiles() {
  const logs = state.logs;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/logfiles?attempt=${logs.attempt}`);
  logs.attempts = data.attempts;
  if (logs.attempt === "latest") logs.latestAttempt = data.attempt;
  logs.files = data.files;
  renderLogPanes();
  dressLogPaneSelects();
}

async function initLogs() {
  state.logs.loaded = true;
  await loadLogfiles();
  await pollLogs();
}

/* ----------------------------------------------------------------- traces */

async function loadRollouts() {
  const traces = state.traces;
  const previousTarget = traces.steps.at(-1);
  const wasFollowing = traces.step == null || traces.step === previousTarget?.step;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/rollouts`);
  traces.steps = data.steps;
  const target = data.steps.at(-1);
  // Follow new work while the user is on the latest preferred step. Keep a
  // manually selected historical step stable, especially while its modal is open.
  if (target && wasFollowing && $("#trace-modal").hidden) traces.step = target.step;
  adjustKindSubset();
  renderStepControl();
}

function stepInfo(step) {
  return state.traces.steps.find((s) => s.step === step);
}

function adjustKindSubset() {
  const traces = state.traces;
  const steps = traces.steps.map((s) => s.step);
  // a step is only meaningful once a cohort shipped at it
  if (traces.mode === "step" && !steps.includes(traces.step)) traces.step = steps.at(-1) ?? null;
}

function renderStepControl() {
  const traces = state.traces;
  const steps = traces.steps;
  const idx = steps.findIndex((s) => s.step === traces.step);
  // over thousands of steps one block per step floods the DOM — cap the bar and
  // let each block stand for a bucket of steps (click selects the bucket's last)
  const perCell = Math.max(1, Math.ceil(steps.length / 240));
  const cells = [];
  for (let b = 0; b * perCell < steps.length; b++) {
    const slice = steps.slice(b * perCell, (b + 1) * perCell);
    const hasEval = slice.some((s) => s.available["eval/all"] || s.available["eval/effective"]);
    const last = b * perCell + slice.length - 1;
    const title = slice.length === 1 ? `step ${slice[0].step}` : `steps ${slice[0].step}–${slice[slice.length - 1].step}`;
    cells.push(
      `<span class="sb-cell${idx >= 0 && b * perCell <= idx ? " on" : ""}${hasEval ? " eval" : ""}" data-i="${last}"` +
        ` title="${title}${hasEval ? " · eval" : ""}"></span>`
    );
  }
  if (!cells.length) cells.push(`<span class="sb-cell placeholder"></span>`);
  const signature = cells.join("");
  if (signature === traces.stepControlSignature) return;
  traces.stepControlSignature = signature;
  $("#step-blocks").innerHTML = signature;
  $("#step-prev").disabled = idx <= 0;
  $("#step-next").disabled = idx < 0 || idx >= steps.length - 1;
  const info = stepInfo(traces.step);
  const hasEval = info && (info.available["eval/all"] || info.available["eval/effective"]);
  $("#step-label").innerHTML =
    traces.step == null
      ? `<span class="muted">no steps yet</span>`
      : `step ${traces.step}${steps.length > 1 ? `<span class="muted">/${steps[steps.length - 1].step}</span>` : ""}` +
        (hasEval ? ' <span class="eval-dot" title="eval rollouts"></span>' : "");
}

function selectStepByIndex(index) {
  const step = state.traces.steps[index];
  if (!step || step.step === state.traces.step) return;
  state.traces.step = step.step;
  adjustKindSubset();
  renderStepControl();
  loadEpisodes();
}

// the table chrome stays in place; the message renders as a spanning row so
// arriving traces cause no layout shift
function showTraceEmpty(title, detail) {
  $("#episode-table tbody").innerHTML = `<tr class="empty"><td colspan="12">${emptyState(title, detail)}</td></tr>`;
}

const PAGE = 128;

/* both kinds on means no filter; exactly one narrows to it. Turning both off would
   only ever show nothing, so the last one on stays on. */
function activeKind() {
  const { train, eval: ev } = state.traces.kinds;
  return train && ev ? "" : train ? "train" : ev ? "eval" : "";
}

function traceSort() {
  return state.traces.sorts[state.traces.mode] ?? "arrival:desc";
}

function traceQuery(extra = {}) {
  const t = state.traces;
  const [sort, order] = traceSort().split(":");
  const qs = new URLSearchParams({ sort, order, errors_only: t.errorsOnly });
  const kind = activeKind();
  if (kind) qs.set("kind", kind);
  if (t.env) qs.set("env", t.env);
  if (t.mode === "step" && t.step != null) qs.set("step", t.step);
  if (t.bin) {
    qs.set("start", t.bin[0]);
    qs.set("end", t.bin[1]);
  }
  for (const [key, value] of Object.entries(extra)) qs.set(key, value);
  return qs;
}

function traceKey() {
  const t = state.traces;
  return JSON.stringify([state.run, t.mode, t.step, activeKind(), t.env, t.errorsOnly, traceSort(), t.bin]);
}

/* the table holds one page at a time and grows as the reader scrolls, so a run of
   any length costs the same to open */
async function loadEpisodes({ append = false } = {}) {
  const traces = state.traces;
  syncTraceFilterControls();
  if (traces.mode === "step" && traces.step == null) {
    $("#trace-status").textContent = "";
    showTraceEmpty("no shipped cohorts yet", "no batch has shipped, so no episode ties to a step");
    return;
  }
  const key = traceKey();
  const fresh = !append || traces.key !== key;
  const offset = fresh ? 0 : traces.episodes.length;
  const qs = traceQuery({ offset, limit: PAGE });
  if (!fresh && offset >= traces.total) return;
  // etag = the stream size the client last saw: an unchanged run answers a poll
  // with {unchanged} instead of a page
  if (fresh && traces.key === key && traces.etag && !append) qs.set("etag", traces.etag);
  let data;
  try {
    data = await api(`/api/runs/${encodeURIComponent(state.run)}/episodes?${qs}`);
  } catch {
    $("#trace-status").textContent = "";
    showTraceEmpty("no traces", "this run has no episode stream yet");
    return;
  }
  if (data.unchanged) return;
  traces.etag = data.etag;
  traces.key = key;
  traces.total = data.total;
  traces.runKinds = data.kinds;
  traces.episodes = fresh ? data.episodes : traces.episodes.concat(data.episodes);
  const currentEnv = traces.env;
  for (const sel of ["#trace-env", "#tm-env"])
    $(sel).innerHTML =
      `<option value="">all envs</option>` +
      data.envs.map((e) => `<option value="${esc(e)}" ${e === currentEnv ? "selected" : ""}>${esc(e)}</option>`).join("");
  syncDressedSelects();
  if (!data.total) {
    $("#trace-status").textContent = "";
    showTraceEmpty("no episodes", "nothing matches the current filters");
    return;
  }
  renderEpisodeRows(fresh);
  if (!$("#trace-modal").hidden) renderRolloutWindow();
  const scope = traces.mode === "step" ? `step ${traces.step}` : traces.bin ? "selected bin" : "stream";
  $("#trace-status").textContent = `${data.total} episode${data.total === 1 ? "" : "s"} · ${scope}`;
}

async function loadMoreEpisodes() {
  const traces = state.traces;
  if (traces.paging || traces.episodes.length >= traces.total) return;
  traces.paging = true;
  try {
    await loadEpisodes({ append: true });
  } finally {
    traces.paging = false;
  }
}

/* episodes finished per time bin: the stream's shape, and the thing you click to
   narrow the table to a moment */
async function loadHistogram() {
  const traces = state.traces;
  if (traces.mode !== "stream") return;
  const qs = traceQuery();
  qs.delete("start");
  qs.delete("end");
  qs.delete("sort");
  qs.delete("order");
  let data;
  try {
    data = await api(`/api/runs/${encodeURIComponent(state.run)}/episodes/histogram?${qs}`);
  } catch {
    return;
  }
  traces.hist = data;
  renderHistogram();
}

const HIST_H = 148;
const HIST_BAR_MAX = 46;

function renderHistogram() {
  const data = state.traces.hist;
  const host = $("#trace-hist");
  const width = Math.max(320, host.clientWidth || 900);
  if (!data || !data.bins.length) {
    // an empty run still gets its frame, so the chart does not pop in later
    $("#trace-chart-sub").textContent = "no episodes yet";
    host.innerHTML =
      `<svg width="${width}" height="${HIST_H}" viewBox="0 0 ${width} ${HIST_H}">` +
      `<line class="hgrid" x1="46" y1="${HIST_H - 20}" x2="${width - 8}" y2="${HIST_H - 20}"></line></svg>`;
    return;
  }
  const bins = data.bins;
  const max = Math.max(...bins.map((b) => b[1]), 1);
  // real pixels, and the plot always spans the width; capping the bar itself is
  // what keeps a two-bar series from becoming two slabs
  const padL = 46, padR = 8, padB = 20, padT = 12;
  const plot = width - padL - padR;
  const slot = plot / bins.length;
  const barW = Math.max(1, Math.min(slot - Math.min(3, slot * 0.25), HIST_BAR_MAX));
  const scale = (count) => (count / max) * (HIST_H - padB - padT);
  const selected = state.traces.bin;
  const bars = bins
    .map(([t, count], i) => {
      const h = scale(count);
      const on = selected && t >= selected[0] && t < selected[1];
      return (
        `<rect class="hbar${on ? " on" : ""}" x="${(padL + i * slot + (slot - barW) / 2).toFixed(2)}" ` +
        `y="${(HIST_H - padB - h).toFixed(2)}" width="${barW.toFixed(2)}" height="${Math.max(h, count ? 1 : 0).toFixed(2)}" ` +
        `data-t="${t}" data-count="${count}"></rect>`
      );
    })
    .join("");
  const grid = [0, max / 2, max]
    .map((v) => {
      const y = HIST_H - padB - scale(v);
      return (
        `<line class="hgrid" x1="${padL}" y1="${y.toFixed(2)}" x2="${(padL + plot).toFixed(2)}" y2="${y.toFixed(2)}"></line>` +
        `<text class="hax hval" x="${padL - 8}" y="${(y + 3).toFixed(2)}">${fmtCompact(Math.round(v))}</text>`
      );
    })
    .join("");
  // label a few bars; include the date once the span crosses a day
  const spanS = data.end - data.start || 1;
  const label = (t) => {
    const d = new Date(t * 1000);
    const clock = d.toLocaleTimeString([], spanS < 300 ? { hour: "2-digit", minute: "2-digit", second: "2-digit" } : { hour: "2-digit", minute: "2-digit" });
    return spanS > 86400 ? `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${clock}` : clock;
  };
  const every = Math.max(1, Math.ceil(bins.length / Math.max(2, Math.floor(plot / 110))));
  const ticks = bins
    .map(([t], i) => (i % every === 0 || i === bins.length - 1 ? i : null))
    .filter((i) => i != null)
    .map((i) => {
      const x = padL + i * slot + slot / 2;
      const anchor = i === 0 ? "start" : i === bins.length - 1 ? "end" : "middle";
      return `<text class="hax" style="text-anchor:${anchor}" x="${x.toFixed(2)}" y="${HIST_H - 5}">${label(bins[i][0])}</text>`;
    })
    .join("");
  host.innerHTML =
    `<svg width="${width}" height="${HIST_H}" viewBox="0 0 ${width} ${HIST_H}">${grid}${bars}${ticks}</svg>`;
  const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const span = `${new Date(data.start * 1000).toLocaleDateString([], { month: "short", day: "numeric" })} ${clock(data.start)} → ${clock(data.end)}`;
  const picked = selected
    ? ` · selected ${clock(selected[0])}–${clock(selected[1])} (${fmtBin(data.bin)})`
    : "";
  $("#trace-chart-sub").textContent = `${data.total} episodes · ${fmtBin(data.bin)} bins · ${span}${picked}`;
}

function histTipHtml(start, count, bin) {
  const end = start + bin;
  const day = new Date(start * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
  const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return (
    `<div class="tip-head">${count} episode${count === 1 ? "" : "s"}</div>` +
    `<div class="tip-row"><span>start</span><span>${day} ${clock(start)}</span></div>` +
    `<div class="tip-row"><span>end</span><span>${day} ${clock(end)}</span></div>` +
    `<div class="tip-row"><span>duration</span><span>${fmtBin(bin)}</span></div>`
  );
}

function fmtBin(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function fmtStamp(epoch) {
  const d = new Date(epoch * 1000);
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function episodeRowHtml(ep) {
  return `<tr data-line="${ep.line}">
        <td class="muted">${ep.line}</td>
        <td class="muted nowrap">${ep.arrival ? fmtStamp(ep.arrival) : ""}</td>
        <td class="muted">${ep.duration != null ? fmtDuration(ep.duration) : ""}</td>
        <td class="muted">${esc(ep.kind ?? "")}</td>
        <td>${esc(ep.env ?? "?")}</td>
        <td class="muted" title="${esc(ep.group ?? "")}">${ep.group ? esc(ep.group.slice(0, 8)) : "n/a"}</td>
        <td class="${rewardClass(ep.reward)}">${fmtReward(ep.reward)}</td>
        <td class="${rewardClass(ep.advantage)}">${ep.advantage != null ? fmtReward(ep.advantage) : "n/a"}</td>
        <td>${
          ep.input_tokens != null || ep.output_tokens != null
            ? `<span class="muted">in</span> ${fmtCompact(ep.input_tokens ?? 0)} <span class="muted">· out</span> ${fmtCompact(ep.output_tokens ?? 0)}`
            : ""
        }</td>
        <td>${ep.turns ?? ""}</td>
        <td class="muted">${esc(ep.stop_condition ?? "")}</td>
        <td class="${ep.ok && !ep.num_errors ? "status-ok" : "status-err"}">${ep.ok && !ep.num_errors ? "ok" : `${ep.num_errors || ""} err`}</td>
      </tr>`;
}

/* windowed table: only rows in (and around) the viewport exist in the DOM, spacer
   rows stand in for the rest, and the page itself grows as the reader scrolls */
let episodeRowH = 0;

function renderEpisodeRows(reset = false) {
  const wrap = $("#episode-table-wrap");
  const tbody = $("#episode-table tbody");
  const episodes = state.traces.episodes || [];
  if (reset) {
    wrap.scrollTop = 0;
    episodeRowH = 0;
  }
  if (!episodeRowH) {
    tbody.innerHTML = episodes.length ? episodeRowHtml(episodes[0]) : "";
    episodeRowH = tbody.firstElementChild?.offsetHeight || 28;
  }
  const start = Math.max(0, Math.floor(wrap.scrollTop / episodeRowH) - 20);
  const end = Math.min(episodes.length, start + Math.ceil(wrap.clientHeight / episodeRowH) + 40);
  const pad = (h) => (h > 0 ? `<tr class="vpad"><td colspan="12" style="height:${h}px"></td></tr>` : "");
  tbody.innerHTML =
    pad(start * episodeRowH) +
    episodes.slice(start, end).map(episodeRowHtml).join("") +
    pad((episodes.length - end) * episodeRowH);
}

async function initTraces() {
  state.traces.loaded = true;
  await refreshTraces();
}

async function refreshTraces() {
  const traces = state.traces;
  if (traces.fetching) return;
  traces.fetching = true;
  try {
    await loadRollouts();
    if (state.traces !== traces) return;
    await loadEpisodes();
    if (state.traces !== traces) return;
    await loadHistogram();
  } finally {
    traces.fetching = false;
  }
}

/* ----------------------------------------------------------- episode view */

let currentEpisode = null;
let currentLine = null;
let currentTraceIdx = 0;
let currentBranchIdx = 0;
let episodeOpenVersion = 0;
let episodeEnrichmentVersion = 0;
let currentTimeline = null;
let traceView = prefs.traceView === "timeline" ? "timeline" : "transcript";
let pendingTimelineNode = null;
let pendingTimelineCall = null;

const COPY_SVG =
  `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">` +
  `<rect x="9" y="9" width="12" height="12"></rect><path d="M5 15V5a2 2 0 0 1 2-2h10"></path></svg>`;

function copyText(text, el) {
  navigator.clipboard
    ?.writeText(text)
    .then(() => {
      el.classList.add("copied");
      setTimeout(() => el.classList.remove("copied"), 700);
    })
    .catch(() => {});
}

function filteredRollouts() {
  return state.traces.episodes || [];
}

function tmItemHtml(e) {
  return (
    `<div class="tm-item ${e.line === currentLine ? "active" : ""}${e.num_errors || !e.ok ? " err" : ""}" data-line="${e.line}">` +
    `<span class="tm-num">#${e.line}</span><span class="tm-env muted" title="${esc(e.env ?? "")}">${esc(e.env ?? "")}</span>` +
    `<span class="tm-reward ${rewardClass(e.reward)}">${fmtReward(e.reward)}</span></div>`
  );
}

/* windowed like the episode table — only the visible slice is in the DOM */
let tmItemH = 0;

function renderRolloutWindow() {
  const list = $("#tm-list");
  const episodes = filteredRollouts();
  $("#tm-count").textContent = fmtCompact(state.traces.total || episodes.length);
  if (!episodes.length) {
    list.innerHTML = "";
    return;
  }
  if (!tmItemH) {
    list.innerHTML = tmItemHtml(episodes[0]);
    tmItemH = list.firstElementChild?.offsetHeight || 33;
  }
  const start = Math.max(0, Math.floor(list.scrollTop / tmItemH) - 20);
  const end = Math.min(episodes.length, start + Math.ceil(list.clientHeight / tmItemH) + 40);
  list.innerHTML =
    `<div style="height:${start * tmItemH}px"></div>` +
    episodes.slice(start, end).map(tmItemHtml).join("") +
    `<div style="height:${(episodes.length - end) * tmItemH}px"></div>`;
}

function renderRolloutList() {
  const list = $("#tm-list");
  const episodes = filteredRollouts();
  const activeIdx = episodes.findIndex((e) => e.line === currentLine);
  if (activeIdx >= 0 && tmItemH) {
    const top = activeIdx * tmItemH;
    if (top < list.scrollTop || top + tmItemH > list.scrollTop + list.clientHeight)
      list.scrollTop = Math.max(0, top - list.clientHeight / 2);
  }
  renderRolloutWindow();
}

async function stepRollout(delta) {
  let episodes = filteredRollouts();
  const idx = episodes.findIndex((e) => e.line === currentLine);
  if (idx + delta >= episodes.length - 1) {
    await loadMoreEpisodes();
    episodes = filteredRollouts();
  }
  const next = episodes[idx + delta];
  if (next) openEpisode(next.line);
}

function renderModalStep() {
  const traces = state.traces;
  if (state.meta?.type === "eval") {
    $("#tm-step-label").innerHTML = "";
    return;
  }
  const idx = traces.steps.findIndex((s) => s.step === traces.step);
  const last = traces.steps[traces.steps.length - 1]?.step;
  $("#tm-step-label").textContent = `step ${traces.step}${traces.steps.length > 1 ? `/${last}` : ""}`;
  $("#tm-step-prev").disabled = idx <= 0;
  $("#tm-step-next").disabled = idx < 0 || idx >= traces.steps.length - 1;
}

async function modalStep(delta) {
  const traces = state.traces;
  const idx = traces.steps.findIndex((s) => s.step === traces.step);
  const target = traces.steps[idx + delta];
  if (!target) return;
  traces.step = target.step;
  adjustKindSubset();
  renderStepControl();
  await loadEpisodes();
  renderModalStep();
  const first = filteredRollouts()[0];
  if (first) {
    openEpisode(first.line);
  } else {
    currentLine = null;
    currentEpisode = null;
    renderRolloutList();
    $("#tm-messages").innerHTML = emptyState("no episodes", "this step has no rollouts for the current filters");
    $("#tm-meta").innerHTML = "";
  }
}

function fetchEpisode(line, withTokens, withRendered = false) {
  const params = new URLSearchParams();
  if (withTokens) params.set("tokens", "true");
  if (withRendered) params.set("rendered", "true");
  const qs = params.size ? `?${params}` : "";
  return api(`/api/runs/${encodeURIComponent(state.run)}/episodes/${line}${qs}`);
}

function fetchEpisodeTimeline(line) {
  return api(`/api/runs/${encodeURIComponent(state.run)}/episodes/${line}/timeline`);
}

async function ensureTimeline() {
  if (currentTimeline || currentLine == null) return;
  const line = currentLine;
  const requestVersion = episodeOpenVersion;
  const timeline = await fetchEpisodeTimeline(line);
  if (line !== currentLine || requestVersion !== episodeOpenVersion) return;
  currentTimeline = timeline;
}

/* token strings multiply the payload of a big episode, so they are fetched only
   for token signals or the rendered-token view — the plain view ships the raw record */
async function ensureTokens() {
  if (!currentEpisode) return;
  const wantsPieces = !!$("#token-signal").value;
  const wantsRendered = state.traces.viewMode === "rendered";
  if ((!wantsPieces || currentEpisode._hasTokens) && (!wantsRendered || currentEpisode._hasRendered)) return;
  const line = currentLine;
  const withTokens = wantsPieces || !!currentEpisode._hasTokens;
  const withRendered = wantsRendered || !!currentEpisode._hasRendered;
  const requestVersion = ++episodeEnrichmentVersion;
  const episode = await fetchEpisode(line, withTokens, withRendered);
  if (line !== currentLine || requestVersion !== episodeEnrichmentVersion) return;
  episode._hasTokens = withTokens;
  episode._hasRendered = withRendered;
  currentEpisode = episode;
}

async function openEpisode(line, target = {}) {
  const requestVersion = ++episodeOpenVersion;
  episodeEnrichmentVersion++;
  $("#trace-modal").hidden = false;
  $("#drawer-backdrop").hidden = false;
  currentLine = line;
  currentEpisode = null;
  renderModalStep();
  renderRolloutList();
  $("#tm-messages").innerHTML = `<div class="chart-empty">loading episode…</div>`;
  $("#tm-timeline").innerHTML = `<div class="chart-empty">loading timeline…</div>`;
  $("#tm-meta").innerHTML = "";
  currentTimeline = null;
  pendingTimelineNode = null;
  pendingTimelineCall = null;
  const withTokens = !!$("#token-signal").value;
  const withRendered = state.traces.viewMode === "rendered";
  const episode = await fetchEpisode(line, withTokens, withRendered);
  if (line !== currentLine || requestVersion !== episodeOpenVersion) return;
  episode._hasTokens = withTokens;
  episode._hasRendered = withRendered;
  currentEpisode = episode;
  currentTraceIdx = target.trace ?? 0;
  currentBranchIdx = target.branch ?? 0;
  if (traceView === "timeline") await ensureTimeline();
  renderEpisode();
  await ensureTokens();
  if (line === currentLine && requestVersion === episodeOpenVersion) renderEpisode();
}

function closeDrawer() {
  episodeOpenVersion++;
  episodeEnrichmentVersion++;
  $("#trace-modal").hidden = true;
  $("#drawer-backdrop").hidden = true;
  $("#tm-back").hidden = true;
  currentEpisode = null;
  currentTimeline = null;
  pendingTimelineNode = null;
  pendingTimelineCall = null;
  currentLine = null;
  pendingHighlight = null;
}

function traceBranches(trace) {
  const nodes = trace.nodes || [];
  const hasChild = new Set();
  nodes.forEach((n) => { if ("parent" in n) hasChild.add(n.parent); });
  const leaves = nodes.map((_, i) => i).filter((i) => !hasChild.has(i));
  return leaves.map((leaf) => {
    const path = [];
    for (let i = leaf; i != null; i = "parent" in nodes[i] ? nodes[i].parent : null) path.push(i);
    return path.reverse();
  });
}

/* branch -1 = the concatenated conversation view: every node once, top to bottom
   in write order, so shared prefixes never repeat */
function currentPath(trace, branches) {
  if (currentBranchIdx === -1) return (trace.nodes || []).map((_, i) => i);
  return branches[Math.min(currentBranchIdx, branches.length - 1)] || [];
}

function traceReward(trace) {
  return Object.values(trace.rewards || {}).reduce(
    (acc, r) => acc + (r.score ?? 0) * (r.weight ?? 1), 0
  );
}

function messageText(message) {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content))
    return content.map((part) => (part.type === "text" ? part.text : `[${part.type}]`)).join("");
  return content == null ? "" : JSON.stringify(content);
}

function reasoningText(content) {
  return typeof content === "string" ? content : content == null ? "" : JSON.stringify(content, null, 2);
}

/* logprobs may cover only masked positions; map token index -> logprob index */
function alignedSignal(node, values) {
  const n = (node.token_ids || []).length;
  if (!Array.isArray(values) || !values.length) return () => null;
  if (values.length === n) return (i) => values[i];
  const mask = node.mask || [];
  const index = new Array(n).fill(null);
  let j = 0;
  for (let i = 0; i < n; i++) if (mask[i]) index[i] = j++;
  if (j !== values.length) return () => null;
  return (i) => (index[i] == null ? null : values[index[i]]);
}

/* per-episode normalization constants for the token-signal overlays, plus the
   run's IPO eps for the stable mask (server-stamped on annotated traces) */
function episodeSignalScales(trace) {
  let maxAbsAdv = 0, maxEntropy = 0, maxKl = 0;
  for (const node of trace.nodes || []) {
    for (const a of node.advantages || []) maxAbsAdv = Math.max(maxAbsAdv, Math.abs(a));
    for (const h of node.entropies || []) if (h != null) maxEntropy = Math.max(maxEntropy, h);
    const trainer = node.trainer_logprobs, inference = node.logprobs;
    if (Array.isArray(trainer) && Array.isArray(inference) && trainer.length === inference.length)
      for (let j = 0; j < trainer.length; j++) {
        if (trainer[j] == null || inference[j] == null) continue;
        const dlp = trainer[j] - inference[j];
        maxKl = Math.max(maxKl, Math.exp(dlp) - dlp - 1);
      }
  }
  return { maxAbsAdv, maxEntropy, maxKl };
}

function renderTokenNode(node, signal, scales) {
  const ids = node.token_ids || [];
  const strs = node.token_strs;
  const logprobAt = alignedSignal(node, node.logprobs);
  const advantageAt = alignedSignal(node, node.advantages);
  const trainerLpAt = alignedSignal(node, node.trainer_logprobs);
  const entropyAt = alignedSignal(node, node.entropies);
  const spans = ids.map((id, i) => {
    const text = strs?.[i] ?? ` ${id} `;
    const logprob = logprobAt(i), advantage = advantageAt(i);
    const trainerLp = trainerLpAt(i), entropy = entropyAt(i);
    const dlp = trainerLp != null && logprob != null ? trainerLp - logprob : null;
    const kl = dlp != null ? Math.exp(dlp) - dlp - 1 : null;
    let bg = "";
    if (signal === "advantage" && advantage != null && scales.maxAbsAdv > 0) {
      const alpha = Math.min(1, Math.abs(advantage) / scales.maxAbsAdv) * 0.45;
      bg = `background:rgba(${advantage > 0 ? "182,255,60" : "255,69,57"},${alpha.toFixed(3)})`;
    } else if (signal === "entropy" && entropy != null && scales.maxEntropy > 0) {
      bg = `background:rgba(94,234,212,${(Math.min(1, entropy / scales.maxEntropy) * 0.5).toFixed(3)})`;
    } else if (signal === "mismatch_kl" && kl != null && scales.maxKl > 0) {
      bg = `background:rgba(255,69,57,${(Math.min(1, kl / scales.maxKl) * 0.55).toFixed(3)})`;
    } else if (signal === "mask" && node.mask?.[i]) {
      bg = "background:rgba(74,158,255,0.3)";
    } else if (signal === "is_content" && node.is_content?.[i]) {
      bg = "background:rgba(252,218,164,0.28)";
    }
    let tip = `#${i} id=${id}`;
    if (signal === "advantage" && advantage != null) tip += ` adv=${fmtNum(advantage)}`;
    else if (signal === "entropy" && entropy != null) tip += ` H=${entropy.toFixed(4)} nats`;
    // the mismatch is the whole comparison: both logprobs, their gap, and the k3 estimator
    else if (signal === "mismatch_kl" && kl != null)
      tip += ` trainer=${trainerLp.toFixed(4)} inference=${logprob.toFixed(4)} Δ=${dlp.toFixed(4)} kl=${kl.toFixed(6)}`;
    else if (signal === "mask") tip += ` mask=${node.mask?.[i] ?? "?"}`;
    else if (signal === "is_content") tip += ` content=${node.is_content?.[i] ?? "?"}`;
    return `<span class="tok" style="${bg}" data-tip="${esc(tip)}">${esc(text)}</span>`;
  });
  return spans.join("");
}

/* tool calls render directly as python-style calls, e.g. ipython("!ls -la /app") */
function pyLiteral(value) {
  if (value === null) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") return String(value);
  return JSON.stringify(value);
}

function toolCallHtml(toolCall) {
  const name = toolCall.function?.name ?? toolCall.name ?? "?";
  const raw = toolCall.function?.arguments ?? toolCall.arguments;
  let args;
  try {
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const entries = Object.entries(parsed);
      // one argument reads best positionally: ipython("!ls -la /app")
      args = entries.length === 1 ? pyLiteral(entries[0][1]) : entries.map(([k, v]) => `${k}=${pyLiteral(v)}`).join(", ");
    } else args = pyLiteral(parsed);
  } catch {
    args = String(raw ?? "");
  }
  return `<div class="tool-call">${esc(name)}(${esc(args)})</div>`;
}

function reasoningBlock(content, marks = null) {
  const text = reasoningText(content);
  // a highlight whose quote lives in the reasoning marks it and opens the block
  const marked = (marks || []).some((h) => h.quote && findQuote(text, h.quote, h.prefix, h.suffix));
  return (
    `<details class="sub"${marked ? " open" : ""}><summary><span class="sub-name">Reasoning</span>` +
    `<span class="entry-preview">${preview(text, 140)}</span>` +
    `<span class="entry-chev">›</span></summary>` +
    `<div class="entry-body">${marked ? quoteMarkedHtml(text, marks) : esc(text)}</div></details>`
  );
}

function normalizedTools(tools) {
  if (tools == null) return [];
  return Array.isArray(tools) ? tools : [tools];
}

function toolParts(tool, index) {
  if (!tool || typeof tool !== "object" || Array.isArray(tool))
    return { name: `tool ${index + 1}`, description: "Malformed tool definition", parameters: tool };
  const value = tool.function && typeof tool.function === "object" ? tool.function : tool;
  return {
    name: typeof value.name === "string" && value.name ? value.name : `tool ${index + 1}`,
    description: typeof value.description === "string" ? value.description : "No description recorded.",
    parameters: value.parameters ?? value.input_schema ?? value.schema ?? null,
  };
}

function toolDefinitionsHtml(trace) {
  const tools = normalizedTools(trace.tools);
  if (!tools.length) return "";
  const names = tools.map((tool, i) => toolParts(tool, i).name);
  const body = tools.map((tool, i) => {
    const parts = toolParts(tool, i);
    const schema = parts.parameters == null ? "No parameters/schema recorded." :
      typeof parts.parameters === "string" ? parts.parameters : JSON.stringify(parts.parameters, null, 2);
    return (
      `<details class="tool-definition"><summary><span class="tool-def-name">${esc(parts.name)}</span>` +
      `<span class="entry-preview">${preview(parts.description, 140)}</span>` +
      `<button class="icon-btn" data-copy-tool="${i}" title="copy full tool definition">${COPY_SVG}</button>` +
      `<span class="entry-chev">›</span></summary>` +
      `<div class="tool-description">${esc(parts.description)}</div>` +
      `<div class="schema-head"><span>Parameters / JSON schema</span>` +
      `<button class="icon-btn" data-copy-schema="${i}" title="copy parameters/schema">${COPY_SVG}</button></div>` +
      `<pre class="tool-schema">${esc(schema)}</pre></details>`
    );
  }).join("");
  return (
    `<details class="tool-definitions"><summary><span class="context-label">Tool definitions</span>` +
    `<span class="chip">${tools.length} tool${tools.length === 1 ? "" : "s"}</span>` +
    `<span class="entry-preview">${esc(names.join(", "))}</span>` +
    `<button class="icon-btn" data-copy-tools title="copy all tool definitions">${COPY_SVG}</button>` +
    `<span class="entry-chev">›</span></summary>` +
    body + `</details>`
  );
}

function renderedTokensHtml(trace, branches) {
  const rendered = trace.rendered_tokens;
  const errors = errorBannersHtml(episodeErrors(currentEpisode, trace));
  if (!rendered) return emptyState("rendered text not loaded", "select this view again to load recorded token IDs") + errors;
  const signal = $("#token-signal").value;
  const path = currentPath(trace, branches);
  const tokenCount = path.reduce((count, index) => count + (trace.nodes[index]?.token_ids?.length || 0), 0);
  const unavailable = {
    missing_token_ids: ["no recorded token IDs", "This trace cannot provide post-renderer text because its nodes have no token_ids."],
    missing_model: ["tokenizer model unavailable", "Neither renderer_model_name nor the run model was recorded."],
    tokenizer_unavailable: ["tokenizer unavailable", `Could not load the recorded renderer tokenizer${rendered.model ? ` (${rendered.model})` : ""}. Token IDs remain authoritative.`],
    decode_error: ["recorded tokens could not be decoded", "The tokenizer was found, but it could not decode this recorded sequence."],
  };
  const selected = currentBranchIdx === -1 ? rendered.all_nodes : rendered.paths?.[currentBranchIdx];
  if (signal && tokenCount) {
    const scales = episodeSignalScales(trace);
    const body = path.map((index) => renderTokenNode(trace.nodes[index], signal, scales)).join("");
    return (
      `<details class="rendered-transcript" open><summary><span class="context-label">Rendered tokens/text</span>` +
      `<span class="chip">${fmtCompact(tokenCount)} tokens</span>` +
      (selected?.text != null ? `<span class="entry-preview">${preview(selected.text, 180)}</span>` : `<span class="entry-preview"></span>`) +
      (selected?.text != null ? `<button class="icon-btn" data-copy-rendered="text" title="copy decoded text">${COPY_SVG}</button>` : "") +
      `<button class="icon-btn" data-copy-rendered="ids" title="copy authoritative token IDs">IDs</button>` +
      `<span class="entry-chev">›</span></summary>` +
      `<pre class="rendered-text">${body}</pre></details>` + errors
    );
  }
  if (selected?.text == null) {
    const [title, detail] = unavailable[rendered.status] ?? ["rendered text unavailable", "The recorded token sequence could not be decoded."];
    return emptyState(title, detail) + errors;
  }
  return (
    `<details class="rendered-transcript" open><summary><span class="context-label">Rendered tokens/text</span>` +
    `<span class="chip">${fmtCompact(selected.token_count)} tokens</span>` +
    `<span class="entry-preview">${preview(selected.text, 180)}</span>` +
    `<button class="icon-btn" data-copy-rendered="text" title="copy decoded text">${COPY_SVG}</button>` +
    `<button class="icon-btn" data-copy-rendered="ids" title="copy authoritative token IDs">IDs</button>` +
    `<span class="entry-chev">›</span></summary>` +
    `<pre class="rendered-text">${esc(selected.text)}</pre></details>` + errors
  );
}

let entriesObserver = null;

function episodeErrors(ep, trace) {
  return [...(ep.errors || []), ...(trace?.errors || [])];
}

function errorBannersHtml(errors) {
  if (!errors.length) return "";
  return (
    `<div class="trace-errors">` +
    errors
      .map((error) => {
        const record = error && typeof error === "object" ? error : { message: String(error) };
        const type = record.type ?? "Error";
        const message = record.message ?? "No error message";
        const traceback = Array.isArray(record.traceback) ? record.traceback.join("") : record.traceback;
        return (
          `<section class="trace-error-banner">` +
          `<div class="trace-error-message"><span class="trace-error-type">${esc(type)}</span> ${esc(message)}</div>` +
          (traceback
            ? `<details class="trace-error-tb"><summary><span>traceback</span><span class="entry-chev">›</span></summary><pre>${esc(traceback)}</pre></details>`
            : "") +
          `</section>`
        );
      })
      .join("") +
    `</div>`
  );
}

function normalizedCallUsage(usage = {}) {
  let input = usage.prompt_tokens;
  let cached = usage.cached_input_tokens;
  if (cached == null) {
    cached = usage.prompt_tokens_details?.cached_tokens;
    if (input != null && cached) input = Math.max(0, input - cached);
  }
  return {
    input,
    cached,
    output: usage.completion_tokens,
    reasoning: usage.reasoning_tokens ?? usage.completion_tokens_details?.reasoning_tokens,
    cost: usage.cost,
  };
}

function renderMessages(ep, trace, branches) {
  const container = $("#tm-messages");
  entriesObserver?.disconnect();
  const errorsHtml = errorBannersHtml(episodeErrors(ep, trace));
  if (!trace) {
    container.innerHTML = emptyState("no traces", "this episode carries no trace data") + errorsHtml;
    return;
  }
  if (state.traces.viewMode === "rendered") {
    container.innerHTML = renderedTokensHtml(trace, branches);
    return;
  }
  const signal = $("#token-signal").value;
  const path = currentPath(trace, branches);
  const concatenated = currentBranchIdx === -1;
  const toolsHtml = toolDefinitionsHtml(trace);
  const systemPosition = path.findIndex((idx) => trace.nodes[idx]?.message?.role === "system");
  const scales = episodeSignalScales(trace);
  const indexedCalls = (trace.calls || []).map((call, index) => ({ call, index }));
  const callsByNode = new Map();
  for (const item of indexedCalls) {
    const calls = callsByNode.get(item.call.node) || [];
    calls.push(item);
    callsByNode.set(item.call.node, calls);
  }
  const callChipHtml = ({ call, index }) => {
    const fields = [`call ${index + 1}`];
    if (call.finish_reason) fields.push(call.finish_reason);
    const { input, cached, output } = normalizedCallUsage(call.usage);
    if (input != null) fields.push(`${fmtCompact(input)} in`);
    if (cached != null) fields.push(`${fmtCompact(cached)} cache`);
    if (output != null) fields.push(`${fmtCompact(output)} out`);
    return `<span class="chip" data-call-index="${index}">${esc(fields.join(" · "))}</span>`;
  };
  // agent highlights (sticky until the next view command or drawer close)
  const hl =
    pendingHighlight &&
    pendingHighlight.run === state.run &&
    pendingHighlight.step === state.traces.step &&
    pendingHighlight.kind === activeKind() &&
    pendingHighlight.subset === (state.traces.mode === "step" ? "effective" : "all") &&
    pendingHighlight.line === currentLine &&
    pendingHighlight.trace === currentTraceIdx
      ? pendingHighlight
      : null;
  const hlByNode = new Map();
  for (const h of hl?.highlights || []) {
    if (!hlByNode.has(h.node)) hlByNode.set(h.node, []);
    hlByNode.get(h.node).push(h);
  }
  const entryHtml = (idx, i) => {
    const node = trace.nodes[idx];
    const role = node.message?.role ?? "?";
    const marks = hlByNode.get(idx) || [];
    const chips = [];
    if (concatenated && node.parent != null && node.parent !== idx - 1) chips.push(`↳ branches from ${node.parent + 1}`);
    if (node.sampled) chips.push("sampled");
    const nodeCalls = callsByNode.get(idx) || [];
    if (!nodeCalls.length && node.token_ids?.length) chips.push(`${node.token_ids.length} tok`);
    const text = messageText(node.message);
    const contentMarks = marks.filter((h) => !h.field || h.field === "content");
    const contentMarked = contentMarks.some((h) => h.quote && findQuote(text, h.quote, h.prefix, h.suffix));
    const reasoningMarks = marks.filter((h) => !h.field || h.field === "reasoning");
    const reasoning = node.message?.reasoning_content ?? node.message?.reasoning;
    const reasoningMarked = reasoningMarks.some(
      (h) => h.quote && findQuote(reasoningText(reasoning), h.quote, h.prefix, h.suffix)
    );
    const marked = contentMarked || reasoningMarked;
    const showingTokens = !contentMarked && signal && node.token_ids?.length;
    const body = contentMarked
      ? quoteMarkedHtml(text, contentMarks)
      : showingTokens ? renderTokenNode(node, signal, scales) : esc(text);
    const subs = [];
    // The token spans already cover the reasoning tokens; a separate box would
    // duplicate them and split the sequence the overlay is coloring.
    if (reasoning && !showingTokens) subs.push(reasoningBlock(reasoning, reasoningMarks));
    const toolCalls = (node.message?.tool_calls || []).map(toolCallHtml);
    const messageHtml =
      `<details class="entry ${esc(role)}${marked ? " hl-entry" : ""}" data-node="${idx}"${role === "system" && !marked ? "" : " open"}>` +
      `<summary><span class="entry-num">${String(i + 1).padStart(2, "0")}</span>` +
      `<span class="entry-role">${esc(role)}</span>` +
      `<span class="entry-preview">${preview(text, 180)}</span>` +
      chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("") +
      nodeCalls.map(callChipHtml).join("") +
      `<button class="icon-btn" data-copy="${idx}" title="copy message">${COPY_SVG}</button>` +
      `<span class="entry-chev">›</span></summary>` +
      subs.join("") +
      (body ? `<div class="entry-body">${body}</div>` : "") +
      toolCalls.join("") +
      `</details>`;
    return messageHtml + (i === systemPosition ? toolsHtml : "");
  };
  // long traces render in chunks as the reader scrolls — a 1MB episode with
  // hundreds of turns paints the first screen immediately; a highlight past the
  // first chunk forces enough entries into the DOM to scroll to
  const CHUNK = 30;
  const lastMark = Math.max(-1, ...[...hlByNode.keys()].map((n) => path.indexOf(n)));
  const targetPosition = pendingTimelineNode == null ? -1 : path.indexOf(pendingTimelineNode);
  let rendered = Math.min(path.length, Math.max(CHUNK, lastMark + 3, targetPosition + 1));
  const unlinkedCallsHtml = indexedCalls
    .filter(({ call }) => !Number.isInteger(call.node) || call.node < 0 || call.node >= (trace.nodes || []).length)
    .map(
      (item) =>
        `<details class="entry model-call" data-call-index="${item.index}" open>` +
        `<summary><span class="entry-num">C${String(item.index + 1).padStart(2, "0")}</span>` +
        `<span class="entry-role">model call</span><span class="entry-preview">${esc(item.call.model || "unlinked call")}</span>` +
        `${callChipHtml(item)}<span class="entry-chev">›</span></summary></details>`,
    )
    .join("");
  container.innerHTML =
    (systemPosition === -1 ? toolsHtml : "") +
    path.slice(0, rendered).map(entryHtml).join("") +
    (rendered < path.length ? `<div id="tm-more" class="chart-empty">scroll for ${path.length - rendered} more entries</div>` : "") +
    unlinkedCallsHtml +
    errorsHtml;
  if (hl && !hl.scrolled) {
    const first = container.querySelector(".hl-entry");
    // consume the one-shot flag only when the scroll lands: openEpisode renders
    // twice (enrichment re-render), which detaches the first render's node
    // before its scheduled scroll fires
    if (first)
      requestAnimationFrame(() => {
        if (!first.isConnected) return;
        hl.scrolled = true;
        first.scrollIntoView({ block: "start", behavior: "smooth" });
      });
  }
  if (rendered < path.length) {
    const sentinel = container.querySelector("#tm-more");
    entriesObserver = new IntersectionObserver((hits) => {
      if (!hits.some((h) => h.isIntersecting)) return;
      const next = path.slice(rendered, rendered + CHUNK).map((idx, j) => entryHtml(idx, rendered + j)).join("");
      rendered += CHUNK;
      sentinel.insertAdjacentHTML("beforebegin", next);
      if (rendered >= path.length) {
        entriesObserver.disconnect();
        sentinel.remove();
      } else {
        sentinel.textContent = `scroll for ${path.length - rendered} more entries`;
      }
    });
    entriesObserver.observe(sentinel);
  }
}

function metaSubRow(key, value) {
  if (value == null) return "";
  return (
    `<div class="meta-row"><span class="k"><span class="tree" style="padding-left:12px">└</span> ${esc(key)}</span>` +
    `<span class="v">${esc(value)}</span></div>`
  );
}

function metaRow(key, value, asId = false) {
  if (value == null) return "";
  return (
    `<div class="meta-row"><span class="k">${esc(key)}</span>` +
    `<span class="v${asId ? " id" : ""}" title="${esc(value)}">${esc(value)}</span>` +
    (asId ? `<button class="icon-btn" data-copytext="${esc(value)}" title="copy">${COPY_SVG}</button>` : "") +
    `</div>`
  );
}

const TRUNCATING_STOPS = new Set(["max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens", "context_length"]);

/* mirrors verifiers Trace.is_truncated (not serialized): framework limits or a
   length-finished final response */
function traceTruncated(trace) {
  if (TRUNCATING_STOPS.has(trace.stop_condition)) return true;
  const last = [...(trace.calls || [])].reverse().find((c) => !c.error);
  return !!(last && last.finish_reason === "length");
}

function renderMeta(ep, trace, branches) {
  const parts = [];
  const headline = (label, value) =>
    `<div class="meta-row"><span class="k">${label}</span>` +
    `<span class="tm-reward-big${value != null && value < 0 ? " neg" : ""}${value == null ? " na" : ""}" style="margin-left:auto">` +
    `${value != null ? fmtReward(value) : "n/a"}</span></div>`;
  const rewardValue = trace && Object.keys(trace.rewards || {}).length ? traceReward(trace) : null;
  parts.push(headline("reward", rewardValue));
  parts.push(headline("advantage", trace?.info?.advantage ?? null));

  if (trace) {
    const rewards = Object.entries(trace.rewards || {});
    if (rewards.length) {
      parts.push(`<div class="meta-sec">rewards</div>`);
      for (const [name, r] of rewards) parts.push(metaRow(name, `${fmtReward(r.score)} × ${fmtNum(r.weight ?? 1)}`));
    }
    const metrics = Object.entries(trace.metrics || {});
    if (metrics.length) {
      parts.push(`<div class="meta-sec">metrics</div>`);
      for (const [name, value] of metrics)
        parts.push(metaRow(name, typeof value === "number" ? fmtNum(value) : String(value)));
    }
  }

  if (trace) {
    const nodes = currentPath(trace, branches).map((i) => trace.nodes[i]);
    parts.push(`<div class="meta-sec">activity</div>`);
    parts.push(metaRow("messages", nodes.filter((n) => n.message).length));
    parts.push(metaRow("turns", nodes.filter((n) => n.sampled).length));
    parts.push(metaRow("branches", branches.length));
    parts.push(metaRow("tool calls", nodes.reduce((acc, n) => acc + (n.message?.tool_calls?.length || 0), 0)));

    const usage = { input: null, output: null, reasoning: null, cached: null, maxContext: null, cost: null };
    const addUsage = (field, value) => {
      if (value != null) usage[field] = (usage[field] ?? 0) + value;
    };
    for (const call of trace.calls || []) {
      const current = normalizedCallUsage(call.usage);
      addUsage("input", current.input);
      addUsage("cached", current.cached);
      addUsage("output", current.output);
      addUsage("reasoning", current.reasoning);
      addUsage("cost", current.cost);
      if (current.input != null) {
        const context = current.input + (current.cached ?? 0);
        usage.maxContext = Math.max(usage.maxContext ?? 0, context);
      }
    }
    const totalInput = usage.input == null ? null : usage.input + (usage.cached ?? 0);
    const totalTokens = totalInput == null || usage.output == null ? null : totalInput + usage.output;
    const hasUsage = usage.input != null || usage.cached != null || usage.output != null;
    if (hasUsage) {
      parts.push(`<div class="meta-sec">usage</div>`);
      if (usage.input != null) parts.push(metaRow("input tokens", fmtCompact(usage.input)));
      if (usage.cached != null) parts.push(metaRow("cached input", fmtCompact(usage.cached)));
      if (totalInput != null) parts.push(metaRow("total input", fmtCompact(totalInput)));
      if (usage.output != null) parts.push(metaRow("output tokens", fmtCompact(usage.output)));
      if (usage.reasoning != null) parts.push(metaRow("reasoning tokens", fmtCompact(usage.reasoning)));
      if (usage.maxContext != null) parts.push(metaRow("max context length", fmtCompact(usage.maxContext)));
      if (usage.cost != null) parts.push(metaRow("cost", fmtCost(usage.cost)));
      if (totalTokens != null) parts.push(metaRow("total tokens", fmtCompact(totalTokens)));
    }

    parts.push(`<div class="meta-sec">state</div>`);
    parts.push(metaRow("stop_condition", trace.stop_condition));
    parts.push(metaRow("is_completed", trace.is_completed));
    parts.push(metaRow("is_truncated", traceTruncated(trace)));
    parts.push(metaRow("ok", trace.ok));

    const durations = [];
    (function walkTiming(obj, prefix) {
      if (!obj || typeof obj !== "object") return;
      if (typeof obj.duration === "number") durations.push([prefix, obj.duration]);
      else if (typeof obj.start === "number" && typeof obj.end === "number") durations.push([prefix, obj.end - obj.start]);
      for (const [k, v] of Object.entries(obj)) if (typeof v === "object") walkTiming(v, prefix ? `${prefix}/${k}` : k);
    })(trace.timing, "");
    if (durations.length) {
      parts.push(`<div class="meta-sec">timing</div>`);
      for (const [name, secs] of durations) {
        // nested phases (agent/model, agent/harness) render as a tree under their parent
        const segments = (name || "total").split("/");
        const depth = segments.length - 1;
        const label = depth
          ? `<span class="tree" style="padding-left:${depth * 12}px">└</span> ${esc(segments[segments.length - 1])}`
          : esc(name || "total");
        parts.push(`<div class="meta-row"><span class="k">${label}</span><span class="v">${secs.toFixed(2)}s</span></div>`);
      }
    }

    // a trace has several steps - dispatched at one, arrived at another, shipped at a
    // third - so each is stamped as its own event. The lag between the first and the
    // last is the staleness the policy trained through, split where it accrued:
    // generating the rollout, then waiting in the sink for a batch.
    const info = trace.info || {};
    const annotations = trace.train_annotations;
    const lag = (steps, seconds) =>
      `${steps} step${Math.abs(steps) === 1 ? "" : "s"}` + (isFinite(seconds) ? ` · ${fmtDuration(seconds)}` : "");
    if (info.dispatch || info.arrival || info.ship || annotations) {
      parts.push(`<div class="meta-sec">timeline</div>`);
      if (info.dispatch?.step != null) parts.push(metaRow("dispatch step", info.dispatch.step));
      if (info.arrival?.step != null) parts.push(metaRow("arrival step", info.arrival.step));
      if (info.ship?.step != null) parts.push(metaRow("ship step", info.ship.step));
      if (info.ship?.step != null && info.dispatch?.step != null) {
        parts.push(metaRow("staleness", lag(info.ship.step - info.dispatch.step, info.ship.time - info.dispatch.time)));
        if (info.arrival?.step != null) {
          parts.push(metaSubRow("in flight", lag(info.arrival.step - info.dispatch.step, info.arrival.time - info.dispatch.time)));
          parts.push(metaSubRow("in queue", lag(info.ship.step - info.arrival.step, info.ship.time - info.arrival.time)));
        }
      }
      if (ep.run?.work?.policy)
        parts.push(metaRow("policy span", `v${ep.run.work.policy.start}–v${ep.run.work.policy.end}`));
      if (annotations) parts.push(metaRow("trainer annotations", `${annotations.nodes} nodes`));
    }
  }

  parts.push(`<div class="meta-sec">identity</div>`);
  parts.push(metaRow("episode ID", ep.id, true));
  if (trace?.id) parts.push(metaRow("trace ID", trace.id, true));
  if (ep.group?.id) parts.push(metaRow("group ID", ep.group.id, true));
  if (ep.task?.key) parts.push(metaRow("task ID", ep.task.key, true));
  if (trace?.agent?.runtime?.id) parts.push(metaRow("runtime ID", trace.agent.runtime.id, true));

  $("#tm-meta").innerHTML = parts.join("");
}

function timelineClock(ts) {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function timelineTipAttr(payload) {
  return ` data-timeline-tip="${esc(JSON.stringify(payload))}"`;
}

function appendTimelineUsage(rows, usage, aggregate = false) {
  if (!usage) return;
  if (aggregate && usage.model_calls != null) rows.push(["model calls", fmtNum(usage.model_calls)]);
  if (usage.input_tokens != null) rows.push(["input tokens", fmtCompact(usage.input_tokens)]);
  if (usage.cached_tokens != null) rows.push(["cached input", fmtCompact(usage.cached_tokens)]);
  if (usage.total_input_tokens != null) rows.push(["total input", fmtCompact(usage.total_input_tokens)]);
  if (usage.output_tokens != null) rows.push(["output tokens", fmtCompact(usage.output_tokens)]);
  if (usage.reasoning_tokens != null) rows.push(["reasoning tokens", fmtCompact(usage.reasoning_tokens)]);
  if (usage.max_context_tokens != null) rows.push(["max context length", fmtCompact(usage.max_context_tokens)]);
  if (usage.total_tokens != null) rows.push(["total tokens", fmtCompact(usage.total_tokens)]);
  if (usage.cost != null) rows.push(["cost", fmtCost(usage.cost)]);
}

function timelineSpanHtml(lane, span, start, total) {
  const partial = span.started_at == null || span.ended_at == null;
  const left = span.started_at == null ? 0 : Math.max(0, Math.min(100, ((span.started_at - start) / total) * 100));
  const width = partial ? 0.35 : Math.max(0.35, Math.min(100 - left, ((span.ended_at - span.started_at) / total) * 100));
  const rows = [
    ["start", span.started_at == null ? "unknown" : timelineClock(span.started_at)],
    ["end", span.ended_at == null ? (span.status === "running" ? "open" : "unknown") : timelineClock(span.ended_at)],
    ["duration", partial ? "—" : fmtDuration(span.ended_at - span.started_at)],
  ];
  if (span.track === "activity") {
    if (span.shared) rows.push(["branch role", "shared prefix"]);
    const totalInput = span.input_tokens == null ? null : span.input_tokens + (span.cached_tokens || 0);
    appendTimelineUsage(rows, {
      input_tokens: span.input_tokens,
      cached_tokens: span.cached_tokens,
      total_input_tokens: totalInput,
      output_tokens: span.output_tokens,
      reasoning_tokens: span.reasoning_tokens,
      max_context_tokens: totalInput,
      total_tokens: totalInput == null || span.output_tokens == null ? null : totalInput + span.output_tokens,
      cost: span.cost,
    });
  } else {
    appendTimelineUsage(rows, lane.usage, true);
  }
  const tip = timelineTipAttr({
    kind: span.track === "activity" ? "activity" : "lifecycle",
    title: `${lane.label} — ${span.label}`,
    snippet: span.snippet || "",
    rows,
    hint: span.track === "activity" ? "Click to open this call in the transcript." : "Click to open this trace transcript.",
  });
  const node = span.node_index == null ? "" : ` data-tl-node="${span.node_index}"`;
  const call = span.call_index == null ? "" : ` data-tl-call="${span.call_index}"`;
  return (
    `<button class="tl-span ${esc(span.track)} ${esc(span.kind)} ${span.shared ? "shared" : ""} ${span.status === "running" ? "running" : ""} ${partial ? "untimed" : ""}"` +
    ` style="left:${left.toFixed(3)}%;width:${width.toFixed(3)}%" data-tl-trace="${lane.trace_index}"${node}${call}${tip}></button>`
  );
}

function timelineLaneHtml(lane, start, total) {
  const grids = [25, 50, 75].map((left) => `<i class="tl-gridline" style="left:${left}%"></i>`).join("");
  const spans = (lane.spans || []).map((span) => timelineSpanHtml(lane, span, start, total)).join("");
  const duration = lane.started_at == null || lane.ended_at == null ? "—" : fmtDuration(lane.ended_at - lane.started_at);
  const ended = lane.ended_at == null ? (lane.status === "running" ? "open" : "unknown") : timelineClock(lane.ended_at);
  const model = lane.model
    ? `<div class="tl-label-meta" title="${esc(lane.model)}">${esc(lane.model)}</div>`
    : "";
  return (
    `<div class="tl-lane" data-tl-trace="${lane.trace_index}">` +
    `<div class="tl-label" style="padding-left:${10 + (lane.depth || 0) * 18}px">` +
    `${lane.depth ? '<span class="tl-tree">└</span>' : ""}<span class="tl-dot" style="background:${PALETTE[lane.trace_index % PALETTE.length]}"></span>` +
    `<span class="tl-label-copy"><div class="tl-label-name" title="${esc(lane.label)}">${esc(lane.label)}</div>` +
    `${model}</span></div>` +
    `<div class="tl-track">${grids}${spans}</div>` +
    `<div class="tl-time"><span>${duration}</span><span class="muted">${ended}</span></div>` +
    `<div class="tl-outcome"><span class="tl-state ${esc(lane.status)}">${esc(lane.outcome || lane.status)}</span>` +
    `${lane.reward == null ? "" : `<span class="tl-reward">reward ${fmtReward(lane.reward)}</span>`}</div></div>`
  );
}

function renderTimeline() {
  const target = $("#tm-timeline");
  const timeline = currentTimeline;
  if (!timeline) {
    target.innerHTML = `<div class="chart-empty">loading timeline…</div>`;
    return;
  }
  if (!(timeline.lanes || []).length) {
    target.innerHTML = emptyState("no timeline", "this episode carries no agent traces");
    return;
  }
  const starts = timeline.lanes.map((lane) => lane.started_at).filter((value) => value != null);
  const ends = timeline.lanes.map((lane) => lane.ended_at ?? lane.started_at).filter((value) => value != null);
  const start = starts.length ? Math.min(...starts) : 0;
  const end = ends.length ? Math.max(...ends) : start + 1;
  const total = Math.max(1, end - start);
  const axis = [0, 0.25, 0.5, 0.75, 1]
    .map((fraction) => `<span style="left:${fraction * 100}%">${fraction ? fmtDuration(total * fraction) : "0"}</span>`)
    .join("");
  target.innerHTML =
    `<div class="tl-shell"><div class="tl-head"><span>branches</span><div class="tl-axis">${axis}</div><span>duration / end</span><span>state / outcome</span></div>` +
    timeline.lanes.map((lane) => timelineLaneHtml(lane, start, total)).join("") +
    `</div>`;
}

async function setTraceView(view) {
  traceView = view;
  setActive("#tm-view", "view", view);
  if (view === "timeline") await ensureTimeline();
  renderEpisode();
  savePrefs();
}

function renderEpisode() {
  const ep = currentEpisode;
  if (!ep) return;
  const traces = ep.traces || [];
  if (currentTraceIdx >= traces.length) currentTraceIdx = 0;
  const trace = traces[currentTraceIdx];
  const branches = trace ? traceBranches(trace) : [];
  if (currentBranchIdx >= branches.length) currentBranchIdx = 0;
  const traceTabs = $("#tm-trace-tabs");
  traceTabs.hidden = traces.length <= 1;
  traceTabs.innerHTML =
    traces.length > 1
      ? traces
          .map((trace, i) => `<button data-trace="${i}" class="${i === currentTraceIdx ? "active" : ""}">${esc(trace.agent?.name || "agent")}</button>`)
          .join("")
      : "";
  const branchTabs = $("#tm-branch-tabs");
  branchTabs.hidden = branches.length <= 1;
  branchTabs.innerHTML =
    branches.length > 1
      ? branches
          .map((_, i) => `<button data-branch="${i}" class="${i === currentBranchIdx ? "active" : ""}">branch ${i}</button>`)
          .join("") +
        `<button data-branch="-1" class="${currentBranchIdx === -1 ? "active" : ""}" title="all branches concatenated top to bottom">all</button>`
      : "";
  setActive("#trace-view-mode", "mode", state.traces.viewMode);
  renderRolloutList();
  const timeline = traceView === "timeline";
  $("#tm-tabs-row").hidden = timeline || (traceTabs.hidden && branchTabs.hidden);
  $("#tm-messages").hidden = timeline;
  $("#tm-timeline").hidden = !timeline;
  $("#token-signal").closest(".dd-select").hidden = timeline;
  $("#trace-view-mode").hidden = timeline;
  $("#tm-collapse").hidden = timeline;
  $("#tm-expand").hidden = timeline;
  setActive("#tm-view", "view", traceView);
  if (timeline) renderTimeline();
  else renderMessages(ep, trace, branches);
  renderMeta(ep, trace, branches);
}

/* ----------------------------------------------------------------- report */
/* Markdown from <run>/reports/*.md. Prose is free; claims are
   addressed: [^id] markers reference JSON citation lines, each one a trace
   address plus a verbatim quote the frontend re-checks against the files. */

async function initReport() {
  state.report.loaded = true;
  await refreshReport();
}

async function refreshReport() {
  const rep = state.report;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/reports`);
  if (state.report !== rep) return;
  rep.files = data.reports;
  if (!rep.files.length) {
    rep.file = null;
    rep.text = null;
    renderReportSelect();
    $("#report-verify").textContent = "";
    $("#report-status").textContent = "";
    $("#report-body").innerHTML = emptyState("no reports yet", "markdown files in <run>/reports/ appear here");
    return;
  }
  const wanted = rep.wanted && rep.files.find((f) => f.file === rep.wanted)?.file;
  rep.wanted = null;
  const target = wanted || (rep.file && rep.files.find((f) => f.file === rep.file)?.file) || rep.files[0].file;
  const entry = rep.files.find((f) => f.file === target);
  const changed = target !== rep.file || entry.mtime !== rep.mtime;
  rep.file = target;
  renderReportSelect();
  if (changed) await loadReport();
}

async function loadReport() {
  const rep = state.report;
  const file = rep.file;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/report?file=${encodeURIComponent(file)}`);
  if (state.report !== rep || rep.file !== file) return;
  rep.text = data.text;
  rep.mtime = data.mtime;
  renderReport();
  if (state.tab === "report") updateHash();
}

function renderReportSelect() {
  const rep = state.report;
  const sel = $("#report-select");
  sel.disabled = !rep.files.length;
  sel.innerHTML = rep.files.length
    ? rep.files
        .map((f) => `<option value="${esc(f.file)}" ${f.file === rep.file ? "selected" : ""}>${esc(f.title || f.file)}</option>`)
        .join("")
    : `<option>no reports</option>`;
  syncDressedSelects();
}

function renderReport() {
  const rep = state.report;
  const { title, body, citations } = parseReport(rep.text || "");
  rep.citations = citations;
  rep.verify = new Map();
  rep.lookup = { summaries: new Map(), episodes: new Map() };
  const ctx = { citations, order: [] };
  const html = mdToHtml(body, ctx);
  rep.order = ctx.order;
  $("#report-body").innerHTML =
    (title ? `<h1 class="report-title">${esc(title)}</h1>` : "") +
    (html || emptyState("empty report", "this report has no content yet"));
  const entry = rep.files.find((f) => f.file === rep.file);
  $("#report-status").textContent = entry ? `${rep.file} · ${fmtAgo(entry.mtime)}` : (rep.file ?? "");
  $("#report-verify").textContent = rep.order.length ? "verifying citations…" : "";
  verifyCitations();
}

/* frontmatter title + [^id]: {json} citation definitions, stripped from the body */
function parseReport(text) {
  let title = null;
  let body = text;
  const fm = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (fm) {
    body = text.slice(fm[0].length);
    const t = /^title:\s*(.+)$/m.exec(fm[1]);
    if (t) title = t[1].trim().replace(/^["']|["']$/g, "");
  }
  const citations = {};
  body = body.replace(/^\[\^([\w-]+)\]:[ \t]*(\{.*\})[ \t]*$/gm, (_, id, json) => {
    try {
      citations[id] = JSON.parse(json);
    } catch {
      citations[id] = { _invalid: json };
    }
    return "";
  });
  return { title, body, citations };
}

/* markdown subset renderer (headings, lists, tables, fences, quotes, inline
   marks) over fully escaped text — reports never inject markup */
function mdToHtml(src, ctx) {
  const lines = src.split("\n");
  const out = [];
  let para = [];
  const flush = () => {
    if (para.length) out.push(`<p>${renderInline(para.join(" "), ctx)}</p>`);
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = /^(```|~~~)\s*(\S*)\s*$/.exec(line);
    if (fence) {
      flush();
      const buf = [];
      for (i++; i < lines.length && !lines[i].startsWith(fence[1]); i++) buf.push(lines[i]);
      out.push(`<pre class="md-code"${fence[2] ? ` data-lang="${esc(fence[2])}"` : ""}><code>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(heading[2], ctx)}</h${level}>`);
      continue;
    }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line) && !para.length) {
      out.push("<hr>");
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const buf = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) buf.push(lines[i].replace(/^\s*>\s?/, ""));
      i--;
      out.push(`<blockquote>${mdToHtml(buf.join("\n"), ctx)}</blockquote>`);
      continue;
    }
    if (line.includes("|") && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1] || "") && (lines[i + 1] || "").includes("-")) {
      flush();
      const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
      const head = cells(line);
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes("|"); i++) rows.push(cells(lines[i]));
      i--;
      out.push(
        `<div class="md-table-wrap"><table class="md-table"><thead><tr>` +
          head.map((h) => `<th>${renderInline(h, ctx)}</th>`).join("") +
          `</tr></thead><tbody>` +
          rows.map((r) => `<tr>${head.map((_, j) => `<td>${renderInline(r[j] ?? "", ctx)}</td>`).join("")}</tr>`).join("") +
          `</tbody></table></div>`
      );
      continue;
    }
    const li = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (li) {
      flush();
      const ordered = /\d/.test(li[2][0]);
      const items = [];
      for (; i < lines.length; i++) {
        const m = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
        if (!m) {
          // indented continuation lines fold into the previous item
          if (/^\s{2,}\S/.test(lines[i]) && items.length) {
            items[items.length - 1].text += " " + lines[i].trim();
            continue;
          }
          break;
        }
        if (m[1].length >= 2 && items.length) (items[items.length - 1].subs ??= []).push(m[3]);
        else items.push({ text: m[3] });
      }
      i--;
      const tag = ordered ? "ol" : "ul";
      out.push(
        `<${tag}>` +
          items
            .map(
              (it) =>
                `<li>${renderInline(it.text, ctx)}` +
                (it.subs ? `<ul>${it.subs.map((s) => `<li>${renderInline(s, ctx)}</li>`).join("")}</ul>` : "") +
                `</li>`
            )
            .join("") +
          `</${tag}>`
      );
      continue;
    }
    if (!line.trim()) {
      flush();
      continue;
    }
    para.push(line.trim());
  }
  flush();
  return out.join("\n");
}

function renderInline(text, ctx) {
  let s = esc(text);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => {
    codes.push(c);
    return `\x00${codes.length - 1}\x00`;
  });
  s = s.replace(/\[\^([\w-]+)\]/g, (_, id) => citeChipHtml(id, ctx));
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|#[^)\s]*)\)/g, `<a href="$2" target="_blank" rel="noopener">$1</a>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>");
  s = s.replace(/\x00(\d+)\x00/g, (_, i) => `<code>${codes[+i]}</code>`);
  return s;
}

function citeChipHtml(id, ctx) {
  let n = ctx.order.indexOf(id) + 1;
  if (!n) n = ctx.order.push(id);
  if (!ctx.citations[id])
    return `<sup class="cite-chip missing" data-cite="${esc(id)}" title="[^${esc(id)}] has no citation definition">?</sup>`;
  return `<sup class="cite-chip" data-cite="${esc(id)}" title="citation ${esc(id)}">${n}</sup>`;
}

/* Whitespace-insensitive exact quote search; returns [start, end) ranges in
   the original text so highlights survive reflowed whitespace. When the quote
   repeats inside the text, optional prefix/suffix (verbatim adjacent text) pick
   the right occurrence. */
function findQuotes(text, quote, prefix = "", suffix = "") {
  const map = [];
  let normed = "";
  for (let i = 0; i < text.length; i++) {
    const ws = /\s/.test(text[i]);
    if (ws && (!normed || normed.endsWith(" "))) continue;
    normed += ws ? " " : text[i];
    map.push(i);
  }
  const norm = (s) => String(s ?? "").replace(/\s+/g, " ").trim();
  const q = norm(quote);
  if (!q) return [];
  const pre = norm(prefix);
  const post = norm(suffix);
  const ranges = [];
  for (let at = normed.indexOf(q); at >= 0; at = normed.indexOf(q, at + 1)) {
    if (pre && !normed.slice(0, at).trimEnd().endsWith(pre)) continue;
    if (post && !normed.slice(at + q.length).trimStart().startsWith(post)) continue;
    ranges.push([map[at], map[at + q.length - 1] + 1]);
  }
  return ranges;
}

const findQuote = (...args) => findQuotes(...args)[0] || null;

function quoteMarkedHtml(text, marks) {
  const ranges = [];
  for (const h of marks) {
    const r = h.quote ? findQuote(text, h.quote, h.prefix, h.suffix) : null;
    if (r) ranges.push([...r, h.reason]);
  }
  if (!ranges.length) return esc(text);
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([...r]);
  }
  let out = "";
  let pos = 0;
  for (const [s, e, tip] of merged) {
    out += esc(text.slice(pos, s)) + `<mark class="hl-quote"${tip ? ` data-tip="${esc(tip)}"` : ""}>${esc(text.slice(s, e))}</mark>`;
    pos = e;
  }
  return out + esc(text.slice(pos));
}

/* Citation resolution uses the same read endpoints as the trace viewer. Lookup
   promises live only for the current report render: duplicate references share
   work, while a changed report or rewritten trace never inherits stale evidence. */
const CITATION_KEYS = new Set([
  "run", "step", "kind", "subset", "episode", "trace", "branch",
  "node", "field", "quote", "prefix", "suffix", "note",
]);

async function resolveCitation(c, cache) {
  if (!c) return { matched: false, reason: "missing citation definition" };
  if (typeof c !== "object" || c._invalid) return { matched: false, reason: "invalid citation JSON" };
  const unknown = Object.keys(c).filter((key) => !CITATION_KEYS.has(key));
  if (unknown.length) return { matched: false, reason: `unknown fields: ${unknown.join(", ")}` };
  const run = c.run || state.run;
  if (typeof run !== "string" || !run) return { matched: false, reason: "citation needs a run" };
  if (!Number.isInteger(c.step) || !["train", "eval"].includes(c.kind) || !["all", "effective"].includes(c.subset))
    return { matched: false, reason: "citation needs a valid step, kind, and subset" };
  for (const key of ["trace", "node"])
    if (c[key] != null && (!Number.isInteger(c[key]) || c[key] < 0)) return { matched: false, reason: `${key} must be a non-negative integer` };
  if (c.branch != null && (!Number.isInteger(c.branch) || c.branch < -1))
    return { matched: false, reason: "branch must be an integer >= -1" };
  if (c.field != null && !["content", "reasoning"].includes(c.field))
    return { matched: false, reason: "field must be content or reasoning" };
  if (typeof c.episode !== "string" || !c.episode) return { matched: false, reason: "citation needs an episode id" };
  if (typeof c.quote !== "string" || !c.quote.trim()) return { matched: false, reason: "citation needs a verbatim quote" };
  if (typeof c.note !== "string" || !c.note.trim()) return { matched: false, reason: "citation needs a note" };
  for (const key of ["prefix", "suffix"])
    if (c[key] != null && typeof c[key] !== "string") return { matched: false, reason: `${key} must be a string` };
  const base = `/api/runs/${encodeURIComponent(run)}/episodes`;
  const summaryKey = `${base}?episode=${encodeURIComponent(c.episode)}&limit=2`;
  if (!cache.summaries.has(summaryKey))
    cache.summaries.set(
      summaryKey,
      api(summaryKey).catch((err) => {
        cache.summaries.delete(summaryKey);
        throw err;
      })
    );
  const list = await cache.summaries.get(summaryKey);
  if (!list.total) return { matched: false, reason: `episode ${c.episode} not found` };
  if (list.total > 1) return { matched: false, reason: `episode ${c.episode} is not unique` };
  const line = list.episodes[0].line;
  const epKey = `${base}/${line}`;
  if (!cache.episodes.has(epKey))
    cache.episodes.set(
      epKey,
      api(epKey).catch((err) => {
        cache.episodes.delete(epKey);
        throw err;
      })
    );
  const ep = await cache.episodes.get(epKey);
  const trace = (ep.traces || [])[c.trace ?? 0];
  if (!trace) return { matched: false, reason: "trace not found", line };
  let nodes = c.node != null ? [[c.node, (trace.nodes || [])[c.node]]] : (trace.nodes || []).map((n, i) => [i, n]);
  nodes = nodes.filter(([, n]) => n);
  if (c.branch != null && c.branch !== -1) {
    const branch = traceBranches(trace)[c.branch];
    if (!branch) return { matched: false, reason: "branch not found", line };
    nodes = nodes.filter(([i]) => branch.includes(i));
  }
  if (!nodes.length) return { matched: false, reason: `node ${c.node} not found`, line };
  const matches = [];
  for (const [i, node] of nodes) {
    const fields = [
      ["content", messageText(node.message)],
      ["reasoning", reasoningText(node.message?.reasoning_content ?? node.message?.reasoning)],
    ].filter(([field]) => !c.field || c.field === field);
    for (const [field, text] of fields) {
      for (const _ of findQuotes(text, c.quote, c.prefix, c.suffix)) matches.push({ nodeIdx: i, field });
    }
  }
  if (!matches.length)
    return { matched: false, reason: c.node != null ? "quote not found in node" : "quote not found in any node", line };
  if (matches.length > 1) return { matched: false, reason: "quote is ambiguous — add node and prefix or suffix", line };
  return { matched: true, line, ...matches[0] };
}

function paintCitation(id, res) {
  document.querySelectorAll(`.cite-chip[data-cite="${CSS.escape(id)}"]`).forEach((el) => {
    el.classList.toggle("ok", !!res.matched);
    el.classList.toggle("bad", !res.matched);
    el.title = res.matched ? "quote verified against the trace" : `⚠ ${res.reason || "quote not found"}`;
  });
}

async function verifyCitations() {
  const rep = state.report;
  const citations = rep.citations;
  const cache = rep.lookup;
  const ids = rep.order;
  if (!ids.length) return;
  let verified = 0;
  let broken = 0;
  for (const id of ids) {
    let res;
    try {
      res = await resolveCitation(citations[id], cache);
    } catch (err) {
      res = { matched: false, reason: String(err) };
    }
    if (state.report !== rep || rep.lookup !== cache) return;
    rep.verify.set(id, res);
    if (res.matched) verified++;
    else broken++;
    paintCitation(id, res);
    $("#report-verify").textContent = `${verified}/${ids.length} verified${broken ? ` · ${broken} broken` : ""}`;
  }
}

/* ------------------------------------------------------ citation click */
/* a chip is a link: clicking jumps straight to the trace with the quote
   highlighted; the citation's note surfaces on hover over the mark */

async function openCitation(id) {
  const c = state.report.citations[id];
  const rep = state.report;
  const cache = rep.lookup;
  let res = rep.verify.get(id);
  if (!res) {
    try {
      res = await resolveCitation(c, cache);
    } catch (err) {
      res = { matched: false, reason: String(err) };
    }
    if (state.report !== rep || rep.lookup !== cache) return;
    rep.verify.set(id, res);
    paintCitation(id, res);
  }
  if (!res.matched) return toastMsg(`[^${esc(id)}] is broken: ${esc(res.reason || "quote not found")}`);
  await applyViewCommand({
    run: c.run || state.run,
    tab: "traces",
    step: c.step,
    kind: c.kind,
    subset: c.subset,
    episode: c.episode,
    line: res.line,
    trace: c.trace,
    branch: c.branch ?? -1,
    highlight: [{ node: res.nodeIdx, field: res.field, quote: c.quote, prefix: c.prefix, suffix: c.suffix, reason: c.note }],
  });
  $("#tm-back").hidden = $("#trace-modal").hidden; // way back to the report
}

/* ----------------------------------------------------------- view command */
/* SSE from /api/view/events: a local agent POSTs an on-disk address, every
   connected tab navigates there through the same functions clicks use. */

let pendingHighlight = null;
let lastViewSeq = 0;
let hadHashRun = false;
let applyingView = false;
const viewCommandQueue = [];
let viewDrain = Promise.resolve();
const MAX_PENDING_VIEW_COMMANDS = 32;

function primeTraceCommand(cmd) {
  const traces = state.traces;
  if (cmd.step != null) traces.step = cmd.step;
  // a command naming a kind narrows to it; both stay on otherwise
  if (cmd.kind) traces.kinds = { train: cmd.kind === "train", eval: cmd.kind === "eval" };
  // a citation addressed to a step wants the cohort view; `all` is the stream
  if (cmd.subset) traces.mode = cmd.subset === "effective" ? "step" : "stream";
  traces.bin = null;
  traces.env = "";
  traces.errorsOnly = false;
  if (cmd.highlight?.length) traces.viewMode = "messages";
  pendingHighlight = null;
}

async function applyOneViewCommand(cmd) {
  let runChanged = false;
  if (cmd.run && cmd.run !== state.run) {
    if (!state.runs.some((r) => r.name === cmd.run)) await loadRuns();
    if (!state.runs.some((r) => r.name === cmd.run)) return toastMsg(`unknown run ${esc(cmd.run)}`);
    await selectRun(cmd.run, true);
    runChanged = true;
  }
  if (cmd.report) {
    state.report.wanted = cmd.report;
    state.report.loaded = false;
  }
  const traceCommand = ["step", "kind", "subset", "episode", "line", "trace", "branch", "highlight"].some(
    (key) => cmd[key] != null
  );
  if (traceCommand) primeTraceCommand(cmd); // target the requested file before the traces tab initializes
  if (cmd.tab && (cmd.tab !== state.tab || cmd.report || runChanged)) await activateTab(cmd.tab, true);
  else if (cmd.report && state.tab === "report") await initReport();
  else if (runChanged) await activateTab(state.tab, true);
  if (traceCommand) await applyTraceCommand(cmd);
}

async function drainViewCommands() {
  while (viewCommandQueue.length) {
    const cmd = viewCommandQueue.shift();
    try {
      await applyOneViewCommand(cmd);
    } catch (err) {
      console.warn("view command failed", err);
    }
  }
  applyingView = false;
}

function applyViewCommand(cmd) {
  viewCommandQueue.push(cmd);
  if (viewCommandQueue.length > MAX_PENDING_VIEW_COMMANDS) viewCommandQueue.shift();
  if (!applyingView) {
    applyingView = true;
    viewDrain = drainViewCommands();
  }
  return viewDrain;
}

async function applyTraceCommand(cmd) {
  const traces = state.traces;
  if (!traces.loaded) await initTraces();
  adjustKindSubset();
  renderStepControl();
  await loadEpisodes();
  if (cmd.line == null && cmd.episode == null) return;
  const episodes = traces.episodes || [];
  const target = cmd.episode != null ? episodes.find((e) => e.id === cmd.episode) : episodes.find((e) => e.line === cmd.line);
  const line = target?.line ?? cmd.line;
  if (line == null) return toastMsg(`episode ${esc(cmd.episode ?? "?")} not found at this address`);
  pendingHighlight = {
    run: state.run,
    step: traces.step,
    kind: activeKind(),
    subset: traces.mode === "step" ? "effective" : "all",
    line,
    trace: cmd.trace ?? 0,
    highlights: (cmd.highlight || []).filter((h) => h && h.node != null),
    scrolled: false,
  };
  await openEpisode(line, { trace: cmd.trace, branch: cmd.branch });
}

let toastEl = null;
let toastTimer = 0;

function toastMsg(html, ms = 6000) {
  if (!toastEl) {
    toastEl = document.createElement("div");
    toastEl.id = "view-toast";
    toastEl.hidden = true;
    document.body.appendChild(toastEl);
    toastEl.addEventListener("click", (e) => {
      if (e.target.closest(".toast-dismiss")) toastEl.hidden = true;
      const go = e.target.closest("[data-cmd]");
      if (go) {
        toastEl.hidden = true;
        applyViewCommand(JSON.parse(go.dataset.cmd));
      }
    });
  }
  toastEl.innerHTML = html;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  if (ms) toastTimer = setTimeout(() => (toastEl.hidden = true), ms);
}

function showViewToast(cmd) {
  const label = [cmd.run, cmd.tab, cmd.step != null ? `step ${cmd.step}` : null, cmd.episode ?? cmd.report]
    .filter(Boolean)
    .join(" · ");
  toastMsg(
    `<span class="t-label">agent</span><span class="toast-text">${esc(label)}</span>` +
      `<button class="btn" data-cmd="${esc(JSON.stringify(cmd))}">go</button><button class="btn toast-dismiss">✕</button>`,
    30000
  );
}

function connectViewEvents() {
  const source = new EventSource("/api/view/events");
  source.onmessage = (e) => {
    let cmd;
    try {
      cmd = JSON.parse(e.data);
    } catch {
      return;
    }
    if (!cmd || (cmd.seq && cmd.seq <= lastViewSeq)) return;
    if (cmd.seq) lastViewSeq = cmd.seq;
    // a stored command replays on connect: apply it only to a fresh bare tab —
    // a deep link or reload states its own target, and stale pointers stay dead
    if (cmd.replay && (hadHashRun || Date.now() / 1000 - (cmd.ts || 0) > 900)) return;
    if (!state.follow) return showViewToast(cmd);
    applyViewCommand(cmd);
  };
}

$("#follow-toggle").addEventListener("change", (e) => {
  state.follow = e.target.checked;
  savePrefs();
});
$("#report-select").addEventListener("change", async (e) => {
  state.report.file = e.target.value;
  await loadReport();
});
$("#report-body").addEventListener("click", (e) => {
  const chip = e.target.closest(".cite-chip");
  if (chip) openCitation(chip.dataset.cite);
});

/* ---------------------------------------------------------------- wiring */

$("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
$("#live-toggle").addEventListener("change", async (e) => {
  state.live = e.target.checked;
  if (state.live) await pollDashboard();
});
$("#compare-menu").addEventListener("change", (e) => {
  const box = e.target.closest("[data-compare]");
  if (box) toggleCompare(box.dataset.compare, box.checked);
});
// one delegated handler for every .dd-wrap dropdown: button toggles its menu,
// clicking anywhere else closes them all (a dropdown nested inside another
// menu, e.g. the env select in the trace filter, keeps its ancestors open)
document.addEventListener("click", (e) => {
  const wrap = e.target.closest(".dd-wrap");
  document.querySelectorAll(".dd-menu").forEach((menu) => {
    if (!wrap || !(wrap.contains(menu) || menu.contains(wrap))) menu.hidden = true;
  });
  const btn = e.target.closest(".dd-btn");
  if (btn && wrap) {
    const menu = wrap.querySelector(".dd-menu");
    if (wrap.classList.contains("dd-select")) rebuildSelectMenu(wrap);
    menu.hidden = !menu.hidden;
    if (!menu.hidden && menu.id === "compare-menu") renderCompareMenu();
  }
});

/* native selects wear the compare-dropdown look: the hidden <select> stays the
   source of truth (existing change listeners keep working), the .dd-menu lists
   its live options each time it opens */
const dressedSelects = new Set();

function rebuildSelectMenu(wrap) {
  const select = wrap.querySelector("select");
  wrap.querySelector(".dd-menu").innerHTML = [...select.options]
    .map((o, i) => `<div class="dd-opt${o.selected ? " active" : ""}${o.disabled ? " disabled" : ""}" data-i="${i}">${esc(o.textContent)}</div>`)
    .join("");
}

function syncDressedSelects() {
  for (const select of dressedSelects) {
    const wrap = select.closest(".dd-wrap");
    if (!wrap) continue;
    wrap.querySelector(".dd-btn span").textContent = select.selectedOptions[0]?.textContent ?? "";
    wrap.querySelector(".dd-btn").disabled = select.disabled;
  }
}

function dressSelect(select) {
  if (!select || select.parentElement?.classList.contains("dd-select")) return;
  const wrap = document.createElement("div");
  wrap.className = "dd-wrap dd-select";
  if (select.title) wrap.title = select.title;
  select.parentNode.insertBefore(wrap, select);
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn dd-btn";
  btn.appendChild(document.createElement("span"));
  const menu = document.createElement("div");
  menu.className = "dd-menu dd-optlist";
  menu.hidden = true;
  menu.addEventListener("click", (e) => {
    const opt = e.target.closest(".dd-opt");
    if (!opt || opt.classList.contains("disabled")) return;
    select.selectedIndex = +opt.dataset.i;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    menu.hidden = true;
    syncDressedSelects();
  });
  wrap.append(btn, menu, select);
  select.hidden = true;
  dressedSelects.add(select);
  syncDressedSelects();
}
/* the table and the trace viewer carry the same filter dropdown over one
   shared state — a change in either view shows up in both */
function syncTraceFilterControls() {
  const t = state.traces;
  for (const sel of ["#trace-env", "#tm-env"]) $(sel).value = t.env;
  for (const sel of ["#trace-kinds", "#tm-kinds"])
    for (const button of document.querySelectorAll(`${sel} button`)) {
      button.classList.toggle("on", !!t.kinds[button.dataset.kind]);
      // note a kind this run never produced, but leave it toggleable: disabling it
      // would strand the toggle off the moment someone turned it off
      const absent = t.runKinds && !t.runKinds.includes(button.dataset.kind);
      button.classList.toggle("absent", !!absent);
      button.title = absent ? `no ${button.dataset.kind} episodes in this run` : "";
    }
  for (const sel of ["#trace-sort", "#tm-sort"]) $(sel).value = traceSort();
  for (const sel of ["#trace-errors", "#tm-errors"]) $(sel).checked = t.errorsOnly;
  const active = [t.env, activeKind(), t.errorsOnly].filter(Boolean).length;
  for (const sel of ["#trace-filter-btn", "#tm-filter-btn"]) $(sel).classList.toggle("active", active > 0);
  const badge = $("#trace-filter-count");
  badge.hidden = !active;
  badge.textContent = active;
  // the stream is not addressed by step, so its controls go away in that mode
  $("#step-bar").hidden = t.mode !== "step";
  $("#tm-stephead").hidden = t.mode !== "step";
  $("#trace-chart").hidden = t.mode !== "stream";
  $("#trace-clear-bin").hidden = !t.bin;
  setActive("#trace-mode", "mode", t.mode);
  setActive("#tm-mode", "mode", t.mode);
  syncDressedSelects();
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => activateTab(b.dataset.tab)));

document.querySelectorAll("#metrics-mode button").forEach((b) =>
  b.addEventListener("click", () => {
    state.metrics.mode = b.dataset.mode;
    setActive("#metrics-mode", "mode", b.dataset.mode);
    $("#all-layout").hidden = b.dataset.mode !== "all";
    renderMetricsBody();
    savePrefs();
  })
);
$("#config-format").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-fmt]");
  if (!btn || btn.disabled || btn.dataset.fmt === state.config.fmt) return;
  state.config.fmt = btn.dataset.fmt;
  state.config.file = configFileFor(btn.dataset.fmt);
  renderConfigFormat();
  loadConfig();
});
$("#config-command-copy").addEventListener("click", (e) => {
  const command = state.config.commandText.trimEnd();
  if (command) copyText(command, e.currentTarget);
});
$("#config-search").addEventListener(
  "input",
  debounce(() => {
    applyConfigSearch();
    savePrefs();
  })
);
document.querySelectorAll("#all-layout button").forEach((b) =>
  b.addEventListener("click", () => {
    state.metrics.allLayout = b.dataset.layout;
    setActive("#all-layout", "layout", b.dataset.layout);
    renderMetricsBody();
    savePrefs();
  })
);
$("#metrics-search").addEventListener(
  "input",
  debounce(() => {
    state.metrics.search = $("#metrics-search").value;
    renderMetricsBody();
    savePrefs();
  }, 250)
);

// remember collapsed sections across re-renders; charts created while hidden
// have zero width, so resize on expand ("toggle" doesn't bubble → capture)
$("#metrics-body").addEventListener(
  "toggle",
  (e) => {
    const section = e.target;
    if (!section.matches?.("details.section")) return;
    if (section.open) resizeCharts();
    // a search force-opens sections - don't let that overwrite the saved state
    if (activeFilter) return;
    if (section.open) state.metrics.collapsedSections.delete(section.dataset.name);
    else state.metrics.collapsedSections.add(section.dataset.name);
    savePrefs();
  },
  true
);

$("#metrics-collapse").addEventListener("click", () =>
  document.querySelectorAll("#metrics-body details.section").forEach((s) => (s.open = false))
);
$("#metrics-expand").addEventListener("click", () =>
  document.querySelectorAll("#metrics-body details.section").forEach((s) => (s.open = true))
);

// drag a pane header to reorder within its section (order persisted by title)
$("#metrics-body").addEventListener("dragover", (e) => {
  if (!dragCard) return;
  const grid = e.target.closest(".chart-grid");
  if (!grid || grid !== dragCard.parentElement) return;
  e.preventDefault();
  const target = e.target.closest(".chart-card");
  if (!target || target === dragCard) return;
  // reorder the moment the cursor enters another pane
  const cards = [...grid.children];
  grid.insertBefore(dragCard, cards.indexOf(dragCard) < cards.indexOf(target) ? target.nextSibling : target);
});

/* wandb-style resize handles: resizing one pane resizes all of them */
$("#metrics-body").addEventListener("pointerdown", (e) => {
  const grip = e.target.closest("[data-rz]");
  if (!grip) return;
  e.preventDefault();
  const mode = grip.dataset.rz;
  const card = grip.closest(".chart-card");
  const startX = e.clientX;
  const startY = e.clientY;
  const startW = card.clientWidth;
  const startH = state.metrics.paneH;
  document.body.classList.add("resizing");
  document.body.style.cursor = mode === "x" ? "ew-resize" : mode === "y" ? "ns-resize" : "nwse-resize";
  let raf = 0;
  const move = (ev) => {
    if (mode !== "y") state.metrics.paneMin = Math.round(Math.max(220, Math.min(900, startW + ev.clientX - startX)));
    if (mode !== "x") state.metrics.paneH = Math.round(Math.max(90, Math.min(420, startH + ev.clientY - startY)));
    if (!raf)
      raf = requestAnimationFrame(() => {
        raf = 0;
        applyPaneSize();
      });
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    document.body.classList.remove("resizing");
    document.body.style.cursor = "";
    applyPaneSize();
    savePrefs();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
});

$("#attempt-select").addEventListener("change", async (e) => {
  state.logs.attempt = e.target.value;
  state.logs.buffers = new Map();
  state.logs.paneFile = {};
  await loadLogfiles();
  await pollLogs();
});
$("#config-attempt-select").addEventListener("change", async (e) => {
  state.config.attempt = e.target.value;
  state.config.file = null;
  await loadConfigAttempt();
});
$("#log-panes").addEventListener("change", async (e) => {
  const select = e.target.closest(".lp-file");
  if (!select) return;
  const comp = select.closest(".log-pane").dataset.comp;
  state.logs.paneFile[comp] = select.value;
  await pollLogs(false);
  renderAllLogPanes();
});
$("#log-panes").addEventListener("click", (e) => {
  const btn = e.target.closest(".lp-max");
  if (!btn) return;
  const comp = btn.closest(".log-pane").dataset.comp;
  state.logs.maximized = state.logs.maximized === comp ? null : comp;
  renderLogPanes();
  dressLogPaneSelects();
  renderAllLogPanes();
});
$("#log-comp-menu").addEventListener("change", async (e) => {
  const box = e.target.closest("[data-comp]");
  if (!box) return;
  const logs = state.logs;
  logs.components ??= new Set(LOG_PANES.filter((p) => paneFiles(p).length).map((p) => p.comp));
  if (box.checked) logs.components.add(box.dataset.comp);
  else logs.components.delete(box.dataset.comp);
  renderLogPanes();
  dressLogPaneSelects();
  await pollLogs(false);
  renderAllLogPanes();
  savePrefs();
});
document.querySelectorAll("#log-view button").forEach((b) =>
  b.addEventListener("click", () => {
    state.logs.view = b.dataset.view;
    setActive("#log-view", "view", b.dataset.view);
    renderLogPanes();
  dressLogPaneSelects();
    renderAllLogPanes();
    savePrefs();
  })
);
const LOG_LEVELS = [
  ["DEBUG", "all"],
  ["INFO", "info+"],
  ["WARNING", "warn+"],
  ["ERROR", "error+"],
];

function renderLogLevel() {
  const idx = LOG_LEVELS.findIndex(([level]) => level === state.logs.level);
  $("#log-level-label").textContent = LOG_LEVELS[idx][1];
  $("#log-level-down").disabled = idx <= 0;
  $("#log-level-up").disabled = idx >= LOG_LEVELS.length - 1;
}

function stepLogLevel(delta) {
  const idx = LOG_LEVELS.findIndex(([level]) => level === state.logs.level);
  const next = LOG_LEVELS[idx + delta];
  if (!next) return;
  state.logs.level = next[0];
  renderLogLevel();
  renderAllLogPanes();
  savePrefs();
}

$("#log-level-down").addEventListener("click", () => stepLogLevel(-1));
$("#log-level-up").addEventListener("click", () => stepLogLevel(1));
$("#log-search").addEventListener(
  "input",
  debounce(() => {
    renderAllLogPanes();
    savePrefs();
  })
);
$("#log-older").addEventListener("click", loadOlder);

$("#step-blocks").addEventListener("click", (e) => {
  const cell = e.target.closest(".sb-cell");
  if (cell) selectStepByIndex(+cell.dataset.i);
});
// scrub across blocks with the button held
$("#step-blocks").addEventListener("pointerover", (e) => {
  if (!(e.buttons & 1)) return;
  const cell = e.target.closest(".sb-cell");
  if (cell) selectStepByIndex(+cell.dataset.i);
});
$("#step-prev").addEventListener("click", () => {
  const idx = state.traces.steps.findIndex((s) => s.step === state.traces.step);
  selectStepByIndex(idx - 1);
});
$("#step-next").addEventListener("click", () => {
  const idx = state.traces.steps.findIndex((s) => s.step === state.traces.step);
  selectStepByIndex(idx + 1);
});
async function setTraceMode(mode, inModal = false) {
  const traces = state.traces;
  traces.mode = mode;
  traces.bin = null;
  adjustKindSubset();
  await loadEpisodes();
  if (mode === "stream") await loadHistogram();
  savePrefs();
  if (inModal) await reopenFirstEpisode();
}

/* filter changes keep the open viewer in sync: hold the current episode when
   it survives the filter, land on the first one otherwise */
async function refreshModalList() {
  if ($("#trace-modal").hidden) return;
  if (filteredRollouts().some((e) => e.line === currentLine)) renderRolloutList();
  else await reopenFirstEpisode();
}

/* after a filter change inside the viewer, land on the first episode of the
   new subset (line numbers don't correspond across files) */
async function reopenFirstEpisode() {
  renderModalStep();
  const first = filteredRollouts()[0];
  if (first) return openEpisode(first.line);
  currentLine = null;
  currentEpisode = null;
  renderRolloutList();
  $("#tm-messages").innerHTML = emptyState("no episodes", "nothing here for the current filters");
  $("#tm-meta").innerHTML = "";
}

for (const [sel, inModal] of [["#trace-mode", false], ["#tm-mode", true]])
  document.querySelectorAll(`${sel} button`).forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.mode === state.traces.mode) return;
      setTraceMode(b.dataset.mode, inModal);
    })
  );
for (const sel of ["#trace-env", "#tm-env"])
  $(sel).addEventListener("change", async (e) => {
    state.traces.env = e.target.value;
    await loadEpisodes();
    await loadHistogram();
    await refreshModalList();
  });
for (const sel of ["#trace-kinds", "#tm-kinds"])
  document.querySelectorAll(`${sel} button`).forEach((b) =>
    b.addEventListener("click", async () => {
      const kinds = state.traces.kinds;
      const kind = b.dataset.kind;
      const other = kind === "train" ? "eval" : "train";
      if (kinds[kind] && !kinds[other]) return; // never leave both off
      kinds[kind] = !kinds[kind];
      await loadEpisodes();
      await loadHistogram();
      savePrefs();
      await refreshModalList();
    })
  );
$("#trace-clear-bin").addEventListener("click", async () => {
  state.traces.bin = null;
  await loadEpisodes();
  renderHistogram();
});
$("#trace-hist").addEventListener("mousemove", (e) => {
  const bar = e.target.closest(".hbar");
  const tip = $("#hist-tip");
  if (!bar) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML = histTipHtml(+bar.dataset.t, +bar.dataset.count, state.traces.hist?.bin ?? 60);
  tip.hidden = false;
  const host = $("#trace-chart").getBoundingClientRect();
  const left = Math.min(e.clientX - host.left + 12, host.width - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${e.clientY - host.top + 14}px`;
});
$("#trace-hist").addEventListener("mouseleave", () => {
  $("#hist-tip").hidden = true;
});
// a bar narrows the table to the episodes that finished in it
$("#trace-hist").addEventListener("click", async (e) => {
  const bar = e.target.closest(".hbar");
  if (!bar) return;
  const start = +bar.dataset.t;
  const width = state.traces.hist?.bin ?? 60;
  const bin = [start, start + width];
  state.traces.bin = state.traces.bin && state.traces.bin[0] === start ? null : bin;
  await loadEpisodes();
  renderHistogram();
});
// infinite scroll: pull the next page as the reader nears the end
$("#episode-table-wrap").addEventListener("scroll", () => {
  const wrap = $("#episode-table-wrap");
  if (wrap.scrollTop + wrap.clientHeight > wrap.scrollHeight - 400) loadMoreEpisodes();
});
for (const sel of ["#trace-errors", "#tm-errors"])
  $(sel).addEventListener("change", async (e) => {
    state.traces.errorsOnly = e.target.checked;
    await loadEpisodes();
    savePrefs();
    await refreshModalList();
  });
for (const sel of ["#trace-sort", "#tm-sort"])
  $(sel).addEventListener("change", async (e) => {
    state.traces.sorts[state.traces.mode] = e.target.value;
    await loadEpisodes();
    savePrefs();
    await refreshModalList();
  });
$("#episode-table").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-line]");
  if (row) openEpisode(+row.dataset.line);
});
function rafThrottle(fn) {
  let pending = false;
  return () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      fn();
    });
  };
}
$("#episode-table-wrap").addEventListener(
  "scroll",
  rafThrottle(() => state.traces.episodes?.length && renderEpisodeRows())
);
$("#tm-list").addEventListener(
  "scroll",
  rafThrottle(() => {
    renderRolloutWindow();
    const list = $("#tm-list");
    if (list.scrollTop + list.clientHeight > list.scrollHeight - 300) loadMoreEpisodes();
  })
);
$("#drawer-close").addEventListener("click", closeDrawer);
$("#tm-back").addEventListener("click", () => {
  closeDrawer();
  activateTab("report");
});
$("#drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") return closeDrawer();
  if ($("#trace-modal").hidden || e.target.matches("input, select, textarea")) return;
  if (e.key === "ArrowDown") { e.preventDefault(); stepRollout(1); }
  if (e.key === "ArrowUp") { e.preventDefault(); stepRollout(-1); }
  if (e.key === "ArrowLeft") { e.preventDefault(); modalStep(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); modalStep(1); }
});
$("#tm-step-prev").addEventListener("click", () => modalStep(-1));
$("#tm-step-next").addEventListener("click", () => modalStep(1));
// instant per-token tooltip (native title is too laggy over thousands of spans)
const tokTip = document.createElement("div");
tokTip.className = "tok-tip";
tokTip.hidden = true;
document.body.appendChild(tokTip);
$("#tm-messages").addEventListener("mouseover", (e) => {
  const tok = e.target.closest("[data-tip]");
  if (!tok) return;
  tokTip.textContent = tok.dataset.tip;
  tokTip.classList.toggle("note", tok.matches("mark.hl-quote")); // citation notes wrap
  tokTip.hidden = false;
  const rect = tok.getBoundingClientRect();
  tokTip.style.left = `${Math.min(rect.left, window.innerWidth - tokTip.offsetWidth - 12)}px`;
  tokTip.style.top = `${rect.top - tokTip.offsetHeight - 6}px`;
});
$("#tm-messages").addEventListener("mouseout", (e) => {
  if (e.target.closest("[data-tip]")) tokTip.hidden = true;
});
$("#token-signal").addEventListener("change", async () => {
  await ensureTokens();
  renderEpisode();
  savePrefs();
});
$("#trace-view-mode").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-mode]");
  if (!btn || btn.dataset.mode === state.traces.viewMode) return;
  state.traces.viewMode = btn.dataset.mode;
  setActive("#trace-view-mode", "mode", state.traces.viewMode);
  await ensureTokens();
  renderEpisode();
  savePrefs();
});
$("#tm-view").addEventListener("click", (e) => {
  const button = e.target.closest("[data-view]");
  if (button && button.dataset.view !== traceView) setTraceView(button.dataset.view);
});
$("#tm-timeline").addEventListener("click", async (e) => {
  const target = e.target.closest("[data-tl-trace]");
  if (!target) return;
  e.stopPropagation();
  currentTraceIdx = +target.dataset.tlTrace;
  const node = target.dataset.tlNode == null ? null : +target.dataset.tlNode;
  const call = target.dataset.tlCall == null ? null : +target.dataset.tlCall;
  if (node != null) {
    const trace = currentEpisode?.traces?.[currentTraceIdx];
    const branches = trace ? traceBranches(trace) : [];
    const branch = branches.findIndex((path) => path.includes(node));
    currentBranchIdx = branch >= 0 ? branch : -1;
    pendingTimelineNode = node;
    pendingTimelineCall = call;
    state.traces.viewMode = "messages";
  } else {
    currentBranchIdx = 0;
    pendingTimelineNode = null;
    pendingTimelineCall = call;
    if (call != null) state.traces.viewMode = "messages";
  }
  await setTraceView("transcript");
  requestAnimationFrame(() => {
    const entry =
      pendingTimelineCall == null
        ? pendingTimelineNode == null
          ? null
          : $(`#tm-messages [data-node="${pendingTimelineNode}"]`)
        : $(`#tm-messages [data-call-index="${pendingTimelineCall}"]`);
    entry?.scrollIntoView({ block: "center" });
    const details = entry?.closest("details");
    if (details) details.open = true;
    pendingTimelineNode = null;
    pendingTimelineCall = null;
  });
});
const timelineTip = document.createElement("div");
timelineTip.className = "tl-tooltip";
timelineTip.hidden = true;
document.body.appendChild(timelineTip);
function moveTimelineTip(e) {
  const gap = 12;
  const left = Math.min(e.clientX + gap, window.innerWidth - timelineTip.offsetWidth - 8);
  const top = Math.min(e.clientY + gap, window.innerHeight - timelineTip.offsetHeight - 8);
  timelineTip.style.left = `${Math.max(8, left)}px`;
  timelineTip.style.top = `${Math.max(8, top)}px`;
}
$("#tm-timeline").addEventListener("mouseover", (e) => {
  const target = e.target.closest("[data-timeline-tip]");
  if (!target) return;
  const payload = JSON.parse(target.dataset.timelineTip);
  timelineTip.innerHTML =
    `<div class="tl-tooltip-kind">${esc(payload.kind)}</div><div class="tl-tooltip-title">${esc(payload.title)}</div>` +
    `${payload.snippet ? `<div class="tl-tooltip-snippet">${esc(payload.snippet)}</div>` : ""}` +
    `<dl>${payload.rows.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("")}</dl>` +
    `<div class="tl-tooltip-hint">${esc(payload.hint)}</div>`;
  timelineTip.hidden = false;
  moveTimelineTip(e);
});
$("#tm-timeline").addEventListener("mousemove", (e) => {
  if (!timelineTip.hidden) moveTimelineTip(e);
});
$("#tm-timeline").addEventListener("mouseout", (e) => {
  if (e.target.closest("[data-timeline-tip]") && !e.relatedTarget?.closest?.("[data-timeline-tip]")) timelineTip.hidden = true;
});
$("#tm-trace-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-trace]");
  if (btn) { currentTraceIdx = +btn.dataset.trace; currentBranchIdx = 0; renderEpisode(); }
});
$("#tm-branch-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-branch]");
  if (btn) { currentBranchIdx = +btn.dataset.branch; renderEpisode(); }
});
$("#tm-list").addEventListener("click", (e) => {
  const item = e.target.closest("[data-line]");
  if (item) openEpisode(+item.dataset.line);
});
$("#tm-collapse").addEventListener("click", () =>
  document.querySelectorAll("#tm-messages details").forEach((d) => (d.open = false))
);
$("#tm-expand").addEventListener("click", () =>
  document.querySelectorAll("#tm-messages details").forEach((d) => (d.open = true))
);
$("#tm-messages").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copy], [data-copy-tool], [data-copy-schema], [data-copy-tools], [data-copy-rendered]");
  if (!btn) return;
  e.preventDefault();
  e.stopPropagation();
  const trace = currentEpisode?.traces?.[currentTraceIdx];
  if (!trace) return;
  if (btn.dataset.copy != null) {
    const node = trace.nodes?.[+btn.dataset.copy];
    if (node) copyText(messageText(node.message), btn);
    return;
  }
  const tools = normalizedTools(trace.tools);
  if (btn.hasAttribute("data-copy-tools")) return copyText(JSON.stringify(tools, null, 2), btn);
  if (btn.dataset.copyTool != null) return copyText(JSON.stringify(tools[+btn.dataset.copyTool], null, 2), btn);
  if (btn.dataset.copySchema != null) {
    const schema = toolParts(tools[+btn.dataset.copySchema], +btn.dataset.copySchema).parameters;
    return copyText(typeof schema === "string" ? schema : JSON.stringify(schema, null, 2), btn);
  }
  if (btn.dataset.copyRendered) {
    const rendered = trace.rendered_tokens;
    const selected = currentBranchIdx === -1 ? rendered?.all_nodes : rendered?.paths?.[currentBranchIdx];
    if (btn.dataset.copyRendered === "text") {
      if (selected?.text != null) copyText(selected.text, btn);
      return;
    }
    const path = currentPath(trace, traceBranches(trace));
    const ids = path.flatMap((index) => trace.nodes?.[index]?.token_ids || []);
    return copyText(JSON.stringify(ids), btn);
  }
});
$("#tm-meta").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copytext]");
  if (btn) copyText(btn.dataset.copytext, btn);
});

function resizeCharts() {
  for (const entry of state.metrics.charts) {
    if (entry.u) entry.u.setSize({ width: chartWidth(entry.card), height: chartHeight() });
    // unmounted (lazy) and no-data cards track the pane height too
    if (entry.card.style.minHeight) entry.card.style.minHeight = `${chartHeight() + 40}px`;
    const empty = entry.card.querySelector(".chart-empty");
    if (empty) empty.style.height = `${chartHeight()}px`;
  }
}
window.addEventListener("resize", resizeCharts);

function savePrefs() {
  localStorage.setItem(
    "prl-dash",
    JSON.stringify({
      smooth: state.metrics.smooth,
      allLayout: state.metrics.allLayout,
      paneMin: state.metrics.paneMin,
      paneH: state.metrics.paneH,
      paneOrder: state.metrics.paneOrder,
      metricsMode: state.metrics.mode,
      metricsSearch: state.metrics.search,
      collapsedSections: [...state.metrics.collapsedSections],
      traceErrorsOnly: state.traces.errorsOnly,
      traceMode: state.traces.mode,
      traceKinds: { ...state.traces.kinds },
      traceSortStream: state.traces.sorts.stream,
      traceSortStep: state.traces.sorts.step,
      traceViewMode: state.traces.viewMode,
      traceView,
      logView: state.logs.view,
      logComponents: state.logs.components ? [...state.logs.components] : null,
      logLevel: state.logs.level,
      logSearch: $("#log-search").value,
      configSearch: $("#config-search").value,
      tokenSignal: $("#token-signal").value,
      follow: state.follow,
    })
  );
}

function applyPaneSize() {
  $("#metrics-body").style.setProperty("--pane-min", `${state.metrics.paneMin}px`);
  resizeCharts();
}

$("#smooth-range").addEventListener("input", (e) => {
  state.metrics.smooth = +e.target.value;
  $("#smooth-val").textContent = state.metrics.smooth > 1 ? String(state.metrics.smooth) : "off";
  updateCharts();
  savePrefs();
});

let ticking = false;
async function pollDashboard() {
  if (!state.live || ticking) return;
  ticking = true;
  try {
    // keep the run list fresh so new runs register without a page refresh;
    // it runs concurrently with the tab poll (they're independent requests)
    const runsRefresh = loadRuns();
    if (!state.run) {
      await runsRefresh;
      const first = state.runs[0]?.name;
      if (first) await selectRun(first);
      return;
    }
    renderOverview(); // keeps the duration field ticking
    syncDressedSelects();
    if (state.tab === "metrics" && state.metrics.loaded) await fetchMetrics();
    else if (state.tab === "logs" && state.logs.loaded) await pollLogs();
    else if (state.tab === "traces" && state.traces.loaded) await refreshTraces();
    else if (state.tab === "report" && state.report.loaded) await refreshReport();
    await runsRefresh;
  } catch (err) {
    console.warn("poll failed", err);
  } finally {
    ticking = false;
  }
}

setInterval(pollDashboard, POLL_MS);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) pollDashboard();
});

(async function init() {
  $("#smooth-range").value = state.metrics.smooth;
  $("#smooth-val").textContent = state.metrics.smooth > 1 ? String(state.metrics.smooth) : "off";
  $("#metrics-search").value = state.metrics.search;
  state.logs.level = LOG_LEVELS.some(([level]) => level === prefs.logLevel) ? prefs.logLevel : "DEBUG";
  renderLogLevel();
  $("#log-search").value = prefs.logSearch ?? "";
  $("#config-search").value = prefs.configSearch ?? "";
  const signal = prefs.tokenSignal ?? "";
  $("#token-signal").value = $(`#token-signal option[value="${CSS.escape(signal)}"]`) ? signal : "";
  $("#follow-toggle").checked = state.follow;
  for (const sel of ["#run-select", "#trace-env", "#trace-sort", "#tm-env", "#tm-sort", "#config-attempt-select", "#attempt-select", "#token-signal", "#report-select"])
    dressSelect($(sel));
  syncTraceFilterControls();
  setActive("#metrics-mode", "mode", state.metrics.mode);
  setActive("#all-layout", "layout", state.metrics.allLayout);
  $("#all-layout").hidden = state.metrics.mode !== "all";
  setActive("#log-view", "view", state.logs.view);
  applyPaneSize();
  const params = new URLSearchParams(location.hash.slice(1));
  state.tab = params.get("tab") || "metrics";
  state.report.wanted = params.get("report");
  hadHashRun = !!params.get("run");
  setActive("#tabs", "tab", state.tab);
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${state.tab}`));
  await loadRuns();
  const wanted = params.get("run");
  const run = state.runs.find((r) => r.name === wanted)?.name ?? state.runs[0]?.name;
  if (run) await selectRun(run);
  else $("#metrics-body").innerHTML = emptyState("no runs found", `nothing to show in ${state.outputDir ?? "the output directory"}`);
  connectViewEvents();
})();
