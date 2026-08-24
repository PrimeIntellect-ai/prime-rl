/* prime-rl dashboard frontend: metrics (wandb-overview replica), merged logs, trace viewer.
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
const POLL_MS = 3000;
const prefs = JSON.parse(localStorage.getItem("prl-dash") || "{}");

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
  config: { loaded: false, files: [], file: null, fmt: "toml", cache: new Map() },
  logs: {
    loaded: false, attempt: "latest", attempts: [], files: [], paneFile: {},
    components: prefs.logComponents ? new Set(prefs.logComponents) : null,
    view: prefs.logView ?? "merge", level: "DEBUG", maximized: null, buffers: new Map(), gseq: 0,
  },
  traces: {
    loaded: false, steps: [], step: null, env: "",
    kind: "train",
    preferred: "effective",
    subset: "effective",
    errorsOnly: prefs.traceErrorsOnly ?? false,
    sort: (prefs.traceSort ?? "line:asc").split(":")[0],
    order: (prefs.traceSort ?? "line:asc").split(":")[1],
  },
};

function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return "–";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e6 || abs < 1e-3) return v.toExponential(2);
  if (abs >= 100) return v.toFixed(1);
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}
const fmtReward = (v) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(3));
/* compact human counts that never overflow: 1.1K, 2.2M, 3.3B */
function fmtCompact(n) {
  if (n == null || Number.isNaN(n)) return "–";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${+(n / 1e9).toFixed(abs >= 1e10 ? 0 : 1)}B`;
  if (abs >= 1e6) return `${+(n / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (abs >= 1000) return `${+(n / 1000).toFixed(abs >= 10000 ? 0 : 1)}K`;
  return String(n);
}
function fmtCost(v) {
  if (v == null || Number.isNaN(v)) return "–";
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
  const fresh = state.runs.find((r) => r.name === current);
  if (fresh && state.meta) {
    Object.assign(state.meta, { updated: fresh.updated, started: fresh.started, last_step: fresh.last_step });
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
  // rl: train/eval + all/effective per step - sft: eval only - eval: neither, no steps
  $("#trace-kind").hidden = isEval || state.meta?.type === "sft";
  $("#trace-subset").hidden = isEval;
  $("#tm-step-prev").hidden = isEval;
  $("#tm-step-next").hidden = isEval;
  $("#tm-kind").hidden = isEval || state.meta?.type === "sft";
  $("#tm-subset").hidden = isEval;
  $(".tm-filterhead").hidden = isEval;
}

async function selectRun(name) {
  if (!name) return;
  state.run = name;
  state.compare = { runs: [], data: new Map() };
  $("#run-select").value = name;
  state.meta = state.runs.find((r) => r.name === name) ?? (await api(`/api/runs/${encodeURIComponent(name)}`));
  state.metrics = {
    ...state.metrics,
    loaded: false, offset: 0, byKey: new Map(), charts: [], renderedKeys: -1,
    timeKeys: new Set(), timeZero: null, maxStep: null,
    evalEtag: null, evalCount: 0, evalCost: null,
  };
  if (state.meta?.type === "eval") fetchEvalSeries(); // populates the overview cost early
  state.config = { loaded: false, files: [], file: null, fmt: state.config.fmt, cache: new Map() };
  state.logs = { ...state.logs, loaded: false, attempt: "latest", files: [], paneFile: {}, maximized: null, buffers: new Map() };
  state.traces = { ...state.traces, loaded: false, steps: [], step: null, env: "", episodes: [], etag: null, kind: "train", subset: state.traces.preferred };
  applyRunTypeControls();
  renderOverview();
  renderCompareMenu();
  updateHash();
  await activateTab(state.tab, true);
}

function fmtDuration(secs) {
  if (secs == null || !isFinite(secs) || secs < 0) return "–";
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
  if (!ts) return "–";
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
function envListField(envs) {
  if (!envs?.length) return `<span class="val">–</span>`;
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
  const duration = meta.started && durationEnd ? fmtDuration(durationEnd - meta.started) : "–";
  const field = ([label, value]) => `<div class="ov-field"><span class="lbl">${label}</span>${value}</div>`;
  // rollout dirs can run one step past max_steps (the final ship drains late
  // arrivals), so the headline caps at the configured horizon
  const shownStep = step != null && meta.max_steps ? Math.min(step, meta.max_steps) : step;
  const stepText = `${shownStep != null ? shownStep.toLocaleString() : "–"}/${meta.max_steps ? meta.max_steps.toLocaleString() : "∞"}`;
  const left = [
    ["status", `<span class="badge st-${status}">${status}</span>`],
    ["type", `<span class="val">${esc((meta.type ?? "–").toUpperCase())}</span>`],
    meta.type === "eval"
      ? ["episodes", `<span class="val">${step != null ? step.toLocaleString() : "–"}</span>`]
      : ["step", `<span class="val">${stepText}</span>`],
    ["model", `<span class="val" title="${esc(meta.model ?? "")}">${esc(meta.model ?? "–")}</span>`],
    ...(meta.type === "eval"
      ? [["env", `<span class="val" title="${esc(meta.env ?? "")}">${esc(meta.env ?? "–")}</span>`]]
      : [
          meta.type === "sft"
            ? ["dataset", `<span class="val" title="${esc(meta.dataset ?? "")}">${esc(meta.dataset ?? "–")}</span>`]
            : ["train envs", envListField(meta.train_envs)],
          ["eval envs", envListField(meta.eval_envs)],
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
  location.hash = `#run=${encodeURIComponent(state.run || "")}&tab=${state.tab}`;
}

async function activateTab(tab, force = false) {
  if (tab === state.tab && !force) return;
  state.tab = tab;
  setActive("#tabs", "tab", tab);
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${tab}`));
  updateHash();
  if (tab === "metrics" && !state.metrics.loaded) await initMetrics();
  if (tab === "config" && !state.config.loaded) await initConfig();
  if (tab === "logs" && !state.logs.loaded) await initLogs();
  if (tab === "traces" && !state.traces.loaded) await initTraces();
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
      if (key === "step" || key === "time" || key === "_timestamp" || typeof value !== "number") continue;
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
      if (data.offset >= (data.size ?? data.offset)) break;
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
    data = await api(`/api/runs/${encodeURIComponent(state.run)}/rollouts/0/eval/all/series?${qs}`);
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
  const costs = (data.series.cost || []).filter((v) => v != null);
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
    if (v == null) return "–";
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
      ...COMMON_METRICS.map((m) => ({ regex: `eval/${envPattern}/${m}` })),
      ...COMMON_REGEXES.map((r) => ({ regex: `eval/${envPattern}/${r}` })),
    ],
  });
  const sections = [];
  const evalEnvs = meta.eval_envs || [];
  if (meta.type === "sft") {
    sections.push({ name: "train", panels: SFT_TRAIN_METRICS.map((m) => ({ metric: m })) });
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
        points: { show: false },
      });
      const m = { label: labels[mainIdx] || "value", stat: statOf(strand.main.key) ?? "value", color, dataIdx: cols.length };
      meta.push(m);
      mainIdx++;
      const aux = [strand.lo, strand.hi, ...strand.overlays].filter(Boolean);
      const bandIdx = {};
      for (const s of aux) {
        cols.push({ s, role: "aux" });
        uSeries.push({ stroke: hexToRgba(color, 0.55), width: 1, dash: [3, 3], spanGaps: true, points: { show: false } });
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
    body.innerHTML = emptyState("no metrics", "this run has no metrics.jsonl yet");
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
        if (section.configured && !activeFilter) grid.innerHTML = `<div class="chart-empty">no eval data yet</div>`;
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


/* both views are fetched once per run, so the TOML/JSON toggle never waits on
   the network */
async function fetchConfigText(file) {
  const cache = state.config.cache;
  if (cache.has(file)) return cache.get(file);
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/config?file=${encodeURIComponent(file)}`);
  let text = data.content;
  try {
    text = JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    /* show raw content if not valid JSON */
  }
  cache.set(file, text);
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
    const test = (line) => {
      re.lastIndex = 0;
      return re.test(line);
    };
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

/* TOML = the launch config as it was passed, JSON = the concatenated resolved dumps */
function configFileFor(fmt) {
  const files = state.config.files || [];
  return fmt === "toml" ? files.find((f) => f.endsWith(".toml")) : files.find((f) => f === "resolved");
}

function renderConfigFormat() {
  for (const btn of document.querySelectorAll("#config-format button")) {
    btn.disabled = !configFileFor(btn.dataset.fmt);
    btn.classList.toggle("active", btn.dataset.fmt === state.config.fmt);
  }
}

async function initConfig() {
  state.config.loaded = true;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/configs`);
  state.config.files = data.files;
  if (!data.files.length) {
    renderConfigFormat();
    $("#config-view").innerHTML = emptyState("no configs", "this run has no configs/ directory");
    return;
  }
  if (!configFileFor(state.config.fmt)) state.config.fmt = configFileFor("toml") ? "toml" : "json";
  state.config.file = configFileFor(state.config.fmt);
  renderConfigFormat();
  await loadConfig();
  const other = configFileFor(state.config.fmt === "toml" ? "json" : "toml");
  if (other) fetchConfigText(other); // warm the other side of the toggle
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
  $("#attempt-select").innerHTML = logs.attempts
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
  logs.attempt = data.attempt;
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
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/rollouts`);
  traces.steps = data.steps;
  if (traces.step == null && data.steps.length) {
    // default to the newest step that already shipped the preferred subset —
    // the newest step is usually in-flight with only "all" (no advantages yet)
    const preferred = traces.preferred;
    const newestFirst = [...data.steps].reverse();
    const shipped =
      newestFirst.find((s) => s.available[`${traces.kind}/${preferred}`]) ??
      newestFirst.find((s) => Object.keys(s.available).some((k) => k.endsWith(`/${preferred}`)));
    traces.step = (shipped ?? data.steps[data.steps.length - 1]).step;
    adjustKindSubset();
  } else if (traces.step != null) {
    // the preferred subset may have shipped since the last poll (a live step's
    // effective file lands late) — re-adjust and reload when it changes
    const before = `${traces.kind}/${traces.subset}`;
    adjustKindSubset();
    if (`${traces.kind}/${traces.subset}` !== before) await loadEpisodes();
  }
  renderStepControl();
}

function stepInfo(step) {
  return state.traces.steps.find((s) => s.step === step);
}

function adjustKindSubset() {
  const traces = state.traces;
  const available = stepInfo(traces.step)?.available || {};
  const hasTrain = available["train/all"] || available["train/effective"];
  const hasEval = available["eval/all"] || available["eval/effective"];
  if (traces.kind === "train" && !hasTrain && hasEval) traces.kind = "eval";
  if (traces.kind === "eval" && !hasEval && hasTrain) traces.kind = "train";
  // fall back when the preferred subset is missing at this step (e.g. the latest
  // step's effective file lands only at ship time), but return to it as soon as
  // it exists again — advantages are only stamped on effective records
  const preferred = traces.preferred;
  const other = preferred === "all" ? "effective" : "all";
  if (available[`${traces.kind}/${preferred}`]) traces.subset = preferred;
  else if (available[`${traces.kind}/${other}`]) traces.subset = other;
  for (const sel of ["#trace-kind", "#tm-kind"]) {
    $(`${sel} [data-kind=train]`).disabled = !hasTrain;
    $(`${sel} [data-kind=eval]`).disabled = !hasEval;
    setActive(sel, "kind", traces.kind);
  }
  setActive("#trace-subset", "subset", traces.subset);
  setActive("#tm-subset", "subset", traces.subset);
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
      ? ""
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

function showTraceEmpty(title, detail) {
  $("#episode-table-wrap").hidden = true;
  const el = $("#trace-empty");
  el.hidden = false;
  el.innerHTML = emptyState(title, detail);
}

async function loadEpisodes() {
  const traces = state.traces;
  updateTraceFilterBtn();
  if (traces.step == null) {
    $("#trace-status").textContent = "";
    showTraceEmpty("no traces", "this run has no saved traces yet");
    return;
  }
  const qs = new URLSearchParams({ sort: traces.sort, order: traces.order, errors_only: traces.errorsOnly });
  if (traces.env) qs.set("env", traces.env);
  // etag = the file size the client last saw: while the file is unchanged the
  // poll gets a tiny {unchanged} response instead of thousands of summaries
  const etagKey = JSON.stringify([state.run, traces.step, traces.kind, traces.subset, traces.env, traces.errorsOnly, traces.sort, traces.order]);
  if (traces.etagKey === etagKey && traces.etag) qs.set("etag", traces.etag);
  let data;
  try {
    data = await api(`/api/runs/${encodeURIComponent(state.run)}/rollouts/${traces.step}/${traces.kind}/${traces.subset}?${qs}`);
  } catch {
    $("#trace-status").textContent = "";
    showTraceEmpty("no traces", `no ${traces.kind}/${traces.subset} traces at step ${traces.step}`);
    return;
  }
  if (data.unchanged) return;
  const fresh = traces.etagKey !== etagKey;
  traces.etag = data.etag;
  traces.etagKey = etagKey;
  traces.episodes = data.episodes;
  $("#trace-empty").hidden = true;
  $("#episode-table-wrap").hidden = false;
  const envSel = $("#trace-env");
  const currentEnv = traces.env;
  envSel.innerHTML =
    `<option value="">all envs</option>` +
    data.envs.map((e) => `<option value="${esc(e)}" ${e === currentEnv ? "selected" : ""}>${esc(e)}</option>`).join("");
  if (!data.total) {
    $("#trace-status").textContent = "";
    showTraceEmpty("no episodes", "nothing matches the current filters");
    return;
  }
  renderEpisodeRows(fresh);
  const fellBack = traces.subset !== traces.preferred && !$("#trace-subset").hidden;
  $("#trace-status").textContent = `${data.total} episodes${fellBack ? ` · no ${traces.preferred} at this step` : ""}`;
}

function episodeRowHtml(ep) {
  return `<tr data-line="${ep.line}">
        <td class="muted">${ep.line}</td>
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
        <td>${ep.branches ?? ""}</td>
        <td class="muted">${esc(ep.stop_condition ?? "")}</td>
        <td class="${ep.ok && !ep.num_errors ? "status-ok" : "status-err"}">${ep.ok && !ep.num_errors ? "ok" : `${ep.num_errors || ""} err`}</td>
      </tr>`;
}

/* windowed table: only rows in (and around) the viewport exist in the DOM,
   spacer rows stand in for the rest — thousands of episodes stay instant */
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
  const pad = (h) => (h > 0 ? `<tr class="vpad"><td colspan="10" style="height:${h}px"></td></tr>` : "");
  tbody.innerHTML =
    pad(start * episodeRowH) +
    episodes.slice(start, end).map(episodeRowHtml).join("") +
    pad((episodes.length - end) * episodeRowH);
}

async function initTraces() {
  state.traces.loaded = true;
  await loadRollouts();
  adjustKindSubset();
  await loadEpisodes();
}

/* ----------------------------------------------------------- episode view */

let currentEpisode = null;
let currentLine = null;
let currentTraceIdx = 0;
let currentBranchIdx = 0;

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
    `<div class="tm-item ${e.line === currentLine ? "active" : ""}" data-line="${e.line}">` +
    `<span class="tm-num">#${e.line}</span><span class="tm-env muted" title="${esc(e.env ?? "")}">${esc(e.env ?? "")}</span>` +
    `<span class="tm-reward ${rewardClass(e.reward)}">${fmtReward(e.reward)}</span></div>`
  );
}

/* windowed like the episode table — only the visible slice is in the DOM */
let tmItemH = 0;

function renderRolloutWindow() {
  const list = $("#tm-list");
  const episodes = filteredRollouts();
  $("#tm-count").textContent = fmtCompact(episodes.length);
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

function stepRollout(delta) {
  const episodes = filteredRollouts();
  const idx = episodes.findIndex((e) => e.line === currentLine);
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

function fetchEpisode(line, withTokens) {
  const traces = state.traces;
  const qs = withTokens ? "?tokens=true" : "";
  return api(`/api/runs/${encodeURIComponent(state.run)}/rollouts/${traces.step}/${traces.kind}/${traces.subset}/${line}${qs}`);
}

/* token strings multiply the payload of a big episode, so they are fetched only
   while a token signal is selected — the plain view ships the raw record */
async function ensureTokens() {
  if (!currentEpisode || currentEpisode._hasTokens || !$("#token-signal").value) return;
  const line = currentLine;
  const episode = await fetchEpisode(line, true);
  if (line !== currentLine) return;
  episode._hasTokens = true;
  currentEpisode = episode;
}

async function openEpisode(line) {
  $("#trace-modal").hidden = false;
  $("#drawer-backdrop").hidden = false;
  currentLine = line;
  renderModalStep();
  renderRolloutList();
  $("#tm-messages").innerHTML = `<div class="chart-empty">loading episode…</div>`;
  $("#tm-meta").innerHTML = "";
  const withTokens = !!$("#token-signal").value;
  const episode = await fetchEpisode(line, withTokens);
  if (line !== currentLine) return; // user already moved to another rollout
  episode._hasTokens = withTokens;
  currentEpisode = episode;
  currentTraceIdx = 0;
  currentBranchIdx = 0;
  renderEpisode();
}

function closeDrawer() {
  $("#trace-modal").hidden = true;
  $("#drawer-backdrop").hidden = true;
  currentEpisode = null;
  currentLine = null;
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

function renderTokenNode(node, signal, maxAbsAdv) {
  const ids = node.token_ids || [];
  const strs = node.token_strs;
  const logprobAt = alignedSignal(node, node.logprobs);
  const advantageAt = alignedSignal(node, node.advantages);
  const spans = ids.map((id, i) => {
    const text = strs ? strs[i] : ` ${id} `;
    const logprob = logprobAt(i), advantage = advantageAt(i);
    let bg = "";
    if (signal === "advantage" && advantage != null && maxAbsAdv > 0) {
      const alpha = Math.min(1, Math.abs(advantage) / maxAbsAdv) * 0.45;
      bg = `background:rgba(${advantage > 0 ? "182,255,60" : "255,69,57"},${alpha.toFixed(3)})`;
    } else if (signal === "logprob" && logprob != null) {
      bg = `background:rgba(183,166,250,${(Math.min(1, -logprob / 6) * 0.6).toFixed(3)})`;
    } else if (signal === "mask" && node.mask?.[i]) {
      bg = "background:rgba(74,158,255,0.3)";
    } else if (signal === "is_content" && node.is_content?.[i]) {
      bg = "background:rgba(252,218,164,0.28)";
    }
    let tip = `#${i} id=${id}`;
    if (signal === "advantage" && advantage != null) tip += ` adv=${fmtNum(advantage)}`;
    else if (signal === "logprob" && logprob != null)
      tip += ` lp=${logprob.toFixed(4)} (${(Math.exp(logprob) * 100).toFixed(1)}%)`;
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

function reasoningBlock(content) {
  const text = typeof content === "string" ? content : JSON.stringify(content, null, 2);
  return (
    `<details class="sub"><summary><span class="sub-name">Reasoning</span>` +
    `<span class="entry-preview">${preview(text, 140)}</span>` +
    `<span class="entry-chev">›</span></summary><div class="entry-body">${esc(text)}</div></details>`
  );
}

let entriesObserver = null;

function renderMessages(trace, branches) {
  const container = $("#tm-messages");
  entriesObserver?.disconnect();
  if (!trace) {
    container.innerHTML = emptyState("no traces", "this episode carries no trace data");
    return;
  }
  const signal = $("#token-signal").value;
  const path = currentPath(trace, branches);
  const concatenated = currentBranchIdx === -1;
  let maxAbsAdv = 0;
  for (const node of trace.nodes || [])
    for (const a of node.advantages || []) maxAbsAdv = Math.max(maxAbsAdv, Math.abs(a));
  const callsByNode = new Map((trace.calls || []).map((c) => [c.node, c]));
  const entryHtml = (idx, i) => {
    const node = trace.nodes[idx];
    const role = node.message?.role ?? "?";
    const call = callsByNode.get(idx);
    const chips = [];
    if (concatenated && node.parent != null && node.parent !== idx - 1) chips.push(`↳ branches from ${node.parent + 1}`);
    if (node.sampled) chips.push("sampled");
    if (call?.finish_reason) chips.push(call.finish_reason);
    if (call?.usage) chips.push(`${call.usage.prompt_tokens ?? "?"}→${call.usage.completion_tokens ?? "?"} tok`);
    else if (node.token_ids?.length) chips.push(`${node.token_ids.length} tok`);
    const text = messageText(node.message);
    const body = signal && node.token_ids?.length ? renderTokenNode(node, signal, maxAbsAdv) : esc(text);
    const subs = [];
    const reasoning = node.message?.reasoning_content ?? node.message?.reasoning;
    if (reasoning) subs.push(reasoningBlock(reasoning));
    const toolCalls = (node.message?.tool_calls || []).map(toolCallHtml);
    return (
      `<details class="entry ${esc(role)}"${role === "system" ? "" : " open"}>` +
      `<summary><span class="entry-num">${String(i + 1).padStart(2, "0")}</span>` +
      `<span class="entry-role">${esc(role)}</span>` +
      `<span class="entry-preview">${preview(text, 180)}</span>` +
      chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("") +
      `<button class="icon-btn" data-copy="${idx}" title="copy message">${COPY_SVG}</button>` +
      `<span class="entry-chev">›</span></summary>` +
      subs.join("") +
      (body ? `<div class="entry-body">${body}</div>` : "") +
      toolCalls.join("") +
      `</details>`
    );
  };
  // long traces render in chunks as the reader scrolls — a 1MB episode with
  // hundreds of turns paints the first screen immediately
  const CHUNK = 30;
  let rendered = Math.min(path.length, CHUNK);
  container.innerHTML =
    path.slice(0, rendered).map(entryHtml).join("") +
    (rendered < path.length ? `<div id="tm-more" class="chart-empty">scroll for ${path.length - rendered} more entries</div>` : "");
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

    const usage = { input: 0, output: 0, reasoning: 0, cached: 0 };
    let hasUsage = false;
    for (const call of trace.calls || []) {
      const u = call.usage || {};
      if (u.prompt_tokens != null || u.completion_tokens != null) hasUsage = true;
      usage.input += u.prompt_tokens ?? 0;
      usage.output += u.completion_tokens ?? 0;
      usage.reasoning += u.completion_tokens_details?.reasoning_tokens ?? 0;
      usage.cached += u.prompt_tokens_details?.cached_tokens ?? 0;
    }
    if (hasUsage) {
      parts.push(`<div class="meta-sec">usage</div>`);
      parts.push(metaRow("input tokens", fmtCompact(usage.input)));
      parts.push(metaRow("output tokens", fmtCompact(usage.output)));
      if (usage.reasoning) parts.push(metaRow("reasoning tokens", fmtCompact(usage.reasoning)));
      if (usage.cached) parts.push(metaRow("cached tokens", fmtCompact(usage.cached)));
      // API-priced runs report per-call cost; local deployments usually don't
      const traceCost = (t) => (t.calls || []).reduce((acc, c) => acc + (c.usage?.cost ?? 0), 0);
      const allTraces = ep.traces || [];
      if (allTraces.some((t) => (t.calls || []).some((c) => c.usage?.cost != null)))
        parts.push(metaRow("cost", fmtCost(allTraces.reduce((acc, t) => acc + traceCost(t), 0))));
      parts.push(metaRow("total tokens", fmtCompact(usage.input + usage.output)));
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
  }

  parts.push(`<div class="meta-sec">identity</div>`);
  parts.push(metaRow("episode ID", ep.id, true));
  if (trace?.id) parts.push(metaRow("trace ID", trace.id, true));
  if (ep.group?.id) parts.push(metaRow("group ID", ep.group.id, true));
  if (trace?.agent?.runtime?.id) parts.push(metaRow("runtime ID", trace.agent.runtime.id, true));

  const errors = [...(ep.errors || []), ...(trace?.errors || [])];
  if (errors.length)
    parts.push(
      `<details class="meta-fold" open><summary>errors (${errors.length})</summary>` +
        `<pre class="json">${esc(JSON.stringify(errors, null, 2))}</pre></details>`
    );

  $("#tm-meta").innerHTML = parts.join("");
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
  // multi-agent episodes: label each trace by its agent name (seat), index only
  // as a tiebreak when names repeat or are missing
  const names = traces.map((t) => t.agent?.name);
  const label = (i) => (names[i] && names.indexOf(names[i]) === names.lastIndexOf(names[i]) ? names[i] : `${names[i] ?? "trace"} ${i}`);
  traceTabs.innerHTML =
    traces.length > 1
      ? traces
          .map((_, i) => `<button data-trace="${i}" class="${i === currentTraceIdx ? "active" : ""}">${esc(label(i))}</button>`)
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
  $("#tm-tabs-row").hidden = traceTabs.hidden && branchTabs.hidden;
  renderRolloutList();
  renderMessages(trace, branches);
  renderMeta(ep, trace, branches);
}

/* ---------------------------------------------------------------- wiring */

$("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
$("#live-toggle").addEventListener("change", (e) => (state.live = e.target.checked));
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
  if (!select || select.closest(".dd-wrap")) return;
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
function updateTraceFilterBtn() {
  const t = state.traces;
  $("#trace-filter-btn").classList.toggle("active", !!(t.env || t.errorsOnly || t.sort !== "line"));
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
async function setTraceKind(kind, inModal = false) {
  state.traces.kind = kind;
  adjustKindSubset();
  await loadEpisodes();
  savePrefs();
  if (inModal) await reopenFirstEpisode();
}

async function setTraceSubset(subset, inModal = false) {
  state.traces.preferred = subset;
  state.traces.subset = subset;
  adjustKindSubset();
  await loadEpisodes();
  savePrefs();
  if (inModal) await reopenFirstEpisode();
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

for (const [sel, inModal] of [["#trace-kind", false], ["#tm-kind", true]])
  document.querySelectorAll(`${sel} button`).forEach((b) =>
    b.addEventListener("click", () => {
      if (b.disabled || b.dataset.kind === state.traces.kind) return;
      setTraceKind(b.dataset.kind, inModal);
    })
  );
for (const [sel, inModal] of [["#trace-subset", false], ["#tm-subset", true]])
  document.querySelectorAll(`${sel} button`).forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.subset === state.traces.subset) return;
      setTraceSubset(b.dataset.subset, inModal);
    })
  );
$("#trace-env").addEventListener("change", (e) => { state.traces.env = e.target.value; loadEpisodes(); });
$("#trace-errors").addEventListener("change", (e) => { state.traces.errorsOnly = e.target.checked; loadEpisodes(); savePrefs(); });
$("#trace-sort").addEventListener("change", (e) => {
  [state.traces.sort, state.traces.order] = e.target.value.split(":");
  loadEpisodes();
  savePrefs();
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
$("#tm-list").addEventListener("scroll", rafThrottle(renderRolloutWindow));
$("#drawer-close").addEventListener("click", closeDrawer);
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
  const tok = e.target.closest(".tok[data-tip]");
  if (!tok) return;
  tokTip.textContent = tok.dataset.tip;
  tokTip.hidden = false;
  const rect = tok.getBoundingClientRect();
  tokTip.style.left = `${Math.min(rect.left, window.innerWidth - tokTip.offsetWidth - 12)}px`;
  tokTip.style.top = `${rect.top - tokTip.offsetHeight - 6}px`;
});
$("#tm-messages").addEventListener("mouseout", (e) => {
  if (e.target.closest(".tok[data-tip]")) tokTip.hidden = true;
});
$("#token-signal").addEventListener("change", async () => {
  await ensureTokens();
  renderEpisode();
  savePrefs();
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
$("#tm-prev").addEventListener("click", () => stepRollout(-1));
$("#tm-next").addEventListener("click", () => stepRollout(1));
$("#tm-collapse").addEventListener("click", () =>
  document.querySelectorAll("#tm-messages details.entry").forEach((d) => (d.open = false))
);
$("#tm-expand").addEventListener("click", () =>
  document.querySelectorAll("#tm-messages details").forEach((d) => (d.open = true))
);
$("#tm-messages").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copy]");
  if (!btn) return;
  e.preventDefault();
  e.stopPropagation();
  const node = currentEpisode?.traces?.[currentTraceIdx]?.nodes?.[+btn.dataset.copy];
  if (node) copyText(messageText(node.message), btn);
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
      traceSort: `${state.traces.sort}:${state.traces.order}`,
      logView: state.logs.view,
      logComponents: state.logs.components ? [...state.logs.components] : null,
      logLevel: state.logs.level,
      logSearch: $("#log-search").value,
      configSearch: $("#config-search").value,
      tokenSignal: $("#token-signal").value,
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

let tickCount = 0;
let ticking = false;
setInterval(async () => {
  if (!state.live || ticking) return;
  ticking = true;
  tickCount++;
  try {
    // keep the run list fresh so new runs register without a page refresh;
    // it runs concurrently with the tab poll (they're independent requests)
    const runsRefresh = tickCount % 3 === 0 || !state.run ? loadRuns() : null;
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
    else if (state.tab === "traces" && state.traces.loaded) {
      await loadRollouts();
      if (tickCount % 5 === 0) await loadEpisodes();
    }
    await runsRefresh;
  } catch (err) {
    console.warn("poll failed", err);
  } finally {
    ticking = false;
  }
}, POLL_MS);

(async function init() {
  $("#smooth-range").value = state.metrics.smooth;
  $("#smooth-val").textContent = state.metrics.smooth > 1 ? String(state.metrics.smooth) : "off";
  $("#metrics-search").value = state.metrics.search;
  state.logs.level = LOG_LEVELS.some(([level]) => level === prefs.logLevel) ? prefs.logLevel : "DEBUG";
  renderLogLevel();
  $("#log-search").value = prefs.logSearch ?? "";
  $("#config-search").value = prefs.configSearch ?? "";
  $("#token-signal").value = prefs.tokenSignal ?? "";
  for (const sel of ["#run-select", "#trace-env", "#trace-sort", "#attempt-select", "#token-signal"])
    dressSelect($(sel));
  $("#trace-sort").value = `${state.traces.sort}:${state.traces.order}`;
  $("#trace-errors").checked = state.traces.errorsOnly;
  setActive("#metrics-mode", "mode", state.metrics.mode);
  setActive("#all-layout", "layout", state.metrics.allLayout);
  $("#all-layout").hidden = state.metrics.mode !== "all";
  setActive("#log-view", "view", state.logs.view);
  applyPaneSize();
  const params = new URLSearchParams(location.hash.slice(1));
  state.tab = params.get("tab") || "metrics";
  setActive("#tabs", "tab", state.tab);
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${state.tab}`));
  await loadRuns();
  const wanted = params.get("run");
  const run = state.runs.find((r) => r.name === wanted)?.name ?? state.runs[0]?.name;
  if (run) await selectRun(run);
  else $("#metrics-body").innerHTML = emptyState("no runs found", `nothing to show in ${state.outputDir ?? "the output directory"}`);
})();
