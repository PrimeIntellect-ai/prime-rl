/* prime-rl dashboard frontend: metrics (wandb-overview replica), merged logs, trace viewer. */

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
    loaded: false, offset: 0, byKey: new Map(), mode: "overview", search: "",
    charts: [], renderedKeys: -1, timeKeys: new Set(), timeZero: null, collapsedSections: new Set(),
    smooth: prefs.smooth ?? 1, paneMin: prefs.paneMin ?? 320, paneH: prefs.paneH ?? 150,
    paneOrder: prefs.paneOrder ?? {},
  },
  compare: { runs: [], data: new Map() },
  config: { loaded: false, files: [], file: null },
  logs: { loaded: false, attempt: "latest", attempts: [], files: [], paneFile: {}, maximized: null, buffers: new Map(), gseq: 0 },
  traces: { loaded: false, steps: [], step: null, kind: "train", subset: "effective", page: 0, limit: 5000, total: 0, env: "", errorsOnly: false, sort: "line", order: "asc" },
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
const fmtBytes = (n) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)}M` : n >= 1024 ? `${(n / 1024).toFixed(0)}K` : `${n}B`);
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const emptyState = (title, detail = "") =>
  `<div class="empty-box"><img src="/static/butterfly-white.svg" alt="">` +
  `<div class="empty-title">${esc(title)}</div>` +
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

async function selectRun(name) {
  if (!name) return;
  state.run = name;
  state.compare = { runs: [], data: new Map() };
  $("#run-select").value = name;
  state.meta = await api(`/api/runs/${encodeURIComponent(name)}`);
  state.metrics = {
    ...state.metrics,
    loaded: false, offset: 0, byKey: new Map(), charts: [], renderedKeys: -1,
    timeKeys: new Set(), timeZero: null,
  };
  state.config = { loaded: false, files: [], file: null };
  state.logs = { ...state.logs, loaded: false, attempt: "latest", files: [], paneFile: {}, maximized: null, buffers: new Map() };
  state.traces = { ...state.traces, loaded: false, steps: [], step: null, page: 0, env: "", kind: "train", subset: "effective" };
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
  let step = null;
  for (const [key, producers] of state.metrics.byKey) {
    if (state.metrics.timeKeys.has(key)) continue; // time-keyed x values are not steps
    for (const series of producers.values()) for (const s of series.keys()) step = Math.max(step ?? 0, s);
  }
  return step ?? state.meta?.last_step ?? null;
}

function runStatus(step) {
  const meta = state.meta;
  if (meta.updated && Date.now() / 1000 - meta.updated < 180) return "running";
  if (step != null && meta.max_steps && step >= meta.max_steps) return "completed";
  return "stopped";
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
  const pct = meta.max_steps && step != null ? Math.min(100, (step / meta.max_steps) * 100) : null;
  const durationEnd = status === "running" ? Date.now() / 1000 : meta.updated;
  const duration = meta.started && durationEnd ? fmtDuration(durationEnd - meta.started) : "–";
  const fields = [
    ["status", `<span class="badge st-${status}">${status}</span>`],
    ["duration", `<span class="val">${duration}</span>`],
    ["model", `<span class="val">${esc(meta.model ?? "–")}</span>`],
    ["created", `<span class="val">${fmtAgo(meta.created)}</span>`],
  ];
  el.innerHTML =
    `<div class="ov-top">` +
    `<div class="ov-pct"><div class="pct">${pct != null ? `${pct.toFixed(2)}%` : step != null ? `step ${step}` : "–"}</div>` +
    `<div class="steps">${(step ?? 0).toLocaleString()}${meta.max_steps ? ` / ${meta.max_steps.toLocaleString()}` : ""} Steps</div></div>` +
    fields.map(([label, value]) => `<div class="ov-field"><span class="lbl">${label}</span>${value}</div>`).join("") +
    `</div>` +
    (pct != null
      ? `<div class="ov-bar"><div class="fill" style="width:${pct.toFixed(2)}%"></div></div>`
      : step != null
        ? // unbounded run: no percentage — grey track, lime sweep while running
          `<div class="ov-bar indeterminate${status === "running" ? " live" : ""}"><div class="sweep"></div></div>`
        : "");
}

function updateHash() {
  location.hash = `#run=${encodeURIComponent(state.run || "")}&tab=${state.tab}`;
}

async function activateTab(tab, force = false) {
  if (tab === state.tab && !force) return;
  state.tab = tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${tab}`));
  updateHash();
  if (tab === "metrics" && !state.metrics.loaded) await initMetrics();
  if (tab === "config" && !state.config.loaded) await initConfig();
  if (tab === "logs" && !state.logs.loaded) await initLogs();
  if (tab === "traces" && !state.traces.loaded) await initTraces();
}

/* ---------------------------------------------------------------- metrics */

const COMMON_METRICS = ["effective/num_total_tokens/mean", "effective/num_turns/mean", "effective/num_branches/mean"];
const COMMON_REGEXES = ["all/[^/]+/has_error/mean", "effective/[^/]+/is_truncated/mean"];
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
    }
    const producer = isTime ? "infer" : rowProducer(row, meta);
    for (const [key, value] of Object.entries(row)) {
      if (key === "step" || key === "time" || key === "_timestamp" || typeof value !== "number") continue;
      let producers = store.byKey.get(key);
      if (!producers) store.byKey.set(key, (producers = new Map()));
      let series = producers.get(producer);
      if (!series) producers.set(producer, (series = new Map()));
      series.set(x, value);
      if (isTime) store.timeKeys.add(key);
    }
  }
}

async function fetchMetrics() {
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/metrics?offset=${state.metrics.offset}`);
  state.metrics.offset = data.offset;
  if (data.rows.length) {
    ingestInto(state.metrics, data.rows, state.meta);
    renderOverview();
  }
  const grew = await fetchCompares();
  if (data.rows.length || grew) {
    if (state.metrics.byKey.size !== state.metrics.renderedKeys) renderMetricsBody();
    else updateCharts();
  }
  return data.rows.length;
}

async function fetchCompares() {
  let grew = false;
  for (const name of state.compare.runs) {
    let store = state.compare.data.get(name);
    if (!store) {
      store = { offset: 0, byKey: new Map(), timeKeys: new Set(), timeZero: null, meta: null };
      state.compare.data.set(name, store);
    }
    try {
      store.meta ??= await api(`/api/runs/${encodeURIComponent(name)}`);
      const data = await api(`/api/runs/${encodeURIComponent(name)}/metrics?offset=${store.offset}`);
      store.offset = data.offset;
      if (data.rows.length) {
        ingestInto(store, data.rows, store.meta);
        grew = true;
      }
    } catch (err) {
      console.warn(`compare fetch failed for ${name}`, err);
    }
  }
  return grew;
}

function buildSections(meta) {
  const trainSection = (name, scope) => ({
    name,
    panels: [
      ...COMMON_METRICS.map((m) => ({ metric: `${scope}/${m}` })),
      { regex: `${escRe(scope)}/all/[^/]+/reward/mean` },
      { regex: `${escRe(scope)}/effective/[^/]+/reward/mean` },
      ...COMMON_REGEXES.map((r) => ({ regex: `${escRe(scope)}/${r}` })),
    ],
  });
  const evalSection = (name, envPattern) => ({
    name,
    panels: [
      { regex: `eval/${envPattern}/all/[^/]+/avg@.*` },
      { regex: `eval/${envPattern}/effective/[^/]+/avg@.*` },
      ...COMMON_METRICS.map((m) => ({ regex: `eval/${envPattern}/${m}` })),
      ...COMMON_REGEXES.map((r) => ({ regex: `eval/${envPattern}/${r}` })),
    ],
  });
  const sections = [];
  const evalEnvs = meta.eval_envs || [];
  if (meta.type === "sft") {
    sections.push({ name: "train", panels: SFT_TRAIN_METRICS.map((m) => ({ metric: m })) });
    if (evalEnvs.length) sections.push(...evalEnvs.map((e) => evalSection(`eval/${e}`, escRe(e))));
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
  if (evalEnvs.length) sections.push(...evalEnvs.map((e) => evalSection(`eval/${e}`, escRe(e))));
  else sections.push(evalSection("eval", ".*"));
  sections.push({ name: "stability", panels: STABILITY_METRICS.map((m) => ({ metric: m })) });
  sections.push({ name: "inference", panels: INFERENCE_PANELS.map((metrics) => ({ metrics })) });
  sections.push({ name: "performance", panels: PERFORMANCE_METRICS.map((m) => ({ metric: m })) });
  return sections;
}

let activeFilter = null;

function metricsFilter() {
  const query = state.metrics.search.trim();
  if (!query) return null;
  try {
    return new RegExp(query, "i");
  } catch {
    const needle = query.toLowerCase();
    return { test: (k) => k.toLowerCase().includes(needle) };
  }
}

function compareStores() {
  const stores = [{ run: state.run, store: state.metrics }];
  for (const name of state.compare.runs) {
    const store = state.compare.data.get(name);
    if (store) stores.push({ run: name, store });
  }
  return stores;
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

/* comparing runs: color by run; otherwise color by series */
function seriesColors(series) {
  if (state.compare.runs.length) {
    const runs = [state.run, ...state.compare.runs];
    return series.map((s) => PALETTE[Math.max(0, runs.indexOf(s.run)) % PALETTE.length]);
  }
  return series.length > 1 ? series.map((_, i) => PALETTE[i % PALETTE.length]) : [SINGLE_SERIES];
}

function rollingMean(values, window) {
  if (window <= 1) return values;
  return values.map((value, i) => {
    if (value == null) return null;
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) {
      const x = values[j];
      if (x != null) {
        sum += x;
        count++;
      }
    }
    return sum / count;
  });
}

/* each logical series feeds two uPlot series: a raw ghost (visible only while
   smoothing) drawn under the smoothed main line */
function panelData(series) {
  const stepSet = new Set();
  for (const s of series) for (const step of s.points.keys()) stepSet.add(step);
  const steps = [...stepSet].sort((a, b) => a - b);
  const window = state.metrics.smooth;
  const out = [steps];
  for (const s of series) {
    const raw = steps.map((st) => s.points.get(st) ?? null);
    out.push(window > 1 ? raw : raw.map(() => null));
    out.push(window > 1 ? rollingMean(raw, window) : raw);
  }
  return out;
}

function chartHeight() {
  return state.metrics.paneH;
}

/* axis labels get one exponent digit so they fit the 50px gutter */
function fmtAxis(v) {
  if (v == null) return "";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e6 || abs < 1e-3) return v.toExponential(1).replace(".0e", "e");
  return fmtNum(v);
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
  return {
    hooks: {
      init: (u) => {
        tip = document.createElement("div");
        tip.className = "u-tip";
        tip.style.display = "none";
        u.over.appendChild(tip);
        u.over.addEventListener("mouseleave", () => (tip.style.display = "none"));
      },
      setCursor: (u) => {
        const { left, top, idx } = u.cursor;
        if (idx == null || left == null || left < 0) {
          tip.style.display = "none";
          return;
        }
        const x = u.data[0][idx];
        let rows = `<div class="u-tip-x">${timeAxis ? fmtTickDur(x) : `step ${x}`}</div>`;
        let any = false;
        meta.forEach((m, i) => {
          const v = u.data[2 + i * 2][idx]; // the main (smoothed) series
          if (v == null) return;
          any = true;
          rows +=
            `<div class="u-tip-row"><span class="sw" style="background:${m.color}"></span>` +
            `${meta.length > 1 ? `<span class="u-tip-l">${esc(m.label)}</span>` : ""}` +
            `<span class="u-tip-v">${fmtNum(v)}</span></div>`;
        });
        if (!any) {
          tip.style.display = "none";
          return;
        }
        tip.innerHTML = rows;
        tip.style.display = "block";
        let tx = left + 14;
        if (tx + tip.offsetWidth > u.over.clientWidth) tx = left - tip.offsetWidth - 14;
        const ty = Math.max(0, Math.min(u.over.clientHeight - tip.offsetHeight, top - tip.offsetHeight / 2));
        tip.style.transform = `translate(${Math.round(tx)}px, ${Math.round(ty)}px)`;
      },
    },
  };
}

function makeChart(el, labels, colorList, width, timeAxis = false) {
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
  const meta = labels.map((label, i) => ({ label: label || "value", color: colorList[i % colorList.length] }));
  const series = [{ label: timeAxis ? "time" : "step" }];
  for (const m of meta) {
    series.push({ stroke: hexToRgba(m.color, 0.25), width: 1, spanGaps: true, points: { show: false } });
    series.push({ label: m.label, stroke: m.color, width: 1.25, spanGaps: true, points: { show: false } });
  }
  return new uPlot(
    {
      width,
      height: chartHeight(),
      padding: [10, 12, 0, 0],
      cursor: { points: { size: 5 }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [xAxis, { ...axis, size: 50, values: (u, vals) => vals.map(fmtAxis) }],
      legend: { show: false },
      plugins: [tooltipPlugin(meta, timeAxis), unzoomPlugin()],
      series,
    },
    [[], ...meta.flatMap(() => [[], []])],
    el
  );
}

/* Charts below the fold mount only when scrolled into view — the all-metrics
   mode renders hundreds of panels. */
let lazyObserver = null;

function mountChart(entry) {
  if (entry.u || !entry.series.length) return;
  const plotEl = document.createElement("div");
  entry.card.appendChild(plotEl);
  const timeAxis = entry.series.every((s) => s.time);
  entry.u = makeChart(plotEl, seriesLabels(entry.series), seriesColors(entry.series), entry.card.clientWidth - 22, timeAxis);
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
  const entry = { card, panel, u: null, series };
  state.metrics.charts.push(entry);
  if (!series.length) {
    card.insertAdjacentHTML("beforeend", `<div class="chart-empty">no data yet</div>`);
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
  const data = panelData(entry.series);
  entry.u.setData(data);
  const last = entry.series[0]?.points;
  if (last?.size) {
    const maxStep = Math.max(...last.keys());
    entry.card.querySelector(".chart-last").textContent = fmtNum(last.get(maxStep));
  }
}

function updateCharts() {
  for (const entry of state.metrics.charts) updateChart(entry);
}

function addSection(body, name, count) {
  const div = document.createElement("details");
  div.className = "section";
  div.dataset.name = name;
  div.open = !state.metrics.collapsedSections.has(name);
  div.innerHTML =
    `<summary>${esc(name)}${count != null ? ` <span class="muted">${count}</span>` : ""}` +
    `<span class="sec-chev">›</span></summary>`;
  const grid = document.createElement("div");
  grid.className = "chart-grid";
  div.appendChild(grid);
  body.appendChild(div);
  return { div, grid };
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
  activeFilter = metricsFilter();
  if (!state.meta?.has_metrics && !m.byKey.size) {
    body.innerHTML = emptyState("no metrics", "this run has no metrics.jsonl yet");
    $("#metrics-status").textContent = "";
    return;
  }
  if (m.mode === "overview") {
    $("#metrics-status").textContent = "";
    for (const section of buildSections(state.meta)) {
      const { div, grid } = addSection(body, section.name);
      for (const panel of section.panels) renderPanelCard(grid, panel);
      if (!grid.children.length) div.remove();
      else applyPaneOrder(grid);
    }
    if (activeFilter && !body.children.length)
      body.innerHTML = emptyState("no keys match", "no overview panels match the filter");
    return;
  }
  // all: every key charted, grouped by top-level namespace, regex-filtered
  const groups = new Map();
  for (const key of [...m.byKey.keys()].sort()) {
    if (activeFilter && !activeFilter.test(key)) continue;
    const group = key.split("/")[0];
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(key);
  }
  const shown = [...groups.values()].reduce((n, keys) => n + keys.length, 0);
  $("#metrics-status").textContent = activeFilter ? `${shown} / ${m.byKey.size} keys` : "";
  for (const [group, keys] of groups) {
    const { grid } = addSection(body, group, keys.length);
    for (const key of keys) renderPanelCard(grid, { metric: key }, true);
    applyPaneOrder(grid);
  }
  if (!groups.size) body.innerHTML = emptyState("no keys match", `0 of ${m.byKey.size} keys match the filter`);
}

async function initMetrics() {
  state.metrics.loaded = true;
  await fetchMetrics();
  renderMetricsBody();
}

/* ----------------------------------------------------------------- config */

function highlightJson(text) {
  let out = "";
  let last = 0;
  const re = /("(?:[^"\\]|\\.)*")(\s*:)?|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b(?:true|false|null)\b/g;
  let match;
  while ((match = re.exec(text))) {
    out += esc(text.slice(last, match.index));
    if (match[1]) {
      out += match[2]
        ? `<span class="j-key">${esc(match[1])}</span>${match[2]}`
        : `<span class="j-str">${esc(match[1])}</span>`;
    } else if (/^(true|false|null)$/.test(match[0])) {
      out += `<span class="j-lit">${match[0]}</span>`;
    } else {
      out += `<span class="j-num">${match[0]}</span>`;
    }
    last = re.lastIndex;
  }
  return out + esc(text.slice(last));
}

async function loadConfig() {
  const data = await api(
    `/api/runs/${encodeURIComponent(state.run)}/config?file=${encodeURIComponent(state.config.file)}`
  );
  let text = data.content;
  try {
    text = JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    /* show raw content if not valid JSON */
  }
  state.config.text = text;
  $("#config-status").textContent = `configs/${data.file}`;
  applyConfigSearch();
}

/* re-render the config with syntax highlighting, then mark every search hit
   by walking text nodes (keeps syntax spans intact) */
function applyConfigSearch() {
  const view = $("#config-view");
  view.innerHTML = highlightJson(state.config.text ?? "");
  const query = $("#config-search").value.trim();
  const hitsEl = $("#config-hits");
  hitsEl.textContent = "";
  if (!query) return;
  let re;
  try {
    re = new RegExp(query, "gi");
  } catch {
    re = new RegExp(escRe(query), "gi");
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

async function initConfig() {
  state.config.loaded = true;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/configs`);
  state.config.files = data.files;
  const sel = $("#config-file");
  sel.innerHTML = data.files.map((f) => `<option value="${esc(f)}">${esc(f)}</option>`).join("");
  if (!data.files.length) {
    sel.innerHTML = `<option>no configs</option>`;
    sel.disabled = true;
    $("#config-view").innerHTML = emptyState("no configs", "this run has no configs/ directory");
    $("#config-status").textContent = "";
    return;
  }
  sel.disabled = false;
  if (!data.files.includes(state.config.file)) state.config.file = data.files[0];
  sel.value = state.config.file;
  await loadConfig();
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
      label: file.label,
      gseq: state.logs.gseq++,
      t: 0,
      level: null,
    });
  }
  return entries;
}

function retime(buffer) {
  let lastSec = null, dayOffset = 0, lastLevel = null;
  for (const line of buffer.lines) {
    if (line.rawTime != null) {
      if (lastSec != null && line.rawTime < lastSec - 12 * 3600) dayOffset++;
      lastSec = line.rawTime;
    }
    line.t = dayOffset * 86400 + (line.rawTime ?? lastSec ?? 0);
    if (line.ownLevel) lastLevel = line.ownLevel;
    line.level = line.ownLevel ?? lastLevel; // continuation lines (tracebacks, noise) inherit
  }
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

const fmtCount = (n) => (n >= 1000 ? `${Math.floor(n / 1000)}K` : String(n));

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
  for (const pane of LOG_PANES) if (paneFiles(pane).length) for (const id of paneSelectedIds(pane)) ids.add(id);
  return ids;
}

function logFilter() {
  const query = $("#log-search").value.trim();
  if (!query) return null;
  try {
    return new RegExp(query, "i");
  } catch {
    const needle = query.toLowerCase();
    return { test: (s) => s.toLowerCase().includes(needle) };
  }
}

function renderLogPanes() {
  const logs = state.logs;
  $("#attempt-select").innerHTML = logs.attempts
    .map((a) => `<option value="${a}" ${String(a) === String(logs.attempt) ? "selected" : ""}>attempt ${a}</option>`)
    .join("");
  const container = $("#log-panes");
  container.innerHTML = "";
  container.classList.toggle("maxed", !!logs.maximized);
  for (const pane of LOG_PANES) {
    const files = paneFiles(pane);
    if (!files.length) continue;
    const isVirtual = pane.comp === "infer" && logs.paneFile[pane.comp] === "__router__";
    if (!pane.merged && !isVirtual && !files.some((f) => f.id === logs.paneFile[pane.comp]))
      logs.paneFile[pane.comp] = (files.find((f) => f.master) ?? files[0]).id;
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
  if (!container.children.length) container.innerHTML = emptyState("no log files", "this run has no logs yet");
}

function renderLogPane(el) {
  const pane = LOG_PANES.find((p) => p.comp === el.dataset.comp);
  if (!pane) return;
  const stream = el.querySelector(".log-pane-stream");
  const minRank = LEVEL_RANK[$("#log-level").value] ?? 0;
  const filter = logFilter();
  const ids = paneSelectedIds(pane);
  // single-node inference.log interleaves router and engine lines: show only
  // engine lines by default, router lines via the virtual "router" file entry
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
  if (ids.length > 1) lines.sort((a, b) => a.t - b.t || a.gseq - b.gseq);
  const shown = lines.slice(-3000);
  // always follow: stick to the bottom unless the user scrolled up to read
  const pinned = !stream.childElementCount || stream.scrollTop + stream.clientHeight >= stream.scrollHeight - 40;
  stream.innerHTML = shown
    .map((line) => `<div class="ll"><span class="ltext">${(line.html ??= ansiToHtml(line.raw))}</span></div>`)
    .join("");
  el.querySelector(".lp-count").textContent = `${fmtCount(lines.length)} lines`;
  if (pinned) stream.scrollTop = stream.scrollHeight;
}

function renderAllLogPanes() {
  document.querySelectorAll("#log-panes .log-pane").forEach(renderLogPane);
}

async function pollLogs(render = true) {
  const logs = state.logs;
  let changed = false;
  await Promise.all(
    [...allSelectedIds()].map(async (id) => {
      const file = logs.files.find((f) => f.id === id);
      if (!file) return;
      let buffer = logs.buffers.get(id);
      if (!buffer) {
        const chunk = await fetchLogChunk(file, { tail: TAIL_BYTES });
        buffer = { file, lines: parseLines(chunk.text, file), headStart: chunk.start, end: chunk.end, size: chunk.size };
        retime(buffer);
        logs.buffers.set(id, buffer);
        changed = true;
      } else {
        const chunk = await fetchLogChunk(file, { start: buffer.end });
        if (chunk.end > buffer.end) {
          buffer.lines.push(...parseLines(chunk.text, file));
          buffer.end = chunk.end;
          buffer.size = chunk.size;
          if (buffer.lines.length > 20000) {
            buffer.lines.splice(0, buffer.lines.length - 20000);
            buffer.headStart = Math.max(buffer.headStart, 1); // older data no longer contiguous
          }
          retime(buffer);
          changed = true;
        }
      }
    })
  );
  if (changed && render) renderAllLogPanes();
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
    traces.step = data.steps[data.steps.length - 1].step;
    adjustKindSubset();
  }
  renderStepControl();
}

function stepInfo(step) {
  return state.traces.steps.find((s) => s.step === step);
}

function adjustKindSubset() {
  const traces = state.traces;
  const counts = stepInfo(traces.step)?.counts || {};
  const hasTrain = counts["train/all"] || counts["train/effective"];
  const hasEval = counts["eval/all"] || counts["eval/effective"];
  if (traces.kind === "train" && !hasTrain && hasEval) traces.kind = "eval";
  if (traces.kind === "eval" && !hasEval && hasTrain) traces.kind = "train";
  if (!counts[`${traces.kind}/${traces.subset}`]) {
    const other = traces.subset === "all" ? "effective" : "all";
    if (counts[`${traces.kind}/${other}`]) traces.subset = other;
  }
  $("#trace-kind [data-kind=train]").disabled = !hasTrain;
  $("#trace-kind [data-kind=eval]").disabled = !hasEval;
  document.querySelectorAll("#trace-kind button").forEach((b) => b.classList.toggle("active", b.dataset.kind === traces.kind));
  document.querySelectorAll("#trace-subset button").forEach((b) => b.classList.toggle("active", b.dataset.subset === traces.subset));
}

function renderStepControl() {
  const traces = state.traces;
  const steps = traces.steps;
  const idx = steps.findIndex((s) => s.step === traces.step);
  $("#step-blocks").innerHTML = steps
    .map((s, i) => {
      const hasEval = s.counts["eval/all"] || s.counts["eval/effective"];
      return (
        `<span class="sb-cell${i <= idx ? " on" : ""}${hasEval ? " eval" : ""}" data-i="${i}"` +
        ` title="step ${s.step}${hasEval ? " · eval" : ""}"></span>`
      );
    })
    .join("");
  $("#step-prev").disabled = idx <= 0;
  $("#step-next").disabled = idx < 0 || idx >= steps.length - 1;
  const info = stepInfo(traces.step);
  const hasEval = info && (info.counts["eval/all"] || info.counts["eval/effective"]);
  $("#step-label").innerHTML =
    traces.step == null
      ? ""
      : `step ${traces.step}${steps.length > 1 ? ` <span class="muted">(${idx + 1}/${steps.length})</span>` : ""}` +
        (hasEval ? ' <span class="eval-dot" title="eval rollouts"></span>' : "");
}

function selectStepByIndex(index) {
  const step = state.traces.steps[index];
  if (!step || step.step === state.traces.step) return;
  state.traces.step = step.step;
  state.traces.page = 0;
  adjustKindSubset();
  renderStepControl();
  loadEpisodes();
}

function showTraceEmpty(title, detail) {
  $("#episode-table-wrap").hidden = true;
  $("#episode-pager").innerHTML = "";
  const el = $("#trace-empty");
  el.hidden = false;
  el.innerHTML = emptyState(title, detail);
}

async function loadEpisodes() {
  const traces = state.traces;
  updateTraceFilterBtn();
  const tbody = $("#episode-table tbody");
  if (traces.step == null) {
    $("#trace-status").textContent = "";
    showTraceEmpty("no rollouts", "this run has no saved rollouts yet");
    return;
  }
  const qs = new URLSearchParams({
    page: traces.page,
    limit: traces.limit,
    sort: traces.sort,
    order: traces.order,
    errors_only: traces.errorsOnly,
  });
  if (traces.env) qs.set("env", traces.env);
  let data;
  try {
    data = await api(`/api/runs/${encodeURIComponent(state.run)}/rollouts/${traces.step}/${traces.kind}/${traces.subset}?${qs}`);
  } catch {
    $("#trace-status").textContent = "";
    showTraceEmpty("no traces", `no ${traces.kind}/${traces.subset} rollouts at step ${traces.step}`);
    return;
  }
  traces.total = data.total;
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
  const rewardClass = (v) => (v == null ? "" : v > 0 ? "r-pos" : v < 0 ? "r-neg" : "r-zero");
  tbody.innerHTML = data.episodes
    .map(
      (ep) => `<tr data-line="${ep.line}">
        <td class="muted">${ep.line}</td>
        <td>${esc(ep.env ?? "?")}</td>
        <td class="muted" title="${esc(ep.group ?? "")}">${esc((ep.group ?? "").slice(0, 8))}</td>
        <td class="${rewardClass(ep.reward)}">${fmtNum(ep.reward)}</td>
        <td class="${rewardClass(ep.advantage)}">${fmtNum(ep.advantage)}</td>
        <td class="muted">${ep.input_tokens ?? ""}</td>
        <td>${ep.output_tokens ?? ""}</td>
        <td>${ep.turns ?? ""}</td>
        <td class="muted">${esc(ep.stop_condition ?? "")}</td>
        <td class="muted">${ep.dispatch_step != null && ep.dispatch_step !== state.traces.step ? ep.dispatch_step : ""}</td>
        <td class="${ep.ok && !ep.num_errors ? "status-ok" : "status-err"}">${ep.ok && !ep.num_errors ? "ok" : `${ep.num_errors || ""} err`}</td>
      </tr>`
    )
    .join("");
  $("#trace-status").textContent = `${data.total} episodes`;
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
  const query = $("#tm-search").value.trim().toLowerCase();
  const episodes = state.traces.episodes || [];
  if (!query) return episodes;
  return episodes.filter((e) =>
    `${e.line} ${e.id ?? ""} ${e.group ?? ""} ${e.env ?? ""} ${e.stop_condition ?? ""}`.toLowerCase().includes(query)
  );
}

function renderRolloutList() {
  const episodes = filteredRollouts();
  $("#tm-count").textContent = episodes.length;
  $("#tm-list").innerHTML = episodes
    .map((e) => {
      const cls = e.reward > 0 ? "r-pos" : e.reward < 0 ? "r-neg" : "r-zero";
      return (
        `<div class="tm-item ${e.line === currentLine ? "active" : ""}" data-line="${e.line}">` +
        `<span>Rollout ${e.line}</span><span class="muted">${esc(e.env ?? "")}</span>` +
        `<span class="tm-reward ${cls}">${fmtNum(e.reward)}</span></div>`
      );
    })
    .join("");
  $("#tm-list .tm-item.active")?.scrollIntoView({ block: "nearest" });
}

function stepRollout(delta) {
  const episodes = filteredRollouts();
  const idx = episodes.findIndex((e) => e.line === currentLine);
  const next = episodes[idx + delta];
  if (next) openEpisode(next.line);
}

function renderModalStep() {
  const traces = state.traces;
  const idx = traces.steps.findIndex((s) => s.step === traces.step);
  $("#tm-step-label").innerHTML =
    `step ${traces.step}` +
    `${traces.steps.length > 1 ? ` <span class="muted">(${idx + 1}/${traces.steps.length})</span>` : ""}` +
    ` <span class="muted">· ${esc(traces.kind)}/${esc(traces.subset)}</span>`;
  $("#tm-step-prev").disabled = idx <= 0;
  $("#tm-step-next").disabled = idx < 0 || idx >= traces.steps.length - 1;
}

async function modalStep(delta) {
  const traces = state.traces;
  const idx = traces.steps.findIndex((s) => s.step === traces.step);
  const target = traces.steps[idx + delta];
  if (!target) return;
  traces.step = target.step;
  traces.page = 0;
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

async function openEpisode(line) {
  const traces = state.traces;
  $("#trace-modal").hidden = false;
  $("#drawer-backdrop").hidden = false;
  currentLine = line;
  renderModalStep();
  renderRolloutList();
  $("#tm-messages").innerHTML = `<div class="chart-empty">loading episode…</div>`;
  $("#tm-meta").innerHTML = "";
  const episode = await api(
    `/api/runs/${encodeURIComponent(state.run)}/rollouts/${traces.step}/${traces.kind}/${traces.subset}/${line}?tokens=true`
  );
  if (line !== currentLine) return; // user already moved to another rollout
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
    const tip = `#${i} id=${id}${logprob != null ? ` lp=${logprob.toFixed(4)}` : ""}${advantage != null ? ` adv=${fmtNum(advantage)}` : ""} mask=${node.mask?.[i] ?? "?"}`;
    return `<span class="tok" style="${bg}" title="${esc(tip)}">${esc(text)}</span>`;
  });
  return spans.join("");
}

function subBlock(name, content) {
  const text = typeof content === "string" ? content : JSON.stringify(content, null, 2);
  return (
    `<details class="sub"><summary><span class="sub-name">${esc(name)}</span>` +
    `<span class="sub-preview">${esc(text.replace(/\s+/g, " ").slice(0, 140))}</span>` +
    `<span class="entry-chev">›</span></summary><div class="entry-body">${esc(text)}</div></details>`
  );
}

function renderMessages(trace, branches) {
  const container = $("#tm-messages");
  if (!trace) {
    container.innerHTML = emptyState("no traces", "this episode carries no trace data");
    return;
  }
  const signal = $("#token-signal").value;
  const path = branches[Math.min(currentBranchIdx, branches.length - 1)] || [];
  let maxAbsAdv = 0;
  for (const node of trace.nodes || [])
    for (const a of node.advantages || []) maxAbsAdv = Math.max(maxAbsAdv, Math.abs(a));
  const callsByNode = new Map((trace.calls || []).map((c) => [c.node, c]));
  const parts = [];
  path.forEach((idx, i) => {
    const node = trace.nodes[idx];
    const role = node.message?.role ?? "?";
    const call = callsByNode.get(idx);
    const chips = [];
    if (node.sampled) chips.push("sampled");
    if (call?.finish_reason) chips.push(call.finish_reason);
    if (call?.usage) chips.push(`${call.usage.prompt_tokens ?? "?"}→${call.usage.completion_tokens ?? "?"} tok`);
    else if (node.token_ids?.length) chips.push(`${node.token_ids.length} tok`);
    const text = messageText(node.message);
    const body = signal && node.token_ids?.length ? renderTokenNode(node, signal, maxAbsAdv) : esc(text);
    const subs = [];
    const reasoning = node.message?.reasoning_content ?? node.message?.reasoning;
    if (reasoning) subs.push(subBlock("Reasoning", reasoning));
    for (const toolCall of node.message?.tool_calls || [])
      subs.push(subBlock(`Tool call · ${toolCall.function?.name ?? "?"}`, toolCall.function?.arguments ?? toolCall));
    parts.push(
      `<details class="entry ${esc(role)}"${role === "system" ? "" : " open"}>` +
        `<summary><span class="entry-num">${String(i + 1).padStart(2, "0")}</span>` +
        `<span class="entry-role">${esc(role)}</span>` +
        `<span class="entry-preview">${esc(text.replace(/\s+/g, " ").slice(0, 180))}</span>` +
        chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("") +
        `<button class="icon-btn" data-copy="${idx}" title="copy message">${COPY_SVG}</button>` +
        `<span class="entry-chev">›</span></summary>` +
        subs.join("") +
        `<div class="entry-body">${body}</div></details>`
    );
  });
  container.innerHTML = parts.join("");
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

function renderMeta(ep, trace, branches) {
  const parts = [];
  const reward = trace ? traceReward(trace) : null;
  if (reward != null)
    parts.push(
      `<div class="meta-row"><span class="k">reward</span>` +
        `<span class="tm-reward-big${reward < 0 ? " neg" : ""}" style="margin-left:auto">${fmtNum(reward)}</span></div>`
    );

  if (trace) {
    const rewards = Object.entries(trace.rewards || {});
    if (rewards.length) {
      parts.push(`<div class="meta-sec">rewards</div>`);
      for (const [name, r] of rewards) parts.push(metaRow(name, `${fmtNum(r.score)} × ${fmtNum(r.weight ?? 1)}`));
    }
    const metrics = Object.entries(trace.metrics || {});
    if (metrics.length) {
      parts.push(`<div class="meta-sec">metrics</div>`);
      for (const [name, value] of metrics)
        parts.push(metaRow(name, typeof value === "number" ? fmtNum(value) : String(value)));
    }
  }

  parts.push(`<div class="meta-sec">identity</div>`);
  parts.push(metaRow("episode ID", ep.id, true));
  if (trace?.id) parts.push(metaRow("trace ID", trace.id, true));
  if (ep.group?.id) parts.push(metaRow("group ID", ep.group.id, true));
  if (trace?.agent?.runtime?.id) parts.push(metaRow("runtime ID", trace.agent.runtime.id, true));
  parts.push(metaRow("env", ep.env?.id ?? ep.env?.name));
  parts.push(metaRow("dispatch step", ep.run?.work?.step ?? ep.run?.metadata?.step));

  if (trace) {
    const path = branches[Math.min(currentBranchIdx, branches.length - 1)] || [];
    const nodes = path.map((i) => trace.nodes[i]);
    parts.push(`<div class="meta-sec">activity</div>`);
    parts.push(metaRow("turns", nodes.filter((n) => n.sampled).length));
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
      parts.push(metaRow("input tokens", usage.input.toLocaleString()));
      parts.push(metaRow("output tokens", usage.output.toLocaleString()));
      if (usage.reasoning) parts.push(metaRow("reasoning tokens", usage.reasoning.toLocaleString()));
      if (usage.cached) parts.push(metaRow("cached tokens", usage.cached.toLocaleString()));
      parts.push(metaRow("total tokens", (usage.input + usage.output).toLocaleString()));
    }

    if (trace.tools?.length)
      parts.push(
        `<details class="meta-fold"><summary>tool definitions (${trace.tools.length})</summary>` +
          `<pre class="json">${esc(JSON.stringify(trace.tools, null, 2))}</pre></details>`
      );

    parts.push(`<div class="meta-sec">state</div>`);
    parts.push(metaRow("stop_condition", trace.stop_condition));
    parts.push(metaRow("is_completed", trace.is_completed));
    parts.push(metaRow("ok", trace.ok));
    parts.push(metaRow("advantage", trace.info?.advantage != null ? fmtNum(trace.info.advantage) : null));

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
  traceTabs.innerHTML =
    traces.length > 1
      ? traces
          .map(
            (t, i) =>
              `<button data-trace="${i}" class="${i === currentTraceIdx ? "active" : ""}">${esc(t.agent?.name ?? "trace")} ${i}</button>`
          )
          .join("")
      : "";
  const branchTabs = $("#tm-branch-tabs");
  branchTabs.hidden = branches.length <= 1;
  branchTabs.innerHTML =
    branches.length > 1
      ? branches
          .map((_, i) => `<button data-branch="${i}" class="${i === currentBranchIdx ? "active" : ""}">branch ${i}</button>`)
          .join("")
      : "";
  renderRolloutList();
  renderMessages(trace, branches);
  renderMeta(ep, trace, branches);
}

/* ---------------------------------------------------------------- wiring */

$("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
$("#live-toggle").addEventListener("change", (e) => (state.live = e.target.checked));
$("#compare-btn").addEventListener("click", () => {
  const menu = $("#compare-menu");
  menu.hidden = !menu.hidden;
  if (!menu.hidden) renderCompareMenu();
});
$("#compare-menu").addEventListener("change", (e) => {
  const box = e.target.closest("[data-compare]");
  if (box) toggleCompare(box.dataset.compare, box.checked);
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#compare-wrap")) $("#compare-menu").hidden = true;
  if (!e.target.closest("#trace-filter-wrap")) $("#trace-filter-menu").hidden = true;
});
$("#trace-filter-btn").addEventListener("click", () => {
  const menu = $("#trace-filter-menu");
  menu.hidden = !menu.hidden;
});

function updateTraceFilterBtn() {
  const t = state.traces;
  $("#trace-filter-btn").classList.toggle("active", !!(t.env || t.errorsOnly || t.sort !== "line"));
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => activateTab(b.dataset.tab)));

document.querySelectorAll("#metrics-mode button").forEach((b) =>
  b.addEventListener("click", () => {
    state.metrics.mode = b.dataset.mode;
    document.querySelectorAll("#metrics-mode button").forEach((x) => x.classList.toggle("active", x === b));
    renderMetricsBody();
  })
);
$("#config-file").addEventListener("change", (e) => {
  state.config.file = e.target.value;
  loadConfig();
});
let configSearchDebounce = 0;
$("#config-search").addEventListener("input", () => {
  clearTimeout(configSearchDebounce);
  configSearchDebounce = setTimeout(applyConfigSearch, 200);
});
let searchDebounce = 0;
$("#metrics-search").addEventListener("input", (e) => {
  state.metrics.search = e.target.value;
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(renderMetricsBody, 250);
});

// remember collapsed sections across re-renders; charts created while hidden
// have zero width, so resize on expand ("toggle" doesn't bubble → capture)
$("#metrics-body").addEventListener(
  "toggle",
  (e) => {
    const section = e.target;
    if (!section.matches?.("details.section")) return;
    if (section.open) {
      state.metrics.collapsedSections.delete(section.dataset.name);
      resizeCharts();
    } else {
      state.metrics.collapsedSections.add(section.dataset.name);
    }
  },
  true
);

// drag a pane header to reorder within its section (order persisted by title)
$("#metrics-body").addEventListener("dragover", (e) => {
  if (!dragCard) return;
  const grid = e.target.closest(".chart-grid");
  if (!grid || grid !== dragCard.parentElement) return;
  e.preventDefault();
  const target = e.target.closest(".chart-card");
  if (!target || target === dragCard) return;
  const rect = target.getBoundingClientRect();
  const after = e.clientX - rect.left > rect.width / 2;
  grid.insertBefore(dragCard, after ? target.nextSibling : target);
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
  renderAllLogPanes();
});
$("#log-level").addEventListener("change", renderAllLogPanes);
let logSearchDebounce = 0;
$("#log-search").addEventListener("input", () => {
  clearTimeout(logSearchDebounce);
  logSearchDebounce = setTimeout(renderAllLogPanes, 200);
});
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
document.querySelectorAll("#trace-kind button").forEach((b) =>
  b.addEventListener("click", () => {
    if (b.disabled) return;
    state.traces.kind = b.dataset.kind;
    state.traces.page = 0;
    adjustKindSubset();
    loadEpisodes();
  })
);
document.querySelectorAll("#trace-subset button").forEach((b) =>
  b.addEventListener("click", () => {
    state.traces.subset = b.dataset.subset;
    state.traces.page = 0;
    adjustKindSubset();
    loadEpisodes();
  })
);
$("#trace-env").addEventListener("change", (e) => { state.traces.env = e.target.value; state.traces.page = 0; loadEpisodes(); });
$("#trace-errors").addEventListener("change", (e) => { state.traces.errorsOnly = e.target.checked; state.traces.page = 0; loadEpisodes(); });
$("#trace-sort").addEventListener("change", (e) => {
  [state.traces.sort, state.traces.order] = e.target.value.split(":");
  state.traces.page = 0;
  loadEpisodes();
});
$("#episode-table").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-line]");
  if (row) openEpisode(+row.dataset.line);
});
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
$("#token-signal").addEventListener("change", renderEpisode);
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
$("#tm-search").addEventListener("input", renderRolloutList);
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
  for (const entry of state.metrics.charts)
    if (entry.u) entry.u.setSize({ width: entry.card.clientWidth - 22, height: chartHeight() });
}
window.addEventListener("resize", resizeCharts);

function savePrefs() {
  localStorage.setItem(
    "prl-dash",
    JSON.stringify({
      smooth: state.metrics.smooth,
      paneMin: state.metrics.paneMin,
      paneH: state.metrics.paneH,
      paneOrder: state.metrics.paneOrder,
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
setInterval(async () => {
  if (!state.live) return;
  tickCount++;
  try {
    // keep the run list fresh so new runs register without a page refresh
    if (tickCount % 3 === 0 || !state.run) await loadRuns();
    if (!state.run) {
      const first = state.runs[0]?.name;
      if (first) await selectRun(first);
      return;
    }
    renderOverview(); // keeps the duration field ticking
    if (state.tab === "metrics" && state.metrics.loaded) await fetchMetrics();
    else if (state.tab === "logs" && state.logs.loaded) await pollLogs();
    else if (state.tab === "traces" && state.traces.loaded) {
      await loadRollouts();
      if (tickCount % 5 === 0) await loadEpisodes();
    }
  } catch (err) {
    console.warn("poll failed", err);
  }
}, POLL_MS);

(async function init() {
  $("#smooth-range").value = state.metrics.smooth;
  $("#smooth-val").textContent = state.metrics.smooth > 1 ? String(state.metrics.smooth) : "off";
  applyPaneSize();
  const params = new URLSearchParams(location.hash.slice(1));
  state.tab = params.get("tab") || "metrics";
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === state.tab));
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${state.tab}`));
  await loadRuns();
  const wanted = params.get("run");
  const run = state.runs.find((r) => r.name === wanted)?.name ?? state.runs[0]?.name;
  if (run) await selectRun(run);
  else $("#metrics-body").innerHTML = emptyState("no runs found", `nothing to show in ${state.outputDir ?? "the output directory"}`);
})();
