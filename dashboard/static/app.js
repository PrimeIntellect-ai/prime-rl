/* prime-rl dashboard frontend: metrics (wandb-overview replica), merged logs, trace viewer. */

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const api = async (path) => {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
};

const PALETTE = ["#a78bfa", "#22d3ee", "#f472b6", "#4ade80", "#fbbf24", "#60a5fa", "#f87171", "#c084fc"];
const POLL_MS = 3000;

const state = {
  runs: [],
  run: null,
  meta: null,
  tab: "metrics",
  live: true,
  metrics: { loaded: false, offset: 0, byKey: new Map(), mode: "overview", search: "", pinned: [], charts: [], renderedKeys: -1 },
  logs: { loaded: false, attempt: "latest", attempts: [], files: [], selected: new Set(), buffers: new Map(), gseq: 0 },
  traces: { loaded: false, steps: [], step: null, kind: "train", subset: "effective", page: 0, limit: 50, total: 0, env: "", errorsOnly: false, sort: "line", order: "asc" },
};

function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return "–";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e6 || abs < 1e-3) return v.toExponential(2);
  if (abs >= 100) return v.toFixed(1);
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(4);
}
const fmtBytes = (n) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)}M` : n >= 1024 ? `${(n / 1024).toFixed(0)}K` : `${n}B`);
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/* ------------------------------------------------------------------- runs */

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  const sel = $("#run-select");
  const current = state.run;
  sel.innerHTML = state.runs.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("");
  if (current && state.runs.some((r) => r.name === current)) sel.value = current;
}

async function selectRun(name) {
  if (!name) return;
  state.run = name;
  $("#run-select").value = name;
  state.meta = await api(`/api/runs/${encodeURIComponent(name)}`);
  state.metrics = { ...state.metrics, loaded: false, offset: 0, byKey: new Map(), pinned: [], charts: [], renderedKeys: -1 };
  state.logs = { ...state.logs, loaded: false, attempt: "latest", files: [], selected: new Set(), buffers: new Map() };
  state.traces = { ...state.traces, loaded: false, steps: [], step: null, page: 0, env: "", kind: "train", subset: "effective" };
  const badge = $("#run-type");
  badge.textContent = state.meta.type;
  badge.className = `badge type-badge ${state.meta.type}`;
  updateProgress();
  updateHash();
  await activateTab(state.tab, true);
}

function updateProgress() {
  const meta = state.meta;
  if (!meta) return;
  let step = null;
  for (const producers of state.metrics.byKey.values())
    for (const series of producers.values()) for (const s of series.keys()) step = Math.max(step ?? 0, s);
  step = step ?? state.runs.find((r) => r.name === state.run)?.last_step;
  $("#run-progress").textContent =
    (meta.model ? `${meta.model}  ·  ` : "") + (step != null ? `step ${step}${meta.max_steps ? `/${meta.max_steps}` : ""}` : "");
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

const TRAINER_KEY_RE = /^(perf|optim|loss|entropy|system|mismatch_kl|kl_ent_ratio|is_masked|masked_|unmasked_|max_vio|routing_|ref_kl|val)[/_]?/;
const ORCH_KEY_RE = /^(train|batch|off_policy|curriculum|eval)\//;

function rowProducer(row) {
  if (state.meta?.type === "sft") return "trainer";
  for (const key of Object.keys(row)) {
    if (TRAINER_KEY_RE.test(key)) return "trainer";
    if (ORCH_KEY_RE.test(key)) return "orch";
  }
  return Object.keys(row).some((k) => k.startsWith("progress/")) ? "orch" : "trainer";
}

function ingestRows(rows) {
  const byKey = state.metrics.byKey;
  for (const row of rows) {
    if (typeof row.step !== "number") continue;
    const producer = rowProducer(row);
    for (const [key, value] of Object.entries(row)) {
      if (key === "step" || key === "time" || typeof value !== "number") continue;
      let producers = byKey.get(key);
      if (!producers) byKey.set(key, (producers = new Map()));
      let series = producers.get(producer);
      if (!series) producers.set(producer, (series = new Map()));
      series.set(row.step, value);
    }
  }
}

async function fetchMetrics() {
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/metrics?offset=${state.metrics.offset}`);
  state.metrics.offset = data.offset;
  if (data.rows.length) {
    ingestRows(data.rows);
    updateProgress();
    if (state.metrics.byKey.size !== state.metrics.renderedKeys) renderMetricsBody();
    else updateCharts();
  }
  return data.rows.length;
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
  sections.push({ name: "performance", panels: PERFORMANCE_METRICS.map((m) => ({ metric: m })) });
  return sections;
}

function resolvePanel(panel) {
  const byKey = state.metrics.byKey;
  let keys;
  if (panel.metric) keys = byKey.has(panel.metric) ? [panel.metric] : [];
  else {
    const re = new RegExp(`^(?:${panel.regex})$`);
    keys = [...byKey.keys()].filter((k) => re.test(k)).sort();
  }
  const series = [];
  for (const key of keys)
    for (const [producer, points] of byKey.get(key)) series.push({ key, producer, points });
  return series;
}

function seriesLabels(series) {
  if (series.length === 1) return [""];
  const keyParts = series.map((s) => s.key.split("/"));
  let start = 0;
  while (keyParts.every((p) => p.length > start + 1 && p[start] === keyParts[0][start])) start++;
  return series.map((s, i) => {
    const tail = keyParts[i].slice(start).join("/");
    const dupKey = series.some((o, j) => j !== i && o.key === s.key);
    return dupKey ? `${tail} (${s.producer})` : tail;
  });
}

function panelData(series) {
  const stepSet = new Set();
  for (const s of series) for (const step of s.points.keys()) stepSet.add(step);
  const steps = [...stepSet].sort((a, b) => a - b);
  return [steps, ...series.map((s) => steps.map((st) => s.points.get(st) ?? null))];
}

function makeChart(el, labels, width) {
  const axis = {
    stroke: "#6b6b7d",
    grid: { stroke: "#1e1e2a", width: 1 },
    ticks: { stroke: "#1e1e2a" },
    font: "10px ui-monospace, monospace",
  };
  return new uPlot(
    {
      width,
      height: 130,
      cursor: { points: { size: 6 }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [{ ...axis, size: 26 }, { ...axis, size: 56, values: (u, vals) => vals.map(fmtNum) }],
      legend: { show: labels.length > 1 },
      series: [
        { label: "step" },
        ...labels.map((label, i) => ({
          label: label || "value",
          stroke: PALETTE[i % PALETTE.length],
          width: 1.5,
          spanGaps: true,
          points: { show: false },
        })),
      ],
    },
    [[], ...labels.map(() => [])],
    el
  );
}

function renderPanelCard(grid, panel) {
  const series = resolvePanel(panel);
  if (!series.length && panel.regex) return; // data-dependent panel with no matches yet
  const card = document.createElement("div");
  card.className = "chart-card";
  const title = panel.metric || panel.regex;
  card.innerHTML = `<div class="chart-head"><div class="chart-title" title="${esc(title)}">${esc(title)}</div><div class="chart-last"></div></div>`;
  grid.appendChild(card);
  if (!series.length) {
    card.insertAdjacentHTML("beforeend", `<div class="chart-empty">no data yet</div>`);
    state.metrics.charts.push({ card, panel, u: null, series: [] });
    return;
  }
  const plotEl = document.createElement("div");
  card.appendChild(plotEl);
  const u = makeChart(plotEl, seriesLabels(series), card.clientWidth - 22);
  const entry = { card, panel, u, series };
  state.metrics.charts.push(entry);
  updateChart(entry);
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

function renderMetricsBody() {
  const m = state.metrics;
  for (const entry of m.charts) entry.u?.destroy();
  m.charts = [];
  m.renderedKeys = m.byKey.size;
  const body = $("#metrics-body");
  body.innerHTML = "";
  $("#metrics-search").hidden = m.mode !== "all";
  if (!state.meta?.has_metrics && !m.byKey.size) {
    body.innerHTML = `<div class="chart-empty">no metrics.jsonl in this run</div>`;
    $("#metrics-status").textContent = "";
    return;
  }
  $("#metrics-status").textContent = `${m.byKey.size} keys`;
  if (m.mode === "overview") {
    for (const section of buildSections(state.meta)) {
      const div = document.createElement("div");
      div.className = "section";
      div.innerHTML = `<h2>${esc(section.name)}</h2>`;
      const grid = document.createElement("div");
      grid.className = "chart-grid";
      div.appendChild(grid);
      body.appendChild(div);
      for (const panel of section.panels) renderPanelCard(grid, panel);
      if (!grid.children.length) div.remove();
    }
  } else {
    body.innerHTML = `<div id="all-metrics"><div id="key-list"></div><div id="pinned-charts"><div class="chart-grid"></div></div></div>`;
    renderKeyList();
    const grid = body.querySelector("#pinned-charts .chart-grid");
    for (const key of m.pinned) renderPanelCard(grid, { metric: key });
  }
}

function renderKeyList() {
  const m = state.metrics;
  const list = $("#key-list");
  if (!list) return;
  const query = m.search.toLowerCase();
  const groups = new Map();
  for (const key of [...m.byKey.keys()].sort()) {
    if (query && !key.toLowerCase().includes(query)) continue;
    const group = key.split("/")[0];
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(key);
  }
  list.innerHTML = [...groups]
    .map(
      ([group, keys]) =>
        `<div class="key-group-name">${esc(group)} <span class="muted">(${keys.length})</span></div>` +
        keys
          .map((k) => `<div class="key-item ${m.pinned.includes(k) ? "pinned" : ""}" data-key="${esc(k)}">${esc(k)}</div>`)
          .join("")
    )
    .join("");
}

async function initMetrics() {
  state.metrics.loaded = true;
  await fetchMetrics();
  renderMetricsBody();
}

/* ------------------------------------------------------------------- logs */

const ANSI_RE = /\x1b\[[0-9;]*m/g;
const TEE_RE = /^\[[A-Za-z]+\d*\]:\s?/;
const LEVEL_RANK = { DEBUG: 0, INFO: 1, SUCCESS: 1, WARNING: 2, ERROR: 3, CRITICAL: 3 };
const TAIL_BYTES = 262144;

function componentClass(component) {
  if (component.startsWith("env:")) return "c-env";
  return { trainer: "c-trainer", orch: "c-orch", infer: "c-infer", evals: "c-evals" }[component] || "c-other";
}

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
    const timeMatch = plain.match(/^(\d\d):(\d\d):(\d\d)\b/);
    const levelMatch = plain.match(/^\d\d:\d\d:\d\d\s+(DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\b/);
    entries.push({
      rawTime: timeMatch ? +timeMatch[1] * 3600 + +timeMatch[2] * 60 + +timeMatch[3] : null,
      ownLevel: levelMatch ? levelMatch[1] : null,
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

async function pollLogs(render = true) {
  const logs = state.logs;
  let changed = false;
  await Promise.all(
    [...logs.selected].map(async (id) => {
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
  if (changed && render) renderLogStream();
}

function renderLogStream() {
  const logs = state.logs;
  const minRank = LEVEL_RANK[$("#log-level").value] ?? 0;
  const query = $("#log-search").value.toLowerCase();
  let lines = [];
  for (const id of logs.selected) {
    const buffer = logs.buffers.get(id);
    if (!buffer) continue;
    for (const line of buffer.lines) {
      if (minRank && (LEVEL_RANK[line.level] ?? minRank) < minRank) continue;
      if (query && !line.plain.toLowerCase().includes(query)) continue;
      lines.push(line);
    }
  }
  lines.sort((a, b) => a.t - b.t || a.gseq - b.gseq);
  const shown = lines.slice(-4000);
  const stream = $("#log-stream");
  const follow = $("#log-follow").checked;
  const pinned = follow || stream.scrollTop + stream.clientHeight >= stream.scrollHeight - 40;
  stream.innerHTML = shown
    .map(
      (line) =>
        `<div class="ll"><span class="lbadge ${componentClass(line.component)}">${esc(line.label)}</span>` +
        `<span class="ltext">${(line.html ??= ansiToHtml(line.raw))}</span></div>`
    )
    .join("");
  $("#log-status").textContent = `${shown.length}${lines.length > shown.length ? ` of ${lines.length}` : ""} lines`;
  if (pinned) stream.scrollTop = stream.scrollHeight;
}

async function loadOlder() {
  const logs = state.logs;
  await Promise.all(
    [...logs.buffers.values()].map(async (buffer) => {
      if (buffer.headStart <= 0) return;
      const start = Math.max(0, buffer.headStart - TAIL_BYTES);
      const chunk = await fetchLogChunk(buffer.file, { start, end: buffer.headStart });
      buffer.lines.unshift(...parseLines(chunk.text, buffer.file));
      buffer.headStart = start;
      retime(buffer);
    })
  );
  renderLogStream();
}

function renderLogSidebar() {
  const logs = state.logs;
  $("#attempt-select").innerHTML = logs.attempts
    .map((a) => `<option value="${a}" ${String(a) === String(logs.attempt) ? "selected" : ""}>attempt ${a}</option>`)
    .join("");
  const groups = new Map();
  for (const file of logs.files) {
    const group = file.component.startsWith("env:") ? "envs" : file.component;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(file);
  }
  $("#log-files").innerHTML = [...groups]
    .map(
      ([group, files]) =>
        `<div class="file-group-name">${esc(group)}</div>` +
        files
          .map(
            (f) =>
              `<label class="file-item"><input type="checkbox" data-file="${esc(f.id)}" ${logs.selected.has(f.id) ? "checked" : ""}>` +
              `<span class="${componentClass(f.component)}">${esc(f.label)}</span><span class="fsize">${fmtBytes(f.size)}</span></label>`
          )
          .join("")
    )
    .join("");
}

async function loadLogfiles() {
  const logs = state.logs;
  const data = await api(`/api/runs/${encodeURIComponent(state.run)}/logfiles?attempt=${logs.attempt}`);
  logs.attempts = data.attempts;
  logs.attempt = data.attempt;
  logs.files = data.files;
  if (!logs.selected.size) for (const f of data.files) if (f.master) logs.selected.add(f.id);
  renderLogSidebar();
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
  renderStepStrip();
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

function renderStepStrip() {
  const traces = state.traces;
  $("#step-strip").innerHTML = traces.steps
    .map((s) => {
      const train = s.counts["train/effective"] ?? s.counts["train/all"];
      return (
        `<div class="step-chip ${s.step === traces.step ? "active" : ""}" data-step="${s.step}">` +
        `${s.step}${train != null ? ` <span class="muted">${train}</span>` : ""}` +
        `${s.counts["eval/all"] || s.counts["eval/effective"] ? '<span class="eval-dot" title="eval rollouts"></span>' : ""}</div>`
      );
    })
    .join("");
}

async function loadEpisodes() {
  const traces = state.traces;
  const tbody = $("#episode-table tbody");
  if (traces.step == null) {
    tbody.innerHTML = "";
    $("#trace-status").textContent = "no rollouts";
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
    tbody.innerHTML = "";
    $("#trace-status").textContent = `no ${traces.kind}/${traces.subset} traces at step ${traces.step}`;
    return;
  }
  traces.total = data.total;
  const envSel = $("#trace-env");
  const currentEnv = traces.env;
  envSel.innerHTML =
    `<option value="">all envs</option>` +
    data.envs.map((e) => `<option value="${esc(e)}" ${e === currentEnv ? "selected" : ""}>${esc(e)}</option>`).join("");
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
  const pages = Math.max(1, Math.ceil(data.total / traces.limit));
  $("#episode-pager").innerHTML =
    pages > 1
      ? `<button class="btn" id="pg-prev" ${traces.page === 0 ? "disabled" : ""}>‹ prev</button>
         <span class="muted">page ${traces.page + 1} / ${pages}</span>
         <button class="btn" id="pg-next" ${traces.page >= pages - 1 ? "disabled" : ""}>next ›</button>`
      : "";
  $("#pg-prev")?.addEventListener("click", () => { traces.page--; loadEpisodes(); });
  $("#pg-next")?.addEventListener("click", () => { traces.page++; loadEpisodes(); });
}

async function initTraces() {
  state.traces.loaded = true;
  await loadRollouts();
  adjustKindSubset();
  await loadEpisodes();
}

/* ----------------------------------------------------------- episode view */

let currentEpisode = null;
let currentTraceIdx = 0;
let currentBranchIdx = 0;

async function openEpisode(line) {
  const traces = state.traces;
  $("#drawer").hidden = false;
  $("#drawer-backdrop").hidden = false;
  $("#drawer-body").innerHTML = `<div class="chart-empty">loading episode…</div>`;
  $("#drawer-title").textContent = `step ${traces.step} · ${traces.kind}/${traces.subset} · #${line}`;
  currentEpisode = await api(
    `/api/runs/${encodeURIComponent(state.run)}/rollouts/${traces.step}/${traces.kind}/${traces.subset}/${line}?tokens=true`
  );
  currentTraceIdx = 0;
  currentBranchIdx = 0;
  renderEpisode();
}

function closeDrawer() {
  $("#drawer").hidden = true;
  $("#drawer-backdrop").hidden = true;
  currentEpisode = null;
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
      const alpha = Math.min(1, Math.abs(advantage) / maxAbsAdv) * 0.55;
      bg = `background:rgba(${advantage > 0 ? "74,222,128" : "248,113,113"},${alpha.toFixed(3)})`;
    } else if (signal === "logprob" && logprob != null) {
      bg = `background:rgba(139,92,246,${(Math.min(1, -logprob / 6) * 0.7).toFixed(3)})`;
    } else if (signal === "mask" && node.mask?.[i]) {
      bg = "background:rgba(139,92,246,0.35)";
    } else if (signal === "is_content" && node.is_content?.[i]) {
      bg = "background:rgba(34,211,238,0.3)";
    }
    const tip = `#${i} id=${id}${logprob != null ? ` lp=${logprob.toFixed(4)}` : ""}${advantage != null ? ` adv=${fmtNum(advantage)}` : ""} mask=${node.mask?.[i] ?? "?"}`;
    return `<span class="tok" style="${bg}" title="${esc(tip)}">${esc(text)}</span>`;
  });
  return spans.join("");
}

function renderEpisode() {
  const ep = currentEpisode;
  if (!ep) return;
  const traces = ep.traces || [];
  const trace = traces[currentTraceIdx];
  const signal = $("#token-signal").value;
  const parts = [];

  const badges = [
    ["env", ep.env?.id ?? ep.env?.name],
    ["group", (ep.group?.id ?? "").slice(0, 12)],
    ["episode", (ep.id ?? "").slice(0, 12)],
    ["dispatch step", ep.run?.work?.step ?? ep.run?.metadata?.step],
    ["ok", ep.ok],
  ];
  parts.push(
    `<div class="ep-meta">` +
      badges.filter(([, v]) => v != null).map(([k, v]) => `<span class="badge">${esc(k)}: ${esc(v)}</span>`).join("") +
      `</div>`
  );

  if (traces.length > 1)
    parts.push(
      `<div class="trace-tabs seg">` +
        traces
          .map((t, i) => `<button data-trace="${i}" class="${i === currentTraceIdx ? "active" : ""}">${esc(t.agent?.name ?? "trace")} ${i}</button>`)
          .join("") +
        `</div>`
    );

  if (!trace) {
    parts.push(`<div class="chart-empty">episode has no traces</div>`);
    if (ep.errors?.length) parts.push(`<h3 class="sec">errors</h3><pre class="json">${esc(JSON.stringify(ep.errors, null, 2))}</pre>`);
    $("#drawer-body").innerHTML = parts.join("");
    return;
  }

  const branches = traceBranches(trace);
  if (branches.length > 1) {
    if (currentBranchIdx >= branches.length) currentBranchIdx = 0;
    parts.push(
      `<div class="trace-tabs seg">` +
        branches.map((_, i) => `<button data-branch="${i}" class="${i === currentBranchIdx ? "active" : ""}">branch ${i}</button>`).join("") +
        `</div>`
    );
  }
  const path = branches[Math.min(currentBranchIdx, branches.length - 1)] || [];

  let maxAbsAdv = 0;
  for (const node of trace.nodes || [])
    for (const a of node.advantages || []) maxAbsAdv = Math.max(maxAbsAdv, Math.abs(a));

  const callsByNode = new Map((trace.calls || []).map((c) => [c.node, c]));
  for (const idx of path) {
    const node = trace.nodes[idx];
    const role = node.message?.role ?? "?";
    const call = callsByNode.get(idx);
    const chips = [];
    if (node.sampled) chips.push("sampled");
    if (call) {
      if (call.finish_reason) chips.push(call.finish_reason);
      if (call.usage) chips.push(`${call.usage.prompt_tokens ?? "?"}→${call.usage.completion_tokens ?? "?"} tok`);
    } else if (node.token_ids?.length) chips.push(`${node.token_ids.length} tok`);
    let body;
    if (signal && node.token_ids?.length) body = renderTokenNode(node, signal, maxAbsAdv);
    else {
      body = esc(messageText(node.message));
      for (const toolCall of node.message?.tool_calls || [])
        body += `\n<span class="badge">tool: ${esc(toolCall.function?.name ?? "?")}</span> ${esc(toolCall.function?.arguments ?? "")}`;
    }
    parts.push(
      `<div class="msg ${esc(role)}"><div class="msg-head">${esc(role)}${chips.map((c) => ` <span class="chip">${esc(c)}</span>`).join("")}</div>` +
        `<div class="msg-body">${body}</div></div>`
    );
  }

  const rewards = Object.entries(trace.rewards || {});
  if (rewards.length) {
    parts.push(`<h3 class="sec">rewards</h3><table class="kv-table"><tr><th>name</th><th>score</th><th>weight</th></tr>`);
    for (const [name, r] of rewards)
      parts.push(`<tr><td>${esc(name)}</td><td>${fmtNum(r.score)}</td><td>${fmtNum(r.weight ?? 1)}</td></tr>`);
    parts.push(`<tr><th>total</th><th colspan="2">${fmtNum(traceReward(trace))}</th></tr></table>`);
  }

  const info = { ...(trace.info || {}) };
  const meta = {
    advantage: info.advantage,
    stop_condition: trace.stop_condition,
    is_completed: trace.is_completed,
    ok: trace.ok,
    model: trace.agent?.config?.model,
    renderer_model: trace.agent?.config?.client?.renderer_model_name,
    temperature: trace.agent?.config?.sampling?.temperature,
    max_tokens: trace.agent?.config?.sampling?.max_tokens,
  };
  parts.push(`<h3 class="sec">trace</h3><table class="kv-table">`);
  for (const [k, v] of Object.entries(meta)) if (v != null) parts.push(`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`);
  parts.push(`</table>`);

  const durations = [];
  (function walkTiming(obj, prefix) {
    if (!obj || typeof obj !== "object") return;
    if (typeof obj.duration === "number") durations.push([prefix, obj.duration]);
    else if (typeof obj.start === "number" && typeof obj.end === "number") durations.push([prefix, obj.end - obj.start]);
    for (const [k, v] of Object.entries(obj)) if (typeof v === "object") walkTiming(v, prefix ? `${prefix}/${k}` : k);
  })(trace.timing, "");
  if (durations.length) {
    parts.push(`<h3 class="sec">timing</h3><table class="kv-table">`);
    for (const [name, secs] of durations) parts.push(`<tr><th>${esc(name || "total")}</th><td>${secs.toFixed(2)}s</td></tr>`);
    parts.push(`</table>`);
  }

  const errors = [...(ep.errors || []), ...(trace.errors || [])];
  if (errors.length) parts.push(`<h3 class="sec">errors</h3><pre class="json">${esc(JSON.stringify(errors, null, 2))}</pre>`);

  $("#drawer-body").innerHTML = parts.join("");
}

/* ---------------------------------------------------------------- wiring */

$("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
$("#live-toggle").addEventListener("change", (e) => (state.live = e.target.checked));

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => activateTab(b.dataset.tab)));

document.querySelectorAll("#metrics-mode button").forEach((b) =>
  b.addEventListener("click", () => {
    state.metrics.mode = b.dataset.mode;
    document.querySelectorAll("#metrics-mode button").forEach((x) => x.classList.toggle("active", x === b));
    renderMetricsBody();
  })
);
$("#metrics-search").addEventListener("input", (e) => {
  state.metrics.search = e.target.value;
  renderKeyList();
});
$("#metrics-body").addEventListener("click", (e) => {
  const item = e.target.closest(".key-item");
  if (!item) return;
  const key = item.dataset.key;
  const pinned = state.metrics.pinned;
  const idx = pinned.indexOf(key);
  if (idx >= 0) pinned.splice(idx, 1);
  else pinned.push(key);
  renderMetricsBody();
});

$("#attempt-select").addEventListener("change", async (e) => {
  state.logs.attempt = e.target.value;
  state.logs.buffers = new Map();
  state.logs.selected = new Set();
  await loadLogfiles();
  await pollLogs();
});
$("#log-files").addEventListener("change", async (e) => {
  const id = e.target.dataset.file;
  if (!id) return;
  if (e.target.checked) state.logs.selected.add(id);
  else { state.logs.selected.delete(id); state.logs.buffers.delete(id); }
  await pollLogs(false);
  renderLogStream();
});
$("#log-level").addEventListener("change", renderLogStream);
$("#log-search").addEventListener("input", renderLogStream);
$("#log-older").addEventListener("click", loadOlder);

$("#step-strip").addEventListener("click", (e) => {
  const chip = e.target.closest(".step-chip");
  if (!chip) return;
  state.traces.step = +chip.dataset.step;
  state.traces.page = 0;
  adjustKindSubset();
  renderStepStrip();
  loadEpisodes();
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
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
$("#token-signal").addEventListener("change", renderEpisode);
$("#drawer-body").addEventListener("click", (e) => {
  const traceBtn = e.target.closest("[data-trace]");
  if (traceBtn) { currentTraceIdx = +traceBtn.dataset.trace; currentBranchIdx = 0; renderEpisode(); return; }
  const branchBtn = e.target.closest("[data-branch]");
  if (branchBtn) { currentBranchIdx = +branchBtn.dataset.branch; renderEpisode(); }
});

window.addEventListener("resize", () => {
  for (const entry of state.metrics.charts)
    if (entry.u) entry.u.setSize({ width: entry.card.clientWidth - 22, height: 130 });
});

let tickCount = 0;
setInterval(async () => {
  if (!state.live || !state.run) return;
  tickCount++;
  try {
    if (state.tab === "metrics" && state.metrics.loaded) await fetchMetrics();
    else if (state.tab === "logs" && state.logs.loaded) await pollLogs();
    else if (state.tab === "traces" && state.traces.loaded) {
      await loadRollouts();
      if (tickCount % 5 === 0) await loadEpisodes();
    }
    if (tickCount % 10 === 0) await loadRuns();
  } catch (err) {
    console.warn("poll failed", err);
  }
}, POLL_MS);

(async function init() {
  const params = new URLSearchParams(location.hash.slice(1));
  state.tab = params.get("tab") || "metrics";
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === state.tab));
  document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.id !== `tab-${state.tab}`));
  await loadRuns();
  const wanted = params.get("run");
  const run = state.runs.find((r) => r.name === wanted)?.name ?? state.runs[0]?.name;
  if (run) await selectRun(run);
  else $("#metrics-body").innerHTML = `<div class="chart-empty">no runs found in the output directory</div>`;
})();
