// rollout-viewer · Direction A — INSTRUMENT / OSCILLOSCOPE
// Vanilla ES module. Two scrubbable playheads (training-step + turn) are the spine.
// Schedule-aware + regime-generic: renders single_turn, single_agent_multiturn,
// and multi_agent against the run's inferred schedule.

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (k === "dataset") for (const [dk, dv] of Object.entries(v)) n.dataset[dk] = dv;
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
};

const api = async (p) => {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${r.status} ${p} :: ${(await r.text()).slice(0, 300)}`);
  return r.json();
};

// ── application state ─────────────────────────────────────────────
const S = {
  run: null,
  steps: [],          // training steps available in the run
  sort: "reward",
  order: "desc",
  rows: [],           // index rows for the rail
  schedule: null,     // {regime, source, slots:[...]}
  selected: null,     // {example_id, step, line, trajectory_id}
  episode: null,      // full Episode (+ alignment) of the selected rollout
  byStep: null,       // examples/{id} -> by_step (training-step scrubber data)
  turn: 0,            // selected turn index within the episode
  demo: false,        // synthetic-regime mode
};

// ── number / text helpers ─────────────────────────────────────────
const fmt = (x, d = 3) => (x == null ? "·" : typeof x === "number" ? x.toFixed(d) : String(x));
const rewardCls = (x) => (x == null ? "zero" : x > 0 ? "pos" : x < 0 ? "neg" : "zero");
const msgText = (c) => (typeof c === "string" ? c : c == null ? "" : JSON.stringify(c, null, 2));
const seatLabel = (a) => (a === "agent" ? "agent" : a);

// answer-flip detection from mar.categorical (first_answer/<seat> != final_answer/<seat>)
function flipsOf(ep) {
  const cat = ep?.mar?.categorical || {};
  const seats = new Set();
  for (const k of Object.keys(cat)) {
    const m = k.match(/^first_answer\/(.+)$/);
    if (m) seats.add(m[1]);
  }
  const out = {};
  for (const seat of seats) {
    const first = cat[`first_answer/${seat}`];
    const final = cat[`final_answer/${seat}`];
    out[seat] = first != null && final != null && first !== final;
  }
  return out; // {seat: bool}
}
const rowFlipped = (r) => {
  // index row: metrics carry flipped/<seat> scalars
  const m = r.metrics || {};
  return Object.keys(m).some((k) => k.startsWith("flipped/") && m[k] > 0);
};

// ── boot ──────────────────────────────────────────────────────────
async function loadRuns() {
  const runs = await api("/api/runs");
  const sel = $("#run");
  sel.replaceChildren(
    ...runs.map((r) => el("option", { value: r.run_id }, `${r.run_id}  (${r.steps.length} steps)`))
  );
  if ((!S.run || !runs.some((r) => r.run_id === S.run)) && runs.length) S.run = runs[0].run_id;
  if (S.run) {
    sel.value = S.run;
    S.steps = runs.find((r) => r.run_id === S.run)?.steps || [];
  }
  S.demo = false;
  await Promise.all([loadSchedule(), loadEpisodes()]);
}

async function loadSchedule() {
  if (!S.run) return;
  try {
    S.schedule = await api(`/api/runs/${S.run}/schedule`);
  } catch (e) {
    S.schedule = null;
  }
  $("#regime").textContent = S.schedule ? S.schedule.regime.replace(/_/g, " ") : "";
  $("#regime").title = S.schedule ? `schedule source: ${S.schedule.source} · ${S.schedule.slots.length} slots` : "";
}

async function loadEpisodes() {
  if (!S.run) return;
  const j = await api(`/api/runs/${S.run}/episodes?sort=${encodeURIComponent(S.sort)}&order=${S.order}&limit=5000`);
  S.rows = j.rows || [];
  renderRail();
}

// ── rail (episode list) ───────────────────────────────────────────
function renderRail() {
  const list = $("#list");
  $("#rail-count").textContent = `${S.rows.length} episodes`;
  if (!S.rows.length) {
    list.replaceChildren(el("div", { class: "placeholder" }, "no episodes"));
    return;
  }
  list.replaceChildren(
    ...S.rows.map((r, i) => {
      const isSel = S.selected && S.selected.step === r.step && S.selected.line === r.transcript_line;
      const flags = [];
      if (rowFlipped(r)) flags.push(el("span", { class: "chip flip" }, "flip"));
      if (r.is_filtered) flags.push(el("span", { class: "chip filt" }, "filtered"));
      if (r.transcript_line < 0) flags.push(el("span", { class: "chip elide" }, "ρ-elided"));
      return el(
        "div",
        { class: `ep ${isSel ? "sel" : ""}`, dataset: { i }, onclick: () => selectRow(i) },
        el("span", { class: "step" }, `s${r.step}`),
        el(
          "div",
          { class: "mid" },
          el("div", { class: "ex" }, r.example_id),
          el(
            "div",
            { class: "meta" },
            el("span", { class: "chip kind" }, (r.kind || "").replace(/_/g, " ")),
            r.winner ? el("span", { class: "winner" }, `▸ ${r.winner}`) : null,
            ...flags
          )
        ),
        el("span", { class: `reward ${rewardCls(r.reward)}` }, fmt(r.reward, 2))
      );
    })
  );
}

// ── selection → load the rollout + its example's step series ──────
async function selectRow(i) {
  const r = S.rows[i];
  if (S.demo) { selectSynthetic(r); return; }
  S.selected = { example_id: r.example_id, step: r.step, line: r.transcript_line, trajectory_id: r.trajectory_id };
  $("#scrubbers").hidden = false;
  // training-step scrubber data: this question's rollouts across steps
  S.byStep = await api(`/api/runs/${S.run}/examples/${encodeURIComponent(r.example_id)}`).catch(() => null);
  await loadEpisode(r.step, r.transcript_line);
  renderRail();
}

async function loadEpisode(step, line) {
  const detail = $("#detail");
  if (line < 0) {
    S.episode = null;
    detail.replaceChildren(
      el("div", { class: "elided" },
        el("b", {}, "transcript not retained (ρ < 1)"),
        "metrics + diagnostics are in the index, but this episode's full transcript was not shipped to the store.")
    );
    renderScrubbers();
    return;
  }
  detail.replaceChildren(el("div", { class: "placeholder" }, "loading transcript…"));
  let ep;
  try {
    ep = await api(`/api/episodes/${S.run}/${step}/${line}`);
  } catch (e) {
    detail.replaceChildren(el("div", { class: "err" }, String(e)));
    return;
  }
  if (ep.retained === false) {
    S.episode = null;
    detail.replaceChildren(el("div", { class: "elided" }, el("b", {}, "transcript not retained"), ep.detail || ""));
    renderScrubbers();
    return;
  }
  S.episode = ep;
  S.selected = { ...S.selected, step, line };
  // default to the last turn (the freshest output), clamped
  S.turn = Math.max(0, (ep.steps?.length || 1) - 1);
  renderScrubbers();
  renderDetail();
}

// ── scrubbers (the spine) ─────────────────────────────────────────
function renderScrubbers() {
  renderScopeMeta();
  renderStepAxis();
  renderTurnAxis();
}

function renderScopeMeta() {
  const ep = S.episode;
  const meta = $("#scope-meta");
  if (!ep) { meta.replaceChildren(); return; }
  const answer = ep.task?.answer;
  const winner = ep.mar?.categorical?.winner ?? ep.metrics?.winner;
  meta.replaceChildren(
    metaItem("example", ep.example_id, "v"),
    metaItem("regime", (ep.kind || "").replace(/_/g, " "), "v"),
    metaItem("traj", (ep.trajectory_id || "·").slice(0, 12), "q"),
    metaItem("reward", fmt(ep.reward, 3), "v"),
    answer != null ? metaItem("gold", String(answer), "v") : null,
    winner ? metaItem("winner", String(winner), "winner-v") : null,
  );
}
const metaItem = (k, v, cls) =>
  el("span", {}, el("span", { class: "k" }, `${k} `), el("span", { class: `v ${cls || ""}` }, v));

// training-step axis: a rail of every step in the run + sparkline traces
// (per-step mean reward of this example's rollouts + flip rate).
function renderStepAxis() {
  const rail = $("#step-rail");
  const series = stepSeries();          // [{step, hasData, meanReward, flipRate}]
  const cur = S.selected?.step;
  rail.replaceChildren(
    ...series.map((s) =>
      el(
        "div",
        {
          class: `step-tick ${s.hasData ? "has" : "empty"} ${s.step === cur ? "cur" : ""}`,
          title: s.hasData
            ? `step ${s.step} · ${s.n} rollouts · mean reward ${fmt(s.meanReward, 2)} · flip ${fmt(s.flipRate, 2)}`
            : `step ${s.step} · no rollouts for example ${S.selected?.example_id}`,
          onclick: () => onStepClick(s),
        },
        String(s.step)
      )
    )
  );
  drawSpark(series, cur);
}

// Build the per-step series for the selected example. Real smoke data has each
// example at one step; absent steps render as empty ticks (honest gaps), so the
// step axis is always the full run axis, not just the populated steps.
function stepSeries() {
  const allSteps = S.demo ? [S.selected?.step ?? 0] : S.steps;
  const map = new Map();
  for (const blk of S.byStep?.by_step || []) {
    const eps = blk.episodes || [];
    const rewards = eps.map((e) => e.reward).filter((x) => x != null);
    map.set(blk.step, {
      n: eps.length,
      meanReward: rewards.length ? rewards.reduce((a, b) => a + b, 0) / rewards.length : null,
      // flip rate isn't in the examples index (it's per-transcript); approximate
      // from the rail rows for the same example+step that carry flipped/* metrics.
      flipRate: flipRateFor(blk.step),
    });
  }
  return (allSteps.length ? allSteps : [...map.keys()]).sort((a, b) => a - b).map((step) => {
    const d = map.get(step);
    return { step, hasData: !!d, n: d?.n ?? 0, meanReward: d?.meanReward ?? null, flipRate: d?.flipRate ?? null };
  });
}

function flipRateFor(step) {
  const ex = S.selected?.example_id;
  const rows = S.rows.filter((r) => r.example_id === ex && r.step === step && r.transcript_line >= 0);
  if (!rows.length) return null;
  return rows.filter(rowFlipped).length / rows.length;
}

function onStepClick(s) {
  if (!s.hasData) return;
  // pick this example's first rollout at that step
  const blk = (S.byStep?.by_step || []).find((b) => b.step === s.step);
  const epRef = blk?.episodes?.[0];
  if (!epRef) return;
  S.selected = { ...S.selected, step: s.step, line: epRef.transcript_line, trajectory_id: epRef.trajectory_id };
  loadEpisode(s.step, epRef.transcript_line);
  renderRail();
}

// faint sparkline traces under the step rail: reward (accent) + flip (amber)
function drawSpark(series, cur) {
  const svg = $("#spark");
  const W = 1000, H = 100, pad = 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const n = series.length;
  const x = (i) => (n <= 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (n - 1));
  const yOf = (v, lo, hi) => H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
  const kids = [];
  // baseline
  kids.push(node("line", { x1: pad, x2: W - pad, y1: H / 2, y2: H / 2, stroke: "var(--line-soft)", "stroke-width": 1 }));
  const tracePath = (key, lo, hi, color, w) => {
    const pts = series.map((s, i) => (s[key] == null ? null : [x(i), yOf(s[key], lo, hi)]));
    let d = "", started = false;
    for (const p of pts) {
      if (p == null) { started = false; continue; }
      d += `${started ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)} `;
      started = true;
    }
    if (d) kids.push(node("path", { d, fill: "none", stroke: color, "stroke-width": w, "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke" }));
    // dots
    series.forEach((s, i) => {
      if (s[key] == null) return;
      kids.push(node("circle", { cx: x(i), cy: yOf(s[key], lo, hi), r: s.step === cur ? 3.2 : 2, fill: color }));
    });
  };
  // reward range derived from data, padded
  const rewards = series.map((s) => s.meanReward).filter((v) => v != null);
  const rLo = Math.min(-1, ...rewards), rHi = Math.max(1, ...rewards);
  tracePath("meanReward", rLo, rHi, "var(--scope)", 1.6);
  tracePath("flipRate", 0, 1, "var(--flip)", 1.2);
  // playhead at current step
  const ci = series.findIndex((s) => s.step === cur);
  if (ci >= 0) kids.push(node("line", { x1: x(ci), x2: x(ci), y1: pad, y2: H - pad, stroke: "var(--scope)", "stroke-width": 1, "stroke-dasharray": "2 3", opacity: 0.7 }));
  svg.replaceChildren(...kids);
  $("#spark-legend").replaceChildren(
    el("span", { class: "reward-l" }, el("i", {}), "mean reward"),
    el("span", { class: "flip-l" }, el("i", {}), "flip rate"),
  );
}
const SVGNS = "http://www.w3.org/2000/svg";
const node = (tag, attrs) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

// turn axis: schedule-aligned ticks. Each tick = an AlignedTurn position.
// matched/deviated/truncated/extra encoded in color; flip + (future) KL markers.
function renderTurnAxis() {
  const ep = S.episode;
  const rail = $("#turn-rail");
  if (!ep) { rail.replaceChildren(); $("#turn-legend").replaceChildren(); return; }
  const aligned = ep.alignment || synthAlignment(ep);
  const flips = flipsOf(ep);
  // bar height encodes completion size (token-ish): chars of the produced completion.
  const sizes = ep.steps.map((s) => s.completion.reduce((a, m) => a + msgText(m.content).length, 0));
  const maxSize = Math.max(1, ...sizes);
  rail.replaceChildren(
    ...aligned.map((a) => {
      // A step is present unless the slot was expected but unobserved (truncated).
      // Both server and synthetic alignment key positionally by index.
      const step = a.status === "truncated" ? null : ep.steps[a.index] || null;
      const stepIdx = step ? step.index : a.index;
      const hasStep = !!step;
      const sz = hasStep ? sizes[stepIdx] ?? 0 : 0;
      const h = hasStep ? 18 + (sz / maxSize) * 78 : 24; // truncated slots still show a short stub
      const actor = a.actor || "·";
      const flip = hasStep && flips[actor];
      const kids = [];
      if (flip) kids.push(el("span", { class: "flip-mark", title: `${actor} flipped its answer` }));
      const klSpike = hasStep && step.diagnostics && step.diagnostics.status === "present" &&
        step.diagnostics.mismatch_kl && step.diagnostics.mismatch_kl.mean > klThreshold();
      if (klSpike) kids.push(el("span", { class: "kl-mark", title: `KL spike: ${fmt(step.diagnostics.mismatch_kl.mean)}` }));
      kids.push(el("div", { class: "bar-fill", style: `height:${h}%` }));
      kids.push(el("div", { class: "seat-lbl" }, seatLabel(actor)));
      return el(
        "div",
        {
          class: `turn-tick ${hasStep && stepIdx === S.turn ? "cur" : ""}`,
          dataset: { st: a.status },
          title: turnTitle(a, actor),
          onclick: hasStep ? () => { S.turn = stepIdx; renderDetail(); renderTurnAxis(); } : null,
        },
        ...kids
      );
    })
  );
  $("#turn-legend").replaceChildren(
    legendItem("i-matched", "matched"),
    legendItem("i-deviated", "deviated"),
    legendItem("i-truncated", "truncated"),
    legendItem("i-extra", "extra"),
    legendItem("i-flip", "answer flip"),
  );
}
const legendItem = (cls, label) => el("span", {}, el("i", { class: cls }), label);
const turnTitle = (a, actor) =>
  `turn ${a.index} · ${actor}${a.phase ? " · " + a.phase : ""} · ${a.status}`;

// KL-spike threshold: a heuristic until the trainer emits diagnostics. Conservative.
const klThreshold = () => 0.05;

// ── selected-turn detail ──────────────────────────────────────────
function renderDetail() {
  const ep = S.episode;
  const detail = $("#detail");
  if (!ep) return;
  const step = ep.steps[S.turn];
  if (!step) { detail.replaceChildren(el("div", { class: "placeholder" }, "no turn selected")); return; }
  const aligned = (ep.alignment || synthAlignment(ep)).find((a) => a.index === step.index && a.status !== "truncated") || {};
  const status = aligned.status || "matched";

  // ── visibility delta: messages NEW vs this actor's previous turn ──
  const prev = prevSameActor(ep, S.turn, step.member_id);
  const prevKeys = new Set((prev?.prompt || []).map(msgKey));
  const newMsgs = step.prompt.filter((m) => !prevKeys.has(msgKey(m)));
  const carriedMsgs = step.prompt.filter((m) => prevKeys.has(msgKey(m)));

  detail.replaceChildren(
    // head
    el(
      "div",
      { class: "turn-head" },
      el("span", { class: "seat" }, seatLabel(step.member_id ?? "agent")),
      el("span", { class: "idx" }, `turn ${step.index}`),
      step.phase ? el("span", { class: "phase" }, step.phase) : null,
      el("span", { class: `status-pill ${status}` }, status),
      el(
        "div",
        { class: "nav" },
        el("button", { onclick: () => stepTurn(-1), disabled: S.turn <= 0 ? "" : null }, "‹ prev"),
        el("button", { onclick: () => stepTurn(1), disabled: S.turn >= ep.steps.length - 1 ? "" : null }, "next ›")
      )
    ),

    // NEW THIS TURN — the visibility delta
    el(
      "div",
      { class: "block" },
      el("div", { class: "block-h" }, "new this turn", el("span", { class: "n" }, `${newMsgs.length} msg${newMsgs.length === 1 ? "" : "s"}`)),
      prev
        ? el("div", { class: "delta-note" }, `vs ${seatLabel(step.member_id ?? "agent")}'s previous turn (#${prev.index})`)
        : el("div", { class: "delta-note" }, `first turn for ${seatLabel(step.member_id ?? "agent")} — entire prompt is new`),
      newMsgs.length
        ? newMsgs.map((m) => renderMsg(m, "prompt", true))
        : el("div", { class: "delta-empty" }, "no new context injected since this actor's last turn")
    ),

    // carried context (collapsed)
    carriedMsgs.length
      ? el(
          "details",
          { class: "carried" },
          el("summary", {}, `carried context · ${carriedMsgs.length} msg${carriedMsgs.length === 1 ? "" : "s"} (seen before)`),
          el("div", { class: "carried-body" }, carriedMsgs.map((m) => renderMsg(m, "prompt", false)))
        )
      : null,

    // completion (what this actor produced)
    el(
      "div",
      { class: "block" },
      el("div", { class: "block-h" }, "produced", el("span", { class: "n" }, `${step.completion.length} msg${step.completion.length === 1 ? "" : "s"}`)),
      step.completion.map((m) => renderMsg(m, "completion", false))
    ),

    // diagnostics line (honest absent/masked/present)
    el("div", { class: "block" }, el("div", { class: "block-h" }, "diagnostics"), renderDiag(step.diagnostics))
  );
  detail.scrollTop = 0;
}

const msgKey = (m) => `${m.role}\u0000${msgText(m.content)}`;

function prevSameActor(ep, turn, member) {
  for (let i = turn - 1; i >= 0; i--) {
    if (ep.steps[i].member_id === member) return ep.steps[i];
  }
  return null;
}

function stepTurn(d) {
  const ep = S.episode;
  S.turn = Math.max(0, Math.min(ep.steps.length - 1, S.turn + d));
  renderDetail();
  renderTurnAxis();
}

function renderMsg(m, src, isNew) {
  const role = m.role || "";
  // surface the [<seat>] visibility prefix when present (cross-seat interpolation)
  const text = msgText(m.content);
  const prefixM = text.match(/^\[([^\]]+)\]/);
  return el(
    "div",
    { class: `msg role-${role} ${isNew ? "is-new" : ""}` },
    el(
      "div",
      { class: "msg-head" },
      el("span", { class: "role" }, role),
      prefixM ? el("span", { class: "seat-prefix" }, `from ${prefixM[1]}`) : null,
      isNew ? el("span", { class: "new-tag" }, "● new") : null,
      el("span", { class: "src" }, src)
    ),
    renderBody(text, role)
  );
}

// render body; dim <think> blocks (debater reasoning channel) without truncating
function renderBody(text, role) {
  const body = el("div", { class: "msg-body" });
  const parts = text.split(/(<think>[\s\S]*?<\/think>)/g);
  for (const p of parts) {
    if (!p) continue;
    if (/^<think>[\s\S]*<\/think>$/.test(p)) body.append(el("span", { class: "think" }, p));
    else body.append(document.createTextNode(p));
  }
  return body;
}

function renderDiag(d) {
  if (!d || d.status === "absent") {
    return el(
      "div",
      { class: "diag" },
      el("span", { class: "stat absent" }, "absent"),
      el("span", { class: "hint" }, "no diagnostics sidecar for this run — KL / IS / entropy light up once the trainer emits them")
    );
  }
  if (d.status === "masked_out") {
    return el(
      "div",
      { class: "diag" },
      el("span", { class: "stat masked_out" }, "masked_out"),
      el("span", { class: "hint" }, "sequence fully clipped by mask_ratio — contributed no gradient (not zero)")
    );
  }
  const metric = (label, v) =>
    el(
      "span",
      { class: "metric" },
      el("span", { class: "ml" }, label),
      el("span", { class: `mv ${v?.mean == null ? "dash" : ""}` }, v?.mean == null ? "·" : fmt(v.mean))
    );
  return el(
    "div",
    { class: "diag" },
    el("span", { class: "stat present" }, "present"),
    metric("IS", d.importance_ratio),
    metric("KL", d.mismatch_kl),
    metric("H", d.entropy),
    el("span", { class: "metric" }, el("span", { class: "ml" }, "tokens"), el("span", { class: "mv" }, d.n_tokens ?? "·"))
  );
}

// ── synthetic alignment fallback (demo episodes have no server alignment) ──
// Mirror of schedule.align() — same statuses, same {index,status,actor,phase}
// shape the server's /alignment returns. Used for demo episodes (no server pass).
function synthAlignment(ep) {
  const slots = S.schedule?.slots;
  if (!slots) {
    return ep.steps.map((s) => ({ index: s.index, status: "matched", actor: s.member_id ?? "agent", phase: s.phase ?? null }));
  }
  const n = Math.max(slots.length, ep.steps.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const slot = slots[i] || null;
    const step = ep.steps[i] || null;
    let status;
    if (slot && step) status = (step.member_id ?? "agent") === slot.actor ? "matched" : "deviated";
    else if (slot) status = "truncated";
    else status = "extra";
    out.push({
      index: i,
      status,
      actor: slot ? slot.actor : step ? step.member_id ?? "agent" : "·",
      phase: (slot && slot.phase) || (step && step.phase) || null,
    });
  }
  return out;
}

// ── demo: synthetic single-agent + multi-turn to PROVE regime-genericity ──
// The real store is multi_agent only; these prove the timeline renders all 3.
function buildSyntheticEpisodes() {
  // 1) single_turn — one slot, member_id None
  const singleTurn = {
    example_id: "synthetic_single_turn",
    trajectory_id: "synthetic-st-0001",
    run_id: S.run || "demo",
    step: S.selected?.step ?? 0,
    kind: "single_turn",
    reward: 1.0,
    task: { answer: "42", prompt: "What is 6 × 7?" },
    members: [],
    mar: null,
    steps: [
      {
        index: 0,
        member_id: null,
        phase: null,
        prompt: [
          { role: "system", content: "You are a careful solver. Answer with the number only.", extra: {} },
          { role: "user", content: "What is 6 × 7?", extra: {} },
        ],
        completion: [{ role: "assistant", content: "<think>6 times 7 is 42.</think>\n42", extra: {} }],
        reward: 1.0,
        diagnostics: { status: "absent" },
      },
    ],
  };

  // 2) single_agent_multiturn — a model↔tool loop (one actor, prompt grows)
  const sys = { role: "system", content: "You are a tool-using agent. Use the search tool, then answer.", extra: {} };
  const q = { role: "user", content: "What is the capital of the country whose flag has a maple leaf?", extra: {} };
  const call1 = { role: "assistant", content: "<think>I should look this up.</think>", extra: { tool_calls: [{ name: "search", args: { q: "maple leaf flag" } }] } };
  const obs1 = { role: "tool", content: "[search] The maple leaf flag belongs to Canada.", extra: {} };
  const call2 = { role: "assistant", content: "<think>Capital of Canada is Ottawa.</think>", extra: { tool_calls: [{ name: "search", args: { q: "capital of Canada" } }] } };
  const obs2 = { role: "tool", content: "[search] Ottawa is the capital of Canada.", extra: {} };
  const multiTurn = {
    example_id: "synthetic_multiturn",
    trajectory_id: "synthetic-mt-0001",
    run_id: S.run || "demo",
    step: S.selected?.step ?? 0,
    kind: "single_agent_multiturn",
    reward: 1.0,
    task: { answer: "Ottawa", prompt: q.content },
    members: [],
    mar: null,
    steps: [
      { index: 0, member_id: null, phase: "act", prompt: [sys, q], completion: [call1], reward: null, diagnostics: { status: "absent" } },
      { index: 1, member_id: null, phase: "act", prompt: [sys, q, call1, obs1], completion: [call2], reward: null, diagnostics: { status: "absent" } },
      { index: 2, member_id: null, phase: "answer", prompt: [sys, q, call1, obs1, call2, obs2], completion: [{ role: "assistant", content: "Ottawa.", extra: {} }], reward: 1.0, diagnostics: { status: "absent" } },
    ],
  };

  return { singleTurn, multiTurn };
}

// synthetic schedules (so the timeline aligns demo episodes the same way the
// server would, regime-generically)
const SYNTH_SCHEDULES = {
  single_turn: { regime: "single_turn", source: "inferred", slots: [{ index: 0, actor: "agent", phase: null }] },
  single_agent_multiturn: {
    regime: "single_agent_multiturn", source: "inferred",
    slots: [
      { index: 0, actor: "agent", phase: "act" },
      { index: 1, actor: "agent", phase: "act" },
      { index: 2, actor: "agent", phase: "answer" },
    ],
  },
};

function loadDemo() {
  S.demo = true;
  const { singleTurn, multiTurn } = buildSyntheticEpisodes();
  window.__demo = { singleTurn, multiTurn };
  // present them as two selectable rows in the rail, plus a truncated variant to
  // exercise the "truncated" status.
  const truncatedMT = JSON.parse(JSON.stringify(multiTurn));
  truncatedMT.example_id = "synthetic_multiturn_truncated";
  truncatedMT.trajectory_id = "synthetic-mt-trunc";
  truncatedMT.steps = truncatedMT.steps.slice(0, 1); // ended early → truncated vs its 3-slot schedule
  truncatedMT.reward = 0.0;

  S.rows = [
    synthRow(singleTurn),
    synthRow(multiTurn),
    synthRow(truncatedMT),
  ];
  S.__synthEpisodes = { [singleTurn.trajectory_id]: singleTurn, [multiTurn.trajectory_id]: multiTurn, [truncatedMT.trajectory_id]: truncatedMT };
  $("#regime").textContent = "demo · 3 synthetic regimes";
  renderRail();
}
function synthRow(ep) {
  return {
    run_id: ep.run_id, step: ep.step, example_id: ep.example_id, trajectory_id: ep.trajectory_id,
    kind: ep.kind, members: ep.members, reward: ep.reward, advantage: 0, is_filtered: false,
    error: null, winner: null, metrics: {}, diagnostics: {}, transcript_shard: "(synthetic)",
    transcript_line: 0,
  };
}

// demo-mode selection: render a synthetic episode entirely client-side (no fetch),
// exercising the SAME client aligner the server's /alignment would feed.
function selectSynthetic(r) {
  const ep = S.__synthEpisodes[r.trajectory_id];
  S.selected = { example_id: r.example_id, step: r.step, line: 0, trajectory_id: r.trajectory_id };
  S.schedule = SYNTH_SCHEDULES[ep.kind] || SYNTH_SCHEDULES.single_turn;
  S.byStep = { example_id: r.example_id, by_step: [{ step: r.step, episodes: [{ step: r.step, transcript_line: 0, trajectory_id: r.trajectory_id, reward: ep.reward, winner: null, kind: ep.kind }] }] };
  S.episode = { ...ep };
  S.episode.alignment = synthAlignment(S.episode); // client aligner == server contract shape
  S.turn = Math.max(0, ep.steps.length - 1);
  $("#scrubbers").hidden = false;
  renderScrubbers();
  renderDetail();
  renderRail();
}

// ── live feed ─────────────────────────────────────────────────────
function connectLive() {
  try {
    const es = new EventSource("/api/stream");
    es.onopen = () => { $("#live").textContent = "live"; $("#live").classList.add("on"); };
    es.onmessage = () => { if (!S.demo) loadEpisodes(); };
    es.onerror = () => { $("#live").textContent = "live offline"; $("#live").classList.remove("on"); };
  } catch { $("#live").textContent = "live n/a"; }
}

// ── wiring ────────────────────────────────────────────────────────
$("#run").addEventListener("change", (e) => { S.run = e.target.value; S.selected = null; S.episode = null; $("#scrubbers").hidden = true; $("#detail").replaceChildren(el("div", { class: "placeholder big" }, "select an episode", el("span", { class: "dim" }, "— the scope is idle"))); loadRuns(); });
$("#sort").addEventListener("change", (e) => { S.sort = e.target.value; loadEpisodes(); });
$("#order").addEventListener("change", (e) => { S.order = e.target.value; loadEpisodes(); });
$("#demo").addEventListener("click", loadDemo);

// keyboard: ←/→ scrub turns, ↑/↓ scrub the rail
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (!S.episode) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); stepTurn(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepTurn(1); }
});

loadRuns().catch((e) => {
  $("#list").replaceChildren(el("div", { class: "err" }, `failed to load runs:\n${e}`));
});
connectLive();
