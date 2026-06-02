// ─────────────────────────────────────────────────────────────────────────
// rollout-viewer · DIRECTION B — wire-protocol analyzer / editorial
//
// Temporality is the spine: a TURN scrubber (frames) + a TRAINING-STEP scrubber,
// both first-class, schedule-aware, regime-generic. The real store only carries
// multi_agent episodes, so single_turn + single_agent_multiturn are exercised via
// synthetic FIXTURE runs that travel the SAME data path (schedule + alignment +
// visibility delta) — nothing about the timeline special-cases debate.
// ─────────────────────────────────────────────────────────────────────────

// ── tiny DOM helpers ──────────────────────────────────────────────────────
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'dataset') Object.assign(n.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, '');
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) n.append(c.nodeType ? c : document.createTextNode(c));
  return n;
};
const fmtNum = (x, d = 3) =>
  x == null ? '·' : (typeof x === 'number' ? (Number.isInteger(x) ? String(x) : x.toFixed(d)) : String(x));
// token-count badge text: thousands-separated when present, an em-dash when the
// backend reports null (genuinely no local id capture) — NEVER a fabricated 0.
const fmtTok = (n) => (n == null ? '— tok' : `${n.toLocaleString()} tok`);
// compact list-cell token figure (e.g. 20.4k); a null total renders an em-dash,
// never a 0 — a missing rollup is not "zero tokens".
function tokCell(n) {
  if (n == null) return el('span', { class: 'tok-na' }, '—');
  const txt = n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
  return el('span', { class: 'tok-c tn' }, txt);
}

// ── seat → hue resolution (debate seats fixed; others assigned cyclically) ──
const SEAT_HUES = {
  debater_a: 'var(--seat-a)', debater_b: 'var(--seat-b)', judge: 'var(--seat-j)', agent: 'var(--seat-x)',
};
const GENERIC_HUES = ['var(--seat-a)', 'var(--seat-b)', 'var(--seat-j)', 'var(--seat-x)', 'var(--cy)', 'var(--amber)'];
function seatHue(actor, order = []) {
  if (SEAT_HUES[actor]) return SEAT_HUES[actor];
  const i = order.indexOf(actor);
  return GENERIC_HUES[(i < 0 ? 0 : i) % GENERIC_HUES.length];
}

// ── visibility delta (client-side; honors the contract conventions) ─────────
// A turn's NEW-THIS-TURN content = messages in this step's prompt not present in
// this actor's *previous* step's prompt, matched by (role, content) membership
// (NEVER by equality with the raw completion). The first turn of an actor: the
// whole prompt is new. Cross-seat injections carry a [<seat>] prefix at pos 0.
function previousStepFor(steps, idx) {
  const actor = stepActor(steps[idx]);
  for (let j = idx - 1; j >= 0; j--) if (stepActor(steps[j]) === actor) return steps[j];
  return null;
}
function stepActor(step) { return step.member_id ?? 'agent'; }
function msgKey(m) {
  const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
  return m.role + '\u0000' + c;
}
function visibilityDelta(steps, idx) {
  const cur = steps[idx];
  const prev = previousStepFor(steps, idx);
  const prevKeys = new Set((prev?.prompt ?? []).map(msgKey));
  const fresh = (prev ? cur.prompt.filter(m => !prevKeys.has(msgKey(m))) : cur.prompt.slice());
  return { fresh, hadPrior: !!prev, freshKeys: new Set(fresh.map(msgKey)) };
}
// attribute a [<seat>] prefix injection to its source seat
const SEAT_PREFIX = /^\s*\[([^\]]+)\]/;
function injectionSource(m) {
  if (m.role !== 'user') return null;
  const c = typeof m.content === 'string' ? m.content : '';
  const mm = c.match(SEAT_PREFIX);
  return mm ? mm[1] : null;
}

// ── bare-LaTeX pre-pass ─────────────────────────────────────────────────────
// Model output mixes properly delimited math ($…$, $$…$$, \(…\), \[…\]) — which
// KaTeX auto-render handles — with BARE commands that escaped delimiting (e.g.
// "1/ \sqrt{2}" or a stray "\langle v | \psi \rangle"). This pass wraps each run
// of bare math in \(…\) so auto-render typesets it, WITHOUT:
//   · touching text already inside a math delimiter (we split on delimiters first),
//   · touching <think>/<reasoning> spans (callers pass only non-think segments),
//   · swallowing prose — a run extends through math tokens only and STOPS at the
//     first prose word, so "\sqrt{2}, in the second row" wraps just "\sqrt{2}".
// Operates on plain strings → plain strings; HTML-escaping is preserved because
// renderContent still emits everything via text nodes.
// Single, self-contained commands only. Environment commands (\begin/\end/…
// pmatrix) are deliberately EXCLUDED: a bare environment fragment is never valid
// standalone, and real matrices in this data arrive already inside $$…$$.
const BARE_CMDS = new Set([
  'sqrt', 'frac', 'cdot', 'times', 'otimes', 'langle', 'rangle', 'psi', 'phi',
  'alpha', 'beta', 'pi', 'theta', 'sum', 'int', 'infty',
  'vec', 'hat', 'lambda', 'propto', 'det',
]);
// Canonicalize paired $$…$$ → \[…\] FIRST. This data sometimes begins mid-stream
// with an unbalanced single '$' (orphaned </think> content), which would corrupt
// auto-render's left-to-right $-pairing and split every $$ block into two inline
// $…$ pairs. Promoting balanced $$ to \[…\] insulates display math from that.
const DISPLAY_DOLLARS = /\$\$([\s\S]*?)\$\$/g;
function normalizeDisplayMath(text) {
  return text.indexOf('$$') < 0 ? text : text.replace(DISPLAY_DOLLARS, (_, body) => `\\[${body}\\]`);
}
// delimiter spans we must leave untouched (already math)
const MATH_DELIM = /(\\\[[\s\S]*?\\\]|\$[^$\n]*?\$|\\\([\s\S]*?\\\))/g;
function wrapBareLatex(text) {
  text = normalizeDisplayMath(text);
  if (text.indexOf('\\') < 0) return text;
  let out = '';
  let last = 0;
  for (const m of text.matchAll(MATH_DELIM)) {
    out += wrapBareSegment(text.slice(last, m.index));
    out += m[0]; // already-delimited — verbatim
    last = m.index + m[0].length;
  }
  out += wrapBareSegment(text.slice(last));
  return out;
}
// token-scan a delimiter-free segment, wrapping maximal bare-math runs.
function wrapBareSegment(seg) {
  if (seg.indexOf('\\') < 0) return seg;
  let out = '', i = 0, n = seg.length;
  while (i < n) {
    // a run must START at a known bare command
    if (seg[i] === '\\' && isBareCmdAt(seg, i)) {
      const end = scanRun(seg, i);
      out += '\\(' + seg.slice(i, end) + '\\)';
      i = end;
    } else {
      out += seg[i++];
    }
  }
  return out;
}
function isBareCmdAt(s, i) {
  const m = /^\\([A-Za-z]+)/.exec(s.slice(i));
  return !!m && BARE_CMDS.has(m[1]);
}
// Extend a run from a starting command through adjacent math tokens. The run ends
// at the last "solid" math boundary (end of a \cmd or a {…} arg); interior glue
// (operators, digits, single-letter vars, spaces) is only KEPT when it bridges to
// another command/brace within the run. So "\sqrt{2}, 0), in the…" → "\sqrt{2}"
// (the ", 0)," is trailing glue, dropped), while "\langle v | \psi \rangle"
// stays whole (the " v | " glue bridges \langle → \psi → \rangle).
function scanRun(s, start) {
  let i = start, n = s.length, lastSolid = start;
  while (i < n) {
    const c = s[i];
    if (c === '\\') {                       // a command token
      const m = /^\\([A-Za-z]+)/.exec(s.slice(i));
      if (m && BARE_CMDS.has(m[1])) { i += m[0].length; lastSolid = i; continue; }
      break;                                // unknown command → stop
    }
    if (c === '{') {                        // a (possibly nested) brace arg
      const end = matchBrace(s, i);
      if (end < 0) break;
      i = end; lastSolid = i; continue;
    }
    // glue: operators, digits, single-letter vars, spaces. Keep scanning, but do
    // NOT advance lastSolid — only a later command/brace will commit this glue.
    if (c === ' ' || c === '\t' || /[0-9^_|=+\-*/().,]/.test(c)) { i++; continue; }
    if (/[A-Za-z]/.test(c)) {
      const w = /^[A-Za-z]+/.exec(s.slice(i))[0];
      if (w.length === 1) { i += 1; continue; }   // single-letter variable = glue
      break;                                       // ≥2 letters = prose word → stop
    }
    break;
  }
  return lastSolid;                          // trim trailing glue
}
function matchBrace(s, i) {
  let depth = 0;
  for (let k = i; k < s.length; k++) {
    if (s[k] === '{') depth++;
    else if (s[k] === '}') { depth--; if (depth === 0) return k + 1; }
  }
  return -1;
}

// ── lightweight prose markdown (PROSE segments only) ────────────────────────
// The debate completions are written in markdown: **bold** run-in headers and the
// occasional leading "## "/"### " ATX header. We render exactly those two inline,
// and NOTHING else (no lists/links/code) — this is a transcript viewer, not a CMS.
// Applied ONLY to non-think prose (callers never pass <think> content here), and
// AFTER wrapBareLatex so math delimiters are intact: KaTeX then typesets the
// resulting text nodes, including any inside a <strong>. Bold markers never live
// inside math ($…$ carries no '**'), so splitting on '**' can't corrupt a formula.
const BOLD_RE = /\*\*([^\n]+?)\*\*/g;
const CODE_RE = /`([^`\n]+)`/g;
const ATX_RE = /^(#{2,3})\s+(.+?)\s*#*$/;
// emit inline markdown for one already-bare-latex'd line into `parent`: inline
// `code` spans FIRST (their content is literal — KaTeX's default ignoredTags skips
// <code>, and no **bold** applies inside), then **bold** in the surrounding prose.
// This is why a model quoting a tag as `<think>` renders as a code chip, not raw
// backticks or a mis-parsed element.
function appendInline(parent, line) {
  let last = 0;
  for (const m of line.matchAll(CODE_RE)) {
    if (m.index > last) appendBold(parent, line.slice(last, m.index));
    parent.append(el('code', { class: 'md-c' }, m[1]));
    last = m.index + m[0].length;
  }
  if (last < line.length) appendBold(parent, line.slice(last));
}
function appendBold(parent, s) {
  let last = 0;
  for (const m of s.matchAll(BOLD_RE)) {
    if (m.index > last) parent.append(document.createTextNode(s.slice(last, m.index)));
    parent.append(el('strong', { class: 'md-b' }, m[1]));
    last = m.index + m[0].length;
  }
  if (last < s.length) parent.append(document.createTextNode(s.slice(last)));
}
// markdown a PROSE segment: split into lines, lift "## "/"### " to a header
// element, render inline bold elsewhere. Newlines are re-emitted as text so the
// `white-space: pre-wrap` body keeps its layout. Returns nodes appended to `frag`.
function appendProse(frag, raw) {
  const text = wrapBareLatex(raw);
  const lines = text.split('\n');
  lines.forEach((line, i) => {
    const h = ATX_RE.exec(line);
    if (h) {
      const lvl = h[1].length;            // 2 or 3
      const hd = el('span', { class: `md-h md-h${lvl}` });
      appendInline(hd, h[2]);
      frag.append(hd);
    } else {
      appendInline(frag, line);
    }
    if (i < lines.length - 1) frag.append(document.createTextNode('\n'));
  });
}

// ── think-block aware text rendering (renderer re-wraps <think>/<reasoning>) ──
// Tag-scanning is INLINE-CODE-AWARE: a `<think>`/`</think>` wrapped in backticks is
// the model QUOTING the format (Qwen3.5 does this constantly), not a span boundary.
// Where reasoning lives in its own reasoning_content field, every <think> in the
// visible content is such a quote — treating it as a span mis-segments and dims
// arbitrary prose. So we only ever split on tags that sit OUTSIDE inline code.
const CLOSE_TAG = { think: '</think>', reasoning: '</reasoning>' };
function inRanges(i, ranges) {
  for (const [a, b] of ranges) if (i >= a && i < b) return true;
  return false;
}
function codeRanges(s) {
  const r = [];
  for (const m of s.matchAll(CODE_RE)) r.push([m.index, m.index + m[0].length]);
  return r;
}
const OPEN_TAG = /<(think|reasoning)>/g;
// next REAL (non-code) think/reasoning span at/after `from`: opener and its matching
// closer both outside inline code. Openers with no valid closer are skipped (treated
// as prose), matching the original "requires a close tag" behavior.
function nextThinkSpan(s, from, ranges) {
  OPEN_TAG.lastIndex = from;
  let m;
  while ((m = OPEN_TAG.exec(s))) {
    if (inRanges(m.index, ranges)) continue;            // quoted opener → not a span
    const tag = CLOSE_TAG[m[1]];
    let ci = s.indexOf(tag, m.index + m[0].length);
    while (ci >= 0 && inRanges(ci, ranges)) ci = s.indexOf(tag, ci + tag.length);
    if (ci >= 0) return { open: m.index, close: ci + tag.length };
    // unclosed opener: keep scanning for a later well-formed span
  }
  return null;
}
function emitProse(frag, text, a, b) {
  const seg = text.slice(a, b);
  // seat-prefix highlight only at the very start of the content
  const pm = a === 0 ? seg.match(SEAT_PREFIX) : null;
  if (pm) {
    frag.append(el('span', { class: 'seatpre' }, pm[0]));
    appendProse(frag, seg.slice(pm[0].length));
  } else appendProse(frag, seg);
}
// extractThink: whether to pull inline <think>…</think> out as dimmed spans. The
// caller passes FALSE when the message already carries reasoning in its own
// reasoning_content field — then every <think> in the visible content is the model
// QUOTING the format, never a real span, and extracting it would mis-segment prose
// (the model's own backticks pair ambiguously, so no in-text heuristic is reliable).
// Passed TRUE for inline-reasoning content (fixtures, models that don't split).
function renderContent(content, extractThink = true) {
  const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
  const frag = document.createDocumentFragment();
  if (extractThink) {
    // A real inline think block is emitted at the very START of the turn (reasoning
    // first): "<think>…</think>\n<answer>". Extract ONLY a leading, matched, non-code
    // think span; any <think> later in the prose is the model quoting the format
    // (and the model's own backticks pair ambiguously, so mid-text heuristics fail).
    const ranges = codeRanges(text);
    const lead = text.match(/^\s*/)[0].length;
    const span = nextThinkSpan(text, lead, ranges);
    if (span && span.open === lead) {
      if (lead) frag.append(document.createTextNode(text.slice(0, lead)));
      // think content: rendered verbatim, NOT math-prepassed, NOT markdown'd
      frag.append(el('span', { class: 'think' }, text.slice(span.open, span.close)));
      emitProse(frag, text, span.close, text.length);
      return frag;
    }
  }
  emitProse(frag, text, 0, text.length);
  return frag;
}

// reasoning_content is ALREADY the model's think channel — render it as prose+math
// but DO NOT re-scan for <think>/<reasoning> delimiters. A model that quotes the
// rendered format inside its reasoning (e.g. `<think>` / `</think>` as inline code,
// as Qwen3.5 routinely does) would otherwise have those literals misread as span
// boundaries, mis-segmenting and dimming arbitrary mid-reasoning prose. The caller
// already wraps the whole thing in a dimmed .think container.
function renderReasoning(text) {
  const frag = document.createDocumentFragment();
  appendProse(frag, typeof text === 'string' ? text : JSON.stringify(text, null, 2));
  return frag;
}

// ── KaTeX typesetting hook ───────────────────────────────────────────────────
// Run after the detail DOM is mounted. auto-render walks text nodes, finds the
// delimiters, and replaces them with typeset math. think spans get rendered too
// (they may contain delimited $…$), which is fine — only the bare-LaTeX prepass
// skips them. throwOnError:false → malformed fragments stay as source, never crash.
const KATEX_DELIMS = [
  { left: '$$', right: '$$', display: true },
  { left: '\\[', right: '\\]', display: true },
  { left: '\\(', right: '\\)', display: false },
  { left: '$', right: '$', display: false },
];
function typesetMath(root) {
  if (!root || typeof window.renderMathInElement !== 'function') return;
  window.renderMathInElement(root, {
    delimiters: KATEX_DELIMS,
    throwOnError: false,
    ignoredClasses: ['seatpre'],
  });
}

// ── synthetic fixtures: the OTHER two regimes (same data path) ──────────────
// Each fixture run = a schedule + a set of episodes whose steps carry prompt/
// completion exactly as the real Episode model does. Alignment is computed by the
// same align() logic the backend uses, locally, so the timeline renders identically.
function buildFixtures() {
  // ---- single_turn: one slot, one step. plus a deviated case (wrong actor). ----
  const stSched = { regime: 'single_turn', source: 'inferred', slots: [{ index: 0, actor: 'agent', phase: null }] };
  const stEpisode = (ex, q, a, reward) => ({
    example_id: ex, run_id: '∅fixtures·single_turn', step: 0, kind: 'single_turn',
    members: [], reward, advantage: 0, mar: null, metrics: { accuracy: reward }, is_filtered: false, error: null,
    task: { question: q }, trajectory_id: 'fx-st-' + ex,
    steps: [{
      index: 0, member_id: null, phase: null, reward, advantage: 0, is_truncated: false, diagnostics: null, extras: {},
      prompt: [
        { role: 'system', content: 'You are a careful solver. Answer with the letter only.' },
        { role: 'user', content: q },
      ],
      completion: [{ role: 'assistant', content: a }],
    }],
  });
  const stRun = {
    run_id: '∅fixtures·single_turn', regime: 'single_turn', schedule: stSched,
    episodes: [
      stEpisode('mmlu-114', 'Which gas is most abundant in dry air?\n(A) O₂ (B) N₂ (C) CO₂ (D) Ar', '<think>Nitrogen is ~78% of dry air.</think>\nB', 1),
      stEpisode('mmlu-209', 'The derivative of sin(x) is?\n(A) cos(x) (B) -cos(x) (C) -sin(x) (D) tan(x)', 'A', 1),
      stEpisode('mmlu-330', 'Capital of Australia?\n(A) Sydney (B) Melbourne (C) Canberra (D) Perth', '<think>Common trap — not Sydney.</think>\nA', 0),
    ],
  };

  // ---- single_agent_multiturn: a tool-loop (model → tool → model → tool → model). ----
  // Schedule = 5 slots, all "agent". One episode runs full; one TRUNCATES early
  // (3 turns) → tail slots render `truncated`; one runs EXTRA (7 turns) → `extra`.
  const maSched = {
    regime: 'single_agent_multiturn', source: 'inferred',
    slots: [0, 1, 2, 3, 4].map(i => ({ index: i, actor: 'agent', phase: i % 2 === 0 ? 'reason' : 'tool' })),
  };
  const tool = (name, args, out) => [
    { role: 'assistant', content: `<think>I should call ${name}.</think>`, extra: { tool_calls: [{ function: { name, arguments: args } }] } },
    { role: 'tool', content: out },
  ];
  function loopEpisode(ex, q, nTurns, reward, truncated) {
    const steps = [];
    let prompt = [
      { role: 'system', content: 'You are a research agent with web_search + calculator. Think, then act.' },
      { role: 'user', content: q },
    ];
    for (let t = 0; t < nTurns; t++) {
      const isTool = t % 2 === 1;
      const completion = isTool
        ? tool('web_search', `{"q":"step ${t} of ${q.slice(0, 18)}…"}`, `result block #${t}: 3 hits, top score 0.82`)
        : [{ role: 'assistant', content: `<think>Turn ${t}: synthesize what I have so far.</think>\n${t === nTurns - 1 ? `FINAL: ${reward ? 'answer settled' : 'gave up'}` : `partial reasoning at depth ${t}`}` }];
      steps.push({
        index: t, member_id: null, phase: isTool ? 'tool' : 'reason',
        reward: t === nTurns - 1 ? reward : null, advantage: 0,
        is_truncated: truncated && t === nTurns - 1, diagnostics: null, extras: { turn: t },
        prompt: prompt.slice(),
        completion,
      });
      // grow the prompt: prior completion(s) reappear, re-materialized
      prompt = prompt.concat(completion.map(m => ({ role: m.role === 'tool' ? 'user' : m.role, content: m.content })));
    }
    return {
      example_id: ex, run_id: '∅fixtures·tool_loop', step: 0, kind: 'single_agent_multiturn',
      members: [], reward, advantage: 0, mar: null, metrics: { success: reward, turns: nTurns },
      is_filtered: false, error: truncated ? null : null, task: { question: q }, trajectory_id: 'fx-loop-' + ex,
      steps,
    };
  }
  const loopRun = {
    run_id: '∅fixtures·tool_loop', regime: 'single_agent_multiturn', schedule: maSched,
    episodes: [
      loopEpisode('agt-7', 'How many moons does the 6th planet from the Sun have?', 5, 1, false),       // full → all matched
      loopEpisode('agt-12', 'GDP of France in 2023 in USD trillions?', 3, 0, true),                      // short → tail truncated
      loopEpisode('agt-19', 'Shortest path A→F in the supplied graph (multi-hop)?', 7, 1, false),         // long → 2 extra
    ],
  };

  return new Map([[stRun.run_id, stRun], [loopRun.run_id, loopRun]]);
}
const FIXTURES = buildFixtures();

// local re-implementation of schedule.align (the contract), so fixtures get the
// SAME alignment shape the backend attaches to real episodes.
function alignLocal(episode, schedule) {
  const slots = schedule.slots, steps = episode.steps;
  const n = Math.max(slots.length, steps.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const slot = slots[i] ?? null, step = steps[i] ?? null;
    let status;
    if (slot && step) status = (stepActor(step) === slot.actor) ? 'matched' : 'deviated';
    else if (slot) status = 'truncated';
    else status = 'extra';
    out.push({
      index: i, status,
      actor: slot ? slot.actor : stepActor(step),
      phase: (slot ? slot.phase : null) ?? (step ? step.phase : null),
    });
  }
  return out;
}

// ── data layer (real API + fixture runs, one interface) ─────────────────────
const api = async (p) => {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${r.status} ${p}`);
  return r.json();
};
const isFixtureRun = (run) => FIXTURES.has(run);

async function fetchRuns() {
  const real = await api('/api/runs').catch(() => []);
  const fx = [...FIXTURES.values()].map(f => ({
    run_id: f.run_id, steps: [0], updated: null, fixture: true, regime: f.regime,
  }));
  return [...real, ...fx];
}
async function fetchSchedule(run) {
  if (isFixtureRun(run)) return FIXTURES.get(run).schedule;
  return api(`/api/runs/${encodeURIComponent(run)}/schedule`);
}
// The declared run setup (config-distilled) + recovered schedule. Returns null
// when the run has no setup.json (synced without --configs-dir → backend 404) and
// for fixtures (which have a schedule but no declared config).
async function fetchSetup(run) {
  if (isFixtureRun(run)) return null;
  try { return await api(`/api/runs/${encodeURIComponent(run)}/setup`); }
  catch (e) { if (String(e).startsWith('404')) return null; throw e; }
}
async function fetchEpisodesList(run, sort, order) {
  if (isFixtureRun(run)) {
    const f = FIXTURES.get(run);
    return f.episodes.map((ep, i) => ({
      run_id: run, step: ep.step, example_id: ep.example_id, trajectory_id: ep.trajectory_id,
      rollout_id: null, kind: ep.kind, members: ep.members, reward: ep.reward, advantage: ep.advantage,
      is_filtered: ep.is_filtered, error: ep.error, winner: ep.mar?.categorical?.winner ?? null,
      answers: null, metrics: ep.metrics, diagnostics: {}, transcript_shard: null, transcript_line: i, _fixture: ep,
    }));
  }
  const j = await api(`/api/runs/${encodeURIComponent(run)}/episodes?sort=${sort}&order=${order}&limit=4000`);
  return j.rows || [];
}
async function fetchEpisode(run, step, line, fixtureEp) {
  if (isFixtureRun(run)) {
    const ep = fixtureEp ?? FIXTURES.get(run).episodes[line];
    return { ...ep, alignment: alignLocal(ep, FIXTURES.get(run).schedule) };
  }
  return api(`/api/episodes/${encodeURIComponent(run)}/${step}/${line}`);
}
async function fetchExample(run, exampleId) {
  if (isFixtureRun(run)) {
    const f = FIXTURES.get(run);
    const eps = f.episodes
      .map((ep, i) => ({ ep, i }))
      .filter(({ ep }) => ep.example_id === exampleId);
    const byStep = new Map();
    for (const { ep, i } of eps) {
      if (!byStep.has(ep.step)) byStep.set(ep.step, []);
      byStep.get(ep.step).push({
        step: ep.step, transcript_line: i, trajectory_id: ep.trajectory_id,
        reward: ep.reward, winner: ep.mar?.categorical?.winner ?? null, kind: ep.kind,
      });
    }
    return { example_id: exampleId, by_step: [...byStep.entries()].sort((a, b) => a[0] - b[0]).map(([step, episodes]) => ({ step, episodes })) };
  }
  return api(`/api/runs/${encodeURIComponent(run)}/examples/${encodeURIComponent(exampleId)}`);
}

// ── app state ───────────────────────────────────────────────────────────────
const S = {
  runs: [], run: null, sort: 'step', order: 'asc',
  rows: [], sel: null,          // selected row index in rows[]
  episode: null,                // loaded Episode (+ alignment)
  turn: 0,                      // selected turn index
  example: null,                // examples-across-steps payload
  curStep: null,                // step shown in the step scrubber
  setup: null,                  // declared run setup (+ recovered schedule), per run
  setupOpen: false,             // is the setup spec-sheet showing in the inspector?
};

// ── episode list (left, table-forward) ──────────────────────────────────────
function rewardClass(x) { return x == null ? 'zero' : x > 0 ? 'pos' : x < 0 ? 'neg' : 'zero'; }
function episodeFlipped(row) {
  const m = row.metrics || {};
  return Object.keys(m).some(k => k.startsWith('flipped/') && m[k] > 0);
}

function renderList() {
  const scroll = $('#list-scroll');
  $('#list-count').textContent = S.rows.length ? `${S.rows.length} ep` : '—';
  if (!S.rows.length) { scroll.replaceChildren(el('div', { class: 'empty' }, el('span', { class: 'big' }, 'no episodes'))); return; }

  const cols = [
    { k: 'step', label: 'st', num: true },
    { k: 'example_id', label: 'example' },
    { k: 'kind', label: 'regime' },
    { k: 'align', label: 'turns' },
    { k: 'winner', label: 'win' },
    { k: 'reward', label: 'r', num: true },
    { k: 'tokens.episode.completion', label: 'tok', num: true },
    { k: 'flip', label: 'flip' },
    { k: 'ret', label: 'tx', num: true },
  ];
  const thead = el('thead', {}, el('tr', {},
    ...cols.map(c => el('th', {
      class: c.num ? 'num' : '', dataset: { sortk: c.k },
      onclick: () => onSortClick(c.k),
    }, c.label, (S.sort === c.k ? el('span', { class: 'ar' }, S.order === 'desc' ? '▾' : '▴') : null)))
  ));

  const seatOrder = [...new Set(S.rows.flatMap(r => r.members || []))];
  const table = el('table', { class: 'episodes' }, thead);
  // Grouped view ONLY when sorted by step: the rows arrive step-ascending, so each
  // step becomes a <tbody> (top-ruled) opened by a summary band, and within it the
  // same-question debates are clustered under a sub-rule. Any other sort → flat.
  if (S.sort === 'step') {
    const ncol = cols.length;
    let i = 0;
    while (i < S.rows.length) {
      const step = S.rows[i].step;
      const grp = [];
      while (i < S.rows.length && S.rows[i].step === step) { grp.push({ r: S.rows[i], idx: i }); i++; }
      const tb = el('tbody', { class: 'step-group' });
      tb.append(stepBandRow(step, grp.map(x => x.r), ncol));
      // cluster same-question debates (preserve first-seen question order)
      const byQ = new Map();
      for (const x of grp) { (byQ.get(x.r.example_id) ?? byQ.set(x.r.example_id, []).get(x.r.example_id)).push(x); }
      for (const [q, qg] of byQ) {
        tb.append(qSubruleRow(q, qg.map(x => x.r), ncol));
        for (const { r, idx } of qg) tb.append(episodeRow(r, idx, seatOrder));
      }
      table.append(tb);
    }
  } else {
    const tbody = el('tbody');
    S.rows.forEach((r, i) => tbody.append(episodeRow(r, i, seatOrder)));
    table.append(tbody);
  }
  scroll.replaceChildren(table);
}

// one episode <tr> (shared by flat + grouped views; idx = position in S.rows)
function episodeRow(r, i, seatOrder) {
  const flipped = episodeFlipped(r);
  const retained = r.transcript_line >= 0;
  return el('tr', {
    class: (S.sel === i ? 'sel ' : '') + (r._fixture ? 'fixture-row' : ''),
    onclick: () => selectEpisode(i),
  },
    el('td', { class: 'num' }, String(r.step)),
    el('td', {}, el('span', { class: 'ex' }, String(r.example_id))),
    el('td', { class: 'kind' }, el('span', { class: `chip k-${r.kind}` }, (r.kind || '').replace('single_agent_multiturn', 'multiturn').replace('multi_agent', 'multi-agent').replace('single_turn', 'single'))),
    el('td', {}, sparkbar(r, seatOrder)),
    el('td', {}, r.winner ? el('span', { class: `win win-${r.winner}` }, r.winner) : '·'),
    el('td', { class: 'num' }, el('span', { class: `reward ${rewardClass(r.reward)}` }, fmtNum(r.reward, 2))),
    el('td', { class: 'num' }, tokCell(r.tokens?.episode?.completion)),
    el('td', {}, flipped ? el('span', { class: 'flip' }, '⇄') : '·'),
    el('td', { class: 'num' }, retained ? '·' : el('span', { class: 'ret-no' }, '∅')),
  );
}

// ── per-group aggregation (drives the step band + question sub-rule) ──────────
// All inputs come from the index row: metrics{} (final_correct/agreement/judge_…),
// winner, and answers{seat: letter}. reward/episode_scalar is deliberately ignored
// — for self-play it carries no signal.
const _m = (r, k) => (typeof r.metrics?.[k] === 'number' ? r.metrics[k] : null);
const _pct = (x) => (x == null ? '—' : `${Math.round(x * 100)}%`);
function aggStats(rows) {
  let acc = 0, accN = 0, agr = 0, agrN = 0, jdg = 0, jdgN = 0, flip = 0, correct = 0, agree = 0;
  const win = { debater_a: 0, debater_b: 0, tie: 0 };
  const qs = new Set();
  for (const r of rows) {
    qs.add(r.example_id);
    const fc = _m(r, 'final_correct/debater_a'); if (fc != null) { acc += fc; accN++; if (fc > 0) correct++; }
    const ag = _m(r, 'agreement'); if (ag != null) { agr += ag; agrN++; if (ag > 0) agree++; }
    const jc = _m(r, 'judge_correct_vs_gt'); if (jc != null) { jdg += jc; jdgN++; }
    if (episodeFlipped(r)) flip++;
    if (r.winner in win) win[r.winner]++;
  }
  return {
    nQ: qs.size, n: rows.length, correct, agree,
    acc: accN ? acc / accN : null, agr: agrN ? agr / agrN : null, jdg: jdgN ? jdg / jdgN : null,
    flip: rows.length ? flip / rows.length : 0, win,
  };
}
// GT letter derived from the data: the answer of any debate that scored correct.
function groundTruth(rows) {
  for (const r of rows) {
    const a = r.answers?.debater_a;
    if (a != null && (_m(r, 'final_correct/debater_a') ?? 0) > 0) return a;
  }
  return null;
}
// answer spread across a question's debates, e.g. "B×3·C×1" (the disagreement view)
function answerSpread(rows) {
  const c = {};
  for (const r of rows) { const a = r.answers?.debater_a; if (a != null) c[a] = (c[a] || 0) + 1; }
  const ent = Object.entries(c).sort((x, y) => y[1] - x[1]);
  return ent.length ? ent.map(([k, v]) => `${k}×${v}`).join('·') : '—';
}
function stepBandRow(step, rows, ncol) {
  const a = aggStats(rows);
  // content wrapped in an inner flex div — a display:flex directly on a colspan <td>
  // detaches it from the table's column model and collapses every other row to 0.
  return el('tr', { class: 'step-band' }, el('td', { class: 'sb', colspan: ncol },
    el('div', { class: 'sb-inner' },
      el('span', { class: 'sb-step' }, `STEP ${step}`),
      el('span', { class: 'sb-stat' }, `${a.nQ}q · ${a.n}ep`),
      el('span', { class: 'sb-stat' }, `acc ${_pct(a.acc)}`),
      el('span', { class: 'sb-stat' }, `agree ${_pct(a.agr)}`),
      el('span', { class: 'sb-stat' }, `judge✓ ${_pct(a.jdg)}`),
      el('span', { class: 'sb-stat' }, `win a${a.win.debater_a}·b${a.win.debater_b}·t${a.win.tie}`),
      el('span', { class: 'sb-stat' }, `flip ${_pct(a.flip)}`),
    )));
}
function qSubruleRow(q, rows, ncol) {
  const a = aggStats(rows);
  const gt = groundTruth(rows);
  return el('tr', { class: 'q-subrule' }, el('td', { colspan: ncol },
    el('span', { class: 'qs-id' }, `#${q}`),
    el('span', { class: 'qs-gt' }, gt ? `gt ${gt}` : 'gt ?'),
    el('span', { class: 'qs-spread' }, answerSpread(rows)),
    el('span', { class: 'qs-corr' }, `${a.correct}/${a.n}✓`),
  ));
}

// a tiny per-row alignment sparkbar (needs the schedule length; we lazily fill)
function sparkbar(row, seatOrder) {
  const bar = el('span', { class: 'sparkbar' });
  const n = S.scheduleLen || (row.members?.length ? 5 : 1);
  // we only know exact statuses once the episode loads; approximate with member count
  const ticks = Math.max(1, row._fixture ? row._fixture.steps.length : (S.scheduleLen || n));
  for (let i = 0; i < Math.min(ticks, 8); i++) bar.append(el('i', { class: 'matched' }));
  return bar;
}

function onSortClick(k) {
  if (k === 'align' || k === 'flip' || k === 'ret') return; // client-only columns
  if (S.sort === k) S.order = S.order === 'desc' ? 'asc' : 'desc';
  else { S.sort = k; S.order = 'desc'; }
  $('#sel-sort').value = S.sort; $('#sel-dir').value = S.order;
  loadEpisodes();
}

// ── selection + load an episode ─────────────────────────────────────────────
async function selectEpisode(i) {
  S.sel = i; renderList();
  const row = S.rows[i];
  const detail = $('#detail-scroll');
  // selecting an episode supersedes the setup sheet
  if (S.setupOpen) { S.setupOpen = false; $('#btn-setup').classList.remove('on'); }
  $('#ep-head').hidden = false; $('#deck').hidden = false;

  if (!row._fixture && row.transcript_line < 0) {
    S.episode = null;
    renderEpHead(row, null);
    $('#deck').replaceChildren();
    detail.replaceChildren(el('div', { class: 'empty' },
      el('span', { class: 'big' }, 'transcript not retained'),
      el('span', {}, 'metrics + diagnostics are still in the index (ρ<1 elision)')));
    return;
  }
  detail.replaceChildren(el('div', { class: 'empty skel' }, el('span', {}, 'decoding frames…')));

  let ep;
  try { ep = await fetchEpisode(S.run, row.step, row.transcript_line, row._fixture); }
  catch (e) { detail.replaceChildren(el('div', { class: 'empty err' }, String(e))); return; }
  S.episode = ep;
  S.turn = 0;
  S.curStep = row.step;
  renderEpHead(row, ep);
  renderDeck();
  renderTurn();

  // training-step scrubber data (a question's rollouts across steps)
  S.example = null;
  fetchExample(S.run, ep.example_id).then(ex => { S.example = ex; renderDeck(); }).catch(() => {});
}

function renderEpHead(row, ep) {
  const h = $('#ep-head');
  const cat = ep?.mar?.categorical || {};
  const winner = ep?.mar?.categorical?.winner ?? row.winner;
  const kv = (k, v, cls = '') => el('div', { class: 'kv' }, el('span', { class: 'k' }, k), el('span', { class: `v ${cls}` }, v));
  const meta = [
    kv('regime', ep?.kind ?? row.kind),
    kv('reward', fmtNum(ep?.reward ?? row.reward, 3), rewardClass(ep?.reward ?? row.reward)),
    winner ? kv('winner', winner, 'win') : null,
    cat['final_answer/debater_a'] ? kv('a→final', cat['final_answer/debater_a']) : null,
    cat['first_answer/debater_a'] && cat['final_answer/debater_a'] && cat['first_answer/debater_a'] !== cat['final_answer/debater_a']
      ? kv('a flip', `${cat['first_answer/debater_a']}→${cat['final_answer/debater_a']}`, 'neg') : null,
    ep ? kv('turns', String(ep.steps.length)) : null,
    ep?.trajectory_id ? kv('traj', String(ep.trajectory_id).slice(0, 10)) : null,
  ].filter(Boolean);

  h.replaceChildren(
    el('button', { class: 'toggle-list', title: 'collapse list', onclick: toggleList }, '⟨'),
    el('div', { class: 'ep-id' },
      el('span', { class: 'big' }, `#${row.example_id}`),
      el('span', { class: 'small' }, `step ${row.step} · line ${row.transcript_line}`)),
    el('div', { class: 'ep-meta' }, ...meta),
  );
}
function toggleList() {
  $('#body').classList.toggle('list-collapsed');
  const t = $('.toggle-list'); if (t) t.textContent = $('#body').classList.contains('list-collapsed') ? '⟩' : '⟨';
}

// ── setup / protocol spec-sheet (the DECLARED config + recovered schedule) ──────
// Renders the run's configured protocol so the user reads it instead of inferring
// it from the timeline. Toggled from the masthead; rendered into the inspector's
// primary scroll context. A missing field shows '—' (config didn't declare it) —
// distinct from a 0 (the judge seat's local token capture), which is a real value.
const SEAT_PHASE_HUES = { propose: 'var(--ok)', critique: 'var(--warn)', final: 'var(--extra)' };
function setupVal(v) { return v == null ? '—' : String(v); }

function renderSetupPanel(setup) {
  const root = el('div', { class: 'detail setup-sheet' });
  root.append(el('div', { class: 'turn-banner setup-banner' },
    el('span', { class: 'seat-badge' }, 'run setup'),
    el('span', { class: 'b-phase' }, 'declared protocol'),
    el('span', { class: 'b-addr' }, S.run),
  ));

  if (!setup) {
    root.append(el('div', { class: 'delta-region empty' },
      'no declared setup for this run — it was synced without --configs-dir, so there is no orchestrator.toml to distill into a spec sheet.'));
    return root;
  }

  const sch = setup.schedule || { slots: [] };
  const samp = setup.sampling || {};
  const model = setup.model || {};
  const lora = model.lora;

  // a labelled spec row of cells
  const spec = (rows) => {
    const wrap = el('div', { class: 'spec-grid' });
    for (const [k, v, cls] of rows) {
      wrap.append(el('div', { class: 'spec-cell' },
        el('span', { class: 'sk' }, k),
        el('span', { class: `sv ${cls || ''}` }, v)));
    }
    return wrap;
  };

  // ── ENV + MODEL ──
  root.append(el('div', { class: 'sec' },
    el('div', { class: 'sec-h' }, 'environment & model', el('span', { class: 'tag' }, 'student')),
    el('div', { class: 'sec-sub' }, 'the task env and the policy being trained')));
  root.append(spec([
    ['env id', setupVal(setup.env_id), 'mono'],
    ['env name', setupVal(setup.env_name), 'mono'],
    ['subset', setupVal(setup.subset), 'mono'],
    ['truth member', setupVal(setup.truth_member), 'seat'],
    ['model', setupVal(model.name), 'mono'],
    ['lora', lora ? `${setupVal(lora.name)} · r${setupVal(lora.rank)} / α${setupVal(lora.alpha)}` : 'none', 'mono'],
  ]));

  // ── SCHEDULE: the explicit declared turn-order chain ──
  root.append(el('div', { class: 'sec' },
    el('div', { class: 'sec-h' }, 'schedule', el('span', { class: 'tag' }, `recovered from rollouts · ${sch.regime}`)),
    el('div', { class: 'sec-sub' }, 'the protocol turn-order (actor·phase, left to right), recovered from the run’s observed turn structure — exact for a fixed-protocol env; not yet read from the env declaration')));
  const chain = el('div', { class: 'sched-chain' });
  sch.slots.forEach((s, i) => {
    if (i > 0) chain.append(el('span', { class: 'sched-arr' }, '→'));
    const hue = seatHue(s.actor, sch.slots.map(x => x.actor));
    chain.append(el('span', { class: 'sched-node' },
      el('span', { class: 'sched-i' }, `T${s.index}`),
      el('span', { class: 'sched-actor' }, el('span', { class: 'seat-dot', style: `background:${hue}` }), s.actor),
      s.phase ? el('span', { class: 'sched-phase', style: `--ph:${SEAT_PHASE_HUES[s.phase] || 'var(--ink-3)'}` }, s.phase) : null));
  });
  root.append(chain);

  // ── SAMPLING ──
  root.append(el('div', { class: 'sec' },
    el('div', { class: 'sec-h' }, 'sampling', el('span', { class: 'tag' }, 'decode knobs')),
    el('div', { class: 'sec-sub' }, 'how the student generates each turn')));
  root.append(spec([
    ['temperature', setupVal(samp.temperature)],
    ['top_p', setupVal(samp.top_p)],
    ['rep. penalty', setupVal(samp.repetition_penalty)],
    ['max completion', samp.max_completion_tokens != null ? `${samp.max_completion_tokens.toLocaleString()} tok` : '—'],
    ['thinking budget', samp.thinking_token_budget != null ? `${samp.thinking_token_budget.toLocaleString()} tok` : '—'],
  ]));

  // ── TRAINING REGIME ──
  root.append(el('div', { class: 'sec' },
    el('div', { class: 'sec-h' }, 'training regime', el('span', { class: 'tag' }, 'on/off-policy · sizes')),
    el('div', { class: 'sec-sub' }, 'group size, off-policy staleness bound, and sequence length')));
  root.append(spec([
    ['group size', setupVal(setup.group_size)],
    ['off-policy steps', setupVal(setup.max_off_policy_steps)],
    ['max steps', setupVal(setup.max_steps)],
    ['batch size', setupVal(setup.batch_size)],
    ['seq len', setup.seq_len != null ? `${setup.seq_len.toLocaleString()} tok` : '—'],
  ]));

  return root;
}

async function showSetup() {
  S.setupOpen = true;
  $('#btn-setup').classList.add('on');
  $('#ep-head').hidden = true;
  $('#deck').hidden = true;
  const detail = $('#detail-scroll');
  if (S.setup === undefined || S.setup === null || S.setup._run !== S.run) {
    detail.replaceChildren(el('div', { class: 'empty skel' }, el('span', {}, 'reading run setup…')));
    let setup = null;
    try { setup = await fetchSetup(S.run); } catch (e) { /* surface below */ }
    if (setup) setup._run = S.run;
    S.setup = setup ?? { _run: S.run, _absent: true };
  }
  const payload = S.setup._absent ? null : S.setup;
  detail.replaceChildren(renderSetupPanel(payload));
  detail.scrollTop = 0;
}

function hideSetup() {
  S.setupOpen = false;
  $('#btn-setup').classList.remove('on');
  const detail = $('#detail-scroll');
  if (S.episode) {
    // restore the drilled episode view
    $('#ep-head').hidden = false; $('#deck').hidden = false;
    renderTurn();
  } else {
    detail.replaceChildren(el('div', { class: 'empty' },
      el('span', { class: 'big' }, 'select an episode'),
      el('span', {}, 'pick a frame from the episode list to inspect its turns')));
  }
}

function toggleSetup() { S.setupOpen ? hideSetup() : showSetup(); }

// ── scrubber deck: TURN frames + TRAINING-STEP ticks ────────────────────────
function detectFlip(ep, idx) {
  // an answer flip marker: this actor's final answer differs from its first (debate)
  const cat = ep?.mar?.categorical || {};
  const actor = stepActor(ep.steps[idx]);
  const first = cat[`first_answer/${actor}`], final = cat[`final_answer/${actor}`];
  // mark on the actor's LAST step in the episode
  const lastForActor = [...ep.steps].reverse().find(s => stepActor(s) === actor);
  return first && final && first !== final && lastForActor && lastForActor.index === idx;
}

function renderDeck() {
  const deck = $('#deck');
  const ep = S.episode;
  if (!ep) { deck.replaceChildren(); return; }
  S.scheduleLen = ep.alignment.length;
  const seatOrder = ep.members.length ? ep.members : ['agent'];

  // — TURN axis —
  const frames = el('div', { class: 'frames' });
  ep.alignment.forEach((al, i) => {
    const step = ep.steps[i] ?? null;
    const hue = seatHue(al.actor, seatOrder);
    const marks = el('div', { class: 'fr-mark' });
    if (step) {
      const vd = visibilityDelta(ep.steps, i);
      if (vd.fresh.length && vd.hadPrior) marks.append(el('span', { class: 'm new', title: `${vd.fresh.length} new message(s) this turn` }, `+${vd.fresh.length}`));
      if (detectFlip(ep, i)) marks.append(el('span', { class: 'm flip', title: 'answer flip' }, '⇄'));
      const d = step.diagnostics;
      if (d && d.status === 'present' && d.mismatch_kl?.mean != null && d.mismatch_kl.mean > 1.0)
        marks.append(el('span', { class: 'm kl', title: 'KL spike' }, 'KL↑'));
    }
    frames.append(el('div', {
      class: `frame ${al.status} ${S.turn === i ? 'sel' : ''}`,
      onclick: () => { S.turn = i; renderDeck(); renderTurn(); },
      title: `turn ${i} · ${al.status}`,
    },
      el('div', { class: 'fr-addr' }, el('span', { class: 'st' }, `T${i}`), el('span', {}, al.status === 'matched' ? 'ok' : al.status.slice(0, 4))),
      el('div', { class: 'fr-actor' }, el('span', { class: 'seat-dot', style: `background:${hue}` }), al.actor),
      el('div', { class: 'fr-phase' }, al.phase ?? '—'),
      marks,
    ));
  });
  const turnAxis = el('div', { class: 'axis' },
    el('div', { class: 'axis-label' }, el('span', { class: 'name' }, 'turn'), el('span', { class: 'pos' }, `T${S.turn}`)),
    el('div', { class: 'axis-track' }, frames),
  );

  // — TRAINING-STEP axis —
  const stepTrack = el('div', { class: 'axis-track' });
  if (S.example && S.example.by_step.length) {
    const ticks = el('div', { class: 'steps' });
    S.example.by_step.forEach(bs => {
      const here = bs.step === S.curStep;
      ticks.append(el('div', {
        class: `steptick ${here ? 'sel current' : ''}`,
        onclick: () => jumpToStep(bs),
        title: `step ${bs.step} · ${bs.episodes.length} rollout(s)`,
      },
        el('span', { class: 'sn' }, `s${bs.step}`),
        el('span', { class: 'sc' }, `${bs.episodes.length}×`),
      ));
    });
    stepTrack.append(ticks);
  } else {
    stepTrack.append(el('span', { class: 'empty-axis' },
      S.example ? `example #${ep.example_id} appears in 1 step of this run` : 'loading step axis…'));
  }
  const stepAxis = el('div', { class: 'axis' },
    el('div', { class: 'axis-label' }, el('span', { class: 'name' }, 'train step'), el('span', { class: 'pos' }, `s${S.curStep ?? '·'}`)),
    stepTrack,
  );

  deck.replaceChildren(turnAxis, stepAxis);
}

async function jumpToStep(bs) {
  // jump to the first rollout of the chosen step for this example
  const target = bs.episodes[0];
  S.curStep = bs.step;
  // try to find an existing list row matching this (step, line); else load directly
  const idx = S.rows.findIndex(r => r.step === target.step && r.transcript_line === target.transcript_line && String(r.example_id) === String(S.episode.example_id));
  if (idx >= 0) { selectEpisode(idx); return; }
  // not in current list view (different sort/limit) — load episode directly, keep deck
  const ep = await fetchEpisode(S.run, target.step, target.transcript_line, null);
  S.episode = ep; S.turn = 0; S.curStep = bs.step;
  renderEpHead({ example_id: ep.example_id, step: target.step, transcript_line: target.transcript_line, kind: ep.kind, reward: ep.reward, winner: target.winner, metrics: {} }, ep);
  renderDeck(); renderTurn();
}

// ── turn detail (the payload) ───────────────────────────────────────────────
function renderTurn() {
  const detail = $('#detail-scroll');
  const ep = S.episode;
  if (!ep) return;
  const al = ep.alignment[S.turn];
  const step = ep.steps[S.turn] ?? null;
  const seatOrder = ep.members.length ? ep.members : ['agent'];
  const hue = seatHue(al.actor, seatOrder);

  const root = el('div', { class: 'detail' });

  // banner — seat / phase / status, then per-turn token read, then the address
  root.append(el('div', { class: 'turn-banner' },
    el('span', { class: 'seat-badge' }, el('span', { class: 'seat-dot', style: `background:${hue}` }), al.actor),
    al.phase ? el('span', { class: 'b-phase' }, al.phase) : null,
    el('span', { class: `b-status ${al.status}` }, al.status),
    step ? el('span', { class: 'b-tok' },
      el('span', { class: 'lab' }, 'prompt'), el('span', { class: 'v' }, fmtTok(step.n_prompt_tokens)),
      el('span', { class: 'arr' }, '→'),
      el('span', { class: 'lab' }, 'completion'), el('span', { class: 'v' }, fmtTok(step.n_completion_tokens))) : null,
    el('span', { class: 'b-addr' }, `turn ${S.turn} / ${ep.alignment.length - 1}`),
  ));

  if (!step) {
    // a truncated slot: expected but no observed step
    root.append(el('div', { class: 'sec' },
      el('div', { class: 'sec-h' }, 'expected turn, not produced', el('span', { class: 'tag' }, 'truncated')),
      el('div', { class: 'sec-sub' }, `the run schedule expects ${al.actor}${al.phase ? ' · ' + al.phase : ''} here; this rollout ended before reaching it.`)));
    detail.replaceChildren(root);
    return;
  }

  // — FULL I/O (untruncated) — the prompt carries the whole context every turn;
  // contiguous runs of messages this seat ALREADY saw on an earlier turn are folded
  // away, leaving only what is NEW this turn expanded. The fold IS the delta — no
  // separate, duplicative "new this turn" view.
  const vd = visibilityDelta(ep.steps, S.turn);
  const subText = !vd.hadPrior
    ? `${al.actor}'s first turn — the whole prompt is new context`
    : vd.fresh.length === 0
      ? 'no new context this turn — this seat saw exactly what it saw last turn (carried-over context folded below)'
      : 'full prompt + completion; context carried over from earlier turns is folded, new context stays expanded';
  root.append(el('div', { class: 'sec' },
    el('div', { class: 'sec-h' }, 'full i/o', el('span', { class: 'tag' }, 'untruncated')),
    el('div', { class: 'sec-sub' }, subText)));

  const promptBlock = el('details', { class: 'io-block', open: true },
    el('summary', {}, el('span', { class: 'caret' }, '▸'), 'prompt',
      el('span', { class: 'count' },
        el('span', { class: 'tok' }, fmtTok(step.n_prompt_tokens)),
        el('span', { class: 'msgn' }, `${step.prompt.length} msg`))),
    renderPromptBody(step.prompt, vd),
  );
  const complTok = {
    totalTok: step.n_completion_tokens,
    totalCh: step.completion.reduce((s, m) => s + msgChars(m), 0),
  };
  const complBlock = el('details', { class: 'io-block', open: true },
    el('summary', {}, el('span', { class: 'caret' }, '▸'), 'completion',
      el('span', { class: 'count' },
        el('span', { class: 'tok' }, fmtTok(step.n_completion_tokens)),
        el('span', { class: 'msgn' }, `${step.completion.length} msg`))),
    el('div', { class: 'io-body' }, ...step.completion.map(m => renderMsg(m, false, complTok))),
  );
  root.append(promptBlock, complBlock);

  // — DIAGNOSTICS line —
  root.append(el('div', { class: 'sec' },
    el('div', { class: 'sec-h' }, 'diagnostics', el('span', { class: 'tag' }, 'per turn · KL / IS / H')),
    el('div', { class: 'sec-sub' }, 'training signal joined per (trajectory, seat, step) — shown absent/masked honestly until the trainer sidecar lands')));
  root.append(renderDiag(step.diagnostics));

  detail.replaceChildren(root);
  detail.scrollTop = 0;
  // typeset math now that the detail subtree is in the document
  typesetMath(detail);
}

// a message carries reasoning out-of-band ⇒ its visible content has no real think
// spans (any <think> there is a quote). Drives extractThink at the render sites.
const hasReasoning = m => !!(m && m.extra && typeof m.extra.reasoning_content === 'string' && m.extra.reasoning_content);
// char count of one message = visible content + any separate reasoning_content.
function msgChars(m) {
  const c = typeof m.content === 'string' ? m.content : (m.content == null ? '' : JSON.stringify(m.content));
  const r = m.extra && typeof m.extra.reasoning_content === 'string' ? m.extra.reasoning_content : '';
  return c.length + r.length;
}
// reasoning carries NO token-id array of its own — it is folded into the single
// completion stream (completion_ids = reasoning + visible content). Estimate its
// token count by char-share of the EXACT completion total: both segments are the
// same model's text in one stream, so they tokenize at near-identical density
// (≈0.1% drift here). Marked ≈ to stay honest vs the exact prompt/completion
// counts. An exact figure would need a dump-time reasoning_tokens field upstream.
function reasoningTok(reasoningCh, tokCtx) {
  if (!tokCtx || tokCtx.totalTok == null || !tokCtx.totalCh) return null;
  return Math.round(tokCtx.totalTok * reasoningCh / tokCtx.totalCh);
}

function renderMsg(m, isNew, tokCtx) {
  const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
  const chars = c.length;
  const src = injectionSource(m);
  // Thinking models (Qwen3.5 etc.) emit reasoning in a SEPARATE reasoning_content
  // field, NOT inline <think>. It is ~45% of the output and must be shown, not
  // dropped. Render it as a dimmed, collapsible block (open by default) before
  // the visible content.
  const reasoning = m.extra && typeof m.extra.reasoning_content === 'string'
    ? m.extra.reasoning_content : '';
  // Prefer the EXACT count the sync computed by re-encoding reasoning_content with
  // the run tokenizer; fall back to the char-share estimate (marked ≈) only when a
  // store was synced without a tokenizer.
  const exactRtok = typeof m.reasoning_tokens === 'number' ? m.reasoning_tokens : null;
  const rtok = exactRtok != null ? exactRtok : (reasoning ? reasoningTok(reasoning.length, tokCtx) : null);
  const approx = exactRtok == null;
  const rfig = reasoning
    ? (rtok != null ? `${approx ? '≈' : ''}${rtok.toLocaleString()} tok · ${reasoning.length.toLocaleString()} ch`
                    : `${reasoning.length.toLocaleString()} ch`)
    : '';
  const rtitle = approx
    ? 'estimate: char-share of the exact completion token total (reasoning is folded into completion_ids with no array of its own)'
    : 'exact: reasoning_content re-encoded with the run tokenizer';
  return el('div', { class: `msg role-${m.role} ${isNew ? 'is-new' : ''}` },
    el('div', { class: 'm-head' },
      el('span', { class: `role r-${m.role}` }, m.role),
      isNew ? el('span', { class: 'inj' }, '· new') : null,
      src ? el('span', { class: 'inj' }, `· [${src}]`) : null,
      reasoning ? el('span', { class: 'inj', title: rtitle }, `· reasoning ${rfig}`) : null,
      (m.extra && m.extra.tool_calls) ? el('span', {}, '· tool_call') : null,
      el('span', { class: 'chars tn' }, `${chars.toLocaleString()} ch`)),
    reasoning ? el('details', { class: 'reasoning-block', open: true },
      el('summary', {}, el('span', { class: 'caret' }, '▸'),
        el('span', { class: 'rlabel' }, 'reasoning'),
        el('span', { class: 'count', title: rtitle }, rfig)),
      el('div', { class: 'think' }, renderReasoning(reasoning))) : null,
    el('div', { class: 'm-body' }, renderContent(m.content, !reasoning)),
    (m.extra && m.extra.tool_calls) ? el('div', { class: 'm-body mono', style: 'color:var(--ink-3)' }, JSON.stringify(m.extra.tool_calls, null, 2)) : null,
  );
}

// Render the prompt as full I/O, folding contiguous runs of messages this seat
// already saw on a previous turn into a quiet collapsible — only what is NEW this
// turn stays expanded (with the is-new accent). On the first turn (no prior) every
// message is "new", so nothing folds and the whole prompt shows.
function renderPromptBody(msgs, vd) {
  const body = el('div', { class: 'io-body' });
  let i = 0;
  while (i < msgs.length) {
    const fresh = vd.freshKeys.has(msgKey(msgs[i]));
    let j = i + 1;
    while (j < msgs.length && vd.freshKeys.has(msgKey(msgs[j])) === fresh) j++;
    const run = msgs.slice(i, j);
    if (fresh) {
      run.forEach(m => body.append(renderMsg(m, true)));
    } else {
      body.append(el('details', { class: 'io-seen' },
        el('summary', {}, el('span', { class: 'caret' }, '▸'),
          el('span', { class: 'seen-lab' }, 'seen earlier'),
          el('span', { class: 'seen-n' }, `${run.length} msg`)),
        ...run.map(m => renderMsg(m, false))));
    }
    i = j;
  }
  return body;
}

function renderDiag(d) {
  if (!d || d.status === 'absent') {
    return el('div', { class: 'diag status-absent' },
      el('span', { class: 'statebadge' }, 'absent'),
      el('div', { class: 'dcell' }, el('span', { class: 'dk' }, 'note'),
        el('span', { class: 'dv muted' }, 'no diagnostics sidecar for this run')));
  }
  if (d.status === 'masked_out') {
    return el('div', { class: 'diag status-masked_out' },
      el('span', { class: 'statebadge' }, 'masked_out'),
      el('div', { class: 'dcell' }, el('span', { class: 'dk' }, 'note'),
        el('span', { class: 'dv muted' }, 'fully clipped by mask_ratio — contributed no gradient')));
  }
  const cell = (k, vs) => el('div', { class: 'dcell' },
    el('span', { class: 'dk' }, k),
    el('span', { class: `dv ${vs?.mean == null ? 'muted' : ''}` }, vs?.mean == null ? 'absent' : fmtNum(vs.mean, 4)));
  return el('div', { class: 'diag status-present' },
    el('span', { class: 'statebadge' }, 'present'),
    cell('IS ratio', d.importance_ratio),
    cell('mismatch KL', d.mismatch_kl),
    cell('entropy', d.entropy),
    el('div', { class: 'dcell' }, el('span', { class: 'dk' }, 'n tokens'),
      el('span', { class: 'dv tn' }, d.n_tokens != null ? d.n_tokens.toLocaleString() : '·')),
  );
}

// ── run / sort wiring ───────────────────────────────────────────────────────
async function loadEpisodes() {
  if (!S.run) return;
  $('#list-scroll').replaceChildren(el('div', { class: 'empty skel' }, el('span', {}, 'querying index…')));
  try {
    S.rows = await fetchEpisodesList(S.run, S.sort, S.order);
  } catch (e) {
    $('#list-scroll').replaceChildren(el('div', { class: 'empty err' }, String(e)));
    return;
  }
  S.sel = null; S.episode = null;
  // prefetch schedule length so the sparkbar has a width
  fetchSchedule(S.run).then(s => { S.scheduleLen = s.slots.length; renderList(); }).catch(() => {});
  renderList();
  // if the setup sheet is open, re-render it for the (possibly new) run
  if (S.setupOpen) showSetup();
}

async function loadRuns() {
  S.runs = await fetchRuns();
  const sel = $('#sel-run');
  sel.replaceChildren(...S.runs.map(r =>
    el('option', { value: r.run_id }, `${r.run_id}${r.fixture ? '  ·fixture' : ''} (${r.steps.length} st)`)));
  if (!S.run && S.runs.length) S.run = S.runs[0].run_id;
  if (S.run) sel.value = S.run;
  await loadEpisodes();
}

$('#sel-run').addEventListener('change', e => { S.run = e.target.value; loadEpisodes(); });
$('#sel-sort').addEventListener('change', e => { S.sort = e.target.value; loadEpisodes(); });
$('#sel-dir').addEventListener('change', e => { S.order = e.target.value; loadEpisodes(); });
$('#btn-refresh').addEventListener('click', () => loadRuns());
$('#btn-setup').addEventListener('click', () => toggleSetup());

// keyboard: ←/→ scrub turns, [ / ] scrub steps
window.addEventListener('keydown', e => {
  if (!S.episode) return;
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') { S.turn = Math.min(S.episode.alignment.length - 1, S.turn + 1); renderDeck(); renderTurn(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { S.turn = Math.max(0, S.turn - 1); renderDeck(); renderTurn(); e.preventDefault(); }
  else if ((e.key === ']' || e.key === '[') && S.example) {
    const steps = S.example.by_step; const cur = steps.findIndex(b => b.step === S.curStep);
    const nx = e.key === ']' ? Math.min(steps.length - 1, cur + 1) : Math.max(0, cur - 1);
    if (nx !== cur && nx >= 0) jumpToStep(steps[nx]);
    e.preventDefault();
  }
});

// ── live feed (SSE; fixtures don't stream) ──────────────────────────────────
function wireLive() {
  const live = $('#live'); const txt = $('.txt', live);
  try {
    const es = new EventSource('/api/stream');
    es.onopen = () => { live.className = 'live on'; txt.textContent = 'live'; };
    es.onmessage = () => { if (S.run && !isFixtureRun(S.run)) loadEpisodes(); };
    es.onerror = () => { live.className = 'live off'; txt.textContent = 'offline'; };
  } catch { live.className = 'live off'; txt.textContent = 'n/a'; }
}

// reflect the default sort (step · asc) in the controls before first load
$('#sel-sort').value = S.sort; $('#sel-dir').value = S.order;
loadRuns();
wireLive();
