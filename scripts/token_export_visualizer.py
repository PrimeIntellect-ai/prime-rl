#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Render prime-rl token export JSONL as HTML.")
    parser.add_argument("input", type=Path, help="Path to a token export JSONL file or directory of JSONL exports.")
    parser.add_argument("--output", "-o", type=Path, help="Path to write the HTML file.")
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Record index to render after filters are applied, or initial record in --all-records mode.",
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Embed every matched record in one navigable HTML page. Implied for directory inputs.",
    )
    parser.add_argument("--step", type=int, help="Only consider records from this trainer step.")
    parser.add_argument("--rank", type=int, help="Only consider records from this trainer rank.")
    parser.add_argument("--env-name", help="Only consider records for this env name.")
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer model name or local path used to decode token ids into text fragments.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading --tokenizer.",
    )
    parser.add_argument(
        "--max-mismatch",
        type=float,
        help="Mismatch KL value mapped to the deepest red. Defaults to the p95 finite trainable mismatch.",
    )
    args = parser.parse_args()

    records = _filter_records(_load_records(args.input), step=args.step, rank=args.rank, env_name=args.env_name)
    if not records:
        raise ValueError("No token export records matched the requested filters.")

    render_all = args.all_records or args.input.is_dir()
    if render_all:
        if args.record_index < 0 or args.record_index >= len(records):
            raise IndexError(f"record-index {args.record_index} is out of range for {len(records)} matched records.")
        selected_records = records
        initial_index = args.record_index
    else:
        selected_records = [_select_record(records, args.record_index)]
        initial_index = 0

    if args.tokenizer:
        tokenizer = _load_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
        for record in selected_records:
            _decode_token_texts(record, tokenizer)
    for record in selected_records:
        _add_derived_token_fields(record)

    output = args.output or _default_output_path(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_html(selected_records, max_mismatch=args.max_mismatch, initial_index=initial_index), encoding="utf-8"
    )
    print(output)


def _default_output_path(path: Path) -> Path:
    if path.is_dir():
        return path / "index.html"
    return path.with_suffix(".html")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted(file for file in path.rglob("*.jsonl") if file.is_file())
        if not files:
            raise FileNotFoundError(f"No JSONL files found under {path}")
        root = path
    else:
        files = [path]
        root = path.parent

    records = []
    for file in files:
        source_file = _relative_source(file, root)
        with file.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source_file"] = source_file
                record["_source_line"] = line_number
                record["_source_record_index"] = len(records)
                records.append(record)
    return records


def _relative_source(file: Path, root: Path) -> str:
    try:
        return str(file.relative_to(root))
    except ValueError:
        return str(file)


def _filter_records(
    records: list[dict[str, Any]],
    *,
    step: int | None,
    rank: int | None,
    env_name: str | None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if (step is None or record.get("step") == step)
        and (rank is None or record.get("rank") == rank)
        and (env_name is None or record.get("env_name") == env_name)
    ]


def _select_record(records: list[dict[str, Any]], record_index: int) -> dict[str, Any]:
    if record_index < 0 or record_index >= len(records):
        raise IndexError(f"record-index {record_index} is out of range for {len(records)} matched records.")
    return records[record_index]


def _load_tokenizer(tokenizer_name: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)


def _decode_token_texts(record: dict[str, Any], tokenizer: Any) -> None:
    tokens = record.get("tokens", [])
    token_ids = [int(token["id"]) for token in tokens]
    token_texts = tokenizer.batch_decode(
        [[token_id] for token_id in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    for token, text in zip(tokens, token_texts):
        token["text"] = text


def _add_derived_token_fields(record: dict[str, Any]) -> None:
    for token in record.get("tokens", []):
        log_ratio = _finite_float(token.get("log_importance_ratio"))
        if log_ratio is None:
            continue
        token.setdefault("sample_kl_trainer_to_inference", log_ratio)
        token.setdefault("sample_kl_inference_to_trainer", -log_ratio)


def _render_html(records: list[dict[str, Any]], max_mismatch: float | None, initial_index: int) -> str:
    scale = _mismatch_scale([token for record in records for token in record.get("tokens", [])], max_mismatch)
    prepared_records = _prepare_records(records)
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>prime-rl token export</title>
  <style>
    :root {
      color-scheme: light;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      line-height: 1.5;
    }
    body {
      margin: 0;
      background: #f8fafc;
      color: #111827;
    }
    main {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 420px;
      min-height: 100vh;
    }
    .records-nav {
      border-right: 1px solid #d1d5db;
      background: #ffffff;
      padding: 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      box-sizing: border-box;
      overflow: auto;
    }
    .record-search {
      box-sizing: border-box;
      width: 100%;
      margin: 10px 0 12px;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      padding: 8px 9px;
      font: inherit;
      font-size: 12px;
      color: #111827;
      background: #ffffff;
    }
    .record-list {
      display: grid;
      gap: 6px;
    }
    .record-button {
      display: block;
      width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      padding: 8px;
      text-align: left;
      font: inherit;
      font-size: 12px;
      color: #111827;
      background: #ffffff;
      cursor: pointer;
    }
    .record-button:hover {
      border-color: #9ca3af;
      background: #f9fafb;
    }
    .record-button.active {
      border-color: #2563eb;
      background: #eff6ff;
    }
    .record-title {
      font-weight: 700;
      margin-bottom: 2px;
    }
    .record-subtitle {
      color: #6b7280;
      overflow-wrap: anywhere;
    }
    .conversation {
      padding: 20px;
    }
    .message {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 12px;
      padding: 14px 0;
      border-bottom: 1px solid #e5e7eb;
    }
    .role {
      align-self: start;
      border-radius: 4px;
      padding: 2px 6px;
      width: max-content;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: #ffffff;
      background: #4b5563;
    }
    .role-system {
      background: #52525b;
    }
    .role-user {
      background: #2563eb;
    }
    .role-assistant {
      background: #059669;
    }
    .role-tool {
      background: #7c3aed;
    }
    .tokens {
      font-size: 16px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .token {
      border-radius: 3px;
      cursor: default;
      transition: outline-color 80ms ease, box-shadow 80ms ease;
    }
    .token:hover {
      outline: 1px solid #991b1b;
      box-shadow: 0 0 0 2px rgba(153, 27, 27, 0.16);
    }
    aside {
      border-left: 1px solid #d1d5db;
      background: #ffffff;
      padding: 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      box-sizing: border-box;
      overflow: auto;
    }
    .label {
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 8px;
    }
    .details {
      font-size: 12px;
    }
    .details table {
      width: 100%;
      border-collapse: collapse;
    }
    .details tr {
      border-bottom: 1px solid #e5e7eb;
    }
    .details th {
      width: 42%;
      padding: 6px 8px 6px 0;
      text-align: left;
      vertical-align: top;
      color: #6b7280;
      font-weight: 600;
    }
    .details td {
      padding: 6px 0;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    .muted {
      color: #6b7280;
    }
    .empty {
      padding: 24px 0;
      color: #6b7280;
      font-size: 13px;
    }
    @media (max-width: 1100px) {
      main {
        display: block;
      }
      .records-nav,
      aside {
        position: static;
        height: auto;
      }
      .records-nav {
        border-right: 0;
        border-bottom: 1px solid #d1d5db;
      }
      .message {
        display: block;
      }
      .role {
        margin-bottom: 8px;
      }
      aside {
        border-left: 0;
        border-top: 1px solid #d1d5db;
      }
    }
  </style>
</head>
<body>
  <main>
    <nav class="records-nav">
      <div class="label">Records</div>
      <div id="record-count" class="muted"></div>
      <input id="record-search" class="record-search" type="search" placeholder="Filter step, rank, env" autocomplete="off">
      <div id="record-list" class="record-list"></div>
    </nav>
    <section id="conversation" class="conversation"></section>
    <aside>
      <div class="label">Hover a token</div>
      <div id="details" class="details muted">Token details will appear here.</div>
    </aside>
  </main>
  <script id="records-data" type="application/json">__RECORDS_JSON__</script>
  <script>
    const records = JSON.parse(document.getElementById("records-data").textContent);
    const mismatchScale = __MISMATCH_SCALE__;
    let activeIndex = __INITIAL_RECORD_INDEX__;

    const conversation = document.getElementById("conversation");
    const details = document.getElementById("details");
    const recordCount = document.getElementById("record-count");
    const recordList = document.getElementById("record-list");
    const recordSearch = document.getElementById("record-search");
    const fieldOrder = [
      "mismatch_kl",
      "is_masked",
      "inference_prob",
      "trainer_prob",
      "prob_delta",
      "importance_ratio",
      "sample_kl_trainer_to_inference",
      "sample_kl_inference_to_trainer",
      "text",
      "id",
      "index",
      "position",
      "loss_mask",
      "reward",
      "advantage",
      "entropy",
      "is_masked_low",
      "is_masked_high",
      "log_importance_ratio",
      "inference_logprob",
      "trainer_logprob",
    ];
    const labels = {
      sample_kl_trainer_to_inference: "sample KL train->infer",
      sample_kl_inference_to_trainer: "sample KL infer->train",
      log_importance_ratio: "log ratio",
      importance_ratio: "ratio",
      prob_delta: "prob delta",
      inference_logprob: "infer logprob",
      trainer_logprob: "train logprob",
      inference_prob: "prob inference",
      trainer_prob: "prob training",
      mismatch_kl: "KL mismatch",
      is_masked: "is_mask",
      is_masked_low: "is_mask low",
      is_masked_high: "is_mask high",
    };

    function renderNav() {
      const query = recordSearch.value.trim().toLowerCase();
      recordList.replaceChildren();
      let shown = 0;
      records.forEach((record, index) => {
        const text = record.meta.search_text;
        if (query && !text.includes(query)) {
          return;
        }
        shown += 1;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "record-button" + (index === activeIndex ? " active" : "");
        button.addEventListener("click", () => renderConversation(index));

        const title = document.createElement("div");
        title.className = "record-title";
        title.textContent = record.meta.label;

        const subtitle = document.createElement("div");
        subtitle.className = "record-subtitle";
        subtitle.textContent = record.meta.subtitle;

        button.appendChild(title);
        button.appendChild(subtitle);
        recordList.appendChild(button);
      });
      recordCount.textContent = `${shown} of ${records.length}`;
    }

    function renderConversation(index) {
      activeIndex = index;
      const record = records[index];
      conversation.replaceChildren();
      details.className = "details muted";
      details.textContent = "Token details will appear here.";

      if (!record || record.segments.length === 0) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No tokens in this record.";
        conversation.appendChild(empty);
        renderNav();
        return;
      }

      for (const segment of record.segments) {
        if (!segment.tokens.length) {
          continue;
        }
        const section = document.createElement("section");
        section.className = "message";

        const role = document.createElement("div");
        role.className = "role " + roleClass(segment.role);
        role.textContent = segment.role;

        const tokens = document.createElement("div");
        tokens.className = "tokens";
        for (const token of segment.tokens) {
          tokens.appendChild(renderToken(token));
        }

        section.appendChild(role);
        section.appendChild(tokens);
        conversation.appendChild(section);
      }
      renderNav();
    }

    function renderToken(token) {
      const span = document.createElement("span");
      span.className = "token";
      span.textContent = token.text === null || token.text === undefined || token.text === "" ? `[token:${token.id}]` : token.text;
      const mismatch = finiteNumber(token.mismatch_kl);
      const useMismatch = Boolean(token.loss_mask) && mismatch !== null;
      const alpha = useMismatch ? Math.min(Math.max(mismatch / mismatchScale, 0), 1) : 0;
      span.style.backgroundColor = alpha > 0 ? `rgba(220, 38, 38, ${(0.08 + 0.72 * alpha).toFixed(3)})` : "transparent";
      span.style.color = !token.loss_mask ? "#6b7280" : (alpha < 0.72 ? "#111827" : "#ffffff");
      if (token.is_masked) {
        span.style.borderBottom = "2px solid #7c2d12";
      }
      span.title = mouseSummary(token);
      span.addEventListener("mouseenter", () => renderDetails(token));
      return span;
    }

    function roleClass(role) {
      const normalized = String(role).toLowerCase().replaceAll("_", "-");
      if (["system", "user", "assistant", "tool"].includes(normalized)) {
        return `role-${normalized}`;
      }
      return "";
    }

    function renderDetails(token) {
      const table = document.createElement("table");
      const tbody = document.createElement("tbody");
      const seen = new Set();
      for (const key of fieldOrder) {
        if (Object.prototype.hasOwnProperty.call(token, key)) {
          addRow(tbody, key, token[key]);
          seen.add(key);
        }
      }
      for (const key of Object.keys(token).sort()) {
        if (!seen.has(key)) {
          addRow(tbody, key, token[key]);
        }
      }
      table.appendChild(tbody);
      details.className = "details";
      details.replaceChildren(table);
    }

    function addRow(tbody, key, value) {
      const row = document.createElement("tr");
      const th = document.createElement("th");
      const td = document.createElement("td");
      th.textContent = labels[key] || key;
      td.textContent = formatValue(value);
      row.appendChild(th);
      row.appendChild(td);
      tbody.appendChild(row);
    }

    function formatValue(value) {
      if (value === null || value === undefined) return "null";
      if (typeof value === "number") {
        if (!Number.isFinite(value)) return String(value);
        if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {
          return value.toExponential(6);
        }
        return Number(value.toPrecision(7)).toString();
      }
      return String(value);
    }

    function mouseSummary(token) {
      return [
        `is_mask=${token.is_masked}`,
        `prob inference=${summaryValue(token.inference_prob)}`,
        `prob training=${summaryValue(token.trainer_prob)}`,
        `kl=${summaryValue(token.mismatch_kl)}`,
      ].join("\\n");
    }

    function summaryValue(value) {
      value = finiteNumber(value);
      if (value === null) return "null";
      if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {
        return value.toExponential(6);
      }
      return Number(value.toPrecision(6)).toString();
    }

    function finiteNumber(value) {
      if (value === null || value === undefined) return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    recordSearch.addEventListener("input", renderNav);
    renderConversation(Math.min(Math.max(activeIndex, 0), records.length - 1));
  </script>
</body>
</html>
"""
    return (
        document.replace("__RECORDS_JSON__", _json_script(prepared_records))
        .replace("__MISMATCH_SCALE__", json.dumps(scale))
        .replace("__INITIAL_RECORD_INDEX__", json.dumps(initial_index))
    )


def _prepare_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "meta": _record_meta(record, ordinal),
            "segments": _chat_segments(record.get("tokens", [])),
        }
        for ordinal, record in enumerate(records)
    ]


def _record_meta(record: dict[str, Any], ordinal: int) -> dict[str, Any]:
    tokens = record.get("tokens", [])
    token_count = len(tokens)
    trainable_count = sum(1 for token in tokens if token.get("loss_mask"))
    label_parts = [f"#{ordinal}"]
    for key, label in (
        ("step", "step"),
        ("rank", "rank"),
        ("env_name", "env"),
        ("export_sequence_idx", "seq"),
    ):
        value = record.get(key)
        if value is not None:
            label_parts.append(f"{label} {value}")

    subtitle_parts = [
        f"{trainable_count}/{token_count} trainable",
        str(record.get("_source_file", "")),
        f"line {record.get('_source_line')}",
    ]
    meta = {
        "ordinal": ordinal,
        "label": " | ".join(label_parts),
        "subtitle": " | ".join(part for part in subtitle_parts if part),
        "step": record.get("step"),
        "rank": record.get("rank"),
        "micro_step": record.get("micro_step"),
        "micro_sequence_idx": record.get("micro_sequence_idx"),
        "export_sequence_idx": record.get("export_sequence_idx"),
        "env_name": record.get("env_name"),
        "source_file": record.get("_source_file"),
        "source_line": record.get("_source_line"),
        "token_count": token_count,
        "trainable_token_count": trainable_count,
    }
    meta["search_text"] = " ".join(str(value) for value in meta.values() if value is not None).lower()
    return meta


def _json_script(value: Any) -> str:
    payload = json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _chat_segments(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tokens or tokens[0].get("text") is None:
        return [{"role": "tokens", "tokens": tokens}]

    segments = []
    idx = 0
    pending = []
    while idx < len(tokens):
        if tokens[idx].get("text") != "<|im_start|>":
            pending.append(tokens[idx])
            idx += 1
            continue

        if pending:
            segments.append({"role": "tokens", "tokens": pending})
            pending = []

        role = "message"
        idx += 1
        if idx < len(tokens):
            role = str(tokens[idx].get("text", role)).strip() or role
            idx += 1
        if idx < len(tokens) and tokens[idx].get("text") == "\n":
            idx += 1

        message_tokens = []
        while idx < len(tokens) and tokens[idx].get("text") != "<|im_end|>":
            message_tokens.append(tokens[idx])
            idx += 1
        if idx < len(tokens) and tokens[idx].get("text") == "<|im_end|>":
            idx += 1
        if idx < len(tokens) and tokens[idx].get("text") == "\n":
            idx += 1
        segments.append({"role": role, "tokens": message_tokens})

    if pending:
        segments.append({"role": "tokens", "tokens": pending})

    return [segment for segment in segments if segment["tokens"]]


def _mismatch_scale(tokens: list[dict[str, Any]], override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--max-mismatch must be positive.")
        return override
    values = [_finite_float(token.get("mismatch_kl")) for token in tokens if token.get("loss_mask")]
    values = [value for value in values if value is not None and value > 0]
    if not values:
        return 1.0
    return max(_percentile(values, 0.95), 0.05)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(max(round((len(ordered) - 1) * q), 0), len(ordered) - 1)
    return ordered[idx]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
