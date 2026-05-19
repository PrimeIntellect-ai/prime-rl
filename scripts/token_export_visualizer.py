#!/usr/bin/env python3
import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a prime-rl token export JSONL record as HTML.")
    parser.add_argument("jsonl", type=Path, help="Path to a token export JSONL file.")
    parser.add_argument("--output", "-o", type=Path, help="Path to write the HTML file.")
    parser.add_argument("--record-index", type=int, default=0, help="Record index to render after filters are applied.")
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
        help="Mismatch KL value mapped to the deepest red. Defaults to the max finite mismatch in the record.",
    )
    args = parser.parse_args()

    records = _load_records(args.jsonl)
    record = _select_record(records, args.record_index, step=args.step, rank=args.rank, env_name=args.env_name)
    if args.tokenizer:
        _decode_token_texts(record, args.tokenizer, trust_remote_code=args.trust_remote_code)
    _add_derived_token_fields(record)
    output = args.output or args.jsonl.with_suffix(".html")
    output.write_text(_render_html(record, max_mismatch=args.max_mismatch), encoding="utf-8")
    print(output)


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _select_record(
    records: list[dict[str, Any]],
    record_index: int,
    *,
    step: int | None,
    rank: int | None,
    env_name: str | None,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if (step is None or record.get("step") == step)
        and (rank is None or record.get("rank") == rank)
        and (env_name is None or record.get("env_name") == env_name)
    ]
    if not matches:
        raise ValueError("No token export records matched the requested filters.")
    if record_index < 0 or record_index >= len(matches):
        raise IndexError(f"record-index {record_index} is out of range for {len(matches)} matched records.")
    return matches[record_index]


def _decode_token_texts(record: dict[str, Any], tokenizer_name: str, trust_remote_code: bool) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
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


def _render_html(record: dict[str, Any], max_mismatch: float | None) -> str:
    tokens = record.get("tokens", [])
    scale = _mismatch_scale(tokens, max_mismatch)
    content = _render_segments(_chat_segments(tokens), scale)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>prime-rl token export</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      line-height: 1.5;
    }}
    body {{
      margin: 0;
      background: #f8fafc;
      color: #111827;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      min-height: 100vh;
    }}
    .conversation {{
      padding: 20px;
    }}
    .message {{
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 12px;
      padding: 14px 0;
      border-bottom: 1px solid #e5e7eb;
    }}
    .role {{
      align-self: start;
      border-radius: 4px;
      padding: 2px 6px;
      width: max-content;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: #ffffff;
      background: #4b5563;
    }}
    .role-system {{
      background: #52525b;
    }}
    .role-user {{
      background: #2563eb;
    }}
    .role-assistant {{
      background: #059669;
    }}
    .role-tool {{
      background: #7c3aed;
    }}
    .tokens {{
      font-size: 16px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .token {{
      border-radius: 3px;
      cursor: default;
      transition: outline-color 80ms ease, box-shadow 80ms ease;
    }}
    .token:hover {{
      outline: 1px solid #991b1b;
      box-shadow: 0 0 0 2px rgba(153, 27, 27, 0.16);
    }}
    aside {{
      border-left: 1px solid #d1d5db;
      background: #ffffff;
      padding: 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      box-sizing: border-box;
      overflow: auto;
    }}
    .label {{
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 8px;
    }}
    .details {{
      font-size: 12px;
    }}
    .details table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .details tr {{
      border-bottom: 1px solid #e5e7eb;
    }}
    .details th {{
      width: 42%;
      padding: 6px 8px 6px 0;
      text-align: left;
      vertical-align: top;
      color: #6b7280;
      font-weight: 600;
    }}
    .details td {{
      padding: 6px 0;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    .muted {{
      color: #6b7280;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      main {{
        display: block;
      }}
      .message {{
        display: block;
      }}
      .role {{
        margin-bottom: 8px;
      }}
      aside {{
        position: static;
        height: auto;
        border-left: 0;
        border-top: 1px solid #d1d5db;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="conversation">{content}</section>
    <aside>
      <div class="label">Hover a token</div>
      <div id="details" class="details muted">Token details will appear here.</div>
    </aside>
  </main>
  <script>
    const details = document.getElementById("details");
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
      "is_clipped",
      "log_importance_ratio",
      "inference_logprob",
      "trainer_logprob",
    ];
    const labels = {{
      sample_kl_trainer_to_inference: "sample KL train→infer",
      sample_kl_inference_to_trainer: "sample KL infer→train",
      log_importance_ratio: "log ratio",
      importance_ratio: "ratio",
      prob_delta: "prob Δ",
      inference_logprob: "infer logprob",
      trainer_logprob: "train logprob",
      inference_prob: "prob inference",
      trainer_prob: "prob training",
      mismatch_kl: "KL mismatch",
      is_masked: "is_mask",
      is_masked_low: "is_mask low",
      is_masked_high: "is_mask high",
    }};

    function formatValue(value) {{
      if (value === null || value === undefined) return "null";
      if (typeof value === "number") {{
        if (!Number.isFinite(value)) return String(value);
        if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {{
          return value.toExponential(6);
        }}
        return Number(value.toPrecision(7)).toString();
      }}
      return String(value);
    }}

    function addRow(tbody, key, value) {{
      const row = document.createElement("tr");
      const th = document.createElement("th");
      const td = document.createElement("td");
      th.textContent = labels[key] || key;
      td.textContent = formatValue(value);
      row.appendChild(th);
      row.appendChild(td);
      tbody.appendChild(row);
    }}

    function renderDetails(rawInfo) {{
      const token = JSON.parse(rawInfo);
      const table = document.createElement("table");
      const tbody = document.createElement("tbody");
      const seen = new Set();
      for (const key of fieldOrder) {{
        if (Object.prototype.hasOwnProperty.call(token, key)) {{
          addRow(tbody, key, token[key]);
          seen.add(key);
        }}
      }}
      for (const key of Object.keys(token).sort()) {{
        if (!seen.has(key)) {{
          addRow(tbody, key, token[key]);
        }}
      }}
      table.appendChild(tbody);
      details.classList.remove("muted");
      details.replaceChildren(table);
    }}

    document.querySelectorAll(".token").forEach((node) => {{
      node.addEventListener("mouseenter", () => {{
        renderDetails(node.dataset.info);
      }});
    }});
  </script>
</body>
</html>
"""


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


def _render_segments(segments: list[dict[str, Any]], scale: float) -> str:
    rendered = []
    for segment in segments:
        role = str(segment["role"])
        token_spans = "".join(_render_token(token, scale) for token in segment["tokens"])
        role_class = _role_class(role)
        rendered.append(
            f'<section class="message">'
            f'<div class="role {role_class}">{html.escape(role)}</div>'
            f'<div class="tokens">{token_spans}</div>'
            f"</section>"
        )
    return "".join(rendered)


def _role_class(role: str) -> str:
    normalized = role.lower().replace("_", "-")
    if normalized in {"system", "user", "assistant", "tool"}:
        return f"role-{normalized}"
    return ""


def _render_token(token: dict[str, Any], scale: float) -> str:
    text = token.get("text")
    if text is None or text == "":
        text = f"[token:{token.get('id')}]"
    mismatch = _finite_float(token.get("mismatch_kl"))
    if not token.get("loss_mask"):
        mismatch = None
    alpha = 0.0 if mismatch is None else min(max(mismatch / scale, 0.0), 1.0)
    background = f"rgba(220, 38, 38, {0.08 + 0.72 * alpha:.3f})" if alpha > 0 else "transparent"
    color = "#6b7280" if not token.get("loss_mask") else ("#111827" if alpha < 0.72 else "#ffffff")
    decoration = "border-bottom: 2px solid #7c2d12;" if token.get("is_masked") else ""
    info = html.escape(json.dumps(token, indent=2, sort_keys=True, ensure_ascii=False), quote=True)
    title = html.escape(_mouse_summary(token), quote=True)
    return (
        f'<span class="token" data-info="{info}" title="{title}" '
        f'style="background-color: {background}; color: {color}; {decoration}">{html.escape(text)}</span>'
    )


def _mouse_summary(token: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"is_mask={token.get('is_masked')}",
            f"prob inference={_summary_value(token.get('inference_prob'))}",
            f"prob training={_summary_value(token.get('trainer_prob'))}",
            f"kl={_summary_value(token.get('mismatch_kl'))}",
        ]
    )


def _summary_value(value: Any) -> str:
    value = _finite_float(value)
    if value is None:
        return "null"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.0001):
        return f"{value:.6e}"
    return f"{value:.6g}"


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
