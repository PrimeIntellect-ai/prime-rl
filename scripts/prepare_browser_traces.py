"""Convert browser-agent traces (jsonl.zst) into the phase-2 SFT parquet format.

Input: datasets/browser_traces.jsonl.zst — one browser-automation trace per line:
  {tools: [OAI function specs], messages: [...], metadata: {...}}
with screenshots as data:image/jpeg;base64 URIs inside user/tool content parts.

Output (LOCAL ONLY — this dataset must never be pushed anywhere):
  datasets/nemotron_vl_sft_phase2/data/browser_use_sft_dataset/{train,validation}.parquet
  datasets/nemotron_vl_sft_phase2/media/browser/<sha1>.jpg   (deduped screenshots)

Schema per row: {id, tools (JSON string), messages: list[{role, content:
list[{type,text,image}], tool_calls: list[{id,type,function:{name,arguments:str}}]}]}
— content is always a uniform part list; tool-call arguments are JSON strings
(deserialized by prime-rl's `deserialize_tool_calls`).

Run from the prime-rl repo root:
    uv run --with zstandard python scripts/prepare_browser_traces.py
"""

import base64
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "datasets" / "browser_traces.jsonl.zst"
OUT = REPO / "datasets" / "nemotron_vl_sft_phase2"
MEDIA = OUT / "media" / "browser"
MEDIA_PREFIX = "datasets/nemotron_vl_sft_phase2/media/browser"

VAL_EVERY = 250  # every Nth trace -> validation (~147 of 36788)

MESSAGE_TYPE = pa.struct(
    [
        ("role", pa.string()),
        (
            "content",
            pa.list_(pa.struct([("type", pa.string()), ("text", pa.string()), ("image", pa.string())])),
        ),
        (
            "tool_calls",
            pa.list_(
                pa.struct(
                    [
                        ("id", pa.string()),
                        ("type", pa.string()),
                        (
                            "function",
                            pa.struct([("name", pa.string()), ("arguments", pa.string())]),
                        ),
                    ]
                )
            ),
        ),
    ]
)
SCHEMA = pa.schema([("id", pa.string()), ("tools", pa.string()), ("messages", pa.list_(MESSAGE_TYPE))])


def save_image(data_uri: str) -> str:
    payload = base64.b64decode(data_uri.split(",", 1)[1])
    digest = hashlib.sha1(payload).hexdigest()
    path = MEDIA / f"{digest}.jpg"
    if not path.exists():
        path.write_bytes(payload)
    return f"{MEDIA_PREFIX}/{digest}.jpg"


def convert_content(content) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content, "image": None}] if content else []
    parts = []
    for p in content:
        if isinstance(p, str):
            if p:
                parts.append({"type": "text", "text": p, "image": None})
        elif p.get("type") in ("image", "image_url"):
            url = p.get("image_url") if p.get("type") == "image_url" else p.get("image")
            url = url.get("url") if isinstance(url, dict) else url
            parts.append({"type": "image", "text": None, "image": save_image(url)})
        elif p.get("text") is not None:
            parts.append({"type": "text", "text": p["text"], "image": None})
    return parts


def convert_tool_calls(tool_calls) -> list[dict] | None:
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        out.append(
            {
                "id": tc.get("id") or "",
                "type": tc.get("type") or "function",
                "function": {
                    "name": fn.get("name") or "",
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                },
            }
        )
    return out


def main() -> None:
    import zstandard

    MEDIA.mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "browser_use_sft_dataset").mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    n = 0
    with open(SRC, "rb") as f:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        reader = io.TextIOWrapper(io.BufferedReader(dctx.stream_reader(f), buffer_size=1 << 24), encoding="utf-8")
        for line in reader:
            row = json.loads(line)
            messages = [
                {
                    "role": m["role"],
                    "content": convert_content(m.get("content")),
                    "tool_calls": convert_tool_calls(m.get("tool_calls")),
                }
                for m in row["messages"]
            ]
            rec = {
                "id": row.get("metadata", {}).get("entry_uuid") or f"trace-{n}",
                "tools": json.dumps(row.get("tools") or []),
                "messages": messages,
            }
            (val_rows if n % VAL_EVERY == VAL_EVERY - 1 else train_rows).append(rec)
            n += 1
            if n % 2000 == 0:
                print(f"{n} traces ...", flush=True)

    for name, rows in (("train", train_rows), ("validation", val_rows)):
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, OUT / "data" / "browser_use_sft_dataset" / f"{name}.parquet")
        print(f"{name}: {len(rows)} rows")
    n_media = sum(1 for _ in MEDIA.iterdir())
    print(f"media: {n_media} unique screenshots")


if __name__ == "__main__":
    main()
