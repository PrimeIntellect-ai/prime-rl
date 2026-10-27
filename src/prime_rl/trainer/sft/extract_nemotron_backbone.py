#!/usr/bin/env python3
"""Extract the text-only `nemotron_h` LLM backbone from the Nemotron 3.5 Super VL
*omni* checkpoint, producing a standalone checkpoint prime-rl can train.

The omni checkpoint (model_type `nemotron_h_omni`) stores its weights under:
  language_model.*      -> the Nemotron-H LLM backbone  (KEEP, strip the prefix)
    language_model.backbone.*   -> backbone.*
    language_model.lm_head.*    -> lm_head.*
    language_model.mtp.*        -> mtp.*   (kept; prime-rl drops it at conversion,
                                            vLLM may use it for speculative decoding)
  vision_model.* / mlp1.* / vision_projector.*  -> C-RADIO vision tower + connector (DROP)
  (sound tower, if present)                      -> (DROP)

The result's keys (`backbone.embeddings.weight`, `backbone.layers.N.mixer...`,
`lm_head.weight`, ...) are exactly the standalone HF NemotronH layout that
prime_rl/trainer/models/nemotron_h/converting_nemotron_h.py recognizes.

CPU-only, streams one shard at a time (never loads the full model into RAM).

Usage:
  python -I extract_nemotron_backbone.py <src_omni_dir> <dst_backbone_dir> [--shard-gib 5]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

KEEP_PREFIX = "language_model."
DROP_PREFIXES = ("vision_model.", "mlp1.", "vision_projector.", "sound_model.", "audio")


def new_key(old: str) -> str | None:
    """Return the standalone-backbone key for an omni key, or None to drop it."""
    if old.startswith(KEEP_PREFIX):
        return old[len(KEEP_PREFIX) :]
    if old.startswith(DROP_PREFIXES):
        return None
    raise ValueError(f"unknown tensor prefix: {old!r}")


def normalize_special_floats(value):
    """Convert GA's tagged Infinity/NaN representations to native Python floats."""
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            special = value["__float__"]
            if special == "Infinity":
                return float("inf")
            if special == "-Infinity":
                return float("-inf")
            if special == "NaN":
                return float("nan")
            raise ValueError(f"unsupported special float: {special!r}")
        return {k: normalize_special_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_special_floats(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_special_floats(v) for v in value)
    return value


def build_standalone_config(src: Path) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    llm = cfg.get("llm_config")
    if not isinstance(llm, dict) or llm.get("model_type") != "nemotron_h":
        raise SystemExit("src config.json has no nemotron_h llm_config — not the expected omni checkpoint")
    llm = normalize_special_floats(dict(llm))
    llm["architectures"] = ["NemotronHForCausalLM"]
    llm.setdefault("torch_dtype", cfg.get("dtype", "bfloat16"))
    return llm


def clean_tokenizer_config(src: Path, dst: Path) -> None:
    tc = json.loads((src / "tokenizer_config.json").read_text())
    # Strip the omni processor wiring so a plain text tokenizer loads via AutoTokenizer.
    for k in ("auto_map", "processor_class"):
        tc.pop(k, None)
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    if (src / "chat_template.jinja").exists():
        tc["chat_template"] = (src / "chat_template.jinja").read_text()
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
    for fn in ("tokenizer.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if (src / fn).exists():
            shutil.copy2(src / fn, dst / fn)


def prepare_destination(src: Path, dst: Path) -> tuple[Path, Path]:
    """Validate paths and return a same-filesystem staging directory."""
    if src.is_symlink() or dst.is_symlink():
        raise SystemExit("source and destination must not be symlinks")
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        raise SystemExit("source and destination must be different directories")
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise SystemExit("destination must be a new or empty directory")
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp-", dir=dst.parent))
    return src, staging


def source_shard_path(src: Path, shard: str) -> Path:
    """Keep index-selected shard paths inside the source directory."""
    candidate = src / shard
    if candidate.is_symlink():
        raise ValueError(f"shard must not be a symlink: {shard!r}")
    candidate = candidate.resolve()
    if not candidate.is_relative_to(src):
        raise ValueError(f"shard escapes source directory: {shard!r}")
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="source omni checkpoint dir")
    ap.add_argument("dst", type=Path, help="output backbone dir (created)")
    ap.add_argument("--shard-gib", type=float, default=5.0, help="max output shard size in GiB")
    args = ap.parse_args()
    if not math.isfinite(args.shard_gib) or args.shard_gib <= 0:
        raise SystemExit("--shard-gib must be finite and greater than zero")

    src, staging = prepare_destination(args.src, args.dst)
    dst = args.dst.resolve()
    try:
        index = json.loads((src / "model.safetensors.index.json").read_text())
        weight_map: dict[str, str] = index["weight_map"]

        # Group kept tensors by their source shard so each shard is opened once.
        kept: dict[str, list[tuple[str, str]]] = {}  # shard_file -> [(old_key, new_key)]
        output_keys: set[str] = set()
        dropped = 0
        for old, shard in weight_map.items():
            nk = new_key(old)
            if nk is None:
                dropped += 1
                continue
            if nk in output_keys:
                raise ValueError(f"duplicate output key: {nk}")
            output_keys.add(nk)
            kept.setdefault(shard, []).append((old, nk))
        n_keep = sum(len(v) for v in kept.values())
        print(f"keeping {n_keep} tensors, dropping {dropped}")
        if n_keep == 0:
            raise SystemExit("nothing kept — check prefixes against the source index")

        source_config = build_standalone_config(src)
        required = {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}
        missing = sorted(required - output_keys)
        if missing:
            raise SystemExit(f"missing required backbone tensors: {missing}")
        layer_types = source_config.get("layers_block_type")
        if not isinstance(layer_types, list) or not layer_types:
            raise SystemExit("config has no non-empty layers_block_type")
        layer_indices = {
            int(match.group(1)) for key in output_keys if (match := re.match(r"backbone\.layers\.(\d+)\.", key))
        }
        expected_layers = set(range(len(layer_types)))
        if layer_indices != expected_layers:
            raise SystemExit(f"incomplete backbone layers: expected {len(layer_types)}, got {len(layer_indices)}")

        cap = int(args.shard_gib * (1024**3))
        out_shards: list[tuple[str, dict, int]] = []  # (filename, keys, tensor_bytes)
        cur: dict = {}
        cur_bytes = 0
        shard_idx = 1

        def flush():
            nonlocal cur, cur_bytes, shard_idx
            if not cur:
                return
            fn = f"model-{shard_idx:05d}.safetensors"
            save_file(cur, str(staging / fn), metadata={"format": "pt"})
            out_shards.append((fn, {k: None for k in cur}, cur_bytes))
            print(f"  wrote {fn}  ({len(cur)} tensors, {cur_bytes / 1024**3:.2f} GiB)")
            cur, cur_bytes, shard_idx = {}, 0, shard_idx + 1

        for shard in sorted(kept):
            with safe_open(str(source_shard_path(src, shard)), framework="pt") as f:
                for old, nk in kept[shard]:
                    t = f.get_tensor(old)
                    nbytes = t.numel() * t.element_size()
                    if cur and cur_bytes + nbytes > cap:
                        flush()
                    cur[nk] = t
                    cur_bytes += nbytes
        flush()

        # Build the HF index (weight_map + tensor-byte total_size).
        total = sum(shard_bytes for _, _, shard_bytes in out_shards)
        weight_index: dict[str, str] = {}
        for fn, keys, _ in out_shards:
            for k in keys:
                weight_index[k] = fn
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_index}, indent=2)
        )

        (staging / "config.json").write_text(json.dumps(source_config, indent=2))
        clean_tokenizer_config(src, staging)
        if dst.exists():
            dst.rmdir()
        staging.replace(dst)
        print(f"done -> {dst}  ({len(out_shards)} shards, {total / 1024**3:.1f} GiB)")
        print("next: validate it loads with prime-rl's trainer (see smoke-test notes).")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
