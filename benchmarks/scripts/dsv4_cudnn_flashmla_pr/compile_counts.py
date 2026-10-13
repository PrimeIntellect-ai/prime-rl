"""Count the kernels the DSv4 sparse attention ops compile in one process, over a sequence of calls.

usage (on a GPU node, from the worktree under test):
  uv run --no-sync python benchmarks/scripts/dsv4_cudnn_flashmla_pr/compile_counts.py CASE [--backend B] [--out F]

CASE is one of:
  corpus      every item of the synthetic corpus (lengths, document mixes, layer types, CP slices; widths <= 640)
  wide        synthetic items at slot widths 128..8320
  two-clamps  the `wide` widths up to 640 under max_slots 8320, then again under max_slots 640

Each call runs the forward and the backward through `prime_rl::dsv4_sparse_attn`. The cuDNN count is read from
the frontend's executor caches (three on SM90, one on SM100), which works on every commit; on commits that log
compiles it is also counted from the `Compiled the cuDNN DSA backward` log lines. TileLang logs compiles but not
disk loads, so its cache is read as its file count before and after (set TILELANG_CACHE_DIR).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn
from prime_rl.utils.logger import get_logger

DEFAULT_CORPUS_DIR = Path("~/tmp/dsv4-sparse-attn-bench/corpus").expanduser()
HEADS = 64
DIM = 512
SM_SCALE = DIM**-0.5
# DeepSeek-V4-Flash's widest slot axis: sliding_window + max_position_embeddings // hca_rate = 128 + 1048576 // 128.
FLASH_MAX_SLOTS = 8320
WIDE_WIDTHS = (128, 256, 384, 640, 1024, 2048, 4096, 8320)
WIDE_QUERIES = 256
WIDE_POSITIONS = 16384


def corpus_calls(corpus_dir: Path) -> list[dict]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    calls = []
    for item in manifest["items"]:
        calls.append(
            {
                "id": item["id"],
                "indices_file": corpus_dir / item["file"],
                "n_queries": item["n_queries"],
                "n_positions": item["n_positions"],
                "seed": item["seed"],
                "max_slots": FLASH_MAX_SLOTS,
            }
        )
    return calls


def wide_calls(widths: tuple[int, ...], max_slots: int, tag: str) -> list[dict]:
    return [
        {
            "id": f"{tag}-width{width}",
            "width": width,
            "n_queries": WIDE_QUERIES,
            "n_positions": WIDE_POSITIONS,
            "seed": width,
            "max_slots": max_slots,
        }
        for width in widths
    ]


def wide_indices(width: int, n_queries: int, n_positions: int, generator: torch.Generator) -> torch.Tensor:
    """Distinct random positions per query, a random valid prefix per query, and an empty first query."""
    keys = torch.rand(n_queries, n_positions, generator=generator, device="cuda")
    indices = keys.topk(width, dim=-1).indices.to(torch.int32)
    lengths = torch.randint(1, width + 1, (n_queries, 1), generator=generator, device="cuda")
    slot_numbers = torch.arange(width, device="cuda").view(1, width)
    indices = torch.where(slot_numbers < lengths, indices, -1)
    indices[0] = -1
    return indices[None, :, None, :].contiguous()


def op_takes_max_slots() -> bool:
    return "max_slots" in str(torch.ops.prime_rl.dsv4_sparse_attn.default._schema)


def cudnn_cache_sizes() -> dict[str, int]:
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import flash_attn_bwd_sm100

    if torch.cuda.get_device_capability()[0] == 10:
        return {"main": len(flash_attn_bwd_sm100.compile_cache)}
    return {
        "preprocess": len(flash_attn_bwd_sm90.compile_cache_pre),
        "main": len(flash_attn_bwd_sm90.compile_cache),
        "postprocess": len(flash_attn_bwd_sm90.compile_cache_post),
    }


def tilelang_cache_files() -> int:
    cache_dir = Path(os.environ["TILELANG_CACHE_DIR"])
    return sum(1 for path in cache_dir.rglob("*") if path.is_file()) if cache_dir.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=["corpus", "wide", "two-clamps"])
    parser.add_argument("--backend", default="cudnn_flashmla")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out", type=Path, help="write the JSON summary here as well")
    args = parser.parse_args()

    compile_lines: list[str] = []
    get_logger().add(
        lambda message: compile_lines.append(message.record["message"]),
        filter=lambda record: record["message"].startswith("Compiled the cuDNN DSA backward"),
    )

    if args.case == "corpus":
        calls = corpus_calls(args.corpus_dir)
    elif args.case == "wide":
        calls = wide_calls(WIDE_WIDTHS, FLASH_MAX_SLOTS, "wide")
    else:
        narrow = tuple(width for width in WIDE_WIDTHS if width <= 640)
        calls = wide_calls(narrow, FLASH_MAX_SLOTS, "clamp16384") + wide_calls(narrow, 640, "clamp1024")

    takes_max_slots = op_takes_max_slots()
    tilelang_before = tilelang_cache_files()
    per_call = []
    for call in calls:
        generator = torch.Generator(device="cuda").manual_seed(call["seed"])
        if "indices_file" in call:
            indices = torch.load(call["indices_file"], weights_only=True).cuda()[None, :, None, :].contiguous()
        else:
            indices = wide_indices(call["width"], call["n_queries"], call["n_positions"], generator)
        q = torch.randn(1, call["n_queries"], HEADS, DIM, generator=generator, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(1, call["n_positions"], 1, DIM, generator=generator, device="cuda", dtype=torch.bfloat16)
        sinks = torch.randn(HEADS, generator=generator, device="cuda", dtype=torch.float32)
        grad_out = torch.randn_like(q)
        for leaf in (q, kv, sinks):
            leaf.requires_grad_(True)
        kwargs = {"backend": args.backend}
        if takes_max_slots:
            kwargs["max_slots"] = call["max_slots"]

        lines_before = len(compile_lines)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out, _ = dsv4_sparse_attn(q, kv, indices, sinks, SM_SCALE, **kwargs)
        torch.autograd.grad(out, (q, kv, sinks), grad_out)
        torch.cuda.synchronize()
        per_call.append(
            {
                "id": call["id"],
                "width": indices.shape[-1],
                "max_slots": call["max_slots"],
                "seconds": round(time.perf_counter() - start, 3),
                "cudnn_compile_lines": compile_lines[lines_before:],
                "cudnn_cache_sizes": cudnn_cache_sizes() if args.backend == "cudnn_flashmla" else None,
            }
        )
        print(json.dumps(per_call[-1]), flush=True)

    summary = {
        "case": args.case,
        "backend": args.backend,
        "op_takes_max_slots": takes_max_slots,
        "n_calls": len(calls),
        "distinct_widths": sorted({call["width"] for call in per_call}),
        "cudnn_cache_sizes": cudnn_cache_sizes() if args.backend == "cudnn_flashmla" else None,
        "cudnn_compile_lines": compile_lines,
        "calls_that_compiled_cudnn": [call["id"] for call in per_call if call["cudnn_compile_lines"]],
        "tilelang_cache_files_before": tilelang_before,
        "tilelang_cache_files_after": tilelang_cache_files(),
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "calls": per_call}, indent=2))


if __name__ == "__main__":
    main()
