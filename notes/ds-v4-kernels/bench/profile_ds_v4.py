#!/usr/bin/env python
"""Profile one DeepSeek V4 module at one sequence length, and write one JSON.

Throwaway measurement harness for the phase-0 kernel survey. One process measures exactly one
`(module, seq_len, ac, mode)` point, because the sweep is designed to hit OOM and an
`OutOfMemoryError` raised mid-backward leaves the allocator in a state where no later peak in the
same process means anything.

Run it through `uv run`, never bare python:

    uv run notes/ds-v4-kernels/bench/profile_ds_v4.py attn-csa 8192 --mode memory --out point.json

Modes:
  memory       one forward and one forward+backward, peak allocation only, no profiler
  timing       `triton.testing.do_bench` quantiles, no profiler, no memory recording
  attribution  a `TorchDispatchMode` allocation log plus a `torch.profiler` chrome trace

The three never run in one process: each instrument perturbs the others, and the profiler in
particular inflates wall time enough that its timings are worthless.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import get_args

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prime_rl.configs.trainer import ActivationCheckpointConfig, DSV4AttnImplementation  # noqa: E402
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper  # noqa: E402
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext  # noqa: E402
from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection  # noqa: E402
from prime_rl.trainer.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer  # noqa: E402
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding  # noqa: E402
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig  # noqa: E402
from prime_rl.utils.utils import default_dtype  # noqa: E402

REAL_CONFIG_JSON = Path(
    "/home/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots"
    "/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json"
)

# Keys in the checkpoint's `config.json` that `DeepseekV4Config` has no use for: quantization
# metadata, the MTP head, and the DSPARK speculative-decoding block.
_CONFIG_KEYS_TO_DROP = frozenset(
    {
        "architectures",
        "model_type",
        "transformers_version",
        "torch_dtype",
        "expert_dtype",
        "quantization_config",
        "num_nextn_predict_layers",
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    }
)


def _load_randomize():
    """`_randomize` out of the GPU test module, by path.

    `tests/unit/train/` carries no `__init__.py`, so the module is not importable by dotted name.
    It is the one helper in that file worth reusing: everything else there is hardwired to the toy
    `_prime_config()` and would silently profile a 100k-parameter model.
    """
    path = REPO_ROOT / "tests/unit/train/models/test_deepseek_v4.py"
    spec = importlib.util.spec_from_file_location("_dsv4_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._randomize


def build_config(**overrides) -> DeepseekV4Config:
    """The real V4-Flash config, from the cached checkpoint's own `config.json`.

    Bare `DeepseekV4Config()` defaults are *not* V4-Flash: their derived `layer_types` contains no
    sliding layer at all, while the checkpoint's `compress_ratios` puts sliding at indices 0 and 1.
    """
    import json as _json

    raw = _json.loads(REAL_CONFIG_JSON.read_text())
    kwargs = {k: v for k, v in raw.items() if k not in _CONFIG_KEYS_TO_DROP}
    kwargs.update(overrides)
    return DeepseekV4Config(**kwargs)


def layer_index_of(config: DeepseekV4Config, layer_type: str) -> int:
    return config.layer_types.index(layer_type)


def assert_real_config(config: DeepseekV4Config) -> None:
    """Refuse to measure anything unless the config is the real one.

    The single check that stops the whole sweep from silently profiling the toy test model.
    """
    assert config.head_dim == 512, config.head_dim
    assert config.num_attention_heads == 64, config.num_attention_heads
    assert config.sliding_window == 128, config.sliding_window
    assert config.hidden_size == 4096, config.hidden_size
    assert config.index_topk == 512, config.index_topk
    assert config.hc_mult == 4, config.hc_mult
    counts = {t: config.layer_types.count(t) for t in set(config.layer_types)}
    assert counts.get("sliding_attention") == 2, counts
    assert counts.get("compressed_sparse_attention") == 21, counts
    assert counts.get("heavily_compressed_attention") == 20, counts


def document_layout(seq_len: int, doc_len: int) -> list[int]:
    """Whole `doc_len` documents packed into `seq_len`, with the remainder as a final document."""
    if seq_len <= doc_len:
        return [seq_len]
    docs = [doc_len] * (seq_len // doc_len)
    remainder = seq_len % doc_len
    if remainder:
        docs.append(remainder)
    return docs


class AllocationLog(TorchDispatchMode):
    """Record every newly allocated CUDA storage above `threshold`, with the op that made it.

    Works identically in forward and backward, names the op *and* its output shape, and is not
    confused by the caching allocator: it keys on `data_ptr`, so a view of a storage already seen
    is not counted twice and a reused block is counted again only under its new op.
    """

    def __init__(self, threshold: int = 16 << 20):
        super().__init__()
        self.threshold = threshold
        self.records: list[dict] = []
        self._seen: set[int] = set()
        self.label = "forward"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in pytree.tree_leaves(out):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            ptr, nbytes = storage.data_ptr(), storage.nbytes()
            if nbytes < self.threshold or ptr in self._seen:
                continue
            self._seen.add(ptr)
            self.records.append(
                {
                    "op": str(func),
                    "phase": self.label,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "bytes": nbytes,
                }
            )
        return out

    def top(self, n: int = 25) -> list[dict]:
        return sorted(self.records, key=lambda r: -r["bytes"])[:n]

    def total_by_op(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for record in self.records:
            totals[record["op"]] = totals.get(record["op"], 0) + record["bytes"]
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


class Point:
    """One measurable module plus the closure that runs its forward.

    `output` is whatever the backward is taken through: a differentiable tensor, or `None` for the
    modules whose output carries no gradient (the indexer returns int64 indices, `PackedContext`
    returns a container of masks and tables).
    """

    def __init__(self, module, forward, differentiable: bool = True, note: str = ""):
        self.module = module
        self.forward = forward
        self.differentiable = differentiable
        self.note = note

    def parameters(self):
        return list(self.module.parameters()) if isinstance(self.module, torch.nn.Module) else []


def _packed(config, seq_lens, dtype, device):
    with torch.device(device), default_dtype(dtype):
        rotary = DeepseekV4RotaryEmbedding(config)
    return rotary, PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(seq_lens, device=device),
        dtype=dtype,
        device=device,
    )


def build_point(name: str, config, seq_lens, dtype, device, ac: str, randomize) -> Point:
    batch, seq_len = 1, sum(seq_lens)
    rotary, packed = _packed(config, seq_lens, dtype, device)
    hidden = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype, requires_grad=True)

    def attention(layer_type: str) -> Point:
        idx = layer_index_of(config, layer_type)
        with torch.device(device), default_dtype(dtype):
            module = DeepseekV4Attention(config, layer_idx=idx)
        randomize(module)
        return Point(module, lambda: module(hidden, packed=packed)[0], note=f"layer_idx={idx}")

    def compressor(layer_type: str) -> Point:
        idx = layer_index_of(config, layer_type)
        with torch.device(device), default_dtype(dtype):
            attn = DeepseekV4Attention(config, layer_idx=idx)
        randomize(attn)
        module = attn.compressor
        q_residual = attn.q_a_norm(attn.q_a_proj(hidden)).detach().requires_grad_(True)
        return Point(module, lambda: module(hidden, q_residual, packed)[0], note=f"layer_idx={idx}")

    if name in ("attn-sliding", "attn-csa", "attn-hca"):
        return attention(
            {
                "attn-sliding": "sliding_attention",
                "attn-csa": "compressed_sparse_attention",
                "attn-hca": "heavily_compressed_attention",
            }[name]
        )

    if name in ("compressor-csa", "compressor-hca"):
        return compressor(
            {"compressor-csa": "compressed_sparse_attention", "compressor-hca": "heavily_compressed_attention"}[name]
        )

    if name in ("indexer", "indexer-scorer"):
        idx = layer_index_of(config, "compressed_sparse_attention")
        with torch.device(device), default_dtype(dtype):
            attn = DeepseekV4Attention(config, layer_idx=idx)
        randomize(attn)
        indexer = attn.compressor.indexer
        q_residual = attn.q_a_norm(attn.q_a_proj(hidden)).detach().requires_grad_(True)
        if name == "indexer":
            # `topk` returns int64 indices, so there is nothing to take a backward through.
            return Point(
                indexer,
                lambda: indexer(hidden, q_residual, packed),
                differentiable=False,
                note="forward only: output is int64 topk indices",
            )
        from prime_rl.trainer.models.deepseek_v4.rotary import apply_rotary_pos_emb_interleaved

        compressed_kv = indexer.compressor.compress(hidden, packed).detach().requires_grad_(True)
        cos, sin = packed.position_embeddings[indexer.compressor.rope_layer_type]
        q = indexer.q_b_proj(q_residual).view(batch, seq_len, -1, indexer.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb_interleaved(q, cos, sin).transpose(1, 2).detach().requires_grad_(True)
        scorer = indexer.scorer
        return Point(
            scorer,
            lambda: scorer(q, compressed_kv, hidden),
            differentiable=False,
            note="fp32 matmul + relu + weighted sum, in place under no_grad",
        )

    if name == "hyperconnection":
        with torch.device(device), default_dtype(dtype):
            module = DeepseekV4HyperConnection(config)
        randomize(module)
        streams = torch.randn(
            batch, seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype, requires_grad=True
        )
        # Backward is taken through `collapsed`, the only one of the three the sublayer consumes
        # as a tensor of activations; `post` and `comb` are gates of size O(t * hc^2).
        return Point(module, lambda: module(streams)[2], note="backward through `collapsed`")

    if name == "rotary":
        return Point(
            rotary,
            lambda: rotary(packed.position_ids, "compress", dtype=dtype)[0],
            differentiable=False,
            note="buffers only, no parameters",
        )

    if name == "rmsnorm":
        with torch.device(device), default_dtype(dtype):
            module = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        randomize(module)
        return Point(module, lambda: module(hidden))

    if name == "packed-context":
        return Point(
            rotary,
            lambda: PackedContext.build(
                rotary_emb=rotary,
                seq_lens=torch.tensor(seq_lens, device=device),
                dtype=dtype,
                device=device,
            ),
            differentiable=False,
            note="once per model forward",
        )

    if name in ("decoder-csa", "decoder-hca"):
        layer_type = "compressed_sparse_attention" if name == "decoder-csa" else "heavily_compressed_attention"
        # The first three layers are hash-routed and read `input_ids`, so take the first layer of
        # the wanted type that runs the standard router instead.
        idx = next(i for i, t in enumerate(config.layer_types) if t == layer_type and i >= config.num_hash_layers)
        with torch.device(device), default_dtype(dtype):
            module = DeepseekV4DecoderLayer(config, idx)
        randomize(module)
        if ac == "full":
            module = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="full", freq=1))(module)
        streams = torch.randn(
            batch, seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype, requires_grad=True
        )
        return Point(module, lambda: module(streams, packed=packed), note=f"layer_idx={idx}, ac={ac}")

    raise ValueError(f"unknown module {name!r}")


MODULES = [
    "attn-sliding",
    "attn-csa",
    "attn-hca",
    "compressor-csa",
    "compressor-hca",
    "indexer",
    "indexer-scorer",
    "hyperconnection",
    "rotary",
    "rmsnorm",
    "packed-context",
    "decoder-csa",
    "decoder-hca",
]


def _scalar(output):
    """A scalar to call `.backward()` on, in fp32 so the reduction itself cannot overflow."""
    leaves = [t for t in pytree.tree_leaves(output) if isinstance(t, torch.Tensor) and t.is_floating_point]
    return sum(t.float().pow(2).mean() for t in leaves)


def measure_memory(point: Point) -> dict:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    output = point.forward()
    torch.cuda.synchronize()
    result = {
        "baseline_bytes": baseline,
        "fwd_peak_allocated": torch.cuda.max_memory_allocated(),
        "fwd_peak_reserved": torch.cuda.max_memory_reserved(),
        "retained_after_fwd": torch.cuda.memory_allocated(),
    }
    if not point.differentiable:
        result["bwd_peak_allocated"] = None
        result["bwd_peak_reserved"] = None
        return result

    loss = _scalar(output)
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    result["bwd_peak_allocated"] = torch.cuda.max_memory_allocated()
    result["bwd_peak_reserved"] = torch.cuda.max_memory_reserved()
    return result


def measure_timing(point: Point, warmup: int = 25, rep: int = 100) -> dict:
    import triton.testing

    quantiles = [0.5, 0.2, 0.8]
    params = point.parameters()

    def forward_only():
        with torch.no_grad():
            point.forward()

    fwd = triton.testing.do_bench(forward_only, warmup=warmup, rep=rep, quantiles=quantiles)
    result = {"fwd_ms": {"p50": fwd[0], "p20": fwd[1], "p80": fwd[2]}}
    if not point.differentiable:
        result["fwd_bwd_ms"] = None
        result["bwd_ms"] = None
        return result

    def forward_backward():
        _scalar(point.forward()).backward()

    both = triton.testing.do_bench(
        forward_backward, warmup=warmup, rep=rep, quantiles=quantiles, grad_to_none=params or None
    )
    result["fwd_bwd_ms"] = {"p50": both[0], "p20": both[1], "p80": both[2]}
    result["bwd_ms"] = both[0] - fwd[0]
    return result


def measure_attribution(point: Point, trace_dir: Path | None) -> dict:
    from torch.profiler import ProfilerActivity, profile, record_function

    log = AllocationLog()
    with log:
        log.label = "forward"
        output = point.forward()
        torch.cuda.synchronize()
        if point.differentiable:
            log.label = "backward"
            _scalar(output).backward()
            torch.cuda.synchronize()
    del output

    result = {"top_allocations": log.top(25), "bytes_by_op": log.total_by_op()}

    if trace_dir is None:
        return result

    torch.cuda.empty_cache()
    started = time.monotonic()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(2):
            with record_function("forward"):
                out = point.forward()
            if point.differentiable:
                with record_function("backward"):
                    _scalar(out).backward()
            del out
            torch.cuda.synchronize()
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace.json.gz"
    prof.export_chrome_trace(str(trace_path))
    result["trace"] = str(trace_path)
    # Two profiled iterations against `do_bench`'s median is the check that the profiler is
    # actually attached: if this is not materially larger, the attribution data is suspect.
    result["profiled_wall_s_for_2_iters"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("module", choices=MODULES)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("--mode", choices=["memory", "timing", "attribution"], default="memory")
    parser.add_argument("--doc-len", type=int, default=8192, help="document length packed into the row")
    parser.add_argument("--ac", choices=["none", "full"], default="none", help="decoder-layer activation checkpointing")
    parser.add_argument(
        "--attn-impl",
        choices=list(get_args(DSV4AttnImplementation)),
        default="kernel",
        help="CSA attention implementation, set on the config the module is built from",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--tiny", action="store_true", help="toy config, for the harness's own smoke test")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--snapshot", type=Path, default=None, help="dump a CUDA memory snapshot pickle here")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    record = {
        "module": args.module,
        "seq_len": args.seq_len,
        "doc_len": args.doc_len,
        "mode": args.mode,
        "ac": args.ac,
        "attn_impl": args.attn_impl,
        "dtype": args.dtype,
        "tiny": args.tiny,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
    }

    if args.tiny:
        config = DeepseekV4Config(
            vocab_size=64,
            hidden_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=5,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=32,
            q_lora_rank=64,
            partial_rotary_factor=0.5,
            max_position_embeddings=256,
            sliding_window=6,
            o_groups=2,
            o_lora_rank=16,
            index_n_heads=4,
            index_head_dim=24,
            index_topk=8,
            n_routed_experts=8,
            num_experts_per_tok=2,
            num_hash_layers=1,
            compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
            layer_types=[
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
                "compressed_sparse_attention",
                "sliding_attention",
            ],
        )
    else:
        config = build_config()
        assert_real_config(config)
    # Before `build_point`, since every attention layer resolves this at construction.
    config.dsv4_attn = args.attn_impl

    seq_lens = document_layout(args.seq_len, args.doc_len)
    record["seq_lens"] = seq_lens
    randomize = _load_randomize()

    if args.snapshot is not None:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    try:
        point = build_point(args.module, config, seq_lens, dtype, device, args.ac, randomize)
        point.note = ", ".join(filter(None, [point.note, f"attn_impl={args.attn_impl}"]))
        record["note"] = point.note
        record["module_params"] = sum(p.numel() for p in point.parameters())
        if not args.tiny and args.module.startswith("attn-"):
            assert record["module_params"] > 1e8, record["module_params"]
        if args.mode == "memory":
            record |= measure_memory(point)
        elif args.mode == "timing":
            record |= measure_timing(point, warmup=args.warmup, rep=args.rep)
        else:
            record |= measure_attribution(point, args.trace_dir)
        record["status"] = "ok"
    except torch.OutOfMemoryError as error:
        # The one place `AGENTS.md`'s minimal-try/except rule is deliberately overridden: the
        # exception *is* the measurement. Nothing is retried and nothing is unwound, because a
        # partially applied backward leaves every later number in this process meaningless.
        record["status"] = "oom"
        record["error"] = str(error)[:400]

    if args.snapshot is not None and record["status"] == "ok":
        import pickle

        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        with open(args.snapshot, "wb") as handle:
            pickle.dump(torch.cuda.memory._snapshot(), handle)
        record["snapshot"] = str(args.snapshot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k not in ("top_allocations", "bytes_by_op", "error")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
