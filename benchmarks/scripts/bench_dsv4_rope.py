"""Benchmark DeepSeek-V4 interleaved RoPE at the op level and one attention layer at the layer level."""

import argparse
import statistics
from collections.abc import Callable, Iterator

import torch

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb_interleaved
from prime_rl.utils.utils import default_dtype

try:
    from prime_rl.trainer.models.kernels.deepseek_v4.interleaved_rope import apply_interleaved_rope_
except ImportError as error:
    apply_interleaved_rope_ = None
    FUSED_IMPORT_ERROR = str(error)

V4FLASH_MODEL = dict(
    vocab_size=64,
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=1,
    head_dim=512,
    q_lora_rank=1024,
    o_groups=8,
    o_lora_rank=1024,
    qk_rope_head_dim=64,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    sliding_window=128,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
    layer_types=["compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention"],
    num_hidden_layers=3,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    max_position_embeddings=65536,
    moe_intermediate_size=64,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=1,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
)
V4FLASH_CONFIG = DeepseekV4Config(**V4FLASH_MODEL)
LAYER_INDICES = {"csa": 0, "hca": 1, "sliding": 2}
DTYPE = torch.bfloat16
HEADS = V4FLASH_MODEL["num_attention_heads"]
HEAD_DIM = V4FLASH_MODEL["head_dim"]
INDEX_HEAD_DIM = V4FLASH_MODEL["index_head_dim"]


def doc_lens_for(tokens: int) -> tuple[int, ...]:
    """Four uneven documents summing to `tokens`."""
    first, second, third = tokens // 8, 3 * tokens // 8, tokens // 4
    return (first, second, third, tokens - first - second - third)


def build_rotary() -> DeepseekV4RotaryEmbedding:
    with torch.device("cuda"), default_dtype(DTYPE):
        return DeepseekV4RotaryEmbedding(V4FLASH_CONFIG)


def build_packed(rotary: DeepseekV4RotaryEmbedding, tokens: int) -> PackedContext:
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(doc_lens_for(tokens), device="cuda"),
        device=torch.device("cuda"),
    )


def fused_cos_sin(rotary: DeepseekV4RotaryEmbedding, rope_type: str, n_rows: int) -> torch.Tensor:
    rows = torch.arange(n_rows, device="cuda")[None]
    cos, sin = rotary(rows, rope_type, dtype=torch.float32)
    return torch.cat([cos[0], sin[0]], dim=-1).contiguous()


def time_ms(step: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def peak_mib(step: Callable[[], None]) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 2**20


def measure(step: Callable[[], None], warmup: int, iters: int) -> str:
    try:
        ms = time_ms(step, warmup, iters)
        mib = peak_mib(step)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM | OOM"
    return f"{ms:.3f} | {mib:.1f}"


def fwd_and_fwd_bwd(forward: Callable[[], torch.Tensor], grad_out: torch.Tensor) -> tuple[Callable, Callable]:
    def fwd() -> None:
        forward()

    def fwd_bwd() -> None:
        forward().backward(grad_out)

    return fwd, fwd_bwd


def eager_main_q(leaf, cos, sin, **_):
    q = (leaf * 1).transpose(1, 2)
    return apply_rotary_pos_emb_interleaved(q, cos, sin).transpose(1, 2).contiguous()


def eager_kv(leaf, cos, sin, **_):
    return apply_rotary_pos_emb_interleaved(leaf * 1, cos, sin, unsqueeze_dim=2)


def eager_indexer_q(leaf, cos, sin, **_):
    q = (leaf * 1).transpose(1, 2)
    return apply_rotary_pos_emb_interleaved(q, cos, sin).transpose(1, 2)


def eager_attn_output(leaf, cos, sin, **_):
    return apply_rotary_pos_emb_interleaved(leaf * 1, cos, -sin, unsqueeze_dim=2)


def fused_forward(leaf, cos_sin, position_ids, **_):
    return apply_interleaved_rope_(leaf * 1, cos_sin, position_ids)


def fused_attn_output(leaf, cos_sin, position_ids, **_):
    return apply_interleaved_rope_((leaf * 1).clone(), cos_sin, position_ids, inverse=True)


OP_CASES = [
    ("main_q", HEADS, HEAD_DIM, "main", True, eager_main_q, fused_forward),
    ("kv", 1, HEAD_DIM, "main", True, eager_kv, fused_forward),
    ("indexer_q", HEADS, INDEX_HEAD_DIM, "compress", False, eager_indexer_q, fused_forward),
    ("attn_output", HEADS, HEAD_DIM, "main", True, eager_attn_output, fused_attn_output),
]


def op_case_rows(
    rotary: DeepseekV4RotaryEmbedding, packed: PackedContext, tokens: int, case: tuple, warmup: int, iters: int
) -> Iterator[str]:
    name, heads, dim, rope_type, has_backward, eager_impl, fused_impl = case
    cos, sin = rotary(packed.position_ids, rope_type, dtype=DTYPE)
    cos_sin = fused_cos_sin(rotary, rope_type, tokens)
    position_ids = packed.position_ids[0]
    leaf = torch.randn(1, tokens, heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    grad_out = torch.randn_like(leaf)
    variants = [("eager", eager_impl)]
    if apply_interleaved_rope_ is not None:
        variants.append(("fused", fused_impl))
    for variant, impl in variants:

        def forward(impl=impl) -> torch.Tensor:
            return impl(leaf, cos=cos, sin=sin, cos_sin=cos_sin, position_ids=position_ids)

        fwd, fwd_bwd = fwd_and_fwd_bwd(forward, grad_out)
        if has_backward:
            fwd_cell, fwd_bwd_cell = measure(fwd, warmup, iters), measure(fwd_bwd, warmup, iters)
        else:
            with torch.no_grad():
                fwd_cell = measure(fwd, warmup, iters)
            fwd_bwd_cell = "n/a | n/a"
        yield f"| {name} | {variant} | {tokens} | {fwd_cell} | {fwd_bwd_cell} |"


def run_op_level(token_counts: list[int], warmup: int, iters: int) -> Iterator[str]:
    rotary = build_rotary()
    for tokens in token_counts:
        packed = build_packed(rotary, tokens)
        for case in OP_CASES:
            yield from op_case_rows(rotary, packed, tokens, case, warmup, iters)
            torch.cuda.empty_cache()


def build_attention(layer_idx: int) -> DeepseekV4Attention:
    with torch.device("cuda"), default_dtype(DTYPE):
        module = DeepseekV4Attention(V4FLASH_CONFIG, layer_idx=layer_idx)
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("scale") or name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            elif name.endswith("sinks") or name.endswith("position_bias"):
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)
    return module


def layer_row(
    module: DeepseekV4Attention, rotary: DeepseekV4RotaryEmbedding, layer: str, tokens: int, warmup: int, iters: int
) -> str:
    try:
        packed = build_packed(rotary, tokens)
        hidden = torch.randn(1, tokens, V4FLASH_MODEL["hidden_size"], device="cuda", dtype=DTYPE, requires_grad=True)
        grad_out = torch.randn_like(hidden)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return f"| {layer} | {tokens} | OOM | OOM | OOM | OOM |"
    fwd, fwd_bwd = fwd_and_fwd_bwd(lambda: module(hidden, packed)[0], grad_out)
    return f"| {layer} | {tokens} | {measure(fwd, warmup, iters)} | {measure(fwd_bwd, warmup, iters)} |"


def run_layer_level(token_counts: list[int], layers: list[str], warmup: int, iters: int) -> Iterator[str]:
    rotary = build_rotary()
    for layer in layers:
        module = build_attention(LAYER_INDICES[layer])
        for tokens in token_counts:
            yield layer_row(module, rotary, layer, tokens, warmup, iters)
            torch.cuda.empty_cache()
        module = None
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("total_tokens", type=int, nargs="*", default=[131072], help="total packed tokens across CP")
    parser.add_argument("--cp-sizes", type=int, nargs="+", default=[1, 4, 8], help="per-rank tokens = total / cp")
    parser.add_argument("--layers", nargs="*", default=list(LAYER_INDICES), choices=list(LAYER_INDICES))
    parser.add_argument("--levels", nargs="+", default=["op", "layer"], choices=["op", "layer"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    token_counts = sorted({total // cp for total in args.total_tokens for cp in args.cp_sizes}, reverse=True)
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}, dtype {DTYPE}")
    print(f"per-rank tokens: {token_counts}, warmup {args.warmup}, iters {args.iters}")
    print("Times are medians of CUDA-event timings. Memory is peak allocated above the pre-case baseline.")
    print()

    if "op" in args.levels:
        print("## Op level")
        print()
        if apply_interleaved_rope_ is None:
            print(f"Fused rows skipped: {FUSED_IMPORT_ERROR}")
            print()
        print("| case | variant | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |")
        print("|---|---|---|---|---|---|---|")
        for row in run_op_level(token_counts, args.warmup, args.iters):
            print(row, flush=True)
        print()

    if "layer" in args.levels and args.layers:
        print("## Layer level (one DeepseekV4Attention, no CP)")
        print()
        print("| layer | tokens | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB |")
        print("|---|---|---|---|---|---|")
        for row in run_layer_level(token_counts, args.layers, args.warmup, args.iters):
            print(row, flush=True)


if __name__ == "__main__":
    main()
