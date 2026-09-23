"""Drive one DSv4 attention layer at V4 Flash shapes through a stream of synthetic packings.

Per packing it records the layer's fwd+bwd wall time, the tilelang compiles that step triggered,
and the sparse attention kernel's own fwd+bwd time, re-run in isolation on the exact inputs the
layer handed it. Document lengths are log-normal and packed up to `seq_len`, the last document
truncated to fit, so the longest document (and with it the HCA gather width) varies step to step.
"""

import argparse
import json
import math
import statistics
import time

import torch
from tilelang.jit import JITImpl

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.kernels.deepseek_v4 import dsv4_sparse_attn as sparse_attn_module
from prime_rl.utils.utils import default_dtype

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
    max_position_embeddings=1 << 20,
    moe_intermediate_size=64,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=1,
    hc_mult=4,
)
LAYER_IDX = {"csa": 0, "hca": 1, "sliding": 2}


def sample_packing(rng: torch.Generator, seq_len: int, median_doc: int, sigma: float) -> list[int]:
    doc_lens = []
    remaining = seq_len
    while remaining > 0:
        draw = math.exp(math.log(median_doc) + sigma * float(torch.randn(1, generator=rng)))
        length = min(max(int(draw), 16), remaining)
        doc_lens.append(length)
        remaining -= length
    return doc_lens


def instrument_compiles() -> list[dict]:
    """Record every tilelang compile the sparse attention kernels trigger, with its wall time."""
    log: list[dict] = []
    for name, jit in vars(sparse_attn_module).items():
        if not isinstance(jit, JITImpl):
            continue

        def timed_compile(*args, _name=name, _real=jit.compile, **kwargs):
            start = time.perf_counter()
            kernel = _real(*args, **kwargs)
            log.append(dict(kernel=_name, key=repr(args[2:3]), seconds=time.perf_counter() - start))
            return kernel

        jit.compile = timed_compile
    return log


def capture_kernel_inputs() -> dict:
    captured: dict = {}
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def kernel(q, kv_buf, indices, sinks, scale):
        captured.update(q=q, kv_buf=kv_buf, indices=indices, sinks=sinks, scale=scale)
        return real_kernel(q, kv_buf, indices, sinks, scale)

    dsv4_attention.dsv4_sparse_attn = kernel
    return captured


def time_kernel(captured: dict, repeats: int) -> tuple[float, float]:
    """Median fwd and bwd milliseconds of the sparse attention op on the captured inputs."""
    q = captured["q"].detach().requires_grad_(True)
    kv = captured["kv_buf"].detach().requires_grad_(True)
    sinks = captured["sinks"].detach().requires_grad_(True)
    fwd_ms, bwd_ms = [], []
    for _ in range(repeats):
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        out, _ = sparse_attn_module.dsv4_sparse_attn(q, kv, captured["indices"], sinks, captured["scale"])
        mid.record()
        out.backward(torch.ones_like(out))
        end.record()
        torch.cuda.synchronize()
        fwd_ms.append(start.elapsed_time(mid))
        bwd_ms.append(mid.elapsed_time(end))
    return statistics.median(fwd_ms), statistics.median(bwd_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("layer", choices=sorted(LAYER_IDX))
    parser.add_argument("seq_len", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--median-doc", type=int, default=4096)
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--kernel-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = DeepseekV4Config(**V4FLASH_MODEL)
    device = torch.device("cuda")
    with torch.device(device), default_dtype(torch.bfloat16):
        layer = DeepseekV4Attention(config, layer_idx=LAYER_IDX[args.layer])
        rotary = DeepseekV4RotaryEmbedding(config)
    for param in layer.parameters():
        torch.nn.init.normal_(param, std=0.02)

    compiles = instrument_compiles()
    captured = capture_kernel_inputs()
    rng = torch.Generator().manual_seed(args.seed)
    steps = []
    for step in range(args.steps):
        doc_lens = sample_packing(rng, args.seq_len, args.median_doc, args.sigma)
        n_compiles_before = len(compiles)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        packed = PackedContext.build(
            rotary_emb=rotary,
            seq_lens=torch.tensor(doc_lens),
            dtype=torch.bfloat16,
            device=device,
        )
        hidden = torch.randn(1, args.seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
        hidden.requires_grad_(True)
        out, _ = layer(hidden, packed)
        out.backward(torch.ones_like(out))
        torch.cuda.synchronize()
        layer_s = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        fwd_ms, bwd_ms = time_kernel(captured, args.kernel_repeats)
        new_compiles = compiles[n_compiles_before:]
        row = dict(
            label=args.label,
            layer=args.layer,
            seq_len=args.seq_len,
            step=step,
            max_doc=max(doc_lens),
            n_docs=len(doc_lens),
            width=sparse_attn_module._pad_slots_to_tile(captured["indices"]).shape[-1],
            layer_s=layer_s,
            compile_s=sum(c["seconds"] for c in new_compiles),
            n_compiles=len(new_compiles),
            kernel_fwd_ms=fwd_ms,
            kernel_bwd_ms=bwd_ms,
            peak_gb=peak_gb,
        )
        steps.append(row)
        print(json.dumps(row), flush=True)
        layer.zero_grad(set_to_none=True)
        captured.clear()

    steady = steps[len(steps) // 4 :]
    summary = dict(
        label=args.label,
        layer=args.layer,
        seq_len=args.seq_len,
        steps=len(steps),
        distinct_widths=len({row["width"] for row in steps}),
        total_compiles=len(compiles),
        total_compile_s=sum(c["seconds"] for c in compiles),
        last_compile_step=max((row["step"] for row in steps if row["n_compiles"]), default=None),
        mean_kernel_ms=statistics.mean(row["kernel_fwd_ms"] + row["kernel_bwd_ms"] for row in steps),
        mean_kernel_fwd_ms=statistics.mean(row["kernel_fwd_ms"] for row in steps),
        mean_kernel_bwd_ms=statistics.mean(row["kernel_bwd_ms"] for row in steps),
        median_steady_layer_s=statistics.median([row["layer_s"] for row in steady if not row["n_compiles"]] or [math.nan]),
        max_peak_gb=max(row["peak_gb"] for row in steps),
    )
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
