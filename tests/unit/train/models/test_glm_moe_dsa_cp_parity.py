"""Parity test for GLM-5 sparse MLA context parallelism.

Compares the CP=2 sharded forward against a CP=1 reference. The reference and
the sharded run share weights and inputs (same RNG seed), so any divergence
should come from the CP plumbing only.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import (
    GlmMoeDsaAttention,
    SparseMlaAttentionArgs,
)

pytestmark = [pytest.mark.gpu]


def _make_args() -> SparseMlaAttentionArgs:
    # The sparse MLA kernel asserts kv_lora_rank + qk_rope_head_dim == 576 (GLM-5 standard).
    # Other dims kept small to make the unit test cheap.
    qk_rope = 64
    qk_nope = 64
    return SparseMlaAttentionArgs(
        hidden_size=512,
        num_attention_heads=16,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_rope_head_dim=qk_rope,
        qk_nope_head_dim=qk_nope,
        qk_head_dim=qk_rope + qk_nope,
        v_head_dim=64,
        attention_bias=False,
        rms_norm_eps=1e-6,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=64,
    )


def _make_cos_sin(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).to(dtype)
    sin = emb.sin().unsqueeze(0).to(dtype)
    return cos, sin


def _build_full_inputs(args: SparseMlaAttentionArgs, seq_len: int, device, dtype):
    g = torch.Generator(device=device).manual_seed(123)
    hidden_full = torch.randn(1, seq_len, args.hidden_size, device=device, dtype=dtype, generator=g)
    cos_full, sin_full = _make_cos_sin(seq_len, args.qk_rope_head_dim, device, dtype)
    ks_full = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke_full = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device)
    return hidden_full, (cos_full, sin_full), ks_full, ke_full


def _worker(rank: int, world_size: int, port: int, seq_len: int, tmpdir: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))

    args = _make_args()
    hidden_full, (cos_full, sin_full), ks_full, ke_full = _build_full_inputs(args, seq_len, device, dtype)

    # Build the CP attention module with deterministic weights.
    torch.manual_seed(42)
    attn_cp = GlmMoeDsaAttention(args).to(device=device, dtype=dtype).eval()
    attn_cp.set_context_parallel_attributes(cp_group, rank, world_size)

    chunk = seq_len // world_size
    start = rank * chunk
    end = start + chunk
    hidden_local = hidden_full[:, start:end].contiguous()

    with torch.no_grad():
        out_local, _ = attn_cp(hidden_local, (cos_full, sin_full), ks_full, ke_full)

    torch.save(out_local.float().cpu(), os.path.join(tmpdir, f"cp_out_rank{rank}.pt"))

    # Rank 0 also computes a CP=1 reference using a fresh module with the same seed.
    if rank == 0:
        torch.manual_seed(42)
        attn_ref = GlmMoeDsaAttention(args).to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            out_ref, _ = attn_ref(hidden_full, (cos_full, sin_full), ks_full, ke_full)
        torch.save(out_ref.float().cpu(), os.path.join(tmpdir, "ref_out.pt"))

    dist.barrier(group=cp_group)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_sparse_mla_cp_parity(tmp_path, free_port, world_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Needs >={world_size} GPUs")

    seq_len = 256

    mp.spawn(
        _worker,
        args=(world_size, free_port, seq_len, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    ref = torch.load(tmp_path / "ref_out.pt")
    chunks = [torch.load(tmp_path / f"cp_out_rank{r}.pt") for r in range(world_size)]
    cp_out = torch.cat(chunks, dim=1)

    assert ref.shape == cp_out.shape, f"shape mismatch: ref {ref.shape} vs cp {cp_out.shape}"
    diff = (ref - cp_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = max_diff / (ref.abs().max().item() + 1e-8)
    print(
        f"\ncp_world_size={world_size}: max_abs_diff={max_diff:.3e} "
        f"mean_abs_diff={mean_diff:.3e} max_rel={rel:.3e}"
    )

    # bf16 + fp8 indexer + triton non-determinism: tolerate small absolute diffs.
    assert max_diff < 5e-2, (
        f"CP parity failed: max_abs_diff={max_diff:.3e} mean_abs_diff={mean_diff:.3e} "
        f"max_rel={rel:.3e}"
    )
