"""Test attention implementations against SDPA reference."""

import tempfile
from pathlib import Path
import shutil
from typing import Generator


import torch
import torch.distributed as dist

from prime_rl.trainer.models.layers.attn import (
    AttentionConfig,
    FlashAttention,
    SDPAAttention,
    substitute_ring_attn,
)

from prime_rl.trainer.models.layers.ulysses_attn import (
    substitute_ulysses_attn,
    update_ulysses_params,
)

from itertools import accumulate
from dataclasses import dataclass, field
import pytest

import contextlib


@contextlib.contextmanager
def preserve_compute_attention():

    from prime_rl.trainer.models.layers.attn import FlashAttention
    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedFlashAttention,
    )

    originals = {
        cls: cls._compute_attention
        for cls in (FlashAttention, AfmoeFlashAttention, Qwen3_5MoeGatedFlashAttention)
    }
    try:
        yield
    finally:
        for cls, method in originals.items():
            cls._compute_attention = method


@pytest.fixture(scope="module")
def single_proc_group():
    if dist.is_initialized():
        yield dist.group.WORLD
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="prime_rl_test_pg_"))
    store_path = tmp_dir / "pg_store"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        world_size=1,
        rank=0,
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def patch_hf_adapter_params(
    cu_seqlens: torch.Tensor, max_seqlen: int, local_k_slice: slice
) -> None:
    from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

    DATA_PARAMS["cu_seqlens_q"] = cu_seqlens
    DATA_PARAMS["cu_seqlens_k"] = cu_seqlens
    DATA_PARAMS["max_seqlen_q"] = max_seqlen
    DATA_PARAMS["max_seqlen_k"] = max_seqlen
    DATA_PARAMS["local_k_slice"] = local_k_slice


def copy_weights(attn_orig, attn_new):
    # Copy weights to ensure same parameters
    attn_new.load_state_dict(attn_orig.state_dict(), strict=True)
    return attn_new


def iter_fa_versions() -> Generator[tuple[int, str], None, None]:
    yield (2, "flash_attention_2")
    yield (3, "flash_attention_3")
    yield (4, "fa4")


@dataclass
class AttentionTestsInputs:
    seq_lens: list[int]
    cu_seqlens: torch.Tensor  # cummulative seq lenths
    max_seqlen: int
    config: AttentionConfig
    dtype: torch.dtype
    hidden_states: torch.Tensor  # shape [1, sum(seq_lens), hidden_size]
    hidden_states_sdpa: torch.Tensor  # [bs, max_seqlen, hidden_size]
    sdpa_output: torch.Tensor  # [bs, max_seqlen, hid_dim]
    sdpa_output_packed: torch.Tensor  # [1, sum(seq_lens), hidden_size]
    sdpa_attn: (
        torch.nn.Module
    )  # Prime-rl sdpa attention implementation with q,k,v layers
    total_tokens: int = field(init=False)

    def __post_init__(self):
        self.total_tokens = sum(self.seq_lens)


def generate_test_inputs(
    seq_lens: list[int] = [128, 64],
    softmax_scale: float | None = None,
    seed: int = 42,
    dtype=torch.bfloat16,
    device="cuda",
) -> AttentionTestsInputs:
    """
    Args:
        seq_lens (list[int]): lengths of each sequence
        ...
    """
    torch.manual_seed(seed)

    cum_sl = [0] + list(accumulate(seq_lens))
    bs = len(seq_lens)
    hidden_size = 4096
    head_dim = 128
    num_attention_heads = 32
    num_key_value_heads = 8

    config = AttentionConfig(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        is_causal=True,
        attention_bias=False,
        use_qk_norm=False,
        rms_norm_eps=1e-5,
        scaling=softmax_scale,
    )

    # Create inputs
    total_tokens = sum(seq_lens)
    hidden_states = torch.randn(
        1, total_tokens, hidden_size, dtype=dtype, device=device
    )
    cu_seqlens = torch.tensor(cum_sl, dtype=torch.int32, device=device)
    max_seqlen = max(seq_lens)

    # Instantiate SDPA and FA3 attention
    sdpa_attn = SDPAAttention(config).cuda().to(dtype)

    # --- Convert to padded (SDPA-compatible) format ---
    hidden_states_sdpa = torch.zeros(
        bs, max_seqlen, hidden_size, dtype=dtype, device=device
    )

    start = 0
    for i, sl in enumerate(seq_lens):
        hidden_states_sdpa[i, :sl] = hidden_states[0, start : start + sl]
        start += sl

    # Get SDPA output
    sdpa_output, _ = sdpa_attn(hidden_states_sdpa)

    # SDPA output is [bs, seq_len, hidden]; varlen FA output is [1, total_tokens, hidden].
    sdpa_output_packed = torch.zeros(
        1, total_tokens, hidden_size, dtype=dtype, device=device
    )

    start = 0
    for i, sl in enumerate(seq_lens):
        sdpa_output_packed[0, start : start + sl] = sdpa_output[i, :sl]
        start += sl

    attn_inputs = AttentionTestsInputs(
        seq_lens=seq_lens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        config=config,
        dtype=dtype,
        hidden_states=hidden_states,
        hidden_states_sdpa=hidden_states_sdpa,
        sdpa_output=sdpa_output,
        sdpa_output_packed=sdpa_output_packed,
        sdpa_attn=sdpa_attn,
    )

    return attn_inputs


@pytest.mark.parametrize(("softmax_scale"), [(None), (1 / 15)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_variants_vs_sdpa(softmax_scale: float | None):
    """Test FA2, FA3, FA4 implementations match SDPA."""

    attn_inputs = generate_test_inputs(
        seq_lens=[64, 64, 64], softmax_scale=softmax_scale, seed=55
    )

    dtype = attn_inputs.dtype
    config = attn_inputs.config
    sdpa_attn = attn_inputs.sdpa_attn
    hidden_states_sdpa = attn_inputs.hidden_states_sdpa
    sdpa_output = attn_inputs.sdpa_output

    for fa_ver, fa_name in iter_fa_versions():

        print(f"Processing fa={fa_ver}")
        fa_attn = FlashAttention(config, flash_attn_version=fa_ver).cuda().to(dtype)
        if not fa_attn._fa_installed():
            # fa not installed
            print(f"check skipped: {fa_name} not installed")
            continue

        fa_attn = copy_weights(sdpa_attn, fa_attn)
        fa_output, _ = fa_attn(hidden_states_sdpa, cu_seqlens=None, max_seqlen=None)

        diff = torch.abs(fa_output.float() - sdpa_output.float()).max()
        torch.testing.assert_close(
            fa_output.float(),
            sdpa_output.float(),
            atol=1e-2,
            rtol=0,
            msg=f"FA{fa_ver} output differs by {diff}",
        )


@pytest.mark.parametrize(("softmax_scale"), [(None), (1 / 15)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ring_varlen_vs_sdpa(softmax_scale: float | None, single_proc_group):
    """Test ring varlen flash attention against SDPA."""

    attn_inputs = generate_test_inputs(
        seq_lens=[128, 64], softmax_scale=softmax_scale, seed=83
    )

    dtype = attn_inputs.dtype
    config = attn_inputs.config
    sdpa_attn = attn_inputs.sdpa_attn
    hidden_states = attn_inputs.hidden_states
    sdpa_output_packed = attn_inputs.sdpa_output_packed
    cu_seqlens = attn_inputs.cu_seqlens
    max_seqlen = attn_inputs.max_seqlen
    total_tokens = attn_inputs.total_tokens

    # local_k_slice indexes the *token* dim of the gathered kv_buffer, not heads.
    # With world_size=1,  local_k_slice = slice(0, total_tokens)
    patch_hf_adapter_params(
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        local_k_slice=slice(0, total_tokens),
    )

    with preserve_compute_attention():
        for fa_ver, fa_name in iter_fa_versions():

            print(f"Processing varlen fa={fa_ver}")
            substitute_ring_attn(
                single_proc_group,
                heads_k_stride=config.num_key_value_heads,
                attn_impl=fa_name,
            )

            varlen_fa_attn = (
                FlashAttention(config, flash_attn_version=fa_ver).cuda().to(dtype)
            )

            if not varlen_fa_attn._fa_installed():
                # fa not installed
                print(f"check skipped: {fa_name} not installed")
                continue

            varlen_fa_attn = copy_weights(sdpa_attn, varlen_fa_attn)
            fa_output, _ = varlen_fa_attn(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            diff = torch.abs(fa_output.float() - sdpa_output_packed.float()).max()
            torch.testing.assert_close(
                fa_output.float(),
                sdpa_output_packed.float(),
                atol=1e-2,
                rtol=0,
                msg=f"Ring Varlen FA{fa_ver} output differs by {diff}",
            )


@pytest.mark.parametrize(("softmax_scale"), [(None), (1 / 15)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ulysses_varlen_vs_sdpa(softmax_scale: float | None, single_proc_group):
    """Test Ulysses varlen attention against SDPA."""

    attn_inputs = generate_test_inputs(
        seq_lens=[128, 64], softmax_scale=softmax_scale, seed=91
    )

    dtype = attn_inputs.dtype
    config = attn_inputs.config
    sdpa_attn = attn_inputs.sdpa_attn
    hidden_states = attn_inputs.hidden_states
    sdpa_output_packed = attn_inputs.sdpa_output_packed
    cu_seqlens = attn_inputs.cu_seqlens
    max_seqlen = attn_inputs.max_seqlen
    total_tokens = attn_inputs.total_tokens

    # local_k_slice indexes the *token* dim of the gathered kv_buffer, not heads.
    # With world_size=1, this rank owns all tokens.
    update_ulysses_params(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    with preserve_compute_attention():
        for fa_ver, fa_name in iter_fa_versions():
            print(f"Processing varlen fa={fa_ver}")

            substitute_ulysses_attn(single_proc_group, attn_impl=fa_name)

            varlen_fa_attn = (
                FlashAttention(config, flash_attn_version=fa_ver).cuda().to(dtype)
            )

            if not varlen_fa_attn._fa_installed():
                # fa not installed
                print(f"check skipped: {fa_name} not installed")
                continue

            varlen_fa_attn = copy_weights(sdpa_attn, varlen_fa_attn)
            fa_output, _ = varlen_fa_attn(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

            diff = torch.abs(fa_output.float() - sdpa_output_packed.float()).max()
            torch.testing.assert_close(
                fa_output.float(),
                sdpa_output_packed.float(),
                atol=1e-2,
                rtol=0,
                msg=f"Ring Ulysses FA{fa_ver} output differs by {diff}",
            )
