"""Trainer/inference parity tests for the recurrent GatedDeltaNet forward.

The trainer's `gdn_recurrent_forward` path calls vLLM's own decode kernel
(`fused_sigmoid_gating_delta_rule_update`) for the forward, so kernel-level
parity with the generator holds by construction. What remains to test is the
integration: that the trainer's full-batch zero-state call reproduces a real
token-by-token vLLM decode through a state cache, that the chunked backward
matches the training path's gradients, and that the module/flag wiring holds.
A separate diagnostic quantifies the conv1d prefill/decode gap, which the
recurrent kernel does not address.
"""

import pytest
import torch

from prime_rl.trainer.models.layers.gdn_recurrent import (
    gdn_gate,
    gdn_recurrent_fwd,
    gdn_recurrent_fwd_chunked_bwd,
)

pytestmark = [pytest.mark.gpu]

# Real Qwen3.5 GDN head dims (K=V=128 drives the kernel tiling); small head
# counts and uneven lengths that cross the 64-token chunk boundary.
H, HV, K, V = 4, 8, 128, 128
SEQ_LENS = [127, 257, 64]
DTYPE = torch.bfloat16


def make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    total = sum(SEQ_LENS)
    q = torch.randn(1, total, HV, K, device="cuda", dtype=DTYPE)
    k = torch.randn(1, total, HV, K, device="cuda", dtype=DTYPE)
    v = torch.randn(1, total, HV, V, device="cuda", dtype=DTYPE)
    a = torch.randn(1, total, HV, device="cuda", dtype=DTYPE)
    b = torch.randn(1, total, HV, device="cuda", dtype=DTYPE)
    A_log = torch.empty(HV, device="cuda", dtype=torch.float32).uniform_(0, 2.7).log()
    dt_bias = torch.rand(HV, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, cu_seqlens


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_recurrent_fwd_matches_vllm_decode_sim(state_dtype):
    """Trainer forward (full packed batch, zero state) vs token-by-token vLLM decode through a state cache.

    This is the parity claim that matters: the trainer's recurrent path, called
    the way the module calls it, must reproduce what vLLM actually did at
    generation time. vLLM's SSM cache dtype defaults to the model dtype (bf16),
    quantizing the recurrent state on every token; the trainer's zero-state
    forward carries fp32 in registers. So parity is bitwise only with an fp32
    cache (server: ``mamba_ssm_cache_dtype = "float32"`` via ``vllm_extra``);
    the bf16 run quantifies the irreducible gap under the default server config.
    """
    from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update

    q, k, v, a, b, A_log, dt_bias, cu_seqlens = make_inputs()

    o_ours = gdn_recurrent_fwd(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, cu_seqlens=cu_seqlens)

    outputs = []
    token_cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    slot = torch.tensor([1], device="cuda", dtype=torch.int32)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        state_cache = torch.zeros(2, HV, V, K, device="cuda", dtype=state_dtype)
        for t in range(start, end):
            o_t, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a[:, t : t + 1].contiguous(),
                b=b[:, t : t + 1].contiguous(),
                dt_bias=dt_bias,
                q=q[:, t : t + 1].contiguous(),
                k=k[:, t : t + 1].contiguous(),
                v=v[:, t : t + 1].contiguous(),
                initial_state=state_cache,
                inplace_final_state=True,
                cu_seqlens=token_cu,
                ssm_state_indices=slot,
                use_qk_l2norm_in_kernel=True,
            )
            outputs.append(o_t)
    o_vllm = torch.cat(outputs, dim=1)

    diff = (o_ours.float() - o_vllm.float()).abs().max().item()
    print(f"\ndecode sim (ssm cache {state_dtype}): bitwise={torch.equal(o_ours, o_vllm)}, max abs diff={diff}")
    if state_dtype == torch.float32:
        assert torch.equal(o_ours, o_vllm), f"not bitwise equal, max abs diff: {diff}"
    else:
        assert diff < 5e-2, f"bf16-cache gap unexpectedly large: {diff}"


def test_recurrent_close_to_chunked():
    """Recurrent and chunked compute the same function; the gap should be small bf16 noise."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, a, b, A_log, dt_bias, cu_seqlens = make_inputs()

    o_rec = gdn_recurrent_fwd(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, cu_seqlens=cu_seqlens)
    o_chunk, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=gdn_gate(a, A_log, dt_bias),
        beta=b.sigmoid(),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    diff = (o_rec.float() - o_chunk.float()).abs().max().item()
    assert diff < 5e-2, f"recurrent vs chunked max abs diff too large: {diff}"


def test_backward_matches_chunked_path():
    """With identical upstream grads, the recurrent path's backward must equal the chunked path's bitwise."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, a, b, A_log, dt_bias, cu_seqlens = make_inputs()
    torch.manual_seed(1)
    w = torch.randn(1, sum(SEQ_LENS), HV, V, device="cuda", dtype=torch.float32)

    def run(path: str) -> dict[str, torch.Tensor]:
        inputs = {
            name: t.clone().requires_grad_()
            for name, t in [("q", q), ("k", k), ("v", v), ("a", a), ("b", b), ("A_log", A_log), ("dt_bias", dt_bias)]
        }
        if path == "recurrent":
            o = gdn_recurrent_fwd_chunked_bwd(
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                a=inputs["a"],
                b=inputs["b"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                cu_seqlens=cu_seqlens,
            )
        else:
            o, _ = chunk_gated_delta_rule(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                g=gdn_gate(inputs["a"], inputs["A_log"], inputs["dt_bias"]),
                beta=inputs["b"].sigmoid(),
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        # Loss linear in o: dL/do = w exactly, identical for both paths even
        # though the forward activations differ in the low bits.
        (o.float() * w).sum().backward()
        return {name: t.grad for name, t in inputs.items()}

    grads_rec = run("recurrent")
    grads_chunk = run("chunked")
    for name in grads_rec:
        assert torch.equal(grads_rec[name], grads_chunk[name]), (
            f"grad mismatch for {name}: "
            f"max abs diff {(grads_rec[name].float() - grads_chunk[name].float()).abs().max().item()}"
        )


def test_gdn_module_recurrent_flag():
    """Module-level wiring: the flag swaps the forward, keeps outputs close, and trains."""
    from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
    from prime_rl.utils.utils import default_dtype

    config = Qwen3_5MoeConfig(
        hidden_size=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=K,
        linear_value_head_dim=V,
        linear_num_key_heads=H,
        linear_num_value_heads=HV,
        rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    with torch.device("cuda"), default_dtype(DTYPE):
        module_chunk = Qwen3_5MoeGatedDeltaNet(config)
        config.gdn_recurrent_forward = True
        module_rec = Qwen3_5MoeGatedDeltaNet(config)
    module_rec.load_state_dict(module_chunk.state_dict())
    assert module_rec.recurrent_forward and not module_chunk.recurrent_forward

    total = sum(SEQ_LENS)
    hidden = torch.randn(1, total, config.hidden_size, device="cuda", dtype=DTYPE)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)

    out_chunk = module_chunk(hidden, cu_seqlens=cu_seqlens)
    out_rec = module_rec(hidden, cu_seqlens=cu_seqlens)
    diff = (out_rec.float() - out_chunk.float()).abs().max().item()
    assert diff < 1e-1, f"module outputs diverged: max abs diff {diff}"

    out_rec.float().sum().backward()
    for name, param in module_rec.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert param.grad.isfinite().all(), f"non-finite grad for {name}"


def test_conv_prefill_decode_parity():
    """Diagnostic: FLA's packed conv (trainer) vs vLLM's prefill conv and decode conv-state update.

    The recurrent GDN kernel does not touch the conv path, so any gap reported
    here is residual trainer/inference mismatch from the conv1d.
    """
    from fla.modules.conv import causal_conv1d as fla_causal_conv1d
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    torch.manual_seed(0)
    dim, width = 64, 4
    n_seqs = len(SEQ_LENS)
    total = sum(SEQ_LENS)
    x = torch.randn(1, total, dim, device="cuda", dtype=DTYPE)
    weight = torch.randn(dim, width, device="cuda", dtype=DTYPE) * 0.1
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)

    o_fla, _ = fla_causal_conv1d(x=x, weight=weight, bias=None, activation="silu", cu_seqlens=cu_seqlens)

    # vLLM prefill over the same packed batch (fresh conv states).
    conv_states = torch.zeros(n_seqs + 1, dim, width - 1, device="cuda", dtype=DTYPE)
    cache_indices = torch.arange(1, n_seqs + 1, device="cuda", dtype=torch.int32)
    o_vllm_prefill = causal_conv1d_fn(
        x=x[0].transpose(0, 1).contiguous(),
        weight=weight,
        bias=None,
        conv_states=conv_states,
        query_start_loc=cu_seqlens,
        cache_indices=cache_indices,
        has_initial_state=torch.zeros(n_seqs, device="cuda", dtype=torch.bool),
        activation="silu",
    ).transpose(0, 1)

    prefill_diff = (o_fla[0].float() - o_vllm_prefill.float()).abs().max().item()
    prefill_bitwise = torch.equal(o_fla[0], o_vllm_prefill)

    # vLLM decode: prime the conv state on a prefix, then step the tail token by token.
    prefix_len, seq_len = 32, SEQ_LENS[0]
    conv_states.zero_()
    prefix_cu = torch.tensor([0, prefix_len], device="cuda", dtype=torch.int32)
    slot = torch.tensor([1], device="cuda", dtype=torch.int32)
    causal_conv1d_fn(
        x=x[0, :prefix_len].transpose(0, 1).contiguous(),
        weight=weight,
        bias=None,
        conv_states=conv_states,
        query_start_loc=prefix_cu,
        cache_indices=slot,
        has_initial_state=torch.zeros(1, device="cuda", dtype=torch.bool),
        activation="silu",
    )
    decode_outs = []
    for t in range(prefix_len, seq_len):
        decode_outs.append(
            causal_conv1d_update(
                x=x[0, t : t + 1].contiguous(),
                conv_state=conv_states,
                weight=weight,
                bias=None,
                activation="silu",
                conv_state_indices=slot,
            )
        )
    o_vllm_decode = torch.cat(decode_outs, dim=0)
    decode_diff = (o_fla[0, prefix_len:seq_len].float() - o_vllm_decode.float()).abs().max().item()
    decode_bitwise = torch.equal(o_fla[0, prefix_len:seq_len], o_vllm_decode)

    print(f"\nconv prefill: bitwise={prefill_bitwise}, max abs diff={prefill_diff}")
    print(f"conv decode:  bitwise={decode_bitwise}, max abs diff={decode_diff}")
    assert prefill_diff < 1e-2, f"conv prefill diff too large: {prefill_diff}"
    assert decode_diff < 1e-2, f"conv decode diff too large: {decode_diff}"
