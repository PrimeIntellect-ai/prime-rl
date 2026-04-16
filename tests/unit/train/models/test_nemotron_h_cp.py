"""Test Mamba-2 all-to-all context parallelism produces identical results to non-CP.

Simulates CP by manually sharding inputs and calling the CP-aware forward,
then comparing against a full-sequence non-CP forward. Uses a fake process
group backed by a single GPU.
"""

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.models.layers.cp_mamba import (
    _all_to_all_head_to_seq,
    _all_to_all_seq_to_head,
)
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHMambaLayer
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

_BASE = dict(
    vocab_size=256,
    hidden_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=128,
    intermediate_size=512,
    mamba_expand=2,
    mamba_num_heads=8,
    mamba_head_dim=64,
    ssm_state_size=64,
    mamba_n_groups=2,  # >1 to test group sharding
    mamba_d_conv=4,
    mamba_chunk_size=64,
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=256,
    moe_shared_expert_intermediate_size=256,
    moe_latent_size=128,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
)


def _init_single_gpu_pg():
    """Initialize a single-GPU process group for testing."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="tcp://localhost:29501", rank=0, world_size=1)
    return dist.group.WORLD


@pytest.fixture(autouse=True)
def _setup_dist():
    _init_single_gpu_pg()


def test_all_to_all_roundtrip():
    """Verify seq_to_head -> head_to_seq is identity for cp_size=1."""
    pg = _init_single_gpu_pg()
    x = torch.randn(1, 64, 512, device="cuda")

    y = _all_to_all_seq_to_head(x, pg, cp_size=1)
    z = _all_to_all_head_to_seq(y, pg, cp_size=1)

    assert torch.allclose(x, z, atol=1e-6), f"Roundtrip error: {(x - z).abs().max()}"


def test_mamba_cp_forward_matches_non_cp():
    """CP forward with cp_size=1 should produce identical results to non-CP forward."""
    pg = _init_single_gpu_pg()

    config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba"],
        use_grouped_mm=False,
    )

    with torch.device("cuda"), default_dtype(torch.float32):
        model = NemotronHForCausalLM._from_config(config)

    inject_prime_lm_head(model, chunk_size=None)

    torch.manual_seed(42)
    input_ids = torch.randint(0, 256, (1, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0)

    # Non-CP forward
    model.zero_grad()
    out_no_cp = model(input_ids, position_ids=position_ids)
    loss_no_cp = out_no_cp["logits"].sum()
    loss_no_cp.backward()
    grads_no_cp = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # CP forward (cp_size=1 should be identical)
    mamba_layer = model.model.layers[0]
    assert isinstance(mamba_layer, NemotronHMambaLayer)
    mamba_layer.set_context_parallel_attributes(pg, cp_rank=0, cp_world_size=1)

    model.zero_grad()
    out_cp = model(input_ids, position_ids=position_ids)
    loss_cp = out_cp["logits"].sum()
    loss_cp.backward()

    logits_diff = (out_cp["logits"] - out_no_cp["logits"]).abs().max()
    assert logits_diff < 1e-2, f"Logits mismatch: {logits_diff}"

    for name, grad_no_cp in grads_no_cp.items():
        grad_cp = dict(model.named_parameters())[name].grad
        if grad_cp is not None:
            diff = (grad_cp - grad_no_cp).abs().max()
            assert diff < 1e-1, f"Grad mismatch for {name}: {diff}"


def test_mamba_cp_forward_backward_runs():
    """Verify CP forward + backward completes without error."""
    pg = _init_single_gpu_pg()

    config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )

    with torch.device("cuda"), default_dtype(torch.float32):
        model = NemotronHForCausalLM._from_config(config)

    inject_prime_lm_head(model, chunk_size=None)

    # Enable CP on Mamba layers
    for layer in model.model.layers:
        if isinstance(layer, NemotronHMambaLayer):
            layer.set_context_parallel_attributes(pg, cp_rank=0, cp_world_size=1)

    input_ids = torch.randint(0, 256, (1, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0)

    output = model(input_ids, position_ids=position_ids)
    output["logits"].sum().backward()

    # Verify all params got gradients
    zero_grads = []
    for name, p in model.named_parameters():
        if p.numel() == 0:
            continue
        if p.grad is None or p.grad.norm().item() == 0:
            zero_grads.append(name)
    assert not zero_grads, f"Parameters with zero/no gradients: {zero_grads}"
