"""GPU integration coverage for Qwen3 direct routing replay.

These tests require the native CUDA/FlashAttention/grouped-GEMM environment.
They are not a claim that full-model GPU replay has been validated on a CPU host.
"""

import copy

import pytest
import torch

from prime_rl.trainer.routing_replay import RoutingReplay

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3 integration requires a CUDA GPU"),
]


def make_model():
    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
    from prime_rl.trainer.models.layers.moe import MoE
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from prime_rl.utils.utils import default_dtype

    pytest.importorskip("flash_attn", reason="Qwen3 integration requires FlashAttention 2")
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        max_position_embeddings=128,
        norm_topk_prob=True,
        load_balance_coeff=None,
        attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM._from_config(config)
    # GroupedExperts owns raw parameters rather than nn.Linear modules.
    for module in model.modules():
        if isinstance(module, MoE):
            module.init_weights(0.02, buffer_device=torch.device("cuda"))
            module.router.gate.requires_grad_(False)
    inject_prime_lm_head(model, chunk_size=None)
    return model.train()


@pytest.mark.parametrize("checkpoint_mode", [None, "full", "selective"])
def test_qwen3_direct_replay_matches_detached_router_objective_and_checkpoint_gradients(checkpoint_mode):
    from prime_rl.configs.trainer import ActivationCheckpointConfig
    from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper

    torch.manual_seed(31)
    reference = make_model()
    actual = copy.deepcopy(reference)
    reference_moes = [layer.mlp for layer in reference.model.layers]
    actual_moes = [layer.mlp for layer in actual.model.layers]
    captured = {}

    def capture_detached(layer_idx):
        def hook(_module, _inputs, output):
            scores, ids, counts, confidence = output
            captured[layer_idx] = RoutingReplay(ids.detach().clone(), scores.detach().clone())
            # This defines the same constant-coefficient objective as direct replay,
            # including removal of the gate's hidden-state derivative.
            return scores.detach(), ids, counts, confidence

        return hook

    hooks = [moe.router.register_forward_hook(capture_detached(i)) for i, moe in enumerate(reference_moes)]
    input_ids = torch.randint(0, reference.config.vocab_size, (1, 16), device="cuda")
    seq_lens = torch.tensor([input_ids.shape[1]], device="cuda")
    expected = reference(input_ids=input_ids, seq_lens=seq_lens)["logits"]
    for hook in hooks:
        hook.remove()
    replay = RoutingReplay(
        torch.stack([captured[i].ids for i in range(len(reference_moes))], dim=1).unsqueeze(0),
        torch.stack([captured[i].weights for i in range(len(reference_moes))], dim=1).unsqueeze(0).requires_grad_(),
    )
    original = RoutingReplay(replay.ids.clone(), replay.weights.detach().clone())

    def fail_gate(*_args):
        raise AssertionError("direct replay must not enter the router module, including during recomputation")

    gate_hooks = [moe.router.register_forward_pre_hook(fail_gate) for moe in actual_moes]
    if checkpoint_mode is not None:
        wrapper = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode=checkpoint_mode))
        for i, layer in enumerate(actual.model.layers):
            actual.model.layers[i] = wrapper(layer)

    output = actual(input_ids=input_ids, seq_lens=seq_lens, routed_experts=replay)["logits"]
    counts_after_forward = [moe.tokens_per_expert.clone() for moe in actual_moes]
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
    output.float().square().mean().backward()
    expected.float().square().mean().backward()
    for hook in gate_hooks:
        hook.remove()

    torch.testing.assert_close(
        actual.model.embed_tokens.weight.grad, reference.model.embed_tokens.weight.grad, rtol=2e-2, atol=1e-5
    )
    assert actual.model.embed_tokens.weight.grad.abs().sum() > 0
    for reference_moe, actual_moe, counts in zip(reference_moes, actual_moes, counts_after_forward):
        for name in ("gate_proj", "up_proj", "down_proj"):
            actual_grad = getattr(actual_moe.experts, name).grad
            reference_grad = getattr(reference_moe.experts, name).grad
            torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-2, atol=1e-5)
            assert actual_grad is not None and actual_grad.abs().sum() > 0
        assert actual_moe.router.gate.weight.grad is None
        torch.testing.assert_close(actual_moe.tokens_per_expert, counts, rtol=0, atol=0)
        torch.testing.assert_close(actual_moe.tokens_per_expert, reference_moe.tokens_per_expert, rtol=0, atol=0)
        assert torch.isnan(actual_moe.routing_confidence_sum)
    assert replay.weights.grad is None
    torch.testing.assert_close(replay.ids, original.ids, rtol=0, atol=0)
    torch.testing.assert_close(replay.weights, original.weights, rtol=0, atol=0)


def test_qwen3_fsdp_preserves_fp32_coefficients_through_root_and_layer_boundaries(free_port):
    import torch.distributed as dist

    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.model import setup_fsdp, validate_full_router_replay_model
    from prime_rl.trainer.parallel_dims import ParallelDims

    if dist.is_initialized():
        pytest.skip("This single-rank integration test requires its own process group")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{free_port}", rank=0, world_size=1)
    router_hooks = []
    try:
        model = make_model()
        expected_router_state = {
            f"model.layers.{i}.mlp.router.gate.weight": layer.mlp.router.gate.weight.detach().cpu().clone()
            for i, layer in enumerate(model.model.layers)
        }
        config = ModelConfig(name="qwen3-moe-routing-replay-test", moe_router_dtype="float32")
        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)
        setup_fsdp(model, config, parallel_dims, preserve_routing_weights=True)
        validate_full_router_replay_model(model)

        def reject_router_forward(*_args):
            raise AssertionError("TRR must not enter the separately sharded frozen router")

        router_hooks = [
            layer.mlp.router.register_forward_pre_hook(reject_router_forward) for layer in model.model.layers
        ]
        ids = torch.arange(64, device="cuda").reshape(1, 16, 2, 2) % model.config.num_experts
        weights = torch.tensor([0.1, 0.9], device="cuda").expand_as(ids).clone().requires_grad_()
        pair = RoutingReplay(ids, weights)
        original = weights.detach().clone()
        tokens = torch.randint(0, model.config.vocab_size, (1, 16), device="cuda")
        output = model(input_ids=tokens, seq_lens=torch.tensor([16], device="cuda"), routed_experts=pair)["logits"]
        output.float().square().mean().backward()
        assert model.model.embed_tokens.weight.grad is not None
        assert pair.weights.grad is None
        assert pair.weights.dtype == torch.float32
        torch.testing.assert_close(pair.weights, original, rtol=0, atol=0)
        for layer in model.model.layers:
            assert layer.mlp.router.gate.weight.grad is None
        # Uncalled nested FSDP units must still own valid checkpoint parameters.
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        for name, expected in expected_router_state.items():
            torch.testing.assert_close(state[name].float(), expected.float(), rtol=0, atol=0)
    finally:
        for handle in router_hooks:
            handle.remove()
        dist.destroy_process_group()
