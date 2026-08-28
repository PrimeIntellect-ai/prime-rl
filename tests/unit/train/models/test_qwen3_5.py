import inspect
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM

from prime_rl.trainer.models.layers.attn import FlashAttention, substitute_ring_attn
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.gated_delta_net import Qwen3_5GatedDeltaNet
from prime_rl.trainer.models.qwen3_5.norm import Qwen3_5RMSNorm
from prime_rl.utils.cp import setup_model_cp


def _tiny_text_config(attn_impl: str = "flash_attention_2") -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
    )
    config._attn_implementation = attn_impl
    return config


def _tiny_vlm_config(attn_impl: str = "flash_attention_2") -> Qwen3_5Config:
    text_config = _tiny_text_config(attn_impl)
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=text_config.hidden_size,
    )
    config = Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = attn_impl
    config.text_config._attn_implementation = attn_impl
    return config


def _tiny_moe_config(attn_impl: str = "flash_attention_2") -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config._attn_implementation = attn_impl
    return config


@pytest.mark.gpu
def test_qwen3_5_dense_matches_hf_state_keys_on_meta():
    config = _tiny_text_config()
    with torch.device("meta"):
        config._attn_implementation = "eager"
        hf_model = HFQwen3_5ForCausalLM(config)
        config._attn_implementation = "flash_attention_2"
        prime_model = Qwen3_5ForCausalLM(config)

    assert set(prime_model.state_dict()) == set(hf_model.state_dict())
    for name, tensor in prime_model.state_dict().items():
        assert tensor.shape == hf_model.state_dict()[name].shape, name


def test_qwen3_5_full_attention_uses_custom_class():
    config = _tiny_text_config(attn_impl="flash_attention_3")
    with torch.device("meta"):
        model = Qwen3_5Model(config)

    assert isinstance(model.layers[1].self_attn, Qwen3_5Attention)
    assert model.config._attn_implementation == "flash_attention_3"
    assert "ALL_ATTENTION_FUNCTIONS" not in inspect.getsource(type(model.layers[1].self_attn).forward)


def test_qwen3_5_norms_remain_zero_centered_after_model_init():
    model = Qwen3_5ForCausalLM(_tiny_text_config())

    norms = [module for module in model.modules() if isinstance(module, Qwen3_5RMSNorm)]
    assert norms
    assert all(torch.count_nonzero(norm.weight) == 0 for norm in norms)


def test_qwen3_5_context_parallel_setup_chain_text_and_vlm():
    cp_group = MagicMock()

    text_model = Qwen3_5ForCausalLM(_tiny_text_config())
    linear_layer = text_model.model.layers[0]
    text_model.model.layers[0] = torch.nn.Sequential(linear_layer)
    setup_model_cp(text_model, cp_group, cp_rank=1, cp_world_size=2)
    assert text_model.model.context_parallel_group is cp_group
    assert text_model.model.context_parallel_rank == 1
    assert text_model.model.context_parallel_world_size == 2
    assert linear_layer.linear_attn.context_parallel_group is cp_group

    vlm_config = _tiny_vlm_config()
    vlm_config.vision_config._attn_implementation = "sdpa"
    vlm_config.vision_config._attn_implementation_internal = "sdpa"
    with torch.device("meta"):
        vlm_model = Qwen3_5ForCausalLM(vlm_config)
    setup_model_cp(vlm_model, cp_group, cp_rank=0, cp_world_size=2)
    assert vlm_model.model.language_model.context_parallel_group is cp_group
    assert vlm_model.model.language_model.layers[0].linear_attn.context_parallel_world_size == 2


@pytest.mark.gpu
def test_qwen3_5_gated_delta_net_context_parallel():
    if int(os.environ.get("WORLD_SIZE", 1)) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    try:
        torch.manual_seed(0)
        config = _tiny_text_config()
        config.linear_key_head_dim = 128
        config.linear_value_head_dim = 128
        config.linear_num_key_heads = 16
        config.linear_num_value_heads = 32
        reference = Qwen3_5GatedDeltaNet(config).cuda().to(torch.bfloat16)
        context_parallel = Qwen3_5GatedDeltaNet(config).cuda().to(torch.bfloat16)
        context_parallel.load_state_dict(reference.state_dict())
        context_parallel.set_context_parallel_attributes(dist.group.WORLD, world_size=2)

        hidden_states = torch.randn(1, 16, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        dist.broadcast(hidden_states, src=0)
        cu_seqlens = torch.tensor([0, 5, 11, 16], device="cuda", dtype=torch.int32)

        reference_input = hidden_states.detach().clone().requires_grad_()
        expected = reference(
            reference_input,
            cu_seqlens,
            cu_seqlens_are_pre_shard=False,
        )
        output_gradient = torch.randn_like(expected)
        dist.broadcast(output_gradient, src=0)
        expected.backward(output_gradient)

        local_slice = slice(local_rank * 8, (local_rank + 1) * 8)
        local_input = hidden_states[:, local_slice].detach().clone().requires_grad_()
        actual = context_parallel(
            local_input,
            cu_seqlens,
            cu_seqlens_are_pre_shard=True,
        )
        actual.backward(output_gradient[:, local_slice])

        gathered = [torch.empty_like(actual) for _ in range(2)]
        dist.all_gather(gathered, actual)
        torch.testing.assert_close(torch.cat(gathered, dim=1), expected, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            local_input.grad,
            reference_input.grad[:, local_slice],
            rtol=5e-2,
            atol=5e-2,
        )
    finally:
        dist.destroy_process_group()


def test_setup_model_cp_requires_hook_only_for_hybrid_models():
    class HybridLayer(torch.nn.Module):
        layer_type = "linear_attention"

    class Inner:
        layers = torch.nn.Sequential(torch.nn.Sequential(HybridLayer()))

    class HybridNoHookModel:
        model = Inner()

    with pytest.raises(ValueError, match="set_context_parallel_attributes"):
        setup_model_cp(HybridNoHookModel(), MagicMock(), cp_rank=0, cp_world_size=2)

    class SoftmaxOnlyModel:
        pass

    setup_model_cp(SoftmaxOnlyModel(), MagicMock(), cp_rank=0, cp_world_size=2)


def test_qwen3_5_ring_patches_dense_flash_attention():
    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    originals = {cls: cls._compute_attention for cls in (FlashAttention, AfmoeFlashAttention)}
    try:
        substitute_ring_attn(process_group=MagicMock(), heads_k_stride=1)
        assert Qwen3_5Attention._compute_attention is FlashAttention._compute_attention
        assert Qwen3_5Attention._compute_attention is not originals[FlashAttention]
    finally:
        for cls, method in originals.items():
            cls._compute_attention = method
