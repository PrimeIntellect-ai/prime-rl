import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import resolve_auto_attn
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.models.fusions import apply_model_fusions
from prime_rl.trainer.models.layers.attn import FlashAttention, substitute_ring_attn
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.gated_delta_net import Qwen3_5GatedDeltaNet
from prime_rl.trainer.models.qwen3_5.norm import Qwen3_5RMSNorm
from prime_rl.utils.cp import setup_model_cp


def get_text_config(config_cls=Qwen3_5TextConfig) -> Qwen3_5TextConfig:
    moe_config = {}
    if config_cls is Qwen3_5MoeTextConfig:
        moe_config = dict(
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
        )
    return config_cls(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        **moe_config,
    )


@pytest.fixture(params=[Qwen3_5TextConfig, Qwen3_5MoeTextConfig], ids=["dense", "moe"])
def text_config(request):
    return get_text_config(request.param)


def get_vlm_config(text_config) -> Qwen3_5Config:
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=text_config.hidden_size,
    )
    config_cls = Qwen3_5MoeConfig if isinstance(text_config, Qwen3_5MoeTextConfig) else Qwen3_5Config
    return config_cls(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )


def get_model(config, device="cuda"):
    runtime_config = ModelConfig()
    resolve_auto_attn(runtime_config)
    with torch.device(device):
        model = AutoModelForCausalLMPrimeRL.from_config(
            config, attn_implementation=runtime_config.attn, dtype=torch.bfloat16
        )
    inject_prime_lm_head(model, chunk_size=None)
    return model


@pytest.mark.gpu
def test_norms_remain_zero_centered_after_model_init(text_config):
    model = get_model(text_config)

    norms = [module for module in model.modules() if isinstance(module, Qwen3_5RMSNorm)]
    assert norms
    assert all(torch.count_nonzero(norm.weight) == 0 for norm in norms)


@pytest.mark.gpu
def test_context_parallel_setup_chain_text_and_vlm(text_config):
    cp_group = MagicMock()

    text_model = get_model(text_config, device="meta")
    linear_layer = text_model.model.layers[0]
    text_model.model.layers[0] = torch.nn.Sequential(linear_layer)
    setup_model_cp(text_model, cp_group, cp_rank=1, cp_world_size=2)
    assert text_model.model.context_parallel_group is cp_group
    assert text_model.model.context_parallel_rank == 1
    assert text_model.model.context_parallel_world_size == 2
    assert linear_layer.linear_attn.context_parallel_group is cp_group
    assert linear_layer.linear_attn.context_parallel_world_size == 2

    vlm_model = get_model(get_vlm_config(text_config), device="meta")
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
        config = get_text_config()
        config.hidden_size = 64
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
        )
        output_gradient = torch.randn_like(expected)
        dist.broadcast(output_gradient, src=0)
        expected.backward(output_gradient)

        local_slice = slice(local_rank * 8, (local_rank + 1) * 8)
        local_input = hidden_states[:, local_slice].detach().clone().requires_grad_()
        actual = context_parallel(
            local_input,
            cu_seqlens,
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


def test_ring_patches_flash_attention():
    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    originals = {cls: cls._compute_attention for cls in (FlashAttention, AfmoeFlashAttention)}
    try:
        substitute_ring_attn(process_group=MagicMock(), heads_k_stride=1)
        assert Qwen3_5Attention._compute_attention is FlashAttention._compute_attention
        assert Qwen3_5Attention._compute_attention is not originals[FlashAttention]
    finally:
        for cls, method in originals.items():
            cls._compute_attention = method


@pytest.mark.gpu
def test_forward_backward_and_packing(text_config):
    prime_model = get_model(text_config)
    fusions = ["qkv", "gate_up"] if isinstance(text_config, Qwen3_5MoeTextConfig) else ["qkv"]
    apply_model_fusions(prime_model, fusions)
    input_ids = torch.randint(0, prime_model.config.vocab_size, (1, 100), device="cuda")
    position_ids = torch.arange(1, 101, device="cuda").unsqueeze(0)
    prime_output = prime_model(
        input_ids,
        position_ids=position_ids,
        seq_lens=torch.tensor([input_ids.shape[1]], device="cuda"),
    )
    prime_output["logits"].sum().backward()
    assert torch.isfinite(prime_output["logits"]).all()
    assert torch.isfinite(prime_model.model.embed_tokens.weight.grad).all()

    packed_position_ids = torch.arange(1, 51, device="cuda").repeat(2).unsqueeze(0)
    # Keep expert selection fixed to isolate packed sequence boundaries.
    config = prime_model.config
    routed_experts = None
    if isinstance(config, Qwen3_5MoeTextConfig):
        routed_experts = (
            torch.rand(1, 100, config.num_hidden_layers, config.num_experts, device="cuda")
            .topk(config.num_experts_per_tok, dim=-1)
            .indices
        )
    with torch.no_grad():
        packed = prime_model(
            input_ids,
            position_ids=packed_position_ids,
            seq_lens=torch.tensor([50, 50], device="cuda"),
            routed_experts=routed_experts,
        )["logits"]
        unpacked = torch.cat(
            [
                prime_model(
                    input_ids[:, start : start + 50],
                    position_ids=packed_position_ids[:, :50],
                    seq_lens=torch.tensor([50], device="cuda"),
                    routed_experts=routed_experts[:, start : start + 50] if routed_experts is not None else None,
                )["logits"]
                for start in (0, 50)
            ],
            dim=1,
        )
    torch.testing.assert_close(packed, unpacked, atol=0.03, rtol=0.01)


@pytest.mark.gpu
def test_moe_router_replay():
    """When routed_experts are provided, the model uses them instead of computing routing."""
    prime_model = get_model(get_text_config(Qwen3_5MoeTextConfig))

    with torch.device("cuda"):
        input_ids = torch.randint(0, prime_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    seq_lens = torch.tensor([input_ids.shape[1]], device="cuda")
    out_normal = prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)

    num_layers = prime_model.config.num_hidden_layers
    topk = prime_model.config.num_experts_per_tok
    routed_experts = torch.randint(0, prime_model.config.num_experts, (1, 100, num_layers, topk), device="cuda")

    prime_model.zero_grad()
    out_replay = prime_model(
        input_ids,
        position_ids=position_ids,
        routed_experts=routed_experts,
        seq_lens=seq_lens,
    )

    assert out_replay["logits"].shape == out_normal["logits"].shape

    out_replay["logits"].sum().backward()
    assert prime_model.model.embed_tokens.weight.grad is not None
