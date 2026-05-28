import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

from prime_rl.trainer.model import can_reinit_empty_buffers
from prime_rl.trainer.models import (
    AutoModelForCausalLMPrimeRL,
    get_custom_vlm_cls,
    supports_custom_impl,
)
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens, set_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGemma4TextExperts
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.vlm import get_language_model, get_vision_encoder, is_vlm_architecture


def _tiny_text_config(
    *,
    enable_moe: bool = False,
    hidden_size_per_layer_input: int = 4,
    use_bidirectional_attention: str | None = None,
) -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        final_logit_softcapping=30.0,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        num_global_key_value_heads=2,
        global_head_dim=8,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        use_bidirectional_attention=use_bidirectional_attention,
        enable_moe_block=enable_moe,
        num_experts=4 if enable_moe else None,
        top_k_experts=2 if enable_moe else None,
        moe_intermediate_size=16 if enable_moe else None,
    )


def _tiny_vision_config() -> Gemma4VisionConfig:
    return Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        pooling_kernel_size=1,
        patch_size=2,
        position_embedding_size=16,
        use_clipped_linears=True,
        standardize=False,
    )


def _tiny_vlm_config(
    *,
    enable_moe: bool = False,
    use_bidirectional_attention: str | None = None,
) -> Gemma4Config:
    return Gemma4Config(
        text_config=_tiny_text_config(
            enable_moe=enable_moe,
            use_bidirectional_attention=use_bidirectional_attention,
        ),
        vision_config=_tiny_vision_config(),
        audio_config=None,
        image_token_id=60,
        audio_token_id=61,
        video_token_id=62,
        boi_token_id=57,
        boa_token_id=58,
        eoi_token_id=59,
        eoa_token_index=63,
    )


def test_gemma4_custom_impl_registered_for_text_and_vlm() -> None:
    from prime_rl.trainer.models.gemma4 import (
        Gemma4ForCausalLM as PrimeRLGemma4ForCausalLM,
    )
    from prime_rl.trainer.models.gemma4 import (
        Gemma4ForConditionalGeneration as PrimeRLGemma4ForConditionalGeneration,
    )

    text_config = _tiny_text_config()
    vlm_config = _tiny_vlm_config()

    assert supports_custom_impl(text_config)
    assert isinstance(AutoModelForCausalLMPrimeRL.from_config(text_config), PrimeRLGemma4ForCausalLM)
    assert is_vlm_architecture(vlm_config)
    assert get_custom_vlm_cls(vlm_config) is PrimeRLGemma4ForConditionalGeneration


def test_gemma4_text_forward_matches_hf_logits() -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as HFGemma4ForCausalLM

    from prime_rl.trainer.models.gemma4 import (
        Gemma4ForCausalLM as PrimeRLGemma4ForCausalLM,
    )

    torch.manual_seed(0)
    config = _tiny_text_config()
    with default_dtype(torch.float32):
        hf_model = HFGemma4ForCausalLM._from_config(config)
        prime_model = PrimeRLGemma4ForCausalLM._from_config(config)

    with torch.no_grad():
        prime_model.load_state_dict(hf_model.state_dict())
        inject_prime_lm_head(prime_model, chunk_size=None)

    input_ids = torch.randint(3, config.vocab_size - 1, (1, 11))
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with torch.no_grad():
        hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
        prime_output = prime_model(input_ids=input_ids, position_ids=position_ids)

    assert torch.allclose(prime_output["logits"], hf_output.logits, atol=1e-5, rtol=1e-5)


def test_gemma4_vlm_text_only_forward_uses_language_model_registry() -> None:
    from prime_rl.trainer.models.gemma4 import (
        Gemma4ForConditionalGeneration as PrimeRLGemma4ForConditionalGeneration,
    )

    config = _tiny_vlm_config(use_bidirectional_attention="vision")
    with default_dtype(torch.float32):
        model = PrimeRLGemma4ForConditionalGeneration._from_config(config)
    inject_prime_lm_head(model, chunk_size=None)

    input_ids = torch.randint(3, config.text_config.vocab_size - 4, (1, 9))
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    out = model(input_ids=input_ids, position_ids=position_ids)

    assert out["logits"].shape == (1, input_ids.shape[1], config.text_config.vocab_size)
    assert get_language_model(model) is model.model.language_model
    assert get_vision_encoder(model) is model.model.vision_tower


def test_gemma4_meta_device_and_buffer_reinit() -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    from prime_rl.trainer.models.gemma4 import (
        Gemma4ForConditionalGeneration as PrimeRLGemma4ForConditionalGeneration,
    )

    config = _tiny_vlm_config()
    with torch.device("meta"):
        model = PrimeRLGemma4ForConditionalGeneration.from_config(config)

    assert can_reinit_empty_buffers(model)

    model.to_empty(device="cpu")
    model.init_buffers_post_meta()

    text_rotary = model.model.language_model.rotary_emb
    vision_rotary = model.model.vision_tower.encoder.rotary_emb
    clippable = next(
        module for module in model.modules() if isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears
    )
    assert text_rotary.full_attention_inv_freq.device.type == "cpu"
    assert text_rotary.sliding_attention_inv_freq.abs().sum() > 0
    assert vision_rotary.inv_freq.abs().sum() > 0
    assert {"input_min", "input_max", "output_min", "output_max"}.issubset(clippable._non_persistent_buffers_set)
    assert torch.isneginf(clippable.input_min)
    assert torch.isposinf(clippable.input_max)
    assert torch.isneginf(clippable.output_min)
    assert torch.isposinf(clippable.output_max)
    assert torch.isclose(
        model.model.language_model.embed_tokens.embed_scale,
        torch.tensor(config.text_config.hidden_size**0.5),
    )


def test_gemma4_moe_lora_matches_base_when_zero_initialized() -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    torch.manual_seed(0)
    config = _tiny_text_config(enable_moe=True, hidden_size_per_layer_input=0)
    base = Gemma4TextExperts(config)
    with torch.no_grad():
        torch.nn.init.normal_(base.gate_up_proj, mean=0.0, std=0.02)
        torch.nn.init.normal_(base.down_proj, mean=0.0, std=0.02)
    set_lora_num_tokens(None, reset_reference=True)
    set_multilora_scaling(None, reset_reference=True)
    set_lora_num_tokens(torch.tensor([6], dtype=torch.int32), reset_reference=True)
    set_multilora_scaling(torch.tensor([2.0], dtype=torch.float32), reset_reference=True)
    wrapped = MultiLoRAGemma4TextExperts(base, rank=2, n_adapters=1)

    hidden_states = torch.randn(6, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
    top_k_weights = torch.full((6, 2), 0.5)

    with torch.no_grad():
        base_out = base(hidden_states, top_k_index, top_k_weights)
        wrapped_out = wrapped(hidden_states, top_k_index, top_k_weights)

    assert torch.allclose(wrapped_out, base_out, atol=1e-6, rtol=1e-6)


def test_gemma4_moe_lora_exports_vllm_3d_adapter_keys() -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    config = _tiny_text_config(enable_moe=True, hidden_size_per_layer_input=0)
    base = Gemma4TextExperts(config)
    set_lora_num_tokens(None, reset_reference=True)
    set_multilora_scaling(None, reset_reference=True)
    set_lora_num_tokens(torch.tensor([1], dtype=torch.int32), reset_reference=True)
    set_multilora_scaling(torch.tensor([1.0], dtype=torch.float32), reset_reference=True)
    wrapped = MultiLoRAGemma4TextExperts(base, rank=2, n_adapters=1)

    adapter_state = wrapped.state_dict_for_adapter(0)

    assert adapter_state["base_layer.lora_A.weight"].shape == (config.num_experts * 2, config.hidden_size)
    assert adapter_state["base_layer.lora_B.weight"].shape == (
        2 * config.moe_intermediate_size,
        2 * config.num_experts,
    )
    assert adapter_state["lora_A.weight"].shape == (config.num_experts * 2, config.moe_intermediate_size)
    assert adapter_state["lora_B.weight"].shape == (config.hidden_size, 2 * config.num_experts)


def test_gemma4_adapter_converter_inserts_vllm_moe_prefix() -> None:
    from prime_rl.trainer.models.gemma4 import (
        Gemma4ForConditionalGeneration as PrimeRLGemma4ForConditionalGeneration,
    )

    state = {
        "model.language_model.layers.0.experts.base_layer.lora_A.weight": torch.empty(8, 32),
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.empty(2, 32),
    }

    converted = PrimeRLGemma4ForConditionalGeneration.convert_adapter_to_hf(state)

    assert "model.language_model.layers.0.moe.experts.base_layer.lora_A.weight" in converted
    assert "model.language_model.layers.0.self_attn.q_proj.lora_A.weight" in converted
