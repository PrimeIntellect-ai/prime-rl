from types import SimpleNamespace

from prime_rl.inference.patches import patch_gemma4_moe_lora_support


def test_patch_gemma4_moe_lora_support_installs_vllm_protocol_methods() -> None:
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

    patch_gemma4_moe_lora_support()

    assert Gemma4ForCausalLM.is_3d_moe_weight
    assert Gemma4ForConditionalGeneration.is_3d_moe_weight

    fake_causal = SimpleNamespace(
        config=SimpleNamespace(num_experts=2),
        num_redundant_experts=0,
        named_parameters=lambda: [],
    )
    mapping = Gemma4ForCausalLM.get_expert_mapping(fake_causal)

    assert ("experts.w13_", "experts.0.gate_proj.", 0, "w1") in mapping
    assert ("experts.w2_", "experts.1.down_proj.", 1, "w2") in mapping
    assert ("experts.w13_", "experts.1.up_proj.", 1, "w3") in mapping

    sentinel = [("param", "weight", 0, "w1")]
    fake_conditional = SimpleNamespace(language_model=SimpleNamespace(get_expert_mapping=lambda: sentinel))

    assert Gemma4ForConditionalGeneration.get_expert_mapping(fake_conditional) is sentinel
