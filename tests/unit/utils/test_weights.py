import pytest
import torch

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4ForCausalLM
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaForCausalLM
from prime_rl.trainer.models.nemotron_h import NemotronHForCausalLM
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from prime_rl.utils.weights import resolve_wire_dtype


def test_resolve_wire_dtype():
    def keep_in_fp32(key: str) -> bool:
        return key.endswith("A_log")

    assert resolve_wire_dtype(keep_in_fp32, "layers.0.linear_attn.A_log", torch.bfloat16) is torch.float32
    assert resolve_wire_dtype(keep_in_fp32, "layers.0.self_attn.q_proj.weight", torch.bfloat16) is torch.bfloat16

    # The default is whatever the transport asked for, not hardcoded bf16.
    assert resolve_wire_dtype(keep_in_fp32, "layers.0.self_attn.q_proj.weight", torch.float8_e4m3fn) is (
        torch.float8_e4m3fn
    )

    # A plain transformers model declares nothing, so every key takes the default.
    assert resolve_wire_dtype(None, "layers.0.linear_attn.A_log", torch.bfloat16) is torch.bfloat16


@pytest.mark.parametrize(
    ("model_cls", "fp32_key", "ordinary_key"),
    [
        (NemotronHForCausalLM, "model.layers.0.mamba.A_log", "model.layers.0.mamba.in_proj.weight"),
        (Qwen3_5ForCausalLM, "model.layers.0.linear_attn.A_log", "model.layers.0.self_attn.q_proj.weight"),
        (Qwen3_5MoeForCausalLM, "model.layers.0.linear_attn.norm.weight", "model.layers.0.mlp.experts.gate_proj"),
        (Glm4MoeForCausalLM, "model.layers.0.mlp.router.selection_bias", "model.layers.0.self_attn.q_proj.weight"),
        (GlmMoeDsaForCausalLM, "model.layers.0.mlp.router.selection_bias", "model.layers.0.self_attn.q_proj.weight"),
        (DeepseekV4ForCausalLM, "model.layers.0.attn_hc.fn", "model.layers.0.self_attn.q_b_proj.weight"),
    ],
)
def test_declared_keys_travel_in_fp32(model_cls, fp32_key, ordinary_key):
    keep_in_fp32 = model_cls.keep_in_fp32_for_weight_transfer

    assert resolve_wire_dtype(keep_in_fp32, fp32_key, torch.bfloat16) is torch.float32
    assert resolve_wire_dtype(keep_in_fp32, ordinary_key, torch.bfloat16) is torch.bfloat16
