"""DeepSeek V4 checks that need no GPU, kept out of the `gpu`-marked whole-model module.

The dequantization math is a pure function over hand-built tensors, and the config checks only
ever construct a `DeepseekV4Config`. Neither needs CUDA, and a module-level `pytest.mark.gpu`
cannot be undone per test, so they live here and run in the CPU job.
"""

import json

import pytest
import torch
from huggingface_hub.errors import StrictDataclassClassValidationError

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_state_dict_, dequantize_weight

from .deepseek_v4_helpers import _MODEL


def test_deepseek_v4_config_rejects_a_foreign_layer_type():
    """V4's own attention vocabulary, not the generic one transformers checks against.

    `PretrainedConfig.validate_layer_type` runs first, from `super().__init__()`, and accepts
    anything in transformers' generic layer-type list; only `DeepseekV4Config`'s own override
    narrows that to the three V4 variants. `compress_rates` carries a rate for the foreign type
    on purpose, so `validate_architecture` cannot be what rejects it.
    """
    kwargs = _MODEL | {"layer_types": ["full_attention"] * 5, "compress_rates": {"full_attention": 4}}

    with pytest.raises(StrictDataclassClassValidationError, match="layer_types entries must be one of"):
        DeepseekV4Config(**kwargs)


def test_deepseek_v4_config_translates_legacy_compress_ratios():
    """Real checkpoints ship the V3-flavoured legacy `compress_ratios`/`num_hash_layers` schema
    instead of `layer_types`/`mlp_layer_types`, which is what prime-rl's model code reads, so the
    config has to translate between them. Loading the real checkpoint without this built the
    wrong per-layer attention schedule outright.
    """
    config = DeepseekV4Config(num_hidden_layers=6, compress_ratios=[0, 0, 4, 128, 4, 128], num_hash_layers=2)

    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config.mlp_layer_types == ["hash_moe", "hash_moe", "moe", "moe", "moe", "moe"]


def test_deepseek_v4_config_serializes_topk_method():
    """The saved `config.json` has to carry `topk_method`, which is what vLLM gates the
    `e_score_correction_bias` parameter on. An explicit value must win."""
    assert json.loads(DeepseekV4Config().to_json_string())["topk_method"] == "noaux_tc"
    assert json.loads(DeepseekV4Config(topk_method="greedy").to_json_string())["topk_method"] == "greedy"


def test_dequantize_weight_dense_fp8():
    """Dense fp8 case: one `float8_e8m0fnu` scale block covers the whole weight."""
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)  # byte 128 -> 2**(128-127) = 2.0

    result = dequantize_weight(weight, scale)

    assert result.dtype == torch.bfloat16
    assert torch.equal(result, torch.tensor([[2.0, 4.0], [-2.0, 1.0]], dtype=torch.bfloat16))


def test_dequantize_weight_packed_mxfp4():
    """Packed MXFP4 expert case: unpack two e2m1 nibbles per byte, then a per-block scale."""
    # Nibble layout per byte is (high << 4) | low; e2m1 LUT indices used here:
    # 2->1.0, 4->2.0, 10->-1.0, 6->4.0, 0->0.0, 7->6.0, 9->-0.5, 3->1.5.
    packed = torch.tensor(
        [
            [(4 << 4) | 2, (6 << 4) | 10],  # row 0 -> unpacks to [1.0, 2.0, -1.0, 4.0]
            [(7 << 4) | 0, (3 << 4) | 9],  # row 1 -> unpacks to [0.0, 6.0, -0.5, 1.5]
        ],
        dtype=torch.int8,
    )
    # [2, 2] scale grid over the unpacked [2, 4] weight -> block_rows=1, block_cols=2.
    scale = torch.tensor([[127, 128], [129, 126]], dtype=torch.uint8).view(torch.float8_e8m0fnu)

    result = dequantize_weight(packed, scale)

    expected = torch.tensor([[1.0, 2.0, -2.0, 8.0], [0.0, 24.0, -0.25, 0.75]], dtype=torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)


def test_dequantize_state_dict_pops_scale_and_leaves_other_keys_untouched():
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    routing = torch.tensor([0, 1, 2], dtype=torch.int64)
    plain = torch.randn(3, dtype=torch.bfloat16)
    state_dict = {
        "layers.0.attn.wq_a.weight": weight,
        "layers.0.attn.wq_a.scale": scale,
        "layers.0.ffn.gate.tid2eid": routing,
        "embed.weight": plain,
    }

    dequantize_state_dict_(state_dict)

    assert set(state_dict) == {"layers.0.attn.wq_a.weight", "layers.0.ffn.gate.tid2eid", "embed.weight"}
    assert state_dict["layers.0.attn.wq_a.weight"].dtype == torch.bfloat16
    assert torch.equal(
        state_dict["layers.0.attn.wq_a.weight"], torch.tensor([[2.0, 4.0], [-2.0, 1.0]], dtype=torch.bfloat16)
    )
    assert torch.equal(state_dict["layers.0.ffn.gate.tid2eid"], routing)
    assert torch.equal(state_dict["embed.weight"], plain)
