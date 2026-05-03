import pytest
import torch

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeForCausalLM
from prime_rl.trainer.mtp import mtp_masks_from_label_mask, roll_tensor
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def _tiny_config(mtp_enabled: bool = False) -> Qwen3_5MoeConfig:
    config = Qwen3_5MoeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        use_grouped_mm=False,
        layer_types=["full_attention", "full_attention"],
        mtp_num_hidden_layers=1,
        prime_mtp_enabled=mtp_enabled,
    )
    config._attn_implementation = "sdpa"
    return config


def _tiny_model(mtp_enabled: bool = False) -> PrimeRLQwen3_5MoeForCausalLM:
    with torch.device("cuda"), default_dtype(torch.float32):
        model = PrimeRLQwen3_5MoeForCausalLM._from_config(_tiny_config(mtp_enabled))
    inject_prime_lm_head(model, chunk_size=None)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    return model


def test_roll_tensor_does_not_cross_packed_boundaries():
    values = torch.tensor([[10, 11, 12, 20, 21]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    rolled = roll_tensor(values, position_ids=position_ids, fill_value=0)

    assert torch.equal(rolled, torch.tensor([[11, 12, 0, 21, 0]]))


def test_mtp_mask_intersection_uses_intermediate_and_target_tokens():
    label_mask = torch.tensor([[True, False, True, True, False]])
    masks = list(mtp_masks_from_label_mask(label_mask, position_ids=None, num_depths=1))

    assert torch.equal(masks[0], torch.tensor([[False, False, True, False, False]]))


def test_qwen_mtp_disabled_by_default():
    model = _tiny_model(mtp_enabled=False)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8), device="cuda")
    labels = roll_tensor(input_ids)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

    out = model(input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask)

    assert "mtp_loss" not in out
    assert len(model.mtp_layers) == 1
    assert model.mtp_layers[0].block.layer_type == "full_attention"


def test_qwen_mtp_loss_only_updates_mtp_params():
    model = _tiny_model(mtp_enabled=True)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8), device="cuda")
    labels = roll_tensor(input_ids)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

    out = model(input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask)
    assert out["mtp_loss"].isfinite()
    out["mtp_loss"].backward()

    assert model.model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert all(param.grad is None for param in model.model.layers.parameters())
    assert any(param.grad is not None for param in model.mtp_layers.parameters())


def test_qwen_mtp_conversion_roundtrip_preserves_hf_keys():
    model = _tiny_model(mtp_enabled=False)
    state_dict = model.state_dict()
    hf_state_dict = PrimeRLQwen3_5MoeForCausalLM.convert_to_hf(dict(state_dict))
    assert "mtp.fc.weight" in hf_state_dict
    assert "mtp.layers.0.self_attn.q_proj.weight" in hf_state_dict
    assert not any(key.startswith("mtp_layers.") for key in hf_state_dict)

    roundtripped = dict(hf_state_dict)
    PrimeRLQwen3_5MoeForCausalLM.convert_to_prime(roundtripped)
    PrimeRLQwen3_5MoeForCausalLM.convert_to_hf(roundtripped)

    for key, value in hf_state_dict.items():
        assert key in roundtripped
        assert torch.allclose(value, roundtripped[key], equal_nan=True), key
