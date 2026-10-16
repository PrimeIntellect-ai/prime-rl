import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models import get_custom_vlm_cls, supports_custom_impl
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerConfig, MuseGlimmerForCausalLM, MuseGlimmerTextConfig
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.vlm import get_language_model, get_layer_prefix


def _text_config(**overrides) -> MuseGlimmerTextConfig:
    values = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    values.update(overrides)
    return MuseGlimmerTextConfig(**values)


def test_config_defaults_match_checkpoint_layout():
    config = MuseGlimmerTextConfig()
    full_layers = [idx for idx, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
    assert full_layers == list(range(3, 52, 4))
    assert config.query_scale == pytest.approx(3.87)
    assert MuseGlimmerTextConfig(qk_scale_factor=43.7840518911).query_scale == pytest.approx(3.87, abs=1e-3)


def test_composite_config_roundtrip_keeps_vision_config():
    config = MuseGlimmerConfig(text_config=_text_config().to_dict(), vision_config={"hidden_size": 32})
    restored = MuseGlimmerConfig.from_dict(config.to_dict())
    assert isinstance(restored.text_config, MuseGlimmerTextConfig)
    assert restored.text_config.layer_types == config.text_config.layer_types
    assert restored.vision_config == {"hidden_size": 32}
    assert supports_custom_impl(restored)
    assert get_custom_vlm_cls(restored) is MuseGlimmerForCausalLM
    assert get_layer_prefix(restored) == "model.language_model.layers."


def test_state_dict_uses_checkpoint_names():
    config = MuseGlimmerConfig(text_config=_text_config(num_hidden_layers=1).to_dict())
    config._attn_implementation = "flash_attention_2"
    model = MuseGlimmerForCausalLM(config)
    assert len(get_language_model(model).layers) == 1
    layer = "model.language_model.layers.0"
    assert set(model.state_dict()) == {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        *(
            f"{layer}.{name}.weight"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "self_attn.gate_proj",
            )
        ),
    }


def _rms(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None, offset: float = 0.0) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * (weight.float() + offset)
    return out.type_as(x)


def _reference_logits(model: MuseGlimmerForCausalLM, input_ids: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    """Eager MuseGlimmer forward over packed documents, written from the architecture description."""
    config = model.config.text_config
    lm = model.model.language_model
    hidden = _rms(lm.embed_tokens(input_ids), config.rms_norm_eps)
    position_ids = torch.cat([torch.arange(n, device=input_ids.device) for n in seq_lens]).unsqueeze(0)
    cos, sin = lm.rotary_emb(hidden, position_ids)
    doc_ids = torch.repeat_interleave(torch.arange(len(seq_lens)), torch.tensor(seq_lens)).to(input_ids.device)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    causal = (positions[None, :] <= positions[:, None]) & (doc_ids[None, :] == doc_ids[:, None])
    heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim

    for layer_idx, layer in enumerate(lm.layers):
        attn = layer.self_attn
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        x = _rms(hidden, config.rms_norm_eps, layer.input_layernorm.weight, 1.0)
        q = _rms(attn.q_proj(x).view(1, -1, heads, head_dim), config.rms_norm_eps) * config.query_scale
        k = _rms(attn.k_proj(x).view(1, -1, kv_heads, head_dim), config.rms_norm_eps)
        v = attn.v_proj(x).view(1, -1, kv_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = causal
        if sliding:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            mask = causal & (positions[:, None] - positions[None, :] < config.sliding_window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, scale=head_dim**-0.5)
        out = out.transpose(1, 2).reshape(1, -1, heads * head_dim)
        out = attn.o_proj(torch.sigmoid(attn.gate_proj(x)) * out)
        hidden = hidden + _rms(out, config.post_norm_eps, layer.post_attention_layernorm.weight, 1.0)

        x = _rms(hidden, config.rms_norm_eps, layer.pre_feedforward_layernorm.weight, 1.0)
        out = layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x))
        hidden = hidden + _rms(out, config.post_norm_eps, layer.post_feedforward_layernorm.weight, 1.0)

    logits = (_rms(hidden, config.rms_norm_eps, lm.norm.weight) @ model.lm_head.weight.t()).float()
    logits = logits * config.output_multiplier
    cap = config.final_logit_softcapping
    return cap * torch.tanh(logits / cap)


@pytest.mark.gpu
def test_forward_and_backward_match_reference():
    torch.manual_seed(0)
    config = MuseGlimmerConfig(text_config=_text_config().to_dict())
    config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = MuseGlimmerForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                param.normal_(0.0, 0.1)
    model.init_buffers_post_meta()
    inject_prime_lm_head(model, chunk_size=None)

    seq_lens = [23, 17]
    input_ids = torch.randint(0, config.text_config.vocab_size, (1, sum(seq_lens)), device="cuda")
    position_ids = torch.cat([torch.arange(n) for n in seq_lens]).unsqueeze(0).cuda()
    out = model(input_ids=input_ids, position_ids=position_ids, seq_lens=torch.tensor(seq_lens, device="cuda"))
    logits = out["logits"].float()
    ref = _reference_logits(model, input_ids, seq_lens)
    torch.testing.assert_close(logits, ref, rtol=0, atol=5e-2)
    assert logits.abs().max() <= config.text_config.final_logit_softcapping

    grad_out = torch.randn_like(ref)
    (logits * grad_out).sum().backward()
    prime_grads = {name: p.grad.float() for name, p in model.named_parameters()}
    model.zero_grad()
    (ref * grad_out).sum().backward()
    for name, p in model.named_parameters():
        # bf16 compute on both sides: compare relative to the gradient's norm.
        rel_err = (prime_grads[name] - p.grad.float()).norm() / p.grad.float().norm()
        assert rel_err < 0.03, f"{name}: relative grad error {rel_err:.4f}"
