# This script should be run with "https://github.com/JJJYmmm/transformers.git" which checks against the (likely) merged transformers PR for Zaya
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import ZayaForCausalLM as HFZayaForCausalLM

# There is something wrong with the quack RMSNorm vs the FP32 implementation
import prime_rl.trainer.models.layers.norms as norms
norms._get_quack_rmsnorm = lambda: None

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.zaya import ZayaConfig
from prime_rl.trainer.models.zaya import ZayaForCausalLM as PrimeRLZayaForCausalLM
from prime_rl.trainer.weights import load_state_dict
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

LOGITS_ATOL = 2e-2
EMBED_GRAD_ATOL = 2


def _tiny_config(attn_implementation: str = "sdpa"):
    config = ZayaConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_hidden_size=16,
        num_hidden_layers=4,
        num_experts=3,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        norm_epsilon=1e-5,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        moe_router_topk=1,
        zaya_mlp_expansion=8,
        zaya_use_mod=True,
        zaya_use_eda=True,
        add_bias_linear=False,
        attention_bias=False,
        lm_head_bias=False,
        tie_word_embeddings=True,
        use_cache=False,
        use_grouped_mm=False,
    )
    config._attn_implementation = attn_implementation
    # HF `ZayaModel` uses `layer_types` and `rope_parameters[layer_type]`; Prime expects `rope_parameters["hybrid"]`.
    config.layer_types = ["hybrid"] * config.num_hidden_layers
    config.rope_parameters = {
        "hybrid": {
            "rope_type": "default",
            "rope_theta": float(config.rope_theta),
            "partial_rotary_factor": float(config.partial_rotary_factor),
        }
    }
    return config


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state_dict.items()}


def get_model_pairs(
    hf_attn_implementation: str = "sdpa",
    prime_attn_implementation: str | None = None,
    dtype: torch.dtype = torch.float32,
):
    if prime_attn_implementation is None:
        prime_attn_implementation = hf_attn_implementation
    hf_config = _tiny_config(hf_attn_implementation)
    prime_config = _tiny_config(prime_attn_implementation)

    with torch.device("cuda"), default_dtype(dtype):
        hf_model = HFZayaForCausalLM(hf_config)
        prime_model = PrimeRLZayaForCausalLM._from_config(prime_config)

    with torch.no_grad():
        state_dict = _clone_state_dict(hf_model.state_dict())
        prime_state_keys = set(prime_model.state_dict().keys())
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    assert prime_state_keys - set(state_dict.keys()) == set()
    return hf_model, prime_model


def _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids, attention_mask=None) -> None:
    hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    prime_output = prime_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    hf_output.logits.float().sum().backward()
    prime_output["logits"].float().sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=LOGITS_ATOL), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )

    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=EMBED_GRAD_ATOL), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


class _PassthroughPrimeZayaBlock(nn.Module):
    def forward(self, hidden_states, prev_router_hidden_states=None, routed_experts=None):
        return hidden_states, None, prev_router_hidden_states


class _PassthroughHfZayaMoe(nn.Module):
    """HF `ZayaSparseMoeBlock` returns `(hidden_states, prev_router_hidden_states)`."""

    def forward(self, hidden_states, prev_router_hidden_states=None):
        return hidden_states, prev_router_hidden_states


def test_zaya_attn_only() -> None:
    hf_model, prime_model = get_model_pairs()

    for layer in hf_model.model.layers:
        layer.mlp = _PassthroughHfZayaMoe()
    for layer in prime_model.model.layers:
        layer.mlp = _PassthroughPrimeZayaBlock()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)


def test_zaya_mlp_only() -> None:
    hf_model, prime_model = get_model_pairs()

    def identity_attn_hf(hidden_states, *args, **kwargs):
        return hidden_states, None

    def identity_attn_prime(
        hidden_states,
        *args,
        **kwargs,
    ):
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = identity_attn_hf
    for layer in prime_model.model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = identity_attn_prime

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)


@pytest.mark.slow
def test_zaya() -> None:
    snapshot = Path(snapshot_download(repo_id="JJJYmmm/ZAYA1-8B-HF", repo_type="model"))
    dtype = torch.bfloat16
    device = torch.device("cuda")

    hf_model = HFZayaForCausalLM.from_pretrained(str(snapshot), torch_dtype=dtype)
    hf_model.to(device)
    attn_impl = getattr(
        hf_model.config,
        "_attn_implementation",
        getattr(hf_model.config, "attn_implementation", "sdpa"),
    )
    prime_config = ZayaConfig.from_pretrained(snapshot)
    prime_config._attn_implementation = attn_impl

    prime_model = PrimeRLZayaForCausalLM._from_config(prime_config)
    sd = load_state_dict(snapshot)
    PrimeRLZayaForCausalLM.convert_to_prime(sd)
    prime_model.load_state_dict(sd, strict=False)

    prime_model.to(device=device, dtype=dtype)
    prime_model.eval()
    hf_model.eval()

    vocab = hf_model.config.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (4, 16), device=device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(4, -1)

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        prime_out = prime_model(input_ids=input_ids, position_ids=position_ids)

    hf_logits = hf_out.logits.float().cpu()
    prime_logits = prime_out["logits"].float().cpu()
    max_abs = (prime_logits - hf_logits).abs().max().item()

    assert torch.allclose(prime_logits, hf_logits, atol=5e-2), (
        f"Forward logits mismatch max abs diff {max_abs} (atol=5e-2)"
    )


def test_zaya_tiny_roundtrip() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFZayaForCausalLM(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_model.zero_grad(set_to_none=True)
    hf_from_prime_model.zero_grad(set_to_none=True)
    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    hf_from_prime_output = hf_from_prime_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    hf_output.logits.sum().backward()
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=LOGITS_ATOL), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=EMBED_GRAD_ATOL), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


def test_zaya_attention_mask() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -3:] = 0

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids, attention_mask)


def test_zaya_flash_attention_2() -> None:
    pytest.importorskip("flash_attn")
    torch.manual_seed(0)
    hf_model, prime_model = get_model_pairs(prime_attn_implementation="flash_attention_2", dtype=torch.bfloat16)

    with torch.device("cuda"):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 12))
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    _assert_logits_and_embed_grads_close(hf_model, prime_model, input_ids, position_ids)