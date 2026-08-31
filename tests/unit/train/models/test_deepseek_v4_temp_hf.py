"""HF-oracle parity checks for the DeepSeek V4 modules, one mechanism at a time.

These tests use `transformers.models.deepseek_v4` as the correctness oracle. That package only
exists from transformers 5.15, and the repo pins an older version so the DS V4 work does not
drag an unrelated dependency bump along with it. Run them explicitly:

    uv run --with 'transformers==5.15.0' pytest tests/unit/train/models/test_deepseek_v4_temp_hf.py -v

Under the pinned version the module skips rather than erroring. Everything these tests share
with the HF-free half lives in `deepseek_v4_temp_helpers.py`.
"""

import pytest

pytest.importorskip("transformers.models.deepseek_v4")

import torch
from torch import nn
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention as HFDeepseekV4Attention,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection as HFDeepseekV4HyperConnection,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperHead as HFDeepseekV4HyperHead,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RotaryEmbedding as HFDeepseekV4RotaryEmbedding,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4SparseMoeBlock as HFDeepseekV4SparseMoeBlock,
)

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, build_sliding_window_mask
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
from prime_rl.utils.utils import default_dtype

from .deepseek_v4_temp_helpers import (
    _ATTN,
    _BASE,
    _BATCH,
    _CSA_LAYER,
    _HASH_LAYER,
    _HASH_MOE,
    _HCA_LAYER,
    _MOE,
    _SEQ,
    _SINGLE_DOC,
    _SLIDING_LAYER,
    _hidden_states,
    _hidden_streams,
    _input_ids,
    _moe_hidden_states,
    _packed_context,
    _position_embeddings,
    _position_ids,
    _randomize,
    _randomize_attention,
    _seed_rng,  # noqa: F401 -- pytest fixture, applied by name
    _tid2eid,
    _torch_rms_norm,  # noqa: F401 -- pytest fixture, applied by name
    prime_attention_config,
)

pytestmark = [pytest.mark.gpu]


def _sync(hf_module: nn.Module, prime_module: nn.Module) -> None:
    hf_state = hf_module.state_dict()
    assert set(hf_state) == set(prime_module.state_dict()), "prime-rl and HF parameter names must match exactly"
    prime_module.load_state_dict(hf_state)


def _compare_grads(hf_module: nn.Module, prime_module: nn.Module, rtol: float = 0, atol: float = 0) -> None:
    prime_grads = dict(prime_module.named_parameters())
    for name, hf_param in hf_module.named_parameters():
        prime_grad = prime_grads[name].grad
        # The Lightning Indexer's parameters reach the loss only through integer top-k
        # indices, so both implementations must agree that they get no gradient at all.
        if hf_param.grad is None:
            assert prime_grad is None, f"{name} received a gradient in prime-rl but not in HF"
            continue
        assert prime_grad is not None, f"{name} received no gradient"
        torch.testing.assert_close(prime_grad, hf_param.grad, rtol=rtol, atol=atol, msg=lambda m, n=name: f"{n}: {m}")


def _hyper_connection_pair():
    hf_config = HFDeepseekV4Config(**_BASE)
    prime_config = DeepseekV4Config(**_BASE)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_module = HFDeepseekV4HyperConnection(hf_config)
        prime_module = DeepseekV4HyperConnection(prime_config)
    _randomize(hf_module)
    _sync(hf_module, prime_module)
    return hf_module, prime_module


def _hyper_head_pair():
    hf_config = HFDeepseekV4Config(**_BASE)
    prime_config = DeepseekV4Config(**_BASE)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_module = HFDeepseekV4HyperHead(hf_config)
        prime_module = DeepseekV4HyperHead(prime_config)
    _randomize(hf_module)
    _sync(hf_module, prime_module)
    return hf_module, prime_module


def test_hyperconnection_matches_hf():
    hf_module, prime_module = _hyper_connection_pair()
    hf_input, prime_input = _hidden_streams()

    hf_post, hf_comb, hf_collapsed = hf_module(hf_input)
    prime_post, prime_comb, prime_collapsed = prime_module(prime_input)

    torch.testing.assert_close(prime_post, hf_post, rtol=0, atol=0)
    torch.testing.assert_close(prime_comb, hf_comb, rtol=0, atol=0)
    torch.testing.assert_close(prime_collapsed, hf_collapsed, rtol=0, atol=0)

    # `comb` is doubly stochastic, so an unweighted sum has a near-constant gradient;
    # weighting the outputs keeps every parameter's gradient informative.
    with torch.device("cuda"):
        post_weight = torch.randn_like(hf_post)
        comb_weight = torch.randn_like(hf_comb)
        collapsed_weight = torch.randn_like(hf_collapsed)

    def loss(post, comb, collapsed):
        return (post * post_weight).sum() + (comb * comb_weight).sum() + (collapsed * collapsed_weight).sum()

    loss(hf_post, hf_comb, hf_collapsed).backward()
    loss(prime_post, prime_comb, prime_collapsed).backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


def test_hyperhead_matches_hf():
    hf_module, prime_module = _hyper_head_pair()
    hf_input, prime_input = _hidden_streams()

    hf_output = hf_module(hf_input)
    prime_output = prime_module(prime_input)

    assert prime_output.shape == (_BATCH, _SEQ, _BASE["hidden_size"])
    torch.testing.assert_close(prime_output, hf_output, rtol=0, atol=0)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


def _attention_configs() -> tuple[HFDeepseekV4Config, DeepseekV4Config]:
    hf_config = HFDeepseekV4Config(**_ATTN)
    # Force the eager path so HF actually runs its sink softmax, not an SDPA kernel.
    hf_config._attn_implementation = "eager"
    return hf_config, prime_attention_config()


def _attention_pair(layer_idx: int = _SLIDING_LAYER) -> tuple[nn.Module, nn.Module]:
    hf_config, prime_config = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_module = HFDeepseekV4Attention(hf_config, layer_idx=layer_idx)
        prime_module = DeepseekV4Attention(prime_config, layer_idx=layer_idx)
    _randomize_attention(hf_module)
    _sync(hf_module, prime_module)
    return hf_module, prime_module


def test_rotary_matches_hf():
    hf_config, prime_config = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_rotary = HFDeepseekV4RotaryEmbedding(hf_config)
        prime_rotary = DeepseekV4RotaryEmbedding(prime_config)
        probe = torch.zeros(_BATCH, _SEQ, _ATTN["hidden_size"])
    position_ids = _position_ids()

    rope_dim = int(_ATTN["head_dim"] * _ATTN["partial_rotary_factor"])
    for rope_type in ("main", "compress"):
        hf_cos, hf_sin = hf_rotary(probe, position_ids, rope_type)
        prime_cos, prime_sin = prime_rotary(probe, position_ids, rope_type)
        # Interleaved RoPE needs one theta per pair, so cos/sin come out at half width.
        assert prime_cos.shape == (_BATCH, _SEQ, rope_dim // 2)
        torch.testing.assert_close(prime_cos, hf_cos, rtol=0, atol=0)
        torch.testing.assert_close(prime_sin, hf_sin, rtol=0, atol=0)

    # The two rope types differ only in their base, so their tables must not coincide.
    assert not torch.equal(prime_rotary.main_inv_freq, prime_rotary.compress_inv_freq)


def test_rotary_matches_hf_yarn():
    """Real checkpoints YaRN-scale the compress branch via a legacy flat `rope_scaling` dict."""
    attn_yarn = _ATTN | {
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        }
    }
    hf_config = HFDeepseekV4Config(**attn_yarn)
    prime_config = DeepseekV4Config(**attn_yarn)

    assert prime_config.rope_parameters["compress"]["rope_type"] == "yarn"
    assert prime_config.rope_parameters["compress"]["attention_factor"] == 1.0
    assert prime_config.rope_parameters["main"]["rope_type"] == "default"

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_rotary = HFDeepseekV4RotaryEmbedding(hf_config)
        prime_rotary = DeepseekV4RotaryEmbedding(prime_config)
        probe = torch.zeros(_BATCH, _SEQ, _ATTN["hidden_size"])
    position_ids = _position_ids()

    for rope_type in ("main", "compress"):
        hf_cos, hf_sin = hf_rotary(probe, position_ids, rope_type)
        prime_cos, prime_sin = prime_rotary(probe, position_ids, rope_type)
        torch.testing.assert_close(prime_cos, hf_cos, rtol=0, atol=0)
        torch.testing.assert_close(prime_sin, hf_sin, rtol=0, atol=0)


@pytest.mark.parametrize("doc_lens", [(_SEQ,), (7, 9)], ids=["single_document", "packed"])
def test_sliding_window_mask_matches_hf(doc_lens):
    """Upstream is the oracle for the local window, packed or not.

    HF recovers document boundaries from `position_ids` restarts and prime-rl is told them as
    `seq_lens`, so the two agree wherever the restarts line up with the boundaries. They do here,
    and on a padded micro-batch they would not: `pad_micro_batch` restarts `position_ids` for the
    padding while folding it into `seq_lens[-1]`.
    """
    hf_config, _ = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        probe = torch.zeros(_BATCH, _SEQ, _ATTN["hidden_size"])
    position_ids = torch.cat([torch.arange(length, device="cuda") for length in doc_lens])
    hf_mask = create_sliding_window_causal_mask(
        config=hf_config,
        inputs_embeds=probe,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids.unsqueeze(0).expand(_BATCH, -1),
        allow_is_causal_skip=False,
    )
    cu_seqlens, _ = get_cu_seqlens_from_seq_lens(torch.tensor(doc_lens, device="cuda"), total_tokens=_SEQ)
    prime_mask = build_sliding_window_mask(
        _SEQ, _ATTN["sliding_window"], torch.bfloat16, torch.device("cuda"), cu_seqlens=cu_seqlens
    )

    assert prime_mask.shape == (1, 1, _SEQ, _SEQ)
    torch.testing.assert_close(prime_mask.expand_as(hf_mask), hf_mask, rtol=0, atol=0)


def test_sliding_attention_matches_hf(_torch_rms_norm):  # noqa: F811
    hf_module, prime_module = _attention_pair()
    hf_input, prime_input = _hidden_states()
    position_embeddings = _position_embeddings()
    packed = _packed_context(prime_module, _SINGLE_DOC, torch.bfloat16)

    hf_output, _ = hf_module(
        hf_input,
        position_embeddings=position_embeddings,
        position_ids=packed.position_ids,
        attention_mask=packed.attention_mask,
    )
    prime_output, prime_weights = prime_module(
        prime_input,
        position_embeddings=position_embeddings,
        packed=packed,
    )

    assert prime_weights is None, "prime-rl attention modules return `None` for attn_weights"
    assert prime_output.shape == (_BATCH, _SEQ, _ATTN["hidden_size"])
    torch.testing.assert_close(prime_output, hf_output, rtol=0, atol=0)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


def test_csa_attention_matches_hf(_torch_rms_norm):  # noqa: F811
    hf_module, prime_module = _attention_pair(_CSA_LAYER)
    hf_input, prime_input = _hidden_states()
    position_embeddings = _position_embeddings()
    packed = _packed_context(prime_module, _SINGLE_DOC, torch.bfloat16)

    hf_output, _ = hf_module(
        hf_input,
        position_embeddings=position_embeddings,
        position_ids=packed.position_ids,
        # HF concatenates the compressor's per-batch block bias onto the mask, so the local
        # window mask has to carry a batch dimension of its own here.
        attention_mask=packed.attention_mask.expand(_BATCH, 1, _SEQ, _SEQ),
    )
    prime_output, _ = prime_module(
        prime_input,
        position_embeddings=position_embeddings,
        packed=packed,
    )

    assert prime_output.shape == (_BATCH, _SEQ, _ATTN["hidden_size"])
    torch.testing.assert_close(prime_output, hf_output, rtol=0, atol=0)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


def test_hca_attention_matches_hf(_torch_rms_norm):  # noqa: F811
    hf_module, prime_module = _attention_pair(_HCA_LAYER)
    hf_input, prime_input = _hidden_states()
    position_embeddings = _position_embeddings()
    packed = _packed_context(prime_module, _SINGLE_DOC, torch.bfloat16)

    hf_output, _ = hf_module(
        hf_input,
        position_embeddings=position_embeddings,
        position_ids=packed.position_ids,
        # As in the CSA case, HF concatenates a per-batch block bias onto the mask.
        attention_mask=packed.attention_mask.expand(_BATCH, 1, _SEQ, _SEQ),
    )
    prime_output, _ = prime_module(
        prime_input,
        position_embeddings=position_embeddings,
        packed=packed,
    )

    assert prime_output.shape == (_BATCH, _SEQ, _ATTN["hidden_size"])
    torch.testing.assert_close(prime_output, hf_output, rtol=0, atol=0)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


# prime-rl's `MoE` owns the router and the load-balancing bias, so both sit one level up
# from where HF keeps them, and its shared expert is singular. HF's fused routed-expert
# `gate_up_proj` splits into prime's `w1`/`w3`; `down_proj` renames to `w2`.
_HF_TO_PRIME_MOE_KEYS = {
    "gate.weight": "router.gate.weight",
    "gate.e_score_correction_bias": "expert_bias",
    "gate.tid2eid": "tid2eid",
    "experts.down_proj": "experts.w2",
}


def _to_prime_moe_items(hf_key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map one HF MoE key/value onto the one or two prime-rl key/value pairs it becomes."""
    if hf_key == "experts.gate_up_proj":
        gate, up = value.chunk(2, dim=1)
        return {"experts.w1": gate, "experts.w3": up}
    key = _HF_TO_PRIME_MOE_KEYS.get(hf_key, hf_key.replace("shared_experts.", "shared_expert.", 1))
    return {key: value}


def _sync_moe(hf_module: nn.Module, prime_module: nn.Module) -> None:
    hf_state: dict[str, torch.Tensor] = {}
    for key, value in hf_module.state_dict().items():
        hf_state.update(_to_prime_moe_items(key, value))
    assert set(hf_state) == set(prime_module.state_dict()), "the HF key set must map onto prime-rl's exactly"
    prime_module.load_state_dict(hf_state)


def _compare_moe_grads(hf_module: nn.Module, prime_module: nn.Module, rtol: float, atol: float) -> None:
    prime_params = dict(prime_module.named_parameters())
    for name, hf_param in hf_module.named_parameters():
        assert hf_param.grad is not None, f"{name} received no gradient in HF"
        for prime_name, expected_grad in _to_prime_moe_items(name, hf_param.grad).items():
            prime_grad = prime_params[prime_name].grad
            assert prime_grad is not None, f"{prime_name} received no gradient"
            torch.testing.assert_close(
                prime_grad, expected_grad, rtol=rtol, atol=atol, msg=lambda m, n=prime_name: f"{n}: {m}"
            )


def _moe_pair() -> tuple[nn.Module, nn.Module]:
    """Build an HF / prime-rl MoE pair from identical weights.

    Everything here runs in float32, unlike the rest of this file: prime-rl's router
    scores in float32 by design (`TokenChoiceTopKRouter` upcasts to keep the training
    loss from exploding) while HF scores in the activation dtype, so bf16 would put a
    ~1e-3 floor under every comparison and hide everything else.
    """
    hf_config = HFDeepseekV4Config(**_MOE)
    # The for-loop expert path keeps the comparison in float32; `use_grouped_mm` casts to
    # bfloat16 internally and is covered separately.
    prime_config = DeepseekV4Config(**_MOE, use_grouped_mm=False)
    with torch.device("cuda"):
        hf_module = HFDeepseekV4SparseMoeBlock(hf_config, layer_idx=0)
        prime_module = DeepseekV4MoE(prime_config, layer_idx=0)
    _randomize(hf_module)
    # The aux-loss-free load-balancing bias is a buffer, so `_randomize` leaves it at
    # zero and the biased selection path would go untested. It maps onto prime-rl's
    # `MoE.expert_bias`, which the forward pass feeds to the router.
    with torch.no_grad():
        hf_module.gate.e_score_correction_bias.normal_(mean=0.0, std=0.1)
    _sync_moe(hf_module, prime_module)
    return hf_module, prime_module


def test_moe_matches_hf():
    hf_module, prime_module = _moe_pair()
    hf_input, prime_input = _moe_hidden_states()

    hf_output = hf_module(hf_input)
    prime_output = prime_module(prime_input)

    assert prime_output.shape == (_BATCH, _SEQ, _MOE["hidden_size"])
    # Both implementations run the same float32 arithmetic on the same weights, but they
    # group it differently: prime-rl sorts the tokens into one contiguous matmul per
    # expert and scatter-adds the results, HF gathers each expert's tokens and index-adds
    # them. Only the summation order differs, hence a tolerance at the float32 floor.
    torch.testing.assert_close(prime_output, hf_output, rtol=1e-5, atol=1e-8)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    # Every parameter here trains, unlike the Lightning Indexer's: nothing on this path
    # goes through an integer selection that the gradient cannot cross.
    _compare_moe_grads(hf_module, prime_module, rtol=1e-4, atol=5e-7)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=1e-5, atol=1e-8)


def _hash_moe_pair() -> tuple[nn.Module, nn.Module]:
    """Build an HF / prime-rl hash-routed MoE pair from identical weights.

    Float32 for the reason `_moe_pair` documents: hash routing changes which experts a
    token reaches, not how the scores that weight them are computed. Both sides start from
    an all-zero table, which would send every token to expert 0, so it is drawn here.
    """
    hf_config = HFDeepseekV4Config(**_HASH_MOE)
    prime_config = DeepseekV4Config(**_HASH_MOE, use_grouped_mm=False)
    with torch.device("cuda"):
        hf_module = HFDeepseekV4SparseMoeBlock(hf_config, layer_idx=_HASH_LAYER)
        prime_module = DeepseekV4MoE(prime_config, layer_idx=_HASH_LAYER)
    _randomize(hf_module)
    with torch.no_grad():
        hf_module.gate.tid2eid.copy_(_tid2eid())
    _sync_moe(hf_module, prime_module)
    return hf_module, prime_module


def test_hash_moe_matches_hf():
    hf_module, prime_module = _hash_moe_pair()
    hf_input, prime_input = _moe_hidden_states()
    input_ids = _input_ids()

    hf_output = hf_module(hf_input, input_ids=input_ids)
    prime_output = prime_module(prime_input, input_ids=input_ids)

    assert prime_output.shape == (_BATCH, _SEQ, _HASH_MOE["hidden_size"])
    # Same float32 arithmetic, different grouping of the expert matmuls, as in
    # `test_moe_matches_hf`: the tolerance sits at the float32 summation-order floor.
    torch.testing.assert_close(prime_output, hf_output, rtol=1e-5, atol=1e-8)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_moe_grads(hf_module, prime_module, rtol=1e-4, atol=5e-7)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=1e-5, atol=1e-8)


def test_hash_moe_trains_the_gate():
    hf_module, prime_module = _hash_moe_pair()
    hf_input, prime_input = _moe_hidden_states()
    input_ids = _input_ids()

    hf_module(hf_input, input_ids=input_ids).sum().backward()
    prime_module(prime_input, input_ids=input_ids).sum().backward()

    # The table only decides which experts run; the gate still produces the weights their
    # outputs are scaled by, so it keeps training, by the same gradient HF gets.
    gate_grad = prime_module.router.gate.weight.grad
    assert gate_grad is not None and (gate_grad != 0).any()
    torch.testing.assert_close(gate_grad, hf_module.gate.weight.grad, rtol=1e-4, atol=5e-7)
    # The table is a buffer, so no optimizer can drift it away from its checkpoint values.
    assert "tid2eid" not in dict(prime_module.named_parameters())
