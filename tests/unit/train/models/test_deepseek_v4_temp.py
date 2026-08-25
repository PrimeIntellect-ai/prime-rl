# Scratch file for development-time correctness checks during the DeepSeek V4 port,
# isolating each mechanism (mHC, rotary, attention variants, MoE) against HF's reference
# as it lands. Not meant to be merged into test_deepseek_v4.py as-is; once the model
# classes exist, split or fold what's still relevant and delete this file. See TODO.md.

import pytest
import torch
import torch.nn.functional as F
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
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4Experts, DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers import norms
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

_BASE = dict(
    hidden_size=128,
    num_hidden_layers=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
)

_ATTN = dict(
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    q_lora_rank=64,
    partial_rotary_factor=0.5,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    max_position_embeddings=256,
    sliding_window=6,
    o_groups=2,
    o_lora_rank=16,
    layer_types=[
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "sliding_attention",
    ],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    index_n_heads=4,
    index_head_dim=24,
    # Smaller than the four compressed entries a 16-token sequence yields, so the
    # Lightning Indexer's selection has to actually discard some of them.
    index_topk=2,
    rms_norm_eps=1e-6,
)

_BATCH, _SEQ = 2, 16
_SLIDING_LAYER, _CSA_LAYER, _HCA_LAYER = 0, 1, 2
_COMPRESS_RATE = _ATTN["compress_rates"]["compressed_sparse_attention"]
_HCA_COMPRESS_RATE = _ATTN["compress_rates"]["heavily_compressed_attention"]


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _randomize(module: nn.Module) -> None:
    """Draw non-degenerate values for every parameter.

    Both classes allocate with `torch.empty`, so a test must fill them. `init_weights`
    zeros the biases and ones the scales, which would leave those paths untested, hence
    the explicit spread here.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("scale"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            else:
                param.normal_(mean=0.0, std=0.02)


def _sync(hf_module: nn.Module, prime_module: nn.Module) -> None:
    hf_state = hf_module.state_dict()
    assert set(hf_state) == set(prime_module.state_dict()), "prime-rl and HF parameter names must match exactly"
    prime_module.load_state_dict(hf_state)


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


def _hidden_streams():
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        streams = torch.randn(_BATCH, _SEQ, _BASE["hc_mult"], _BASE["hidden_size"])
    return streams.clone().requires_grad_(True), streams.clone().requires_grad_(True)


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


def test_hyperconnection_collapses_streams_with_pre_gate():
    _, prime_module = _hyper_connection_pair()
    _, streams = _hidden_streams()

    post, comb, collapsed = prime_module(streams)

    assert post.shape == (_BATCH, _SEQ, _BASE["hc_mult"])
    assert comb.shape == (_BATCH, _SEQ, _BASE["hc_mult"], _BASE["hc_mult"])
    assert collapsed.shape == (_BATCH, _SEQ, _BASE["hidden_size"])
    assert collapsed.dtype == streams.dtype
    # `post` is 2 * sigmoid(.), `comb` is a positive Sinkhorn iterate: both stay in range.
    assert (post >= 0).all() and (post <= 2).all()
    assert (comb > 0).all()


def test_hyperconnection_comb_is_doubly_stochastic():
    _, prime_module = _hyper_connection_pair()
    _, streams = _hidden_streams()

    _, comb, _ = prime_module(streams)

    ones = torch.ones_like(comb.sum(dim=-1))
    torch.testing.assert_close(comb.sum(dim=-1), ones, rtol=0, atol=1e-5)
    torch.testing.assert_close(comb.sum(dim=-2), ones, rtol=0, atol=1e-5)


def test_hyperconnection_init_weights():
    config = DeepseekV4Config(**_BASE)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        module = DeepseekV4HyperConnection(config)
    module.init_weights(0.02)

    assert (module.base == 0).all()
    assert (module.scale == 1).all()
    assert module.fn.float().std().item() == pytest.approx(0.02, rel=0.1)


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


def test_hyperhead_init_weights():
    config = DeepseekV4Config(**_BASE)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        module = DeepseekV4HyperHead(config)
    module.init_weights(0.02)

    assert (module.hc_base == 0).all()
    assert (module.hc_scale == 1).all()
    assert module.hc_fn.float().std().item() == pytest.approx(0.02, rel=0.1)


@pytest.fixture
def _torch_rms_norm(monkeypatch):
    """Make the shared `RMSNorm` take its PyTorch path instead of the quack kernel.

    The kernel is a project-wide choice that predates this model and drifts from HF's
    fp32 reference by up to ~1e-2 in bf16, which would swamp everything the V4-specific
    math contributes. Disabling it is what lets the parity assertions stay exact.
    """
    monkeypatch.setattr(norms, "_get_quack_rmsnorm", lambda: None)


def _attention_configs() -> tuple[HFDeepseekV4Config, DeepseekV4Config]:
    hf_config = HFDeepseekV4Config(**_ATTN)
    # Force the eager path so HF actually runs its sink softmax, not an SDPA kernel.
    hf_config._attn_implementation = "eager"
    return hf_config, DeepseekV4Config(**_ATTN)


def _randomize_attention(module: nn.Module) -> None:
    """Draw non-degenerate values for every attention parameter.

    Norm gains default to ones, the sinks and the compressors' position biases to zeros,
    which would leave all three paths indistinguishable from an identity, hence the
    explicit spread. The position bias is drawn wide because it is a softmax logit: at the
    projections' std it would leave the pooling gate all but uniform.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name == "sinks" or name.endswith("position_bias"):
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)


def _position_ids() -> torch.Tensor:
    return torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)


def _hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hidden = torch.randn(_BATCH, _SEQ, _ATTN["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _attention_pair(layer_idx: int = _SLIDING_LAYER) -> tuple[nn.Module, nn.Module]:
    hf_config, prime_config = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_module = HFDeepseekV4Attention(hf_config, layer_idx=layer_idx)
        prime_module = DeepseekV4Attention(prime_config, layer_idx=layer_idx)
    _randomize_attention(hf_module)
    _sync(hf_module, prime_module)
    return hf_module, prime_module


def _position_embeddings() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    _, prime_config = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        rotary = DeepseekV4RotaryEmbedding(prime_config)
        probe = torch.zeros(_BATCH, _SEQ, _ATTN["hidden_size"])
    position_ids = _position_ids()
    return {rope_type: rotary(probe, position_ids, rope_type) for rope_type in ("main", "compress")}


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


def test_sliding_window_mask_matches_hf():
    hf_config, _ = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        probe = torch.zeros(_BATCH, _SEQ, _ATTN["hidden_size"])
    hf_mask = create_sliding_window_causal_mask(
        config=hf_config,
        inputs_embeds=probe,
        attention_mask=None,
        past_key_values=None,
        position_ids=_position_ids(),
        allow_is_causal_skip=False,
    )
    prime_mask = build_sliding_window_mask(_SEQ, _ATTN["sliding_window"], torch.bfloat16, torch.device("cuda"))

    assert prime_mask.shape == (1, 1, _SEQ, _SEQ)
    torch.testing.assert_close(prime_mask.expand_as(hf_mask), hf_mask, rtol=0, atol=0)


def test_sliding_attention_matches_hf(_torch_rms_norm):
    hf_module, prime_module = _attention_pair()
    hf_input, prime_input = _hidden_states()
    position_embeddings = _position_embeddings()
    mask = build_sliding_window_mask(_SEQ, _ATTN["sliding_window"], torch.bfloat16, torch.device("cuda"))

    hf_output, _ = hf_module(
        hf_input,
        position_embeddings=position_embeddings,
        position_ids=_position_ids(),
        attention_mask=mask,
    )
    prime_output, prime_weights = prime_module(
        prime_input,
        position_embeddings=position_embeddings,
        attention_mask=mask,
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


def test_sliding_attention_builds_its_own_mask():
    _, prime_module = _attention_pair()
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()
    mask = build_sliding_window_mask(_SEQ, _ATTN["sliding_window"], torch.bfloat16, torch.device("cuda"))

    explicit, _ = prime_module(hidden, position_embeddings=position_embeddings, attention_mask=mask)
    implicit, _ = prime_module(hidden, position_embeddings=position_embeddings)

    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


def test_sliding_attention_only_reads_the_local_window():
    _, prime_module = _attention_pair()
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()
    window = _ATTN["sliding_window"]

    baseline, _ = prime_module(hidden, position_embeddings=position_embeddings)
    perturbed_input = hidden.clone()
    perturbed_input[:, 0] += 1.0
    perturbed, _ = prime_module(perturbed_input, position_embeddings=position_embeddings)

    # Token 0 is the last key inside the window of query `window - 1` and the first one
    # outside the window of query `window`.
    assert not torch.equal(perturbed[:, window - 1], baseline[:, window - 1])
    torch.testing.assert_close(perturbed[:, window:], baseline[:, window:], rtol=0, atol=0)


def test_csa_attention_matches_hf(_torch_rms_norm):
    hf_module, prime_module = _attention_pair(_CSA_LAYER)
    hf_input, prime_input = _hidden_states()
    position_embeddings = _position_embeddings()
    position_ids = _position_ids()
    # HF concatenates the compressor's per-batch block bias onto the mask, so the local
    # window mask has to carry a batch dimension of its own here.
    mask = build_sliding_window_mask(_SEQ, _ATTN["sliding_window"], torch.bfloat16, torch.device("cuda"))
    mask = mask.expand(_BATCH, 1, _SEQ, _SEQ)

    hf_output, _ = hf_module(
        hf_input,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=mask,
    )
    prime_output, _ = prime_module(
        prime_input,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        position_ids=position_ids,
    )

    assert prime_output.shape == (_BATCH, _SEQ, _ATTN["hidden_size"])
    torch.testing.assert_close(prime_output, hf_output, rtol=0, atol=0)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


def test_csa_attention_defaults_to_sequential_positions():
    _, prime_module = _attention_pair(_CSA_LAYER)
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()

    explicit, _ = prime_module(hidden, position_embeddings=position_embeddings, position_ids=_position_ids())
    implicit, _ = prime_module(hidden, position_embeddings=position_embeddings)

    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


def test_csa_attention_reads_beyond_the_local_window():
    _, prime_module = _attention_pair(_CSA_LAYER)
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()
    window = _ATTN["sliding_window"]

    baseline, _ = prime_module(hidden, position_embeddings=position_embeddings)
    perturbed_input = hidden.clone()
    perturbed_input[:, 0] += 1.0
    perturbed, _ = prime_module(perturbed_input, position_embeddings=position_embeddings)

    # Token 0 is outside the local window of every query from `window` on, and a sliding
    # layer ignores it there (see `test_sliding_attention_only_reads_the_local_window`).
    # A CSA layer still reaches it through the compressed entries it pools into.
    assert not torch.equal(perturbed[:, window:], baseline[:, window:])


def test_csa_compressor_pools_overlapping_windows():
    _, prime_module = _attention_pair(_CSA_LAYER)
    compressor = prime_module.compressor
    _, hidden = _hidden_states()

    compressed = compressor.compress(hidden)
    assert compressed.shape == (_BATCH, _SEQ // _COMPRESS_RATE, _ATTN["head_dim"])

    token = _COMPRESS_RATE + 1
    perturbed_input = hidden.clone()
    perturbed_input[:, token] += 1.0
    perturbed = compressor.compress(perturbed_input)

    changed = {w for w in range(compressed.shape[1]) if not torch.equal(perturbed[:, w], compressed[:, w])}
    # A token feeds its own window's entry through the `Cb` series and the next window's
    # through `Ca`; nothing earlier and nothing later may move.
    assert changed == {token // _COMPRESS_RATE, token // _COMPRESS_RATE + 1}


def test_csa_compressor_drops_the_trailing_partial_window():
    _, prime_module = _attention_pair(_CSA_LAYER)
    compressor = prime_module.compressor
    _, hidden = _hidden_states()

    full = compressor.compress(hidden)
    truncated = compressor.compress(hidden[:, : _SEQ - 1])

    assert truncated.shape[1] == full.shape[1] - 1
    torch.testing.assert_close(truncated, full[:, : truncated.shape[1]], rtol=0, atol=0)


def test_csa_indexer_keeps_only_readable_entries():
    _, prime_module = _attention_pair(_CSA_LAYER)
    indexer = prime_module.compressor.indexer
    _, hidden = _hidden_states()
    position_ids = _position_ids()
    q_residual = prime_module.q_a_norm(prime_module.q_a_proj(hidden))

    top_k_indices = indexer(hidden, q_residual, position_ids)

    top_k = _ATTN["index_topk"]
    assert top_k_indices.shape == (_BATCH, _SEQ, top_k)
    # Entry `w` pools tokens up to `(w + 1) * compress_rate - 1`, so query `t` may read
    # `(t + 1) // compress_rate` of them.
    readable = (position_ids + 1) // _COMPRESS_RATE
    assert readable.max() > top_k, "config must leave the indexer something to discard"
    assert (top_k_indices < readable.unsqueeze(-1)).all(), "an unreadable entry was selected"
    # `-1` pads the picks of queries with fewer readable entries than `index_topk`.
    kept = (top_k_indices >= 0).sum(dim=-1)
    torch.testing.assert_close(kept, readable.clamp(max=top_k))


def test_csa_attention_init_weights_reaches_the_compressor():
    _, prime_module = _attention_pair(_CSA_LAYER)
    assert (prime_module.compressor.indexer.position_bias != 0).any(), "fixture must start from a spread"

    prime_module.init_weights(0.02)

    assert (prime_module.sinks == 0).all()
    assert (prime_module.compressor.position_bias == 0).all()
    assert (prime_module.compressor.indexer.position_bias == 0).all()


def test_csa_indexer_selection_is_not_differentiable():
    _, prime_module = _attention_pair(_CSA_LAYER)
    _, hidden = _hidden_states()

    output, _ = prime_module(hidden, position_embeddings=_position_embeddings())
    output.sum().backward()

    compressor = prime_module.compressor
    for name, param in compressor.named_parameters():
        got_grad = param.grad is not None
        # The compressed entries are attended over, so the compressor trains; the indexer
        # only emits integer indices, so nothing differentiates back into it. DeepSeek
        # trains it with a separate auxiliary loss that prime-rl does not have yet.
        assert got_grad == (not name.startswith("indexer.")), f"unexpected gradient state for {name}"


def test_hca_attention_matches_hf(_torch_rms_norm):
    hf_module, prime_module = _attention_pair(_HCA_LAYER)
    hf_input, prime_input = _hidden_states()
    position_embeddings = _position_embeddings()
    position_ids = _position_ids()
    # As in the CSA case, HF concatenates a per-batch block bias onto the mask.
    mask = build_sliding_window_mask(_SEQ, _ATTN["sliding_window"], torch.bfloat16, torch.device("cuda"))
    mask = mask.expand(_BATCH, 1, _SEQ, _SEQ)

    hf_output, _ = hf_module(
        hf_input,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=mask,
    )
    prime_output, _ = prime_module(
        prime_input,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        position_ids=position_ids,
    )

    assert prime_output.shape == (_BATCH, _SEQ, _ATTN["hidden_size"])
    torch.testing.assert_close(prime_output, hf_output, rtol=0, atol=0)

    with torch.device("cuda"):
        weight = torch.randn_like(hf_output)
    (hf_output * weight).sum().backward()
    (prime_output * weight).sum().backward()

    _compare_grads(hf_module, prime_module)
    torch.testing.assert_close(prime_input.grad, hf_input.grad, rtol=0, atol=0)


def test_hca_attention_reads_every_readable_compressed_entry():
    _, prime_module = _attention_pair(_HCA_LAYER)
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()
    window = _ATTN["sliding_window"]

    baseline, _ = prime_module(hidden, position_embeddings=position_embeddings)
    perturbed_input = hidden.clone()
    perturbed_input[:, 0] += 1.0
    perturbed, _ = prime_module(perturbed_input, position_embeddings=position_embeddings)

    # Token 0 leaves the local window at query `window` and only re-enters through
    # compressed entry 0, which covers tokens `0 .. compress_rate - 1` and so is unreadable
    # until the query reaches the last of them. In between, nothing carries it.
    first_readable = _HCA_COMPRESS_RATE - 1
    assert first_readable > window, "config must leave a gap between the window and the first entry"
    torch.testing.assert_close(perturbed[:, window:first_readable], baseline[:, window:first_readable], rtol=0, atol=0)
    assert not torch.equal(perturbed[:, first_readable:], baseline[:, first_readable:])


def test_hca_compressor_pools_non_overlapping_windows():
    _, prime_module = _attention_pair(_HCA_LAYER)
    compressor = prime_module.compressor
    _, hidden = _hidden_states()

    compressed = compressor.compress(hidden)
    assert compressed.shape == (_BATCH, _SEQ // _HCA_COMPRESS_RATE, _ATTN["head_dim"])

    token = _HCA_COMPRESS_RATE + 1
    perturbed_input = hidden.clone()
    perturbed_input[:, token] += 1.0
    perturbed = compressor.compress(perturbed_input)

    changed = {w for w in range(compressed.shape[1]) if not torch.equal(perturbed[:, w], compressed[:, w])}
    # The windows do not overlap, so a token feeds its own entry and no other. This is the
    # whole structural difference from CSA, whose `Ca` series spills into the next window.
    assert changed == {token // _HCA_COMPRESS_RATE}


def test_hca_compressor_drops_the_trailing_partial_window():
    _, prime_module = _attention_pair(_HCA_LAYER)
    compressor = prime_module.compressor
    _, hidden = _hidden_states()

    full = compressor.compress(hidden)
    truncated = compressor.compress(hidden[:, : _SEQ - 1])

    # One token short of a full window is one entry short, and the entries that survive are
    # bit-identical: the dropped tokens never fed them.
    assert truncated.shape[1] == full.shape[1] - 1
    torch.testing.assert_close(truncated, full[:, : truncated.shape[1]], rtol=0, atol=0)


def test_hca_compressor_masks_unreadable_entries():
    _, prime_module = _attention_pair(_HCA_LAYER)
    compressor = prime_module.compressor
    _, hidden = _hidden_states()
    position_ids = _position_ids()

    q_residual = prime_module.q_a_norm(prime_module.q_a_proj(hidden))

    compressed_kv, block_bias = compressor(hidden, q_residual, position_ids)

    n_windows = _SEQ // _HCA_COMPRESS_RATE
    assert compressed_kv.shape == (_BATCH, 1, n_windows, _ATTN["head_dim"])
    assert block_bias.shape == (_BATCH, 1, _SEQ, n_windows)
    # Every readable entry is unbiased: there is no indexer to gate them any further.
    readable = (position_ids + 1) // _HCA_COMPRESS_RATE
    entries = torch.arange(n_windows, device=block_bias.device).view(1, 1, 1, -1)
    expected = torch.where(entries < readable.unsqueeze(1).unsqueeze(-1), 0.0, float("-inf"))
    torch.testing.assert_close(block_bias, expected.to(block_bias.dtype), rtol=0, atol=0)


def test_hca_compressor_is_fully_differentiable():
    _, prime_module = _attention_pair(_HCA_LAYER)
    _, hidden = _hidden_states()

    output, _ = prime_module(hidden, position_embeddings=_position_embeddings())
    output.sum().backward()

    # Unlike CSA, every compressed entry is attended over directly, so there is no
    # non-differentiable selection step and no parameter left without a gradient.
    for name, param in prime_module.compressor.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} received a non-finite gradient"


def test_hca_attention_init_weights_reaches_the_compressor():
    _, prime_module = _attention_pair(_HCA_LAYER)
    assert (prime_module.compressor.position_bias != 0).any(), "fixture must start from a spread"

    prime_module.init_weights(0.02)

    assert (prime_module.sinks == 0).all()
    assert (prime_module.compressor.position_bias == 0).all()


_MOE = dict(
    hidden_size=64,
    num_hidden_layers=2,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    # Small enough that the parameter spread `_randomize` draws actually reaches the
    # clamp; the 10.0 default would leave that branch untested.
    swiglu_limit=0.1,
    mlp_layer_types=["moe", "moe"],
    rms_norm_eps=1e-6,
)
_MOE_TOKENS = _BATCH * _SEQ

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


def _moe_hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"):
        hidden = torch.randn(_BATCH, _SEQ, _MOE["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


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


def test_moe_router_scores_with_sqrt_softplus():
    _, prime_module = _moe_pair()
    router = prime_module.router
    _, hidden = _moe_hidden_states()
    x = hidden.detach().reshape(-1, _MOE["hidden_size"])

    top_scores, indices, num_tokens_per_expert, _ = router(x)

    scores = F.softplus(F.linear(x, router.gate.weight)).sqrt()
    expected_scores, expected_indices = torch.topk(scores, _MOE["num_experts_per_tok"], dim=1)
    expected = expected_scores / expected_scores.sum(dim=-1, keepdim=True) * _MOE["routed_scaling_factor"]

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(top_scores, expected, rtol=0, atol=0)
    # Normalization happens before the scale, so every token's weights sum to it.
    torch.testing.assert_close(
        top_scores.sum(dim=-1), torch.full((x.shape[0],), _MOE["routed_scaling_factor"], device=x.device)
    )
    assert num_tokens_per_expert.sum().item() == _MOE_TOKENS * _MOE["num_experts_per_tok"]


def test_moe_expert_bias_steers_selection_but_not_the_gate():
    _, prime_module = _moe_pair()
    router = prime_module.router
    _, hidden = _moe_hidden_states()
    x = hidden.detach().reshape(-1, _MOE["hidden_size"])
    favored = 5

    _, unbiased_indices, _, _ = router(x)
    with torch.device("cuda"):
        expert_bias = torch.zeros(_MOE["n_routed_experts"])
    expert_bias[favored] = 100.0
    top_scores, indices, num_tokens_per_expert, _ = router(x, expert_bias=expert_bias)

    assert not torch.equal(indices, unbiased_indices), "the bias must change the selection"
    assert num_tokens_per_expert[favored].item() == _MOE_TOKENS, "every token must reach the favored expert"
    # The bias only steers the argmax: the gating values stay the unbiased scores.
    scores = F.softplus(F.linear(x, router.gate.weight)).sqrt().gather(dim=1, index=indices)
    expected = scores / scores.sum(dim=-1, keepdim=True) * _MOE["routed_scaling_factor"]
    torch.testing.assert_close(top_scores, expected, rtol=0, atol=0)


def test_moe_shared_expert_clamps_the_swiglu():
    _, prime_module = _moe_pair()
    shared_expert = prime_module.shared_expert
    _, hidden = _moe_hidden_states()
    x = hidden.detach()

    output = shared_expert(x)

    gate, up = shared_expert.gate_proj(x), shared_expert.up_proj(x)
    limit = shared_expert.limit
    assert (gate > limit).any() and (up.abs() > limit).any(), "config must push the pre-activations past the clamp"
    clamped = shared_expert.down_proj(F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit))
    torch.testing.assert_close(output, clamped, rtol=0, atol=0)
    assert not torch.equal(output, shared_expert.down_proj(F.silu(gate) * up))


def _experts(use_grouped_mm: bool) -> DeepseekV4Experts:
    return DeepseekV4Experts(
        dim=_MOE["hidden_size"],
        hidden_dim=_MOE["moe_intermediate_size"],
        num_experts=_MOE["n_routed_experts"],
        swiglu_limit=_MOE["swiglu_limit"],
        use_grouped_mm=use_grouped_mm,
    )


def test_moe_grouped_mm_experts_match_the_for_loop():
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        for_loop, grouped_mm = _experts(use_grouped_mm=False), _experts(use_grouped_mm=True)
        x = torch.randn(_MOE_TOKENS, _MOE["hidden_size"])
    _randomize(for_loop)
    grouped_mm.load_state_dict(for_loop.state_dict())
    # Tokens arrive pre-sorted by expert, exactly as `MoE`'s reorderer hands them over.
    expert_of_token = torch.arange(_MOE_TOKENS, device="cuda") % _MOE["n_routed_experts"]
    num_tokens_per_expert = torch.histc(expert_of_token, bins=_MOE["n_routed_experts"], min=0, max=8)

    # The grouped GEMM computes the same function through a different kernel, so the
    # tolerance sits at the bfloat16 rounding floor for outputs of this magnitude.
    torch.testing.assert_close(
        grouped_mm(x, num_tokens_per_expert), for_loop(x, num_tokens_per_expert), rtol=1e-2, atol=1e-5
    )


def test_moe_init_weights():
    prime_config = DeepseekV4Config(**_MOE, use_grouped_mm=False)
    with torch.device("cuda"):
        module = DeepseekV4MoE(prime_config, layer_idx=0)

    module.init_weights(0.5, torch.device("cuda"))

    # The gated branches keep the shared `MoE`'s fixed 0.02, the rest scales with init_std.
    expected_std = {
        "router.gate.weight": 0.5,
        "experts.w1": 0.02,
        "experts.w3": 0.02,
        "experts.w2": 0.5,
        "shared_expert.gate_proj.weight": 0.02,
        "shared_expert.up_proj.weight": 0.5,
        "shared_expert.down_proj.weight": 0.5,
    }
    for name, param in module.named_parameters():
        assert param.std().item() == pytest.approx(expected_std[name], rel=0.15), name
    assert (module.expert_bias == 0).all()


# A vocabulary small enough that a 32-token batch hits most rows of the table several times.
_HASH_MOE = dict(_MOE, mlp_layer_types=["hash_moe", "moe"], vocab_size=16)
_HASH_LAYER = 0


def _tid2eid() -> torch.Tensor:
    """A frozen token id -> expert ids table, distinct experts per row as a real one has."""
    rows = [
        torch.randperm(_HASH_MOE["n_routed_experts"])[: _HASH_MOE["num_experts_per_tok"]]
        for _ in range(_HASH_MOE["vocab_size"])
    ]
    return torch.stack(rows).to(device="cuda", dtype=torch.long)


def _input_ids() -> torch.Tensor:
    return torch.randint(_HASH_MOE["vocab_size"], (_BATCH, _SEQ), device="cuda")


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


def test_hash_moe_routes_by_token_id_not_by_score():
    _, prime_module = _hash_moe_pair()
    _, hidden = _moe_hidden_states()
    hidden = hidden.detach()
    table = prime_module.tid2eid
    assert set(table[0].tolist()) != set(table[1].tolist()), "the two rows must differ for this to bite"

    # Identical hidden states throughout: only the token ids move the routing.
    for token_id in (0, 1):
        token_ids = torch.full((_BATCH, _SEQ), token_id, device="cuda", dtype=torch.long)
        prime_module.tokens_per_expert.zero_()
        prime_module(hidden, input_ids=token_ids)

        expected = torch.zeros_like(prime_module.tokens_per_expert)
        expected[table[token_id]] = _MOE_TOKENS
        torch.testing.assert_close(prime_module.tokens_per_expert, expected)

    # The learned scores are still computed, and they would have picked other experts.
    _, learned_indices, _, _ = prime_module.router(hidden.reshape(-1, _HASH_MOE["hidden_size"]))
    assert set(learned_indices.flatten().tolist()) != set(table[0].tolist())


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


def test_moe_ignores_input_ids_when_not_hash_routed():
    _, prime_module = _moe_pair()
    _, hidden = _moe_hidden_states()

    prime_module.tokens_per_expert.zero_()
    with_ids = prime_module(hidden, input_ids=_input_ids())
    counts_with_ids = prime_module.tokens_per_expert.clone()
    prime_module.tokens_per_expert.zero_()
    without_ids = prime_module(hidden)

    # The selection is identical down to the token count per expert. The outputs only match
    # to the float32 floor: the scatter-add over experts is not bitwise reproducible.
    torch.testing.assert_close(counts_with_ids, prime_module.tokens_per_expert, rtol=0, atol=0)
    torch.testing.assert_close(with_ids, without_ids, rtol=1e-5, atol=1e-8)
