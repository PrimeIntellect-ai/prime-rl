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

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import (
    CompressionLayout,
    DeepseekV4Attention,
    PackedContext,
    build_compression_layout,
    build_sliding_window_mask,
)
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4Experts, DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers import norms
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
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


def test_sliding_attention_matches_hf(_torch_rms_norm):
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


def test_sliding_attention_builds_its_own_mask():
    _, prime_module = _attention_pair()
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()
    packed = _packed_context(prime_module, _SINGLE_DOC, torch.bfloat16)

    explicit, _ = prime_module(hidden, position_embeddings=position_embeddings, packed=packed)
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


def test_csa_attention_defaults_to_sequential_positions():
    _, prime_module = _attention_pair(_CSA_LAYER)
    _, hidden = _hidden_states()
    position_embeddings = _position_embeddings()
    packed = _packed_context(prime_module, _SINGLE_DOC, torch.bfloat16)

    explicit, _ = prime_module(hidden, position_embeddings=position_embeddings, packed=packed)
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
    num_hash_layers=0,
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
_HASH_MOE = dict(_MOE, num_hash_layers=1, vocab_size=16)
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


# Everything below exercises a *packed* batch: several documents concatenated into one row,
# with `position_ids` restarting per document and the lengths handed over as `seq_lens`, the
# way `trainer/batch.py` builds a micro-batch. HF never runs this model packed, so there is no
# upstream reference to copy. The oracle is self-consistency instead: a packed run must equal
# running each document on its own and concatenating. That is the production requirement,
# because vLLM serves each rollout alone, and a trainer that disagrees optimizes a model the
# sampler does not implement.
#
# Float32 throughout, unlike the bfloat16 parity tests above. `kv_proj` sees a different number
# of rows packed than alone and cuBLAS is free to tile the two differently, so the two runs can
# never be bit-identical; in bfloat16 that floor would sit around 1e-2 and swallow exactly the
# cross-document leakage these tests exist to catch.

_SINGLE_DOC = (_SEQ,)
# The boundary falls inside a window of both rates, so both drop tokens at it.
_MID_WINDOW_DOCS = (7, 9)
# Every document is a whole number of windows at both rates, so the per-document entry count
# equals the row-global one and only the numbering (and with it the RoPE position) moves.
_EXACT_MULTIPLE_DOCS = (8, 8)
# The first document is shorter than either rate, so it compresses to nothing while the rest of
# the row still carries entries.
_SHORT_FIRST_DOCS = (3, 13)
# Shorter than the HCA rate everywhere, so HCA compresses the whole row to zero entries.
_ALL_SHORT_DOCS = (5, 5, 6)
# One CSA entry in the first document and three in the second: `index_topk` of 2 has to discard,
# and a pick renumbered into the second document cannot coincide with its own local index.
_INDEXER_DOCS = (4, 12)

# Packed and unpacked run the same arithmetic through differently shaped matmuls, so they agree
# to the float32 rounding floor and no further. A query reading another document's tokens moves
# an output by a fraction of its own scale, orders of magnitude above this.
_PACKED_RTOL, _PACKED_ATOL = 1e-5, 1e-6
# Gradients get a bound against the tensor's own scale instead. They are sums over the whole row,
# and their near-zero entries are the ones whose summands cancelled, so an element-wise relative
# bound reads out cancellation noise rather than anything a document leak would move: one entry
# of `kv_proj`'s gradient lands 3e-4 off in relative terms while being 2e-6 off in absolute ones.
# Against the tensor's own scale the worst deviation measured over these tests is 8e-7.
_PACKED_GRAD_RTOL = 1e-5


def _cu_seqlens(doc_lens: tuple[int, ...]) -> torch.Tensor:
    return get_cu_seqlens_from_seq_lens(torch.tensor(doc_lens, device="cuda"), total_tokens=sum(doc_lens))[0]


def _layout(doc_lens: tuple[int, ...], compress_rate: int) -> CompressionLayout:
    return build_compression_layout(_cu_seqlens(doc_lens), compress_rate, sum(doc_lens))


def _layouts(doc_lens: tuple[int, ...]) -> dict[int, CompressionLayout]:
    """One layout per rate the config compresses at, as `DeepseekV4Model` hands them to a layer."""
    return {rate: _layout(doc_lens, rate) for rate in set(_ATTN["compress_rates"].values())}


def _entry_counts(doc_lens: tuple[int, ...], compress_rate: int) -> list[int]:
    return [length // compress_rate for length in doc_lens]


def _doc_slice(doc_lens: tuple[int, ...], index: int) -> slice:
    start = sum(doc_lens[:index])
    return slice(start, start + doc_lens[index])


def _entry_slice(doc_lens: tuple[int, ...], compress_rate: int, index: int) -> slice:
    """Where one document's compressed entries sit, the entry axis being laid out document by
    document exactly as the token axis is."""
    return _doc_slice(tuple(_entry_counts(doc_lens, compress_rate)), index)


def _doc_of_token(doc_lens: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([torch.full((length,), index, device="cuda") for index, length in enumerate(doc_lens)])


def _packed_position_ids(doc_lens: tuple[int, ...]) -> torch.Tensor:
    positions = torch.cat([torch.arange(length, device="cuda") for length in doc_lens])
    return positions.unsqueeze(0).expand(_BATCH, -1)


def _alone_position_ids(length: int, batch: int = _BATCH) -> torch.Tensor:
    return torch.arange(length, device="cuda").unsqueeze(0).expand(batch, -1)


def _compress_rates(module: nn.Module) -> set[int]:
    """The rates an attention layer needs a layout for: its compressor's, or none at all."""
    return set() if module.compressor is None else {module.compressor.compress_rate}


def _packed_context(module: nn.Module, doc_lens: tuple[int, ...], dtype: torch.dtype) -> PackedContext:
    """The context `DeepseekV4Model` would hand `module` for a row laid out as `doc_lens`.

    `_SINGLE_DOC` gives back what the layer builds for itself when passed no context at all, so
    the two are comparable. `dtype` is the mask's, and has to be the one the caller runs at.
    """
    return PackedContext.build(
        cu_seqlens=_cu_seqlens(doc_lens),
        position_ids=_packed_position_ids(doc_lens),
        total_tokens=sum(doc_lens),
        compress_rates=_compress_rates(module),
        sliding_window=_ATTN["sliding_window"],
        dtype=dtype,
        device=torch.device("cuda"),
    )


def _fp32_hidden_states(seq_len: int = _SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two leaves carrying identical values, one for the packed run and one for the lone runs."""
    with torch.device("cuda"):
        hidden = torch.randn(_BATCH, seq_len, _ATTN["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _fp32_attention(layer_idx: int) -> nn.Module:
    _, prime_config = _attention_configs()
    with torch.device("cuda"):
        module = DeepseekV4Attention(prime_config, layer_idx=layer_idx)
    _randomize_attention(module)
    return module


def _fp32_position_embeddings(position_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    _, prime_config = _attention_configs()
    with torch.device("cuda"):
        rotary = DeepseekV4RotaryEmbedding(prime_config)
        probe = torch.zeros(*position_ids.shape, _ATTN["hidden_size"])
    return {rope_type: rotary(probe, position_ids, rope_type) for rope_type in ("main", "compress")}


def _take_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    """Detach whatever gradients have accumulated and clear them for the next run."""
    grads = {name: None if param.grad is None else param.grad.clone() for name, param in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    return grads


def _assert_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    deviation = (actual - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _compare_accumulated_grads(
    module: nn.Module, expected: dict[str, torch.Tensor | None], rtol: float = _PACKED_GRAD_RTOL
) -> None:
    """Compare the gradients now on `module` against a snapshot taken from an earlier backward.

    Same shape as `_compare_grads`, including its allowance for a parameter that legitimately
    receives nothing: the Lightning Indexer reaches the loss only through integer top-k indices,
    so neither run may hand its parameters a gradient.
    """
    for name, param in module.named_parameters():
        if expected[name] is None:
            assert param.grad is None, f"{name} received a gradient per document but not packed"
            continue
        assert param.grad is not None, f"{name} received no gradient per document"
        _assert_relative(param.grad, expected[name], rtol, name)


def _backward_if_differentiable(loss: torch.Tensor) -> None:
    """Backward `loss` unless it carries no graph at all.

    A document too short to fill a window compresses to nothing, and an empty output has no
    graph to walk back through. Its contribution to every gradient is zero, so there is nothing
    to accumulate and nothing to assert beyond that.
    """
    if loss.requires_grad:
        loss.backward()


def _assert_layout_is_consistent(layout: CompressionLayout, doc_lens: tuple[int, ...], compress_rate: int) -> None:
    """Pin the layout against the per-document construction it claims to describe.

    Document `d` of length `L_d` gets `L_d // compress_rate` entries; entry `j` covers that
    document's local tokens `j * compress_rate` through `(j + 1) * compress_rate - 1` and is
    rotated at local position `j * compress_rate`. The entries are ordered document by document,
    which is what lets every comparison below slice them per document.
    """
    counts = _entry_counts(doc_lens, compress_rate)
    starts = [sum(doc_lens[:index]) for index in range(len(doc_lens))]
    expected_doc = [index for index, count in enumerate(counts) for _ in range(count)]
    expected_local = [entry for count in counts for entry in range(count)]
    expected_src = [
        [starts[doc] + local * compress_rate + offset for offset in range(compress_rate)]
        for doc, local in zip(expected_doc, expected_local)
    ]

    def as_tensor(values: list, dtype: torch.dtype = torch.long) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device="cuda")

    assert torch.equal(layout.entry_doc, as_tensor(expected_doc)), "entries are not ordered document by document"
    assert torch.equal(layout.entry_local, as_tensor(expected_local)), "entries are not numbered within a document"
    assert torch.equal(layout.src_idx, as_tensor(expected_src).reshape(-1, compress_rate)), (
        "an entry pools source tokens outside its own document's window"
    )
    assert torch.equal(layout.entry_pos, layout.entry_local * compress_rate), "entry_pos must be a local position"
    assert torch.equal(layout.is_first, layout.entry_local == 0), "is_first must mark entry 0 of each document"
    assert torch.equal(layout.doc_of_token, _doc_of_token(doc_lens)), "doc_of_token must follow the document lengths"


def _assert_compress_matches_per_document(
    compressor: nn.Module, doc_lens: tuple[int, ...], compress_rate: int, layout: CompressionLayout
) -> None:
    """Compressing a packed row must equal compressing each of its documents on its own.

    Forward and backward both: entry `n` of the packed run must pool the same source tokens, at
    the same compress-RoPE position, as the corresponding entry of its own document's run, and
    the gradient the packed run sends into the weights must equal the one the per-document runs
    accumulate. One random weight tensor is drawn over the packed entries and sliced per
    document, so the packed loss and the summed per-document losses are literally the same
    function of the same numbers.
    """
    _assert_layout_is_consistent(layout, doc_lens, compress_rate)
    counts = _entry_counts(doc_lens, compress_rate)
    packed_input, alone_input = _fp32_hidden_states(sum(doc_lens))

    packed = compressor.compress(packed_input, layout=layout)
    assert packed.shape == (_BATCH, sum(counts), compressor.head_dim)

    with torch.device("cuda"):
        weight = torch.randn_like(packed)
    (packed * weight).sum().backward()
    packed_grads = _take_grads(compressor)

    for index, count in enumerate(counts):
        entries = _entry_slice(doc_lens, compress_rate, index)
        alone = compressor.compress(alone_input[:, _doc_slice(doc_lens, index)])
        assert alone.shape == (_BATCH, count, compressor.head_dim), f"document {index} compressed to the wrong count"
        torch.testing.assert_close(
            packed[:, entries],
            alone,
            rtol=_PACKED_RTOL,
            atol=_PACKED_ATOL,
            msg=lambda m, i=index: f"document {i} compresses differently packed than alone: {m}",
        )
        _backward_if_differentiable((alone * weight[:, entries]).sum())

    _compare_accumulated_grads(compressor, packed_grads)
    torch.testing.assert_close(alone_input.grad, packed_input.grad, rtol=_PACKED_RTOL, atol=_PACKED_ATOL)


def test_csa_compressor_single_document_matches_today():
    """One document is the unpacked case, which packing must leave exactly where it was.

    The per-document comparison degenerates into `compress(x, layout)` against `compress(x)`
    here, so it pins the layout path against the layout-free one: four entries at packed
    positions 0, 4, 8, 12, only the first of them without a predecessor to pool.
    """
    module = _fp32_attention(_CSA_LAYER)
    layout = _layout(_SINGLE_DOC, _COMPRESS_RATE)

    assert layout.entry_pos.tolist() == [0, 4, 8, 12]
    assert layout.is_first.tolist() == [True, False, False, False]
    _assert_compress_matches_per_document(module.compressor, _SINGLE_DOC, _COMPRESS_RATE, layout)


def test_csa_compressor_splits_a_mid_window_boundary():
    """The boundary at token 7 falls inside a window, and the window must not span it.

    A row-global compression pools tokens 4 through 7 into one entry, blending the tail of the
    first document with the head of the second. Per document, the first document's tokens 4
    through 6 fill no window and are dropped, exactly as a trailing partial window is dropped
    when the document runs alone, and the second document starts a fresh window at token 7.
    """
    module = _fp32_attention(_CSA_LAYER)
    layout = _layout(_MID_WINDOW_DOCS, _COMPRESS_RATE)

    assert _entry_counts(_MID_WINDOW_DOCS, _COMPRESS_RATE) == [1, 2]
    assert layout.src_idx.shape[0] == 3 < _SEQ // _COMPRESS_RATE, "a row-global compression would emit more entries"
    _assert_compress_matches_per_document(module.compressor, _MID_WINDOW_DOCS, _COMPRESS_RATE, layout)


def test_csa_compressor_handles_an_exact_multiple():
    """Both documents are a whole number of windows, so only the numbering can be wrong.

    Nothing is dropped at the boundary and the entry count is the row-global one, which isolates
    the two remaining defects: the second document's entries have to be rotated at *its* local
    positions 0 and 4 rather than at packed 8 and 12, and its first entry has to be marked as
    first so the dual series' backward-looking half is gated off instead of reaching into the
    first document's last window.
    """
    module = _fp32_attention(_CSA_LAYER)
    layout = _layout(_EXACT_MULTIPLE_DOCS, _COMPRESS_RATE)

    assert layout.entry_pos.tolist() == [0, 4, 0, 4], "the second document must be rotated at its own positions"
    assert layout.is_first.tolist() == [True, False, True, False], "each document must have a first entry"
    _assert_compress_matches_per_document(module.compressor, _EXACT_MULTIPLE_DOCS, _COMPRESS_RATE, layout)


def test_csa_compressor_emits_no_entry_for_a_short_first_document():
    """A document below the compress rate contributes no entry while the row still has some.

    This is the path that does not exist today: the compressor early-returns only when the whole
    row is too short, never for one empty document among non-empty ones.
    """
    module = _fp32_attention(_CSA_LAYER)
    layout = _layout(_SHORT_FIRST_DOCS, _COMPRESS_RATE)

    assert _entry_counts(_SHORT_FIRST_DOCS, _COMPRESS_RATE) == [0, 3]
    assert layout.entry_doc.tolist() == [1, 1, 1], "every entry must belong to the second document"
    _assert_compress_matches_per_document(module.compressor, _SHORT_FIRST_DOCS, _COMPRESS_RATE, layout)


def test_csa_compressor_emits_only_first_entries_when_every_document_is_short():
    """Three documents of one window each: every entry is the first of its own document.

    A row-global compression would emit four entries, three of which straddle a boundary, and
    would pool a predecessor window into all but the first. Per document there is nothing for
    the backward-looking half of the dual series to reach, on any of the three.
    """
    module = _fp32_attention(_CSA_LAYER)
    layout = _layout(_ALL_SHORT_DOCS, _COMPRESS_RATE)

    assert _entry_counts(_ALL_SHORT_DOCS, _COMPRESS_RATE) == [1, 1, 1]
    assert layout.is_first.all(), "no entry has a predecessor inside its own document"
    _assert_compress_matches_per_document(module.compressor, _ALL_SHORT_DOCS, _COMPRESS_RATE, layout)


def test_hca_compressor_single_document_matches_today():
    """One document is the unpacked case, which packing must leave exactly where it was.

    Two entries at packed positions 0 and 8. HCA's windows do not overlap, so there is no
    predecessor to gate, and `is_first` carries no meaning beyond bookkeeping here.
    """
    module = _fp32_attention(_HCA_LAYER)
    layout = _layout(_SINGLE_DOC, _HCA_COMPRESS_RATE)

    assert layout.entry_pos.tolist() == [0, 8]
    _assert_compress_matches_per_document(module.compressor, _SINGLE_DOC, _HCA_COMPRESS_RATE, layout)


def test_hca_compressor_splits_a_mid_window_boundary():
    """At rate 8 the first document cannot fill a window at all and the second fills exactly one.

    A row-global compression pools tokens 0 through 7, which crosses the boundary at 7 and hands
    the second document an entry that is seven eighths someone else's rollout.
    """
    module = _fp32_attention(_HCA_LAYER)
    layout = _layout(_MID_WINDOW_DOCS, _HCA_COMPRESS_RATE)

    assert _entry_counts(_MID_WINDOW_DOCS, _HCA_COMPRESS_RATE) == [0, 1]
    assert layout.src_idx.tolist() == [[7, 8, 9, 10, 11, 12, 13, 14]], "the entry must pool the second document only"
    _assert_compress_matches_per_document(module.compressor, _MID_WINDOW_DOCS, _HCA_COMPRESS_RATE, layout)


def test_hca_compressor_handles_an_exact_multiple():
    """One whole window per document, so the entry count matches the row-global one exactly.

    What is left to get wrong is the rotation: the second document's entry covers its own local
    positions 0 through 7 and must be rotated at 0, not at packed position 8.
    """
    module = _fp32_attention(_HCA_LAYER)
    layout = _layout(_EXACT_MULTIPLE_DOCS, _HCA_COMPRESS_RATE)

    assert layout.entry_pos.tolist() == [0, 0], "the second document must be rotated at its own position"
    _assert_compress_matches_per_document(module.compressor, _EXACT_MULTIPLE_DOCS, _HCA_COMPRESS_RATE, layout)


def test_hca_compressor_emits_no_entry_for_a_short_first_document():
    """The three-token first document has nothing to compress while the row still yields an entry.

    Same new path as on the CSA side: an empty document among non-empty ones, which the
    whole-row early return never reached.
    """
    module = _fp32_attention(_HCA_LAYER)
    layout = _layout(_SHORT_FIRST_DOCS, _HCA_COMPRESS_RATE)

    assert _entry_counts(_SHORT_FIRST_DOCS, _HCA_COMPRESS_RATE) == [0, 1]
    assert layout.entry_doc.tolist() == [1], "the only entry must belong to the second document"
    _assert_compress_matches_per_document(module.compressor, _SHORT_FIRST_DOCS, _HCA_COMPRESS_RATE, layout)


def test_hca_compressor_emits_no_entries_when_every_document_is_short():
    """Every document is below rate 8, so the whole row compresses to nothing.

    A row-global compression fills two windows out of the sixteen packed tokens and hands them
    to whichever document happens to be far enough along, which is pure leakage: run alone, none
    of these documents has a long-range pathway at all. Written out rather than run through
    `_assert_compress_matches_per_document` because an empty compression is not differentiable:
    with no source token feeding it there is no graph to walk back through, so there are no
    gradients for the two runs to agree on.
    """
    module = _fp32_attention(_HCA_LAYER)
    compressor = module.compressor
    layout = _layout(_ALL_SHORT_DOCS, _HCA_COMPRESS_RATE)
    _assert_layout_is_consistent(layout, _ALL_SHORT_DOCS, _HCA_COMPRESS_RATE)
    assert _SEQ // _HCA_COMPRESS_RATE == 2, "a row-global compression must emit entries, or this is vacuous"
    packed_input, alone_input = _fp32_hidden_states()

    packed = compressor.compress(packed_input, layout=layout)
    assert packed.shape == (_BATCH, 0, compressor.head_dim)
    assert not packed.requires_grad, "an empty compression must not carry a graph back to the row"

    for index, length in enumerate(_ALL_SHORT_DOCS):
        assert length < _HCA_COMPRESS_RATE, f"document {index} must be too short to fill a window"
        alone = compressor.compress(alone_input[:, _doc_slice(_ALL_SHORT_DOCS, index)])
        assert alone.shape == (_BATCH, 0, compressor.head_dim), f"document {index} compressed to something"


def _selected_entries(top_k_indices: torch.Tensor, n_entries: int) -> torch.Tensor:
    """Boolean `[batch, seq, n_entries]`: which compressed entries each query's picks name.

    The `-1` sentinels are scattered into one throwaway column and sliced back off, the way the
    CSA compressor turns the same picks into its block bias. Comparing the selected *set* rather
    than the pick vector is what makes the comparison meaningful across runs: `index_topk` is
    clipped to the number of entries in the row, so a lone document with fewer entries than the
    packed row comes back with fewer columns even when it selects the same entries.
    """
    selected = torch.zeros(*top_k_indices.shape[:2], n_entries + 1, dtype=torch.bool, device=top_k_indices.device)
    safe = torch.where(top_k_indices >= 0, top_k_indices, torch.full_like(top_k_indices, n_entries))
    return selected.scatter_(-1, safe, True)[..., :n_entries]


def test_csa_indexer_selects_only_within_its_own_document():
    """What the Lightning Indexer hands a query must not depend on how the row was packed.

    Its picks are indices into the outer compressor's entries, and those get renumbered by a
    per-document compression, so a packed pick must equal its lone counterpart shifted by the
    document's entry offset, with `-1` preserved. Asserted on the indices rather than only on
    what attention then reads: a top-k that flips to a neighbouring entry is a real regression,
    and it should fail here rather than surface downstream as an unexplained numerical drift.
    """
    module = _fp32_attention(_CSA_LAYER)
    indexer = module.compressor.indexer
    counts = _entry_counts(_INDEXER_DOCS, _COMPRESS_RATE)
    assert counts == [1, 3] and counts[1] > _ATTN["index_topk"], "config must leave the indexer something to discard"
    layout = _layout(_INDEXER_DOCS, _COMPRESS_RATE)
    _, hidden = _fp32_hidden_states()
    hidden = hidden.detach()
    position_ids = _packed_position_ids(_INDEXER_DOCS)
    q_residual = module.q_a_norm(module.q_a_proj(hidden))

    packed_picks = indexer(hidden, q_residual, position_ids, layout=layout)
    packed_selected = _selected_entries(packed_picks, sum(counts))

    own_document = layout.entry_doc.view(1, 1, -1) == layout.doc_of_token.view(1, -1, 1)
    assert not (packed_selected & ~own_document).any(), "a query selected an entry from another document"
    assert (packed_picks < 0).any(), "vacuous probe: no query is early enough to have a surplus pick"
    for index, length in enumerate(_INDEXER_DOCS):
        span = _doc_slice(_INDEXER_DOCS, index)
        entries = _entry_slice(_INDEXER_DOCS, _COMPRESS_RATE, index)
        alone_picks = indexer(hidden[:, span], q_residual[:, span], _alone_position_ids(length))
        expected = torch.zeros_like(packed_selected[:, span])
        expected[..., entries] = _selected_entries(alone_picks, counts[index])
        assert torch.equal(packed_selected[:, span], expected), (
            f"document {index} selects different entries packed than on its own"
        )
    assert packed_selected[:, _doc_slice(_INDEXER_DOCS, 1)].any(), "vacuous probe: the second document picks nothing"
    assert packed_selected.sum(-1).max().item() == _ATTN["index_topk"], "some query must fill all of its picks"


def test_csa_indexer_marks_every_pick_invalid_with_no_readable_entry():
    """A query in a document that compressed to nothing must read nothing, not the row's entries.

    The first document is three tokens long, below the compress rate, so it has no entries of its
    own, while the row carries three built from the second document. The threshold is counted in
    the query's own document, so every one of those queries has to come back with `-1` picks and
    no long-range pathway at all, rather than being pointed at the start of the row.
    """
    module = _fp32_attention(_CSA_LAYER)
    indexer = module.compressor.indexer
    layout = _layout(_SHORT_FIRST_DOCS, _COMPRESS_RATE)
    assert layout.src_idx.shape[0] == 3, "the row must carry entries, or the invalid picks are vacuous"
    _, hidden = _fp32_hidden_states()
    hidden = hidden.detach()
    position_ids = _packed_position_ids(_SHORT_FIRST_DOCS)
    q_residual = module.q_a_norm(module.q_a_proj(hidden))

    packed_picks = indexer(hidden, q_residual, position_ids, layout=layout)

    first, second = (_doc_slice(_SHORT_FIRST_DOCS, index) for index in (0, 1))
    assert (packed_picks[:, first] < 0).all(), "a query whose document compressed to nothing was given a pick"
    assert (packed_picks[:, second] >= 0).any(), "vacuous probe: the second document picks nothing either"
    alone_picks = indexer(hidden[:, first], q_residual[:, first], _alone_position_ids(_SHORT_FIRST_DOCS[0]))
    assert (alone_picks >= 0).sum().item() == 0, "run alone the same document has nothing to pick from"


def _assert_attention_matches_per_document(module: nn.Module, doc_lens: tuple[int, ...]) -> None:
    """A packed attention layer must equal the same layer run on each document separately.

    Forward and backward, with one random weight drawn over the packed output and sliced per
    document so the two losses are the same function. The lone runs pass no context at all, which
    is the contract's default of a single document and the shape a rollout arrives in at
    inference time.
    """
    compress_rate = module.compressor.compress_rate
    packed_input, alone_input = _fp32_hidden_states(sum(doc_lens))
    packed = _packed_context(module, doc_lens, torch.float32)

    q_residual = module.q_a_norm(module.q_a_proj(packed_input.detach()))
    _, block_bias = module.compressor(
        packed_input.detach(), q_residual, packed.position_ids, layout=packed.compression_layouts[compress_rate]
    )
    assert (block_bias[:, :, _doc_slice(doc_lens, 1)] == 0).any(), (
        "vacuous probe: no query of the second document reads a compressed entry"
    )

    packed_output, _ = module(
        packed_input,
        position_embeddings=_fp32_position_embeddings(packed.position_ids),
        packed=packed,
    )
    with torch.device("cuda"):
        weight = torch.randn_like(packed_output)
    (packed_output * weight).sum().backward()
    packed_grads = _take_grads(module)

    for index, length in enumerate(doc_lens):
        span = _doc_slice(doc_lens, index)
        alone_position_ids = _alone_position_ids(length)
        alone_output, _ = module(
            alone_input[:, span],
            position_embeddings=_fp32_position_embeddings(alone_position_ids),
        )
        torch.testing.assert_close(
            packed_output[:, span],
            alone_output,
            rtol=_PACKED_RTOL,
            atol=_PACKED_ATOL,
            msg=lambda m, i=index: f"document {i} attends differently packed than alone: {m}",
        )
        (alone_output * weight[:, span]).sum().backward()

    _compare_accumulated_grads(module, packed_grads)
    torch.testing.assert_close(alone_input.grad, packed_input.grad, rtol=_PACKED_RTOL, atol=_PACKED_ATOL)


def _assert_attention_is_finite(module: nn.Module, doc_lens: tuple[int, ...]) -> None:
    """Run one packed forward and backward and require every number that comes out to be finite."""
    packed_input, _ = _fp32_hidden_states(sum(doc_lens))
    packed = _packed_context(module, doc_lens, torch.float32)

    output, _ = module(
        packed_input,
        position_embeddings=_fp32_position_embeddings(packed.position_ids),
        packed=packed,
    )
    assert torch.isfinite(output).all(), "the attention output is not finite"

    with torch.device("cuda"):
        weight = torch.randn_like(output)
    (output * weight).sum().backward()
    assert torch.isfinite(packed_input.grad).all(), "the input gradient is not finite"
    for name, param in module.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), f"{name} received a non-finite gradient"


def test_csa_attention_packed_matches_unpacked(_torch_rms_norm):
    """The invariant that makes the trainer agree with vLLM, on a CSA layer.

    Everything the layer reads past its local window comes through the compressor, so this is
    where a leaking entry, a misnumbered pick and a misrotated entry all show up at once.
    """
    _assert_attention_matches_per_document(_fp32_attention(_CSA_LAYER), _MID_WINDOW_DOCS)


def test_hca_attention_packed_matches_unpacked(_torch_rms_norm):
    """The same invariant on an HCA layer, which has no indexer to narrow the damage.

    One whole window per document at rate 8, so both documents own an entry and the second one's
    has to be rotated at its own position rather than at packed position 8.
    """
    _assert_attention_matches_per_document(_fp32_attention(_HCA_LAYER), _EXACT_MULTIPLE_DOCS)


def test_csa_attention_survives_a_zero_entry_document(_torch_rms_norm):
    """A document that compressed to nothing leaves its queries an all `-inf` block bias.

    The row still carries the second document's three entries, so the compressed half of the
    logit row exists and is masked off in full for every query of the first document. The local
    window and the attention sink keep the softmax normalizable; what this pins is that the
    masked row does not come back as NaN, forward or backward.
    """
    module = _fp32_attention(_CSA_LAYER)
    layout = _layouts(_SHORT_FIRST_DOCS)[_COMPRESS_RATE]
    hidden, _ = _fp32_hidden_states()
    position_ids = _packed_position_ids(_SHORT_FIRST_DOCS)
    q_residual = module.q_a_norm(module.q_a_proj(hidden.detach()))

    _, block_bias = module.compressor(hidden.detach(), q_residual, position_ids, layout=layout)
    assert block_bias.shape[-1] == 3, "the row must carry entries the first document cannot read"
    assert (block_bias[:, :, _doc_slice(_SHORT_FIRST_DOCS, 0)] == float("-inf")).all(), (
        "vacuous probe: the first document was allowed to read an entry"
    )

    _assert_attention_is_finite(module, _SHORT_FIRST_DOCS)


def test_hca_attention_survives_a_zero_entry_document(_torch_rms_norm):
    """Every document below rate 8, so the compressed half of the layer is empty for the row.

    This is the other new path: not one empty document among non-empty ones but a compressor
    that returns zero entries and a block bias with zero columns, which the attention block has
    to concatenate onto its local mask without producing an empty softmax or a NaN.
    """
    module = _fp32_attention(_HCA_LAYER)
    layout = _layouts(_ALL_SHORT_DOCS)[_HCA_COMPRESS_RATE]
    hidden, _ = _fp32_hidden_states()
    hidden = hidden.detach()
    position_ids = _packed_position_ids(_ALL_SHORT_DOCS)
    q_residual = module.q_a_norm(module.q_a_proj(hidden))

    compressed_kv, block_bias = module.compressor(hidden, q_residual, position_ids, layout=layout)
    assert compressed_kv.shape[2] == 0 and block_bias.shape[-1] == 0, "vacuous probe: the row compressed to something"

    _assert_attention_is_finite(module, _ALL_SHORT_DOCS)


# A whole model, one layer of every attention type over standard MoE blocks. Dropout is off so
# the packed and the lone runs cannot diverge through their random draws.
_MODEL = dict(
    vocab_size=64,
    hidden_size=128,
    moe_intermediate_size=64,
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
    index_topk=2,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
)

# Looser than the module-level floor, and it has to be: `DeepseekV4Experts` sorts the tokens by
# expert assignment, so a packed row and a lone document accumulate the same expert's matmul in a
# different order, and four layers of hyper-connected residual amplify that. Measured on this
# config the logits deviate by up to 3e-7 in absolute terms on a scale of 0.8, and the parameter
# gradients by up to 9e-6 relative to their own scale. Both tolerances sit an order of magnitude
# above that, because which tokens share an expert (and so what cancels) moves with the seed.
_MODEL_RTOL, _MODEL_ATOL = 1e-4, 1e-5
_MODEL_GRAD_RTOL = 1e-4


def _randomize_model(model: nn.Module) -> None:
    """Draw non-degenerate values for every parameter of the whole stack.

    The union of what `_randomize` and `_randomize_attention` each do for their own module: norm
    gains and hyper-connection scales default to ones, sinks, position biases and
    hyper-connection bases to zeros, and every one of those defaults would leave the path it
    controls indistinguishable from a no-op.
    """
    for name, param in model.named_parameters():
        with torch.no_grad():
            if name.endswith("scale") or name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            elif name.endswith("sinks") or name.endswith("position_bias"):
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)


def _packed_model() -> nn.Module:
    """A float32 model in eval mode, with the LM head the trainer wraps it in."""
    config = DeepseekV4Config(**_MODEL, use_grouped_mm=False)
    with torch.device("cuda"):
        model = DeepseekV4ForCausalLM._from_config(config)
    _randomize_model(model)
    inject_prime_lm_head(model, chunk_size=None)
    return model.eval()


def test_model_packed_matches_unpacked(_torch_rms_norm):
    """End to end, the whole point: packing rollouts together must not change any of them.

    The trainer packs; vLLM serves each rollout alone. Every layer contributes here, so unlike
    the compressor tests this one also covers the threading of `seq_lens` down from
    `DeepseekV4Model.forward` into the layouts the compressors receive. Gradients are compared
    as well, since a leak that barely moves the logits can still move the update.
    """
    model = _packed_model()
    doc_lens = _MID_WINDOW_DOCS
    input_ids = torch.randint(0, _MODEL["vocab_size"], (1, sum(doc_lens)), device="cuda")
    seq_lens = torch.tensor(doc_lens, device="cuda")

    packed = model(input_ids, position_ids=_packed_position_ids(doc_lens)[:1], seq_lens=seq_lens)["logits"]
    with torch.device("cuda"):
        weight = torch.randn_like(packed)
    (packed * weight).sum().backward()
    packed_grads = _take_grads(model)

    for index, length in enumerate(doc_lens):
        span = _doc_slice(doc_lens, index)
        alone = model(
            input_ids[:, span],
            position_ids=_alone_position_ids(length, batch=1),
            seq_lens=torch.tensor([length], device="cuda"),
        )["logits"]
        torch.testing.assert_close(
            packed[:, span],
            alone,
            rtol=_MODEL_RTOL,
            atol=_MODEL_ATOL,
            msg=lambda m, i=index: f"document {i} produces different logits packed than alone: {m}",
        )
        (alone * weight[:, span]).sum().backward()

    _compare_accumulated_grads(model, packed_grads, rtol=_MODEL_GRAD_RTOL)


def test_model_segments_by_seq_lens_on_a_padded_row(_torch_rms_norm, monkeypatch):
    """`seq_lens` decides the document layout, and `position_ids` restarting does not.

    `pad_micro_batch` (`trainer/batch.py:714-719`) folds its padding into the last document: it
    extends `seq_lens[-1]` while restarting `position_ids` at 0 over the pad block, so on a padded
    micro-batch the two disagree by construction. Following the restart would cut the last rollout
    in two and drop the entries that straddle the cut, which is a real capability loss on every
    padded step; following `seq_lens` treats the padding as a continuation, which costs nothing,
    since causality keeps it away from every real token and it is loss-masked. This is a design
    decision, not an accident, so it is asserted directly rather than through a packing oracle.

    Read off the width of the key stream each layer actually attends over, which is the local
    window plus that layer's compressed entries, rather than off the layout builder, so it keeps
    holding if the threading moves.
    """
    doc_len, pad_size = 7, 5
    total = doc_len + pad_size
    csa_rate = _MODEL["compress_rates"]["compressed_sparse_attention"]
    hca_rate = _MODEL["compress_rates"]["heavily_compressed_attention"]
    by_seq_lens = {
        "sliding_attention": 0,
        "compressed_sparse_attention": total // csa_rate,
        "heavily_compressed_attention": total // hca_rate,
    }
    by_restart = {
        "compressed_sparse_attention": doc_len // csa_rate + pad_size // csa_rate,
        "heavily_compressed_attention": doc_len // hca_rate + pad_size // hca_rate,
    }
    assert by_restart != {key: by_seq_lens[key] for key in by_restart}, "the two segmentations must disagree"

    recorded: list[int] = []
    real_attention = dsv4_attention.eager_attention_with_sinks

    def record(query, key, value, sinks, attention_mask, **kwargs):
        recorded.append(key.shape[2])
        return real_attention(query, key, value, sinks, attention_mask, **kwargs)

    monkeypatch.setattr(dsv4_attention, "eager_attention_with_sinks", record)

    model = _packed_model()
    input_ids = torch.randint(0, _MODEL["vocab_size"], (1, total), device="cuda")
    position_ids = torch.cat([torch.arange(doc_len, device="cuda"), torch.arange(pad_size, device="cuda")])
    model(input_ids, position_ids=position_ids.unsqueeze(0), seq_lens=torch.tensor([total], device="cuda"))

    assert len(recorded) == _MODEL["num_hidden_layers"]
    for layer_idx, layer_type in enumerate(_MODEL["layer_types"]):
        expected = total + by_seq_lens[layer_type]
        assert recorded[layer_idx] == expected, (
            f"layer {layer_idx} ({layer_type}) attends over {recorded[layer_idx]} keys, not {expected}: "
            "the compressed entries follow the position_ids restart rather than seq_lens"
        )
