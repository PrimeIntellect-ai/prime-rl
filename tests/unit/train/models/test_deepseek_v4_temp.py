import pytest
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

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, build_sliding_window_mask
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead
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
    layer_types=["sliding_attention"] * 4,
    rms_norm_eps=1e-6,
)

_BATCH, _SEQ = 2, 16


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

    Norm gains default to ones and the sinks to zeros, which would leave both paths
    indistinguishable from an identity, hence the explicit spread.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name == "sinks":
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)


def _position_ids() -> torch.Tensor:
    return torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)


def _hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hidden = torch.randn(_BATCH, _SEQ, _ATTN["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _attention_pair() -> tuple[nn.Module, nn.Module]:
    hf_config, prime_config = _attention_configs()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_module = HFDeepseekV4Attention(hf_config, layer_idx=0)
        prime_module = DeepseekV4Attention(prime_config, layer_idx=0)
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
