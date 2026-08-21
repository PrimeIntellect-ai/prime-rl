import pytest
import torch
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection as HFDeepseekV4HyperConnection,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperHead as HFDeepseekV4HyperHead,
)

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead
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


def _compare_grads(hf_module: nn.Module, prime_module: nn.Module) -> None:
    prime_grads = dict(prime_module.named_parameters())
    for name, hf_param in hf_module.named_parameters():
        prime_grad = prime_grads[name].grad
        assert prime_grad is not None, f"{name} received no gradient"
        torch.testing.assert_close(prime_grad, hf_param.grad, rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}")


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
