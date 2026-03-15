import importlib.util

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.llama import LlamaForCausalLM as PrimeRLLlamaForCausalLM
from prime_rl.utils.utils import default_dtype

QUACK_INSTALLED = importlib.util.find_spec("quack") is not None
requires_quack_rms = pytest.mark.skipif(
    not (QUACK_INSTALLED and torch.cuda.is_available()),
    reason="quack RMSNorm requires quack-kernels and CUDA",
)


@pytest.mark.gpu
@requires_quack_rms
def test_quack_rms_norm_matches_reference_forward_and_backward():
    torch.manual_seed(0)

    reference = RMSNorm(RMSNormConfig(hidden_size=128, eps=1e-6)).to("cuda")
    quack = RMSNorm(RMSNormConfig(hidden_size=128, eps=1e-6, impl="quack")).to("cuda")
    quack.load_state_dict(reference.state_dict())

    hidden_ref = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hidden_quack = hidden_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(hidden_ref)

    out_ref = reference(hidden_ref)
    out_quack = quack(hidden_quack)

    out_ref.backward(grad_out)
    out_quack.backward(grad_out)

    torch.testing.assert_close(out_quack, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(hidden_quack.grad, hidden_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(quack.weight.grad, reference.weight.grad, atol=2e-2, rtol=2e-2)


@requires_quack_rms
def test_quack_rms_norm_requires_cuda_tensors():
    layer = RMSNorm(RMSNormConfig(hidden_size=8, impl="quack"))

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        layer(torch.randn(2, 8))


@pytest.mark.gpu
@requires_quack_rms
def test_prime_llama_selects_quack_rms_norm():
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        vocab_size=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    config._attn_implementation = "sdpa"
    config.rms_norm_impl = "quack"

    with torch.device("cuda"), default_dtype(torch.float32):
        model = PrimeRLLlamaForCausalLM._from_config(config)

    assert model.model.layers[0].input_layernorm.impl == "quack"
    assert model.model.layers[0].post_attention_layernorm.impl == "quack"
    assert model.model.norm.impl == "quack"
