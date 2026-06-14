from prime_rl.trainer.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM
from transformers import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM
import torch
from torch import nn
from prime_rl.utils.utils import default_dtype
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from copy import deepcopy
import pytest
import re


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _IdentityMLP(nn.Identity):
    def forward(self, x, **kwargs):
        return super().forward(x)


def get_configs(
    n_group: int = None,
    rope_interleave: bool = True,
    rope_type: str = "default",
    num_hidden_layers: int = 8,
    vocab_size: int = 50272,
):

    if rope_type == "yarn":
        rope_scaling = {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    else:
        rope_scaling = None

    hf_conf = DeepseekV3Config(
        vocab_size=vocab_size,
        max_position_embeddings=4096,
        hidden_size=1024,
        intermediate_size=1024,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=1,
        num_nextn_predict_layers=0,
        moe_intermediate_size=1024,
        norm_topk_prob=True,
        n_shared_experts=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=4 if n_group is None else n_group,
        topk_group=min(2, n_group) if isinstance(n_group, int) else 1,
        rope_theta=1000000.0,
        use_qk_norm=True,
        use_grouped_mm=False,
        rope_scaling=rope_scaling,
        rope_interleave=rope_interleave,
        load_balance_coeff=1,
    )

    prime_config = deepcopy(hf_conf)
    return hf_conf, prime_config


def fix_seed(seed_val: int = 42):
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)


def get_model_pairs(
    n_group: int = None,
    rope_interleave: bool = True,
    rope_type: str = "default",
    device_prime="cpu",
    device_hf=None,
    **config_kwgs,
):
    if device_hf is None:
        device_hf = device_prime

    hf_conf, prime_config = get_configs(
        n_group=n_group,
        rope_interleave=rope_interleave,
        rope_type=rope_type,
        **config_kwgs,
    )

    with torch.device(device_prime), default_dtype(torch.float32):
        prime_model = DeepseekV3ForCausalLM(prime_config)

    with torch.device(device_hf), default_dtype(torch.float32):
        hf_model = HFDeepseekV3ForCausalLM(hf_conf)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    return hf_model, prime_model


def assert_models_close(
    hf_model, prime_model, bs=1, sl=100, atol_logits=2e-2, atol_grad=2.0
):

    device = hf_model.device
    input_ids = torch.randint(0, hf_model.config.vocab_size, (bs, sl), device=device)
    position_ids = torch.arange(sl).unsqueeze(0).to(device)
    mask = torch.triu(torch.ones(sl, sl)).bool().to(device)

    hf_out = hf_model(
        input_ids=input_ids, attention_mask=mask, position_ids=position_ids
    )
    prime_out = prime_model(input_ids, position_ids)

    # logits
    logits_diff = torch.abs(hf_out.logits - prime_out["logits"]).max()
    assert logits_diff < atol_logits, f"Logits differ by {atol_logits:.3e}"

    # Grads
    hf_out.logits.sum().backward()
    prime_out["logits"].sum().backward()

    grad_diff = (
        hf_model.model.embed_tokens.weight.grad
        - prime_model.model.embed_tokens.weight.grad
    )
    grad_diff = grad_diff.abs().max()
    assert grad_diff < atol_grad, f"Grad differ by: {grad_diff:.3e}"


def assert_eval_models_close(hf_model, prime_model, device):
    """Compare models outputs without checking their gradients."""
    hf_model.eval()
    prime_model.eval()

    with torch.device(device), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    with torch.no_grad():
        hf_output = hf_model(input_ids, position_ids)
        prime_output = prime_model(input_ids, position_ids)

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(
        logits_diff, torch.zeros_like(logits_diff), atol=2e-2
    ), f"Max logits diff: {logits_diff.abs().max()}"


def test_empty_config():
    """Empty config shouldn't raise errors."""
    try:
        conf = DeepseekV3Config()
        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(conf)
    except:
        raise


@pytest.mark.parametrize(
    ("rope_interleave", "rope_type"),
    [
        (True, "default"),
        (False, "default"),
        (True, "yarn"),
        (False, "yarn"),
    ],
)
def test_deepseekv3_attention_only(rope_interleave: bool, rope_type: str, device):

    fix_seed(45)

    hf_model, prime_model = get_model_pairs(
        rope_interleave=rope_interleave, rope_type=rope_type, device_prime=device
    )

    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()

    for layer in prime_model.model.layers:
        layer.mlp = _IdentityMLP()

    assert_models_close(hf_model, prime_model, bs=1, sl=100)


@pytest.mark.parametrize(("n_group"), [1, 2, 4])
def test_deepseekv3_mlp_only(n_group: int, device):

    fix_seed(48)

    hf_model, prime_model = get_model_pairs(n_group=n_group, device_prime=device)

    def foo(hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states, None

    # replace attention layers
    for layer in hf_model.model.layers:
        layer.self_attn.forward = foo
    for layer in prime_model.model.layers:
        layer.self_attn.forward = foo

    assert_models_close(hf_model, prime_model, bs=1, sl=100)

    ## check that expert bias works
    num_experts = prime_model.config.n_routed_experts
    bias_val = 100 * torch.rand(num_experts, dtype=torch.float32, device=device)

    hf_model.model.layers[3].mlp.gate.e_score_correction_bias = bias_val
    prime_model.model.layers[3].mlp.expert_bias = bias_val

    assert_models_close(hf_model, prime_model, bs=1, sl=100)


def test_embeddings(device):

    fix_seed(49)

    hf_model, prime_model = get_model_pairs(device_prime=device)

    bs, sl = 1, 1024
    tokens = torch.randint(low=0, high=20000, size=(bs, sl))
    device = hf_model.device

    hf_embs = hf_model.model.embed_tokens(tokens.to(device))
    prime_embs = prime_model.model.embed_tokens(tokens.to(device))
    assert torch.allclose(hf_embs, prime_embs)


def test_deepseekv3(device):
    """Test full models are close."""

    fix_seed(51)

    hf_model, prime_model = get_model_pairs(
        rope_type="yarn", rope_interleave=True, n_group=4, device_prime=device
    )
    assert_models_close(hf_model, prime_model, bs=1, sl=100)


def test_deepseek_v3_cp_patching():
    """Verify substitute_ring_attn patches DeepSeekAttentionCore._compute_attention."""
    from unittest.mock import MagicMock

    from prime_rl.trainer.models.layers.attn import substitute_ring_attn
    from prime_rl.trainer.models.deepseek_v3.attention_deepseek_v3 import (
        DeepSeekAttentionCore,
    )

    original_method = DeepSeekAttentionCore._compute_attention

    mock_group = MagicMock()
    substitute_ring_attn(process_group=mock_group, heads_k_stride=1)

    assert DeepSeekAttentionCore._compute_attention is not original_method

    # Restore to avoid polluting other tests
    DeepSeekAttentionCore._compute_attention = original_method


def test_deepseek_v3_ulysses_patching():
    """Verify substitute_ulysses_attn patches DeepSeekAttentionCore._compute_attention."""
    from unittest.mock import MagicMock

    from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn
    from prime_rl.trainer.models.deepseek_v3.attention_deepseek_v3 import (
        DeepSeekAttentionCore,
    )

    original_method = DeepSeekAttentionCore._compute_attention

    mock_group = MagicMock()
    substitute_ulysses_attn(process_group=mock_group)

    assert DeepSeekAttentionCore._compute_attention is not original_method

    # Restore to avoid polluting other tests
    DeepSeekAttentionCore._compute_attention = original_method


def test_hf_to_prime_conversion(device):

    fix_seed(52)

    hf_conf, prime_conf = get_configs(n_group=None)

    with torch.device(device), default_dtype(torch.float32):
        prime_model = DeepseekV3ForCausalLM(prime_conf)
        hf_model = HFDeepseekV3ForCausalLM(hf_conf)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)

    assert_eval_models_close(hf_model, prime_model, device=device)


def test_prime_to_hf_conversion(device):

    fix_seed(53)

    hf_conf, prime_conf = get_configs(n_group=None)

    with torch.device(device), default_dtype(torch.float32):
        prime_model = DeepseekV3ForCausalLM(prime_conf)
        hf_model = HFDeepseekV3ForCausalLM(hf_conf)

    with torch.no_grad():
        state_dict = prime_model.state_dict()
        prime_model.convert_to_hf(state_dict)
        hf_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)

    assert_eval_models_close(hf_model, prime_model, device=device)


@pytest.mark.parametrize(("layer_idx"), [1, 4, 13, 20])
def test_layers_conversion(layer_idx: int, device):
    fix_seed(55)

    device = "meta"
    hf_conf, prime_conf = get_configs(n_group=None, num_hidden_layers=20)

    with torch.device(device), default_dtype(torch.float32):
        prime_model = DeepseekV3ForCausalLM(prime_conf)
        hf_model = HFDeepseekV3ForCausalLM(hf_conf)

    hf_state_dict = hf_model.state_dict()
    prime_state_dict = prime_model.state_dict()

    # prime to hf conversion
    new_state_dict = prime_model.convert_layer_to_prime(
        hf_state_dict.copy(), layer_idx=layer_idx
    )

    for k in new_state_dict.keys():
        p = re.search(r"model.layers.(\d+)", k)
        if not p:
            continue

        k_layer_idx = int(p.group(1))
        if k_layer_idx == layer_idx:
            assert k in prime_state_dict
        else:
            assert k in hf_state_dict

    # hf to prime conversion
    new_state_dict = prime_model.convert_layer_to_hf(
        prime_state_dict.copy(), layer_idx=layer_idx
    )

    for k in new_state_dict.keys():
        p = re.search(r"model.layers.(\d+)", k)
        if not p:
            continue

        k_layer_idx = int(p.group(1))
        if k_layer_idx == layer_idx:
            assert k in hf_state_dict
        else:
            assert k in prime_state_dict
