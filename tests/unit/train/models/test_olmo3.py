import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM as HFOlmo3ForCausalLM

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, supports_custom_impl
from prime_rl.trainer.models.layers.attn import flash_attn_3_varlen_func, flash_attn_4_varlen_func
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.olmo3 import Olmo3ForCausalLM as PrimeRLOlmo3ForCausalLM
from prime_rl.utils.sequence_packing import build_cu_seqlens
from prime_rl.utils.utils import default_dtype


def _tiny_config(
    attn_impl: str = "sdpa",
    *,
    hidden_size: int = 64,
    intermediate_size: int = 160,
) -> Olmo3Config:
    config = Olmo3Config(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        max_position_embeddings=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=4,
        pad_token_id=1,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        sliding_window=4,
        tie_word_embeddings=False,
        vocab_size=128,
    )
    config._attn_implementation = attn_impl
    return config


def _model_pair(attn_impl: str = "sdpa") -> tuple[HFOlmo3ForCausalLM, PrimeRLOlmo3ForCausalLM]:
    config = _tiny_config(attn_impl)
    with default_dtype(torch.float32):
        hf_model = HFOlmo3ForCausalLM._from_config(config)
        prime_model = PrimeRLOlmo3ForCausalLM._from_config(config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = set(prime_model.state_dict().keys())
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
        inject_prime_lm_head(prime_model, chunk_size=None)
    assert prime_state_keys == set(state_dict.keys())
    hf_model.eval()
    prime_model.eval()
    return hf_model, prime_model


def test_olmo3_custom_impl_registered() -> None:
    config = _tiny_config()

    assert supports_custom_impl(config)
    assert isinstance(AutoModelForCausalLMPrimeRL.from_config(config), PrimeRLOlmo3ForCausalLM)


def test_olmo3_custom_impl_uses_yarn_rope_from_real_config_shape() -> None:
    config = _tiny_config()
    config.max_position_embeddings = 65536
    config.rope_parameters = None
    config.rope_scaling = {
        "rope_type": "yarn",
        "attention_factor": 1.2079441541679836,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    }

    with default_dtype(torch.float32):
        model = PrimeRLOlmo3ForCausalLM._from_config(config)

    assert model.model.rotary_emb.rope_type == "yarn"


def test_olmo3_custom_impl_uses_yarn_rope_from_nested_layer_config_shape() -> None:
    config = _tiny_config()
    yarn_rope = {
        "rope_type": "yarn",
        "attention_factor": 1.2079441541679836,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    }
    config.rope_parameters = {
        "full_attention": dict(yarn_rope),
        "sliding_attention": dict(yarn_rope),
        "rope_type": "default",
        "rope_theta": 10000.0,
    }
    config.rope_scaling = config.rope_parameters

    with default_dtype(torch.float32):
        model = PrimeRLOlmo3ForCausalLM._from_config(config)

    assert model.model.rotary_emb.rope_type == "yarn"


def test_olmo3_custom_impl_rejects_mixed_nested_rope_types() -> None:
    config = _tiny_config()
    config.rope_parameters = {
        "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }

    with pytest.raises(ValueError, match="single RoPE type"):
        with default_dtype(torch.float32):
            PrimeRLOlmo3ForCausalLM._from_config(config)


def test_olmo3_matches_hf_unpacked_sliding_and_full_attention() -> None:
    torch.manual_seed(0)
    hf_model, prime_model = _model_pair()
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 13))
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with torch.no_grad():
        hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
        prime_output = prime_model(input_ids=input_ids, position_ids=position_ids)

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-5, rtol=1e-5), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )


def test_olmo3_packed_sdpa_matches_independent_sequences() -> None:
    torch.manual_seed(0)
    _, prime_model = _model_pair()
    sequences = [
        torch.randint(0, prime_model.config.vocab_size, (1, 7)),
        torch.randint(0, prime_model.config.vocab_size, (1, 5)),
        torch.randint(0, prime_model.config.vocab_size, (1, 9)),
    ]
    packed_input_ids = torch.cat(sequences, dim=1)
    packed_position_ids = torch.cat([torch.arange(seq.shape[1]) for seq in sequences]).unsqueeze(0)
    cu_seqlens, max_seqlen = build_cu_seqlens([seq.shape[1] for seq in sequences])

    with torch.no_grad():
        packed_output = prime_model(
            input_ids=packed_input_ids,
            position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )["logits"]
        independent_output = torch.cat(
            [
                prime_model(
                    input_ids=seq,
                    position_ids=torch.arange(seq.shape[1]).unsqueeze(0),
                )["logits"]
                for seq in sequences
            ],
            dim=1,
        )

    logits_diff = packed_output - independent_output
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-5, rtol=1e-5), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("attn_impl", "kernel_func"),
    [
        ("flash_attention_3", flash_attn_3_varlen_func),
        ("flash_attention_4", flash_attn_4_varlen_func),
    ],
)
def test_olmo3_packed_flash_attention_matches_independent_sequences(attn_impl, kernel_func) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention numerics")
    if kernel_func is None:
        pytest.skip(f"{attn_impl} is not installed")

    probe = Path(__file__).with_name("olmo3_flash_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), attn_impl],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{attn_impl} probe failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
