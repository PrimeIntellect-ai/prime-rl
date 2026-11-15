import pytest
import torch
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM, Qwen3Config, Qwen3ForCausalLM

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM as PrimeRLQwen3ForCausalLM
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.parity import assert_matches_hf

SEQ_LEN = 64

pytestmark = [pytest.mark.gpu]


def save_checkpoint(model_cls, config, path) -> None:
    torch.manual_seed(0)
    with default_dtype(torch.bfloat16):
        model_cls(config).save_pretrained(path)


def load_model(path):
    model = get_model(ModelConfig(name=str(path), attn="flash_attention_2")).to("cuda")
    # setup_model wraps the LM head; the custom forward needs it to accept labels/temperature.
    inject_prime_lm_head(model, chunk_size=None)
    return model


def test_dense_checkpoint_matches_hf(tmp_path):
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    save_checkpoint(Qwen3ForCausalLM, config, tmp_path)
    model = load_model(tmp_path)
    assert isinstance(model, PrimeRLQwen3ForCausalLM)
    hf_model = Qwen3ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    assert_matches_hf(hf_model.to("cuda"), model, tol=2e-2)

    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]
        default_positions_logits = model(input_ids=input_ids, seq_lens=seq_lens)["logits"]
    torch.testing.assert_close(default_positions_logits, logits, rtol=0, atol=0)


def test_moe_checkpoint_uses_custom_impl(tmp_path):
    config = Glm4MoeConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
        max_position_embeddings=512,
    )
    save_checkpoint(Glm4MoeForCausalLM, config, tmp_path)
    model = load_model(tmp_path)
    assert isinstance(model, PrimeRLGlm4MoeForCausalLM)

    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), device="cuda")
    logits = model(input_ids=input_ids, seq_lens=torch.tensor([SEQ_LEN], device="cuda"))["logits"]
    assert logits.shape == (1, SEQ_LEN, config.vocab_size)
