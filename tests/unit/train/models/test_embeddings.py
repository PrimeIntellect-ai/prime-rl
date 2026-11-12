import pytest
import torch

from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_MODELS, AutoModelForCausalLMPrimeRL

PAD_TOKEN_ID = 1


@pytest.mark.parametrize(
    "config_cls", [config_cls for config_cls, _ in _CUSTOM_CAUSAL_LM_MODELS], ids=lambda cls: cls.model_type
)
def test_pad_token_embedding_receives_gradient(config_cls):
    """get_model falls back to EOS for pad_token_id, a real token in the training data."""
    config = config_cls(dsa_backend="tilelang")
    text_config = config.get_text_config()
    text_config.vocab_size = 8
    text_config.pad_token_id = PAD_TOKEN_ID
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(config, attn_implementation="flash_attention_3")
    embed_tokens = model.get_input_embeddings().to_empty(device="cpu")
    embed_tokens(torch.tensor([PAD_TOKEN_ID])).sum().backward()
    assert embed_tokens.weight.grad[PAD_TOKEN_ID].all()
