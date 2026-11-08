from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import AutoConfig, PreTrainedTokenizerFast

from prime_rl.trainer.model import check_vocab_size
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nanochat import NanochatConfig, NanochatForCausalLM

EXAMPLE_MODEL = Path(__file__).parents[4] / "examples" / "extra" / "pretrain" / "nanochat-d12"


def test_nanochat_builds_from_the_example_config():
    with torch.device("meta"):
        model = NanochatForCausalLM(AutoConfig.from_pretrained(EXAMPLE_MODEL))
    assert len(model.model.layers) == 12
    assert model.model.embed_tokens.weight.shape == (50304, 768)


def test_nanochat_trains_on_cpu():
    """A random-init forward and backward over two packed documents."""
    config = NanochatConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = NanochatForCausalLM(config)
    model.initialize_weights()
    inject_prime_lm_head(model, chunk_size=None)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    loss = model(input_ids=input_ids, labels=torch.roll(input_ids, -1), seq_lens=torch.tensor([7, 5]))["loss"]
    loss.backward()
    assert loss.isfinite()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0


def test_a_tokenizer_larger_than_the_vocabulary_fails():
    tokens = {f"t{i}": i for i in range(64)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(tokens, unk_token="t0")))
    check_vocab_size(NanochatConfig(vocab_size=64), tokenizer)
    with pytest.raises(ValueError, match="64 tokens, more than the model's vocab_size \\(32\\)"):
        check_vocab_size(NanochatConfig(vocab_size=32), tokenizer)
