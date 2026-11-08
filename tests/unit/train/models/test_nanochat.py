import torch
from transformers import AutoConfig

from prime_rl.configs.pretrain import DEFAULT_MODEL, DEFAULT_TOKENIZER, PretrainConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nanochat import NanochatConfig, NanochatForCausalLM


def test_pretraining_defaults_to_nanochat(tmp_path):
    """The default pretraining model is the nanochat config shipped with prime-rl, with Qwen3's tokenizer."""
    config = PretrainConfig(data={"paths": [tmp_path]}, output_dir=tmp_path)
    assert (config.model.name, config.tokenizer.name) == (DEFAULT_MODEL, DEFAULT_TOKENIZER)
    assert config.run.name.startswith("nanochat-d12--")
    model_config = AutoConfig.from_pretrained(config.model.name)
    with torch.device("meta"):
        model = NanochatForCausalLM(model_config)
    assert len(model.model.layers) == 12
    assert model.model.embed_tokens.weight.shape == (151936, 768)


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
