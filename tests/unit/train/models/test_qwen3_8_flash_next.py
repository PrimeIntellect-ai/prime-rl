import pytest
import torch

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.trainer.model import freeze_ngram_embedding
from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.vlm import get_language_model


@pytest.fixture(params=[False, True], ids=["text", "composite"])
def config(request):
    text = Qwen3_8FlashNextTextConfig(
        vocab_size=32,
        bos_token_id=31,
        eos_token_id=31,
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        hc_count=2,
        hc_lowrank=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        ple_layer_ids=[1],
        ple_embed_dim=16,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=3,
    )
    return Qwen3_8FlashNextConfig(text_config=text) if request.param else text


@pytest.mark.parametrize("shard_count", [3, 128])
def test_config_and_checkpoint_roundtrip(config, shard_count):
    getattr(config, "text_config", config).split_ngram_parts = shard_count
    restored = type(config).from_dict(config.to_dict())
    model = Qwen3_8FlashNextForCausalLM(restored)
    model.init_buffers_post_meta()
    assert len(get_language_model(model).layers) == 1
    original = {
        name: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for name, value in model.state_dict().items()
    }
    converted = dict(original)
    operations = model.conversion_chain(restored)
    apply_prime_to_hf(converted, operations)
    assert model.is_hf_state_dict(converted)
    apply_hf_to_prime(converted, operations)
    assert converted.keys() == original.keys()
    for name, value in original.items():
        torch.testing.assert_close(converted[name], value, rtol=0, atol=0)


def test_freeze_ngram_embedding(config):
    model = Qwen3_8FlashNextForCausalLM(config)
    freeze_ngram_embedding(model)

    table = get_language_model(model).layers[0].ple.ple_embedding.ngram_embedding.weight
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    assert trainable == {id(param) for param in model.parameters() if param is not table}

    optimizer, _ = setup_optimizer(
        AdamWConfig(weight_decay=0.0),
        list(model.named_parameters()),
        ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1),
    )
    assert {id(param) for group in optimizer.param_groups for param in group["params"]} == trainable
