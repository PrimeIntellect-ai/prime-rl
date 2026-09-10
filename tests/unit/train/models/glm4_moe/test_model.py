import pytest
import torch

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

BS = 1
SEQ_LEN = 8

pytestmark = [pytest.mark.gpu]


def test_moe_custom_impl():
    config = ModelConfig(name="PrimeIntellect/GLM-0.5B", attn="flash_attention_2", impl="custom")
    model = get_model(config)
    model = model.to("cuda")
    # we need to wrap the lm head as custom forward only works with it, this is done in setup_model
    inject_prime_lm_head(model, chunk_size=None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids, seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)
