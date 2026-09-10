import pytest
import torch

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

BS = 1
SEQ_LEN = 8

pytestmark = [pytest.mark.gpu]


@pytest.mark.skip(reason="need special token for meta stuff in ci")
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_model_forward_custom_impl(model_name):
    config = ModelConfig(name=model_name, impl="custom", attn="flash_attention_2")
    model = get_model(config)
    # we need to wrap the lm head as custom forward only works with it, this is done in setup_model
    inject_prime_lm_head(model, chunk_size=None)
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids, seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)
