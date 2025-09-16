import pytest
import torch

from prime_rl.trainer.models import init_model


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_init_model(model_name):
    model = init_model(pretrained_model_name_or_path=model_name).to("cuda")

    inputs_ids = torch.randint(0, 256, (1, 1024)).to("cuda")
    outputs = model(input_ids=inputs_ids).logits
    assert outputs.shape == (1, 1024, model.model_args.vocab_size)
