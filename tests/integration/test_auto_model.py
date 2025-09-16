import pytest
import torch

from prime_rl.trainer.models import from_pretrained


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test(model_name):
    model = from_pretrained(pretrained_model_name_or_path=model_name).to("cuda")

    inputs_ids = torch.randint(0, 256, (1, 1024)).to("cuda")
    outputs = model(input_ids=inputs_ids)
    assert outputs.shape == (1, 1024, model.model_args.vocab_size)
