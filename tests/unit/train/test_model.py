import pytest
import torch

from prime_rl.configs.trainer import AttnImplementation, ModelConfig
from prime_rl.trainer.model import get_model

BS = 1
SEQ_LEN = 8

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.filterwarnings("ignore:torch.get_autocast_gpu_dtype\\(\\) is deprecated:DeprecationWarning"),
]


@pytest.fixture(params=["flash_attention_2"])
def attn(request) -> AttnImplementation:
    """
    Fixture to test different attention implementations.
    """
    try:
        # ruff: noqa: F401
        import flash_attn
    except ImportError:
        pytest.skip("Flash Attention not available")
    return request.param


@pytest.fixture
def model(attn):
    config = ModelConfig(name="Qwen/Qwen3-0.6B", attn=attn)
    return get_model(config)


def position_ids_for(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)


def test_model_to_gpu(model):
    model = model.to("cuda")


def test_model_forward(model):
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(inputs_ids, position_ids_for(inputs_ids), seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)


def test_moe_custom_impl():
    config = ModelConfig(name="PrimeIntellect/GLM-0.5B", attn="flash_attention_2")
    model = get_model(config)
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(inputs_ids, position_ids_for(inputs_ids), seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)
