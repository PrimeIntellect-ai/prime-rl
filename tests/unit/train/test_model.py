import pytest
import torch
import torch.distributed as dist

from prime_rl.configs.trainer import AttnImplementation, DebugModelConfig, ModelConfig
from prime_rl.trainer.model import clear_padding_idx, get_model, initialize_from_scratch
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

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
    model = get_model(config)
    # Mirror setup_model: the custom Qwen3 forward calls lm_head with
    # (hidden_states, labels, temperature=...), which only VanillaOutputLinear
    # / FusedOutputLinear accept. Plain nn.Linear errors with
    # `Linear.forward() got an unexpected keyword argument 'temperature'`.
    inject_prime_lm_head(model, chunk_size=None)
    return model


def test_model_to_gpu(model):
    model = model.to("cuda")


def test_model_forward(model):
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids, seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)


def test_model_with_position_ids(model):
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).repeat(BS, 1).to("cuda")

        outputs = model(
            input_ids=inputs_ids,
            position_ids=position_ids,
            seq_lens=torch.tensor([SEQ_LEN], device="cuda"),
        )
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)


def test_moe_custom_impl():
    config = ModelConfig(name="PrimeIntellect/GLM-0.5B", attn="flash_attention_2")
    model = get_model(config)
    model = model.to("cuda")
    # we need to wrap the lm head as custom forward only works with it, this is done in setup_model
    inject_prime_lm_head(model, chunk_size=None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids, seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)


def test_initialize_from_scratch_initializes_the_padding_embedding(attn):
    config = ModelConfig(name="Qwen/Qwen3-0.6B", attn=attn, debug=DebugModelConfig(num_layers=1))
    model = get_model(config, device=torch.device("meta"), dtype=torch.float32)
    padding_idx = model.model.embed_tokens.padding_idx
    assert padding_idx is not None
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29513", world_size=1, rank=0)
    try:
        clear_padding_idx(model)
        initialize_from_scratch(model, seed=0)
    finally:
        dist.destroy_process_group()
    embedding = model.model.embed_tokens
    assert embedding.padding_idx is None
    assert embedding.weight[padding_idx].abs().sum() > 0
