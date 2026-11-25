import pytest
import torch
import torch.distributed as dist

from prime_rl.configs.trainer import AttnImplementation, DebugModelConfig, ModelConfig
from prime_rl.trainer.model import get_model, load_dcp_from_hf
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import get_free_port

BS = 1
SEQ_LEN = 8

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.filterwarnings("ignore:torch.get_autocast_gpu_dtype\\(\\) is deprecated:DeprecationWarning"),
]


@pytest.fixture(scope="module", autouse=True)
def process_group():
    if dist.is_initialized():
        yield
        return
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{get_free_port()}", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


def load_model(config: ModelConfig) -> torch.nn.Module:
    """Build on the meta device and load the checkpoint, as the trainer does."""
    model = get_model(config)
    # Mirror setup_model: the custom Qwen3 forward calls lm_head with
    # (hidden_states, labels, temperature=...), which only VanillaOutputLinear
    # / FusedOutputLinear accept. Plain nn.Linear errors with
    # `Linear.forward() got an unexpected keyword argument 'temperature'`.
    inject_prime_lm_head(model, chunk_size=None)
    model.to_empty(device="cuda")
    model.tie_weights()
    load_dcp_from_hf(model, config)
    return model


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
    return load_model(ModelConfig(name="Qwen/Qwen3-0.6B", attn=attn))


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
    model = get_model(ModelConfig(name="PrimeIntellect/GLM-0.5B", attn="flash_attention_2"))
    inject_prime_lm_head(model, chunk_size=None)
    model.to_empty(device="cuda")
    model.tie_weights()
    model.init_buffers_post_meta()
    model.initialize_weights()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids, seq_lens=torch.tensor([SEQ_LEN], device="cuda"))
        logits = outputs["logits"]

        assert logits.shape == (BS, SEQ_LEN, model.config.vocab_size)
