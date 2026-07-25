import pytest
import torch
from torch.distributed.checkpoint.state_dict import _get_fqns
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFGptOssForCausalLM

from prime_rl.configs.trainer import AttnImplementation, ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import GroupedExperts

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


def test_moe_checkpoint_format_keeps_split_gate_up_weights():
    experts = GroupedExperts(dim=16, hidden_dim=8, num_experts=4)

    assert set(dict(experts.named_parameters())) == {"input_weight", "w2"}
    checkpoint = {name: torch.randn_like(weight) for name, weight in experts.state_dict().items()}
    assert set(checkpoint) == {"w1", "w2", "w3"}

    experts.load_state_dict(checkpoint)
    torch.testing.assert_close(experts.input_weight, torch.cat((checkpoint["w1"], checkpoint["w3"]), dim=1))
    torch.testing.assert_close(experts.w2, checkpoint["w2"])

    model = torch.nn.Sequential(experts)
    assert _get_fqns(model, "0.w1") == {"0.w1"}
    assert _get_fqns(model, "0.w3") == {"0.w3"}


def test_gpt_oss_checkpoint_format_matches_hf():
    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=4,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_experts_per_tok=2,
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
    )

    with torch.device("meta"):
        hf_model = HFGptOssForCausalLM(config)
        prime_model = PrimeRLGptOssForCausalLM(config)

    hf_state_dict = hf_model.state_dict()
    prime_state_dict = prime_model.state_dict()
    assert set(prime_state_dict) == set(hf_state_dict)
    for name, tensor in prime_state_dict.items():
        assert tensor.shape == hf_state_dict[name].shape, name


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


@pytest.mark.skip(reason="Sequence packing for Qwen not working.")
@pytest.mark.parametrize("correct_position_ids", [True, False])
def test_model_with_sequence_packing(model, correct_position_ids):
    """
    The goal of this test is to check that the sequence packing works correctly.

    The idea is that is to check that the logits is the same when doing

    [B, seq]  and doing [1, B*seq] with the proper masking.

    """
    if model.config.attn != "flash_attention_2":
        pytest.skip("Test only works with flash attention")

    model = model.to("cuda")
    inputs = [0, 1, 2, 3]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.Tensor(inputs).repeat(1, 1).int().to("cuda")
        outputs = model(input_ids=inputs_ids, seq_lens=torch.tensor([len(inputs)], device="cuda"))
        output_base = outputs["logits"]

        assert output_base.shape == (1, len(inputs), model.config.vocab_size)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.Tensor(inputs + inputs).repeat(1, 1).int().to("cuda")
        if correct_position_ids:
            position_ids = torch.Tensor([0, 1, 2, 3, 0, 1, 2, 3]).repeat(1, 1).int().to("cuda")
            # should work
        else:
            position_ids = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(1, 1).int().to("cuda")
            # should fail
        outputs = model(
            input_ids=inputs_ids,
            position_ids=position_ids,
            seq_lens=torch.tensor([len(inputs), len(inputs)], device="cuda"),
        )
        outputs_packed = outputs["logits"]

        assert outputs_packed.shape == (1, 2 * len(inputs), model.config.vocab_size)

    output_packed_left = outputs_packed[:, : len(inputs), :]
    output_packed_right = outputs_packed[:, len(inputs) :, :]

    assert output_packed_left.shape == output_base.shape == output_packed_right.shape

    if correct_position_ids:
        torch.testing.assert_close(output_packed_left, output_base)
        torch.testing.assert_close(output_packed_right, output_base)
    else:
        torch.testing.assert_close(output_packed_left, output_base)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(output_packed_right, output_base)


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
