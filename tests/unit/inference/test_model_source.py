import pytest
from transformers import Qwen3Config

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.server import setup_model_source


@pytest.fixture
def step_dir(tmp_path):
    step_dir = tmp_path / "checkpoints" / "step_5"
    (step_dir / "trainer").mkdir(parents=True)
    Qwen3Config().save_pretrained(step_dir)
    return step_dir


def test_a_checkpoint_step_dir_starts_inference_from_placeholder_weights(step_dir):
    config = InferenceConfig(vllm={"model": str(step_dir)})
    setup_model_source(config)
    assert config.vllm.load_format == "dummy"

    explicit = InferenceConfig(vllm={"model": str(step_dir), "load_format": "auto"})
    setup_model_source(explicit)
    assert explicit.vllm.load_format == "auto"

    hf = InferenceConfig(vllm={"model": "Qwen/Qwen3-0.6B"})
    setup_model_source(hf)
    assert getattr(hf.vllm, "load_format", None) is None


@pytest.mark.parametrize(
    ("vllm", "match"),
    [
        ({"enable_lora": True}, "LoRA sends only the adapter"),
        ({"speculative_config": {"method": "mtp"}}, "reads its draft weights from model.name"),
    ],
)
def test_a_checkpoint_step_dir_rejects_what_needs_hf_weights(step_dir, vllm: dict, match: str):
    with pytest.raises(ValueError, match=match):
        setup_model_source(InferenceConfig(vllm={"model": str(step_dir), **vllm}))
