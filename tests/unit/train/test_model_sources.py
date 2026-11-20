import pytest
from transformers import AutoTokenizer, Qwen3Config

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import check_ckpt_dir, setup_tokenizer, use_resumed_model_files
from prime_rl.utils.pathing import is_checkpoint_step_dir


def test_a_checkpoint_step_dir_is_recognized_and_needs_a_config_json(tmp_path):
    step_dir = tmp_path / "step_5"
    (step_dir / "trainer").mkdir(parents=True)
    Qwen3Config().save_pretrained(tmp_path / "hf")
    assert is_checkpoint_step_dir(step_dir)
    assert not is_checkpoint_step_dir(tmp_path / "hf") and not is_checkpoint_step_dir("Qwen/Qwen3-0.6B")
    with pytest.raises(FileNotFoundError, match="has no config.json"):
        check_ckpt_dir(ModelConfig(name=str(step_dir)))
    Qwen3Config().save_pretrained(step_dir)
    check_ckpt_dir(ModelConfig(name=str(step_dir)))


def test_a_model_name_without_a_tokenizer_asks_for_tokenizer_name(tmp_path):
    Qwen3Config().save_pretrained(tmp_path)
    with pytest.raises(FileNotFoundError, match="set tokenizer.name"):
        setup_tokenizer(TokenizerConfig(name=str(tmp_path)), model_name=str(tmp_path))
    AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    assert len(setup_tokenizer(TokenizerConfig(name=str(tmp_path)), model_name=str(tmp_path))) > 1


def test_a_run_from_a_checkpoint_resumes_with_the_files_of_its_own_checkpoint(tmp_path):
    source, step_dir = tmp_path / "source" / "step_5", tmp_path / "run" / "step_2"
    model = ModelConfig(name=str(source))
    tokenizer = TokenizerConfig(name=str(source))
    use_resumed_model_files(model, tokenizer, step_dir)  # a checkpoint without config.json
    assert model.name == tokenizer.name == str(source)
    Qwen3Config().save_pretrained(step_dir)
    use_resumed_model_files(model, tokenizer, step_dir)
    assert model.name == tokenizer.name == str(step_dir)
