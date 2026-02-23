import pytest

from prime_rl.trainer.config import ModelConfig
from prime_rl.trainer.sft.config import SFTDataConfig, SFTEvalConfig, SFTTrainerConfig


def test_sft_eval_requires_val_data():
    with pytest.raises(ValueError, match="both eval and val_data"):
        SFTTrainerConfig(eval=SFTEvalConfig(interval=10, num_batches=2))


def test_sft_val_data_requires_eval():
    with pytest.raises(ValueError, match="both eval and val_data"):
        SFTTrainerConfig(val_data=SFTDataConfig())


def test_sft_eval_with_val_data_is_valid():
    config = SFTTrainerConfig(
        eval=SFTEvalConfig(interval=10, num_batches=2),
        val_data=SFTDataConfig(name="willcb/R1-reverse-wikipedia-paragraphs-v1-1000", splits=["train[:5%]"]),
    )
    assert config.eval is not None
    assert config.val_data is not None


def test_sft_val_data_requires_cp_compatible_pack_function():
    with pytest.raises(ValueError, match="Validation packing function must be 'cat' when CP is enabled"):
        SFTTrainerConfig(
            model=ModelConfig(cp=2),
            eval=SFTEvalConfig(interval=10, num_batches=2),
            val_data=SFTDataConfig(pack_function="stack", seq_len=256),
        )


def test_sft_val_data_requires_cp_compatible_seq_len():
    with pytest.raises(ValueError, match="Validation sequence length must be divisible by CP degree"):
        SFTTrainerConfig(
            model=ModelConfig(cp=2),
            eval=SFTEvalConfig(interval=10, num_batches=2),
            val_data=SFTDataConfig(seq_len=127),
        )


def test_sft_val_data_requires_cp_compatible_micro_batch_size():
    with pytest.raises(ValueError, match="Validation micro batch size must be 1 when CP is enabled"):
        SFTTrainerConfig(
            model=ModelConfig(cp=2),
            eval=SFTEvalConfig(interval=10, num_batches=2),
            val_data=SFTDataConfig(micro_batch_size=2),
        )
