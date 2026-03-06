import pytest

from prime_rl.configs.sft import SFTConfig, SFTDataConfig, SFTValConfig
from prime_rl.configs.trainer import ModelConfig


def test_sft_val_config_is_valid():
    config = SFTConfig(
        val=SFTValConfig(
            interval=10,
            eval_on_start=True,
            data=SFTDataConfig(name="willcb/R1-reverse-wikipedia-paragraphs-v1-1000", splits=["train[:5%]"]),
        ),
    )
    assert config.val is not None
    assert config.val.eval_on_start is True
    assert config.val.data.name == "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"


def test_sft_val_data_requires_cp_compatible_pack_function():
    with pytest.raises(ValueError, match="Validation packing function must be 'cat' when CP is enabled"):
        SFTConfig(
            model=ModelConfig(cp=2),
            val=SFTValConfig(
                interval=10,
                data=SFTDataConfig(pack_function="stack", seq_len=256),
            ),
        )


def test_sft_val_data_requires_cp_compatible_seq_len():
    with pytest.raises(ValueError, match="Validation sequence length must be divisible by CP degree"):
        SFTConfig(
            model=ModelConfig(cp=2),
            val=SFTValConfig(
                interval=10,
                data=SFTDataConfig(seq_len=127),
            ),
        )


def test_sft_val_data_requires_cp_compatible_micro_batch_size():
    with pytest.raises(ValueError, match="Validation micro batch size must be 1 when CP is enabled"):
        SFTConfig(
            model=ModelConfig(cp=2),
            val=SFTValConfig(
                interval=10,
                data=SFTDataConfig(micro_batch_size=2),
            ),
        )
