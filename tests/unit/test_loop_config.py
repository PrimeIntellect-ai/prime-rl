import pytest
from pydantic import ValidationError

from prime_rl.configs.trainer import LoopConfig, LoRAConfig, ModelConfig, ScratchInit


def test_loop_config_defaults_and_bounds():
    cfg = LoopConfig()
    assert cfg.num_loops == 2
    assert cfg.start_layer == 6
    assert cfg.resolve_bounds(52) == (6, 48)

    cfg_neg = LoopConfig(start_layer=4, end_layer=-2)
    assert cfg_neg.resolve_bounds(52) == (4, 50)

    with pytest.raises(ValueError, match="Invalid loop bounds"):
        LoopConfig(start_layer=48, end_layer=48).resolve_bounds(52)


def test_loop_config_rejects_invalid_combinations():
    with pytest.raises(ValidationError):
        LoopConfig(num_loops=1)

    with pytest.raises(ValidationError, match="cannot both be set"):
        ModelConfig(loop=LoopConfig(), lora=LoRAConfig())

    with pytest.raises(ValidationError, match="freeze_base"):
        ModelConfig(init=ScratchInit(), loop=LoopConfig(freeze_base=True))

    m = ModelConfig(init=ScratchInit(), loop=LoopConfig(freeze_base=False))
    assert m.loop is not None and not m.loop.freeze_base
