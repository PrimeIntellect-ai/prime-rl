"""The registry covers every supported model_type and yields valid chains."""

from types import SimpleNamespace

import pytest

from prime_rl.trainer.models.conversion_chains import (
    CONVERSION_CHAIN_MODEL_TYPES,
    build_conversion_chain,
)
from prime_rl.trainer.models.conversion_ops import ConvOp

# gpt_oss is the identity conversion -> empty chain; everything else is non-empty.
_IDENTITY = {"gpt_oss"}


@pytest.mark.parametrize("model_type", sorted(CONVERSION_CHAIN_MODEL_TYPES))
def test_registry_builds_chain(model_type):
    config = SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=2,
        layers_block_type=["mamba", "moe"],
    )
    ops = build_conversion_chain(model_type, config)
    assert isinstance(ops, list)
    assert all(isinstance(op, ConvOp) for op in ops)
    if model_type in _IDENTITY:
        assert ops == []
    else:
        assert ops, f"{model_type} produced an empty chain"


def test_unknown_model_type_raises():
    with pytest.raises(NotImplementedError):
        build_conversion_chain("not_a_model", SimpleNamespace())
