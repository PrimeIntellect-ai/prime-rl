from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.worker.weight_transfer import check_update_covers_dummy_weights


def test_an_update_must_cover_every_parameter_of_a_dummy_loaded_model():
    expected = {"weight", "bias"}
    model_config = SimpleNamespace(quantization=None)
    dummy = SimpleNamespace(load_config=SimpleNamespace(load_format="dummy"))
    check_update_covers_dummy_weights(expected, {"weight", "bias"}, model_config, dummy)
    with pytest.raises(ValueError, match="does not cover 1 parameters of the dummy-loaded model: bias"):
        check_update_covers_dummy_weights(expected, {"weight"}, model_config, dummy)
    # An engine loaded from disk keeps the disk values of what an update leaves out.
    check_update_covers_dummy_weights(
        expected, {"weight"}, model_config, SimpleNamespace(load_config=SimpleNamespace(load_format="auto"))
    )
