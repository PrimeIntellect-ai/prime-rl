import math

import pytest
from pydantic import ValidationError

from prime_rl.configs.shared import FileSystemTransportConfig, ZMQTransportConfig


@pytest.mark.parametrize("config_type", [FileSystemTransportConfig, ZMQTransportConfig])
def test_rollout_send_timeout_has_finite_positive_default(config_type):
    timeout = config_type().send_timeout_seconds

    assert timeout == 300.0
    assert math.isfinite(timeout)


@pytest.mark.parametrize("config_type", [FileSystemTransportConfig, ZMQTransportConfig])
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_rollout_send_timeout_rejects_nonpositive_or_nonfinite_values(config_type, timeout: float):
    with pytest.raises(ValidationError):
        config_type(send_timeout_seconds=timeout)
