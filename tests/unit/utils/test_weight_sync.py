import pytest

from prime_rl.utils.weight_sync import generation_weight_error


@pytest.mark.parametrize(
    ("active", "dirty", "ready", "required", "minimum", "status"),
    [
        ("base", False, True, False, None, None),
        ("3", False, True, True, "3", None),
        ("4", False, True, True, "3", None),
        ("2", False, True, True, "3", 503),
        ("base", False, True, True, "0", 503),
        ("3", True, True, True, "3", 503),
        ("3", False, False, True, "3", 503),
        ("3", False, True, True, None, 400),
        ("3", False, True, True, "-1", 400),
        ("3", False, True, True, "invalid", 400),
    ],
)
def test_generation_weight_admission(active, dirty, ready, required, minimum, status) -> None:
    error = generation_weight_error(
        active_version=active,
        weights_dirty=dirty,
        serving_ready=ready,
        require_version=required,
        minimum_version=minimum,
    )
    assert (error[0] if error else None) == status
