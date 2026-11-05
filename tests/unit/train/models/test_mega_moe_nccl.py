import pytest

from prime_rl.trainer.models.layers.mega_moe import check_nccl_max_ctas


@pytest.mark.parametrize("name", ["NCCL_MIN_CTAS", "NCCL_MIN_NCHANNELS"])
@pytest.mark.parametrize("value", ["9", "many"])
def test_nccl_minimum_above_max_ctas_is_refused(monkeypatch, name, value):
    monkeypatch.setenv("NCCL_MAX_CTAS", "8")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        check_nccl_max_ctas(num_reserved_sms=16, num_concurrent=2)


def test_nccl_minimum_within_max_ctas_is_accepted(monkeypatch):
    monkeypatch.setenv("NCCL_MAX_CTAS", "8")
    monkeypatch.setenv("NCCL_MIN_CTAS", "8")
    check_nccl_max_ctas(num_reserved_sms=16, num_concurrent=2)
