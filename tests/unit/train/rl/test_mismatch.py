import pytest
import torch

from prime_rl.trainer.rl.mismatch import mismatch_diagnostics, parameter_change_counts


def test_mismatch_diagnostics_detects_roundoff_and_signed_zero():
    inference = torch.tensor([-1.0, -1e-4, 0.0, -2.0])
    trainer = inference.clone()
    trainer[1] = torch.nextafter(inference[1], torch.tensor(0.0))
    trainer[2] = -0.0
    metrics = mismatch_diagnostics(trainer, inference)
    assert metrics["logprob_bit_mismatch"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert metrics["logprob_nonfinite"].sum() == 0
    delta = trainer[1].double() - inference[1].double()
    assert metrics["mismatch_k3_stable"][1].item() == pytest.approx(0.5 * delta.item() ** 2, rel=1e-4, abs=0)
    assert metrics["logprob_abs_error"][1] > 0
    assert metrics["logprob_abs_error"][[0, 2, 3]].sum() == 0


def test_mismatch_diagnostics_rejects_nonfinite_matches_and_broadcasting():
    values = torch.tensor([float("nan"), float("-inf"), -0.5])
    metrics = mismatch_diagnostics(values, values)
    assert metrics["logprob_nonfinite"].tolist() == [1.0, 1.0, 0.0]
    assert metrics["logprob_bit_mismatch"].tolist() == [1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="identical shapes"):
        mismatch_diagnostics(values, values[:1])


def test_parameter_changes_distinguish_updates_and_nonfinite_values():
    before = torch.tensor([1.0, 0.0, -0.5, float("nan"), float("inf")])
    after = before.clone()
    assert parameter_change_counts(before, after).tolist() == [0, 2]
    after[0] = torch.nextafter(before[0], torch.tensor(2.0))
    after[1] = -0.0
    assert parameter_change_counts(before, after).tolist() == [2, 2]
    with pytest.raises(TypeError, match="FP32"):
        parameter_change_counts(before.bfloat16(), after.bfloat16())
    with pytest.raises(ValueError, match="identical shapes"):
        parameter_change_counts(before, after[:1])
