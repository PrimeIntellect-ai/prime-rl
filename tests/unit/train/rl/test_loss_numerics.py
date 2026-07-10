import math
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from prime_rl.configs.trainer import DefaultLossConfig, IPOLossConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    compute_importance_ratio_and_mismatch_kl,
    default_loss_fn,
    ipo_loss_fn,
    ref_kl_loss_fn,
)
from prime_rl.trainer.rl.token_export import _export_columns
from prime_rl.trainer.utils import raise_if_nonfinite_gradients


def _assert_nonfinite_gradient_reaches_rank_without_grad(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dp",))
        parameter = torch.nn.Parameter(
            DTensor.from_local(
                torch.ones(1),
                device_mesh=mesh,
                placements=[Shard(0)],
                shape=torch.Size([world_size]),
                stride=(1,),
            )
        )
        if rank == 0:
            parameter.grad = DTensor.from_local(
                torch.full((1,), float("inf")),
                device_mesh=mesh,
                placements=[Shard(0)],
                shape=parameter.shape,
                stride=parameter.stride(),
            )

        with pytest.raises(RuntimeError, match="Non-finite gradients detected"):
            raise_if_nonfinite_gradients([parameter])
    finally:
        dist.destroy_process_group()


def test_nonfinite_gradient_check_synchronizes_ranks_without_local_grads(tmp_path):
    mp.spawn(
        _assert_nonfinite_gradient_reaches_rank_without_grad,
        args=(2, str(tmp_path / "nonfinite_grad_store")),
        nprocs=2,
        join=True,
    )


def _inputs(
    trainer_logprob: float,
    inference_logprob: float,
    *,
    advantage: float = 1.0,
    ref_logprob: float | None = None,
) -> LossInputs:
    return LossInputs(
        trainer_logprobs=torch.tensor([trainer_logprob], dtype=torch.float32),
        inference_logprobs=torch.tensor([inference_logprob], dtype=torch.float32),
        ref_logprobs=None if ref_logprob is None else torch.tensor([ref_logprob], dtype=torch.float32),
        advantages=torch.tensor([advantage], dtype=torch.float32),
        loss_mask=torch.tensor([True]),
    )


def _assert_finite_output(loss, metrics):
    assert torch.isfinite(loss)
    for value in metrics.values():
        assert torch.isfinite(value).all()


def test_mismatch_kl_stays_finite_without_saturating_raw_ratio():
    log_ratio, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        torch.tensor([-2.0], dtype=torch.float32),
        torch.tensor([-91.0], dtype=torch.float32),
    )

    assert log_ratio.item() == pytest.approx(89.0)
    assert torch.isinf(ratio).all()
    assert torch.isfinite(mismatch_kl).all()
    assert (mismatch_kl >= 0).all()


def test_mismatch_kl_reduction_stays_finite_for_many_overflow_ratios():
    num_tokens = 1024
    output = default_loss_fn(
        LossInputs(
            trainer_logprobs=torch.full((num_tokens,), -2.0),
            inference_logprobs=torch.full((num_tokens,), -91.0),
            ref_logprobs=None,
            advantages=torch.ones(num_tokens),
            loss_mask=torch.ones(num_tokens, dtype=torch.bool),
        ),
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0),
    )

    _assert_finite_output(output.loss, output.metrics)


def test_mismatch_kl_reduction_stays_finite_for_float16_ratios():
    num_tokens = 1024
    _, _, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        torch.full((num_tokens,), -2.0, dtype=torch.float16),
        torch.full((num_tokens,), -91.0, dtype=torch.float16),
    )

    assert torch.isfinite(mismatch_kl.sum())


def test_token_export_preserves_raw_importance_ratio_semantics():
    columns = _export_columns(
        {
            "input_ids": torch.tensor([1, 2]),
            "position_ids": torch.tensor([0, 1]),
            "loss_mask": torch.tensor([True, True]),
            "advantages": torch.ones(2),
            "inference_logprobs": torch.tensor([-3.0, -91.0]),
            "env_names": ["test", "test"],
        },
        {
            "logprobs": torch.tensor([-2.0, -2.0]),
            "entropy": torch.zeros(2),
        },
        DefaultLossConfig(),
    )

    assert columns["log_importance_ratio"] == pytest.approx([1.0, 89.0])
    assert columns["importance_ratio"][0] == pytest.approx(math.e)
    assert columns["importance_ratio"][1] is None
    assert all(value is not None and math.isfinite(value) for value in columns["mismatch_kl"])


def test_default_loss_clips_unmasked_overflow_ratio():
    inputs = _inputs(-2.0, -91.0)
    inputs.trainer_logprobs.requires_grad_()
    output = default_loss_fn(
        inputs,
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert output.loss.item() == pytest.approx(-20.0)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(-20.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_default_loss_preserves_unclipped_importance_gradient():
    inputs = _inputs(-2.0, -3.0)
    inputs.trainer_logprobs.requires_grad_()
    output = default_loss_fn(
        inputs,
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )
    output.loss.backward()

    expected = -math.e
    assert output.loss.item() == pytest.approx(expected)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(expected)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(0.0)


def test_default_loss_masked_overflow_ratio_does_not_nan():
    inputs = _inputs(-1.0, -91.0)
    inputs.trainer_logprobs.requires_grad_()
    output = default_loss_fn(inputs, DefaultLossConfig(kl_tau=0.0))
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(0.0)
    assert output.metrics["is_masked"].item() == pytest.approx(1.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_ipo_loss_clips_overflow_ratio():
    inputs = _inputs(-2.0, -91.0)
    inputs.trainer_logprobs.requires_grad_()
    output = ipo_loss_fn(
        inputs,
        IPOLossConfig(ipo_threshold=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert output.loss.item() == pytest.approx(-20.0)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(-20.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_ref_kl_loss_clips_overflow_ratio():
    inputs = _inputs(-2.0, -91.0, ref_logprob=-3.0)
    inputs.trainer_logprobs.requires_grad_()
    output = ref_kl_loss_fn(inputs)
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(20.178)
    assert output.metrics["ref_kl/importance_ratio_clipped"].item() == pytest.approx(1.0)
