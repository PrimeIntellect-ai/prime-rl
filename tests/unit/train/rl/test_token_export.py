from pathlib import Path

import torch

from prime_rl.trainer.rl.token_export import TokenExporter


def test_token_exporter_marks_run_local_step_stable(tmp_path: Path):
    exporter = TokenExporter(tmp_path, rank=0)

    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "loss_mask": torch.tensor([[False, True, True]]),
        "advantages": torch.tensor([[0.0, 1.0, 1.0]]),
        "rewards": torch.tensor([[0.0, 1.0, 1.0]]),
        "inference_logprobs": torch.tensor([[0.0, -0.2, -0.3]]),
        "training_mode": "sft",
        "env_names": ["reverse_text", "reverse_text", "reverse_text"],
        "run_id": "run_alpha",
        "run_step": 7,
    }
    model_output = {
        "logprobs": torch.tensor([[0.0, -0.1, -0.4]]),
        "entropy": torch.tensor([[1.0, 2.0, 3.0]]),
    }

    exporter.export(
        step=0,
        micro_step=0,
        micro_batch=micro_batch,
        model_output=model_output,
        response_lengths=[3],
        loss_config=object(),
    )

    step_dir = tmp_path / "run_alpha" / "token_exports" / "step_7"
    assert not (step_dir / "STABLE").exists()

    # A run step can span multiple trainer steps; don't finalize until the run is
    # reported ready (its step fully consumed). Marking with the run absent is a no-op.
    exporter.mark_stable(ready_run_ids=set())
    assert not (step_dir / "STABLE").exists()

    exporter.mark_stable(ready_run_ids={"run_alpha"})
    assert (step_dir / "STABLE").exists()
