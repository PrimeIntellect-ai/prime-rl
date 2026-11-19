from unittest.mock import Mock

import pytest
import torch.distributed.checkpoint as dcp
from torch import nn

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import CheckpointManager, ModelFiles, check_model_holds_checkpoint
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


def test_keep_last_ignores_a_step_cut_off_mid_write(tmp_path):
    ckpt_dir = get_ckpt_dir(tmp_path)
    # Steps 1 and 2 finished writing; the run crashed while writing step 3 and resumes from step 2
    for step in (1, 2, 3):
        (get_step_path(ckpt_dir, step) / "trainer").mkdir(parents=True)
    for step in (1, 2):
        (get_step_path(ckpt_dir, step) / "trainer" / ".metadata").touch()

    manager = CheckpointManager(tmp_path, CheckpointConfig(keep_last=1), Mock(spec=ModelFiles))
    manager.maybe_clean()

    assert manager.ckpt_steps == [2]
    assert not get_step_path(ckpt_dir, 1).exists()
    assert get_step_path(ckpt_dir, 2).exists()


def test_a_checkpoint_with_tensors_the_model_lacks_does_not_load(tmp_path):
    with_head = nn.ModuleDict({"body": nn.Linear(2, 2), "head": nn.Linear(2, 1)})
    dcp.save({"app": {"model": with_head.state_dict()}}, checkpoint_id=tmp_path, no_dist=True)
    check_model_holds_checkpoint(tmp_path, with_head)
    # A checkpoint load skips the tensors its template lacks, so the head would be dropped silently
    with pytest.raises(ValueError, match=r"holds 2 tensors ModuleDict lacks \(head.bias, head.weight\)"):
        check_model_holds_checkpoint(tmp_path, nn.ModuleDict({"body": nn.Linear(2, 2)}))
