from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import CheckpointManager
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


def test_keep_last_ignores_a_step_cut_off_mid_write(tmp_path):
    ckpt_dir = get_ckpt_dir(tmp_path)
    # Steps 1 and 2 finished writing; the run crashed while writing step 3 and resumes from step 2
    for step in (1, 2, 3):
        (get_step_path(ckpt_dir, step) / "trainer").mkdir(parents=True)
    for step in (1, 2):
        (get_step_path(ckpt_dir, step) / "trainer" / ".metadata").touch()

    manager = CheckpointManager(tmp_path, CheckpointConfig(keep_last=1))
    manager.maybe_clean()

    assert manager.ckpt_steps == [2]
    assert not get_step_path(ckpt_dir, 1).exists()
    assert get_step_path(ckpt_dir, 2).exists()
