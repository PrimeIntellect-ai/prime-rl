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


def test_resume_rejects_a_checkpoint_that_trained_a_now_frozen_weight(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from prime_rl.trainer import ckpt

    model = SimpleNamespace(frozen_weight_keys=lambda: {"model.embed_tokens.weight"})

    def checkpoint(*keys):
        metadata = SimpleNamespace(state_dict_metadata=dict.fromkeys(keys))
        monkeypatch.setattr(ckpt, "FileSystemReader", lambda path: SimpleNamespace(read_metadata=lambda: metadata))

    checkpoint("app.model.model.norm.weight")  # saved with the weight frozen: left out
    ckpt.check_frozen_weights_not_trained(model, tmp_path)
    for saved in ("app.optimizers.state.model.embed_tokens.weight.exp_avg", "app.model.model.embed_tokens.weight"):
        # Trained unfrozen: AdamW leaves optimizer state; SignSGD only the weight
        checkpoint(saved)
        with pytest.raises(ValueError, match="trained weights this run freezes"):
            ckpt.check_frozen_weights_not_trained(model, tmp_path)
