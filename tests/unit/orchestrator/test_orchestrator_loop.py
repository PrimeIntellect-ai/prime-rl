from prime_rl.orchestrator.config import EvalConfig
from prime_rl.orchestrator.orchestrator import _should_run_eval, _should_save_checkpoint


def test_should_run_eval_base_model_on_start() -> None:
    eval_config = EvalConfig(interval=10, eval_base_model=True)
    assert _should_run_eval(eval_config, ckpt_step=0, last_eval_step=-1, is_final_step=False)


def test_should_not_run_eval_base_model_when_disabled() -> None:
    eval_config = EvalConfig(interval=10, eval_base_model=False)
    assert not _should_run_eval(eval_config, ckpt_step=0, last_eval_step=-1, is_final_step=False)


def test_should_run_eval_on_resume_when_due() -> None:
    eval_config = EvalConfig(interval=10, eval_base_model=True)
    assert _should_run_eval(eval_config, ckpt_step=20, last_eval_step=10, is_final_step=False)


def test_should_skip_eval_when_already_evaluated() -> None:
    eval_config = EvalConfig(interval=10, eval_base_model=True)
    assert not _should_run_eval(eval_config, ckpt_step=20, last_eval_step=20, is_final_step=False)


def test_should_save_checkpoint_on_interval() -> None:
    assert _should_save_checkpoint(ckpt_interval=10, ckpt_step=20, is_final_step=False)


def test_should_not_save_checkpoint_when_not_due() -> None:
    assert not _should_save_checkpoint(ckpt_interval=10, ckpt_step=21, is_final_step=False)


def test_should_force_save_final_checkpoint_without_interval() -> None:
    assert _should_save_checkpoint(ckpt_interval=None, ckpt_step=20, is_final_step=True, force_final=True)


def test_should_not_save_checkpoint_without_interval_unless_forced() -> None:
    assert not _should_save_checkpoint(ckpt_interval=None, ckpt_step=20, is_final_step=True, force_final=False)


def test_should_not_save_checkpoint_at_step_zero() -> None:
    assert not _should_save_checkpoint(ckpt_interval=10, ckpt_step=0, is_final_step=True, force_final=True)
