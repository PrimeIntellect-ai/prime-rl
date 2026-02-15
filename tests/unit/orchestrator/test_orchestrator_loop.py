from prime_rl.orchestrator.config import EvalConfig
from prime_rl.orchestrator.orchestrator import _should_run_eval


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
