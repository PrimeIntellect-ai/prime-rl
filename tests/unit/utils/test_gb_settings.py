import pytest

from prime_rl.utils.gb_settings import (
    GBSettings,
    GBSettingsError,
    check_resume_step,
    load_gb_settings,
    metric_name,
    validate_orchestrator_settings,
    validate_trainer_settings,
)

ENV = "genebench-heldout"


def test_no_variables_is_inert():
    s, warnings = load_gb_settings({}, environ={"PATH": "/bin"})
    assert s == GBSettings()
    assert not s.active and not s.records_enabled and not s.keep3 and not s.final_export
    assert warnings == []
    # Stock retention settings are fine when keep-3 is off
    validate_trainer_settings(s, ckpt_enabled=True, keep_last=1, keep_interval=40, has_hf_token=False)
    assert validate_orchestrator_settings(s, eval_intervals=None, ckpt_interval=None, retrigger_on_resume=False) == []


def test_config_env_vars_win_over_the_environment():
    s, _ = load_gb_settings(
        {"GB_KEEP3": "1", "GB_MAX_EVAL_ERROR": "0.1"},
        environ={"GB_KEEP3": "0", "GB_FINAL_EXPORT": "true", "GB_EXPORT_REPO_PREFIX": "Org/Model-rl"},
    )
    assert s.keep3 and s.final_export and s.max_eval_error == 0.1
    assert s.export_repo_prefix == "Org/Model-rl"
    assert s.sources == {
        "GB_KEEP3": "config.env_vars",
        "GB_MAX_EVAL_ERROR": "config.env_vars",
        "GB_FINAL_EXPORT": "os.environ",
        "GB_EXPORT_REPO_PREFIX": "os.environ",
    }
    assert "keep3=True" in s.describe()


def test_metric_keys():
    s, _ = load_gb_settings(
        {"GB_KEEP3": "1", "GB_KEEP_METRICS": "train_reward, eval/x/all/agent/rewards/pass/mean"}, {}
    )
    assert s.metric_keys == {
        "train_reward": f"eval/{ENV}/all/agent/metrics/train_reward/mean",
        "pass": "eval/x/all/agent/rewards/pass/mean",
    }
    assert GBSettings().metric_keys == {
        "train_reward": f"eval/{ENV}/all/agent/metrics/train_reward/mean",
        "hurdle_frac": f"eval/{ENV}/all/agent/metrics/hurdle_frac/mean",
    }
    assert metric_name("eval/e/all/agent/solved_some") == "solved_some"


def test_unknown_keys():
    with pytest.raises(GBSettingsError, match="unknown GB_"):
        load_gb_settings({"GB_KEEP_3": "1"}, {})
    s, warnings = load_gb_settings({}, {"GB_SOMETHING": "1"})
    assert not s.active and "GB_SOMETHING" in warnings[0]


@pytest.mark.parametrize(
    "env_vars",
    [
        {"GB_KEEP3": "maybe"},
        {"GB_MAX_EVAL_ERROR": "x"},
        {"GB_MAX_EVAL_ERROR": "1.5"},
        {"GB_MAX_EVAL_ERROR": "nan"},
        {"GB_FINAL_EVAL_TIMEOUT_S": "0"},
        {"GB_WAIT_FOR_EXPORT_S": "-1"},
        {"GB_EXPORT_REPO_PREFIX": "no-org"},
        {"GB_EXPORT_REPO_PREFIX": "Org/Model-step"},
        {"GB_EXPORT_REPO_PREFIX": "Org/Model-step200"},
        {"GB_KEEP_METRICS": " , "},
        {"GB_KEEP_METRICS": "train_reward,train_reward"},
        {"GB_KEEP_METRICS": "bad name"},
        {"GB_EVAL_ENV": "a/b"},
    ],
)
def test_invalid_values(env_vars):
    with pytest.raises(GBSettingsError):
        load_gb_settings(env_vars, {})


def test_settings_without_a_feature_warn():
    s, warnings = load_gb_settings({"GB_EVAL_ENV": "x"}, {})
    assert s.active and not s.records_enabled
    assert "no effect" in warnings[0]


def keep3(**extra):
    return load_gb_settings({"GB_KEEP3": "1", **extra}, {})[0]


def test_trainer_validation():
    validate_trainer_settings(keep3(), ckpt_enabled=True, keep_last=None, keep_interval=None, has_hf_token=False)
    with pytest.raises(GBSettingsError, match="keep_interval"):
        validate_trainer_settings(keep3(), ckpt_enabled=True, keep_last=None, keep_interval=40, has_hf_token=True)
    with pytest.raises(GBSettingsError, match="keep_last"):
        validate_trainer_settings(keep3(), ckpt_enabled=True, keep_last=1, keep_interval=None, has_hf_token=True)
    with pytest.raises(GBSettingsError, match="checkpointing"):
        validate_trainer_settings(keep3(), ckpt_enabled=False, keep_last=None, keep_interval=None, has_hf_token=True)
    export = keep3(GB_FINAL_EXPORT="1")
    with pytest.raises(GBSettingsError, match="HF_TOKEN"):
        validate_trainer_settings(export, ckpt_enabled=True, keep_last=None, keep_interval=None, has_hf_token=False)
    validate_trainer_settings(export, ckpt_enabled=True, keep_last=None, keep_interval=None, has_hf_token=True)
    only_export = load_gb_settings({"GB_FINAL_EXPORT": "1"}, {})[0]
    with pytest.raises(GBSettingsError, match="needs GB_KEEP3"):
        validate_trainer_settings(only_export, ckpt_enabled=True, keep_last=None, keep_interval=None, has_hf_token=True)


def test_orchestrator_validation():
    assert (
        validate_orchestrator_settings(keep3(), eval_intervals={ENV: 20}, ckpt_interval=20, retrigger_on_resume=True)
        == []
    )
    with pytest.raises(GBSettingsError, match="not an eval env"):
        validate_orchestrator_settings(
            keep3(), eval_intervals={"other": 20}, ckpt_interval=20, retrigger_on_resume=True
        )
    with pytest.raises(GBSettingsError, match="not an eval env"):
        validate_orchestrator_settings(keep3(), eval_intervals=None, ckpt_interval=20, retrigger_on_resume=True)
    with pytest.raises(GBSettingsError, match="checkpointing"):
        validate_orchestrator_settings(keep3(), eval_intervals={ENV: 20}, ckpt_interval=None, retrigger_on_resume=True)
    warnings = validate_orchestrator_settings(
        keep3(), eval_intervals={ENV: 40}, ckpt_interval=20, retrigger_on_resume=False
    )
    assert len(warnings) == 2


def test_rendered_trainer_config_accepts_gb_env_vars():
    """The hosted backend renders [trainer.env_vars] into the trainer config's env_vars table."""
    from prime_rl.configs.trainer import TrainerConfig

    env_vars = {"GB_KEEP3": "1", "GB_FINAL_EXPORT": "1"}
    assert TrainerConfig.model_fields["env_vars"].annotation is not None
    config = TrainerConfig.model_validate({"env_vars": env_vars, "model": {"name": "x"}})
    s, _ = load_gb_settings(config.env_vars, {})
    assert s.keep3 and s.final_export


def test_expect_fork_step():
    s, warnings = load_gb_settings({"GB_KEEP3": "1", "GB_EXPECT_FORK_STEP": "200"}, environ={})
    assert s.expect_fork_step == 200 and warnings == []
    check_resume_step(s, 200)  # the fork
    check_resume_step(s, 260)  # a restart from the run's own checkpoint
    with pytest.raises(GBSettingsError, match="would start from scratch"):
        check_resume_step(s, None)  # resume.dir dropped by the backend
    with pytest.raises(GBSettingsError, match="from step 180"):
        check_resume_step(s, 180)
    check_resume_step(GBSettings(), None)  # unset: inert
    # Alone it is a guard, not a feature: no "no effect" warning.
    assert load_gb_settings({"GB_EXPECT_FORK_STEP": "200"}, environ={})[1] == []
    for bad in ("2x0", "0", "-5"):
        with pytest.raises(GBSettingsError, match="GB_EXPECT_FORK_STEP"):
            load_gb_settings({"GB_EXPECT_FORK_STEP": bad}, environ={})
