from copy import deepcopy

import orjson
import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.pathing import create_attempt_dirs, write_launch_artifacts
from prime_rl.utils.resume import check_resume_config


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (RLConfig, {"trainer": {}, "orchestrator": {"train": {"source": [{"env": {"id": "single_agent"}}]}}}),
        (SFTConfig, {}),
    ],
)
@pytest.mark.parametrize("skip_checks", [False, True])
def test_training_resume_checks_saved_configs(tmp_path, config_cls, kwargs, skip_checks):
    kwargs = kwargs | {"run": {"name": "resume-test"}, "output_dir": tmp_path}
    saved = config_cls(**deepcopy(kwargs))
    config_dir, _ = create_attempt_dirs(saved.run_dir)
    write_launch_artifacts(config_dir, "training", saved)
    snapshot = config_dir / "launcher.json"
    current = config_cls(**deepcopy(kwargs), resume={"step": 10, "skip_checks": skip_checks})

    # RL's top-level resume is copied into trainer and orchestrator.
    check_resume_config(current, [snapshot])
    saved_dict = orjson.loads(snapshot.read_bytes())
    saved_dict["log"]["level"] = "debug"
    snapshot.write_bytes(orjson.dumps(saved_dict))
    newest = tmp_path / "newest.json"
    newest.write_bytes(orjson.dumps(current.model_dump(mode="json")))
    if skip_checks:
        check_resume_config(current, [snapshot, newest, tmp_path / "missing.json"])
    else:
        with pytest.raises(ValueError, match=r"in \[log\].*--resume.skip-checks"):
            check_resume_config(current, [snapshot, newest])


def test_launch_snapshot_preserves_full_submission_config(tmp_path):
    submission = SFTConfig(run={"name": "resume-test"}, slurm={}, dry_run=True)
    config_dir, _ = create_attempt_dirs(tmp_path)
    write_launch_artifacts(config_dir, "sft", submission)
    snapshot = config_dir / "launcher.json"
    saved = snapshot.read_bytes()
    child = submission.model_copy(update={"slurm": None, "dry_run": False})
    write_launch_artifacts(config_dir, "sft", child)
    assert snapshot.read_bytes() == saved
    assert orjson.loads(saved) == submission.model_dump(mode="json")
