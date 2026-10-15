import subprocess
from copy import deepcopy

import orjson
import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints.rl import write_slurm_script as write_rl_script
from prime_rl.entrypoints.sft import write_slurm_script as write_sft_script
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
@pytest.mark.parametrize("resume_dry_run", [False, True])
def test_training_resume_checks_saved_configs(tmp_path, config_cls, kwargs, skip_checks, resume_dry_run):
    kwargs = kwargs | {"run": {"name": "resume-test"}, "output_dir": tmp_path}
    dry_run = config_cls(**deepcopy(kwargs), dry_run=True, log={"level": "debug"})
    dry_run_dir, _ = create_attempt_dirs(dry_run.run_dir)
    write_launch_artifacts(dry_run_dir, "training", dry_run)
    dry_run_snapshot = dry_run_dir / "launcher.json"
    saved = config_cls(**deepcopy(kwargs))
    config_dir, _ = create_attempt_dirs(saved.run_dir)
    write_launch_artifacts(config_dir, "training", saved)
    snapshot = config_dir / "launcher.json"
    current = config_cls(**deepcopy(kwargs), dry_run=resume_dry_run, resume={"step": 10, "skip_checks": skip_checks})

    # RL's top-level resume is copied into trainer and orchestrator.
    check_resume_config(current, [dry_run_snapshot, snapshot])
    saved_dict = orjson.loads(snapshot.read_bytes())
    saved_dict["log"]["level"] = "debug"
    snapshot.write_bytes(orjson.dumps(saved_dict))
    newest = tmp_path / "newest.json"
    newest.write_bytes(orjson.dumps(current.model_dump(mode="json")))
    if skip_checks:
        check_resume_config(current, [dry_run_snapshot, snapshot, newest, tmp_path / "missing.json"])
    else:
        with pytest.raises(ValueError, match=r"in \[log\].*--resume.skip-checks"):
            check_resume_config(current, [dry_run_snapshot, snapshot, newest])


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (RLConfig, {"trainer": {}, "orchestrator": {"train": {"source": [{"env": {"id": "single_agent"}}]}}}),
        (SFTConfig, {}),
    ],
)
def test_launch_snapshot_preserves_full_submission_config(tmp_path, config_cls, kwargs):
    template = tmp_path / "custom.sbatch.j2"
    header = "#!/bin/bash\n\n#SBATCH --job-name=resume-test\n \t\n  # Custom setup\n#SBATCH --nodes=1\n\n"
    template.write_text(header + "true\n")
    submission = config_cls(
        **deepcopy(kwargs),
        output_dir=tmp_path,
        run={"name": "resume-test"},
        slurm={"template_path": template},
        dry_run=True,
    )
    config_dir, log_dir = create_attempt_dirs(tmp_path / "run 'with spaces")
    write_launch_artifacts(config_dir, "sft", submission)
    snapshot = config_dir / "launcher.json"
    saved = snapshot.read_bytes()
    child = submission.model_copy(update={"slurm": None, "dry_run": False})
    write_launch_artifacts(config_dir, "sft", child)
    assert snapshot.read_bytes() == saved
    assert orjson.loads(saved) == submission.model_dump(mode="json")

    current = submission.model_copy(update={"dry_run": False})
    changed = current.model_copy(update={"max_steps": 100})
    check_resume_config(changed, [snapshot])
    script_path = tmp_path / "custom.sbatch"
    if config_cls is RLConfig:
        write_rl_script(submission, config_dir, log_dir, script_path)
    else:
        write_sft_script(submission, config_dir / "sft.json", log_dir, script_path, "test-run", False)
    assert script_path.read_text().startswith(header)
    assert not snapshot.with_suffix(".started").exists()
    # Custom templates record execution without requiring their own marker command.
    subprocess.run(["bash", str(script_path)], cwd=tmp_path, check=True)
    assert snapshot.with_suffix(".started").is_file()
    check_resume_config(current, [snapshot])
    with pytest.raises(ValueError, match=r"in \[max_steps\].*--resume.skip-checks"):
        check_resume_config(changed, [snapshot])
