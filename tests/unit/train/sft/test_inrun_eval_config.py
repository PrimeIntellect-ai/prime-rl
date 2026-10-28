"""The in-run eval's config: the TB4 run and smoke configs parse as SFTConfig with the eval as intended, and the
cross-section validators reject the combinations the eval cannot run."""

import tomllib
from pathlib import Path

import pytest
import tomli_w

from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.config import cli

CONFIGS = Path(__file__).parents[4] / "configs" / "advanced" / "nemotron-3-super"
PLATFORM_KEYS = ("image_tag", "gpu_type", "source_ref", "env_file", "env_files", "volume", "volume_size", "type")


def parse(raw: dict, tmp_path: Path) -> SFTConfig:
    raw = {k: v for k, v in raw.items() if k not in PLATFORM_KEYS}
    # The hosted platform launches multi-node runs itself; locally the config needs a [slurm] table to validate.
    raw.setdefault("slurm", {})
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(raw))
    return cli(SFTConfig, args=["@", str(path)])


def load(name: str) -> dict:
    return tomllib.loads((CONFIGS / name).read_text())


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PRL_OUTPUT_DIR", str(tmp_path / "outputs"))


def test_tb4_run_config(tmp_path):
    config = parse(load("sft-hosted-tb4-v2.toml"), tmp_path)
    cfg = config.inrun_eval
    assert cfg is not None and cfg.checkpoints == ["best", "last"] and not cfg.eval_only and cfg.model_dir is None
    assert cfg.schedule == "sequential" and cfg.budget_hours is None  # no cap beyond the collection's timeouts
    assert config.env_vars["PRIME_TEAM_ID"] == "cmlr3u2er002zhr01tj8f48ts" and cfg.harness.prime_team_id is None
    assert cfg.harness.d5_exclude_newer == "2026-10-24T07:23:00Z" and cfg.harness.in_box_uv_version == "0.13.5"
    assert not cfg.evaluate_aliases and config.val.keep_best_max_step is None
    assert cfg.serving.tensor_parallel_size * cfg.serving.data_parallel_size == config.deployment.gpus_per_node
    assert cfg.serving.max_model_len == 1048576 and cfg.serving.override_generation_config == {
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert cfg.harness.expected_tasks == 500 and cfg.harness.k is None and cfg.harness.tasks_file is None
    assert cfg.harness.tb_sci_dataset == "tb-science-vendor@71124a84e6f3"
    assert config.hf_export.cleanup.best_snapshot and not config.hf_export.cleanup.final_checkpoint


def test_smoke_config(tmp_path):
    """The smoke runs the run's production path: real training on 2 nodes, val + keep_best, the export of best and
    last without a push, then the eval of both (two rounds) on 3 tasks x 1 rollout."""
    raw = load("tb4-inrun-eval-smoke.toml")
    run = parse(load("sft-hosted-tb4-v2.toml"), tmp_path)
    config = parse(raw, tmp_path)
    cfg = config.inrun_eval
    assert not cfg.eval_only and cfg.model_dir is None and cfg.checkpoints == ["best", "last"]
    assert cfg.schedule == "sequential" and cfg.evaluate_aliases and cfg.budget_hours is not None
    assert cfg.harness.k == 1 and cfg.harness.expected_tasks == 3 and cfg.harness.tasks_file is not None
    assert (
        cfg.harness.model_dump() | {"k": None, "expected_tasks": 500, "tasks_file": None, "credential_probe": False}
        == run.inrun_eval.harness.model_dump()
    )
    assert cfg.serving == run.inrun_eval.serving
    assert config.deployment.num_train_nodes == 2 and config.max_steps == 3 and config.ckpt is None
    assert config.model.cp == run.model.cp and config.model.ep == run.model.ep and config.data.seq_len == 65536
    assert config.val.after_optimizer_step and config.val.keep_best and config.val.keep_best_max_step == 2
    assert config.val.interval == 1 and config.hf_export.checkpoints == ["best", "last"]
    assert config.hf_export.best_repo is None and config.hf_export.last_repo is None
    assert config.hf_export.cleanup.best_snapshot and config.env_vars["PRIME_TEAM_ID"] == run.env_vars["PRIME_TEAM_ID"]
    # Launched without any -e (no HF_TOKEN: nothing can be pushed).
    assert "-e " not in "".join(
        line
        for line in (CONFIGS / "tb4-inrun-eval-smoke.toml").read_text().splitlines()
        if line.startswith("# prime train")
    )

    evalonly = parse(load("tb4-inrun-eval-smoke-evalonly.toml"), tmp_path).inrun_eval
    assert evalonly.eval_only and str(evalonly.model_dir) == "/volume/models/nemotron-3.5-super-text"
    assert evalonly.harness.k == 1 and evalonly.harness.expected_tasks == 3


def test_launch_commands_pass_only_hf_env():
    """`prime train -e` forwards only HF_TOKEN / WANDB_API_KEY for SFT runs: no other env file may be passed."""
    for name in ("sft-hosted-tb4-v2.toml", "tb4-inrun-eval-smoke.toml", "tb4-inrun-eval-smoke-evalonly.toml"):
        text = (CONFIGS / name).read_text()
        assert "tb4-eval.env" not in text and "TB4_EVAL_PRIME_API_KEY" not in text
    assert "-e ~/.config/prime-export/hf.env -y" in (CONFIGS / "sft-hosted-tb4-v2.toml").read_text()


def test_validators(tmp_path):
    raw = load("sft-hosted-tb4-v2.toml")
    with pytest.raises(Exception, match="gpus_per_node"):
        parse({**raw, "inrun_eval": {"serving": {"tensor_parallel_size": 4, "data_parallel_size": 4}}}, tmp_path)
    with pytest.raises(Exception, match="not in hf_export.checkpoints"):
        parse({**raw, "hf_export": {**raw["hf_export"], "checkpoints": ["last"]}}, tmp_path)
    no_export = {k: v for k, v in raw.items() if k != "hf_export"}
    with pytest.raises(Exception, match="configure \\[hf_export\\] or model_dir"):
        parse(no_export, tmp_path)
    with pytest.raises(Exception, match="eval_only needs"):
        parse({**raw, "inrun_eval": {"eval_only": True}}, tmp_path)
    with pytest.raises(Exception, match="RFC 3339"):
        parse({**raw, "inrun_eval": {"harness": {"d5_exclude_newer": "2026-10-24 07:23"}}}, tmp_path)
    # Off by default: the run config without the section parses with no eval.
    assert parse({k: v for k, v in raw.items() if k != "inrun_eval"}, tmp_path).inrun_eval is None
