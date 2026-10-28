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
    assert cfg.schedule == "sequential" and cfg.budget_hours == 48
    assert cfg.serving.tensor_parallel_size * cfg.serving.data_parallel_size == config.deployment.gpus_per_node
    assert cfg.serving.max_model_len == 1048576 and cfg.serving.override_generation_config == {
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert cfg.harness.expected_tasks == 500 and cfg.harness.k is None and cfg.harness.tasks_file is None
    assert cfg.harness.tb_sci_dataset == "tb-science-vendor@71124a84e6f3"
    assert config.hf_export.cleanup.best_snapshot and not config.hf_export.cleanup.final_checkpoint


def test_smoke_config(tmp_path):
    config = parse(load("tb4-inrun-eval-smoke.toml"), tmp_path)
    cfg = config.inrun_eval
    assert cfg.eval_only and str(cfg.model_dir) == "/volume/models/nemotron-3.5-super-text"
    assert cfg.harness.k == 1 and cfg.harness.expected_tasks == 3 and config.deployment.num_train_nodes == 1


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
    # Off by default: the run config without the section parses with no eval.
    assert parse({k: v for k, v in raw.items() if k != "inrun_eval"}, tmp_path).inrun_eval is None
