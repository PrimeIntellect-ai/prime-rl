from pathlib import Path

import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.doctor import (
    CheckStatus,
    HostProbe,
    check_ckpt,
    check_disk,
    check_parallelism,
    check_ports,
    check_tokens,
    run_checks,
)


class FakeProbe(HostProbe):
    """HostProbe with fully controlled host state — no sockets, disk, or env vars."""

    def __init__(
        self,
        free_ports: bool = True,
        disk_free: int = 500 * 10**9,
        env: dict[str, str] | None = None,
        netrc_hosts: set[str] | None = None,
    ):
        self.free_ports = free_ports
        self.disk_free = disk_free
        self.env = env or {}
        self.netrc_hosts = netrc_hosts or set()

    def port_is_free(self, port: int) -> bool:
        return self.free_ports

    def disk_free_bytes(self, path: Path) -> int:
        return self.disk_free

    def getenv(self, name: str) -> str | None:
        return self.env.get(name)

    def netrc_has_host(self, host: str) -> bool:
        return host in self.netrc_hosts


def make_config(**overrides) -> RLConfig:
    data = {"trainer": {}, "orchestrator": {}, "inference": {}}
    data.update(overrides)
    return RLConfig.model_validate(data)


def statuses(results) -> list[CheckStatus]:
    return [r.status for r in results]


### check_ports


def test_ports_pass_when_defaults_align():
    # Default client base_url (localhost:8000/v1) matches default server port (8000).
    results = check_ports(make_config(), FakeProbe(free_ports=True))
    assert statuses(results) == [CheckStatus.PASS, CheckStatus.PASS]


def test_ports_pass_when_client_auto_syncs_to_server_port():
    # base_url not explicitly set -> auto_setup_inference_client syncs it to
    # the server port, so there is no mismatch to catch.
    config = make_config(inference={"server": {"port": 8001}})
    results = check_ports(config, FakeProbe())
    match = next(r for r in results if r.name == "client/server port match")
    assert match.status == CheckStatus.PASS


def test_ports_fail_on_client_server_port_mismatch():
    # A mismatch only survives config validation when base_url is explicitly
    # set (auto-sync respects model_fields_set) — that's the case this guards.
    config = make_config(
        orchestrator={"model": {"client": {"base_url": ["http://localhost:8000/v1"]}}},
        inference={"server": {"port": 8001}},
    )
    results = check_ports(config, FakeProbe())
    match = next(r for r in results if r.name == "client/server port match")
    assert match.status == CheckStatus.FAIL
    assert "8001" in match.detail


def test_ports_fail_when_port_busy():
    results = check_ports(make_config(), FakeProbe(free_ports=False))
    bind = next(r for r in results if r.name == "inference port free")
    assert bind.status == CheckStatus.FAIL
    assert bind.hint is not None


def test_ports_skip_without_inference_block():
    config = RLConfig.model_validate({"trainer": {}, "orchestrator": {}})
    results = check_ports(config, FakeProbe())
    assert statuses(results) == [CheckStatus.SKIP]


def test_ports_bind_probe_skipped_on_slurm():
    # Launcher host is not the compute host — bind probe is meaningless there.
    config = make_config(slurm={})
    results = check_ports(config, FakeProbe(free_ports=False))
    bind = next(r for r in results if r.name == "inference port free")
    assert bind.status == CheckStatus.SKIP


### check_parallelism


def test_parallelism_pass_default():
    results = check_parallelism(make_config(), FakeProbe())
    assert statuses(results) == [CheckStatus.PASS]


def test_parallelism_fail_on_non_divisible_cp():
    config = make_config(
        trainer={"model": {"cp": 2}},
        deployment={"type": "single_node", "gpus_per_node": 8, "num_train_gpus": 3, "num_infer_gpus": 1},
    )
    results = check_parallelism(config, FakeProbe())
    assert statuses(results) == [CheckStatus.FAIL]
    assert "cp" in results[0].detail


### check_ckpt


def test_ckpt_skip_when_not_resuming():
    results = check_ckpt(make_config(), FakeProbe())
    assert statuses(results) == [CheckStatus.SKIP]


def test_ckpt_pass_when_resume_step_exists(tmp_path: Path):
    (tmp_path / "checkpoints" / "step_10").mkdir(parents=True)
    config = make_config(output_dir=str(tmp_path), ckpt={"resume_step": 10})
    results = check_ckpt(config, FakeProbe())
    assert statuses(results) == [CheckStatus.PASS]


def test_ckpt_fail_when_resume_step_missing(tmp_path: Path):
    (tmp_path / "checkpoints" / "step_10").mkdir(parents=True)
    config = make_config(output_dir=str(tmp_path), ckpt={"resume_step": 7})
    results = check_ckpt(config, FakeProbe())
    assert statuses(results) == [CheckStatus.FAIL]
    assert "10" in results[0].detail  # lists available steps


def test_ckpt_latest_resolves_or_fails(tmp_path: Path):
    config = make_config(output_dir=str(tmp_path), ckpt={"resume_step": -1})
    assert statuses(check_ckpt(config, FakeProbe())) == [CheckStatus.FAIL]

    (tmp_path / "checkpoints" / "step_20").mkdir(parents=True)
    results = check_ckpt(config, FakeProbe())
    assert statuses(results) == [CheckStatus.PASS]
    assert "20" in results[0].detail


def test_ckpt_respects_ckpt_output_dir_override(tmp_path: Path):
    ckpt_dir = tmp_path / "elsewhere"
    (ckpt_dir / "checkpoints" / "step_5").mkdir(parents=True)
    config = make_config(output_dir=str(tmp_path / "out"), ckpt={"resume_step": 5, "output_dir": str(ckpt_dir)})
    assert statuses(check_ckpt(config, FakeProbe())) == [CheckStatus.PASS]


### check_disk


@pytest.mark.parametrize(
    ("free_gb", "expected"),
    [(2, CheckStatus.FAIL), (50, CheckStatus.WARN), (500, CheckStatus.PASS)],
)
def test_disk_thresholds(free_gb: int, expected: CheckStatus):
    results = check_disk(make_config(), FakeProbe(disk_free=free_gb * 10**9))
    assert statuses(results) == [expected]


### check_tokens


def test_tokens_warn_when_wandb_configured_without_creds():
    config = make_config(wandb={})
    results = check_tokens(config, FakeProbe())
    wandb = next(r for r in results if r.name == "wandb auth")
    assert wandb.status == CheckStatus.WARN


@pytest.mark.parametrize(
    "env",
    [{"WANDB_API_KEY": "x"}, {"WANDB_MODE": "disabled"}, {"WANDB_MODE": "offline"}],
)
def test_tokens_pass_with_wandb_creds_or_disabled(env: dict[str, str]):
    config = make_config(wandb={})
    results = check_tokens(config, FakeProbe(env=env))
    wandb = next(r for r in results if r.name == "wandb auth")
    assert wandb.status == CheckStatus.PASS


def test_tokens_pass_with_wandb_netrc():
    config = make_config(wandb={})
    results = check_tokens(config, FakeProbe(netrc_hosts={"api.wandb.ai"}))
    wandb = next(r for r in results if r.name == "wandb auth")
    assert wandb.status == CheckStatus.PASS


def test_tokens_skip_when_wandb_not_configured():
    results = check_tokens(make_config(), FakeProbe())
    wandb = next(r for r in results if r.name == "wandb auth")
    assert wandb.status == CheckStatus.SKIP


def test_hf_token_is_informational_only():
    # Never a WARN/FAIL — gated-model detection is unreliable.
    for env in ({}, {"HF_TOKEN": "x"}):
        results = check_tokens(make_config(), FakeProbe(env=env))
        hf = next(r for r in results if r.name == "hf auth")
        assert hf.status == CheckStatus.PASS


### run_checks (exit codes; static tier only — full tier needs subprocess/network)


def test_run_checks_exit_zero_when_healthy(tmp_path: Path):
    config = make_config(output_dir=str(tmp_path))
    assert run_checks(config, probe=FakeProbe()) == 0


def test_run_checks_warnings_do_not_fail(tmp_path: Path):
    # 50 GB free -> disk WARN, but exit code stays 0 so CI can gate on --check.
    config = make_config(output_dir=str(tmp_path))
    assert run_checks(config, probe=FakeProbe(disk_free=50 * 10**9)) == 0


def test_run_checks_exit_one_on_failure(tmp_path: Path):
    config = make_config(output_dir=str(tmp_path))
    assert run_checks(config, probe=FakeProbe(free_ports=False)) == 1
