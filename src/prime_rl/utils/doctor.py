"""Preflight checks for the ``rl`` entrypoint (``rl @ config.toml --check``).

Answers "will this run actually start?" before any GPU-hour is spent. Checks
are pure functions ``(RLConfig, HostProbe) -> list[CheckResult]`` so they can
be unit-tested on CPU-only CI with a fake probe. Host/config checks run in
well under a second; the env check spawns each configured env server (bounded
concurrency, per-env timeout) and the endpoint check probes frozen/external
inference servers over the network, so a full run takes seconds to a couple
of minutes.

Invariant: ``--check`` never writes to ``output_dir`` (unlike ``--dry-run``,
which writes resolved configs). Env-server logs go to a temp directory.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from prime_rl.configs.algorithm import FrozenModelConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir

if TYPE_CHECKING:
    from prime_rl.configs.rl import RLConfig

# Disk thresholds. FAIL below the floor (can't even hold logs + resolved
# configs safely); WARN below the soft limit since full checkpoints with
# optimizer state commonly run to hundreds of GB.
DISK_FAIL_GB = 5
DISK_WARN_GB = 100

# Per-env timeout for the env spawn check. Deliberately much shorter than the
# orchestrator's ENV_SERVER_SPAWN_TIMEOUT (600s): a slow dataset download is a
# SKIP with a hint, not a FAIL, so --check stays usable in CI.
ENV_CHECK_TIMEOUT = 120.0

# Max env servers spawned concurrently. Each is a real child process that
# loads a taskset (RAM + network), so many-env configs shouldn't fork-bomb the
# launcher host; 8 keeps the worst case at a few timeout windows without
# contending badly on downloads.
ENV_CHECK_CONCURRENCY = 8

# Timeout for endpoint reachability probes.
ENDPOINT_PROBE_TIMEOUT = 5.0


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    hint: str | None = None


class HostProbe:
    """Thin, mockable layer over host state (sockets, disk, env vars)."""

    def port_is_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return True
            except OSError:
                return False

    def disk_free_bytes(self, path: Path) -> int:
        # output_dir may not exist yet (--check must not create it); walk up
        # to the nearest existing ancestor.
        current = path.resolve()
        while not current.exists() and current != current.parent:
            current = current.parent
        return shutil.disk_usage(current).free

    def getenv(self, name: str) -> str | None:
        return os.environ.get(name)

    def netrc_has_host(self, host: str) -> bool:
        netrc_path = Path(self.getenv("NETRC") or "~/.netrc").expanduser()
        try:
            return host in netrc_path.read_text()
        except OSError:
            return False

    def hf_token_present(self) -> bool:
        if self.getenv("HF_TOKEN"):
            return True
        hf_home = Path(self.getenv("HF_HOME") or "~/.cache/huggingface").expanduser()
        return (hf_home / "token").is_file()


def _is_local_single_node(config: "RLConfig") -> bool:
    return config.slurm is None and config.deployment.type == "single_node"


### Static tier


def check_ports(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    results: list[CheckResult] = []
    if config.inference is None:
        return [
            CheckResult(
                "ports",
                CheckStatus.SKIP,
                "no [inference] block — the external server is probed by the endpoints check",
            )
        ]

    # Client base_url port must match the managed server's port (mirrors the
    # launch-time validation in entrypoints/rl.py, but before any mutation).
    client = config.orchestrator.model.client
    if not client.is_elastic:
        base_url = client.base_url[0]
        client_port = urlparse(base_url).port
        server_port = config.inference.server.port
        if client_port != server_port:
            results.append(
                CheckResult(
                    "client/server port match",
                    CheckStatus.FAIL,
                    f"orchestrator client points at port {client_port}, inference server serves on {server_port}",
                    hint=f"set orchestrator.model.client.base_url to use port {server_port}, "
                    "or change inference.server.port",
                )
            )
        else:
            results.append(
                CheckResult(
                    "client/server port match",
                    CheckStatus.PASS,
                    f"client base_url and inference server agree on port {server_port}",
                )
            )
    else:
        results.append(CheckResult("client/server port match", CheckStatus.SKIP, "elastic client — no static URL"))

    # Bind probe is only meaningful when this host is the compute host.
    if _is_local_single_node(config):
        port = config.inference.server.port
        if probe.port_is_free(port):
            results.append(CheckResult("inference port free", CheckStatus.PASS, f"port {port} is free"))
        else:
            results.append(
                CheckResult(
                    "inference port free",
                    CheckStatus.FAIL,
                    f"port {port} is already in use",
                    hint="a vLLM server from a previous run may still be alive "
                    "(check `nvidia-smi` / `pgrep -f vllm`), or change inference.server.port",
                )
            )
    else:
        results.append(
            CheckResult(
                "inference port free",
                CheckStatus.SKIP,
                "launcher host is not the compute host (SLURM/multi-node)",
            )
        )
    return results


def check_parallelism(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    # tp×dp vs num_infer_gpus and multi-node divisibility are enforced by
    # config validators at parse time. A non-divisible train-GPU/cp allocation
    # is caught by the trainer's ParallelDims assert — but only after the
    # launcher has cleaned the output dir and spawned all processes, with the
    # error buried in logs/trainer.log. Catch it here, before any side effects.
    if config.deployment.type != "single_node":
        return [CheckResult("parallelism", CheckStatus.SKIP, "multi-node — validated at config parse time")]

    num_train_gpus = config.deployment.num_train_gpus
    cp = config.trainer.model.cp
    if cp > num_train_gpus or num_train_gpus % cp != 0:
        return [
            CheckResult(
                "parallelism",
                CheckStatus.FAIL,
                f"num_train_gpus ({num_train_gpus}) is not divisible by trainer.model.cp ({cp})",
                hint="the trainer would crash at startup (ParallelDims: dp_shard * cp must equal its "
                "world size) — adjust deployment.num_train_gpus or trainer.model.cp",
            )
        ]
    return [
        CheckResult(
            "parallelism",
            CheckStatus.PASS,
            f"{num_train_gpus} train GPU(s), cp={cp} → {num_train_gpus // cp} data-parallel worker(s)",
        )
    ]


def check_ckpt(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    if config.ckpt is None or config.ckpt.resume_step is None:
        return [CheckResult("ckpt resume", CheckStatus.SKIP, "not resuming")]

    ckpt_base = config.ckpt.output_dir if config.ckpt.output_dir is not None else config.output_dir
    ckpt_dir = get_ckpt_dir(ckpt_base)
    steps = get_all_ckpt_steps(ckpt_dir) if ckpt_dir.is_dir() else []
    resume_step = config.ckpt.resume_step

    if resume_step == -1:
        if steps:
            return [
                CheckResult(
                    "ckpt resume", CheckStatus.PASS, f"resume_step=-1 resolves to step {steps[-1]} in {ckpt_dir}"
                )
            ]
        return [
            CheckResult(
                "ckpt resume",
                CheckStatus.FAIL,
                f"resume_step=-1 but no checkpoints found in {ckpt_dir}",
                hint="remove ckpt.resume_step to train from scratch, or point ckpt.output_dir/output_dir "
                "at the directory holding the checkpoints",
            )
        ]

    if resume_step in steps:
        return [CheckResult("ckpt resume", CheckStatus.PASS, f"step {resume_step} found in {ckpt_dir}")]
    available = ", ".join(map(str, steps[-5:])) if steps else "none"
    return [
        CheckResult(
            "ckpt resume",
            CheckStatus.FAIL,
            f"step {resume_step} not found in {ckpt_dir} (latest steps: {available})",
            hint="pick an existing step, or use resume_step = -1 for the latest",
        )
    ]


def check_disk(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    free_gb = probe.disk_free_bytes(config.output_dir) / 1e9
    if free_gb < DISK_FAIL_GB:
        return [
            CheckResult(
                "disk",
                CheckStatus.FAIL,
                f"{free_gb:.1f} GB free at {config.output_dir} (< {DISK_FAIL_GB} GB floor)",
                hint="free up space or point output_dir at a larger volume",
            )
        ]
    if free_gb < DISK_WARN_GB:
        return [
            CheckResult(
                "disk",
                CheckStatus.WARN,
                f"{free_gb:.1f} GB free at {config.output_dir}",
                hint=f"full checkpoints with optimizer state commonly exceed this — consider ckpt.keep_last, "
                f"or a volume with > {DISK_WARN_GB} GB free",
            )
        ]
    return [CheckResult("disk", CheckStatus.PASS, f"{free_gb:.0f} GB free at {config.output_dir}")]


def check_tokens(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    results: list[CheckResult] = []

    if config.wandb is not None:
        wandb_mode = (probe.getenv("WANDB_MODE") or "").lower()
        has_creds = (
            probe.getenv("WANDB_API_KEY") is not None
            or probe.netrc_has_host("api.wandb.ai")
            or wandb_mode in ("disabled", "offline")
        )
        if has_creds:
            results.append(CheckResult("wandb auth", CheckStatus.PASS, "credentials found"))
        else:
            results.append(
                CheckResult(
                    "wandb auth",
                    CheckStatus.WARN,
                    "wandb is configured but no credentials were found",
                    hint="headless launches hang at the interactive wandb login prompt — "
                    "set WANDB_API_KEY, run `uv run wandb login`, or set WANDB_MODE=disabled",
                )
            )
    else:
        results.append(CheckResult("wandb auth", CheckStatus.SKIP, "wandb not configured"))

    if probe.hf_token_present():
        results.append(CheckResult("hf auth", CheckStatus.PASS, "HF token found"))
    else:
        results.append(
            CheckResult("hf auth", CheckStatus.PASS, "no HF token found — required only for gated/private models")
        )
    return results


### Full tier (network + subprocesses)


def _probe_endpoint(base_url: str) -> str | None:
    """Return None if reachable, else a short error description.

    ``base_url`` is an OpenAI-compatible base like ``http://host:8000/v1``;
    any HTTP response (including 401/404) proves a server is listening.
    """
    import httpx

    url = base_url.rstrip("/") + "/models"
    try:
        httpx.get(url, timeout=ENDPOINT_PROBE_TIMEOUT)
        return None
    except httpx.HTTPError as e:
        return f"{type(e).__name__}: {e}"


def check_endpoints(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    """Probe endpoints the rl entrypoint does not manage: frozen/teacher models
    always, and the policy server when no [inference] block is configured.

    Upgrades the launch-time "make sure these are serving, otherwise rollouts
    will hang" log lines into actual probes.
    """
    results: list[CheckResult] = []

    # Frozen model references (teachers, OPD sources, ...) — never launched by prime-rl.
    frozen: dict[str, FrozenModelConfig] = {}
    for env in config.orchestrator.train.env:
        algo = env.algo
        if algo is None:
            continue
        for ref in (algo.sampling.source, getattr(algo, "teacher", None)):
            if isinstance(ref, FrozenModelConfig) and not ref.is_elastic:
                frozen.setdefault(ref.name, ref)
    for name, ref in frozen.items():
        error = _probe_endpoint(ref.base_url[0])
        if error is None:
            results.append(CheckResult(f"frozen model '{name}'", CheckStatus.PASS, f"reachable at {ref.base_url[0]}"))
        else:
            results.append(
                CheckResult(
                    f"frozen model '{name}'",
                    CheckStatus.FAIL,
                    f"unreachable at {ref.base_url[0]} ({error})",
                    hint="the rl entrypoint does not start frozen models — start this server before "
                    "launching, or rollouts will hang",
                )
            )

    # Policy inference server, when not managed by this entrypoint.
    if config.inference is None:
        client = config.orchestrator.model.client
        if client.is_elastic:
            results.append(CheckResult("external inference", CheckStatus.SKIP, "elastic pool — discovered via DNS"))
        else:
            error = _probe_endpoint(client.base_url[0])
            if error is None:
                results.append(
                    CheckResult("external inference", CheckStatus.PASS, f"reachable at {client.base_url[0]}")
                )
            else:
                results.append(
                    CheckResult(
                        "external inference",
                        CheckStatus.FAIL,
                        f"unreachable at {client.base_url[0]} ({error})",
                        hint="no [inference] block is configured, so the orchestrator expects a running "
                        "server at this URL and will hang waiting for it",
                    )
                )
    return results


async def _check_one_env(env_config, log_dir: Path) -> CheckResult:
    from prime_rl.orchestrator.envs import Env

    env = Env(env_config)
    name = env.name
    try:
        await asyncio.wait_for(env.start(log_dir), timeout=ENV_CHECK_TIMEOUT)
        return CheckResult(
            f"env '{name}'",
            CheckStatus.PASS,
            f"loaded: num_tasks={env.num_tasks}, group_scoring={env.requires_group_scoring}",
        )
    except asyncio.TimeoutError:
        return CheckResult(
            f"env '{name}'",
            CheckStatus.SKIP,
            f"did not come up within {ENV_CHECK_TIMEOUT:.0f}s",
            hint=f"often a slow first-time dataset download, not a bug — see {log_dir / f'{name}.log'}",
        )
    except Exception as e:  # noqa: BLE001 — report, don't crash the check run
        detail = str(e).strip().split("\n")[0][:200] or type(e).__name__
        return CheckResult(
            f"env '{name}'",
            CheckStatus.FAIL,
            detail,
            hint=f"full server output: {log_dir / f'{name}.log'}",
        )
    finally:
        # shutdown blocks (join with timeout, kill if stuck) — keep it off the
        # event loop so one wedged env doesn't stall the other parallel checks.
        await asyncio.to_thread(env.shutdown)


def check_envs(config: "RLConfig", probe: HostProbe) -> list[CheckResult]:
    """Spawn each configured env server briefly and read its ``info``.

    Catches missing env packages, dataset auth/download failures, and broken
    env code before a run ever starts — today these surface minutes in, buried
    in the orchestrator's env logs.
    """
    env_configs = list(config.orchestrator.train.env)
    if config.orchestrator.eval is not None:
        env_configs += list(config.orchestrator.eval.env)
    if not env_configs:
        return [CheckResult("envs", CheckStatus.SKIP, "no environments configured")]

    # Never write into output_dir from --check; env-server logs go to a temp dir.
    log_dir = Path(tempfile.mkdtemp(prefix="prime-rl-check-envs-"))

    async def run_all() -> list[CheckResult]:
        semaphore = asyncio.Semaphore(ENV_CHECK_CONCURRENCY)

        async def run_one(env_config) -> CheckResult:
            async with semaphore:
                return await _check_one_env(env_config, log_dir)

        return list(await asyncio.gather(*(run_one(env_config) for env_config in env_configs)))

    return asyncio.run(run_all())


CHECKS = [check_ports, check_parallelism, check_ckpt, check_disk, check_tokens, check_envs, check_endpoints]

_STATUS_SYMBOL = {
    CheckStatus.PASS: "✓",
    CheckStatus.WARN: "⚠",
    CheckStatus.FAIL: "✗",
    CheckStatus.SKIP: "○",
}


def run_checks(config: "RLConfig", probe: HostProbe | None = None) -> int:
    """Run preflight checks and render a report. Returns the process exit code:
    0 when nothing failed (warnings allowed, so CI can gate on --check without
    flaking), 1 when any check failed."""
    logger = get_logger()
    probe = probe or HostProbe()

    logger.info("Running preflight checks")

    results: list[CheckResult] = []
    for check in CHECKS:
        results.extend(check(config, probe))

    for result in results:
        line = f"{_STATUS_SYMBOL[result.status]} {result.name:<28} {result.detail}"
        if result.status == CheckStatus.FAIL:
            logger.error(line)
        elif result.status == CheckStatus.WARN:
            logger.warning(line)
        else:
            logger.info(line)
        if result.hint and result.status in (CheckStatus.FAIL, CheckStatus.WARN):
            logger.info(f"    ↳ {result.hint}")

    num_failed = sum(r.status == CheckStatus.FAIL for r in results)
    num_warned = sum(r.status == CheckStatus.WARN for r in results)
    num_passed = sum(r.status == CheckStatus.PASS for r in results)
    num_skipped = sum(r.status == CheckStatus.SKIP for r in results)
    summary = f"{num_passed} passed, {num_warned} warning(s), {num_failed} failure(s), {num_skipped} skipped"

    if num_failed:
        logger.error(f"Preflight failed: {summary}. Fix the failures above before launching.")
        return 1
    logger.success(f"Preflight passed: {summary}")
    return 0
