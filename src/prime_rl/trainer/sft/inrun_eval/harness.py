"""The collection harness on node 0: its bundle, its own virtualenv, its flow configs and the flow runs.

Bundle layout (``[inrun_eval.harness] bundle_dir``, staged on the volume; see PREREQS.md):

    manifest.json                    {"files": {"<relative path>": "<sha256>", ...}}: every file below
    tasks.tsv                        <name>\\t<source>\\t<split> for every task (tasks.py)
    flow-configs/<source>.toml       the collection runs' configs/attempt_2/flow.toml (sha256-pinned in the config)
    sft-gen/                         sft-gen @ bca802f (sft_gen/, plugins/, prompts/, ...): the flow runs from a copy
    harness/requirements.txt         the flow host's packages, exact versions (from sft-gen bca802f's uv.lock)
    harness/wheels/                  wheels the install may take (at least the private prime-flow); with a
                                     ``.complete`` marker every requirement is there and nothing is downloaded
    harness/python/*.tar.gz          optional: python-build-standalone 3.13 (no Python download)
    harness/frpc/                    optional: prime_tunnel's frpc binary and .frpc_version (no GitHub download)
    harness/host-uv/<ver>/<target>/uv optional: verifiers' host uv (``runtimes/host_uv``) for boxes that cannot
                                     install uv themselves, as its cache holds it (no GitHub download)
    harbor-cache/<cache name>/       the Harbor dataset exports, as verifiers caches them (tasks.harbor_cache_name)
    kernels/<landlock bundle>.tgz    the kernel bundle the 2 landlock tasks' configs name
"""

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from prime_rl.trainer.sft.inrun_eval.serving import clean_env

MANIFEST = "manifest.json"
TASKS = "tasks.tsv"
REQUIREMENTS = "harness/requirements.txt"
WHEELS = "harness/wheels"
PYTHONS = "harness/python"
FRPC = "harness/frpc"
HOST_UV = "harness/host-uv"
SFT_GEN = "sft-gen"
HARBOR_CACHE = "harbor-cache"
KERNELS = "kernels"
FLOW_POLL = 30.0
PLUGINS = Path(__file__).resolve().parent / "plugins"
"""Harness plugins of the eval (the D5 mini_swe_agent harness), put on the flow runs' PYTHONPATH."""
D5_MODULE = "tb4_mini_swe_agent_d5"
PRIME_API_KEY = "PRIME_API_KEY"

# Never handed to the harness (it gets the platform's PRIME_API_KEY, PRIME_TEAM_ID and its own vLLM key only).
SECRET_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN", "GH_TOKEN")
UV_PROJECT_VARS = (
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON_PREFERENCE",
    "UV_PYTHON",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "UV_SYSTEM_PYTHON",
)

VERIFY_SCRIPT = r"""
import hashlib, json, sys, importlib.metadata as md, pathlib
out = {"python": sys.version.split()[0], "packages": {}, "files": {}}
for name in ("verifiers", "prime-flow", "prime-sandboxes", "harbor", "litellm", "prime-tunnel"):
    try:
        out["packages"][name] = md.version(name)
    except md.PackageNotFoundError:
        out["packages"][name] = None
import verifiers
root = pathlib.Path(verifiers.__file__).resolve().parent.parent
for rel in json.loads(sys.argv[1]):
    p = root / rel
    out["files"][rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
    from verifiers.v1.utils.loaders import harness_class, harness_config_type
    cls = harness_class(sys.argv[3])
    out["plugin"] = {"harness": cls.__name__, "config": harness_config_type(sys.argv[3]).__name__}
print(json.dumps(out))
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(bundle: Path) -> list[str]:
    """Every file the manifest lists exists with its sha256 (problems; empty when the bundle is intact)."""
    path = bundle / MANIFEST
    if not path.is_file():
        return [f"{path} missing"]
    files = json.loads(path.read_text()).get("files", {})
    problems = []
    for rel, digest in sorted(files.items()):
        target = bundle / rel
        if not target.is_file():
            problems.append(f"{rel} missing")
        elif sha256_file(target) != digest:
            problems.append(f"{rel} sha256 differs from the manifest")
    return problems


def check_bundle(harness, bundle: Path) -> list[str]:
    """What the eval needs from the bundle beyond the manifest: the task table, the pinned source configs, the
    sft-gen tree and the requirements."""
    problems = verify_manifest(bundle)
    for rel in (TASKS, REQUIREMENTS, f"{SFT_GEN}/plugins/sft_gen_task_qc.py", f"{SFT_GEN}/sft_gen/task_qc.py"):
        if not (bundle / rel).is_file():
            problems.append(f"{rel} missing")
    for label in harness.sources:
        path = bundle / "flow-configs" / f"{label}.toml"
        if not path.is_file():
            problems.append(f"flow-configs/{label}.toml missing")
        elif sha256_file(path) != harness.source_config_sha256[label]:
            problems.append(f"flow-configs/{label}.toml sha256 {sha256_file(path)} is not the pinned collection config")
    return problems


@dataclass
class HarnessEnv:
    venv: Path
    cwd: Path
    env: dict[str, str]
    info: dict = field(default_factory=dict)


def uv_env(base: dict[str, str], work: Path) -> dict[str, str]:
    env = {k: v for k, v in clean_env(base).items() if k not in UV_PROJECT_VARS and k not in SECRET_VARS}
    env.update(
        UV_CACHE_DIR=str(work / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(work / "uv-python"),
        HOME=str(work / "home"),
    )
    return env


def wheel_mode(bundle: Path) -> str:
    """``offline`` (every requirement has a wheel: ``.complete`` marker), ``mixed`` (some wheels, the rest from
    PyPI / git) or ``online``."""
    wheels = bundle / WHEELS
    if not wheels.is_dir() or not any(wheels.glob("*.whl")):
        return "online"
    return "offline" if (wheels / ".complete").is_file() else "mixed"


def install_commands(uv: str, python: str, venv: Path, bundle: Path, mode: str) -> list[list[str]]:
    """``uv venv`` with the harness Python, then the pinned requirements without resolving anything (``--no-deps``:
    the file is the complete closure)."""
    install = [uv, "pip", "install", "--python", str(venv / "bin" / "python"), "--no-deps"]
    install += ["-r", str(bundle / REQUIREMENTS)]
    if mode in ("offline", "mixed"):
        install += ["--find-links", str(bundle / WHEELS)]
    if mode == "offline":
        install += ["--no-index"]
    return [[uv, "venv", str(venv), "--python", python], install]


def python_for_venv(harness, bundle: Path, work: Path, env: dict[str, str], log) -> str:
    """The staged python-build-standalone if the bundle has one, else a uv-managed download of ``harness.python``."""
    tarballs = sorted((bundle / PYTHONS).glob("*.tar.gz")) if (bundle / PYTHONS).is_dir() else []
    if tarballs:
        target = work / "python-standalone"
        if not (target / "python" / "bin").is_dir():
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tarballs[0]) as tar:
                tar.extractall(target, filter="data")
        candidates = sorted((target / "python" / "bin").glob("python3.*"))
        candidates = [c for c in candidates if c.name.count(".") == 1]
        if candidates:
            log(f"harness Python: {candidates[0]} (from {tarballs[0].name})")
            return str(candidates[0])
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    env["UV_PYTHON_DOWNLOADS"] = "automatic"
    log(f"harness Python: uv-managed {harness.python} (downloaded)")
    return harness.python


def build_env(harness, bundle: Path, work: Path, log=print, run=subprocess.run) -> HarnessEnv:
    """Build the harness virtualenv under ``work`` and check its pins. Raises on any failure."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    work.mkdir(parents=True, exist_ok=True)
    (work / "home").mkdir(exist_ok=True)
    env = uv_env(dict(os.environ), work)
    python = python_for_venv(harness, bundle, work, env, log)
    venv = work / "venv"
    mode = wheel_mode(bundle)
    log(f"harness install: {mode} ({bundle / WHEELS})")
    for cmd in install_commands(uv, python, venv, bundle, mode):
        done = run(cmd, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])} failed ({done.returncode}): {(done.stderr or '')[-2000:]}")
    plugins = work / "plugins"
    if plugins.exists():
        shutil.rmtree(plugins)
    shutil.copytree(PLUGINS, plugins, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    verify = [str(venv / "bin" / "python"), "-I", "-c", VERIFY_SCRIPT, json.dumps(sorted(harness.pinned_files_sha256))]
    if harness.d5_exclude_newer is not None:
        from prime_rl.trainer.sft.inrun_eval.flowcfg import D5_HARNESS

        verify += [str(plugins), D5_HARNESS]
    done = run(verify, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"harness check failed: {(done.stderr or '')[-2000:]}")
    info = json.loads(done.stdout.strip().splitlines()[-1])
    info["plugins_sha256"] = {path.name: sha256_file(path) for path in sorted(plugins.glob("*.py"))}
    problems = check_versions(harness, info)
    if problems:
        raise RuntimeError(f"harness pins: {problems}")

    cwd = work / SFT_GEN
    if cwd.exists():
        shutil.rmtree(cwd)
    shutil.copytree(bundle / SFT_GEN, cwd, symlinks=True)
    frpc = bundle / FRPC
    if frpc.is_dir():
        shutil.copytree(frpc, work / "home" / ".prime" / "bin", dirs_exist_ok=True)
    stage_host_uv(bundle, env)
    hen = HarnessEnv(venv=venv, cwd=cwd, env=env, info=info)
    hen.info["plugins_dir"] = str(plugins)
    return hen


def stage_host_uv(bundle: Path, env: dict[str, str]) -> Path | None:
    """Put the bundle's host uv where verifiers' ``host_uv.cache_dir()`` looks for it under the harness env."""
    src = bundle / HOST_UV
    if not src.is_dir():
        return None
    cache = Path(env.get("XDG_CACHE_HOME") or Path(env["HOME"]) / ".cache") / "verifiers" / "uv"
    shutil.copytree(src, cache, dirs_exist_ok=True)
    return cache


def check_versions(harness, info: dict) -> list[str]:
    problems = []
    packages = info.get("packages", {})
    if packages.get("verifiers") != harness.verifiers_version:
        problems.append(f"verifiers {packages.get('verifiers')} != {harness.verifiers_version}")
    if harness.prime_flow_version is not None and packages.get("prime-flow") != harness.prime_flow_version:
        problems.append(f"prime-flow {packages.get('prime-flow')} != {harness.prime_flow_version}")
    for rel, digest in harness.pinned_files_sha256.items():
        got = info.get("files", {}).get(rel)
        if got != digest:
            problems.append(f"{rel} sha256 {got} != {digest}")
    return problems


def team_id(harness, base: dict[str, str], fallback: str | None = None) -> tuple[str | None, str]:
    """The Prime team id and where it came from: the config, the environment, the trainer's view (its [env_vars])."""
    if harness.prime_team_id:
        return harness.prime_team_id, "inrun_eval.harness.prime_team_id"
    if base.get("PRIME_TEAM_ID"):
        return base["PRIME_TEAM_ID"], "environment"
    if fallback:
        return fallback, "run config [env_vars]"
    return None, "unset"


def credentials(harness, base: dict[str, str], fallback_team: str | None = None) -> dict:
    """What the eval has for Prime (sandboxes, tunnels, the team's private images): the platform's per-run
    ``PRIME_API_KEY`` (only its presence is recorded) and the team id. ``problem`` is None when both are there."""
    team, source = team_id(harness, base, fallback_team)
    present = bool(base.get(PRIME_API_KEY))
    problem = None
    if not present:
        problem = (
            "PRIME_API_KEY is not set in the trainer pod: the platform did not provide its per-run Prime credential "
            "(prime train -e cannot pass it), so no sandbox can start"
        )
    elif not team:
        problem = "PRIME_TEAM_ID is not set (run config [env_vars] or inrun_eval.harness.prime_team_id)"
    return {"prime_api_key_present": present, "prime_team_id": team, "prime_team_id_source": source, "problem": problem}


def run_env(harness, hen: HarnessEnv, base: dict[str, str], fallback_team: str | None = None) -> dict[str, str]:
    """The flow runs' environment: the install env plus the venv, the sft-gen modules and the eval's plugins, the
    platform's Prime credential and team, and a fresh vLLM key. Raises when the credentials are missing."""
    env = dict(hen.env)
    env["PATH"] = f"{hen.venv / 'bin'}:{base.get('PATH', '/usr/bin:/bin')}"
    env["VIRTUAL_ENV"] = str(hen.venv)
    path = [str(hen.cwd), str(hen.cwd / "plugins")]
    if hen.info.get("plugins_dir"):
        path.append(hen.info["plugins_dir"])
    env["PYTHONPATH"] = ":".join(path)
    creds = credentials(harness, base, fallback_team)
    if creds["problem"]:
        raise RuntimeError(creds["problem"])
    env[PRIME_API_KEY] = base[PRIME_API_KEY]
    env["PRIME_TEAM_ID"] = creds["prime_team_id"]
    env[harness.api_key_var] = secrets.token_hex(16)
    for var in ("UV_PYTHON_PREFERENCE", "UV_PYTHON_DOWNLOADS"):
        env.pop(var, None)
    return env


def stage_harbor_caches(bundle: Path, home: Path, names: list[str]) -> list[str]:
    """Copy the bundle's Harbor dataset exports to ``~/.cache/harbor`` of the harness (missing ones: problems)."""
    problems = []
    for name in names:
        src = bundle / HARBOR_CACHE / name
        dst = home / ".cache" / "harbor" / name
        if not src.is_dir():
            problems.append(f"{HARBOR_CACHE}/{name} missing")
            continue
        if not dst.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, symlinks=True)
    return problems


def stage_kernel_bundle(bundle: Path, target: str, work: Path, log=print, run=subprocess.run) -> dict[str, str]:
    """Put the landlock bundle where the configs name it. Returns the path rewrite to apply to the configs when that
    location cannot be created (the path is only read on the host, never shown to the agent)."""
    src = bundle / KERNELS / Path(target).name
    if not src.is_file():
        log(f"no {KERNELS}/{Path(target).name} in the bundle: the landlock tasks fail to set up their kernel")
        return {}
    dst = Path(target)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or sha256_file(dst) != sha256_file(src):
            shutil.copyfile(src, dst)
        return {}
    except OSError:
        pass
    done = run(["sudo", "-n", "mkdir", "-p", str(dst.parent)], capture_output=True)
    if done.returncode == 0:
        run(["sudo", "-n", "chown", str(os.getuid()), str(dst.parent)], capture_output=True)
        try:
            shutil.copyfile(src, dst)
            return {}
        except OSError:
            pass
    local = work / KERNELS / src.name
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, local)
    log(f"{dst.parent} cannot be created: the configs point the landlock bundle at {local} instead")
    return {target: str(local)}


def flow_cmd(harness, venv: Path, config: Path) -> list[str]:
    return [part.replace("{venv}", str(venv)).replace("{config}", str(config)) for part in harness.flow_command]


def wait_flow_runs(
    procs: dict, deadline: float | None, log=print, poll: float | None = None, grace: float = 300.0
) -> dict[str, dict]:
    """Wait for the flow runs (label -> ``serving.Proc``) to exit; with a ``deadline`` (the optional cap, wall clock)
    stop the rest then (a flow's first SIGTERM stops it now; what landed stays in its store, and the scoring reports
    the draws it cut as ``cut``). Returns each run's exit code, timing and whether the cap stopped it."""
    from prime_rl.trainer.sft.inrun_eval import serving

    poll = FLOW_POLL if poll is None else poll
    start = time.time()
    status: dict[str, dict] = {}
    last_note = 0.0
    while True:
        running = [label for label, proc in procs.items() if proc.alive()]
        for label, proc in procs.items():
            if label not in status and not proc.alive():
                status[label] = {"exit_code": proc.popen.returncode, "seconds": time.time() - start, "stopped": False}
                log(f"flow run {label} exited with {proc.popen.returncode} after {status[label]['seconds']:.0f}s")
        if not running:
            return status
        if deadline is not None and time.time() >= deadline:
            log(f"eval safety cap reached: stopping the flow runs {running} (their unfinished draws are 'cut')")
            serving.stop([procs[label] for label in running], grace=grace)
            for label in running:
                status[label] = {
                    "exit_code": procs[label].popen.returncode,
                    "seconds": time.time() - start,
                    "stopped": True,
                }
            return status
        if time.time() - last_note > 1800:
            last_note = time.time()
            left = "no cap" if deadline is None else f"{(deadline - time.time()) / 3600:.1f} h to the cap"
            log(f"flow runs still running: {running} ({(time.time() - start) / 3600:.1f} h in, {left})")
        time.sleep(poll)
