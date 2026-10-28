"""The collection harness on node 0: its bundle, its own virtualenv, its flow configs and the flow runs.

Bundle layout (``[inrun_eval.harness] bundle_dir``, staged on the volume; see PREREQS.md):

    manifest.json                    {"version": 2, "files": {"<relative path>": "<sha256>", ...},
                                     "modes": {"<relative path>": "0755" | "0644" | ..., ...}}: every file below.
                                     A Prime volume keeps no file modes (S3 objects): the eval restores them from
                                     ``modes`` on its copies (frpc, the Harbor task files: tests/, test.sh, ...)
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
import stat
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from prime_rl.trainer.sft.inrun_eval.serving import PRIME_API_KEY, SECRET_VARS, clean_env

MANIFEST = "manifest.json"
MANIFEST_VERSION = 2
"""Version 2 records every file's mode (``modes``) next to its sha256."""
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
PRIME_API_BASE_URL = "https://api.primeintellect.ai"
"""prime-sandboxes' / prime-tunnel's default API (``PRIME_API_BASE_URL`` / ``PRIME_BASE_URL`` override it, as there)."""

# SECRET_VARS (serving) are never handed to the harness: it gets the platform's PRIME_API_KEY, PRIME_TEAM_ID and
# its own vLLM key only.
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


def load_manifest(bundle: Path) -> dict:
    return json.loads((bundle / MANIFEST).read_text())


def file_modes(bundle: Path) -> dict[str, int]:
    """The manifest's file modes (relative path -> permission bits)."""
    return {rel: int(mode, 8) for rel, mode in load_manifest(bundle).get("modes", {}).items()}


def verify_manifest(bundle: Path) -> list[str]:
    """Every file the manifest lists exists with its sha256, and has a recorded mode (problems; empty when the bundle
    is intact). The modes of the bundle's own files are not checked: the volume does not keep them."""
    path = bundle / MANIFEST
    if not path.is_file():
        return [f"{path} missing"]
    manifest = json.loads(path.read_text())
    files = manifest.get("files", {})
    modes = manifest.get("modes")
    problems = []
    if manifest.get("version", 1) < MANIFEST_VERSION or not isinstance(modes, dict):
        problems.append(
            f"{MANIFEST} records no file modes (version {manifest.get('version', 1)}): the volume loses them and the "
            "eval could not restore frpc's or the task files' exec bits; rebuild it with stage_bundle.py"
        )
        modes = {}
    else:
        unmoded = sorted(set(files) - set(modes))
        if unmoded:
            problems.append(f"{MANIFEST} has no mode for {len(unmoded)} files, e.g. {unmoded[:3]}")
        bad = sorted(rel for rel, mode in modes.items() if not _valid_mode(mode))
        if bad:
            problems.append(f"{MANIFEST} has invalid modes for {bad[:3]}")
    for rel, digest in sorted(files.items()):
        target = bundle / rel
        if not target.is_file():
            problems.append(f"{rel} missing")
        elif sha256_file(target) != digest:
            problems.append(f"{rel} sha256 differs from the manifest")
    return problems


def _valid_mode(mode) -> bool:
    try:
        value = int(mode, 8)
    except (TypeError, ValueError):
        return False
    return 0 <= value <= 0o777


def restore_modes(modes: dict[str, int], prefix: str, target: Path) -> tuple[int, list[str]]:
    """Give each copy under ``target`` of a bundle file under ``prefix`` its recorded mode (the volume kept none) and
    check it took. Returns how many files were set and the problems (a copy missing, a chmod that did not take)."""
    problems = []
    done = 0
    lead = prefix.rstrip("/") + "/"
    for rel, mode in sorted(modes.items()):
        if not rel.startswith(lead):
            continue
        path = target / rel[len(lead) :]
        if path.is_symlink():
            problems.append(f"{rel}: {path} is a symlink (the bundle has regular files only)")
            continue
        try:
            os.chmod(path, mode)
            got = stat.S_IMODE(os.stat(path).st_mode)
        except OSError as exc:
            problems.append(f"{rel}: cannot set mode {mode:04o} on {path}: {exc}")
            continue
        if got != mode:
            problems.append(f"{rel}: mode {got:04o} after chmod {mode:04o} on {path}")
            continue
        done += 1
    return done, problems


def require_modes(modes: dict[str, int], prefix: str, target: Path, what: str, log=print) -> int:
    """``restore_modes``, raising on any problem (the eval phase then fails cleanly at 'harness ready')."""
    done, problems = restore_modes(modes, prefix, target)
    if problems:
        raise RuntimeError(f"{what}: {len(problems)} file modes not restored: {problems[:5]}")
    executable = sum(1 for rel, mode in modes.items() if rel.startswith(prefix.rstrip("/") + "/") and mode & 0o111)
    log(f"{what}: restored the modes of {done} files ({executable} executable) from the manifest")
    return done


def make_executable(path: Path) -> None:
    """``path`` executable (0755) whatever the volume gave it, checked."""
    os.chmod(path, 0o755)
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"{path} is not executable after chmod 0755")


def check_bundle(harness, bundle: Path) -> list[str]:
    """What the eval needs from the bundle beyond the manifest: the task table, the pinned source configs, the
    sft-gen tree and the requirements."""
    problems = verify_manifest(bundle)
    for rel in (TASKS, REQUIREMENTS, f"{SFT_GEN}/plugins/sft_gen_task_qc.py", f"{SFT_GEN}/sft_gen/task_qc.py"):
        if not (bundle / rel).is_file():
            problems.append(f"{rel} missing")
    if harness.expected_val_tasks is not None and (bundle / TASKS).is_file():
        from prime_rl.trainer.sft.inrun_eval.tasks import parse_task_table

        try:
            val = sum(row.split == "val" for row in parse_task_table((bundle / TASKS).read_text()))
        except ValueError as exc:
            problems.append(f"{TASKS}: {exc}")
        else:
            if val != harness.expected_val_tasks:
                problems.append(
                    f"{TASKS} labels {val} tasks val, expected {harness.expected_val_tasks} (the dataset's validation "
                    "split, data-prep/tb4-val-tasks-v2.txt): rebuild the bundle's task table"
                )
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
            # The tarball carries its own modes (tarfile keeps them); check the interpreter can run all the same.
            if not os.access(candidates[0], os.X_OK):
                raise RuntimeError(f"harness Python {candidates[0]} is not executable after extraction")
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

    modes = file_modes(bundle)
    cwd = work / SFT_GEN
    if cwd.exists():
        shutil.rmtree(cwd)
    shutil.copytree(bundle / SFT_GEN, cwd, symlinks=True)
    info["modes_restored"] = {SFT_GEN: require_modes(modes, SFT_GEN, cwd, "sft-gen", log)}
    staged = stage_frpc(bundle, work / "home", modes, log)
    if staged is not None:
        info["modes_restored"][FRPC] = staged
    cache = stage_host_uv(bundle, env)
    if cache is not None:
        info["modes_restored"][HOST_UV] = require_modes(modes, HOST_UV, cache, "host uv", log)
    hen = HarnessEnv(venv=venv, cwd=cwd, env=env, info=info)
    hen.info["plugins_dir"] = str(plugins)
    return hen


def stage_frpc(bundle: Path, home: Path, modes: dict[str, int], log=print) -> int | None:
    """Put the bundle's frpc where prime_tunnel looks for it (``~/.prime/bin``), with its recorded mode and
    executable: prime_tunnel runs it as is when ``.frpc_version`` matches (it never re-downloads or chmods it), so a
    copy without its exec bit would fail every tunnel. Returns the files whose modes were restored, None without
    an frpc in the bundle (prime_tunnel then downloads it)."""
    src = bundle / FRPC
    if not src.is_dir():
        return None
    bin_dir = home / ".prime" / "bin"
    shutil.copytree(src, bin_dir, dirs_exist_ok=True)
    done = require_modes(modes, FRPC, bin_dir, "frpc", log)
    make_executable(bin_dir / "frpc")
    return done


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


def api_base_url(base: dict[str, str]) -> str:
    """The Prime API the harness's clients talk to, resolved as prime-sandboxes 0.4.3 / prime-tunnel 0.1.14 do for a
    home without a CLI config (``PRIME_API_BASE_URL`` / ``PRIME_BASE_URL``, else the default)."""
    url = base.get("PRIME_API_BASE_URL") or base.get("PRIME_BASE_URL") or PRIME_API_BASE_URL
    return url.rstrip("/").removesuffix("/api/v1")


PROBES = (
    # (name, path, query): what every rollout needs from the key — the team's sandboxes and the tunnel service.
    ("sandboxes", "/api/v1/sandbox", {"page": 1, "per_page": 1, "team_id": None}),
    ("tunnels", "/api/v1/tunnel", {"page": 1, "perPage": 1, "teamId": None}),
)
PROBE_STATUS = {401: "no_credentials", 402: "forbidden", 403: "forbidden"}
"""HTTP status of a probe -> the eval status it means (the key is not accepted / not allowed for the team)."""


def _probe_get(url: str, api_key: str, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "prime-rl-inrun-eval",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(65536)
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return 0, f"HTTP {response.status} without JSON"
            if not isinstance(payload, dict):
                return 0, f"HTTP {response.status} with a {type(payload).__name__}"
            return response.status, ""
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(2048).decode(errors="replace")
        except Exception:
            detail = ""
        return exc.code, detail
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def probe_credentials(
    api_key: str,
    team: str,
    base: dict[str, str],
    attempts: int = 6,
    wait: float = 60.0,
    timeout: float = 30.0,
    get=_probe_get,
    sleep=time.sleep,
) -> dict:
    """Cheap authenticated calls with the run's key before anything is served: list one of the team's sandboxes and
    one tunnel (what every rollout needs: creating the team's sandboxes, its tunnel to the router). Returns
    ``{"status": "ok" | "no_credentials" | "forbidden" | "inconclusive", "checks": {...}, "error": ...}``: 401 is
    ``no_credentials``, 402 / 403 ``forbidden`` (both end the eval phase before serving); a network error or a 5xx
    is retried and, if it persists, ``inconclusive`` (the eval goes on: it only fails fast on a definite refusal).
    The key is only sent, never logged or recorded (any echo of it in an error body is masked)."""
    root = api_base_url(base)
    checks: dict[str, dict] = {}
    for name, path, query in PROBES:
        params = {k: (team if v is None else v) for k, v in query.items()}
        url = f"{root}{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(1, attempts + 1):
            code, detail = get(url, api_key, timeout)
            detail = detail.replace(api_key, "***")[:300] if api_key else detail[:300]
            checks[name] = {"http": code, "attempts": attempt, **({"detail": detail} if detail else {})}
            if 200 <= code < 300:
                break
            if code in PROBE_STATUS:
                status = PROBE_STATUS[code]
                error = f"the Prime API refused PRIME_API_KEY for {name} of team {team}: HTTP {code} {detail}".strip()
                return {"status": status, "base_url": root, "checks": checks, "error": error}
            if 400 <= code < 500:
                break  # e.g. a 404 / 422 from a changed API: not a credential verdict
            if attempt < attempts:
                sleep(wait)
    failed = {name: check for name, check in checks.items() if not 200 <= check["http"] < 300}
    if failed:
        error = f"credential probe inconclusive (the eval goes on): {failed}"
        return {"status": "inconclusive", "base_url": root, "checks": checks, "error": error}
    return {"status": "ok", "base_url": root, "checks": checks, "error": None}


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


def stage_harbor_caches(bundle: Path, home: Path, names: list[str], log=print) -> list[str]:
    """Copy the bundle's Harbor dataset exports to ``~/.cache/harbor`` of the harness and give every file its
    recorded mode (the task files' exec bits: tests/test.sh, helper scripts; the volume kept none). Problems: missing
    caches, modes that could not be restored."""
    problems = []
    modes = file_modes(bundle)
    for name in names:
        src = bundle / HARBOR_CACHE / name
        dst = home / ".cache" / "harbor" / name
        if not src.is_dir():
            problems.append(f"{HARBOR_CACHE}/{name} missing")
            continue
        if not dst.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, symlinks=True)
        prefix = f"{HARBOR_CACHE}/{name}"
        done, failed = restore_modes(modes, prefix, dst)
        if not done:
            failed.append(f"{prefix}: no file modes in the manifest")
        problems += failed[:20]
        executable = sum(1 for rel, mode in modes.items() if rel.startswith(prefix + "/") and mode & 0o111)
        log(f"{prefix}: restored the modes of {done} files ({executable} executable) from the manifest")
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
            log(f"eval safety cap reached: stopping the flow runs {running} (their draws still running are 'cut')")
            stopped_at = time.time()
            serving.stop([procs[label] for label in running], grace=grace)
            for label in running:
                status[label] = {
                    "exit_code": procs[label].popen.returncode,
                    "seconds": time.time() - start,
                    "stopped": True,
                    "stopped_at": stopped_at,
                }
            return status
        if time.time() - last_note > 1800:
            last_note = time.time()
            left = "no cap" if deadline is None else f"{(deadline - time.time()) / 3600:.1f} h to the cap"
            log(f"flow runs still running: {running} ({(time.time() - start) / 3600:.1f} h in, {left})")
        time.sleep(poll)
