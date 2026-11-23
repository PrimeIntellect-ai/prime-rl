"""Run prime-rl's GPU test suites on Modal, one container per test group.

    uvx modal run scripts/modal_ci.py --suite tiny|full|parallel [--group name]

The image installs the third-party dependencies from uv.lock and every pyproject.toml, then the local packages
(packages/ and the deps/ submodules) as editable installs. The source, tests and configs are added when a container
starts, so a code change does not rebuild the image. Each group's compile caches (Triton, Inductor, vLLM, TileLang, C++
extensions) persist in a Modal Volume, keyed by uv.lock, the GPU type and the suite, and so does its Hugging Face Hub
cache (the models and datasets its tests download), keyed by the suite.

A suite's group without paths runs every test with the suite's marker that no other group claims, so a new test
always runs somewhere.
"""

import asyncio
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import modal

# Modal may serve an H200 instead, the same architecture
GPU = "H100"
REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/repo"
CACHE = "/cache"
OUTPUTS = "/tmp/outputs"
IGNORE = ["**/.git", "**/.venv", "**/__pycache__", "**/*.pyc"]


@dataclass(frozen=True)
class Group:
    name: str
    gpus: int
    paths: tuple[str, ...] = ()
    timeout: int = 20
    """Minutes"""


@dataclass(frozen=True)
class Suite:
    marker: str
    groups: tuple[Group, ...]

    def pytest_args(self, group: Group) -> list[str]:
        if group.paths:
            return [*group.paths, "-m", self.marker]
        claimed = [path for other in self.groups for path in other.paths]
        return ["tests/unit", "tests/integration", "-m", self.marker, *(f"--ignore={path}" for path in claimed)]


SUITES = {
    "tiny": Suite(
        marker="tiny",
        groups=(
            Group("rl", gpus=2, paths=("tests/integration/test_reverse_text_tiny.py",)),
            Group("sft", gpus=2, paths=("tests/integration/test_reverse_text_sft.py",)),
            Group("invariance", gpus=2, paths=("tests/unit/train/test_distributed_invariance.py",)),
            Group("offload", gpus=2, paths=("tests/unit/train/test_offload_checkpoint.py",)),
            Group("rest", gpus=1),
        ),
    ),
    # gpu-full.yml runs each group as its own job: a new group needs an entry in its matrix
    "full": Suite(
        marker="gpu",
        groups=(
            Group("dsv4-kernels", gpus=1, timeout=25, paths=("tests/unit/train/models/test_deepseek_v4_kernels.py",)),
            Group("dsv4", gpus=1, paths=("tests/unit/train/models/test_deepseek_v4.py",)),
            Group("invariance", gpus=2, timeout=35, paths=("tests/unit/train/test_distributed_invariance.py",)),
            Group("offload", gpus=2, paths=("tests/unit/train/test_offload_checkpoint.py",)),
            Group("vllm-parity", gpus=1, paths=("tests/unit/train/models/test_vllm_parity.py",)),
            Group("compile", gpus=2, timeout=60, paths=("tests/unit/train/test_compile.py",)),
            Group("rest", gpus=2, timeout=30),
            # Each integration test starts its own RL or SFT runs
            *(
                Group(test, gpus=2, timeout=30, paths=(f"tests/integration/test_{test}.py",))
                for test in (
                    "reverse_text",
                    "reverse_text_lora",
                    "reverse_text_moe",
                    "reverse_text_rl_opd",
                    "reverse_text_rl_sft",
                    "reverse_text_sft",
                    "reverse_text_sft_lora",
                    "alphabet_sort",
                    "pretrain",
                )
            ),
        ),
    ),
    "parallel": Suite(
        marker="gpu",
        groups=(Group("invariance", gpus=4, paths=("tests/regression/test_parallel_invariance.py",)),),
    ),
}


def tracked_files(*paths: str) -> list[str]:
    """The files git tracks under `paths`, including those of initialized submodules."""
    command = ["git", "ls-files", "-z", "--recurse-submodules", "--", *paths]
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\0")[:-1]


def staged(name: str, paths: list[str]) -> Path:
    """Copies of repo files with the modes git checks out. Image layers are keyed by file content and mode, so a
    local checkout (any umask, untracked files) and CI build the same layers."""
    root = Path(STAGING.name, name)
    for path in paths:
        source, target = REPO / path, root / path
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o755 if source.stat().st_mode & 0o100 else 0o644)
    return root


# The container imports this file too, without building the image
if modal.is_local():
    STAGING = tempfile.TemporaryDirectory(prefix="modal-ci-")
    manifests = staged(
        "manifests",
        [
            path
            for path in tracked_files("pyproject.toml", "uv.lock", "packages", "deps")
            if path.endswith(("pyproject.toml", "uv.lock"))
        ],
    )
    local_packages = staged("local-packages", tracked_files("README.md", "packages", "deps"))
else:
    manifests = local_packages = Path()

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e",
        add_python="3.12",
    )
    .apt_install("git", "build-essential")
    .dockerfile_commands("COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /uvx /usr/local/bin/")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "UV_PROJECT_ENVIRONMENT": "/opt/venv",
            "UV_PYTHON_PREFERENCE": "only-system",
            "UV_LINK_MODE": "copy",
        }
    )
    .workdir(REMOTE_REPO)
    .add_local_dir(manifests, REMOTE_REPO, copy=True)
    # uv cannot check the lock without the local sources, which also lack their git versions here; the CPU tests
    # job runs `uv sync --locked`
    .run_commands("uv sync --frozen --all-extras --all-packages --no-install-local --compile-bytecode --no-cache")
    # The integration tests' dashboard smoke test drives the headless Chromium of the locked playwright version
    .run_commands("/opt/venv/bin/playwright install --with-deps --only-shell chromium")
    .add_local_dir(local_packages, REMOTE_REPO, copy=True)
    # hatchling puts src/ on the path only if it finds a module there; the source is added when a container starts.
    # Modal's builder sets UV_COMPILE_BYTECODE=1, which would recompile the whole venv in this layer.
    .run_commands(
        "mkdir -p src/prime_rl && touch src/prime_rl/_stub.py"
        " && UV_COMPILE_BYTECODE=0 uv sync --frozen --all-extras --all-packages --no-cache"
        " && rm src/prime_rl/_stub.py"
    )
)
for directory in ("src", "tests", "configs", "examples", "tools"):
    image = image.add_local_dir(REPO / directory, f"{REMOTE_REPO}/{directory}", ignore=IGNORE)

app = modal.App("prime-rl-ci", image=image)
cache = modal.Volume.from_name("prime-rl-ci-cache", create_if_missing=True)


def restore_cache(archive: Path, target: Path) -> set[Path]:
    """Unpack a cache from the Volume. Returns the files it holds."""
    target.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        subprocess.run(["tar", "-xf", archive, "-C", target], check=True)
    return cached_files(target)


def save_cache(archive: Path, target: Path, restored: set[Path]) -> None:
    """Repack a cache into the Volume if the run added files to it."""
    if cached_files(target) <= restored:
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(f"{archive.name}.partial")
    subprocess.run(["tar", "-cf", partial, "-C", target, "."], check=True)
    partial.replace(archive)


def cached_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


@app.function(
    gpu=GPU,
    volumes={CACHE: cache},
    secrets=[modal.Secret.from_name("prime-rl-ci")],
    timeout=20 * 60,
    # Otherwise a group can run in the container another group just finished in, with its outputs and caches
    single_use_containers=True,
)
def run_group(
    name: str, pytest_args: list[str], suite: str, lock_hash: str, ref_name: str
) -> tuple[int, bytes, float, float]:
    """Run one pytest group. Returns its exit code, a tarball of its logs, and its start time and duration."""
    started = time.time()
    compile_cache = Path("/root/.cache/compile")
    hub_cache = Path("/root/.cache/huggingface/hub")
    # A Volume reads small files slowly (50 ms to 0.5 s each, by region), so each cache moves as one tar file. Hub
    # downloads are slow or fail in some regions, and the Hub cache depends on neither uv.lock nor the GPU.
    archives = {
        compile_cache: Path(CACHE, f"{lock_hash}-{GPU}", suite, f"{name}.tar"),
        hub_cache: Path(CACHE, "huggingface", suite, f"{name}.tar"),
    }
    restored = {target: restore_cache(archive, target) for target, archive in archives.items()}
    for kind in ("triton", "inductor", "vllm", "tilelang", "torch_extensions"):
        (compile_cache / kind).mkdir(exist_ok=True)
    # The rl and sft launchers pin these two caches to /tmp paths (prime_rl/utils/process.py)
    os.symlink(compile_cache / "triton", f"/tmp/triton-{os.getuid()}")
    os.symlink(compile_cache / "vllm", f"/tmp/.vllm-cache-{os.getuid()}")
    # Modal's PYTHONPATH points at its own client packages
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env |= {
        "USERNAME_CI": "CI_RUNNER",
        "UV_NO_SYNC": "1",
        "WANDB_MODE": "offline",
        "PYTEST_OUTPUT_DIR": OUTPUTS,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "GITHUB_REF_NAME": ref_name,
        "TRITON_CACHE_DIR": f"{compile_cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{compile_cache}/inductor",
        "VLLM_CACHE_ROOT": f"{compile_cache}/vllm",
        "TILELANG_CACHE_DIR": f"{compile_cache}/tilelang",
        # The CPU AdamW of full offload is a C++ extension built on first use
        "TORCH_EXTENSIONS_DIR": f"{compile_cache}/torch_extensions",
        "HF_HUB_CACHE": str(hub_cache),
    }
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    region = os.environ.get("MODAL_REGION")
    print(f"[{name}] {', '.join(gpus.stdout.splitlines())} in {region}; cache restored in {time.time() - started:.0f}s")
    os.makedirs(OUTPUTS, exist_ok=True)
    with open(f"{OUTPUTS}/pytest.log", "w") as log:
        process = subprocess.Popen(
            ["uv", "run", "pytest", *pytest_args, "-vv", "--durations=0"],
            cwd=REMOTE_REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)
            log.write(line)
    exit_code = process.wait()
    for target, archive in archives.items():
        save_cache(archive, target, restored[target])
    cache.commit()

    artifacts = io.BytesIO()
    with tarfile.open(fileobj=artifacts, mode="w:gz") as tar:
        for path in Path(OUTPUTS).rglob("*.log"):
            if not path.is_symlink():
                tar.add(path, arcname=path.relative_to(OUTPUTS))
    return exit_code, artifacts.getvalue(), started, time.time() - started


@app.local_entrypoint()
async def main(suite: str = "tiny", group: str = ""):
    """`group`: run only the suite's group of that name."""
    lock_hash = hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()[:16]
    ref_name = os.environ.get("GITHUB_REF_NAME", "local")
    artifacts_dir = REPO / "outputs" / "modal-ci"
    test_suite = SUITES[suite]
    groups = [{g.name: g for g in test_suite.groups}[group]] if group else test_suite.groups

    async def run(group: Group) -> tuple[int, float, float]:
        fn = run_group.with_options(
            gpu=f"{GPU}:{group.gpus}", cpu=8.0 * group.gpus, memory=65536 * group.gpus, timeout=group.timeout * 60
        )
        spawned = time.time()
        exit_code, artifacts, started, duration = await fn.remote.aio(
            group.name, test_suite.pytest_args(group), suite, lock_hash, ref_name
        )
        if (artifacts_dir / group.name).exists():
            shutil.rmtree(artifacts_dir / group.name)
        with tarfile.open(fileobj=io.BytesIO(artifacts)) as tar:
            tar.extractall(artifacts_dir / group.name, filter="data")
        return exit_code, started - spawned, duration

    start = time.time()
    results = await asyncio.gather(*(run(group) for group in groups), return_exceptions=True)
    wall = time.time() - start

    width = max(len(group.name) for group in groups) + 2
    print(f"\n{'group':<{width}}{'gpus':>5}  {'result':<8}{'startup':>9}{'run':>8}")
    failed = False
    for group, result in zip(groups, results):
        if isinstance(result, BaseException):
            failed = True
            print(f"{group.name:<{width}}{group.gpus:>5}  error: {result!r}")
            continue
        exit_code, startup, duration = result
        failed |= exit_code != 0
        status = "passed" if exit_code == 0 else f"exit {exit_code}"
        print(f"{group.name:<{width}}{group.gpus:>5}  {status:<8}{startup:>8.0f}s{duration:>7.0f}s")
    print(f"Suite {suite} on {GPU}: {'failed' if failed else 'passed'} in {wall:.0f}s. Logs: {artifacts_dir}")
    if failed:
        sys.exit(1)
