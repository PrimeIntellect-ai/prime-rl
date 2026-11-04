"""prebuilt-harbor: Harbor tasks graded in separate verifier boxes from prebuilt images,
loaded from a local directory or from a task archive in a Hugging Face dataset repo
instead of Harbor Hub.

Every task asks for a separate verifier (`[verifier].environment_mode = "separate"`):
the agent works in its own box, the declared artifacts (`/app`, ...) are collected,
and `tests/test.sh` grades them in a second box. Verifiers never builds Dockerfiles,
so each task needs two prebuilt, pullable images:

- the agent image, built from `environment/`, named by `image_template`;
- the verifier image, built from `tests/` (it rebuilds the agent environment and adds
  `/tests`), named by `verifier_image_template`.

What is specific to one dataset (tasks held back and why, per-task agent users, ...)
lives in its overrides file (see `prebuilt_harbor.overrides`). Where the task dirs come
from is `prebuilt_harbor.source`. Artifact symlinks travel to the verifier box the way
Harbor moves them (see `prebuilt_harbor.artifacts`).

`vf-validate prebuilt-harbor` checks each task without a model: the gold check runs
`solution/solve.sh` like Harbor's oracle agent and must score 1; the noop check
grades the untouched box and must score below 1.
"""

import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.errors import TaskError
from verifiers.v1.rollout import resolve_agent_runtime
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.deployment import Deployment
from verifiers.v1.task import task_key
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset
from verifiers.v1.tasksets.harbor.taskset import make_tar, parse_task
from verifiers.v1.trace import Trace

from prebuilt_harbor import artifacts, source
from prebuilt_harbor.overrides import Overrides

# Under the library logger, so vf-eval/vf-validate route these records to the console
# and to the run's log file.
logger = logging.getLogger("verifiers.v1.tasksets.prebuilt_harbor")

GOLD_AS_ROOT_ENV = "PREBUILT_HARBOR_GOLD_AS_ROOT"
"""Set to 1 to run every gold check's solve.sh as root instead of the agent user: a
triage switch that tells a reference solution needing root from a broken task. The
overrides' `gold_as_root` tasks always run it as root."""
SOLUTION_DIR = "/solution"
"""Where Harbor's oracle agent stages `solution/`; some solve.sh files assume it."""
MAX_ORACLE_OUTPUT_CHARS = 16 * 1024
ISOLATED_TEST_SH = r"""
UP_LO=$(cat <<'UP_LO_END'
up_lo() {
    ip link set lo up 2>/dev/null && return 0
    ifconfig lo up 2>/dev/null && return 0
    for py in python3 python; do
        command -v "$py" >/dev/null 2>&1 || continue
        "$py" -c 'import fcntl, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
r = bytearray(40); r[:2] = b"lo"
fcntl.ioctl(s, 0x8913, r)
r[16:18] = (int.from_bytes(r[16:18], "little") | 1).to_bytes(2, "little")
fcntl.ioctl(s, 0x8914, bytes(r))' 2>/dev/null && return 0
    done
    return 1
}
UP_LO_END
)
RUN='bash /tests/test.sh; status=$?; exit $status'
if unshare --pid --fork --mount-proc true 2>/dev/null; then
    if [ "${PREBUILT_HARBOR_ISOLATE_NET:-0}" = 1 ]; then
        if unshare --net sh -c "$UP_LO
up_lo" 2>/dev/null; then
            exec unshare --net --pid --fork --mount-proc sh -c "$UP_LO
up_lo || exit 125
$RUN"
        fi
        echo 'prebuilt-harbor: cannot bring up loopback in a private network namespace; test.sh shares the box network' >&2
    fi
    exec unshare --pid --fork --mount-proc sh -c "$RUN"
fi
echo 'prebuilt-harbor: unshare is unavailable; running test.sh without a private PID namespace' >&2
exec bash /tests/test.sh
"""
"""How tests/test.sh is launched: as root, as `bash /tests/test.sh` (verifiers that scan
/proc/*/cmdline for candidate processes rely on that command line), in a private PID
and /proc namespace with a plain `sh` as PID 1, the way a container shows a `docker
exec` only the container's own processes. A Prime VM otherwise shows the platform's
root daemons, which "same-uid process" sweeps in verifiers take for leftover candidate
processes. With PREBUILT_HARBOR_ISOLATE_NET=1 (tasks without internet) also in a
private network namespace with only loopback up, like Docker's `network_mode: none`:
Prime blocks egress by dropping packets, so every DNS lookup (`getfqdn`,
`HTTPServer.server_bind`) stalls ~10 s instead of failing at once, which breaks
verifiers with startup deadlines."""
EXPOSE_IMAGE_RUN = r"""
grep -qE '^[0-9]+ [0-9]+ [^ ]+ [^ ]+ /run ' /proc/self/mountinfo || exit 0
d=$(mktemp -d /tmp/vf-prebuilt-rootfs.XXXXXX) || exit 1
mount --bind / "$d" 2>/dev/null || mount -o bind / "$d" || { rmdir "$d"; exit 3; }
s=0
for p in "$d"/run/* "$d"/run/.[!.]* "$d"/run/..?*; do
    [ -e "$p" ] || [ -L "$p" ] || continue
    n=${p##*/}
    if [ ! -e "/run/$n" ] && [ ! -L "/run/$n" ]; then cp -a "$p" "/run/$n" || s=1; fi
done
if [ -d "$d/run" ]; then
    chown "$(stat -c %u:%g "$d/run")" /run || s=1
    chmod "$(stat -c %a "$d/run")" /run || s=1
fi
umount "$d" && rmdir "$d"
exit $s
"""
"""A Prime sandbox mounts an empty tmpfs on /run, hiding whatever the image put there
(images that set up daemons often create directories under /run). Docker shows the
image's /run, so copy it into the tmpfs from a bind mount of / (which does not carry
the tmpfs), without replacing anything already there, and give the mount point the
image /run's owner and mode (some images chown /run to the agent)."""


class PrebuiltHarborConfig(HarborConfig):
    tasks_dir: Path | None = None
    """Local directory of task dirs (each with a task.toml), on the machine that loads
    the taskset (the eval / orchestrator / vf-validate process). Unset, with no
    `hf_repo`, reads $PREBUILT_HARBOR_TASKS_DIR."""
    overrides: Path | None = None
    """Overrides file (see `prebuilt_harbor.overrides`). Unset uses `overrides.json`
    beside the task dirs' directory if there is one: the bundle layout,
    `<root>/overrides.json` next to `<root>/tasks/`. Without one, no task is held back."""
    hf_repo: str | None = None
    """Hugging Face dataset repo holding the task archive, in place of `tasks_dir`. The
    archive's root holds `tasks/<name>/` and, optionally, `overrides.json`."""
    hf_revision: str | None = None
    """Revision of `hf_repo` to download. Pin a commit sha: a branch or tag is resolved
    again by every process that loads the taskset. Unset is `main`."""
    hf_filename: str = "tasks.tar.gz"
    """The archive's path in `hf_repo`."""
    hf_sha256: str | None = Field(None, pattern=r"^[0-9a-fA-F]{64}$")
    """Expected sha256 of the archive; a mismatch fails the load."""
    hf_token_env: str | None = "HF_TOKEN"
    """Environment variable holding the Hugging Face token; None downloads anonymously."""
    cache_dir: Path | None = None
    """Where downloaded and extracted archives are kept. Unset reads
    $PREBUILT_HARBOR_CACHE_DIR, else uses a writable `~/.cache/prebuilt-harbor`, else a
    per-user directory under the system temp dir."""
    registry: str | None = None
    """Prefix the images were pushed under: the Image Reference that
    `prime images list` shows, minus `/<name>:<tag>` (e.g. `prime/<team-slug>`).
    Required."""
    tag: str = "v1"
    """Tag both images were pushed with."""
    image_template: str = "{registry}/{task}:{tag}"
    """Agent image reference; `{task}` is the task directory name."""
    verifier_image_template: str = "{registry}/{task}-verifier:{tag}"
    """Verifier image reference, built from `tests/Dockerfile`."""
    agent_user: str | None = "agent"
    """User the agent runs as when the task declares no `[agent].user`. Harbor runs the
    agent as the image's `USER`, and tasks that hide their oracles from the agent rely on
    that account; the overrides' `agent_users` name it for tasks whose image calls it
    something else. None runs as the runtime's default user, usually root."""
    include_groups: list[str] = Field(default_factory=list)
    """Optional hold-back groups of the overrides file to load anyway."""
    ignore_dockerfile: bool = True
    """Parse Dockerfile-only tasks; `image` and `verifier_image` are set from the
    templates right after."""
    artifact_max_bytes: int = Field(2 * 1024 * 1024 * 1024, gt=0)
    """`/app` and the other declared paths travel to the verifier box (Harbor's
    `docker cp` has no cap); verifiers' 32 MiB default is too small for whole
    workspaces. Archives are held in memory and saved with eval runs, so this is a
    ceiling, not a typical size."""
    min_disk_gb: float | None = Field(16.0, gt=0)
    """Disk floor for the agent and verifier boxes. A Prime sandbox's disk holds the
    image too and is enforced; Docker does not enforce `storage_mb`, which tasks often
    set low. None keeps each task's `storage_mb`."""
    verifier_cpus: float | None = Field(2.0, gt=0)
    """vCPUs for the verifier box (at least the task's own). Docker grants a task's
    `cpus` as a CFS quota with all host cores visible; a Prime box gets exactly its
    vCPUs, and verifiers that need parallelism (a listen backlog, signal delivery to
    helpers) can fail on one. None keeps the task's `cpus`. The agent box always keeps
    the task's `cpus`."""

    @model_validator(mode="after")
    def _one_source(self) -> "PrebuiltHarborConfig":
        if self.tasks_dir is not None and self.hf_repo is not None:
            raise ValueError("set taskset.tasks_dir or taskset.hf_repo, not both")
        return self


class PrebuiltHarborTask(HarborTask):
    gold_as_root: bool = False
    """Whether the gold check runs solve.sh as root (the overrides' `gold_as_root`)."""

    @property
    def key(self) -> str:
        """The content hash without `task_dir`, which names wherever this machine keeps
        the tasks (a downloaded archive lands in an environment-dependent cache), so a
        task keeps its key across machines and restarts."""
        return task_key(self.data.model_dump(mode="json", exclude_none=True, exclude={"task_dir"}))

    async def setup(self, runtime: Runtime, deployment: Deployment | None = None) -> None:
        """Show the image's /run as Docker would (see `EXPOSE_IMAGE_RUN`), in the agent
        box and in the verifier box alike, then do Harbor's setup."""
        result = await runtime.run(["sh", "-c", EXPOSE_IMAGE_RUN], {})
        if result.exit_code == 3:
            logger.warning(
                "%s: cannot bind-mount / to expose the image's /run: %s",
                self.data.name,
                (result.stderr or result.stdout).strip()[-300:],
            )
        elif result.exit_code:
            raise TaskError(
                f"exposing the image's /run failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        await super().setup(runtime, deployment)

    async def run_verifier(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        """Run tests/test.sh as root with a container's view of the process table (see
        `ISOLATED_TEST_SH`), and with /logs/verifier at 0777, as Harbor leaves it. Root
        is the identity Harbor verifiers are written for. Here every process test.sh
        started dies with it, before the reward is read, and only reward.txt counts (see
        `read_reward_json`)."""
        opened = await runtime.run(["sh", "-c", "mkdir -p /logs/verifier && chmod 0777 /logs/verifier"], {})
        if opened.exit_code:
            raise TaskError(f"preparing /logs/verifier failed: {(opened.stderr or opened.stdout).strip()[-300:]}")
        # A box without network (Harbor's `network_mode: none`): loopback only, failing
        # fast. The box's own policy, so a scoring or runtime override counts.
        config = runtime.config
        no_network = isinstance(config, NetworkPolicyConfig) and not config.allow
        return await super().run_verifier(_IsolatedTests(runtime, no_network), trace)

    async def read_reward_json(self, runtime: Runtime) -> dict[str, float] | None:
        """No reward.json: the reward is reward.txt's. Harbor reads a reward.json first
        whenever one exists, so code test.sh runs could otherwise set its own reward by
        writing one."""
        return None

    def grading_task(self, runtime: Runtime, *, scoring: ScoringConfig):
        """Every task grades in its own verifier image, which holds what `test.sh`
        needs; the agent image does not. Refuse shared scoring rather than grade there."""
        if scoring.mode != "isolated":
            raise TaskError(
                f"{self.data.name}: prebuilt-harbor grades in each task's verifier image; "
                "shared scoring is not supported"
            )
        return super().grading_task(runtime, scoring=scoring)

    async def finalize(
        self,
        trace: Trace,
        runtime: Runtime,
        sidecars: dict[str, Runtime] | None = None,
    ) -> None:
        """Collect artifacts, then stage each root for `stage_verifier` to put in place
        the way Harbor's separate verifier uploads it (see `prebuilt_harbor.artifacts`):
        a root the agent box lacks is skipped, a symlinked directory root travels as its
        content, every file arrives root-owned, every symlink is set aside, and each
        directory root's owner and mode in the agent box are recorded."""
        await super().finalize(trace, runtime, sidecars)
        if sidecars:
            raise TaskError("prebuilt-harbor tasks have no Compose services")
        for key in (artifacts.INFO_KEY, artifacts.ROOTS_KEY):
            trace.info.pop(key, None)
        if self.data.verifier is None:
            return
        collected = trace.state.artifacts
        roots: list[dict] = []
        staged: dict[str, bytes] = {}
        carried: dict[str, list[dict]] = {}
        items = list(collected.items())
        for index, (root, archive) in enumerate(items):
            path = PurePosixPath(root)
            if path.is_relative_to("/tests") or PurePosixPath(artifacts.STAGED_DIR).is_relative_to(path):
                raise TaskError(f"artifact {root!r} cannot be restored into the verifier box")
            if archive is None:
                continue  # Harbor skips a root the agent box lacks; the image's stays.
            if artifacts.root_is_symlink(root, archive):
                if not await _is_dir(runtime, root):
                    # Harbor copies a link to a file, or a dangling one, out as a link and
                    # uploads it only if it resolves on the orchestrator host, which a
                    # path inside the box does not: the image's copy stays.
                    continue
                # Harbor copies `<root>/.`: the target's content, as a real directory.
                if any(a.exclude for a in self.data.artifacts if str(PurePosixPath(a.source)) == root):
                    raise TaskError(f"artifact {root!r} is a symlink and has excludes")
                # The cap covers what travels: roots staged so far and those still raw.
                used = sum(len(part) for part in staged.values()) + sum(len(a or b"") for _, a in items[index + 1 :])
                archive = await _collect_dereferenced(runtime, root, self.data.artifact_max_bytes - used)
            entry = artifacts.root_entry(root, archive)
            if entry is None:
                continue  # Nothing collectable there (a FIFO or device node was dropped).
            scratch = f"{artifacts.STAGED_DIR}/{len(roots)}{root}"
            part, links = artifacts.split_symlinks(scratch, artifacts.relocate(root, scratch, archive))
            staged[scratch] = artifacts.owned_by_root(scratch, part)
            if links:
                carried[scratch] = [{**link, "uid": 0, "gid": 0} for link in links]
            if entry["kind"] == "file":
                entry = {"kind": "file"}
            roots.append({"root": root, **entry, "scratch": scratch})
        if not staged:
            # Keep one harmless entry: with none, the caller would collect raw archives
            # again and `restore` would delete roots the agent box lacks from the image.
            staged[artifacts.EMPTY_KEY] = artifacts.empty_archive()
        collected.clear()
        collected.update(staged)
        trace.info[artifacts.ROOTS_KEY] = roots
        if carried:
            trace.info[artifacts.INFO_KEY] = carried

    async def stage_verifier(self, trace: Trace, runtime: Runtime) -> None:
        """After `restore` has extracted the staged roots into this verifier box,
        recreate the symlinks `finalize` set aside and move each root into place, in
        declared order, then stage the tests."""
        roots = trace.info.get(artifacts.ROOTS_KEY) or []
        carried = trace.info.get(artifacts.INFO_KEY) or {}
        if self.data.verifier is not None and (roots or carried):
            scratch = {entry["scratch"] for entry in roots}
            if unknown := (scratch | set(carried)) - set(trace.state.artifacts):
                raise TaskError(f"artifact staging recorded for uncollected keys {sorted(unknown)}")
            if unknown := set(carried) - scratch:
                raise TaskError(f"symlinks recorded for unstaged keys {sorted(unknown)}")
            script = artifacts.link_script(carried) + artifacts.place_script(roots)
            path = f"/tmp/vf-prebuilt-restore-{uuid.uuid4().hex}.sh"
            await runtime.write(path, script)
            result = await runtime.run(["sh", "-c", 'sh "$1"; s=$?; rm -f "$1"; exit $s', "sh", path], {})
            if result.exit_code:
                raise TaskError(
                    f"putting artifacts in place failed (exit {result.exit_code}): "
                    f"{(result.stderr or result.stdout).strip()[-500:]}"
                )
        await super().stage_verifier(trace, runtime)

    async def validate(self, trace: Trace, runtime: Runtime) -> bool:
        """Gold check for `vf-validate`: stage `solution/` at /solution, run solve.sh
        as the agent user the way Harbor's oracle agent does (as root for the overrides'
        `gold_as_root` tasks, or for all with $PREBUILT_HARBOR_GOLD_AS_ROOT=1), then
        grade it in a fresh verifier box as a rollout would. Valid iff it scores 1.
        Needs task dirs that include `solution/`."""
        solution = Path(self.data.task_dir) / "solution"
        if not solution.is_dir():
            raise TaskError(f"{self.data.name}: the gold check needs {solution}, which these task dirs lack")
        archive = make_tar(solution)
        await runtime.write("/tmp/solution.tgz", archive)
        staged = await runtime.run(
            [
                "sh",
                "-c",
                f"rm -rf {SOLUTION_DIR} && mkdir -p {SOLUTION_DIR} && "
                f"tar --no-same-owner -xzf /tmp/solution.tgz -C {SOLUTION_DIR} && "
                f"rm -f /tmp/solution.tgz && chmod -R a+rX {SOLUTION_DIR} && "
                f"chmod +x {SOLUTION_DIR}/solve.sh",
            ],
            {},
        )
        if staged.exit_code:
            raise TaskError(
                f"staging solution failed (exit {staged.exit_code}): {(staged.stderr or staged.stdout).strip()[-500:]}"
            )
        as_root = os.environ.get(GOLD_AS_ROOT_ENV) == "1" or self.gold_as_root
        oracle_user = None if as_root else self.data.agent_user
        agent = await resolve_agent_runtime(runtime, oracle_user)
        result = await agent.run([f"{SOLUTION_DIR}/solve.sh"], {"DEBIAN_FRONTEND": "noninteractive"})
        trace.info["oracle"] = {
            "exit_code": result.exit_code,
            "stdout": result.stdout[-MAX_ORACLE_OUTPUT_CHARS:],
            "stderr": result.stderr[-MAX_ORACLE_OUTPUT_CHARS:],
        }
        # Harbor's oracle agent records a failing exit code and grades anyway.
        try:
            await self.finalize(trace, runtime)
            await vf.grade_submission(self, trace, runtime)
        except Exception as exc:
            self._log_gold_failure(trace, result, f"raised {type(exc).__name__}: {exc}")
            raise
        if trace.reward == 1.0:
            return True
        self._log_gold_failure(trace, result, f"scored {trace.reward}")
        return False

    def _log_gold_failure(self, trace: Trace, oracle, outcome: str) -> None:
        """vf-validate keeps only valid/invalid/error; log why, so a failed sweep is
        triageable from its log."""
        verifier = trace.info.get("scoring_diagnostics", {}).get("harbor", {})
        logger.warning(
            "prebuilt-harbor gold check: %s %s (solve.sh exit %s)\n"
            "--- solve.sh stdout (tail) ---\n%s\n--- solve.sh stderr (tail) ---\n%s\n"
            "--- test.sh exit %s, stdout (tail) ---\n%s\n--- test.sh stderr (tail) ---\n%s",
            self.data.name,
            outcome,
            oracle.exit_code,
            oracle.stdout[-2000:],
            oracle.stderr[-2000:],
            verifier.get("exit_code"),
            verifier.get("stdout", "")[-6000:],
            verifier.get("stderr", "")[-3000:],
        )


class _IsolatedTests:
    """`runtime`, launching HarborTask's `bash /tests/test.sh` through
    `ISOLATED_TEST_SH`; every other call passes through unchanged."""

    def __init__(self, runtime: Runtime, no_network: bool) -> None:
        self._runtime = runtime
        # A plain shell variable: test.sh does not inherit it.
        self._script = ("PREBUILT_HARBOR_ISOLATE_NET=1\n" if no_network else "") + ISOLATED_TEST_SH

    def __getattr__(self, name: str):
        return getattr(self._runtime, name)

    async def run(self, argv: list[str], env: dict[str, str]):
        if argv == ["bash", "/tests/test.sh"]:
            argv = ["sh", "-c", self._script]
        return await self._runtime.run(argv, env)


async def _is_dir(runtime: Runtime, path: str) -> bool:
    result = await runtime.run(["sh", "-c", 'test -d "$1"', "sh", path], {})
    return result.exit_code == 0


async def _collect_dereferenced(runtime: Runtime, root: str, budget: int) -> bytes:
    """The content of the directory `root` links to, archived under `root`."""
    path = f"/tmp/vf-prebuilt-artifact-{uuid.uuid4().hex}.tar"
    try:
        result = await runtime.run(["sh", "-c", 'COPYFILE_DISABLE=1 tar -cf "$1" -C "$2/" .', "sh", path, root], {})
        if result.exit_code:
            raise TaskError(
                f"collect artifact {root!r} failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        return artifacts.rebase(root, await runtime.read(path, max_bytes=max(budget, 0)))
    finally:
        await runtime.run(["rm", "-f", path], {})


class _FallbackImageWarning(logging.Filter):
    """Drop parse_task's per-task "grading in the agent's image" warning: with
    `ignore_dockerfile` the verifier image parses as None, and `load` sets it from
    `verifier_image_template` immediately after."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "names no docker_image" not in str(record.msg)


@contextmanager
def quiet_fallback_warnings() -> Iterator[None]:
    harbor_logger = logging.getLogger("verifiers.v1.tasksets.harbor.taskset")
    quiet = _FallbackImageWarning()
    harbor_logger.addFilter(quiet)
    try:
        yield
    finally:
        harbor_logger.removeFilter(quiet)


def verifier_box(data, resources, config: PrebuiltHarborConfig):
    """`data.verifier` with its own box: the agent box's workdir and resources, except at
    least `verifier_cpus` vCPUs. A task that declares no `[verifier.environment]` gets
    a fresh copy of the agent's box under Harbor; stating the same box explicitly is
    what lets its CPUs differ."""
    verifier = data.verifier
    if not verifier.fresh_copy:
        raise ValueError(f"{data.name}: declares its own [verifier.environment]")
    box = resources
    if config.verifier_cpus is not None and (box.cpu or 0) < config.verifier_cpus:
        box = box.model_copy(update={"cpu": config.verifier_cpus})
    return verifier.model_copy(update={"fresh_copy": False, "workdir": data.workdir, "resources": box})


class PrebuiltHarborTaskset(HarborTaskset, vf.Taskset[PrebuiltHarborTask, PrebuiltHarborConfig]):
    def load(self) -> Iterator[PrebuiltHarborTask]:
        config = self.config
        if not config.registry:
            raise ValueError(
                "prebuilt-harbor needs the registry its images were pushed under: set "
                "taskset.registry (--taskset.registry for vf-validate, "
                "--env.taskset.registry for vf-eval and prime-rl), e.g. prime/<team-slug>"
            )
        if config.ignore_separate_verifier:
            raise ValueError(
                "prebuilt-harbor grades in each task's verifier image; "
                "taskset.ignore_separate_verifier is not supported"
            )
        found = source.resolve(config)
        root = found.tasks_dir
        task_dirs = sorted(path.parent for path in root.glob("*/task.toml"))
        if not task_dirs:
            raise ValueError(f"no Harbor tasks found in {root}")
        overrides = Overrides.read(found.overrides)
        overrides.check({path.name for path in task_dirs}, config.include_groups)
        if config.tasks is not None:
            if unknown := set(config.tasks) - {path.name for path in task_dirs}:
                raise ValueError(f"unknown prebuilt-harbor tasks: {sorted(unknown)}")
            task_dirs = [path for path in task_dirs if path.name in config.tasks]
        held = overrides.held_back(config.include_groups)
        if skipped := [path.name for path in task_dirs if path.name in held]:
            logger.info(
                "prebuilt-harbor: holding back %d task(s) listed in %s: %s",
                len(skipped),
                found.overrides,
                ", ".join(skipped),
            )
        task_dirs = [path for path in task_dirs if path.name not in held]
        if not task_dirs:
            raise ValueError(f"every selected prebuilt-harbor task is held back by {found.overrides}")
        with quiet_fallback_warnings():
            for idx, task_dir in enumerate(task_dirs):
                data = parse_task(task_dir, idx, config)
                names = {
                    "registry": config.registry,
                    "task": task_dir.name,
                    "tag": config.tag,
                }
                if data.verifier is None:
                    raise ValueError(f"{task_dir.name}: declares no separate verifier")
                update = {
                    "image": config.image_template.format(**names),
                    "verifier_image": config.verifier_image_template.format(**names),
                }
                if data.agent_user is None and config.agent_user is not None:
                    update["agent_user"] = overrides.agent_users.get(task_dir.name, config.agent_user)
                resources = data.resources
                if config.min_disk_gb is not None and (resources.disk or 0) < config.min_disk_gb:
                    resources = resources.model_copy(update={"disk": config.min_disk_gb})
                update["resources"] = resources
                update["verifier"] = verifier_box(data, resources, config)
                if excludes := overrides.home_artifact_excludes.get(task_dir.name):
                    update["artifacts"] = [
                        artifact.model_copy(
                            update={
                                "exclude": [
                                    *artifact.exclude,
                                    *excludes.get(artifact.source, []),
                                ]
                            }
                        )
                        for artifact in data.artifacts
                    ]
                task = PrebuiltHarborTask(data, config.task).with_data(**update)
                task.gold_as_root = task_dir.name in overrides.gold_as_root
                yield task
