"""Where the task dirs come from: a local directory, or a task archive in a Hugging Face
dataset repo, downloaded and extracted once per machine.

An archive holds the bundle layout at its root: `tasks/<name>/` (each with a
`task.toml`) and, optionally, `overrides.json` (see `prebuilt_harbor.overrides`).

The cache keeps the downloaded archive (`hub/`, Hugging Face's own layout), each
extracted archive under its sha256 (`trees/<sha256>/`), and which archive a repo,
revision and filename resolved to (`refs/`). Every process that loads the taskset
(eval, orchestrator, vf-validate) downloads under a cross-process lock per repo,
revision and filename, and extracts under one per archive, so concurrent starts
download and extract it once; a tree appears only complete, by rename, and is reused
by every later load. Env servers rebuild tasks from the data they are sent and never
load the archive.
"""

import contextlib
import fcntl
import hashlib
import logging
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from prebuilt_harbor.overrides import OVERRIDES_FILE

logger = logging.getLogger("verifiers.v1.tasksets.prebuilt_harbor")

TASKS_DIR_ENV = "PREBUILT_HARBOR_TASKS_DIR"
CACHE_DIR_ENV = "PREBUILT_HARBOR_CACHE_DIR"
TASKS_SUBDIR = "tasks"
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class TaskSource:
    tasks_dir: Path
    """Directory whose children are the task dirs."""
    overrides: Path | None
    """Overrides file, if any."""


def resolve(config) -> TaskSource:
    """The task dirs and overrides file `config` selects (a `PrebuiltHarborConfig`)."""
    if config.hf_repo is not None:
        root = fetch(config)
        tasks_dir = root / TASKS_SUBDIR
        if not tasks_dir.is_dir():
            raise ValueError(
                f"{config.hf_filename!r} from {config.hf_repo!r} has no {TASKS_SUBDIR}/ directory at its root"
            )
    else:
        local = config.tasks_dir or os.environ.get(TASKS_DIR_ENV)
        if not local:
            raise ValueError(
                "prebuilt-harbor needs its tasks: set taskset.tasks_dir (or "
                f"${TASKS_DIR_ENV}) to a local directory of task dirs, or taskset.hf_repo "
                "to a Hugging Face dataset repo holding a task archive"
            )
        tasks_dir = Path(local).expanduser().resolve()
        if not tasks_dir.is_dir():
            raise ValueError(f"prebuilt-harbor task directory {tasks_dir} does not exist")
    if config.overrides is not None:
        overrides = Path(config.overrides).expanduser().resolve()
        if not overrides.is_file():
            raise ValueError(f"prebuilt-harbor overrides file {overrides} does not exist")
    else:
        beside = tasks_dir.parent / OVERRIDES_FILE
        overrides = beside if beside.is_file() else None
    if overrides is None:
        logger.warning(
            "prebuilt-harbor: no overrides file (taskset.overrides, or %s beside %s), so no task is held back",
            OVERRIDES_FILE,
            tasks_dir,
        )
    logger.info("prebuilt-harbor: tasks from %s, overrides from %s", tasks_dir, overrides)
    return TaskSource(tasks_dir, overrides)


def cache_root(configured: Path | None) -> Path:
    """`configured`, else $PREBUILT_HARBOR_CACHE_DIR, else a writable
    `$XDG_CACHE_HOME/prebuilt-harbor` (default `~/.cache`), else a per-user directory
    under the system temp dir (pod homes may be read-only)."""
    if configured is not None:
        return Path(configured).expanduser()
    if env := os.environ.get(CACHE_DIR_ENV):
        return Path(env).expanduser()
    with contextlib.suppress(RuntimeError, KeyError):  # no $HOME and no passwd entry
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        if _writable(base / "prebuilt-harbor"):
            return base / "prebuilt-harbor"
    fallback = Path(tempfile.gettempdir()) / f"prebuilt-harbor-{os.getuid()}"
    fallback.mkdir(mode=0o700, exist_ok=True)
    if fallback.stat().st_uid != os.getuid():
        raise RuntimeError(f"{fallback} belongs to another user; set ${CACHE_DIR_ENV} or taskset.cache_dir")
    return fallback


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            return True
    except OSError:
        return False


def fetch(config) -> Path:
    """Download `config.hf_filename` from the dataset repo `config.hf_repo` at
    `config.hf_revision`, check it against `config.hf_sha256` when given, and return
    the directory it is extracted to."""
    revision = config.hf_revision or "main"
    pinned = COMMIT_SHA.fullmatch(revision) is not None
    if not pinned:
        logger.warning(
            "prebuilt-harbor: hf_revision %r is not a commit sha; pin one so every process "
            "and every restart loads the same tasks",
            revision,
        )
    expected = config.hf_sha256.lower() if config.hf_sha256 else None
    cache = cache_root(config.cache_dir)
    key = hashlib.sha256("\0".join((config.hf_repo, revision, config.hf_filename)).encode()).hexdigest()[:32]
    ref = cache / "refs" / key
    # A commit's files never change: once resolved, reuse its tree without the network
    # or a token.
    if pinned and (tree := _cached_tree(cache, ref, expected)):
        return tree
    with _locked(cache / "locks" / f"{key}.lock"):
        if pinned and (tree := _cached_tree(cache, ref, expected)):
            return tree
        token = _token(config)
        _point_xet_cache(cache)
        archive = Path(_download(config, revision, token, cache / "hub"))
        digest = _sha256(archive)
        if expected is not None and digest != expected:
            raise ValueError(
                f"{config.hf_filename!r} from {config.hf_repo!r} at {revision!r} has sha256 "
                f"{digest}, but taskset.hf_sha256 is {expected}"
            )
        tree = cache / "trees" / digest
        # Refs that resolve to the same archive share one tree and one extraction.
        with _locked(cache / "locks" / f"tree-{digest}.lock"):
            if not tree.is_dir():
                logger.info("prebuilt-harbor: extracting %s into %s", archive, tree)
                _extract(archive, tree)
        _write_atomic(ref, digest)
    return tree


def _token(config) -> str | bool:
    """The token in `$hf_token_env`, or False (never a token saved on the machine) when
    `hf_token_env` is None."""
    if config.hf_token_env is None:
        return False
    token = os.environ.get(config.hf_token_env, "")
    if not token:
        raise ValueError(
            f"${config.hf_token_env} is not set: prebuilt-harbor needs a Hugging Face token "
            f"that can read the dataset repo {config.hf_repo!r} (taskset.hf_token_env names "
            "the variable; set it to null for a public repo)"
        )
    return token


def _cached_tree(cache: Path, ref: Path, expected: str | None) -> Path | None:
    try:
        digest = ref.read_text().strip()
    except FileNotFoundError:
        return None
    if expected is not None and digest != expected:
        return None
    tree = cache / "trees" / digest
    return tree if tree.is_dir() else None


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _point_xet_cache(cache: Path) -> None:
    """Unless $HF_XET_CACHE is set, keep hf_xet's cache and logs (which record each
    download's repo and path) in `cache` rather than under $HF_HOME, which may be shared
    with other users or read-only. hf_xet reads it when it first starts in a process."""
    if "HF_XET_CACHE" not in os.environ:
        os.environ["HF_XET_CACHE"] = str(cache / "xet")


def _download(config, revision: str, token: str | bool, cache_dir: Path) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    where = f"{config.hf_filename!r} in the dataset repo {config.hf_repo!r} at revision {revision!r}"
    access = (
        f"check that ${config.hf_token_env} holds a token that can read it"
        if config.hf_token_env is not None
        else "it may be private: set taskset.hf_token_env"
    )
    try:
        return hf_hub_download(
            repo_id=config.hf_repo,
            filename=config.hf_filename,
            repo_type="dataset",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    except GatedRepoError as exc:
        raise PermissionError(f"no access to {where}; {access}") from exc
    except RepositoryNotFoundError as exc:
        raise ValueError(f"cannot download {where}: the repo does not exist or is not visible; {access}") from exc
    except RevisionNotFoundError as exc:
        raise ValueError(f"cannot download {where}: no such revision") from exc
    except LocalEntryNotFoundError as exc:  # before its base class EntryNotFoundError
        # The Hub lookup failed (no network, Hub error, or $HF_HUB_OFFLINE) and nothing is cached.
        raise ConnectionError(
            f"cannot download {where}, and {cache_dir} holds no copy: {exc.__cause__ or exc}"
        ) from exc
    except EntryNotFoundError as exc:
        raise ValueError(f"cannot download {where}: no such file at that revision") from exc
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            raise PermissionError(f"no access to {where} (HTTP {status}); {access}") from exc
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(archive: Path, tree: Path) -> None:
    """Extract into a scratch dir beside `tree`, then rename it into place. The caller
    holds the tree's lock, so a scratch dir already there was left by a killed process."""
    tree.parent.mkdir(parents=True, exist_ok=True)
    for stale in tree.parent.glob(f".{tree.name}.*"):
        shutil.rmtree(stale, ignore_errors=True)
    scratch = Path(tempfile.mkdtemp(dir=tree.parent, prefix=f".{tree.name}."))
    try:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(scratch, filter="data")
        scratch.rename(tree)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.{os.getpid()}")
    scratch.write_text(text)
    os.replace(scratch, path)
