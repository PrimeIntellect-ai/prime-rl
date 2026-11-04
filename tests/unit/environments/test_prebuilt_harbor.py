import hashlib
import io
import json
import os
import tarfile
import threading
from pathlib import Path

import huggingface_hub
import pytest
from prebuilt_harbor import source
from prebuilt_harbor.overrides import Overrides
from prebuilt_harbor.taskset import PrebuiltHarborConfig, PrebuiltHarborTaskset

REGISTRY = "registry.invalid/team"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TASK_TOML = """\
schema_version = "1.1"
artifacts = ["/app", "/srv/state"]

[agent]
timeout_sec = 600.0

[verifier]
environment_mode = "separate"
timeout_sec = 600.0

[environment]
cpus = 1
memory_mb = 2048
storage_mb = 1024
"""
OVERRIDES = {
    "version": 1,
    "holdback_groups": {
        "never": {"optional": False, "tasks": {"task-c": "cannot run here"}},
        "extra": {"optional": True, "description": "opt-in", "tasks": {"task-d": "needs a daemon"}},
    },
    "agent_users": {"task-b": "builder"},
    "gold_as_root": {"task-a": "solve.sh needs root"},
    "home_artifact_excludes": {"task-a": {"/srv/state": ["srv/state/scratch"]}},
}
NAMES = ["task-a", "task-b", "task-c", "task-d"]


def write_tasks(root: Path, names=NAMES) -> Path:
    for name in names:
        task = root / name
        for rel, text in {
            "task.toml": TASK_TOML,
            "instruction.md": f"Do {name}.\n",
            "environment/Dockerfile": "FROM scratch\n",
            "tests/Dockerfile": "FROM scratch\nCOPY . /tests\n",
        }.items():
            (task / rel).parent.mkdir(parents=True, exist_ok=True)
            (task / rel).write_text(text)
    return root


@pytest.fixture
def bundle(tmp_path) -> Path:
    root = tmp_path / "bundle"
    write_tasks(root / "tasks")
    (root / "overrides.json").write_text(json.dumps(OVERRIDES))
    return root


@pytest.fixture
def archive(bundle, tmp_path) -> Path:
    path = tmp_path / "tasks.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(bundle / "overrides.json", arcname="overrides.json")
        tar.add(bundle / "tasks", arcname="tasks")
    return path


@pytest.fixture
def downloads(monkeypatch, archive, tmp_path) -> list[dict]:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "xet"))  # loads would set it for the whole session
    return calls


def load(**fields) -> list:
    config = PrebuiltHarborConfig(registry=REGISTRY, **fields)
    return list(PrebuiltHarborTaskset(config).load())


def hf_fields(tmp_path, **extra) -> dict:
    return {"hf_repo": "owner/tasks", "hf_revision": COMMIT, "cache_dir": tmp_path / "cache", **extra}


def test_overrides_and_image_templating(bundle):
    tasks = {task.data.name: task for task in load(tasks_dir=bundle / "tasks", tag="v7")}
    assert sorted(tasks) == ["task-a", "task-b"]
    a, b = tasks["task-a"], tasks["task-b"]
    assert a.data.image == f"{REGISTRY}/task-a:v7"
    assert a.data.verifier_image == f"{REGISTRY}/task-a-verifier:v7"
    assert (a.data.agent_user, b.data.agent_user) == ("agent", "builder")
    assert (a.gold_as_root, b.gold_as_root) == (True, False)
    assert {x.source: x.exclude for x in a.data.artifacts} == {"/app": [], "/srv/state": ["srv/state/scratch"]}
    assert {x.source: x.exclude for x in b.data.artifacts} == {"/app": [], "/srv/state": []}
    assert a.data.resources.disk == 16.0 and a.data.resources.cpu == 1
    assert a.data.verifier.resources.cpu == 2.0 and not a.data.verifier.fresh_copy
    assert [task.data.idx for task in tasks.values()] == [0, 1]


def test_holdback_groups(bundle):
    assert [t.data.name for t in load(tasks_dir=bundle / "tasks", include_groups=["extra"])] == [
        "task-a",
        "task-b",
        "task-d",
    ]
    assert [t.data.name for t in load(tasks_dir=bundle / "tasks", tasks=["task-a", "task-d"])] == ["task-a"]
    with pytest.raises(ValueError, match="not optional"):
        load(tasks_dir=bundle / "tasks", include_groups=["never"])
    with pytest.raises(ValueError, match="unknown hold-back groups"):
        load(tasks_dir=bundle / "tasks", include_groups=["typo"])
    assert Overrides.model_validate(OVERRIDES).held_back(["extra"]) == {"task-c": "cannot run here"}


def test_unknown_tasks_raise(bundle, tmp_path):
    with pytest.raises(ValueError, match="unknown prebuilt-harbor tasks"):
        load(tasks_dir=bundle / "tasks", tasks=["task-a", "task-x"])
    overrides = tmp_path / "bad.json"
    overrides.write_text(json.dumps({"agent_users": {"task-x": "agent"}}))
    with pytest.raises(ValueError, match="no task dir"):
        load(tasks_dir=bundle / "tasks", overrides=overrides)


def test_hf_download_and_cache_reuse(bundle, downloads, tmp_path):
    local = load(tasks_dir=bundle / "tasks")
    remote = load(**hf_fields(tmp_path))
    assert [t.data.model_dump(exclude={"task_dir"}) for t in remote] == [
        t.data.model_dump(exclude={"task_dir"}) for t in local
    ]
    assert [t.key for t in remote] == [t.key for t in local]
    assert len(downloads) == 1
    call = downloads[0]
    assert (call["repo_id"], call["repo_type"], call["revision"], call["token"]) == (
        "owner/tasks",
        "dataset",
        COMMIT,
        "hf_test",
    )
    assert Path(call["cache_dir"]).is_relative_to(tmp_path / "cache")
    assert Path(remote[0].data.task_dir).is_relative_to(tmp_path / "cache" / "trees")
    load(**hf_fields(tmp_path))
    assert len(downloads) == 1


def test_hf_checksum_mismatch(downloads, tmp_path):
    with pytest.raises(ValueError, match="has sha256"):
        load(**hf_fields(tmp_path, hf_sha256="0" * 64))


def test_hf_missing_token(downloads, monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError, match=r"\$HF_TOKEN is not set"):
        load(**hf_fields(tmp_path))
    assert downloads == []
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    load(**hf_fields(tmp_path))
    monkeypatch.delenv("HF_TOKEN")
    assert len(load(**hf_fields(tmp_path))) == 2  # a pinned, extracted archive needs no token
    assert len(downloads) == 1


def test_hf_concurrent_loads_extract_once(downloads, archive, tmp_path):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    stale = tmp_path / "cache" / "trees" / f".{digest}.killed"  # left by a killed extraction
    stale.mkdir(parents=True)
    results, errors = [], []

    def worker():
        try:
            results.append(len(load(**hf_fields(tmp_path))))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and results == [2] * 6
    assert len(downloads) == 1
    assert list((tmp_path / "cache" / "trees").iterdir()) == [tmp_path / "cache" / "trees" / digest]


def test_cache_falls_back_when_home_is_read_only(monkeypatch, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can write a read-only home")
    home = tmp_path / "home"
    home.mkdir(mode=0o500)
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv(source.CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(source.tempfile, "tempdir", str(temp))
    assert source.cache_root(None) == temp / f"prebuilt-harbor-{os.getuid()}"
    monkeypatch.setenv(source.CACHE_DIR_ENV, str(tmp_path / "chosen"))
    assert source.cache_root(None) == tmp_path / "chosen"
    monkeypatch.setenv("HF_XET_CACHE", "unset-at-teardown")
    monkeypatch.delenv("HF_XET_CACHE")
    source._point_xet_cache(tmp_path / "chosen")
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "chosen" / "xet")
    source._point_xet_cache(tmp_path / "other")
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "chosen" / "xet")


def test_archive_extraction_rejects_escaping_members(downloads, archive, tmp_path):
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        load(**hf_fields(tmp_path))
    assert not (tmp_path / "cache" / "trees" / "escape").exists()
    assert list((tmp_path / "cache" / "trees").iterdir()) == []
