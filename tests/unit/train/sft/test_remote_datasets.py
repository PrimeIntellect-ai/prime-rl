import hashlib
import json
import shutil
import socket
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import s3fs
from datasets import Dataset

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.sft.data import load_sft_dataset, parse_dataset_name, pre_download_data

S3_SERVER_IMAGE = "versity/versitygw:v1.8.0"
ACCESS_KEY, SECRET_KEY = "testkey", "testsecret"
ROWS = [{"messages": [{"role": "assistant", "content": f"row {i}"}]} for i in range(4)]


@pytest.fixture
def hf_dataset_server():
    state = {"commit": "a" * 40, "rows": ROWS, "downloads": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond(head=False)

        def log_message(self, *_args):
            pass

        def respond(self, *, head):
            if self.path.startswith("/api/datasets/test/selected/revision/"):
                body = json.dumps(
                    {
                        "id": "test/selected",
                        "sha": state["commit"],
                        "siblings": [{"rfilename": "train.jsonl"}],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif self.path.endswith("/train.jsonl"):
                body = "".join(json.dumps(row) + "\n" for row in state["rows"]).encode()
                self.send_response(200)
                self.send_header("X-Repo-Commit", state["commit"])
                self.send_header("ETag", hashlib.sha256(body).hexdigest())
                if not head:
                    state["downloads"] += 1
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["endpoint"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hf_dataset_is_cached_on_volume_across_runs(hf_dataset_server, tmp_path, monkeypatch):
    from huggingface_hub import constants

    volume = tmp_path / "volume"
    cache = volume / ".cache" / "datasets"
    monkeypatch.setenv("HF_ENDPOINT", hf_dataset_server["endpoint"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "model-cache"))
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    config = SFTDataConfig(name="hf://datasets/test/selected", cache_dir=cache)

    first = load_sft_dataset(config)
    second = load_sft_dataset(config.model_copy(update={"name": "test/selected"}))

    assert first["messages"] == second["messages"] == [row["messages"] for row in ROWS]
    assert hf_dataset_server["downloads"] == 1
    assert list(cache.rglob("train.jsonl"))
    assert list(cache.rglob("*.arrow"))
    assert not any(p.is_symlink() for p in cache.rglob("*"))
    assert constants.HF_HUB_OFFLINE

    hf_dataset_server["commit"] = "b" * 40
    hf_dataset_server["rows"] = ROWS[:1]
    updated = load_sft_dataset(config)

    assert updated["messages"] == [ROWS[0]["messages"]]
    assert hf_dataset_server["downloads"] == 2

    launcher_config = config.model_copy()
    pre_download_data(launcher_config, {})
    assert Path(launcher_config.name).is_relative_to(cache)
    assert load_sft_dataset(launcher_config)["messages"] == updated["messages"]
    assert hf_dataset_server["downloads"] == 2


def test_parse_dataset_name():
    assert parse_dataset_name("hf://datasets/PrimeIntellect/Reverse-Text-SFT") == (
        None,
        "PrimeIntellect/Reverse-Text-SFT",
    )
    assert parse_dataset_name("PrimeIntellect/Reverse-Text-SFT") == (None, "PrimeIntellect/Reverse-Text-SFT")
    assert parse_dataset_name("/data/sft") == (None, "/data/sft")
    assert parse_dataset_name("s3://bucket/prefix") == ("s3", "s3://bucket/prefix")
    assert parse_dataset_name("gs://bucket/file.parquet") == ("gs", "gs://bucket/file.parquet")


@pytest.mark.parametrize("name", ["r2://bucket/key", "https://example.com/data.parquet", "hf://models/foo/bar"])
def test_parse_dataset_name_rejects_unsupported_scheme(name):
    with pytest.raises(ValueError, match="hf://datasets/<owner>/<repo>, s3://"):
        parse_dataset_name(name)


@pytest.fixture(scope="module")
def s3_server():
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("docker is not available")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    name = f"prime-rl-test-s3-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        # Versity S3 gateway serving a tmpfs directory; every top-level directory is a bucket.
        ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:7070", "--tmpfs", "/gw"]
        + [
            "-e",
            f"ROOT_ACCESS_KEY={ACCESS_KEY}",
            "-e",
            f"ROOT_SECRET_KEY={SECRET_KEY}",
            S3_SERVER_IMAGE,
            "posix",
            "/gw",
        ],
        check=True,
        capture_output=True,
    )
    endpoint = f"http://127.0.0.1:{port}"
    try:
        fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, endpoint_url=endpoint, skip_instance_cache=True)
        for _ in range(60):
            try:
                fs.mkdir("sft")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("S3 server did not become ready")
        for i, chunk in enumerate([ROWS[:2], ROWS[2:]]):
            with fs.open(f"sft/parquet/part-{i}.parquet", "wb") as f:
                Dataset.from_list(chunk).to_parquet(f)
            with fs.open(f"sft/jsonl/part-{i}.jsonl", "w") as f:
                f.write("".join(json.dumps(row) + "\n" for row in chunk))
        fs.pipe("sft/jsonl/_SUCCESS", b"")
        fs.pipe("sft/mixed/a.parquet", fs.cat("sft/parquet/part-0.parquet"))
        fs.pipe("sft/mixed/b.txt", b"hello")
        yield endpoint
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def s3_env(s3_server, monkeypatch):
    # Only the standard env vars: nothing S3-specific is passed through code.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_KEY)
    monkeypatch.setenv("AWS_ENDPOINT_URL", s3_server)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("s3://sft/parquet/part-0.parquet", ROWS[:2]),
        ("s3://sft/parquet", ROWS),
        ("s3://sft/jsonl/part-1.jsonl", ROWS[2:]),
        ("s3://sft/jsonl/", ROWS),
    ],
)
def test_load_sft_dataset_from_s3(s3_env, name, expected):
    dataset = load_sft_dataset(SFTDataConfig(name=name))
    assert dataset["messages"] == [row["messages"] for row in expected]
    assert set(dataset["__split"]) == {"train"}


def test_load_sft_dataset_from_s3_rejects_mixed_prefix(s3_env):
    with pytest.raises(ValueError, match=r"must all be one of .*got: \['.parquet', '.txt'\]"):
        load_sft_dataset(SFTDataConfig(name="s3://sft/mixed"))


def test_load_sft_dataset_from_s3_rejects_subsets(s3_env):
    with pytest.raises(ValueError, match="single 'train' split"):
        load_sft_dataset(SFTDataConfig(name="s3://sft/parquet", subsets=["default"]))
