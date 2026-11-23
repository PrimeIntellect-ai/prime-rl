"""Record the inference engines' weight fingerprints before and right after the trainer's startup broadcast (v0)."""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from safetensors_hash import hash_safetensors_dir

POLL_S = 0.2


def get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"X-Prime-Admin-Token": os.environ["PRIME_RL_ADMIN_TOKEN"]})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def server_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def fingerprint(base_url: str) -> list[dict]:
    workers = get_json(f"{base_url}/weight_fingerprint/model", timeout=1800)["results"]
    return [
        {name: {key: tensor[key] for key in ("shape", "dtype", "sha256", "nonzero")} for name, tensor in worker.items()}
        for worker in workers
    ]


def orchestrator_loop_started(run_dir: Path) -> bool:
    for log in run_dir.glob("logs/**/orchestrator*.log"):
        if "Starting orchestrator loop" in log.read_text(errors="replace"):
            return True
    return False


def link_step0(run_dir: Path, dest: Path) -> bool:
    step0 = run_dir / "broadcasts" / "step_0"
    if not (step0 / ".finished").exists():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    for path in step0.iterdir():
        if path.is_file() and not path.name.startswith("."):
            os.link(path, dest / path.name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8100")
    parser.add_argument("--filesystem", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    broadcasts = args.run_dir / "broadcasts"
    deadline = time.monotonic() + args.timeout
    record: dict = {}

    def save() -> None:
        (args.out_dir / "fingerprints.json").write_text(json.dumps(record, indent=1, sort_keys=True))

    while not server_up(args.base_url):
        if time.monotonic() > deadline:
            raise TimeoutError("inference server never came up")
        time.sleep(1)

    if not (broadcasts / "step_0" / ".receiver_ready").exists():
        record["pre"] = fingerprint(args.base_url)
        record["pre_valid"] = not (broadcasts / "step_0" / ".receiver_ready").exists()
        save()

    step0_copy = args.out_dir / "fs_step0"
    linked = False
    while not orchestrator_loop_started(args.run_dir):
        if args.filesystem and not linked:
            linked = link_step0(args.run_dir, step0_copy)
        if time.monotonic() > deadline:
            raise TimeoutError("orchestrator never synced to v0")
        time.sleep(POLL_S)
    if args.filesystem and not linked:
        linked = link_step0(args.run_dir, step0_copy)

    record["v0"] = fingerprint(args.base_url)
    record["v0_valid"] = not (broadcasts / "step_1" / ".receiver_ready").exists()
    save()

    if args.filesystem:
        record["fs_step0_linked"] = linked
        if linked:
            (args.out_dir / "fs_step0_tensors.json").write_text(
                json.dumps(hash_safetensors_dir(step0_copy), indent=1, sort_keys=True)
            )
            for path in step0_copy.iterdir():
                path.unlink()
            step0_copy.rmdir()
        save()


if __name__ == "__main__":
    main()
