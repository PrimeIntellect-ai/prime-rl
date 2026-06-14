from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_checkpoint_pause_dir, get_step_path

REQUEST = "REQUEST"
PAUSED = "PAUSED"
RELEASE = "RELEASE"
RESUMED = "RESUMED"

PAUSE_ACK_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 0.5


def get_pause_step_dir(output_dir: Path, step: int) -> Path:
    return get_step_path(get_checkpoint_pause_dir(output_dir), step)


def write_pause_request(output_dir: Path, step: int) -> str:
    request_id = uuid.uuid4().hex
    step_dir = get_pause_step_dir(output_dir, step)
    if step_dir.exists():
        shutil.rmtree(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / REQUEST).write_text(request_id)
    get_logger().debug(f"Requested inference pause for trainer checkpoint step {step}")
    return request_id


def read_marker(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text().strip()


def write_marker(step_dir: Path, marker: str, request_id: str) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / marker).write_text(request_id)


def wait_for_marker(step_dir: Path, marker: str, request_id: str, *, timeout_s: float = PAUSE_ACK_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    marker_path = step_dir / marker
    while time.monotonic() < deadline:
        if read_marker(marker_path) == request_id:
            return
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Timed out waiting for checkpoint pause marker {marker_path}")


def write_pause_release(output_dir: Path, step: int, request_id: str) -> None:
    write_marker(get_pause_step_dir(output_dir, step), RELEASE, request_id)
    get_logger().debug(f"Released inference pause for trainer checkpoint step {step}")


def get_pending_pause_requests(output_dir: Path) -> list[tuple[int, Path, str]]:
    pause_dir = get_checkpoint_pause_dir(output_dir)
    requests: list[tuple[int, Path, str]] = []
    for step_dir in sorted(pause_dir.glob("step_*")):
        try:
            step = int(step_dir.name.split("_")[-1])
        except ValueError:
            continue
        request_id = read_marker(step_dir / REQUEST)
        if request_id is None or read_marker(step_dir / RESUMED) == request_id:
            continue
        requests.append((step, step_dir, request_id))
    return sorted(requests, key=lambda request: request[0])
