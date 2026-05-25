"""PersistentBlender + BlenderPool: long-lived Blender process management.

PersistentBlender manages a single Blender subprocess communicating via
Unix domain socket. BlenderPool manages 1-N PersistentBlender instances
per GPU with asyncio.Queue-based worker dispatch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import time
from pathlib import Path
from subprocess import Popen
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_LENGTH_FMT = "!I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FMT)

_WORKER_LOOP_PATH = Path(__file__).parent / "worker_loop.py"


class RenderRequest(BaseModel):
    """Inbound render request schema."""

    blend_file: str
    code: str
    output_dir: str
    timeout: int = 600


class RenderResponse(BaseModel):
    """Outbound render response, aligned with RenderResult fields."""

    success: bool = False
    image_paths: list[str] = []
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    gpu_id: int = -1
    duration_s: float = 0
    code_path: str = ""


def _send_json(sock: socket.socket, data: dict) -> None:
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(_LENGTH_FMT, len(payload)) + payload)


def _recv_json(sock: socket.socket) -> dict:
    header = b""
    while len(header) < _LENGTH_SIZE:
        chunk = sock.recv(_LENGTH_SIZE - len(header))
        if not chunk:
            raise ConnectionError("peer closed connection during header")
        header += chunk
    (length,) = struct.unpack(_LENGTH_FMT, header)
    buf = b""
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection during payload")
        buf += chunk
    return json.loads(buf.decode("utf-8"))


class PersistentBlender:
    """Single long-lived Blender process communicating via Unix socket."""

    def __init__(
        self,
        gpu_id: int,
        worker_idx: int,
        blender_bin: Path,
    ):
        self._gpu_id = gpu_id
        self._worker_idx = worker_idx
        self._blender_bin = blender_bin
        self._socket_path = f"/tmp/blendergym_blender_{gpu_id}_{worker_idx}.sock"
        self._blender_user = Path(f"/tmp/blendergym_user_{gpu_id}_{worker_idx}")
        self._blender_user.mkdir(parents=True, exist_ok=True)
        self._consecutive_failures = 0
        self._process = self._spawn()

    def _spawn(self) -> Popen:
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self._gpu_id),
            "BLENDERGYM_WORKER_SOCKET": self._socket_path,
            "BLENDER_USER_RESOURCES": str(self._blender_user),
            "PYTHONUNBUFFERED": "1",
        }
        return Popen(
            [
                str(self._blender_bin),
                "--background",
                "--factory-startup",
                "--python",
                str(_WORKER_LOOP_PATH),
            ],
            env=env,
        )

    async def render(self, req: RenderRequest) -> RenderResponse:
        msg: dict[str, Any] = {
            "blend_file": req.blend_file,
            "code": req.code,
            "output_dir": req.output_dir,
        }
        try:
            resp = await asyncio.to_thread(self._send_recv, msg, req.timeout)
            self._consecutive_failures = 0
            return RenderResponse(
                success=(resp.get("status") == "ok"),
                image_paths=resp.get("image_paths", []),
                stdout=resp.get("stdout", ""),
                stderr=resp.get("stderr", ""),
                duration_s=resp.get("duration_s", 0),
                timed_out=False,
                gpu_id=self._gpu_id,
                code_path=resp.get("code_path", ""),
            )
        except socket.timeout:
            self._consecutive_failures += 1
            self._restart()
            return RenderResponse(
                success=False,
                timed_out=True,
                gpu_id=self._gpu_id,
                stderr=f"Unix socket timeout after {req.timeout}s",
            )
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._restart()
            return RenderResponse(
                success=False,
                gpu_id=self._gpu_id,
                stderr=str(e),
            )

    def _send_recv(self, msg: dict, timeout: int) -> dict:
        sock = socket.socket(socket.AF_UNIX)
        sock.settimeout(timeout)
        sock.connect(self._socket_path)
        try:
            _send_json(sock, msg)
            return _recv_json(sock)
        finally:
            sock.close()

    def _restart(self) -> None:
        logger.warning(
            "Restarting Blender worker gpu=%d idx=%d (failures=%d)",
            self._gpu_id,
            self._worker_idx,
            self._consecutive_failures,
        )
        try:
            self._process.kill()
            self._process.wait(timeout=10)
        except Exception:
            pass
        self._consecutive_failures = 0
        self._process = self._spawn()

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    @property
    def socket_path(self) -> str:
        return self._socket_path


class BlenderPool:
    """Per-GPU pool of PersistentBlender workers with queue-based dispatch."""

    def __init__(
        self,
        gpu_id: int,
        blender_bin: Path,
        pool_size: int = 1,
    ):
        self._gpu_id = gpu_id
        self._workers: list[PersistentBlender] = []
        self._queue: asyncio.Queue[PersistentBlender] = asyncio.Queue()
        for i in range(pool_size):
            w = PersistentBlender(gpu_id, worker_idx=i, blender_bin=blender_bin)
            self._workers.append(w)
            self._queue.put_nowait(w)
        self.pool_size = pool_size

    async def render(self, req: RenderRequest) -> RenderResponse:
        worker = await self._queue.get()
        try:
            return await worker.render(req)
        finally:
            self._queue.put_nowait(worker)

    @property
    def alive(self) -> bool:
        return all(w.alive for w in self._workers)

    async def wait_ready(self, timeout: float = 120) -> None:
        """Wait until all workers' Unix sockets are connectable."""
        deadline = time.monotonic() + timeout
        for w in self._workers:
            while time.monotonic() < deadline:
                if os.path.exists(w.socket_path):
                    try:
                        sock = socket.socket(socket.AF_UNIX)
                        sock.settimeout(2)
                        sock.connect(w.socket_path)
                        sock.close()
                        break
                    except (ConnectionRefusedError, FileNotFoundError, OSError):
                        pass
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError(
                    f"Worker gpu={w._gpu_id} idx={w._worker_idx} "
                    f"not ready after {timeout}s"
                )

    def shutdown(self) -> None:
        """Gracefully terminate all worker subprocesses."""
        for w in self._workers:
            try:
                w._process.terminate()
                w._process.wait(timeout=10)
            except Exception:
                try:
                    w._process.kill()
                except Exception:
                    pass
