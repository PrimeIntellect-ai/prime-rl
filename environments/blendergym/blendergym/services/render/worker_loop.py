"""Persistent Blender worker loop — runs inside Blender's embedded Python.

Launched as: blender --background --factory-startup --python worker_loop.py

Communicates with PersistentBlender via Unix domain socket using a 4-byte
big-endian length prefix + UTF-8 JSON protocol. Each request is a short
connection: accept -> recv -> process -> send -> close.

Reuses _enable_gpu_cycles and _render_camera1 from pipeline_render_script.py
to avoid duplicating Cycles configuration logic.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import socket
import struct
import sys
import time
import traceback

import bpy

from blendergym.assets.pipeline_render_script import (
    _enable_gpu_cycles,
    _render_camera1,
)

SOCKET_PATH = os.environ["BLENDERGYM_WORKER_SOCKET"]
_LENGTH_FMT = "!I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FMT)


def _cleanup_socket():
    try:
        os.unlink(SOCKET_PATH)
    except OSError:
        pass


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


def _exec_user_code_str(code: str, code_path: str) -> None:
    """Write code to disk then exec. code_path appears in tracebacks."""
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    exec(compile(code, code_path, "exec"), {"__name__": "__main__", "bpy": bpy})


def _handle_request(msg: dict) -> dict:
    """Process a single render request. Returns response dict."""
    blend_file = msg["blend_file"]
    code = msg["code"]
    output_dir = msg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    code_path = os.path.join(output_dir, "code.py")
    render_path = os.path.join(output_dir, "render1.png")

    # Remove stale render artifact
    if os.path.exists(render_path):
        os.unlink(render_path)

    # Capture stdout/stderr for observability
    old_stdout, old_stderr = sys.stdout, sys.stderr
    cap_stdout, cap_stderr = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = cap_stdout, cap_stderr

    t0 = time.monotonic()
    try:
        # Blend file caching: revert for same file, open for different
        current = getattr(_handle_request, "_current_blend", None)
        if current == blend_file:
            bpy.ops.wm.revert_mainfile(use_scripts=False)
        else:
            bpy.ops.wm.open_mainfile(filepath=blend_file)
            _handle_request._current_blend = blend_file

        _enable_gpu_cycles()
        _exec_user_code_str(code, code_path)
        _render_camera1(output_dir)

        duration = time.monotonic() - t0
        sys.stdout, sys.stderr = old_stdout, old_stderr

        image_paths = [render_path] if os.path.isfile(render_path) else []
        return {
            "status": "ok",
            "image_paths": image_paths,
            "code_path": code_path,
            "duration_s": round(duration, 3),
            "stdout": cap_stdout.getvalue(),
            "stderr": cap_stderr.getvalue(),
        }
    except Exception:
        duration = time.monotonic() - t0
        sys.stdout, sys.stderr = old_stdout, old_stderr
        tb = traceback.format_exc()
        return {
            "status": "error",
            "stderr": cap_stderr.getvalue() + "\n" + tb,
            "stdout": cap_stdout.getvalue(),
            "duration_s": round(duration, 3),
        }


def main() -> None:
    # Clean up any residual socket file
    _cleanup_socket()
    atexit.register(_cleanup_socket)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    srv.listen(1)

    # Initial Cycles configuration (before any blend file is loaded)
    # This triggers OPTIX kernel JIT on first call
    _enable_gpu_cycles()

    while True:
        conn, _ = srv.accept()
        try:
            msg = _recv_json(conn)
            resp = _handle_request(msg)
            _send_json(conn, resp)
        except Exception as e:
            try:
                _send_json(conn, {
                    "status": "error",
                    "stderr": f"worker_loop protocol error: {e}",
                    "stdout": "",
                    "duration_s": 0,
                })
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
