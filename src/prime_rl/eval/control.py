"""Run control: steer and resize a running eval from another process.

The eval runner serves a small request/reply endpoint (zmq, JSON frames) on a local
TCP port and writes its address to ``<run>/control.json`` (with the host and pid that
serve it), so any process on the host finds it. Requests:

- ``status``: the in-flight episodes (task key, env, and each streamed trace with its
  seat, phase and turns) and the run's concurrency.
- ``steer``: a user message into running agents, picked by ``task_key`` or
  ``trace_id``, optionally narrowed to one ``seat``. The runner
  calls the env server's ``steer`` route for each picked trace (verifiers delivers
  natively or through the interception and records the steer on the trace) and
  returns the receipts.
- ``resize``: move the run's in-flight concurrency (``ConcurrencyController.resize``).

There is no cancel: a task is stopped by the taskset that handed it out
(``Taskset.cancel``), which owns what a cancel means (flow: ``flow cancel``).

A seat is the env agent's name
(``Trace.agent.name``) as streamed in the trace's deltas; a seat steer goes to that
seat's latest open trace in each picked episode, like ``vf.steer`` by seat.

``Client`` is the Python client; ``uv run eval control --run <dir> ...`` its CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.eval.runner import EvalRunner
    from prime_rl.orchestrator.types import InflightEpisode

CONTROL_FILE = "control.json"
OPEN_STAGES = ("pending", "boot", "setup", "running")
"""Trace phases in which the agent can still take a message."""


DEFAULT_TIMEOUT_S = 10.0
"""How long a client waits for a live run's answer (a steer waits on its env servers)."""


class ControlError(RuntimeError):
    """The run refused a request, or could not be reached."""


def control_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / CONTROL_FILE


def read_address(run_dir: Path | str) -> str:
    """The control address of the eval running in ``run_dir``. Raises ``ControlError``
    when there is none, or when ``control.json`` names a process on another host (the
    endpoint binds loopback) or one that no longer runs."""
    path = control_path(run_dir)
    try:
        info = json.loads(path.read_text())
    except FileNotFoundError:
        raise ControlError(f"no run control at {path} (is the eval running?)") from None
    except (OSError, ValueError) as error:
        raise ControlError(f"unreadable run control file {path}: {error}") from None
    host, pid, address = info.get("host"), info.get("pid"), info.get("address")
    if host != socket.gethostname():
        raise ControlError(f"the eval of {path} runs on {host}, not on this host")
    if not isinstance(pid, int) or not pid_alive(pid):
        raise ControlError(f"stale run control at {path}: its eval (pid {pid}) is gone")
    if not isinstance(address, str):
        raise ControlError(f"no control address in {path}")
    return address


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ControlServer:
    """Serves the run control of one ``EvalRunner`` while it runs."""

    def __init__(self, runner: EvalRunner) -> None:
        self.runner = runner
        self.ctx: zmq.asyncio.Context | None = None
        self.socket: zmq.asyncio.Socket | None = None
        self.address: str | None = None
        self.loop_task: asyncio.Task | None = None
        self.handlers: set[asyncio.Task] = set()

    async def start(self) -> None:
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.address = f"tcp://127.0.0.1:{port}"
        self.loop_task = asyncio.create_task(self.serve(), name="run-control")
        path = control_path(self.runner.run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"address": self.address, "host": socket.gethostname(), "pid": os.getpid()}))
        get_logger().info(f"Run control at {self.address} ({path})")

    async def stop(self) -> None:
        if self.loop_task is not None:
            self.loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.loop_task
            self.loop_task = None
        for task in list(self.handlers):
            task.cancel()
        await asyncio.gather(*self.handlers, return_exceptions=True)
        path = control_path(self.runner.run_dir)
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
            if json.loads(path.read_text()).get("address") == self.address:
                path.unlink()
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.ctx is not None:
            self.ctx.term()
            self.ctx = None

    async def serve(self) -> None:
        assert self.socket is not None
        while True:
            identity, empty, payload = await self.socket.recv_multipart()
            task = asyncio.create_task(self.answer(identity, empty, payload))
            self.handlers.add(task)
            task.add_done_callback(self.handlers.discard)

    async def answer(self, identity: bytes, empty: bytes, payload: bytes) -> None:
        try:
            reply = await self.handle(json.loads(payload))
        except Exception as error:
            reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if self.socket is not None:
            await self.socket.send_multipart([identity, empty, json.dumps(reply).encode()])

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method")
        if method == "status":
            return {"ok": True, **self.status()}
        if method == "steer":
            return {"ok": True, "receipts": await self.steer(**request.get("args", {}))}
        if method == "resize":
            return {"ok": True, "max_inflight": self.resize(**request.get("args", {}))}
        return {"ok": False, "error": f"unknown request {method!r}"}

    def inflight(self) -> list[InflightEpisode]:
        return [meta for meta in self.runner.dispatcher.inflight.values() if meta.kind == "eval"]

    def status(self) -> dict[str, Any]:
        dispatcher = self.runner.dispatcher
        concurrency = getattr(self.runner, "concurrency", None)
        return {
            "episodes": [
                {
                    "task_key": meta.task.key,
                    "env": meta.env_name,
                    "group": str(meta.group_id),
                    "seats": sorted({trace.seat for trace in meta.live.values() if trace.seat}),
                    "traces": [
                        {"trace_id": trace_id, "seat": trace.seat, "stage": trace.stage, "turns": trace.turns}
                        for trace_id, trace in meta.live.items()
                    ],
                }
                for meta in self.inflight()
            ],
            "inflight": dispatcher.current_inflight,
            "max_inflight": int(dispatcher.max_inflight),
            "min_inflight": concurrency.floor if concurrency is not None else None,
        }

    async def steer(
        self,
        text: str,
        task_key: str | None = None,
        trace_id: str | None = None,
        seat: str | None = None,
        sender: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("steering requires a nonempty message")
        if (task_key is None) == (trace_id is None):
            raise ValueError("steer one of task_key or trace_id")
        episodes = [
            meta
            for meta in self.inflight()
            if (task_key is not None and meta.task.key == task_key) or (trace_id is not None and trace_id in meta.live)
        ]
        target = task_key or trace_id
        if not episodes:
            return [{"queued": False, "reason": f"no running episode for {target}", "trace_id": trace_id}]
        picked: list[tuple[InflightEpisode, str, str | None]] = []
        for meta in episodes:
            if trace_id is not None:
                picked.append((meta, trace_id, meta.live[trace_id].seat))
                continue
            open_traces = [
                (tid, trace)
                for tid, trace in meta.live.items()
                if trace.stage in OPEN_STAGES and (seat is None or trace.seat == seat)
            ]
            if seat is not None:
                open_traces = open_traces[-1:]  # the seat's latest open rollout
            picked += [(meta, tid, trace.seat) for tid, trace in open_traces]
        if not picked:
            who = f"seat {seat!r}" if seat is not None else "agent"
            return [{"queued": False, "reason": f"no running {who} in {target}", "trace_id": None}]
        message_id = uuid.uuid4().hex

        async def one(meta: InflightEpisode, tid: str, trace_seat: str | None) -> dict[str, Any]:
            client = self.runner.eval_envs.get(meta.env_name).env_client
            try:
                response = await client.steer(text, trace_id=tid, sender=sender, message_id=message_id)
                receipt = {"queued": response.queued, "reason": response.reason or response.error}
                receipt["delivery"] = response.delivery
            except Exception as error:
                receipt = {"queued": False, "reason": f"{type(error).__name__}: {error}", "delivery": None}
            return {
                "task_key": meta.task.key,
                "env": meta.env_name,
                "seat": trace_seat,
                "trace_id": tid,
                "message_id": message_id,
                **receipt,
            }

        receipts = await asyncio.gather(*(one(*entry) for entry in picked))
        get_logger().info(f"Steered {target}: {sum(r['queued'] for r in receipts)}/{len(receipts)} queued")
        return list(receipts)

    def resize(self, max_inflight: int) -> int:
        self.runner.concurrency.resize(int(max_inflight))
        return int(self.runner.dispatcher.max_inflight)


class Client:
    """The run control of a running eval: ``Client(run_dir).status()``,
    ``.steer(text, task_key=...)``, ``.resize(n)``. A ``control.json`` left behind by an
    eval that died without its teardown fails here, not after ``timeout``."""

    def __init__(self, run_dir: Path | str, *, timeout: float = DEFAULT_TIMEOUT_S):
        self.address = read_address(run_dir)
        self.timeout = timeout

    def request(self, method: str, **args: Any) -> dict[str, Any]:
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        socket.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        try:
            socket.connect(self.address)
            socket.send(json.dumps({"method": method, "args": args}).encode())
            reply = json.loads(socket.recv())
        except zmq.Again as error:
            raise ControlError(f"run control at {self.address} did not answer in {self.timeout}s") from error
        finally:
            socket.close()
        if not reply.get("ok"):
            raise ControlError(reply.get("error") or "request failed")
        return reply

    def status(self) -> dict[str, Any]:
        reply = self.request("status")
        reply.pop("ok")
        return reply

    def steer(
        self,
        text: str,
        *,
        task_key: str | None = None,
        trace_id: str | None = None,
        seat: str | None = None,
        sender: str | None = None,
    ) -> list[dict[str, Any]]:
        """One receipt per steered trace: ``queued``, ``reason``, ``delivery``,
        ``trace_id``, ``seat``, ``task_key``."""
        args = {"task_key": task_key, "trace_id": trace_id, "seat": seat, "sender": sender}
        return self.request("steer", text=text, **{k: v for k, v in args.items() if v is not None})["receipts"]

    def resize(self, max_inflight: int) -> int:
        return self.request("resize", max_inflight=max_inflight)["max_inflight"]


def format_receipt(receipt: dict[str, Any]) -> str:
    where = " ".join(
        f"{name}={receipt[key]}"
        for key, name in (("task_key", "task"), ("seat", "seat"), ("trace_id", "trace"))
        if receipt.get(key)
    )
    if receipt["queued"]:
        return f"queued   {where} ({receipt.get('delivery')})"
    return f"refused  {where}: {receipt.get('reason')}"


def main(argv: list[str] | None = None) -> None:
    """``uv run eval control --run <dir> status|steer|resize``."""
    parser = argparse.ArgumentParser(prog="eval control", description="Control a running eval.")
    parser.add_argument("--run", required=True, type=Path, help="the eval's run directory (holds control.json)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="in-flight episodes and concurrency")
    status.add_argument("--json", action="store_true")
    steer = commands.add_parser("steer", help="send a user message to running agents")
    target = steer.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", dest="task_key")
    target.add_argument("--trace", dest="trace_id")
    steer.add_argument("--seat")
    steer.add_argument("--sender", default="operator")
    steer.add_argument("text")
    resize = commands.add_parser("resize", help="set the in-flight concurrency")
    resize.add_argument("max_inflight", type=int)
    args = parser.parse_args(argv)

    try:
        client = Client(args.run, timeout=args.timeout)
        if args.command == "status":
            state = client.status()
            if args.json:
                print(json.dumps(state, indent=2))
                return
            print(f"inflight {state['inflight']} / max {state['max_inflight']}")
            for episode in state["episodes"]:
                print(f"{episode['task_key']}  env={episode['env']}")
                for trace in episode["traces"]:
                    print(
                        f"    {trace['seat'] or '?':<16} {trace['trace_id']}  {trace['stage']}  turns={trace['turns']}"
                    )
        elif args.command == "steer":
            receipts = client.steer(
                args.text,
                task_key=args.task_key,
                trace_id=args.trace_id,
                seat=args.seat,
                sender=args.sender,
            )
            for receipt in receipts:
                print(format_receipt(receipt))
            if not any(receipt["queued"] for receipt in receipts):
                sys.exit(1)
        elif args.command == "resize":
            print(f"max_inflight {client.resize(args.max_inflight)}")
    except ControlError as error:
        sys.exit(f"eval control: {error}")
