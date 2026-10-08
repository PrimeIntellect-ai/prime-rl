"""CPU-only saved-delta replay. No training, sandbox, or model API calls."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import hashlib
import json
import multiprocessing as mp
import os
import resource
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import msgpack
import msgspec
import numpy as np
import psutil
import zmq
import zmq.asyncio
from verifiers.v1.episode import WireEpisode
from verifiers.v1.graph import MessageNode
from verifiers.v1.serve.delta import EpisodeAssembly, TraceSummary, unpack
from verifiers.v1.serve.encoding import msgpack_encoder


def containers(value):
    if isinstance(value, msgspec.Struct):
        return {
            field.encode_name: (
                getattr(value, field.name) if field.type is Any else containers(getattr(value, field.name))
            )
            for field in msgspec.structs.fields(type(value))
            if getattr(value, field.name) is not msgspec.UNSET
        }
    if isinstance(value, list):
        return [containers(item) for item in value]
    if isinstance(value, dict):
        return {key: containers(item) for key, item in value.items()}
    return value


def decoder(binary_type):
    array = msgspec.defstruct(
        "ArrayBuffer",
        [
            (name, binary_type if name == "data" else Any, msgspec.UNSET)
            for name in ("__nd__", "dtype", "shape", "data")
        ],
        forbid_unknown_fields=True,
    )
    mask = msgspec.defstruct("MaskBuffer", [("ids", array), ("counts", array)], forbid_unknown_fields=True)
    node = msgspec.defstruct(
        "BufferNode",
        [
            (
                name,
                array | None if name == "routed_experts" else mask | None if name == "sampling_mask" else Any,
                msgspec.UNSET,
            )
            for name, field in MessageNode.model_fields.items()
            if not field.exclude
        ],
        forbid_unknown_fields=True,
    )
    delta = msgspec.defstruct(
        "BufferDelta",
        [
            (name, Any, msgspec.UNSET)
            for name in (
                "trace",
                "discard",
                "open",
                "calls",
                "errors",
                "extra_usage",
                "request_rewrites",
                "response_rewrites",
                "set",
                "pending",
                "links",
                "dispatch",
            )
        ]
        + [("nodes", list[node], msgspec.UNSET), ("routing_repairs", dict[Any, array], msgspec.UNSET)],
        forbid_unknown_fields=True,
    )
    return msgspec.msgpack.Decoder(delta)


def owned_binary_fields(delta):
    # Preserve the public callback's bytes fields, without traversing token lists.
    arrays = list((delta.get("routing_repairs") or {}).values())
    for node in delta.get("nodes", []):
        if node.get("routed_experts") is not None:
            arrays.append(node["routed_experts"])
        if node.get("sampling_mask") is not None:
            arrays.extend(node["sampling_mask"].values())
    for array in arrays:
        if isinstance(array.get("data"), memoryview):
            array["data"] = bytes(array["data"])
    return delta


def observe(delta):
    return {
        "trace": delta["trace"],
        "keys": sorted(delta),
        "nodes": len(delta.get("nodes", [])),
        "calls": len(delta.get("calls", [])),
        "repairs": len(delta.get("routing_repairs", {})),
        "binary_fields_are_bytes": all(
            isinstance(node["routed_experts"]["data"], bytes)
            for node in delta.get("nodes", [])
            if node.get("routed_experts") is not None
        ),
    }


def raw_episode(assembly):
    first = next(iter(assembly.traces.values()))
    head = {"id": "offline-contract-episode", "task": first["task"], "ok": True}
    summaries = [
        TraceSummary(id=key, nodes=len(trace["nodes"]), calls=len(trace["calls"]))
        for key, trace in assembly.traces.items()
    ]
    return assembly.finish(head, summaries)


def summary(assembly):
    # This is an experimental envelope, not a replacement for Prime RL metrics.
    return {
        "traces": len(assembly.traces),
        "nodes": sum(len(trace["nodes"]) for trace in assembly.traces.values()),
        "calls": sum(len(trace["calls"]) for trace in assembly.traces.values()),
        "tokens": sum(len(node.get("token_ids", [])) for trace in assembly.traces.values() for node in trace["nodes"]),
    }


def fingerprint(episode):
    assert type(episode) is WireEpisode
    value = episode.model_dump(mode="python", exclude_computed_fields=True)
    return hashlib.sha256(msgpack.packb(value, use_bin_type=True, default=msgpack_encoder)).hexdigest()


def verify_native(episode, expected, allow_views):
    assert type(episode) is WireEpisode and isinstance(episode.traces, list)
    for trace in episode.traces:
        assert isinstance(trace.nodes, list) and isinstance(trace.calls, list)
        for node in trace.nodes:
            for name in ("token_ids", "mask", "is_content", "logprobs", "semantic_parents"):
                assert isinstance(getattr(node, name), list)
            if node.routed_experts is not None:
                base = node.routed_experts
                assert isinstance(base, np.ndarray)
                while isinstance(base, np.ndarray) and base.base is not None:
                    base = base.base
                assert isinstance(base, bytes) or (allow_views and isinstance(base, memoryview) and base.readonly)
    assert fingerprint(episode) == expected


STATE = {}
ORDINALS = {}
INPUT = None
OUTPUTS = {}


def initialize(name):
    global INPUT
    INPUT = SharedMemory(name=name)


def apply_packet(request, kind, length, ordinal, total):
    start = time.process_time()
    assert ordinal == ORDINALS.get(request, 0)
    assembly = STATE.setdefault(request, EpisodeAssembly())
    if kind == b"delta":
        view = INPUT.buf[:length]
        try:
            assembly.apply(unpack(view))
        finally:
            view.release()
        ORDINALS[request] = ordinal + 1
    else:
        assert kind == b"reply" and ordinal == total
        raw_episode(assembly)
    return time.process_time() - start


def describe(request):
    return summary(STATE[request])


def export(request):
    started = time.perf_counter()
    wire = msgpack.packb(raw_episode(STATE[request]), use_bin_type=True)
    output = SharedMemory(create=True, size=len(wire))
    output.buf[:] = wire
    OUTPUTS[request] = output
    return output.name, len(wire), time.perf_counter() - started


def release_output(request):
    output = OUTPUTS.pop(request)
    output.close()
    output.unlink()


def release_graphs(requests):
    for request in requests:
        STATE.pop(request)
        ORDINALS.pop(request)
    return len(STATE)


def cleanup():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


class Lane:
    def __init__(self, span):
        self.slot = SharedMemory(create=True, size=span)
        self.pool = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"), initializer=initialize, initargs=(self.slot.name,)
        )

    async def apply(self, *args):
        request, kind, data, ordinal, total = args
        self.slot.buf[: len(data)] = data
        return await asyncio.wrap_future(self.pool.submit(apply_packet, request, kind, len(data), ordinal, total))

    async def call(self, function, *args):
        return await asyncio.wrap_future(self.pool.submit(function, *args))

    def close(self):
        self.pool.shutdown(wait=True)
        self.slot.close()
        self.slot.unlink()


@dataclass
class PayloadRef:
    lane: Lane
    request: int
    released: bool = False

    async def materialize(self, pool):
        if self.released:
            raise RuntimeError("Payload already released")
        name, length, encode_seconds = await self.lane.call(export, self.request)

        def reconstruct():
            allocation = SharedMemory(name=name)
            view = allocation.buf[:length]
            try:
                record = unpack(view)
            finally:
                view.release()
                allocation.close()
            return WireEpisode.model_validate(record)

        try:
            episode = await asyncio.get_running_loop().run_in_executor(pool, reconstruct)
        finally:
            await self.lane.call(release_output, self.request)
        return episode, length, encode_seconds

    async def release(self):
        if not self.released:
            await self.lane.call(release_graphs, [self.request])
            self.released = True


@dataclass
class CompletedRollout:
    statistics: dict
    payload: PayloadRef


async def main(args):
    assert msgpack.__version__ == "1.1.2", "Use the same decoder version as the saved benchmark"
    meta = json.loads((args.fixture / "fixture.json").read_text())
    wires = [(args.fixture / "frames" / frame["file"]).read_bytes() for frame in meta["frames"]]
    assert all(hashlib.sha256(wire).hexdigest() == frame["sha256"] for wire, frame in zip(wires, meta["frames"]))
    reference = EpisodeAssembly()
    expected_observations = []
    for wire in wires:
        value = unpack(wire)
        reference.apply(value)
        expected_observations.append(observe(value))
    native = WireEpisode.model_validate(raw_episode(reference))
    expected = fingerprint(native)
    expected_summary = summary(reference)
    typed = (
        None if args.mode in ("thread", "deferred") else decoder(bytes if args.mode == "typed-bytes" else memoryview)
    )
    if typed is not None:
        # Exact decoded frame bytes, ordering, callback mutation and final graph.
        proposed = EpisodeAssembly()
        for wire in wires:
            baseline = unpack(wire)
            value = owned_binary_fields(containers(typed.decode(wire)))
            assert value == baseline
            proposed.apply(value)
            info = (value.get("set") or {}).get("info")
            if isinstance(info, dict):
                marker = object()
                info["offline_callback_control"] = marker
                assert proposed.traces[value["trace"]]["info"]["offline_callback_control"] is marker
                del info["offline_callback_control"]
        verify_native(WireEpisode.model_validate(raw_episode(proposed)), expected, False)
        del proposed, baseline, value
    if args.mode == "deferred":
        assert not args.callbacks, "Owner-final path does not preserve parent callbacks"
    else:
        assert args.consumption == 1.0, "Native-return controls must return every episode"
    del native, reference, wire
    cleanup()
    args.output.mkdir(parents=True, exist_ok=False)
    pool = ThreadPoolExecutor(max_workers=1)
    lanes = [Lane(max(map(len, wires))) for _ in range(args.queues)] if args.mode == "deferred" else []
    rows = []
    buffered = []
    proc = psutil.Process()

    async def measured(phase, function, cycle):
        lag = []

        async def heartbeat():
            while True:
                start = time.perf_counter()
                await asyncio.sleep(0.005)
                lag.append(max(0, time.perf_counter() - start - 0.005))

        task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        started, cpu = time.perf_counter(), time.process_time()
        value = await function()
        elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        row = {
            "phase": phase,
            "cycle": cycle,
            "seconds": elapsed,
            "parent_cpu_seconds": cpu,
            "max_loop_lag_seconds": max(lag, default=0),
            "p95_loop_lag_seconds": float(np.percentile(lag, 95)) if lag else 0,
            "rss_bytes": proc.memory_info().rss,
            "pss_bytes": proc.memory_full_info().pss,
            "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        if isinstance(value, dict):
            row.update(value)
        rows.append(row)
        print(json.dumps(row), flush=True)
        return value

    async def receive(cycle):
        context = zmq.asyncio.Context()
        rx = context.socket(zmq.PAIR)
        rx.setsockopt(zmq.RCVHWM, 8)
        endpoint = f"ipc:///tmp/hubert-deferred-replay-{os.getpid()}.sock"
        rx.bind(endpoint)
        sender = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).with_name("sender.py")), endpoint, str(args.replicas), str(args.fixture)
        )
        assemblies = [EpisodeAssembly() for _ in range(args.replicas)] if not lanes else []
        observations = [[] for _ in range(args.replicas)]
        queues = [asyncio.Queue() for _ in range(args.queues)]
        condition = asyncio.Condition()
        budget_bytes = budget_items = peak_bytes = peak_items = 0
        ordinal = [0] * args.replicas
        worker_cpu = 0.0

        async def worker(index):
            nonlocal budget_bytes, budget_items, worker_cpu
            while True:
                packet = await queues[index].get()
                if packet is None:
                    return
                request, kind, data, order = packet
                if lanes:
                    worker_cpu += await lanes[index].apply(
                        cycle * args.replicas + request, kind, data, order, len(wires)
                    )
                elif kind == b"delta":

                    def decode():
                        value = unpack(data) if typed is None else containers(typed.decode(data))
                        return owned_binary_fields(value) if args.callbacks else value

                    value = await asyncio.get_running_loop().run_in_executor(pool, decode)
                    assemblies[request].apply(value)
                    if args.callbacks:
                        observations[request].append(observe(value))
                else:
                    assert order == len(wires)
                    raw_episode(assemblies[request])
                async with condition:
                    budget_bytes -= len(data)
                    budget_items -= 1
                    condition.notify_all()

        try:
            async with asyncio.TaskGroup() as tasks:
                for index in range(args.queues):
                    tasks.create_task(worker(index))
                for _ in range((len(wires) + 1) * args.replicas):
                    request, kind, data = await rx.recv_multipart()
                    request = int(request)
                    assert 0 <= request < args.replicas and kind in (b"delta", b"reply")
                    order = ordinal[request]
                    if kind == b"delta":
                        ordinal[request] += 1
                    else:
                        assert order == len(wires)
                    async with condition:
                        await condition.wait_for(lambda: budget_bytes + len(data) <= 2 * 1024**3 and budget_items < 64)
                        budget_bytes += len(data)
                        budget_items += 1
                        peak_bytes, peak_items = max(peak_bytes, budget_bytes), max(peak_items, budget_items)
                    await queues[request % args.queues].put((request, kind, data, order))
                for queue in queues:
                    await queue.put(None)
            assert budget_bytes == budget_items == 0
            await rx.send(b"consumed")
            assert await sender.wait() == 0
        finally:
            if sender.returncode is None:
                sender.terminate()
            await sender.wait()
            rx.close(0)
            context.term()
            Path(endpoint.removeprefix("ipc://")).unlink(missing_ok=True)
        started = time.perf_counter()
        outputs = []
        payload_bytes, encode_seconds, summary_seconds = 0, 0.0, 0.0
        envelopes = []
        for request in range(args.replicas):
            if lanes:
                lane = lanes[request % args.queues]
                identifier = cycle * args.replicas + request
                before = time.perf_counter()
                statistics = await lane.call(describe, identifier)
                summary_seconds += time.perf_counter() - before
                assert statistics == expected_summary
                envelope = CompletedRollout(statistics, PayloadRef(lane, identifier))
                envelopes.append(envelope)
                if request < round(args.replicas * args.consumption):
                    episode, length, encode = await envelope.payload.materialize(pool)
                    payload_bytes += length
                    encode_seconds += encode
                    outputs.append(episode)
            else:
                if args.callbacks:
                    assert observations[request] == expected_observations
                record = raw_episode(assemblies[request])
                outputs.append(
                    await asyncio.get_running_loop().run_in_executor(pool, WireEpisode.model_validate, record)
                )
                del record
        # Include owner and temporary graph destruction in the measured receive path.
        if lanes:
            for envelope in envelopes:
                await envelope.payload.release()
        else:
            await asyncio.get_running_loop().run_in_executor(pool, assemblies.clear)
        return {
            "outputs": outputs,
            "materialize_seconds": time.perf_counter() - started,
            "complete_native_episodes": len(outputs),
            "scalar_envelopes": len(envelopes),
            "summary_handoff_seconds": summary_seconds,
            "worker_encode_seconds": encode_seconds,
            "exported_payload_bytes": payload_bytes,
            "worker_cpu_seconds": worker_cpu,
            "peak_accounted_input_bytes": peak_bytes,
            "peak_accounted_input_items": peak_items,
        }

    async def release(episodes):
        def clear():
            episodes.clear()
            cleanup()

        await asyncio.get_running_loop().run_in_executor(pool, clear)

    try:
        for cycle in range(args.cycles):
            held = []

            async def receive_into():
                result = await receive(cycle)
                held.extend(result.pop("outputs"))
                return result

            await measured("receive", receive_into, cycle)

            async def verify():
                def check():
                    for episode in held:
                        verify_native(episode, expected, args.mode == "typed-views" and not args.callbacks)

                await asyncio.get_running_loop().run_in_executor(pool, check)

            await measured("verify", verify, cycle)
            if buffered:
                await measured("release_old_buffer", lambda: release(buffered), cycle)
            keep = len(held) // 4
            buffered.extend(held[:keep])
            del held[:keep]
            await measured("publication_release", lambda: release(held), cycle)
        await measured("final_release", lambda: release(buffered), args.cycles)
        for lane in lanes:
            assert await lane.call(release_graphs, []) == 0
    finally:
        pool.shutdown(wait=True)
        for lane in lanes:
            lane.close()
    native_root = Path(sys.modules["verifiers.v1.graph"].__file__).parent
    result = {
        "mode": args.mode,
        "callbacks": args.callbacks,
        "consumption": args.consumption,
        "replicas": args.replicas,
        "cycles": args.cycles,
        "queues": args.queues,
        "rows": rows,
        "native_episode_fingerprint": expected,
        "fixture_wire_bytes": meta["wire_bytes"],
        "fixture_frames": len(wires),
        "fixture_sha256": meta["fixture_sha256"],
        "node": socket.gethostname(),
        "cpu_affinity": proc.cpu_affinity(),
        "versions": {
            "python": sys.version,
            "msgpack": msgpack.__version__,
            "msgspec": msgspec.__version__,
            "numpy": np.__version__,
            "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        },
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path(__file__).with_name("sender.py")]
            + [
                native_root / name
                for name in (
                    "episode.py",
                    "trace.py",
                    "graph.py",
                    "serve/delta.py",
                    "serve/client.py",
                    "configs/retries.py",
                )
            ]
        },
        "validation": {
            "all_selected_episode_fingerprints_match": True,
            "selected_native_field_types_preserved": True,
            "all_deferred_summaries_match": True,
            "all_worker_graphs_released": True,
        },
        "limits": "Same saved delta frames, independent repetitions, reconstructed minimal head. "
        "No live RPC, production callbacks or training sample shipping. Timed callbacks use bounded "
        "native observer signatures. Deferred changes return/admission contract; summary is experimental. "
        "Automatic GC enabled; forced GC/malloc_trim on release match the historical control. "
        "Typed binary views pin immutable input frames. Typed Struct maps change dictionary insertion order. "
        "Schema checked against archived native source.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["thread", "typed-bytes", "typed-views", "deferred"], required=True)
    parser.add_argument("--callbacks", action="store_true")
    parser.add_argument("--consumption", type=float, choices=[0.0, 0.25, 1.0], default=1.0)
    parser.add_argument("--queues", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=2)
    asyncio.run(main(parser.parse_args()))
