"""CPU-only experiment; no changes to runtime interfaces or live training."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import msgpack
import msgspec
import numpy as np
import verifiers.v1 as vf
from verifiers.v1.episode import WireEpisode
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import SamplingMask

from prime_rl.orchestrator.metrics import MetricWindow, RolloutWindow, TrainEpisodes
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.utils.logger import setup_logger

MODES = ("native", "typed-bytes", "typed-views", "deferred-views")


def typed_decoder(binary_type: type) -> msgspec.msgpack.Decoder:
    """Use current native field names; keep non-binary fields as native Python types."""
    nd = msgspec.defstruct(
        "ArrayBuffer",
        [("marker", bool, msgspec.field(name="__nd__")), ("dtype", str), ("shape", list[int]), ("data", binary_type)],
        forbid_unknown_fields=True,
    )
    masks = msgspec.defstruct("MaskBuffer", [("ids", nd), ("counts", nd)], forbid_unknown_fields=True)
    node = msgspec.defstruct(
        "BufferNode",
        [
            (
                name,
                nd | None if name == "routed_experts" else masks | None if name == "sampling_mask" else Any,
                msgspec.UNSET,
            )
            for name, info in MessageNode.model_fields.items()
            if not info.exclude
        ],
        forbid_unknown_fields=True,
    )
    trace = msgspec.defstruct(
        "BufferTrace",
        [(name, list[node] if name == "nodes" else Any, msgspec.UNSET) for name in serialized_fields(vf.Trace)],
        forbid_unknown_fields=True,
    )
    episode = msgspec.defstruct(
        "BufferEpisode",
        [(name, list[trace] if name == "traces" else Any, msgspec.UNSET) for name in serialized_fields(WireEpisode)],
        forbid_unknown_fields=True,
    )
    return msgspec.msgpack.Decoder(episode)


def serialized_fields(model: type) -> list[str]:
    return [name for name, info in model.model_fields.items() if not info.exclude] + list(model.model_computed_fields)


def native_containers(value: Any) -> Any:
    # asdict/to_builtins can copy buffers; preserve the views until native validation.
    if isinstance(value, msgspec.Struct):
        return {
            field.encode_name: (
                getattr(value, field.name) if field.type is Any else native_containers(getattr(value, field.name))
            )
            for field in msgspec.structs.fields(type(value))
            if getattr(value, field.name) is not msgspec.UNSET
        }
    if isinstance(value, list):
        return [native_containers(item) for item in value]
    if isinstance(value, dict):
        return {key: native_containers(item) for key, item in value.items()}
    return value


def decode_episode(wire: bytes | memoryview, decoder: msgspec.msgpack.Decoder | None) -> WireEpisode:
    raw = (
        msgpack.unpackb(wire, raw=False, strict_map_key=False)
        if decoder is None
        else native_containers(decoder.decode(wire))
    )
    return WireEpisode.model_validate(raw)


def capture_summary(episode: WireEpisode, accepted: bool) -> dict:
    """Exactly the numeric observation path from PR #3806, for a finalized group."""
    episodes = TrainEpisodes()
    episodes.extend([episode], admitted=accepted)
    window = RolloutWindow()
    queued = {trace.id for trace in episode.traces} if accepted else set()
    window.observe(episodes, [], None, env_name=episode.env.name, queued_trace_ids=queued)
    return msgspec.msgpack.decode(msgspec.msgpack.encode(asdict(window.metrics)))


class Envelope(msgspec.Struct, forbid_unknown_fields=True):
    episode_id: str
    summary: dict[str, Any]
    payload: memoryview


@dataclass
class PayloadRef:
    buffer: memoryview | None
    decoder: msgspec.msgpack.Decoder

    def materialize(self) -> WireEpisode:
        if self.buffer is None:
            raise RuntimeError("Payload has been released")
        return decode_episode(self.buffer, self.decoder)

    def release(self) -> None:
        # NumPy views owned by a materialized Episode retain their backing frame.
        self.buffer = None


@dataclass
class CompletedRollout:
    episode_id: str
    summary: dict
    payload: PayloadRef


def samples(episode: WireEpisode) -> list:
    return [sample for trace in episode.traces for sample in trace_to_samples(trace, env_name=episode.env.name)]


def checksum(episode: WireEpisode) -> str:
    return hashlib.sha256(msgpack.packb(episode.model_dump(mode="python"), use_bin_type=True)).hexdigest()


def fixture(index: int, tokens: int, image_bytes: int) -> WireEpisode:
    """Independent image-bearing branched graphs, with routing and sampling masks."""
    nodes = []
    calls = []
    span = max(1, tokens // 32)
    image = "data:image/png;base64," + "A" * image_bytes
    for i in range(32):
        ids = list(range(1000 + i * span, 1000 + (i + 1) * span))
        if i == 0:
            ids[0] = 900000
            message = vf.UserMessage(
                content=[
                    vf.TextContentPart(text="Synthetic benchmark 東京"),
                    vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image)),
                ]
            )
        elif i % 2:
            message = vf.AssistantMessage(content=f"Synthetic answer {i}")
        else:
            message = vf.ToolMessage(content="synthetic tool output " * 16, tool_call_id=f"call-{i}", name="browser")
        sampled = i % 2 == 1
        parent = None if i == 0 else 15 if i in (29, 31) else i - 1
        node = MessageNode(
            parent=parent,
            message=message,
            sampled=sampled,
            timestamp=float(i),
            token_ids=ids,
            mask=[sampled] * span,
            is_content=[True] * span,
            logprobs=[-0.2] * span if sampled else [],
            advantages=[float(index % 2 * 2 - 1)] * span if sampled else None,
            routed_experts=np.full((span, 32, 8), i, dtype=np.uint8),
            sampling_mask=SamplingMask(
                ids=np.tile(np.array([1, 2], dtype=np.int32), span), counts=np.full(span, 2, dtype=np.int32)
            )
            if sampled
            else None,
        )
        nodes.append(node)
        if sampled:
            calls.append(ModelCall(node=i, finish_reason="stop"))
    task = vf.TraceTask(type="SyntheticPayloadTask", data=vf.TaskData(idx=index, prompt=None))
    trace = vf.Trace(
        id=f"trace-{index}",
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=calls,
        mm_token_type_id_map={900000: 1},
        rewards={"solved": vf.Reward(score=float(index % 2))},
        metrics={"custom_control": float(index)},
        ok=True,
        is_completed=True,
        timing=vf.Timing(start=0),
    )
    return WireEpisode.model_validate(
        vf.Episode(
            id=f"episode-{index}",
            task=task,
            traces=[trace],
            ok=True,
            env=vf.EnvInfo(id="synthetic", name="synthetic"),
            group=vf.GroupInfo(id=f"group-{index}"),
            run=vf.TrainRunInfo(id="benchmark", work=vf.TrainWorkInfo(step=1)),
        ).model_dump(mode="python")
    )


def memory() -> dict:
    out = {"rss_bytes": None, "pss_bytes": None}
    path = Path("/proc/self/smaps_rollup")
    if path.exists():
        for line in path.read_text().splitlines():
            key, *value = line.split()
            if key in ("Rss:", "Pss:"):
                out["rss_bytes" if key == "Rss:" else "pss_bytes"] = int(value[0]) * 1024
    return out


async def run_case(args: argparse.Namespace) -> dict:
    decoder = None if args.mode == "native" else typed_decoder(bytes if args.mode == "typed-bytes" else memoryview)
    envelope_decoder = msgspec.msgpack.Decoder(Envelope)
    selected = {i for i in range(args.episodes) if i < round(args.episodes * args.acceptance)}
    inputs = []
    expected = []
    producer_cpu = 0.0
    producer_wall = 0.0
    input_bytes = 0
    for i in range(args.episodes):
        episode = (
            WireEpisode.model_validate(msgpack.unpackb(args.episode.read_bytes(), raw=False, strict_map_key=False))
            if args.episode
            else fixture(i, args.tokens, args.image_bytes)
        )
        wire = msgpack.packb(episode.model_dump(mode="python"), use_bin_type=True)
        summary = capture_summary(episode, i in selected)
        expected.append(
            (checksum(episode), summary, hashlib.sha256(msgspec.msgpack.encode(samples(episode))).hexdigest())
        )
        if args.mode == "deferred-views":
            wall, cpu = time.perf_counter(), time.process_time()
            summary = capture_summary(episode, i in selected)
            wire = msgpack.packb({"episode_id": episode.id, "summary": summary, "payload": wire}, use_bin_type=True)
            producer_wall += time.perf_counter() - wall
            producer_cpu += time.process_time() - cpu
        inputs.append(wire)
        input_bytes += len(wire)
        del episode, wire
    gc.collect()
    memories = {"before_receive": memory()}
    observations = []
    resident = []
    lags = []
    running = True

    async def heartbeat():
        while running:
            start = time.perf_counter()
            await asyncio.sleep(0.002)
            lags.append(max(0.0, time.perf_counter() - start - 0.002))

    async def measure(function, *values):
        wall, cpu = time.perf_counter(), time.process_time()
        value = await asyncio.to_thread(function, *values)
        return value, time.perf_counter() - wall, time.process_time() - cpu

    timer = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.01)
    phase = {}
    materialized = 0
    for i in range(args.episodes):

        def receive(wire, admitted):
            if args.mode == "deferred-views":
                value = envelope_decoder.decode(wire)
                return CompletedRollout(value.episode_id, value.summary, PayloadRef(value.payload, decoder))
            episode = decode_episode(wire, decoder)
            return episode, capture_summary(episode, admitted)

        result, wall, cpu = await measure(receive, inputs[i], i in selected)
        phase.setdefault("receive_wall_seconds", 0.0)
        phase.setdefault("receive_cpu_seconds", 0.0)
        phase["receive_wall_seconds"] += wall
        phase["receive_cpu_seconds"] += cpu
        inputs[i] = None
        if isinstance(result, CompletedRollout):
            observations.append(result.summary)
            resident.append(result)
        else:
            episode, summary = result
            observations.append(summary)
            resident.append(episode if i in selected else None)
            materialized += 1
            del episode
        del result
    del inputs
    memories["after_receive"] = memory()
    # The delayed admission phase is an explicit experimental assumption, not a new sink API.
    for i, value in enumerate(resident):
        if isinstance(value, CompletedRollout):
            if i in selected:
                episode, wall, cpu = await measure(value.payload.materialize)
                phase["materialize_wall_seconds"] = phase.get("materialize_wall_seconds", 0.0) + wall
                phase["materialize_cpu_seconds"] = phase.get("materialize_cpu_seconds", 0.0) + cpu
                materialized += 1
            value.payload.release()
            resident[i] = episode if i in selected else None
            if i in selected:
                del episode
    value = None
    memories["after_materialize"] = memory()
    outputs = []
    for i in sorted(selected):
        payload, wall, cpu = await measure(lambda episode: msgspec.msgpack.encode(samples(episode)), resident[i])
        phase["training_payload_wall_seconds"] = phase.get("training_payload_wall_seconds", 0.0) + wall
        phase["training_payload_cpu_seconds"] = phase.get("training_payload_cpu_seconds", 0.0) + cpu
        outputs.append((i, payload))
    memories["after_training_payload"] = memory()
    await asyncio.sleep(0.01)
    work_lags = list(lags)
    # Full comparisons happen outside the reported receive/materialize/sample timings.
    verify = time.perf_counter()
    aggregate, reference = MetricWindow(), MetricWindow()
    for i, observation in enumerate(observations):
        assert observation == expected[i][1], "Numeric observations changed"
        aggregate.extend(MetricWindow(**observation))
        reference.extend(MetricWindow(**expected[i][1]))
    assert aggregate.to_dict() == reference.to_dict(), "Metric keys or values changed"
    for i, payload in outputs:
        assert hashlib.sha256(payload).hexdigest() == expected[i][2], "Native TrainingSample bytes changed"
        assert checksum(resident[i]) == expected[i][0], "Native episode fields changed"
    verify_seconds = time.perf_counter() - verify
    payload_bytes = sum(len(payload) for _, payload in outputs)
    resident.clear()
    outputs.clear()
    payload = None
    gc.collect()
    await asyncio.sleep(0.01)
    running = False
    await timer
    memories["after_release"] = memory()
    return {
        "mode": args.mode,
        "acceptance": args.acceptance,
        "episodes": args.episodes,
        "selected": len(selected),
        "materialized": materialized,
        "input_bytes": input_bytes,
        "training_payload_bytes": payload_bytes,
        "phases": phase,
        "receiver_total_wall_seconds": sum(v for k, v in phase.items() if "wall" in k),
        "receiver_total_cpu_seconds": sum(v for k, v in phase.items() if "cpu" in k),
        "producer_summary_envelope_wall_seconds": producer_wall,
        "producer_summary_envelope_cpu_seconds": producer_cpu,
        "verification_wall_seconds": verify_seconds,
        "work_event_loop_lag_max_seconds": max(work_lags, default=0),
        "work_event_loop_lag_p95_seconds": float(np.percentile(work_lags, 95)) if work_lags else 0,
        "whole_run_lag_including_verification_seconds": max(lags, default=0),
        "memory_boundaries": memories,
        "process_peak_rss_including_setup_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1024 if sys.platform != "darwin" else 1),
        "metric_key_count": len(aggregate.to_dict()),
        "validation": "exact accepted native episodes and sample bytes; all numeric observations and aggregate keys/values",
        "fixture": "external complete WireEpisode"
        if args.episode
        else "synthetic independent branched image-bearing graphs",
        "versions": {
            "python": platform.python_version(),
            "msgpack": msgpack.__version__,
            "msgspec": msgspec.__version__,
            "numpy": np.__version__,
        },
    }


def safety_checks():
    decoder = typed_decoder(memoryview)
    wire = msgpack.packb(fixture(0, 64, 64).model_dump(mode="python"), use_bin_type=True)
    raw = native_containers(decoder.decode(wire))
    view = raw["traces"][0]["nodes"][0]["routed_experts"]["data"]
    assert isinstance(view, memoryview) and view.readonly
    reference = decode_episode(wire, None)
    episode = decode_episode(wire, decoder)
    assert checksum(episode) == checksum(reference)
    ref = PayloadRef(memoryview(wire), decoder)
    materialized = ref.materialize()
    ref.release()
    assert checksum(materialized) == checksum(reference), "Release invalidated native array owners"
    try:
        ref.materialize()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Released handle remained usable")
    mutable = bytearray(wire)
    raw = native_containers(decoder.decode(mutable))
    view = raw["traces"][0]["nodes"][0]["routed_experts"]["data"]
    view[0] = 255
    assert decode_episode(mutable, None).traces[0].nodes[0].routed_experts.flat[0] == 255
    for malformed in (b"\xc1", wire[:-1]):
        try:
            decode_episode(malformed, decoder)
        except (msgspec.DecodeError, msgspec.ValidationError):
            pass
        else:
            raise AssertionError("Malformed payload accepted")
    return {
        "native_fields_preserved": True,
        "read_only_input_view": True,
        "materialized_episode_survives_handle_release": True,
        "use_after_release_rejected": True,
        "mutable_input_corruption_demonstrated": True,
        "malformed_input_rejected": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--image-bytes", type=int, default=1048576)
    parser.add_argument(
        "--episode", type=Path, help="Private complete Python-mode MessagePack WireEpisode, including tensors"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--acceptance", type=float, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.episodes < 1 or args.tokens < 32 or args.image_bytes < 0 or args.repeats < 1:
        parser.error("episodes/repeats must be positive; tokens >= 32; image-bytes >= 0")
    setup_logger("warning")
    if args.mode:
        result = asyncio.run(run_case(args))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    checks = safety_checks()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repeat in range(args.repeats):
        # Rotate mode order to reduce fixed warm-cache/order bias; never overlap cases.
        modes = MODES[repeat % len(MODES) :] + MODES[: repeat % len(MODES)]
        for acceptance in (0.0, 0.25, 1.0):
            for mode in modes:
                path = args.output / f"{mode}-{acceptance}-{repeat}.json"
                command = [
                    sys.executable,
                    __file__,
                    str(path),
                    "--mode",
                    mode,
                    "--acceptance",
                    str(acceptance),
                    "--episodes",
                    str(args.episodes),
                    "--tokens",
                    str(args.tokens),
                    "--image-bytes",
                    str(args.image_bytes),
                ]
                if args.episode:
                    command.extend(["--episode", str(args.episode.resolve())])
                subprocess.run(command, check=True)
                results.append(json.loads(path.read_text()))
                print(f"Completed {mode}, acceptance={acceptance}, repeat={repeat}", flush=True)
    (args.output / "summary.json").write_text(json.dumps({"checks": checks, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
