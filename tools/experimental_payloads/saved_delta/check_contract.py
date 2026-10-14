"""Replay native EnvClient.run with deterministic saved deltas, without an RPC."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import replay
import verifiers.v1.serve.client as client_module
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.types import SamplingConfig


async def main(args):
    meta = json.loads((args.fixture / "fixture.json").read_text())
    wires = [(args.fixture / "frames" / frame["file"]).read_bytes() for frame in meta["frames"]]
    assert all(hashlib.sha256(wire).hexdigest() == frame["sha256"] for wire, frame in zip(wires, meta["frames"]))
    reference = replay.EpisodeAssembly()
    for wire in wires:
        reference.apply(replay.unpack(wire))
    record = replay.raw_episode(reference)
    head = {key: value for key, value in record.items() if key != "traces"}
    summaries = [
        replay.TraceSummary(id=key, nodes=len(trace["nodes"]), calls=len(trace["calls"]))
        for key, trace in reference.traces.items()
    ]
    expected = replay.fingerprint(replay.WireEpisode.model_validate(record))
    checks = []
    original_unpack = client_module.unpack
    for fault in (False, True):
        baseline_sha = baseline_errors = None
        for mode in ("native", "typed-bytes", "typed-views"):
            typed = None if mode == "native" else replay.decoder(bytes if mode == "typed-bytes" else memoryview)
            client_module.unpack = (
                original_unpack
                if typed is None
                else (lambda data: replay.owned_binary_fields(replay.containers(typed.decode(data))))
            )
            observations, errors = [], []

            class ReplayClient(EnvClient):
                async def _request(self, request, response_type, timeout=None, on_delta=None):
                    for wire in wires:
                        try:
                            on_delta(wire)
                        except RuntimeError as error:
                            errors.append(str(error))
                    return response_type.model_validate(
                        {"success": True, "head": head, "traces": [summary.model_dump() for summary in summaries]}
                    )

            def observer(delta):
                ordinal = len(observations)
                assert delta == original_unpack(wires[ordinal])
                observations.append(replay.observe(delta))
                info = (delta.get("set") or {}).get("info")
                if fault and isinstance(info, dict):
                    info["offline_callback_mutation"] = "same-parent-object"
                if fault and len(observations) == 2:
                    raise RuntimeError("expected-offline-callback-error")

            client = ReplayClient("ipc:///tmp/offline-deferred-contract-unused")
            try:
                episode = await client.run(
                    EvalClientConfig(base_url="http://127.0.0.1:1", api_key_var="OFFLINE_UNUSED"),
                    "offline",
                    SamplingConfig(),
                    {},
                    on_delta=observer,
                )
                sha = replay.fingerprint(episode)
                replay.verify_native(episode, sha, False)
                if mode == "native":
                    baseline_sha, baseline_errors = sha, errors
                else:
                    assert sha == baseline_sha and errors == baseline_errors
                if not fault:
                    assert sha == expected
                assert len(observations) == len(wires)
                checks.append(
                    {
                        "mode": mode,
                        "mutation_and_exception": fault,
                        "callback_count": len(observations),
                        "caught_exceptions": len(errors),
                        "native_episode_fingerprint": sha,
                        "passed": True,
                    }
                )
            finally:
                await client.close()
                client_module.unpack = original_unpack
    result = {
        "all_passed": True,
        "checks": checks,
        "fixture_sha256": meta["fixture_sha256"],
        "limits": "Unchanged native EnvClient.run and validation, with offline _request byte source. "
        "Values and native types match; raw dictionary insertion order may differ. "
        "No live receive-loop RPC, cancellation/timeout API, or Prime RL integration exercised.",
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    asyncio.run(main(parser.parse_args()))
