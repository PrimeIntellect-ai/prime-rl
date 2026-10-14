"""Use the previous full-native-return harness unchanged, replacing only decoding."""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import replay


class ZeroCopyReceiver:
    def __init__(self, socket):
        self.socket = socket

    async def recv_multipart(self):
        parts = await self.socket.recv_multipart(copy=False)
        # Keep the Frame owner alive through its readonly buffer; never reuse an input slot.
        return [parts[0].bytes, parts[1].bytes, parts[2].buffer.toreadonly()]


def main(args):
    sys.path.insert(0, str(args.harness.parent))
    contract = importlib.import_module(args.harness.stem)
    contract.ROOT = args.fixture
    if args.zero_copy_receive:
        assert args.mode in ("thread", "typed-bytes", "typed-views")
        base_consume = contract.o.b.consume_frames

        async def consume(socket, *values, **kwargs):
            return await base_consume(ZeroCopyReceiver(socket), *values, **kwargs)

        contract.o.b.consume_frames = consume
    if args.mode.startswith("typed-"):
        typed = replay.decoder(bytes if args.mode == "typed-bytes" else memoryview)

        def unpack(data):
            value = replay.containers(typed.decode(data))
            return replay.owned_binary_fields(value) if args.callbacks else value

        contract.unpack = unpack
        if args.mode == "typed-views" and not args.callbacks:

            def verify_types(episode):
                assert type(episode) is contract.WireEpisode
                counts = {"nodes": 0, "token_ids": 0, "routing_arrays": 0}
                for trace in episode.traces:
                    assert isinstance(trace.nodes, list) and isinstance(trace.calls, list)
                    for node in trace.nodes:
                        counts["nodes"] += 1
                        counts["token_ids"] += len(node.token_ids)
                        for name in ("token_ids", "mask", "is_content", "logprobs", "semantic_parents"):
                            assert isinstance(getattr(node, name), list)
                        if node.routed_experts is not None:
                            counts["routing_arrays"] += 1
                            base = node.routed_experts
                            assert isinstance(base, np.ndarray)
                            while isinstance(base, np.ndarray) and base.base is not None:
                                base = base.base
                            assert isinstance(base, bytes) or isinstance(base, memoryview) and base.readonly
                return counts

            contract.verify_types = verify_types
    args.variant = "thread" if args.mode.startswith("typed-") else args.mode
    asyncio.run(contract.main(args))
    path = args.fixture / "results" / args.name / "result.json"
    result = json.loads(path.read_text())
    result["mode"] = args.mode
    result["zero_copy_receive"] = args.zero_copy_receive
    result["comparison"] = "Previous saved-delta full native episode harness, same input, return, callbacks and release"
    result["source_hashes"].update(
        {
            "exact_harness.py": replay.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "typed_decoder_replay.py": replay.hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        }
    )
    result["limits"] += "; typed decoder maps do not retain original dictionary insertion order"
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("harness", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--mode",
        choices=["thread", "delta-pickle", "delta-shm", "owner-shm", "typed-bytes", "typed-views"],
        required=True,
    )
    parser.add_argument("--callbacks", action="store_true")
    parser.add_argument("--zero-copy-receive", action="store_true")
    parser.add_argument("--queues", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--queue-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--queue-items", type=int, default=64)
    parser.add_argument("--name", required=True)
    main(parser.parse_args())
