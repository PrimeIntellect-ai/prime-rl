"""Per-token training arrays passed by reference (``[payloads]``).

An env-server worker keeps each finished train episode's per-branch arrays (exactly what
`trace_to_samples` would put on the sample) and serves them over HTTP; the trace carries
only `{url, branch -> id}` in `trace.info["payload"]`. The orchestrator packs `PayloadSlice`
references, and each trainer rank fetches its micro batches' arrays before the step and
assembles them as the packer assembles inline arrays.

Arrays live until no pending batch can train on them: every train dispatch stamps its
policy version (`epoch`) and the oldest version a pending batch can still train on
(`release_below`), and the worker drops arrays of episodes dispatched before that.
"""

from __future__ import annotations

import socket
import threading
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import msgspec
import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.shared import PayloadsConfig
from prime_rl.orchestrator.trajectories import _encode_routed_experts, _encode_sampling_mask, _encode_top_logprobs
from prime_rl.transports.batch.types import BranchArrays, MicroBatch, RoutedExperts, SamplingMask, TopLogprobs
from prime_rl.utils.logger import get_logger
from prime_rl.utils.sequence import repeat_to_length

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(BranchArrays)


class PayloadServer:
    """One env-server worker's arrays, served at `GET <url>/<id>`."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.blobs: OrderedDict[str, tuple[int, bytes]] = OrderedDict()
        self.nbytes = 0
        self.release_below = 0
        self.evicted = 0
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                blob = server.get(self.path.lstrip("/"))
                self.send_response(200 if blob is not None else 404)
                self.send_header("Content-Length", str(len(blob or b"")))
                self.end_headers()
                if blob is not None:
                    self.wfile.write(blob)

            def log_message(self, *args: Any) -> None:
                pass

        self.http = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://{socket.gethostbyname(socket.gethostname())}:{self.http.server_port}"

    def put(self, key: str, blob: bytes, epoch: int, release_below: int) -> None:
        with self.lock:
            # Blobs arrive in dispatch-version order up to in-flight overlap, so dropping
            # from the oldest end frees every released blob that has reached it.
            self.release_below = max(self.release_below, release_below)
            while self.blobs and next(iter(self.blobs.values()))[0] < self.release_below:
                self.nbytes -= len(self.blobs.popitem(last=False)[1][1])
            self.blobs[key] = (epoch, blob)
            self.nbytes += len(blob)
            while self.nbytes > self.max_bytes and len(self.blobs) > 1:
                self.nbytes -= len(self.blobs.popitem(last=False)[1][1])
                self.evicted += 1
                if self.evicted == 1 or self.evicted % 1000 == 0:
                    get_logger().warning(
                        f"Payload store is over {self.max_bytes / 2**30:.0f} GiB and dropped "
                        f"{self.evicted} unreleased arrays; raise [payloads] max_gib"
                    )

    def get(self, key: str) -> bytes | None:
        with self.lock:
            entry = self.blobs.get(key)
        return entry[1] if entry is not None else None


class PayloadHook:
    """`verifiers.v1.serve.set_episode_hook` hook: keep a finished train episode's arrays."""

    def __init__(self, config: PayloadsConfig):
        self.config = config
        self.server: PayloadServer | None = None

    def __call__(self, episode: vf.Episode, sampling: vf.SamplingConfig) -> None:
        stamp = (getattr(sampling, "extra_args", None) or {}).get("prl_payload")
        for trace in episode.traces:
            branches: dict[str, str] = {}
            fields: set[str] = set()
            if stamp is not None:
                for branch in trace.branches:
                    if not branch.trainable:
                        continue
                    num_tokens = len(branch.token_ids)
                    arrays = BranchArrays(
                        routed_experts=_encode_routed_experts(branch.routed_experts, num_tokens),
                        sampling_mask=_encode_sampling_mask(branch.sampling_mask, num_tokens),
                        top_logprobs=_encode_top_logprobs(branch.top_logprobs, num_tokens),
                    )
                    present = {name for name in BranchArrays.__struct_fields__ if getattr(arrays, name) is not None}
                    if not present:
                        continue
                    if self.server is None:
                        self.server = PayloadServer(self.config.max_gib << 30)
                    key = f"{trace.id}-{branch.index}"
                    self.server.put(key, _encoder.encode(arrays), stamp["epoch"], stamp["release_below"])
                    branches[str(branch.index)] = key
                    fields |= present
            if branches:
                trace.info["payload"] = {"url": self.server.url, "branches": branches, "fields": sorted(fields)}
            for node in trace.nodes:
                node.routed_experts = node.sampling_mask = node.top_logprobs = None


def _fetch(url: str) -> BranchArrays | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return _decoder.decode(response.read())
    except OSError:
        return None


def resolve_payloads(micro_batches: list[MicroBatch], pool: ThreadPoolExecutor) -> list[int]:
    """Fetch the micro batches' arrays into their inline fields, assembled as the packer
    assembles inline arrays: each sample truncated to its slice, concatenated in packing
    order, and the padding tail filled (routing repeats the micro batch's own rows, as
    `pad_micro_batch` does; masks and heads get empty rows).

    Returns the indices of micro batches whose arrays could not be fetched; they get
    arrays of the right shapes, and the caller must keep them out of the loss."""
    urls = {f"{s.url}/{s.id}" for mb in micro_batches for s in mb.payloads or []}
    fetched = dict(zip(urls, pool.map(_fetch, urls)))
    failed = []
    for index, mb in enumerate(micro_batches):
        if mb.payloads is None:
            continue
        parts = [fetched[f"{s.url}/{s.id}"] for s in mb.payloads]
        if any(part is None for part in parts):
            failed.append(index)
        _assemble(mb, parts)
        mb.payloads = None
    if failed:
        get_logger().warning(f"Could not fetch the payloads of {len(failed)} micro batches")
    return failed


def _assemble(mb: MicroBatch, parts: list[BranchArrays | None]) -> None:
    lengths = [s.num_tokens for s in mb.payloads]
    tail = len(mb.input_ids) - sum(lengths)
    fields = {name for s in mb.payloads for name in s.fields}
    if "routed_experts" in fields:
        rows, shape, dtype = [], None, "uint8"
        for part, n in zip(parts, lengths, strict=True):
            routed = part.routed_experts if part is not None else None
            if routed is not None:
                shape, dtype = routed.shape[1:], routed.dtype
                rows.append(np.frombuffer(routed.data, dtype=dtype).reshape(routed.shape)[:n])
            else:
                rows.append(None)
        shape = shape or [1, 1]
        rows = [r if r is not None else np.zeros((n, *shape), dtype=dtype) for r, n in zip(rows, lengths)]
        data = b"".join(r.tobytes() for r in rows)
        row_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        data += repeat_to_length(data, tail * row_size) if tail else b""
        mb.routed_experts = RoutedExperts(data=data, shape=[len(mb.input_ids), *shape], dtype=dtype)
    for name, flat in (("sampling_mask", ("ids",)), ("top_logprobs", ("ids", "logprobs"))):
        if name not in fields:
            continue
        columns: dict[str, list[bytes]] = {key: [] for key in (*flat, "counts")}
        for part, n in zip(parts, lengths, strict=True):
            head = getattr(part, name) if part is not None else None
            counts = np.frombuffer(head.counts, dtype=np.int32)[:n] if head is not None else np.zeros(n, np.int32)
            columns["counts"].append(counts.tobytes())
            for key in flat:
                columns[key].append(bytes(getattr(head, key))[: int(counts.sum()) * 4] if head is not None else b"")
        columns["counts"].append(bytes(4 * tail))
        joined = {key: b"".join(chunks) for key, chunks in columns.items()}
        setattr(mb, name, (SamplingMask if name == "sampling_mask" else TopLogprobs)(**joined))
