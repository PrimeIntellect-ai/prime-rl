"""Per-token training arrays passed by reference.

An env-server worker keeps each finished train episode's per-branch arrays (exactly what
`trace_to_samples` would put on the sample) and serves them over HTTP; the trace carries
only `{url, branch -> id}` in `trace.info["payload"]`. The orchestrator packs `PayloadSlice`
references, and each trainer rank fetches its micro batches' arrays before the step and
assembles them as the packer assembles inline arrays.

Every kept blob is deleted exactly once, by whoever holds it last: the trainer deletes a
step's blobs once every rank fetched them; the orchestrator deletes the blobs of samples
and traces it drops without shipping; the env server deletes the blobs of episodes it
kept but did not deliver (`PayloadHook.discard`).
"""

from __future__ import annotations

import socket
import threading
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import msgspec
import numpy as np
import verifiers.v1 as vf

from prime_rl.orchestrator.trajectories import encode_branch_arrays
from prime_rl.transports.batch.types import BranchArrays, MicroBatch, PayloadSlice, RoutedExperts, TrainingSample
from prime_rl.utils.logger import get_logger

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(BranchArrays)


def advertised_host(bind_address: str | None) -> str:
    """The host trainer ranks reach this env server at: the host it binds (the launcher
    binds the node address), else the address the hostname resolves to."""
    host = urlparse(bind_address).hostname if bind_address else None
    if host and host not in ("*", "0.0.0.0"):
        return host
    return socket.gethostbyname(socket.gethostname())


class PayloadServer:
    """One env-server worker's arrays, served at `GET <url>/<id>` and freed by
    `DELETE <url>/<id>`."""

    def __init__(self, host: str):
        self.blobs: dict[str, bytes] = {}
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

            def do_DELETE(self) -> None:
                server.delete(self.path.lstrip("/"))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self.http = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://{host}:{self.http.server_port}"

    def put(self, key: str, blob: bytes) -> None:
        with self.lock:
            self.blobs[key] = blob

    def delete(self, key: str) -> None:
        with self.lock:
            self.blobs.pop(key, None)

    def get(self, key: str) -> bytes | None:
        with self.lock:
            return self.blobs.get(key)


def _fields(arrays: BranchArrays) -> set[str]:
    fields = {name for name in BranchArrays.__struct_fields__ if getattr(arrays, name) is not None}
    if arrays.sampling_mask is not None and arrays.sampling_mask.logprobs is not None:
        fields.add("sampling_mask_logprobs")
    return fields


class PayloadHook:
    """`verifiers.v1.serve.set_episode_hook` hook: keep a finished train episode's arrays."""

    def __init__(self, bind_address: str | None = None):
        self.server = PayloadServer(advertised_host(bind_address))

    def keep(self, episode: vf.Episode, sampling: vf.SamplingConfig) -> None:
        stored: list[str] = []
        try:
            for trace in episode.traces:
                payload: dict[str, Any] = {"url": self.server.url, "branches": {}, "fields": set()}
                for branch in trace.branches:
                    if not branch.trainable:
                        continue
                    arrays = encode_branch_arrays(branch, len(branch.token_ids))
                    fields = _fields(arrays)
                    if not fields:
                        continue
                    key = f"{trace.id}-{branch.index}"
                    self.server.put(key, _encoder.encode(arrays))
                    stored.append(key)
                    payload["branches"][str(branch.index)] = key
                    payload["fields"] |= fields
                    if arrays.routed_experts is not None:
                        payload["routed_shape"] = arrays.routed_experts.shape[1:]
                        payload["routed_dtype"] = arrays.routed_experts.dtype
                if payload["branches"]:
                    payload["fields"] = sorted(payload["fields"])
                    trace.info["payload"] = payload
                for node in trace.nodes:
                    node.routed_experts = node.sampling_mask = node.top_logprobs = None
        except BaseException:
            # The server never hands this episode to `discard`, so free what it kept here.
            for key in stored:
                self.server.delete(key)
            raise

    def discard(self, episode: vf.Episode) -> None:
        for trace in episode.traces:
            for key in (trace.info.get("payload") or {}).get("branches", {}).values():
                self.server.delete(key)


def trace_payload_urls(trace: vf.Trace) -> list[str]:
    """URLs of every blob an env server kept for the trace."""
    payload = trace.info.get("payload")
    return [f"{payload['url']}/{key}" for key in payload["branches"].values()] if payload else []


def sample_payload_urls(samples: Iterable[TrainingSample]) -> list[str]:
    return [f"{s.payload.url}/{s.payload.id}" for s in samples if s.payload is not None]


def _fetch(url: str) -> BranchArrays | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return _decoder.decode(response.read())
    except Exception:
        return None


def _delete(url: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="DELETE"), timeout=60).close()
    except OSError:
        pass


def delete_payloads(urls: list[str], pool: ThreadPoolExecutor) -> None:
    """Free fetched arrays on their workers, without waiting. Call only once every rank
    that may read them has fetched them."""
    for url in urls:
        pool.submit(_delete, url)


def resolve_payloads(micro_batches: list[MicroBatch], pool: ThreadPoolExecutor) -> tuple[list[int], list[str]]:
    """Fetch the micro batches' arrays into their inline fields, assembled with the packer's
    own helpers: each sample truncated to its slice, concatenated in packing order, and
    extended over the padding tail.

    Returns the indices of micro batches whose arrays could not be fetched (they get
    arrays of the right shapes, and the caller must keep them out of the loss) and the
    fetched URLs, for `delete_payloads`."""
    urls = {f"{s.url}/{s.id}" for mb in micro_batches for s in mb.payloads or []}
    fetched = dict(zip(urls, pool.map(_fetch, urls)))
    failed = []
    for index, mb in enumerate(micro_batches):
        if mb.payloads is None:
            continue
        if not _assemble(mb, [fetched[f"{s.url}/{s.id}"] for s in mb.payloads]):
            failed.append(index)
        mb.payloads = None
    if failed:
        get_logger().warning(f"Could not fetch the payloads of {len(failed)} micro batches")
    return failed, sorted(urls)


def _assemble(mb: MicroBatch, parts: list[BranchArrays | None]) -> bool:
    """Set the micro batch's arrays from its fetched parts; False if a part is missing.
    A missing part's routing rows repeat the micro batch's other rows, so a micro batch
    kept out of the loss still routes like real data; its masks and heads are empty."""
    # Trainer-only, and they pull in torch, which env-server workers importing this module
    # would otherwise load.
    from prime_rl.orchestrator.batch import (
        _slice_routed_experts,
        _slice_sampling_mask,
        _slice_top_logprobs,
        pack_token_arrays,
        pad_token_arrays,
    )

    slices = mb.payloads
    routed = next((s for s in slices if "routed_experts" in s.fields), None)
    ok = True
    packed: list[tuple[int, BranchArrays]] = []
    for s, part in zip(slices, parts, strict=True):
        if part is None or (routed is not None and part.routed_experts is None):
            ok = False
            packed.append((s.num_tokens, BranchArrays()))
            continue
        packed.append(
            (
                s.num_tokens,
                BranchArrays(
                    routed_experts=_slice_routed_experts(part.routed_experts, s.num_tokens)
                    if part.routed_experts is not None
                    else None,
                    sampling_mask=_slice_sampling_mask(part.sampling_mask, s.num_tokens)
                    if part.sampling_mask is not None
                    else None,
                    top_logprobs=_slice_top_logprobs(part.top_logprobs, s.num_tokens)
                    if part.top_logprobs is not None
                    else None,
                ),
            )
        )
    if routed is not None and not ok:
        _fill_routing(packed, routed)
    arrays = pack_token_arrays(packed)
    mb.routed_experts, mb.sampling_mask, mb.top_logprobs = (
        arrays.routed_experts,
        arrays.sampling_mask,
        arrays.top_logprobs,
    )
    pad_token_arrays(mb, mb.padding)
    return ok


def _fill_routing(packed: list[tuple[int, BranchArrays]], routed: PayloadSlice) -> None:
    from prime_rl.utils.sequence import repeat_to_length

    shape, dtype = routed.routed_shape, np.dtype(routed.routed_dtype)
    rows = b"".join(arrays.routed_experts.data for _, arrays in packed if arrays.routed_experts is not None)
    if not rows:
        # Distinct expert ids per token: every top-k slot routes to a different expert.
        rows = np.broadcast_to(np.arange(shape[-1], dtype=dtype), (1, *shape)).tobytes()
    row_size = int(np.prod(shape)) * dtype.itemsize
    for index, (num_tokens, arrays) in enumerate(packed):
        if arrays.routed_experts is None:
            data = repeat_to_length(rows, num_tokens * row_size)
            routed_experts = RoutedExperts(data=data, shape=[num_tokens, *shape], dtype=dtype.name)
            packed[index] = (num_tokens, BranchArrays(routed_experts, arrays.sampling_mask, arrays.top_logprobs))
