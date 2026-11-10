"""Per-token training arrays passed by reference.

Training arrays (routing rows, sampling masks, top-k heads) never leave an env worker
inline. When a train episode finishes, each env worker (`PayloadHook`) encodes every
trainable branch's arrays, keeps the blob and serves it over HTTP; the trace carries only
references in `trace.info[PAYLOAD_INFO_KEY]`. The orchestrator packs one `PayloadSlice` per
sample, and each trainer rank fetches its micro batches' blobs before the step and assembles
them into the micro batch's array fields.

Every kept blob is deleted exactly once, by whoever holds it last: the trainer deletes a
step's blobs once every rank fetched them; the orchestrator deletes the blobs of samples
and traces it drops without shipping; the env worker deletes the blobs of episodes it
kept but did not deliver (`PayloadHook.discard`).

With `rollout_transport.retain`, a pool's workers write their blobs as files into the
directory of one `RolloutStore`, which serves them for the whole pool. The trainer then
marks a step's blobs as trained instead of deleting them, and the store deletes them once
a trainer checkpoint covers the step, so a restarted trainer fetches them again.

Nothing here imports torch: every env worker imports this module.
"""

from __future__ import annotations

import queue
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import msgspec
import numpy as np
import verifiers.v1 as vf

from prime_rl.transports.batch.types import (
    PAYLOAD_INFO_KEY,
    BranchArrays,
    MicroBatch,
    PayloadSlice,
    RoutedExperts,
    SamplerSupport,
    TrainingSample,
)
from prime_rl.utils.logger import get_logger

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(BranchArrays)


def _tile(data: bytes, size: int) -> bytes:
    """`data` repeated end to end and cut to `size` bytes."""
    return (data * (size // len(data) + 1))[:size]


def _encode_routed_experts(arr: np.ndarray | None, num_tokens: int) -> RoutedExperts | None:
    """The branch's router-replay array (`[tokens, layers, top_k]`) -> the transport
    `RoutedExperts` the trainer replays. Defensively realigns the token axis to `num_tokens`
    (the trainer asserts `routed_experts.shape[0] == len(token_ids)`): truncate if longer,
    zero-pad the tail if shorter. `Branch.routed_experts` already guarantees alignment, so this
    is a backstop."""
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if arr.shape[0] > num_tokens:
        arr = arr[:num_tokens]
    elif arr.shape[0] < num_tokens:
        pad = np.zeros((num_tokens - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return RoutedExperts(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _encode_sampler_support(support: vf.SamplerSupport | None, num_tokens: int) -> SamplerSupport | None:
    """Encode a branch's sampler support for a fixed token count: rows past ``num_tokens`` are cut
    with their entries, missing rows are empty."""
    if support is None:
        return None
    ids, counts, replayed, logprobs = support.ids, support.counts, support.replayed, support.logprobs
    if len(counts) > num_tokens:
        counts, replayed = counts[:num_tokens], replayed[:num_tokens]
        size = int(counts.sum())
        ids, logprobs = ids[:size], logprobs[:size] if logprobs is not None else None
    elif len(counts) < num_tokens:
        missing = num_tokens - len(counts)
        counts = np.concatenate([counts, np.zeros(missing, dtype=np.int32)])
        replayed = np.concatenate([replayed, np.zeros(missing, dtype=np.bool_)])
    if int(counts.sum()) != len(ids) or (logprobs is not None and len(logprobs) != len(ids)):
        raise ValueError("Sampler support is not aligned to branch tokens")
    return SamplerSupport(
        ids=np.ascontiguousarray(ids, dtype=np.int32).tobytes(),
        counts=np.ascontiguousarray(counts, dtype=np.int32).tobytes(),
        replayed=np.ascontiguousarray(replayed, dtype=np.bool_).tobytes(),
        logprobs=np.ascontiguousarray(logprobs, dtype=np.float32).tobytes() if logprobs is not None else None,
    )


def encode_branch_arrays(branch: vf.Branch, num_tokens: int) -> BranchArrays:
    """The branch's routing rows and sampler support as the trainer replays them."""
    return BranchArrays(
        routed_experts=_encode_routed_experts(branch.routed_experts, num_tokens),
        sampler_support=_encode_sampler_support(branch.sampler_support, num_tokens),
    )


def _routed_experts_row_size(routed_experts: RoutedExperts) -> int:
    return routed_experts.shape[1] * routed_experts.shape[2] * np.dtype(routed_experts.dtype).itemsize


def _slice_routed_experts(routed_experts: RoutedExperts, seq_len: int) -> RoutedExperts:
    row_size = _routed_experts_row_size(routed_experts)
    return RoutedExperts(
        data=routed_experts.data[: seq_len * row_size],
        shape=[seq_len, routed_experts.shape[1], routed_experts.shape[2]],
        dtype=routed_experts.dtype,
    )


def _pad_routed_experts(micro_batch: MicroBatch, padding_size: int) -> None:
    routed_experts = micro_batch.routed_experts
    assert routed_experts is not None
    row_size = _routed_experts_row_size(routed_experts)
    routed_experts.data += _tile(routed_experts.data, padding_size * row_size)
    routed_experts.shape[0] += padding_size


def _empty_sampler_support(num_tokens: int) -> SamplerSupport:
    return SamplerSupport(ids=b"", counts=bytes(num_tokens * 4), replayed=bytes(num_tokens))


def _slice_sampler_support(support: SamplerSupport, seq_len: int) -> SamplerSupport:
    counts = np.frombuffer(support.counts, dtype=np.int32)[:seq_len]
    size = int(counts.sum()) * 4
    return SamplerSupport(
        ids=support.ids[:size],
        counts=counts.tobytes(),
        replayed=support.replayed[:seq_len],
        logprobs=support.logprobs[:size] if support.logprobs is not None else None,
    )


def pack_token_arrays(parts: Sequence[tuple[int, BranchArrays]]) -> BranchArrays:
    """Concatenate samples' routing rows and sampler supports in packing order.
    Each part is a sample's token count and arrays. Routing must be on every part or none;
    the support is per-token optional, so parts without one get empty rows."""
    routed = [arrays.routed_experts for _, arrays in parts if arrays.routed_experts is not None]
    routed_experts = None
    if routed:
        assert len(routed) == len(parts), "routing must be on every part or none"
        assert all(r.dtype == routed[0].dtype and r.shape[1:] == routed[0].shape[1:] for r in routed)
        routed_experts = RoutedExperts(
            data=b"".join(r.data for r in routed),
            shape=[sum(r.shape[0] for r in routed), *routed[0].shape[1:]],
            dtype=routed[0].dtype,
        )
    sampler_support = None
    if any(arrays.sampler_support is not None for _, arrays in parts):
        supports = [
            arrays.sampler_support if arrays.sampler_support is not None else _empty_sampler_support(num_tokens)
            for num_tokens, arrays in parts
        ]
        logprobs = None
        if any(support.logprobs is not None for support in supports):
            # NaN marks entries that carry no sampler logprobs.
            logprobs = b"".join(
                support.logprobs
                if support.logprobs is not None
                else np.full(len(support.ids) // 4, np.nan, np.float32).tobytes()
                for support in supports
            )
        sampler_support = SamplerSupport(
            ids=b"".join(support.ids for support in supports),
            counts=b"".join(support.counts for support in supports),
            replayed=b"".join(support.replayed for support in supports),
            logprobs=logprobs,
        )
    return BranchArrays(routed_experts=routed_experts, sampler_support=sampler_support)


def pad_token_arrays(micro_batch: MicroBatch, padding_size: int) -> None:
    """Extend the micro batch's routing rows and sampler support over ``padding_size`` trailing
    padding tokens, which have no support."""
    if micro_batch.routed_experts is not None:
        _pad_routed_experts(micro_batch, padding_size)
    if micro_batch.sampler_support is not None:
        micro_batch.sampler_support.counts += bytes(padding_size * 4)
        micro_batch.sampler_support.replayed += bytes(padding_size)


def payload_host(bind_address: str | None) -> tuple[str, str]:
    """The host the payload server binds and the host trainer ranks reach it at: the env
    server's own bind host (the launcher binds the node address; unset means the loopback
    default), or for a wildcard bind, every interface and a routable address of this node."""
    if bind_address is None:
        return "127.0.0.1", "127.0.0.1"
    host = urlparse(bind_address).hostname
    if host and host not in ("*", "0.0.0.0"):
        return host, host
    return "0.0.0.0", _routable_address()


def _routable_address() -> str:
    """A non-loopback address of this node: the hostname often maps to 127.0.1.1."""
    _, _, addresses = socket.gethostbyname_ex(socket.gethostname())
    if routable := [address for address in addresses if not address.startswith("127.")]:
        return routable[0]
    # Connecting a UDP socket sends nothing; it only picks the outbound interface.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]


class _HTTPServer(ThreadingHTTPServer):
    # Every trainer rank fetches its blobs at once at step start; the default backlog
    # of 5 drops connections, which retry only after a second or more.
    request_queue_size = 1024
    daemon_threads = True


class PayloadServer:
    """One env worker's arrays, served at `GET <url>/<id>` and freed by
    `DELETE <url>/<id>`."""

    def __init__(self, bind_host: str, url_host: str | None = None):
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

            def do_POST(self) -> None:
                server.trained(_trained_decoder.decode(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self.http = _HTTPServer((bind_host, 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://{url_host or bind_host}:{self.http.server_port}"

    def put(self, key: str, blob: bytes) -> None:
        with self.lock:
            self.blobs[key] = blob

    def delete(self, key: str) -> None:
        with self.lock:
            self.blobs.pop(key, None)

    def get(self, key: str) -> bytes | None:
        with self.lock:
            return self.blobs.get(key)

    def trained(self, trained: Trained) -> None:
        raise NotImplementedError("Only a rollout store retains trained arrays")


class Trained(msgspec.Struct):
    """A trainer telling a rollout store that `ids` went into `step`, and that a checkpoint
    covers every step up to `committed`."""

    step: int
    ids: list[str]
    committed: int


_trained_decoder = msgspec.msgpack.Decoder(Trained)


class RolloutStore(PayloadServer):
    """One env worker pool's arrays as files, kept until a trainer checkpoint covers them.

    The pool's workers write each blob to `live/<id>` (`StoreWriter`). A blob the trainer
    trained on moves to `step_<N>/<id>` and stays fetchable; the step directories up to the
    trainer's committed step are deleted."""

    def __init__(self, directory: Path, bind_host: str, url_host: str | None = None):
        self.directory = directory
        self.live = directory / "live"
        self.live.mkdir(parents=True)
        self.steps: dict[str, int] = {}
        """The step each trained blob went into."""
        super().__init__(bind_host, url_host)

    def path(self, key: str) -> Path:
        if Path(key).name != key:
            raise ValueError(f"Invalid blob id: {key}")
        with self.lock:
            step = self.steps.get(key)
        return self.live / key if step is None else self.directory / f"step_{step}" / key

    def get(self, key: str) -> bytes | None:
        try:
            return self.path(key).read_bytes()
        except FileNotFoundError:
            return None

    def delete(self, key: str) -> None:
        """Free a blob that will never ship. A trained blob stays until its step is committed."""
        path = self.path(key)
        if path.parent == self.live:
            path.unlink(missing_ok=True)

    def trained(self, trained: Trained) -> None:
        if trained.ids:
            step_dir = self.directory / f"step_{trained.step}"
            step_dir.mkdir(exist_ok=True)
            for key in trained.ids:
                path = self.path(key)
                if path.parent != self.live:
                    continue
                # Another data rank may mark the same blob, and a step trained again after a
                # trainer restart has its blobs there already.
                try:
                    path.rename(step_dir / key)
                except FileNotFoundError:
                    continue
                with self.lock:
                    self.steps[key] = trained.step
        for step_dir in self.directory.glob("step_*"):
            if int(step_dir.name.removeprefix("step_")) <= trained.committed:
                shutil.rmtree(step_dir, ignore_errors=True)
        with self.lock:
            self.steps = {key: step for key, step in self.steps.items() if step > trained.committed}


class StoreWriter:
    """An env worker's side of its pool's `RolloutStore`: blobs go straight to the store's
    directory, and the store serves them at `url`."""

    def __init__(self, directory: Path, url: str):
        self.live = directory / "live"
        self.url = url

    def put(self, key: str, blob: bytes) -> None:
        tmp = self.live / f".{key}.tmp"
        tmp.write_bytes(blob)
        tmp.rename(self.live / key)

    def delete(self, key: str) -> None:
        (self.live / key).unlink(missing_ok=True)


def _fields(arrays: BranchArrays) -> set[str]:
    fields = {name for name in BranchArrays.__struct_fields__ if getattr(arrays, name) is not None}
    if arrays.sampler_support is not None:
        if np.frombuffer(arrays.sampler_support.replayed, dtype=np.bool_).any():
            fields.add("sampler_support_replayed")
        if arrays.sampler_support.logprobs is not None:
            fields.add("sampler_support_logprobs")
    return fields


class PayloadHook:
    """`prime_rl.env.set_episode_hook` hook: keep a finished train episode's arrays."""

    def __init__(self, bind_address: str | None = None, store: StoreWriter | None = None):
        self._hosts = payload_host(bind_address)
        self._server: PayloadServer | StoreWriter | None = store
        self._lock = threading.Lock()

    @property
    def server(self) -> PayloadServer | StoreWriter:
        """The pool's rollout store if it has one, else this worker's own server, started on
        first use: a pool broker installs the hook too but never keeps arrays."""
        with self._lock:
            if self._server is None:
                self._server = PayloadServer(*self._hosts)
            return self._server

    def keep(self, episode: vf.Episode, sampling: vf.SamplingConfig) -> None:
        stored: list[str] = []
        try:
            for trace in episode.traces:
                references: dict[str, dict[str, Any]] = {}
                for branch in trace.branches:
                    if not branch.trainable:
                        continue
                    try:
                        arrays = encode_branch_arrays(branch, len(branch.token_ids))
                    except ValueError as error:
                        # Keep the episode and its routing rows: eval never needs the sampler
                        # evidence, and a train run that needs it stops at the train sink.
                        get_logger().warning(
                            f"Not keeping the sampling mask and top-k head of trace {trace.id} "
                            f"branch {branch.index}: {error}"
                        )
                        arrays = BranchArrays(
                            routed_experts=_encode_routed_experts(branch.routed_experts, len(branch.token_ids))
                        )
                    fields = _fields(arrays)
                    if not fields:
                        continue
                    key = f"{trace.id}-{branch.index}"
                    self.server.put(key, _encoder.encode(arrays))
                    stored.append(key)
                    references[str(branch.index)] = {"id": key, "fields": sorted(fields)}
                    if arrays.routed_experts is not None:
                        references[str(branch.index)] |= {
                            "routed_shape": arrays.routed_experts.shape[1:],
                            "routed_dtype": arrays.routed_experts.dtype,
                        }
                if references:
                    trace.info[PAYLOAD_INFO_KEY] = {"url": self.server.url, "branches": references}
                for node in trace.nodes:
                    node.routed_experts = node.sampler_support = None
        except BaseException:
            # `discard` finds only references already in `trace.info`; free the rest here.
            for key in stored:
                self.server.delete(key)
            raise

    def discard(self, episode: vf.Episode) -> None:
        if self._server is None:
            return
        for trace in episode.traces:
            for reference in (trace.info.get(PAYLOAD_INFO_KEY) or {}).get("branches", {}).values():
                self.server.delete(reference["id"])


def trace_payload_urls(trace: vf.Trace) -> list[str]:
    """URLs of every blob an env worker kept for the trace."""
    payload = trace.info.get(PAYLOAD_INFO_KEY)
    return [f"{payload['url']}/{reference['id']}" for reference in payload["branches"].values()] if payload else []


def sample_payload_urls(samples: Iterable[TrainingSample]) -> list[str]:
    return [f"{s.payload.url}/{s.payload.id}" for s in samples if s.payload is not None]


# Payload URLs are node-local addresses: never route them through an `HTTP_PROXY`.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _fetch(url: str, attempts: int = 3) -> BranchArrays | None:
    """The blob at `url`, or None if it is gone (404) or every attempt failed."""
    for attempt in range(attempts):
        try:
            with _opener.open(url, timeout=60) as response:
                return _decoder.decode(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(0.5 * 2**attempt)
    return None


class _Releaser:
    """Fire-and-forget release requests on daemon threads, so requests still queued at exit
    (say, to an env worker that is gone) never hold the process up."""

    def __init__(self, num_threads: int = 8) -> None:
        self.num_threads = num_threads
        self.queue: queue.Queue[urllib.request.Request] = queue.Queue()
        self._started = False
        self._lock = threading.Lock()

    def submit(self, requests: Iterable[urllib.request.Request]) -> None:
        with self._lock:
            if not self._started:
                for _ in range(self.num_threads):
                    threading.Thread(target=self._run, daemon=True, name="payload-release").start()
                self._started = True
        for request in requests:
            self.queue.put(request)

    def _run(self) -> None:
        while True:
            request = self.queue.get()
            try:
                _opener.open(request, timeout=10).close()
            except Exception:
                pass
            finally:
                self.queue.task_done()


_releaser = _Releaser()


def delete_payloads(urls: Iterable[str]) -> None:
    """Free kept arrays on their env workers, without waiting. Call only once nothing will
    read them again."""
    _releaser.submit(urllib.request.Request(url, method="DELETE") for url in urls)


def mark_trained(stores: Iterable[str], urls: Iterable[str], step: int, committed: int) -> None:
    """Tell each of the rollout `stores` which of the blobs at `urls` it holds went into
    `step`, and that a trainer checkpoint covers every step up to `committed`, without
    waiting."""
    ids: dict[str, list[str]] = {store: [] for store in stores}
    for url in urls:
        store, _, key = url.rpartition("/")
        ids.setdefault(store, []).append(key)
    _releaser.submit(
        urllib.request.Request(store, data=msgspec.msgpack.encode(Trained(step, keys, committed)), method="POST")
        for store, keys in ids.items()
    )


def wait_for_deletes() -> None:
    """Block until every queued release request has run."""
    _releaser.queue.join()


def resolve_payloads(micro_batches: list[MicroBatch], pool: ThreadPoolExecutor) -> tuple[list[int], list[str]]:
    """Fetch the micro batches' arrays into their array fields: each sample's arrays cut
    to its slice, concatenated in packing order, and extended over the padding tail.

    Returns the indices of micro batches whose arrays could not be fetched (they get
    arrays of the right shapes, and the caller must keep them out of the loss) and the
    fetched URLs, for `delete_payloads`."""
    urls = {f"{s.url}/{s.id}" for mb in micro_batches for s in mb.payloads or [] if s is not None}
    fetched = dict(zip(urls, pool.map(_fetch, urls)))
    failed = []
    for index, mb in enumerate(micro_batches):
        if mb.payloads is None:
            continue
        if not _assemble(mb, fetched):
            failed.append(index)
        mb.payloads = None
    if failed:
        get_logger().warning(f"Could not fetch the payloads of {len(failed)} micro batches")
    return failed, sorted(urls)


def _assemble(mb: MicroBatch, fetched: dict[str, BranchArrays | None]) -> bool:
    """Set the micro batch's arrays from its fetched blobs; False if one is missing.
    A missing blob's routing rows repeat the micro batch's other rows, so a micro batch
    kept out of the loss still routes like real data; its masks and heads are empty."""
    lengths = list(mb.sequence_lengths)
    lengths[-1] -= mb.padding
    ok = True
    parts: list[tuple[int, BranchArrays]] = []
    for s, num_tokens in zip(mb.payloads, lengths, strict=True):
        if s is None:
            parts.append((num_tokens, BranchArrays()))
            continue
        assert s.num_tokens == num_tokens, f"reference holds {s.num_tokens} tokens, its sequence {num_tokens}"
        part = fetched[f"{s.url}/{s.id}"]
        arrays = _cut(part, s, num_tokens) if part is not None else None
        if arrays is None:
            ok = False
        parts.append((num_tokens, arrays or BranchArrays()))
    routed = next((s for s in mb.payloads if s is not None and "routed_experts" in s.fields), None)
    if routed is not None and not ok:
        _fill_routing(parts, routed)
    arrays = pack_token_arrays(parts)
    mb.routed_experts, mb.sampler_support = arrays.routed_experts, arrays.sampler_support
    pad_token_arrays(mb, mb.padding)
    _assert_aligned(mb, sum(mb.sequence_lengths))
    return ok


def _assert_aligned(mb: MicroBatch, num_tokens: int) -> None:
    """The assembled arrays must stay position-aligned with the micro batch's tokens; a
    misaligned array would corrupt training silently."""
    if mb.routed_experts is not None:
        assert mb.routed_experts.shape[0] == num_tokens, f"routed_experts: {mb.routed_experts.shape} for {num_tokens}"
        assert len(mb.routed_experts.data) == num_tokens * _routed_experts_row_size(mb.routed_experts), (
            f"routed_experts: {len(mb.routed_experts.data)} bytes for {mb.routed_experts.shape}"
        )
    support = mb.sampler_support
    if support is not None:
        counts = np.frombuffer(support.counts, dtype=np.int32)
        assert len(counts) == num_tokens, f"sampler_support counts: {len(counts)} for {num_tokens}"
        assert len(support.replayed) == num_tokens, f"sampler_support replayed: {len(support.replayed)}"
        assert (counts >= 0).all() and len(support.ids) == int(counts.sum()) * 4
        assert support.logprobs is None or len(support.logprobs) == len(support.ids)


def _cut(part: BranchArrays, s: PayloadSlice, num_tokens: int) -> BranchArrays | None:
    """The arrays the slice lists, cut to its first `num_tokens`; None if one is missing."""
    cut = {
        "routed_experts": _slice_routed_experts,
        "sampler_support": _slice_sampler_support,
    }
    arrays = {}
    for name in BranchArrays.__struct_fields__:
        if name not in s.fields:
            continue
        value = getattr(part, name)
        if value is None:
            return None
        arrays[name] = cut[name](value, num_tokens)
    return BranchArrays(**arrays)


def _fill_routing(parts: list[tuple[int, BranchArrays]], routed: PayloadSlice) -> None:
    shape, dtype = routed.routed_shape, np.dtype(routed.routed_dtype)
    rows = b"".join(arrays.routed_experts.data for _, arrays in parts if arrays.routed_experts is not None)
    if not rows:
        # Distinct expert ids per token: every top-k slot routes to a different expert.
        rows = np.broadcast_to(np.arange(shape[-1], dtype=dtype), (1, *shape)).tobytes()
    row_size = int(np.prod(shape)) * dtype.itemsize
    for index, (num_tokens, arrays) in enumerate(parts):
        if arrays.routed_experts is None:
            routed_experts = RoutedExperts(
                data=_tile(rows, num_tokens * row_size), shape=[num_tokens, *shape], dtype=dtype.name
            )
            parts[index] = (num_tokens, BranchArrays(routed_experts, arrays.sampler_support))
