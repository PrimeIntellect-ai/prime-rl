"""Per-token training arrays passed by reference.

Training arrays (routing rows, sampling masks, top-k heads) never leave an env server
inline. When a train episode finishes, each env-server worker (`PayloadHook`) encodes every
trainable branch's arrays, keeps the blob and serves it over HTTP; the trace carries only
references in `trace.info[PAYLOAD_INFO_KEY]`. The orchestrator packs one `PayloadSlice` per
sample, and each trainer rank fetches its micro batches' blobs before the step and assembles
them into the micro batch's array fields.

Every kept blob is deleted exactly once, by whoever holds it last: the trainer deletes a
step's blobs once every rank fetched them; the orchestrator deletes the blobs of samples
and traces it drops without shipping; the env server deletes the blobs of episodes it
kept but did not deliver (`PayloadHook.discard`).

Nothing here imports torch: every env-server worker imports this module.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    SamplingMask,
    TopLogprobs,
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


def _encode_sampler_evidence(
    mask: vf.SamplingMask | None, head: vf.TopLogprobs | None, num_tokens: int
) -> tuple[SamplingMask | None, TopLogprobs | None]:
    """Encode a branch's sampling mask and sampler top logprobs for a fixed token count.

    The mask stores row sizes in `counts` and concatenates every row in `ids`.
    For example, `counts=[2, 1]` and `ids=[4, 7, 9]` become `counts=[2]` and
    `ids=[4, 7]` for one token. For three tokens, they become
    `counts=[2, 1, 0]` with unchanged ids. Zero-count rows disable replay.

    A masked token's top logprobs cover exactly its mask, so they move onto the
    mask; the remaining top-logprob rows are heads of untruncated distributions.
    """
    if head is not None and (
        len(head.counts) != num_tokens or int(head.counts.sum()) != len(head.ids) or len(head.ids) != len(head.logprobs)
    ):
        raise ValueError("Top logprobs are not aligned to branch tokens")
    if mask is None:
        if head is None:
            return None, None
        return None, TopLogprobs(
            ids=np.ascontiguousarray(head.ids, dtype=np.int32).tobytes(),
            logprobs=np.ascontiguousarray(head.logprobs, dtype=np.float32).tobytes(),
            counts=np.ascontiguousarray(head.counts, dtype=np.int32).tobytes(),
        )

    mask_ids, mask_counts = np.asarray(mask.ids), np.asarray(mask.counts)
    if len(mask_counts) > num_tokens:
        mask_counts = mask_counts[:num_tokens]
        mask_ids = mask_ids[: int(mask_counts.sum())]
    elif len(mask_counts) < num_tokens:
        mask_counts = np.concatenate([mask_counts, np.zeros(num_tokens - len(mask_counts), dtype=np.int32)])
    if head is None:
        return SamplingMask(
            ids=np.ascontiguousarray(mask_ids, dtype=np.int32).tobytes(),
            counts=np.ascontiguousarray(mask_counts, dtype=np.int32).tobytes(),
        ), None

    head_ids, head_logprobs, head_counts = np.asarray(head.ids), np.asarray(head.logprobs), np.asarray(head.counts)
    if np.array_equal(mask_counts, head_counts) and np.array_equal(mask_ids, head_ids):
        return SamplingMask(
            ids=np.ascontiguousarray(head_ids, dtype=np.int32).tobytes(),
            counts=np.ascontiguousarray(mask_counts, dtype=np.int32).tobytes(),
            logprobs=np.ascontiguousarray(head_logprobs, dtype=np.float32).tobytes(),
        ), None
    rows = np.arange(num_tokens, dtype=np.int64)
    head_rows = np.repeat(rows, head_counts)
    masked = mask_counts > 0
    on_mask = masked[head_rows]
    mask_keys = np.repeat(rows, mask_counts) << 32 | mask_ids.astype(np.int64)
    head_keys = head_rows[on_mask] << 32 | head_ids[on_mask].astype(np.int64)
    if not np.array_equal(np.sort(mask_keys), np.sort(head_keys)):
        raise ValueError("Top logprobs must cover exactly the recorded sampling support")
    sampling_mask = SamplingMask(
        ids=np.ascontiguousarray(head_ids[on_mask], dtype=np.int32).tobytes(),
        counts=np.ascontiguousarray(mask_counts, dtype=np.int32).tobytes(),
        logprobs=np.ascontiguousarray(head_logprobs[on_mask], dtype=np.float32).tobytes(),
    )
    if on_mask.all():
        return sampling_mask, None
    return sampling_mask, TopLogprobs(
        ids=np.ascontiguousarray(head_ids[~on_mask], dtype=np.int32).tobytes(),
        logprobs=np.ascontiguousarray(head_logprobs[~on_mask], dtype=np.float32).tobytes(),
        counts=np.ascontiguousarray(np.where(masked, 0, head_counts), dtype=np.int32).tobytes(),
    )


def encode_branch_arrays(branch: vf.Branch, num_tokens: int) -> BranchArrays:
    """The branch's routing rows, sampling mask and top-k head as the trainer replays them."""
    sampling_mask, top_logprobs = _encode_sampler_evidence(branch.sampling_mask, branch.top_logprobs, num_tokens)
    return BranchArrays(
        routed_experts=_encode_routed_experts(branch.routed_experts, num_tokens),
        sampling_mask=sampling_mask,
        top_logprobs=top_logprobs,
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


_SAMPLING_MASK_ITEMSIZE = np.dtype(np.int32).itemsize


def _empty_sampling_mask(num_tokens: int) -> SamplingMask:
    return SamplingMask(ids=b"", counts=b"\0" * (num_tokens * _SAMPLING_MASK_ITEMSIZE))


def _slice_sampling_mask(sampling_mask: SamplingMask, seq_len: int) -> SamplingMask:
    counts = np.frombuffer(sampling_mask.counts, dtype=np.int32)[:seq_len]
    size = int(counts.sum()) * _SAMPLING_MASK_ITEMSIZE
    return SamplingMask(
        ids=sampling_mask.ids[:size],
        counts=counts.tobytes(),
        logprobs=sampling_mask.logprobs[:size] if sampling_mask.logprobs is not None else None,
    )


def _slice_top_logprobs(head: TopLogprobs, seq_len: int) -> TopLogprobs:
    counts = np.frombuffer(head.counts, dtype=np.int32)[:seq_len]
    size = int(counts.sum()) * 4
    return TopLogprobs(ids=head.ids[:size], logprobs=head.logprobs[:size], counts=counts.tobytes())


def _validate_top_logprobs(head: TopLogprobs, num_tokens: int) -> None:
    counts = np.frombuffer(head.counts, dtype=np.int32)
    if (
        len(counts) != num_tokens
        or (counts < 0).any()
        or len(head.ids) != len(head.logprobs)
        or len(head.ids) != int(counts.sum()) * 4
    ):
        raise ValueError("Top logprobs ids, values and counts must align with packed tokens")


def _pad_sampling_mask(micro_batch: MicroBatch, padding_size: int) -> None:
    """Add zero-count mask rows for sequence-padding tokens.

    Padding adds token positions but no eligible token ids, so only `counts` grows.
    """
    sampling_mask = micro_batch.sampling_mask
    assert sampling_mask is not None
    sampling_mask.counts += b"\0" * (padding_size * _SAMPLING_MASK_ITEMSIZE)


def pack_token_arrays(parts: Sequence[tuple[int, BranchArrays]]) -> BranchArrays:
    """Concatenate samples' routing rows, sampling masks and top-k heads in packing order.
    Each part is a sample's token count and arrays. Routing must be on every part or none;
    masks and heads are per-token optional, so parts without them get empty rows."""
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
    sampling_mask = None
    if any(arrays.sampling_mask is not None for _, arrays in parts):
        masks = [
            arrays.sampling_mask if arrays.sampling_mask is not None else _empty_sampling_mask(num_tokens)
            for num_tokens, arrays in parts
        ]
        logprobs = None
        if any(mask.logprobs is not None for mask in masks):
            # NaN marks mask rows that carry no sampler logprobs.
            logprobs = b"".join(
                mask.logprobs
                if mask.logprobs is not None
                else np.full(len(mask.ids) // _SAMPLING_MASK_ITEMSIZE, np.nan, np.float32).tobytes()
                for mask in masks
            )
        sampling_mask = SamplingMask(
            ids=b"".join(mask.ids for mask in masks),
            counts=b"".join(mask.counts for mask in masks),
            logprobs=logprobs,
        )
    top_logprobs = None
    if any(arrays.top_logprobs is not None for _, arrays in parts):
        heads = [arrays.top_logprobs for _, arrays in parts]
        top_logprobs = TopLogprobs(
            ids=b"".join(head.ids for head in heads if head is not None),
            logprobs=b"".join(head.logprobs for head in heads if head is not None),
            counts=b"".join(
                head.counts if head is not None else bytes(num_tokens * 4)
                for head, (num_tokens, _) in zip(heads, parts, strict=True)
            ),
        )
    return BranchArrays(routed_experts=routed_experts, sampling_mask=sampling_mask, top_logprobs=top_logprobs)


def pad_token_arrays(micro_batch: MicroBatch, padding_size: int) -> None:
    """Extend the micro batch's routing rows, sampling masks and top-k heads over
    ``padding_size`` trailing padding tokens."""
    if micro_batch.routed_experts is not None:
        _pad_routed_experts(micro_batch, padding_size)
    if micro_batch.sampling_mask is not None:
        _pad_sampling_mask(micro_batch, padding_size)
    if micro_batch.top_logprobs is not None:
        micro_batch.top_logprobs.counts += b"\0" * (padding_size * 4)


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
    """One env-server worker's arrays, served at `GET <url>/<id>` and freed by
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


def _fields(arrays: BranchArrays) -> set[str]:
    fields = {name for name in BranchArrays.__struct_fields__ if getattr(arrays, name) is not None}
    if arrays.sampling_mask is not None and arrays.sampling_mask.logprobs is not None:
        fields.add("sampling_mask_logprobs")
    return fields


class PayloadHook:
    """`verifiers.v1.serve.set_episode_hook` hook: keep a finished train episode's arrays."""

    def __init__(self, bind_address: str | None = None):
        self.server = PayloadServer(*payload_host(bind_address))

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
                        # Keep the episode: eval never needs the arrays, and a train run
                        # that needs them stops at the train sink's guard.
                        get_logger().warning(
                            f"Not keeping the arrays of trace {trace.id} branch {branch.index}: {error}"
                        )
                        continue
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
                    node.routed_experts = node.sampling_mask = node.top_logprobs = None
        except BaseException:
            # `discard` finds only references already in `trace.info`; free the rest here.
            for key in stored:
                self.server.delete(key)
            raise

    def discard(self, episode: vf.Episode) -> None:
        for trace in episode.traces:
            for reference in (trace.info.get(PAYLOAD_INFO_KEY) or {}).get("branches", {}).values():
                self.server.delete(reference["id"])


def trace_payload_urls(trace: vf.Trace) -> list[str]:
    """URLs of every blob an env server kept for the trace."""
    payload = trace.info.get(PAYLOAD_INFO_KEY)
    return [f"{payload['url']}/{reference['id']}" for reference in payload["branches"].values()] if payload else []


def sample_payload_urls(samples: Iterable[TrainingSample]) -> list[str]:
    return [f"{s.payload.url}/{s.payload.id}" for s in samples if s.payload is not None]


def _fetch(url: str, attempts: int = 3) -> BranchArrays | None:
    """The blob at `url`, or None if it is gone (404) or every attempt failed."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return _decoder.decode(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(0.5 * 2**attempt)
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
    mb.routed_experts, mb.sampling_mask, mb.top_logprobs = (
        arrays.routed_experts,
        arrays.sampling_mask,
        arrays.top_logprobs,
    )
    pad_token_arrays(mb, mb.padding)
    _assert_aligned(mb, sum(mb.sequence_lengths))
    return ok


def _assert_aligned(mb: MicroBatch, num_tokens: int) -> None:
    """The assembled arrays must stay position-aligned with the micro batch's tokens; a
    misaligned array would corrupt training silently."""
    if mb.routed_experts is not None:
        assert mb.routed_experts.shape[0] == num_tokens, f"routed_experts: {mb.routed_experts.shape} for {num_tokens}"
    if mb.sampling_mask is not None:
        counts = np.frombuffer(mb.sampling_mask.counts, dtype=np.int32)
        assert len(counts) == num_tokens, f"sampling_mask counts: {len(counts)} for {num_tokens}"
        assert len(mb.sampling_mask.ids) == int(counts.sum()) * _SAMPLING_MASK_ITEMSIZE
        assert mb.sampling_mask.logprobs is None or len(mb.sampling_mask.logprobs) == len(mb.sampling_mask.ids)
    if mb.top_logprobs is not None:
        _validate_top_logprobs(mb.top_logprobs, num_tokens)


def _cut(part: BranchArrays, s: PayloadSlice, num_tokens: int) -> BranchArrays | None:
    """The arrays the slice lists, cut to its first `num_tokens`; None if one is missing."""
    cut = {
        "routed_experts": _slice_routed_experts,
        "sampling_mask": _slice_sampling_mask,
        "top_logprobs": _slice_top_logprobs,
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
            parts[index] = (num_tokens, BranchArrays(routed_experts, arrays.sampling_mask, arrays.top_logprobs))
