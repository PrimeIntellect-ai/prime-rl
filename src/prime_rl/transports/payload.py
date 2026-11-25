"""Training samples passed by reference.

A training sample never leaves the env worker that ran it inline. When a train episode finishes,
each env worker (`PayloadHook`) builds every trainable branch's sample: its per-token columns
(tokens, masks, logprobs, policy versions, loss weights, reference logprobs, multimodal ids)
and training arrays (routing rows, sampling masks, top-k heads). It keeps the encoded
`SampleBody` and serves it over HTTP; the trace carries only a reference and a small summary per
branch in `trace.info[PAYLOAD_INFO_KEY]`. The orchestrator decides about samples from the
summaries and ships `MicroBatchPlan`s of `SampleRef`s, and each trainer rank fetches its micro
batches' bodies and builds them (`materialize_micro_batches`).

Every kept body is deleted exactly once, by whoever holds it last: the trainer deletes a
step's bodies once every rank fetched them; the orchestrator deletes the bodies of samples
and traces it drops without shipping; the env worker deletes the bodies of episodes it
kept but did not deliver (`PayloadHook.discard`).

With `rollout_transport.retain`, a pool's workers write their bodies as files into the
directory of one `RolloutStore`, which serves them for the whole pool. The trainer then
marks a step's bodies as trained instead of deleting them, and the store deletes them once
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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import msgspec
import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.trajectories import BranchSample, trace_to_samples
from prime_rl.transports.batch.materialize import (
    assert_token_arrays_aligned,
    fill_routing,
    make_dummy_batch,
    materialize_bin,
    mm_cut,
    pad_micro_batch,
    prepare_sample,
    slice_routed_experts,
    slice_sampler_support,
    truncate_mm_refs,
)
from prime_rl.transports.batch.types import (
    PAYLOAD_INFO_KEY,
    UNLABELED,
    UNVERSIONED,
    MicroBatch,
    MicroBatchPlan,
    RoutedExperts,
    SampleBody,
    SampleRef,
    SamplerSupport,
    TrainingSample,
)
from prime_rl.utils.logger import get_logger

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(SampleBody)


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
                trained = _trained_decoder.decode(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(204 if server.trained(trained) else 501)
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

    def trained(self, trained: Trained) -> bool:
        """Whether this server retains trained arrays: only a `RolloutStore` does."""
        return False


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

    def trained(self, trained: Trained) -> bool:
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
        return True


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


def _arrays(values: np.ndarray | None, dtype: type[np.generic]) -> bytes | None:
    return None if values is None else np.ascontiguousarray(values, dtype=dtype).tobytes()


def _version_counts(versions: np.ndarray, selected: np.ndarray) -> dict[str, int]:
    """How many of the ``selected`` tokens each policy version (see ``SampleBody.versions``) sampled."""
    values, counts = np.unique(versions[selected], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def encode_sample(sample: BranchSample) -> tuple[SampleBody, dict[str, Any]]:
    """A branch sample as the body an env worker keeps, and the summary the orchestrator
    decides with: its token count, which arrays it holds, and per loss component how many of
    its tokens each policy version sampled."""
    branch, num_tokens = sample.branch, len(sample.token_ids)
    support = None
    try:
        support = _encode_sampler_support(branch.sampler_support, num_tokens)
    except ValueError as error:
        # Keep the sample and its routing rows: a train run that needs the sampler
        # evidence stops at the train sink.
        get_logger().warning(f"Not keeping the sampling mask and top-k head of branch {branch.index}: {error}")
    body = SampleBody(
        token_ids=_arrays(sample.token_ids, np.int32),
        mask=_arrays(sample.mask, np.bool_),
        sampled=_arrays(sample.sampled, np.bool_),
        logprobs=_arrays(sample.logprobs, np.float32),
        versions=_arrays(sample.versions, np.int32),
        rl_weights=_arrays(sample.rl_weights, np.float32),
        ce_weights=_arrays(sample.ce_weights, np.float32),
        ref_kl_weights=_arrays(sample.ref_kl_weights, np.float32),
        ref_logprobs=_arrays(sample.ref_logprobs, np.float32),
        mm_token_type_ids=_arrays(sample.mm_token_type_ids, np.int32),
        mm_refs=sample.mm_refs,
        routed_experts=_encode_routed_experts(branch.routed_experts, num_tokens),
        sampler_support=support,
    )
    rl = sample.mask if sample.rl_weights is None else sample.mask & (sample.rl_weights != 0)
    loss = {"rl": _version_counts(sample.versions, rl)}
    for name, weights in (("ce", sample.ce_weights), ("ref_kl", sample.ref_kl_weights)):
        if weights is not None and weights.any():
            loss[name] = _version_counts(sample.versions, weights != 0)
    summary: dict[str, Any] = {
        "num_tokens": num_tokens,
        "fields": sorted(_fields(body)),
        "trainable": _version_counts(sample.versions, sample.mask),
        "loss": loss,
    }
    if body.routed_experts is not None:
        summary |= {"routed_shape": body.routed_experts.shape[1:], "routed_dtype": body.routed_experts.dtype}
    if sample.mm_refs is not None:
        summary["images"] = [[image.offset, image.length] for image in sample.mm_refs.images]
    return body, summary


def _fields(body: SampleBody) -> set[str]:
    """The optional training arrays a body holds."""
    fields = {name for name in ("routed_experts", "sampler_support") if getattr(body, name) is not None}
    if body.sampler_support is not None:
        if np.frombuffer(body.sampler_support.replayed, dtype=np.bool_).any():
            fields.add("sampler_support_replayed")
        if body.sampler_support.logprobs is not None:
            fields.add("sampler_support_logprobs")
    return fields


class PayloadHook:
    """`prime_rl.env.set_episode_hook` hook: keep a finished train episode's samples."""

    def __init__(self, bind_address: str | None = None, store: StoreWriter | None = None):
        self._hosts = payload_host(bind_address)
        self._server: PayloadServer | StoreWriter | None = store
        self._lock = threading.Lock()

    @property
    def server(self) -> PayloadServer | StoreWriter:
        """The pool's rollout store if it has one, else this worker's own server, started on
        first use: a pool broker installs the hook too but never keeps samples."""
        with self._lock:
            if self._server is None:
                self._server = PayloadServer(*self._hosts)
            return self._server

    def keep(self, episode: vf.Episode, action_loss_type: ActionLossType) -> None:
        """Keep the samples of the episode's trainable traces, their action tokens routed into
        ``action_loss_type``, and leave references to them in each trace's info. Every node's
        training arrays and loss streams go, so the final flush streams none of them."""
        stored: list[str] = []
        try:
            for _, trace in iter_trainable_traces([episode]):
                references: dict[str, dict[str, Any]] = {}
                for sample in trace_to_samples(trace):
                    stamp_loss_routing(sample, action_loss_type)
                    body, summary = encode_sample(sample)
                    key = f"{trace.id}-{sample.branch.index}"
                    self.server.put(key, _encoder.encode(body))
                    stored.append(key)
                    references[str(sample.branch.index)] = {"id": key, **summary}
                # Every trainable trace names its samples, even none: the orchestrator tells a
                # trace without samples from one whose worker kept nothing.
                trace.info[PAYLOAD_INFO_KEY] = {"url": self.server.url, "branches": references}
            for trace in episode.traces:
                for node in trace.nodes:
                    node.routed_experts = node.sampler_support = None
                    node.reference_logprobs = node.loss_weights = None
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
    """URLs of every sample an env worker kept for the trace."""
    payload = trace.info.get(PAYLOAD_INFO_KEY)
    return [f"{payload['url']}/{reference['id']}" for reference in payload["branches"].values()] if payload else []


def sample_url(ref: SampleRef) -> str:
    return f"{ref.url}/{ref.id}"


def sample_payload_urls(refs: Iterable[SampleRef]) -> list[str]:
    return [sample_url(ref) for ref in refs]


# Payload URLs are node-local addresses: never route them through an `HTTP_PROXY`.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _fetch(url: str, attempts: int = 3) -> SampleBody | None:
    """The body at `url`, or None if it is gone (404) or every attempt failed."""
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


def require_rollout_store(url: str) -> None:
    """Raise unless the env worker pool at `url` retains trained arrays. A pool without a
    rollout store would neither keep nor free the arrays marked on it."""
    request = urllib.request.Request(url, data=msgspec.msgpack.encode(Trained(0, [], -1)), method="POST")
    try:
        _opener.open(request, timeout=60).close()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"The env worker pool at {url} has no rollout store, which rollout_transport.retain needs: "
            "set store_dir in its config"
        ) from error


def wait_for_deletes() -> None:
    """Block until every queued release request has run."""
    _releaser.queue.join()


def build_sample(body: SampleBody, ref: SampleRef, plan: MicroBatchPlan) -> TrainingSample:
    """The first ``ref.num_tokens`` of a kept sample, with the orchestrator's decisions applied:
    the trace's advantage spread over the sampled tokens and its rl weight over the trainable
    ones, zero-advantage tokens out of the rl component, and tokens of a policy older than the
    step's ``min_version`` out of every component."""
    n = ref.num_tokens
    mask = np.frombuffer(body.mask, np.bool_)[:n].copy()
    sampled = np.frombuffer(body.sampled, np.bool_)[:n]
    versions = np.frombuffer(body.versions, np.int32)[:n]

    def stream(values: bytes | None) -> np.ndarray | None:
        return None if values is None else np.frombuffer(values, np.float32)[:n].copy()

    rl_weights, ce_weights, ref_kl_weights = (
        stream(body.rl_weights),
        stream(body.ce_weights),
        stream(body.ref_kl_weights),
    )
    if ref.rl_weight is not None:
        rl_weights = np.where(mask, np.float32(ref.rl_weight), np.float32(0.0))
    advantages = None
    if ref.advantage is not None:
        advantages = np.where(sampled, np.float32(ref.advantage), np.float32(0.0))
        pruned = mask & (advantages == 0.0)
        if rl_weights is not None:
            pruned &= rl_weights != 0.0
        if pruned.any():
            rl_weights = mask.astype(np.float32) if rl_weights is None else rl_weights
            rl_weights[pruned] = 0.0
    staleness = None
    if ref.policy_start is not None:
        versioned = versions != UNVERSIONED
        effective = np.where(versions == UNLABELED, ref.policy_start, versions)
        stale = versioned & (effective < plan.min_version)
        if stale.any():
            mask[stale] = False
            for weights in (rl_weights, ce_weights, ref_kl_weights):
                if weights is not None:
                    weights[stale] = 0.0
        staleness = np.where(versioned, np.maximum(0, plan.trained_version - effective), -1).astype(np.int16)
    return TrainingSample(
        token_ids=np.frombuffer(body.token_ids, np.int32)[:n],
        mask=mask,
        logprobs=np.frombuffer(body.logprobs, np.float32)[:n],
        temperatures=np.full(n, ref.temperature, dtype=np.float32),
        env_name=ref.env_name,
        ref_logprobs=None if body.ref_logprobs is None else np.frombuffer(body.ref_logprobs, np.float32)[:n],
        mm_refs=truncate_mm_refs(body.mm_refs, n),
        mm_token_type_ids=None
        if body.mm_token_type_ids is None
        else np.frombuffer(body.mm_token_type_ids, np.int32)[:n],
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=ref_kl_weights,
        advantages=advantages,
        trace_id=ref.trace_id,
        branch_index=ref.branch_index,
        staleness=staleness,
        routed_experts=None if body.routed_experts is None else slice_routed_experts(body.routed_experts, n),
        sampler_support=None if body.sampler_support is None else slice_sampler_support(body.sampler_support, n),
    )


def _placeholder(ref: SampleRef, token_ids: np.ndarray) -> TrainingSample:
    """A sample whose body could not be fetched, to keep its micro batch's shapes: tokens
    borrowed from the micro batch's other samples (identical tokens would route to the same
    MoE experts), nothing to train."""
    n = ref.num_tokens
    return TrainingSample(
        token_ids=np.resize(token_ids if len(token_ids) else np.zeros(1, np.int32), n).astype(np.int32),
        mask=np.zeros(n, dtype=np.bool_),
        logprobs=np.zeros(n, dtype=np.float32),
        temperatures=np.ones(n, dtype=np.float32),
        env_name=ref.env_name,
        trace_id=ref.trace_id,
        branch_index=ref.branch_index,
    )


def _multimodal_placeholder(ref: SampleRef, donors: list[TrainingSample]) -> TrainingSample | None:
    """A placeholder for a multimodal sample whose body could not be fetched: another fetched
    multimodal sample of this rank's step, cut to the missing sample's length without splitting an
    image, so the micro batch still runs the vision encoder the other data ranks run at this
    step. None if no fetched sample keeps an image within that length."""
    n = ref.num_tokens
    for donor in donors:
        images = [[image.offset, image.length] for image in donor.mm_refs.images]
        cut = mm_cut(images, len(donor.token_ids), n)
        mm_refs = truncate_mm_refs(donor.mm_refs, cut)
        if mm_refs is None:
            continue
        padding = np.zeros(n - cut, dtype=np.int32)
        return TrainingSample(
            token_ids=np.concatenate([donor.token_ids[:cut], padding]),
            mask=np.zeros(n, dtype=np.bool_),
            logprobs=np.zeros(n, dtype=np.float32),
            temperatures=np.ones(n, dtype=np.float32),
            env_name=ref.env_name,
            mm_refs=mm_refs,
            mm_token_type_ids=np.concatenate([donor.mm_token_type_ids[:cut], padding]),
            trace_id=ref.trace_id,
            branch_index=ref.branch_index,
        )
    return None


def _fill_missing_routing(samples: list[TrainingSample], refs: list[SampleRef]) -> None:
    """Routing rows for the placeholders of a micro batch that replays routing: the other
    samples' rows repeated, so a micro batch kept out of the loss still routes like real data."""
    routed = next((ref for ref in refs if "routed_experts" in ref.fields), None)
    if routed is None:
        return
    rows = b"".join(sample.routed_experts.data for sample in samples if sample.routed_experts is not None)
    for sample in samples:
        if sample.routed_experts is None:
            sample.routed_experts = fill_routing(len(sample.token_ids), routed.routed_shape, routed.routed_dtype, rows)


@dataclass
class Materialized:
    """A rank's micro batches of one step, built from its plans."""

    micro_batches: list[MicroBatch]
    failed: list[int]
    """Micro batches with a sample that could not be fetched: it holds a placeholder of the right
    shape, and the caller must keep the micro batch out of the loss."""
    unbuildable: list[int]
    """Failed multimodal micro batches left without an image: they cannot run the vision encoder
    the other data ranks run at this step, so the step cannot train."""
    urls: list[str]
    """The fetched samples, for `delete_payloads`."""


def materialize_micro_batches(plans: list[MicroBatchPlan], pool: ThreadPoolExecutor) -> Materialized:
    """Fetch the plans' sample bodies and build their micro batches. Never raises for a sample
    that could not be fetched: every data rank must reach the step's next collective."""
    urls = sorted({sample_url(ref) for plan in plans for ref in plan.samples})
    bodies = dict(zip(urls, pool.map(_fetch, urls)))
    built = [
        [
            build_sample(body, ref, plan) if (body := bodies[sample_url(ref)]) is not None else None
            for ref in plan.samples
        ]
        for plan in plans
    ]
    donors = [sample for samples in built for sample in samples if sample is not None and sample.mm_refs is not None]
    result = Materialized(micro_batches=[], failed=[], unbuildable=[], urls=urls)
    for index, (plan, samples) in enumerate(zip(plans, built, strict=True)):
        if any(sample is None for sample in samples):
            result.failed.append(index)
            token_ids = np.concatenate([s.token_ids for s in samples if s is not None] or [np.zeros(0, np.int32)])
            for position, ref in enumerate(plan.samples):
                if samples[position] is None:
                    placeholder = _multimodal_placeholder(ref, donors) if ref.images else None
                    samples[position] = placeholder or _placeholder(ref, token_ids)
            _fill_missing_routing(samples, plan.samples)
        micro_batch = pad_micro_batch(materialize_bin([prepare_sample(sample) for sample in samples]), plan.padding)
        if any(ref.images for ref in plan.samples) and micro_batch.mm_refs is None:
            result.unbuildable.append(index)
        if plan.dummy:
            micro_batch = make_dummy_batch(micro_batch)
        assert_token_arrays_aligned(micro_batch)
        result.micro_batches.append(micro_batch)
    if result.failed:
        get_logger().warning(f"Could not fetch the samples of {len(result.failed)} micro batches")
    return result
