"""Publish HF safetensors checkpoints from disk as ModelExpress weight versions, so that running engines can install
them (``POST /swap-checkpoint``) instead of restarting on new weights.

A trainer publishes the weights it trains; a checkpoint publisher publishes an exported checkpoint. Its ranks (one
process each, joined by a gloo group) split the checkpoint's tensors between them, read their share into page-locked
host memory and serve it from there: ModelExpress reads those buffers directly (``IN_PLACE``), so the publisher holds
one copy of the checkpoint. The tensors keep the checkpoint's names and layout, which is what vLLM's own loader reads.

The publisher keeps one trainer mesh for its lifetime. Engines stage the first version of a mesh while paused (it
records how the mesh's tensors map onto theirs) and every later one while they serve. It holds one version at a time:
to publish the next checkpoint, it marks the current version RELEASING (no new reads start), waits until the reads in
flight end, reads the new checkpoint into the same buffers and publishes it, numbered with its step.

ModelExpress publishes floating-point tensors only, so integer tensors stay out of a version and engines keep the ones
they loaded (see ``checkpoint_versions``). The version's uid carries their digest.

Rank 0 takes commands over HTTP: ``GET /status``, ``POST /publish {"model_dir", "step"}`` (blocks until the version is
READY) and ``POST /release`` (retire the version and exit; SIGTERM does the same).
"""

import ctypes
import json
import logging
import os
import queue
import resource
import signal
import struct
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.transports.weights.checkpoint_versions import checkpoint_version_uid, integer_tables_digest

logger = logging.getLogger("prime_rl.checkpoint_publisher")

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
PUBLISHED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
"""The wire dtypes ModelExpress publishes; a tensor is published in its checkpoint dtype."""

READ_CHUNK_BYTES = 256 << 20
PR_SET_PDEATHSIG = 1
READY_POLL_S = 0.1
RELEASE_RETRY_S = 1.0


@dataclass(frozen=True)
class TensorEntry:
    """Where a checkpoint holds one tensor: its file, dtype, shape and byte range in that file."""

    file: Path
    dtype: torch.dtype
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def read_checkpoint(model_dir: Path) -> dict[str, TensorEntry]:
    """Every tensor of an HF safetensors checkpoint, from its files' headers."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")
    entries = {}
    for name in files:
        path = model_dir / name
        with path.open("rb") as file:
            (header_size,) = struct.unpack("<Q", file.read(8))
            header = json.loads(file.read(header_size))
        header.pop("__metadata__", None)
        data_start = 8 + header_size
        for tensor, info in header.items():
            begin, end = info["data_offsets"]
            entries[tensor] = TensorEntry(
                path, SAFETENSORS_DTYPES[info["dtype"]], tuple(info["shape"]), data_start + begin, data_start + end
            )
    return entries


def published_tensors(entries: dict[str, TensorEntry]) -> dict[str, TensorEntry]:
    """The tensors a version carries: the floating-point ones. Fails on floating dtypes ModelExpress cannot publish."""
    published = {name: entry for name, entry in entries.items() if entry.dtype.is_floating_point}
    if unsupported := sorted(name for name, entry in published.items() if entry.dtype not in PUBLISHED_DTYPES):
        raise ValueError(
            f"{len(unsupported)} tensors have a dtype ModelExpress cannot publish (e.g. {unsupported[0]}: "
            f"{published[unsupported[0]].dtype}); publish a bf16 export"
        )
    return published


def split_tensors(entries: dict[str, TensorEntry], ranks: int) -> list[list[str]]:
    """Split tensors between ranks: runs in file order, of about equal bytes, so each rank reads long ranges."""
    names = sorted(entries, key=lambda name: (entries[name].file, entries[name].begin))
    if len(names) < ranks:
        raise ValueError(f"{len(names)} tensors cannot be split between {ranks} publisher ranks")
    total = max(sum(entries[name].nbytes for name in names), 1)
    parts: list[list[str]] = [[] for _ in range(ranks)]
    before = 0
    for index, name in enumerate(names):
        size = entries[name].nbytes
        # Leave at least one tensor for every later rank
        rank = min((before + size // 2) * ranks // total, ranks - 1, index)
        rank = max(rank, ranks - (len(names) - index))
        parts[rank].append(name)
        before += size
    return parts


class HostTensors:
    """One rank's share of a checkpoint in host memory: a page-locked buffer per file region it reads, and every tensor
    a view into it. The buffers stay put for the publisher's lifetime, since ModelExpress serves them in place."""

    def __init__(self, entries: dict[str, TensorEntry], names: list[str], pin: bool):
        self.tensors: dict[str, torch.Tensor] = {}
        self._bytes: dict[str, torch.Tensor] = {}
        self.buffers: list[torch.Tensor] = []
        by_file = defaultdict(list)
        for name in names:
            by_file[entries[name].file].append(name)
        for file_names in by_file.values():
            low = min(entries[name].begin for name in file_names)
            high = max(entries[name].end for name in file_names)
            buffer = self._allocate(high - low, pin)
            for name in file_names:
                entry = entries[name]
                raw = buffer[entry.begin - low : entry.end - low]
                if (entry.begin - low) % entry.dtype.itemsize:
                    raw = self._allocate(entry.nbytes, pin)
                self._bytes[name] = raw
                self.tensors[name] = raw.view(entry.dtype).view(entry.shape)

    def _allocate(self, size: int, pin: bool) -> torch.Tensor:
        buffer = torch.empty(max(size, 1), dtype=torch.uint8)
        if pin:
            # Pin exactly these pages. torch's pinned allocator rounds every allocation up to a power of two.
            error = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel(), 0)
            if int(error) != 0:
                raise RuntimeError(f"cudaHostRegister of {buffer.numel()} bytes failed: {error}")
        self.buffers.append(buffer)
        return buffer[:size]

    @property
    def nbytes(self) -> int:
        return sum(buffer.numel() for buffer in self.buffers)

    def check_layout(self, entries: dict[str, TensorEntry]) -> list[str]:
        """Why ``entries`` cannot be read into these buffers: tensors that are missing or changed shape or dtype."""
        problems = []
        for name, tensor in self.tensors.items():
            entry = entries.get(name)
            if entry is None:
                problems.append(f"{name}: missing")
            elif entry.dtype != tensor.dtype or entry.shape != tuple(tensor.shape):
                problems.append(f"{name}: {entry.dtype}{list(entry.shape)} != {tensor.dtype}{list(tensor.shape)}")
        return problems

    def read(self, entries: dict[str, TensorEntry], threads: int) -> int:
        """Read every tensor of this share from ``entries``' files into its buffer; returns the bytes read."""
        chunks = [
            (entries[name].file, entries[name].begin + offset, raw[offset : offset + READ_CHUNK_BYTES])
            for name, raw in self._bytes.items()
            for offset in range(0, raw.numel(), READ_CHUNK_BYTES)
        ]
        descriptors = {path: os.open(path, os.O_RDONLY) for path in {path for path, _, _ in chunks}}
        try:

            def read_chunk(chunk) -> int:
                path, offset, raw = chunk
                view = memoryview(raw.numpy())
                done = 0
                while done < len(view):
                    count = os.preadv(descriptors[path], [view[done:]], offset + done)
                    if count == 0:
                        raise EOFError(f"{path} ended at byte {offset + done}")
                    done += count
                return done

            with ThreadPoolExecutor(max_workers=threads) as pool:
                return sum(pool.map(read_chunk, chunks))
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)


def read_integer_tables(entries: dict[str, TensorEntry]) -> list[torch.Tensor]:
    """A checkpoint's integer tensors, which a version leaves out."""
    tables = []
    for entry in entries.values():
        if entry.dtype.is_floating_point:
            continue
        with entry.file.open("rb") as file:
            file.seek(entry.begin)
            data = bytearray(file.read(entry.nbytes))
        tables.append(torch.frombuffer(data, dtype=entry.dtype).view(entry.shape) if data else torch.empty(entry.shape))
    return tables


@dataclass
class PublisherConfig:
    model_dir: Path
    """The first checkpoint to publish. Every later one must hold the same tensors (names, dtypes and shapes)."""

    step: int
    """The first checkpoint's step: the version number engines label the tokens they sample with."""

    model_name: str
    """ModelExpress model name to publish under: the engines' ``weight_broadcast.model_name`` (by default their
    ``vllm.model``)."""

    mx_host: str = "localhost"
    mx_port: int = 8001
    ranks: int = 1
    """Publisher processes. Each reads, holds and registers its share of the checkpoint."""

    read_threads: int = 16
    """Reader threads per rank."""

    pin: bool = True
    """Page-lock the host buffers with CUDA (needs a GPU visible to the publisher). Registering them with NIXL pins
    them as well; this makes the cost explicit at load and keeps the memory pinned in between versions."""

    control_host: str = "0.0.0.0"
    control_port: int = 8090
    metadata_port: int = 15555
    """First NIXL metadata port; rank r listens on ``metadata_port + r``."""

    worker_port: int = 16555
    """First ModelExpress worker gRPC port; rank r listens on ``worker_port + r``."""

    release_timeout: float = 1800.0
    """Seconds to wait for reads of the version being retired before giving up on publishing the next one."""

    ready_timeout: float = 600.0
    serve_modelexpress: Path | None = None
    """Start a ModelExpress server (and its Valkey) on ``mx_port`` beside the publisher, with its state and logs in
    this directory, instead of publishing to one at ``mx_host``. It stops with the publisher."""

    dist_port: int = 0
    """Port of the ranks' gloo group; 0 picks a free one."""

    session: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class Command:
    name: str
    """``publish`` or ``release``."""

    model_dir: str | None = None
    step: int | None = None


class Publisher:
    """One rank of the publisher. Every rank runs the same commands; rank 0 also talks to the control plane."""

    def __init__(self, config: PublisherConfig, rank: int, world: int):
        from modelexpress_rl import (
            FSDPTrainerContext,
            ModelExpressControlClient,
            ModelExpressTrainerClient,
            ModelExpressTrainerConfig,
            TrainerStagingMode,
            WeightPayloadFormat,
        )

        self.config = config
        self.rank = rank
        self.world = world
        self.server_url = f"{config.mx_host}:{config.mx_port}"
        self.version: str | None = None
        """The uid of the version this rank serves (READY) or retires (RELEASING)."""
        self.step: int | None = None
        self.model_dir: str | None = None
        self.state = "starting"
        self.timings: dict[str, float] = {}

        start = time.perf_counter()
        entries = published_tensors(read_checkpoint(config.model_dir))
        names = split_tensors(entries, world)[rank]
        self.host = HostTensors(entries, names, config.pin)
        allocated = time.perf_counter()
        os.environ["MX_METADATA_PORT"] = str(config.metadata_port + rank)
        os.environ["MX_WORKER_GRPC_PORT"] = str(config.worker_port + rank)
        overrides = {name: tensor.dtype for name, tensor in self.host.tensors.items() if tensor.dtype != torch.bfloat16}
        self.trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(wire_dtype_overrides=overrides),
                model_name=config.model_name,
                device_id=0,
                server_url=self.server_url,
                worker_id=f"ckpt-publisher-{config.session}-{rank}",
                staging_mode=TrainerStagingMode.IN_PLACE,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
        binding = self.trainer.bind_tensors(self.host.tensors)
        workers = [None] * world
        dist.all_gather_object(workers, (self.trainer.worker_id, binding.logical_shard_id, binding.metadata_endpoint))
        self.control = None
        self.mesh_id = None
        if rank == 0:
            from modelexpress_rl import TrainerTensorsMetadata

            self.control = ModelExpressControlClient.connect(server_url=self.server_url)
            mesh = self.control.create_trainer_mesh(
                model_name=config.model_name,
                idempotency_key=uuid.uuid4().hex,
                workers={
                    worker_id: TrainerTensorsMetadata(logical_shard_id=shard, metadata_endpoint=endpoint)
                    for worker_id, shard, endpoint in workers
                },
            )
            self.mesh_id = mesh.mesh_id
        self.timings = {"allocate_s": allocated - start, "init_s": time.perf_counter() - start}
        logger.info(
            f"Rank {rank}: {len(self.host.tensors)} tensors in {self.host.nbytes / 2**30:.2f} GiB of "
            f"{'pinned ' if config.pin else ''}host memory, allocated in {allocated - start:.1f}s"
        )

    def run(self, command: Command) -> dict:
        if command.name == "publish":
            return self.publish(Path(command.model_dir), command.step)
        if command.name == "release":
            return {"released": self.retire()}
        raise ValueError(f"Unknown command {command.name}")

    def publish(self, model_dir: Path, step: int) -> dict:
        """Publish the checkpoint at ``model_dir`` as the version of ``step``, retiring the current one."""
        from modelexpress_rl import WeightPayloadFormat, WeightVersionRef

        timings = {}
        start = time.perf_counter()
        entries = read_checkpoint(model_dir)
        problems = self.host.check_layout(entries)
        if self.rank == 0:
            published = published_tensors(entries)
            bound = set().union(*self._gather(set(self.host.tensors)))
            problems += [f"{name}: not published before" for name in sorted(published.keys() - bound)]
        else:
            self._gather(set(self.host.tensors))
        problems = [problem for rank_problems in self._gather(problems) for problem in rank_problems]
        if problems:
            raise ValueError(
                f"{model_dir} does not match the published checkpoint's tensors ({len(problems)} differ: "
                f"{'; '.join(problems[:5])}); start a new publisher for it"
            )
        timings["index_s"] = time.perf_counter() - start

        mark = time.perf_counter()
        retired = self.retire()
        timings["release_s"] = time.perf_counter() - mark

        mark = time.perf_counter()
        self.state = "loading"
        read = self.host.read(entries, self.config.read_threads)
        timings["read_s"] = time.perf_counter() - mark
        timings["read_gib"] = read / 2**30

        uid = None
        if self.rank == 0:
            uid = checkpoint_version_uid(step, integer_tables_digest(read_integer_tables(entries)))
            self.control.create_weight_version(
                model_name=self.config.model_name,
                idempotency_key=uuid.uuid4().hex,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                trainer_mesh_id=self.mesh_id,
                version_number=step,
                uid=uid,
            )
        uid = self._broadcast(uid)
        mark = time.perf_counter()
        # The first publish also registers the buffers with NIXL
        self.trainer.publish_version(version=WeightVersionRef(uid))
        self.version, self.step, self.model_dir = uid, step, str(model_dir)
        timings["publish_s"] = time.perf_counter() - mark
        dist.barrier()
        mark = time.perf_counter()
        if self.rank == 0:
            self._wait_ready(uid)
        timings["ready_s"] = time.perf_counter() - mark
        timings["total_s"] = time.perf_counter() - start
        self.state = "ready"
        self.timings.update(timings)
        logger.info(f"Rank {self.rank}: published step {step} as {uid} ({retired=}): {timings}")
        return {"step": step, "version_uid": uid, "retired": retired, "timings": timings}

    def retire(self) -> str | None:
        """Retire the current version: no new reads start, and the buffers are free once the reads in flight end."""
        import grpc
        from modelexpress_rl import WeightVersionRef

        if self.version is None:
            return None
        uid = self.version
        self.state = "releasing"
        if self.rank == 0:
            self.control.delete_weight_version(uid)
        dist.barrier()
        deadline = time.monotonic() + self.config.release_timeout
        while True:
            try:
                # Retries for its RPC timeout while a reader holds a lease
                self.trainer.release_version(version=WeightVersionRef(uid))
                break
            except grpc.RpcError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Reads of version {uid} did not end within {self.config.release_timeout}s: {error.details()}"
                    ) from error
                logger.info(f"Rank {self.rank}: version {uid} still has readers, waiting ({error.details()})")
                time.sleep(RELEASE_RETRY_S)
        dist.barrier()
        self.version = None
        self.state = "idle"
        return uid

    def _wait_ready(self, uid: str) -> None:
        from modelexpress_rl import WeightVersionState

        deadline = time.monotonic() + self.config.ready_timeout
        while (state := self.control.get_weight_version(uid).state) is not WeightVersionState.READY:
            if state is WeightVersionState.RELEASING or time.monotonic() >= deadline:
                raise RuntimeError(f"Version {uid} did not become READY (state {state.value})")
            time.sleep(READY_POLL_S)

    def _gather(self, value) -> list:
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def _broadcast(self, value):
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def status(self) -> dict:
        return {
            "model_name": self.config.model_name,
            "state": self.state,
            "step": self.step,
            "version_uid": self.version,
            "model_dir": self.model_dir,
            "mesh_id": self.mesh_id,
            "ranks": self.world,
            "host_gib": self.host.nbytes / 2**30,
            "timings": self.timings,
        }


def init_rank(config: PublisherConfig, rank: int, world: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{config.dist_port}",
        rank=rank,
        world_size=world,
        # Ranks other than 0 wait in a collective until the next command
        timeout=timedelta(days=365),
    )


def serve_rank(config: PublisherConfig, rank: int, world: int) -> None:
    """A rank other than 0: run every command rank 0 broadcasts, until it releases."""
    # Exit with rank 0 rather than wait in a collective forever
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    init_rank(config, rank, world)
    publisher = Publisher(config, rank, world)
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0)
        command: Command = box[0]
        try:
            result = publisher.run(command)
        except Exception as error:
            logger.exception(f"Rank {rank}: {command.name} failed")
            result = {"error": f"rank {rank}: {error!r}"}
        dist.all_gather_object([None] * world, result)
        if command.name == "release":
            break
    publisher.trainer.close()
    dist.destroy_process_group()


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, commands: "queue.Queue[tuple[Command, Future]]", status):
        super().__init__(address, ControlHandler)
        self.commands = commands
        self.status = status
        self.busy = threading.Lock()


class ControlHandler(BaseHTTPRequestHandler):
    server: ControlServer

    def _reply(self, status: HTTPStatus, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path != "/status":
            return self._reply(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
        self._reply(HTTPStatus.OK, self.server.status())

    def do_POST(self) -> None:
        if self.path not in ("/publish", "/release"):
            return self._reply(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/publish":
                command = Command("publish", model_dir=str(body["model_dir"]), step=int(body["step"]))
            else:
                command = Command("release")
        except (ValueError, KeyError, TypeError) as error:
            return self._reply(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {error!r}"})
        if not self.server.busy.acquire(blocking=False):
            return self._reply(HTTPStatus.CONFLICT, {"error": "the publisher is running another command"})
        try:
            future: Future = Future()
            self.server.commands.put((command, future))
            result = future.result()
        finally:
            self.server.busy.release()
        if "error" in result:
            return self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, result)
        self._reply(HTTPStatus.OK, result)

    def log_message(self, format, *args) -> None:
        logger.info(f"control: {format % args}")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def run_publisher(config: PublisherConfig) -> None:
    """Start the publisher's ranks, publish ``config.model_dir`` and serve commands until released."""
    if config.dist_port == 0:
        from prime_rl.utils.utils import get_free_port

        config.dist_port = get_free_port()
    modelexpress = None
    if config.serve_modelexpress is not None:
        from prime_rl.entrypoints.modelexpress_server import start_modelexpress
        from prime_rl.utils.process import ProcessGroup

        modelexpress = ProcessGroup()
        start_modelexpress(modelexpress, config.mx_port, config.serve_modelexpress, dict(os.environ))
        config.mx_host = "127.0.0.1"
    context = mp.get_context("spawn")
    children = [
        context.Process(target=serve_rank, args=(config, rank, config.ranks), name=f"publisher-{rank}", daemon=True)
        for rank in range(1, config.ranks)
    ]
    for child in children:
        child.start()
    init_rank(config, 0, config.ranks)
    publisher = Publisher(config, 0, config.ranks)

    commands: "queue.Queue[tuple[Command, Future]]" = queue.Queue()
    server = ControlServer(
        (config.control_host, config.control_port),
        commands,
        lambda: {**publisher.status(), "rank0_max_rss_gib": rss_gib()},
    )
    threading.Thread(target=server.serve_forever, name="publisher-control", daemon=True).start()
    logger.info(f"Publisher control on http://{config.control_host}:{config.control_port}")

    def release_on_signal(signum, frame) -> None:
        logger.info(f"Received signal {signum}: releasing the version and exiting")
        commands.put((Command("release"), Future()))

    signal.signal(signal.SIGTERM, release_on_signal)
    signal.signal(signal.SIGINT, release_on_signal)
    commands.put((Command("publish", model_dir=str(config.model_dir), step=config.step), Future()))

    while True:
        try:
            command, future = commands.get(timeout=1.0)
        except queue.Empty:
            if dead := [child.name for child in children if not child.is_alive()]:
                logger.error(f"Publisher ranks {dead} exited; retiring the version and exiting")
                if publisher.version is not None:
                    publisher.control.delete_weight_version(publisher.version)
                os._exit(1)
            continue
        dist.broadcast_object_list([command], src=0)
        try:
            result = publisher.run(command)
        except Exception as error:
            logger.exception(f"Rank 0: {command.name} failed")
            result = {"error": f"rank 0: {error!r}"}
        results = [None] * config.ranks
        dist.all_gather_object(results, result)
        if errors := [result["error"] for result in results if "error" in result]:
            result = {"error": "; ".join(errors), "status": publisher.status()}
        future.set_result(result)
        if command.name == "release":
            break
        if "error" in result and command.model_dir == str(config.model_dir) and publisher.step is None:
            # The first checkpoint did not publish: nothing to serve
            logger.error(f"Publishing {config.model_dir} failed: {result['error']}")
            break
    server.shutdown()
    for child in children:
        child.join(timeout=60)
    publisher.trainer.close()
    if publisher.control is not None:
        publisher.control.delete_trainer_mesh(publisher.mesh_id)
        publisher.control.close()
    dist.destroy_process_group()
    if modelexpress is not None:
        modelexpress.cleanup()
