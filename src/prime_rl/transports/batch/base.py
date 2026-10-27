from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import msgspec

from prime_rl.transports.batch.journal import BatchJournal
from prime_rl.transports.batch.types import MicroBatch, encode_array
from prime_rl.utils.logger import get_logger


class BatchSender(ABC):
    """Base class for sending micro batches from the orchestrator to the train workers.

    Trainer ranks pull: a step goes out only once every trainer rank requested it, so the
    trainer only ever holds the batch it is training on, and the last step sent is always the
    one the trainer is on."""

    def __init__(self, output_dir: Path, data_world_size: int):
        self.logger = get_logger()
        # Per-token streams are numpy arrays; each goes out as its raw bytes.
        self.encoder = msgspec.msgpack.Encoder(enc_hook=encode_array)
        self.output_dir = output_dir
        self.data_world_size = data_world_size
        self.journal: BatchJournal | None = None
        self.journal_session = uuid4().hex
        self.journal_write_time = 0.0

    def encode_batch(self, micro_batch_grid: list[list[MicroBatch]], step: int) -> list[bytes]:
        if self.journal is None:
            return [self.encode(micro_batches) for micro_batches in micro_batch_grid]
        start = perf_counter()
        batch_id = f"{self.journal_session}:{step}"
        grid = []
        for micro_batches in micro_batch_grid:
            for micro_batch in micro_batches:
                if micro_batch.mm_refs is not None:
                    raise ValueError("Batch journaling does not yet retain external multimodal inputs")
            batches = [msgspec.structs.replace(mb, journal_batch_id=batch_id) for mb in micro_batches]
            grid.append(batches)
        batches = [mb for rank in grid for mb in rank]
        urls = []
        if any(mb.payloads is not None for mb in batches):
            from prime_rl.transports.payload import delete_payloads, resolve_payloads

            with ThreadPoolExecutor(32, thread_name_prefix="journal-payload") as pool:
                failed, urls = resolve_payloads(batches, pool)
            if failed:
                raise RuntimeError("Cannot journal a batch with missing training payloads")
        payloads = [self.encode(rank) for rank in grid]
        self.journal.write(batch_id, payloads)
        if urls:
            # The committed bytes retain the arrays independently of the env servers.
            delete_payloads(urls)
        self.journal_write_time = perf_counter() - start
        return payloads

    @abstractmethod
    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers once every trainer rank requested the step.
        Returns the seconds spent handing the step to the transport, excluding the wait."""
        pass

    def encode(self, micro_batches: list[MicroBatch]) -> bytes:
        """Byte-identical to ``self.encoder.encode(micro_batches)``, but encodes one
        micro batch at a time so a worker thread releases the GIL between them."""
        n = len(micro_batches)
        if n <= 1:
            return self.encoder.encode(micro_batches)
        if n < 16:
            header = bytes([0x90 | n])
        elif n < 2**16:
            header = b"\xdc" + n.to_bytes(2, "big")
        else:
            header = b"\xdd" + n.to_bytes(4, "big")
        with BytesIO() as payload:
            payload.write(header)
            for micro_batch in micro_batches:
                payload.write(self.encoder.encode(micro_batch))
            return payload.getvalue()

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class BatchReceiver(ABC):
    """Base class for receiving micro batches from the orchestrator."""

    def __init__(self, output_dir: Path, data_rank: int, rank: int, world_size: int):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.data_rank = data_rank
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def wait(self) -> None:
        """Request the current step from the orchestrator and block until it is available."""
        pass

    @abstractmethod
    def receive(self) -> list[MicroBatch]:
        """Receive the micro batches ``wait`` requested."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
