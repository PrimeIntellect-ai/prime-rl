from datetime import timedelta
from pathlib import Path
from time import sleep, time

import torch.distributed as dist

from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender
from prime_rl.transport.config import TCPStoreTransportConfig
from prime_rl.transport.types import MicroBatch

# Key prefixes for TCPStore
_MICRO_BATCH_KEY_PREFIX = "micro_batch"
_READY_KEY_PREFIX = "ready"
_STEP_KEY_PREFIX = "step"

# Polling interval for can_receive checks
_POLL_INTERVAL_SECONDS = 0.01


def _get_micro_batch_key(step: int, data_rank: int) -> str:
    """Generate the key for a micro batch in the TCPStore."""
    return f"{_MICRO_BATCH_KEY_PREFIX}_step_{step}_rank_{data_rank}"


def _get_ready_key(data_rank: int) -> str:
    """Generate the key for a ready signal in the TCPStore."""
    return f"{_READY_KEY_PREFIX}_rank_{data_rank}"


def _get_step_key() -> str:
    """Generate the key for the current step in the TCPStore."""
    return f"{_STEP_KEY_PREFIX}_current"


class TCPStoreMicroBatchSender(MicroBatchSender):
    """TCPStore-based micro batch sender that broadcasts micro batches via torch.distributed.TCPStore.

    The master creates the TCPStore and sets keys for each data rank.
    Workers read from the TCPStore based on their data rank.
    """

    def __init__(
        self,
        output_dir: Path,
        data_world_size: int,
        current_step: int,
        transport: TCPStoreTransportConfig,
    ):
        super().__init__(output_dir, data_world_size)
        self._current_step = current_step
        self._ready = False
        self._transport = transport

        # Create TCPStore as master (is_master=True)
        self._store = dist.TCPStore(
            host_name=transport.host,
            port=transport.port,
            world_size=data_world_size + 1,  # +1 for the sender
            is_master=True,
            timeout=timedelta(seconds=transport.timeout_seconds),
        )

        # Initialize the current step in the store
        self._store.set(_get_step_key(), str(self._current_step).encode("utf-8"))

        self.logger.info(
            f"TCPStore micro batch sender initialized: host={transport.host} "
            f"port={transport.port} data_world_size={data_world_size}"
        )

    def _wait_for_ready(self) -> None:
        """Wait for all data ranks to signal they are ready."""
        if self._ready:
            return

        self.logger.debug(f"Waiting for {self.data_world_size} ranks to be ready")
        ready_ranks: set[int] = set()

        while len(ready_ranks) < self.data_world_size:
            for rank in range(self.data_world_size):
                if rank in ready_ranks:
                    continue
                ready_key = _get_ready_key(rank)
                try:
                    # Check if the key exists
                    num_keys = self._store.num_keys()
                    if num_keys > 0:
                        # Try to get the key - will raise if not found
                        try:
                            value = self._store.get(ready_key)
                            if value == b"1":
                                ready_ranks.add(rank)
                                self.logger.debug(f"Rank {rank} is ready")
                        except Exception:
                            pass
                except Exception:
                    pass
            if len(ready_ranks) < self.data_world_size:
                sleep(_POLL_INTERVAL_SECONDS)

        self.logger.debug(f"All {self.data_world_size} ranks ready, starting broadcast")
        self._ready = True

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers via TCPStore."""
        # Validation
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        # Wait for all receivers to be ready before the first send
        self._wait_for_ready()

        self.logger.debug(f"Sending micro batch grid for step {self._current_step}")

        # Store micro batches for each data rank
        for data_rank in range(self.data_world_size):
            buffer = self.encoder.encode(micro_batch_grid[data_rank])
            key = _get_micro_batch_key(self._current_step, data_rank)
            self._store.set(key, buffer)

        # Update the current step after all batches are written
        self._store.set(_get_step_key(), str(self._current_step).encode("utf-8"))

        self._current_step += 1

    def close(self) -> None:
        """Clean up resources."""
        # TCPStore doesn't have an explicit close method, but we log cleanup
        self.logger.info("TCPStore micro batch sender closed")


class TCPStoreMicroBatchReceiver(MicroBatchReceiver):
    """TCPStore-based micro batch receiver that receives micro batches via torch.distributed.TCPStore.

    Each receiver connects to the TCPStore as a client and reads micro batches
    for its specific data rank.
    """

    def __init__(
        self,
        output_dir: Path,
        data_rank: int,
        current_step: int,
        transport: TCPStoreTransportConfig,
    ):
        super().__init__(output_dir, data_rank)
        self._current_step = current_step
        self._transport = transport

        # Connect to TCPStore as client (is_master=False)
        self._store = dist.TCPStore(
            host_name=transport.host,
            port=transport.port,
            is_master=False,
            timeout=timedelta(seconds=transport.timeout_seconds),
        )

        # Signal that we are ready
        ready_key = _get_ready_key(data_rank)
        self._store.set(ready_key, b"1")

        self.logger.info(
            f"TCPStore micro batch receiver initialized: host={transport.host} "
            f"port={transport.port} data_rank={data_rank}"
        )

    def _get_current_key(self) -> str:
        """Get the key for the current step and rank."""
        return _get_micro_batch_key(self._current_step, self.data_rank)

    def wait(self) -> None:
        """Wait for the micro batch to be available in the TCPStore."""
        key = self._get_current_key()
        start_time = time()
        timeout = self._transport.wait_timeout_seconds

        self.logger.debug(f"Waiting for key {key}")

        while True:
            try:
                # Try to wait for the key using TCPStore's wait method
                self._store.wait([key], timedelta(seconds=min(1.0, timeout)))
                self.logger.debug(f"Key {key} is available")
                return
            except Exception:
                elapsed = time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Timeout waiting for micro batch key {key} after {elapsed:.1f}s")
                # Key not yet available, continue waiting
                sleep(_POLL_INTERVAL_SECONDS)

    def can_receive(self) -> bool:
        """Check if a micro batch is available in the TCPStore."""
        key = self._get_current_key()
        try:
            # Attempt to get the key - if it exists, we can receive
            self._store.get(key)
            return True
        except Exception:
            return False

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the TCPStore."""
        key = self._get_current_key()

        try:
            buffer = self._store.get(key)
        except Exception as e:
            raise RuntimeError(f"Failed to get micro batch from TCPStore for key {key}: {e}")

        micro_batches: list[MicroBatch] = self.decoder.decode(buffer)
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self._current_step}")

        self._current_step += 1
        return micro_batches

    def close(self) -> None:
        """Clean up resources."""
        # TCPStore doesn't have an explicit close method, but we log cleanup
        self.logger.info("TCPStore micro batch receiver closed")
