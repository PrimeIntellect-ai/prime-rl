import copy
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import AsyncFileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


class AsyncFileSystemWeightBroadcast(WeightBroadcast):
    """Non-blocking filesystem weight broadcast.

    The FSDP gather runs synchronously (all ranks must participate), but
    serialization and disk I/O run in a background thread so the trainer
    can continue to the next step immediately.
    """

    def __init__(
        self, output_dir: Path, config: AsyncFileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        super().__init__(output_dir, lora_config)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded if lora_config is None else False
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight_broadcast")
        self._pending: Future | None = None
        self._last_broadcast_time: float = 0.0
        self.logger.info(
            f"Async filesystem broadcast initialized (save_format={config.save_format}, "
            f"save_sharded={self.save_sharded})"
        )

    def _wait_for_pending(self) -> float:
        """Block until the previous background write completes. Returns wait time in seconds."""
        wait_time = 0.0
        if self._pending is not None and not self._pending.done():
            wait_start = time.perf_counter()
            self._pending.result()
            wait_time = time.perf_counter() - wait_start
            self.logger.info(f"[BENCH] async: waited {wait_time:.3f}s for previous write to finish")
        if self._pending is not None and self._pending.done():
            # Re-raise any exception from the background thread
            self._pending.result()
        self._pending = None
        return wait_time

    def _write_and_notify(
        self,
        state_dicts: dict[int, dict],
        run_metadata: dict[int, tuple[Path, int]],
        adapter_only: bool,
        lora_configs: dict[int, tuple[int, float, float]] | None,
        model: nn.Module | None,
        cuda_event: torch.cuda.Event | None = None,
    ) -> None:
        """Runs in background thread: serialize weights to disk and write STABLE marker.

        Args:
            state_dicts: {idx: state_dict} per run, each owned exclusively by this thread.
            run_metadata: {idx: (run_dir, progress_step)} captured on the main thread.
            lora_configs: {idx: (rank, alpha, dropout)} for adapter runs, None otherwise.
            model: Model reference for save_lora_config. Only reads immutable module
                structure and config metadata — safe while trainer mutates parameter data.
            cuda_event: If provided, synchronize before reading tensor data to
                ensure non-blocking GPU->CPU DMA transfers have completed.
        """
        write_start = time.perf_counter()

        if cuda_event is not None:
            cuda_event.synchronize()
            event_sync_time = time.perf_counter() - write_start
            self.logger.info(f"[BENCH] async: CUDA event sync={event_sync_time:.3f}s (DMA drain)")

        for idx, state_dict in state_dicts.items():
            try:
                run_dir, progress_step = run_metadata[idx]
                save_dir = get_step_path(get_broadcast_dir(run_dir), progress_step)
                save_dir.mkdir(parents=True, exist_ok=True)

                self.logger.debug(f"[async] Saving weights for run {idx} to {save_dir}")
                save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)

                if adapter_only and lora_configs is not None and model is not None and idx in lora_configs:
                    rank, alpha, dropout = lora_configs[idx]
                    save_lora_config(model, save_dir, rank=rank, alpha=alpha, dropout=dropout)

                self._notify_orchestrator(save_dir)

                # Avoid zombie run directories that get recreated by mkdir
                # after the orchestrator deletes them during broadcast
                if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                    shutil.rmtree(run_dir)
            except FileNotFoundError:
                self.logger.warning(f"[async] Run {idx} directory deleted during broadcast, skipping")
            except Exception as e:
                self.logger.error(f"[async] Error broadcasting weights for run {idx}: {e}")

        self._last_broadcast_time = time.perf_counter() - write_start
        self.logger.info(f"[BENCH] async background write completed in {self._last_broadcast_time:.3f}s")

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights: gather synchronously, write asynchronously."""
        self.logger.debug(f"Starting async weight broadcast at step {step}")
        start_time = time.perf_counter()

        # Block until any in-flight background write finishes.
        # This provides backpressure: if the disk write from step N is slower
        # than one training step, we block here at step N+1 (no worse than sync).
        pending_wait = self._wait_for_pending()

        adapter_only = self.lora_config is not None
        cuda_event = None

        # Phase 1: FSDP gather (synchronous — all ranks must participate).
        # With non_blocking=True the per-tensor GPU->CPU DMAs are queued
        # into pinned memory without blocking. A CUDA event recorded
        # afterwards lets the background thread wait for the DMA to finish.
        if not adapter_only:
            state_dict = gather_weights_on_master(model, is_master=self.world.is_master, non_blocking=True)
            if self.world.is_master and torch.cuda.is_available():
                cuda_event = torch.cuda.Event()
                cuda_event.record()

            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_to_hf(state_dict)
            else:
                from transformers.core_model_loading import revert_weight_conversion

                state_dict = revert_weight_conversion(model, state_dict)

        gather_time = time.perf_counter() - start_time
        self.logger.debug(f"Gather completed in {gather_time:.2f}s")

        # Phase 2: Prepare per-run state dicts (synchronous for DTensor ops).
        ready_idxs = list(self.multi_run_manager.ready_to_update_idxs)
        prepared: dict[int, dict] = {}
        lora_configs: dict[int, tuple[int, float, float]] | None = None if not adapter_only else {}

        for idx in ready_idxs:
            if adapter_only:
                run_state_dict = self.multi_run_manager.get_state_dict_for_run(idx)
                for key, value in run_state_dict.items():
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    if self.world.is_master:
                        pinned = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                        pinned.copy_(value, non_blocking=True)
                        run_state_dict[key] = pinned
                if self.world.is_master:
                    prepared[idx] = run_state_dict
                    if torch.cuda.is_available():
                        cuda_event = torch.cuda.Event()
                        cuda_event.record()
                    if lora_configs is not None:
                        orch_lora = self.multi_run_manager.config[idx].model.lora
                        lora_configs[idx] = (orch_lora.rank, orch_lora.alpha, self.lora_config.dropout)
            else:
                if self.world.is_master:
                    # save_state_dict mutates (del keys) the dict it receives,
                    # so each run needs its own copy to avoid corruption.
                    if len(ready_idxs) > 1:
                        prepared[idx] = copy.copy(state_dict)
                    else:
                        prepared[idx] = state_dict

        # Snapshot run metadata on the main thread before the trainer's
        # progress.step increments during the next training step.
        run_metadata: dict[int, tuple[Path, int]] = {}
        for idx in prepared:
            run_metadata[idx] = (
                self.multi_run_manager.get_run_dir(idx),
                self.multi_run_manager.progress[idx].step,
            )

        # Clear ready flags on the main thread to prevent the next broadcast
        # from double-processing these runs (thread-safety fix).
        for idx in ready_idxs:
            self.multi_run_manager.ready_to_update[idx] = False

        # Phase 3: Submit background serialize + write (master only).
        blocking_time = time.perf_counter() - start_time
        if self.world.is_master and prepared:
            self._pending = self._executor.submit(
                self._write_and_notify,
                prepared,
                run_metadata,
                adapter_only,
                lora_configs,
                model if adapter_only else None,
                cuda_event,
            )
            self.logger.info(
                f"[BENCH] async broadcast: blocking={blocking_time:.3f}s "
                f"(pending_wait={pending_wait:.3f}s gather={gather_time:.3f}s) "
                f"write=background prev_write={self._last_broadcast_time:.3f}s"
            )

    def _notify_orchestrator(self, save_dir: Path):
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, max_async_level: int, interval_to_keep: int | None):
        # Cleanup targets step-(async_level+1), which is always a completed
        # write. The pending write is for the current step and won't be cleaned.
        for idx in self.multi_run_manager.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
                max_async_level,
                interval_to_keep,
            )

    def shutdown(self):
        """Wait for pending work and shut down the thread pool."""
        self._wait_for_pending()
        self._executor.shutdown(wait=True)
