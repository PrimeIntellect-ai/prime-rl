import asyncio
import io
from pathlib import Path
from time import time

import msgspec

from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, RoutedExperts, TrainingBatch, TrainingSample
from prime_rl.utils.pathing import get_rollout_dir, get_step_path, sync_wait_for_path

BATCH_FILE_TMP_NAME = "train_rollouts.bin.tmp"
BATCH_FILE_NAME = "train_rollouts.bin"
SIDECAR_FILE_TMP_NAME = "train_rollouts.routed_experts.bin.tmp"
SIDECAR_FILE_NAME = "train_rollouts.routed_experts.bin"
FORMAT_VERSION = 2
LOG_FREQ_SECONDS = 10


class FileSystemTrainingBatchSender(TrainingBatchSender):
    """Filesystem-based training batch sender.

    Writes `train_rollouts.bin` and a `train_rollouts.routed_experts.bin` sidecar.
    `routed_experts.data` (the bulk of the payload, ~85%) is peeled out of msgpack
    encoding and written as a single concatenated raw-bytes blob via a threaded
    write. Encoded TrainingSample frames are streamed one-at-a-time with
    `await asyncio.sleep(0)` between samples so the event loop stays responsive
    during step transitions.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.rollout_dir = get_rollout_dir(output_dir)

    async def send(self, batch: TrainingBatch) -> None:
        step_path = get_step_path(self.rollout_dir, batch.step)
        step_path.mkdir(parents=True, exist_ok=True)

        offsets: list[int] = []
        shapes: list[list[int] | None] = []
        dtypes: list[str | None] = []
        re_payloads: list[bytes] = []
        running = 0

        samples_buf = io.BytesIO()
        for sample in batch.examples:
            if sample.routed_experts is None:
                offsets.append(-1)
                shapes.append(None)
                dtypes.append(None)
                out_sample = sample
            else:
                re = sample.routed_experts
                offsets.append(running)
                shapes.append(re.shape)
                dtypes.append(re.dtype)
                re_payloads.append(re.data)
                running += len(re.data)
                out_sample = msgspec.structs.replace(
                    sample,
                    routed_experts=RoutedExperts(data=b"", shape=re.shape, dtype=re.dtype),
                )
            frame = self.encoder.encode(out_sample)
            samples_buf.write(len(frame).to_bytes(4, "little"))
            samples_buf.write(frame)
            await asyncio.sleep(0)

        manifest = self.encoder.encode(
            {
                "version": FORMAT_VERSION,
                "format": "stream_sidecar",
                "step": batch.step,
                "run_idx": batch.run_idx,
                "n_examples": len(batch.examples),
                "offsets": offsets,
                "shapes": shapes,
                "dtypes": dtypes,
                "sidecar_total_bytes": running,
                "sidecar_filename": SIDECAR_FILE_NAME,
            }
        )

        await asyncio.to_thread(
            self._write_files, step_path, manifest, samples_buf.getvalue(), re_payloads
        )

    @staticmethod
    def _write_files(
        step_path: Path, manifest: bytes, samples_buf: bytes, re_payloads: list[bytes]
    ) -> None:
        # Sidecar must be visible before the main file, since the receiver
        # treats the main file's existence as the "batch is ready" signal.
        tmp_sidecar = step_path / SIDECAR_FILE_TMP_NAME
        with open(tmp_sidecar, "wb") as sf:
            for p in re_payloads:
                sf.write(p)
        tmp_sidecar.rename(step_path / SIDECAR_FILE_NAME)

        tmp_main = step_path / BATCH_FILE_TMP_NAME
        with open(tmp_main, "wb") as mf:
            mf.write(len(manifest).to_bytes(4, "little"))
            mf.write(manifest)
            mf.write(samples_buf)
        tmp_main.rename(step_path / BATCH_FILE_NAME)


def _decode_training_batch_from_disk(main_path: Path) -> TrainingBatch:
    """Read a stream_sidecar v2 train_rollouts.bin + sidecar back into a TrainingBatch."""
    with open(main_path, "rb") as f:
        data = f.read()

    cursor = 0
    manifest_len = int.from_bytes(data[cursor : cursor + 4], "little")
    cursor += 4
    manifest = msgspec.msgpack.decode(data[cursor : cursor + manifest_len])
    cursor += manifest_len

    n = manifest["n_examples"]
    samples: list[TrainingSample] = []
    for _ in range(n):
        frame_len = int.from_bytes(data[cursor : cursor + 4], "little")
        cursor += 4
        sample: TrainingSample = msgspec.msgpack.decode(
            data[cursor : cursor + frame_len], type=TrainingSample
        )
        cursor += frame_len
        samples.append(sample)

    sidecar_path = main_path.parent / manifest["sidecar_filename"]
    if manifest["sidecar_total_bytes"] > 0:
        with open(sidecar_path, "rb") as sf:
            sidecar = sf.read()
        offsets = manifest["offsets"]
        shapes = manifest["shapes"]
        dtypes = manifest["dtypes"]
        total = manifest["sidecar_total_bytes"]
        # End of each present routed_experts payload = next present offset (or total).
        next_present_offset = [total] * len(offsets)
        last_end = total
        for i in range(len(offsets) - 1, -1, -1):
            if offsets[i] >= 0:
                next_present_offset[i] = last_end
                last_end = offsets[i]
        for i, off in enumerate(offsets):
            if off < 0:
                continue
            samples[i].routed_experts = RoutedExperts(
                data=sidecar[off : next_present_offset[i]],
                shape=shapes[i],
                dtype=dtypes[i],
            )

    return TrainingBatch(
        examples=samples, step=manifest["step"], run_idx=manifest.get("run_idx")
    )


class FileSystemTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem-based training batch receiver supporting the v2 stream_sidecar format."""

    def __init__(self) -> None:
        super().__init__()
        self.multi_run_manager = get_multi_run_manager()
        self._last_logged_paths: list[Path] | None = None
        self._last_logged_time = time()
        self._waiting_since: float | None = None
        self._received_steps: dict[int, int] = {}

    def _get_received_step(self, idx: int) -> int:
        if idx not in self._received_steps:
            self._received_steps[idx] = self.multi_run_manager.progress[idx].step
        return self._received_steps[idx]

    def _get_batch_path(self, idx: int) -> Path:
        run_dir = self.multi_run_manager.get_run_dir(idx)
        rollout_dir = get_rollout_dir(run_dir)
        step = self._get_received_step(idx)
        return get_step_path(rollout_dir, step) / BATCH_FILE_NAME

    def can_receive(self) -> bool:
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx] and self._get_batch_path(idx).exists():
                return True
        return False

    def receive(self) -> list[TrainingBatch]:
        batches: list[TrainingBatch] = []
        now = time()

        if self.can_receive():
            self._waiting_since = None
        else:
            self._waiting_since = self._waiting_since or now

        current_paths = [self._get_batch_path(idx) for idx in self.multi_run_manager.used_idxs]
        if current_paths != self._last_logged_paths or now - self._last_logged_time > LOG_FREQ_SECONDS:
            if len(current_paths) == 0:
                self.logger.debug(
                    "Did you set the output dir of the orchestrator to a run_* subdirectory of the trainer output dir?"
                )
            waiting_suffix = ""
            if self._waiting_since is not None:
                waiting_suffix = f" (waiting {now - self._waiting_since:.1f}s)"
            self.logger.debug(f"Looking for batches in {current_paths}{waiting_suffix}")
            self._last_logged_paths = current_paths
            self._last_logged_time = now

        for idx in self.multi_run_manager.used_idxs:
            if self.multi_run_manager.ready_to_update[idx]:
                continue
            batch_path = self._get_batch_path(idx)
            if batch_path.exists():
                try:
                    batch = _decode_training_batch_from_disk(batch_path)
                    batch.run_idx = idx
                    batches.append(batch)
                    self._received_steps[idx] = self._get_received_step(idx) + 1
                except Exception as e:
                    self.logger.error(f"Error loading rollouts for run {idx}: {e}")
        return batches

    def reset_run(self, idx: int) -> None:
        if idx in self._received_steps:
            del self._received_steps[idx]


class FileSystemMicroBatchSender(MicroBatchSender):
    """Filesystem-based micro batch sender that writes micro batches to disk."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int = 0):
        super().__init__(output_dir, data_world_size)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        step_path = get_step_path(self.rollout_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)

        for data_rank in range(self.data_world_size):
            buffer = self.encoder.encode(micro_batch_grid[data_rank])
            tmp_path = step_path / f"rank_{data_rank}.bin.tmp"
            with open(tmp_path, "wb") as f:
                f.write(buffer)
            tmp_path.rename(step_path / f"rank_{data_rank}.bin")
        self.current_step += 1


class FileSystemMicroBatchReceiver(MicroBatchReceiver):
    """Filesystem-based micro batch receiver that reads micro batches from disk."""

    def __init__(self, output_dir: Path, data_rank: int, current_step: int = 0):
        super().__init__(output_dir, data_rank)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def _get_micro_batch_path(self) -> Path:
        return get_step_path(self.rollout_dir, self.current_step) / f"rank_{self.data_rank}.bin"

    def wait(self) -> None:
        sync_wait_for_path(self._get_micro_batch_path())

    def can_receive(self) -> bool:
        return self._get_micro_batch_path().exists()

    def receive(self) -> list[MicroBatch]:
        with open(self._get_micro_batch_path(), "rb") as f:
            micro_batches: list[MicroBatch] = self.decoder.decode(f.read())
        self.current_step += 1
        return micro_batches
