import shutil
from pathlib import Path

from prime_rl.utils.pathing import get_all_ckpt_steps, get_batch_dir, get_step_path


class BatchLog:
    """Shipped steps on disk, one file per step (``<batch_dir>/step_<N>/batch.bin``). A step
    is written ahead of its delivery and published with it, so every published step is one
    the trainer was handed."""

    def __init__(self, output_dir: Path):
        self.dir = get_batch_dir(output_dir)

    def path(self, step: int) -> Path:
        return get_step_path(self.dir, step) / "batch.bin"

    def write(self, step: int, payload: bytes) -> None:
        get_step_path(self.dir, step).mkdir(parents=True, exist_ok=True)
        self.path(step).with_suffix(".bin.tmp").write_bytes(payload)

    def publish(self, step: int) -> None:
        path = self.path(step)
        path.with_suffix(".bin.tmp").rename(path)

    def read(self, step: int) -> bytes | None:
        try:
            return self.path(step).read_bytes()
        except FileNotFoundError:
            return None

    def prune(self, through: int) -> None:
        """Delete the steps up to ``through``."""
        for step in get_all_ckpt_steps(self.dir):
            if step <= through:
                shutil.rmtree(get_step_path(self.dir, step), ignore_errors=True)
