import shutil
import tempfile
from pathlib import Path

from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker
from prime_rl.utils.sparse_weights import (
    SPARSE_WEIGHTS_MANIFEST,
    apply_sparse_delta,
    parse_step_from_dir,
    read_sparse_manifest,
)


class SparseFileSystemWeightUpdateWorker(FileSystemWeightUpdateWorker):
    """vLLM worker extension for sparse filesystem weight updates."""

    def init_broadcaster(self) -> None:
        self._sparse_local_step = 0
        self._sparse_local_weight_dir: Path | None = None
        self._sparse_cache_dir = Path(tempfile.mkdtemp(prefix="prime_rl_sparse_weights_"))

    def update_weights_from_path(self, weight_path: str) -> None:
        weight_dir = Path(weight_path)
        target_dir = self._materialize_weight_dir(weight_dir)
        super().update_weights_from_path(target_dir.as_posix())

    def _materialize_weight_dir(self, weight_dir: Path) -> Path:
        if not hasattr(self, "_sparse_cache_dir"):
            self.init_broadcaster()

        manifest = read_sparse_manifest(weight_dir)
        if manifest is None or manifest.get("type") == "full":
            self._set_full_weight_dir(weight_dir, manifest)
            return weight_dir

        if manifest.get("type") != "delta":
            raise ValueError(f"Unknown sparse weight manifest type in {weight_dir}: {manifest.get('type')}")

        target_step = int(manifest["step"])
        self._materialize_step(weight_dir.parent, target_step)
        assert self._sparse_local_weight_dir is not None
        return self._sparse_local_weight_dir

    def _set_full_weight_dir(self, weight_dir: Path, manifest: dict | None) -> None:
        if manifest is not None and "step" in manifest:
            self._sparse_local_step = int(manifest["step"])
        else:
            self._sparse_local_step = parse_step_from_dir(weight_dir)
        self._sparse_local_weight_dir = weight_dir

    def _materialize_step(self, broadcast_dir: Path, target_step: int) -> None:
        if self._sparse_local_step == target_step:
            return
        if self._sparse_local_step > target_step:
            raise ValueError(
                f"Cannot apply sparse weights for step {target_step}; worker already has step {self._sparse_local_step}"
            )

        step_dir = broadcast_dir / f"step_{target_step}"
        if not step_dir.exists():
            raise FileNotFoundError(f"Cannot materialize sparse weights; missing {step_dir}")

        manifest = read_sparse_manifest(step_dir)
        if manifest is None or manifest.get("type") == "full":
            self._set_full_weight_dir(step_dir, manifest)
            return

        if manifest.get("type") != "delta":
            raise ValueError(f"Unknown sparse weight manifest type in {step_dir}: {manifest.get('type')}")

        base_step = int(manifest["base_step"])
        self._materialize_step(broadcast_dir, base_step)
        if base_step != self._sparse_local_step:
            raise ValueError(
                f"Sparse delta base mismatch for {step_dir}: base={base_step}, local={self._sparse_local_step}"
            )
        if self._sparse_local_weight_dir is None:
            raise ValueError(f"Cannot apply sparse delta {step_dir} before a full weight update")

        self._ensure_private_materialized_dir()
        assert self._sparse_local_weight_dir is not None
        apply_sparse_delta(step_dir, self._sparse_local_weight_dir)
        self._sparse_local_step = int(manifest["step"])

    def _ensure_private_materialized_dir(self) -> None:
        assert self._sparse_local_weight_dir is not None
        if self._sparse_local_weight_dir == self._sparse_cache_dir:
            return

        shutil.rmtree(self._sparse_cache_dir, ignore_errors=True)
        shutil.copytree(
            self._sparse_local_weight_dir,
            self._sparse_cache_dir,
            ignore=shutil.ignore_patterns("STABLE", SPARSE_WEIGHTS_MANIFEST),
        )
        self._sparse_local_weight_dir = self._sparse_cache_dir
