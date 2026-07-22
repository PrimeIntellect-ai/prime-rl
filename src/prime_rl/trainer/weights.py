import fcntl
import json
import os
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from prime_rl.trainer.lora import (
    clean_lora_state_dict,
)
from prime_rl.utils.logger import get_logger

STREAMING_SHARD_SIZE = 512 * 1024**2
_LAYER_KEY_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def load_state_dict_keys(save_dir: Path) -> list[str]:
    """Load only the key names from safetensor files without reading tensor data."""
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def _group_safetensor_keys(save_dir: Path) -> dict[int, list[tuple[Path, str]]]:
    groups: dict[int, list[tuple[Path, str]]] = defaultdict(list)
    seen_keys: set[str] = set()
    for safetensor_path in sorted(save_dir.glob("*.safetensors")):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in seen_keys:
                    raise ValueError(f"duplicate tensor key {key!r} in {save_dir}")
                seen_keys.add(key)
                match = _LAYER_KEY_PATTERN.search(key)
                layer_idx = int(match.group(1)) if match else -1
                groups[layer_idx].append((safetensor_path, key))
    return groups


def _load_safetensor_group(entries: list[tuple[Path, str]]) -> dict[str, Tensor]:
    entries_by_path: dict[Path, list[str]] = defaultdict(list)
    for path, key in entries:
        entries_by_path[path].append(key)

    state_dict: dict[str, Tensor] = {}
    for path, keys in entries_by_path.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)
    return state_dict


class _StreamingSafetensorsWriter:
    def __init__(self, save_dir: Path, max_shard_size: int):
        self.save_dir = save_dir
        self.max_shard_size = max_shard_size
        self.buffer: dict[str, Tensor] = {}
        self.buffer_size = 0
        self.total_size = 0
        self.shards: list[tuple[Path, list[str]]] = []
        self.keys: set[str] = set()

    def add(self, key: str, tensor: Tensor) -> None:
        if key in self.keys:
            raise ValueError(f"conversion produced duplicate tensor key {key!r}")
        self.keys.add(key)

        tensor = tensor.contiguous()
        tensor_size = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_size + tensor_size > self.max_shard_size:
            self.flush()
        self.buffer[key] = tensor
        self.buffer_size += tensor_size
        self.total_size += tensor_size
        if self.buffer_size >= self.max_shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_path = self.save_dir / f".model-{len(self.shards) + 1:05d}.safetensors"
        shard_keys = list(self.buffer)
        save_file(self.buffer, shard_path, metadata={"format": "pt"})
        self.shards.append((shard_path, shard_keys))
        self.buffer = {}
        self.buffer_size = 0

    def finish(self) -> None:
        self.flush()
        if not self.shards:
            raise ValueError("conversion produced an empty state dict")
        if len(self.shards) == 1:
            self.shards[0][0].rename(self.save_dir / SAFE_WEIGHTS_NAME)
            return

        weight_map: dict[str, str] = {}
        shard_count = len(self.shards)
        for shard_idx, (provisional_path, shard_keys) in enumerate(self.shards, start=1):
            shard_name = f"model-{shard_idx:05d}-of-{shard_count:05d}.safetensors"
            provisional_path.rename(self.save_dir / shard_name)
            weight_map.update(dict.fromkeys(shard_keys, shard_name))
        index = {"metadata": {"total_size": self.total_size}, "weight_map": weight_map}
        with open(self.save_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")


def stream_convert_state_dict(
    source_dir: Path,
    save_dir: Path,
    convert_layer: Callable[[dict[str, Tensor], int], dict[str, Tensor] | None],
    is_converted: Callable[[dict[str, None]], bool],
    *,
    overwrite: bool = False,
    max_shard_size: int = STREAMING_SHARD_SIZE,
) -> bool:
    """Convert safetensors one layer group at a time and atomically publish them."""
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=f".{save_dir.name}.tmp-", dir=save_dir.parent))
    try:
        writer = _StreamingSafetensorsWriter(staged_dir, max_shard_size)
        groups = _group_safetensor_keys(Path(source_dir))
        for layer_idx in sorted(groups):
            state_dict = _load_safetensor_group(groups[layer_idx])
            converted = convert_layer(state_dict, layer_idx)
            if converted is not None:
                state_dict = converted
            for key in list(state_dict):
                writer.add(key, state_dict.pop(key))
        writer.finish()
        if not is_converted(dict.fromkeys(writer.keys)):
            raise ValueError("streaming conversion did not produce the expected state-dict format")

        def existing_is_converted(path: Path) -> bool:
            if not is_state_dict_complete(path):
                return False
            keys = dict.fromkeys(load_state_dict_keys(path))
            return is_converted(keys)

        return atomic_publish_state_dict_dir(
            staged_dir,
            save_dir,
            overwrite=overwrite,
            existing_is_valid=existing_is_converted,
        )
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory in safetensors or torch format.

    Sharded saves consume ``state_dict`` as shards are written to keep peak memory
    bounded. Callers that need the mapping afterward must pass a copy.
    """
    logger = get_logger()
    if adapter:
        weights_name = ADAPTER_SAFE_WEIGHTS_NAME if save_format == "safetensors" else ADAPTER_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        # Save weights (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4252)
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]
            if save_format == "safetensors":
                save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, save_dir / shard_file)
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
            save_index_file = save_dir / save_index_file
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        if save_format == "safetensors":
            save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_dir / weights_name)


def is_state_dict_complete(save_dir: Path) -> bool:
    """Return whether ``save_dir`` contains a complete state-dict save."""
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        return False

    for index_name, is_safetensors in (
        (SAFE_WEIGHTS_INDEX_NAME, True),
        (WEIGHTS_INDEX_NAME, False),
    ):
        index_path = save_dir / index_name
        if not index_path.is_file():
            continue
        try:
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not isinstance(weight_map, dict) or not weight_map:
            return False

        shard_names = set(weight_map.values())
        shard_paths = [save_dir / shard_name for shard_name in shard_names]
        if not all(path.is_file() and path.stat().st_size > 0 for path in shard_paths):
            return False
        if not is_safetensors:
            return True

        actual_keys: set[str] = set()
        try:
            for shard_path in shard_paths:
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    actual_keys.update(f.keys())
        except (OSError, SafetensorError):
            return False
        return actual_keys == set(weight_map)

    for weights_name, is_safetensors in (
        (SAFE_WEIGHTS_NAME, True),
        (ADAPTER_SAFE_WEIGHTS_NAME, True),
        (WEIGHTS_NAME, False),
        (ADAPTER_WEIGHTS_NAME, False),
    ):
        weights_path = save_dir / weights_name
        if not weights_path.is_file() or weights_path.stat().st_size == 0:
            continue
        if not is_safetensors:
            return True
        try:
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                return bool(f.keys())
        except (OSError, SafetensorError):
            return False
    return False


@contextmanager
def _lock_save_dir(save_dir: Path):
    lock_path = save_dir.parent / f".{save_dir.name}.lock"
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def atomic_publish_state_dict_dir(
    staged_dir: Path,
    save_dir: Path,
    *,
    overwrite: bool = False,
    existing_is_valid: Callable[[Path], bool] | None = None,
) -> bool:
    """Atomically publish a complete staged state-dict directory."""
    staged_dir = Path(staged_dir)
    save_dir = Path(save_dir)
    if not is_state_dict_complete(staged_dir):
        raise ValueError(f"staged state dict is incomplete: {staged_dir}")

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    with _lock_save_dir(save_dir):
        if save_dir.exists():
            if existing_is_valid is not None:
                if existing_is_valid(save_dir):
                    return False
            elif not overwrite and is_state_dict_complete(save_dir):
                return False
            _remove_path(save_dir)
        try:
            os.rename(staged_dir, save_dir)
        except OSError:
            if existing_is_valid is not None and existing_is_valid(save_dir):
                return False
            if existing_is_valid is None and not overwrite and is_state_dict_complete(save_dir):
                return False
            raise
        return True


def atomic_save_state_dict(state_dict: dict[str, Tensor], save_dir: Path, *, overwrite: bool = False, **kwargs) -> bool:
    """Save into a unique temp sibling and atomically publish the directory.

    Concurrent writers serialize publication with a file lock. A complete
    destination written by another process wins unless ``overwrite`` is true;
    an incomplete destination is repaired. Returns whether this writer
    published its output. Like ``save_state_dict``, sharded saves consume
    ``state_dict``.
    """
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{save_dir.name}.tmp-", dir=save_dir.parent))
    try:
        save_state_dict(state_dict, tmp_dir, **kwargs)
        return atomic_publish_state_dict_dir(tmp_dir, save_dir, overwrite=overwrite)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """Gather distributed weights on CPU on master rank."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if is_master:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                cpu_state[key] = value.to("cpu", non_blocking=False)
        torch.distributed.barrier()

    # Always clean up the state dict for HF compatibility
    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
        cpu_state = clean_lora_state_dict(cpu_state)

    return cpu_state
