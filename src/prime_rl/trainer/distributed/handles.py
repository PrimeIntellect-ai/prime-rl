from typing import Any

import torch

# Custom ops cannot take communication handles, so they carry them as a one-element CPU tensor ID.
_handles: dict[int, Any] = {}
_next_handle_id = 0


class _HandleOwner(bytearray):
    """Storage of a handle ID tensor. The handle lives exactly as long as some tensor shares that storage."""

    def __del__(self) -> None:
        _handles.pop(int.from_bytes(self, "little"), None)


def store_handle(handle: Any) -> torch.Tensor:
    global _next_handle_id
    _next_handle_id += 1
    _handles[_next_handle_id] = handle
    return torch.frombuffer(_HandleOwner(_next_handle_id.to_bytes(8, "little")), dtype=torch.int64)


def get_handle(handle_id: torch.Tensor) -> Any:
    return _handles[handle_id.item()]
