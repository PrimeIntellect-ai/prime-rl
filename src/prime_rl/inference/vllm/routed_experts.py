from __future__ import annotations

from typing import Any

import numpy as np
import pybase64
from vllm.distributed.aux_output_connector.connector import AuxRequestOutput


def serialize_routed_experts(routed_experts: Any, start: int = 0) -> dict[str, Any] | None:
    if routed_experts is None:
        return None

    array = np.asarray(routed_experts)
    assert array.ndim == 3
    assert np.issubdtype(array.dtype, np.integer)
    dtype = np.uint8
    if array.size:
        assert array.min() >= 0
        if array.max() > np.iinfo(np.uint8).max:
            # Models with >256 experts (e.g. NemotronH Super/Ultra: 512) need wider
            # indices. The payload self-describes via "dtype" so consumers pick it up.
            assert array.max() <= np.iinfo(np.uint16).max
            dtype = np.uint16

    compact = np.ascontiguousarray(array.astype(dtype, copy=False))
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "start": start,
        "dtype": np.dtype(dtype).name,
    }


class PackedAuxOutputs:
    """One step's ``dict[request_id, AuxRequestOutput]`` as one rows array.

    Answers the scheduler's ``in`` / ``[]`` lookups and pickles as three byte strings instead of one buffer per
    request. Unpickled rows own their memory rather than viewing the shared-memory ring buffer.
    """

    __slots__ = ("_index", "_token_starts", "_offsets", "_rows")

    def __init__(self, index: dict[str, int], token_starts: np.ndarray, offsets: np.ndarray, rows: np.ndarray):
        self._index = index
        self._token_starts = token_starts
        self._offsets = offsets
        self._rows = rows

    @classmethod
    def pack(cls, outputs: dict[str, AuxRequestOutput]) -> PackedAuxOutputs:
        rows = [output.rows for output in outputs.values()]
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum([len(r) for r in rows], out=offsets[1:])
        token_starts = np.fromiter((output.token_start for output in outputs.values()), np.int64, len(rows))
        return cls(dict(zip(outputs, range(len(rows)))), token_starts, offsets, np.concatenate(rows))

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._index

    def __getitem__(self, request_id: str) -> AuxRequestOutput:
        i = self._index[request_id]
        return AuxRequestOutput(int(self._token_starts[i]), self._rows[self._offsets[i] : self._offsets[i + 1]])

    def __reduce__(self):
        rows = self._rows
        return (
            _unpack_aux_outputs,
            (
                list(self._index),
                self._token_starts.tobytes(),
                self._offsets.tobytes(),
                rows.tobytes(),
                rows.dtype.str,
                rows.shape,
            ),
        )


def _unpack_aux_outputs(request_ids, token_starts, offsets, rows, dtype, shape) -> PackedAuxOutputs:
    return PackedAuxOutputs(
        dict(zip(request_ids, range(len(request_ids)))),
        np.frombuffer(token_starts, np.int64),
        np.frombuffer(offsets, np.int64),
        np.frombuffer(rows, dtype).reshape(shape),
    )
