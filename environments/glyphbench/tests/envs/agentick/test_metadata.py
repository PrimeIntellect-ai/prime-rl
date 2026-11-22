"""Metadata conversion does not require the optional AgenticK backend."""

from __future__ import annotations

import json

import numpy as np

from glyphbench.envs.agentick.env import _compact_json


def test_numpy_metadata_serializes_nested_arrays_and_scalars() -> None:
    metadata = {
        "positions": np.array([[1, 2], [3, 4]]),
        "count": np.int64(2),
        "success": np.bool_(True),
        "empty": np.array([]),
        "scalar_array": np.array(7),
    }

    assert json.loads(_compact_json(metadata)) == {
        "positions": [[1, 2], [3, 4]],
        "count": 2,
        "success": True,
        "empty": [],
        "scalar_array": 7,
    }
