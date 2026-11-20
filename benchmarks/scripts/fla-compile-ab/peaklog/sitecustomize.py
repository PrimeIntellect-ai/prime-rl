"""Append each rank's max_memory_allocated and max_memory_reserved to $PEAK_LOG_DIR/rank<R>.jsonl before every torch.cuda.reset_peak_memory_stats call and at exit."""

import os

if os.environ.get("PEAK_LOG_DIR") and os.environ.get("RANK") is not None:
    import atexit
    import json

    import torch

    _path = os.path.join(os.environ["PEAK_LOG_DIR"], f"rank{os.environ['RANK']}.jsonl")
    _original = torch.cuda.reset_peak_memory_stats
    _calls = [0]

    def _record(event):
        if not torch.cuda.is_initialized():
            return
        row = {
            "call": _calls[0],
            "event": event,
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    def _hooked(*args, **kwargs):
        _record("before_reset")
        _calls[0] += 1
        return _original(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats = _hooked
    atexit.register(_record, "exit")
