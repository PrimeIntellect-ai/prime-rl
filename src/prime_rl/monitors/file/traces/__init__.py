"""Where the file monitor puts the traces, and everything written about them.

The stream and its annotations live under one directory so nothing beside them has to
be read as belonging to them, and every index is named for the stream it indexes and
sits beside it.
"""

from pathlib import Path

from prime_rl.utils.pathing import get_file_monitor_dir


def get_trace_dir(output_dir: Path) -> Path:
    return get_file_monitor_dir(output_dir) / "traces"


def get_trace_stream(output_dir: Path) -> Path:
    """Every episode, appended as it arrives."""
    return get_trace_dir(output_dir) / "stream.jsonl"


def get_annotations_dir(output_dir: Path) -> Path:
    """One file per producer of trace updates: orch ship-time facts, trainer streams."""
    return get_trace_dir(output_dir) / "annotations"


def get_index_path(path: Path) -> Path:
    """A stream's index, beside it, so a reader pairs the two without knowing who
    wrote them."""
    return path.with_name(path.name.replace(".jsonl", ".index.jsonl"))


__all__ = ["get_annotations_dir", "get_index_path", "get_trace_dir", "get_trace_stream"]
