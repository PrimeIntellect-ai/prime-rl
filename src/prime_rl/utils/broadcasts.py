"""The markers of a weight broadcast's step dir, which the trainer, the orchestrator and the engines read. Kept
apart from ``prime_rl.transports.weights`` so the inference server can read them without the transports."""

import os
from pathlib import Path

# Broadcast-dir sentinels. Every transport goes through the same four stages:
# the trainer resets the step dir and raises ``.sender_ready``, the consumer
# joins the transfer and replies ``.receiver_ready`` (for NCCL with the
# engines paused inside the receive RPC), the trainer raises ``.started`` and
# moves the weights, then raises ``.finished`` — for filesystem, the weights
# are fully on disk. Engines that pull (``weight_broadcast.pull``) acknowledge
# nothing: the trainer goes from ``.sender_ready`` straight to ``.started``.
SENDER_READY_MARKER = ".sender_ready"
RECEIVER_READY_MARKER = ".receiver_ready"
STARTED_MARKER = ".started"
FINISHED_MARKER = ".finished"


def marker_exists(marker: Path) -> bool:
    """``marker.exists()`` that sees a marker the other host just created on a
    shared NFS mount. NFS clients cache "not found" answers for the attribute
    timeout (30 s by default); opening a directory forces the client to
    revalidate it (close-to-open). The step dir may not exist yet, so the
    broadcast dir above it is opened too."""
    for directory in (marker.parent.parent, marker.parent):
        try:
            os.close(os.open(directory, os.O_RDONLY | os.O_DIRECTORY))
        except FileNotFoundError:
            pass
    return marker.exists()
