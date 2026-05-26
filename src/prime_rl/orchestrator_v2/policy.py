"""Shared mutable policy state between watcher and dispatcher.

The orchestrator v2 routes the current policy via a single ``Policy`` instance.
The ``WeightWatcher`` writes ``version`` (and ``model_name`` after a LoRA
swap) when a new checkpoint becomes available; the ``RolloutDispatcher`` and
all in-flight rollout meta read these fields at dispatch time. Passed by
reference — never copied — so observers see new versions immediately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Policy:
    """Mutable shared view of the current policy.

    Attributes:
        version: Latest policy step the watcher has installed. Bumped by
            ``WeightWatcher`` after a successful ``update_weights``. Used as the
            ``policy_version`` snapshot on each emitted ``Trajectory``.
        model_name: Model name to send on rollout requests. For LoRA this
            advances to the adapter name after the first weight update; for
            plain runs it stays at the base model name.
    """

    version: int = 0
    model_name: str = ""
