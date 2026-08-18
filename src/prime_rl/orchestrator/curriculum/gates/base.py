"""Admission gate interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import EpisodeRun


class AdmissionGate:
    """Base class for user-authored training-sample admission policies."""

    def admit(self, group: list[EpisodeRun]) -> bool:
        """Return whether a finalized group should enter the training batch."""
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this gate."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before results resume."""

    def metrics(self) -> dict[str, float]:
        """Return metrics relative to this gate's namespace."""
        return {}
