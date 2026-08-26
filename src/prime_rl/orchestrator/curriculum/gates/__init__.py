"""Admission gate interfaces and implementations."""

from prime_rl.orchestrator.curriculum.gates.adv import AdvRangeGate
from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

__all__ = ["AdmissionGate", "AdvRangeGate"]
