"""Task selection and finalized-sample admission for training environments."""

from prime_rl.orchestrator.curriculum.advantage_range_gate import AdvantageRangeGate
from prime_rl.orchestrator.curriculum.base import (
    AdmissionGate,
    Curriculum,
    CurriculumResult,
    TaskSampler,
    setup_curriculum,
)
from prime_rl.orchestrator.curriculum.difficulty_pools import DifficultyPools

__all__ = [
    "AdmissionGate",
    "AdvantageRangeGate",
    "Curriculum",
    "CurriculumResult",
    "DifficultyPools",
    "TaskSampler",
    "setup_curriculum",
]
