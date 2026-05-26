from __future__ import annotations

from typing import Literal

Split = Literal["train", "dev", "frozen_test"]

TRAIN_YEAR_MAX = 2023
FROZEN_TEST_YEARS = {2024, 2025}
FROZEN_COMPETITIONS = {
    "HiPhO",
    "IPhO",
    "APhO",
    "EuPhO",
    "NBPhO",
    "PanPhO",
    "PanMechanics",
    "F=MA",
}
BLOCKED_TRAIN_SOURCES = {
    "PHYSICS_test",
    "OlympiadBench_eval",
    "JEEBench",
    "SciBench",
}


def choose_split(
    *,
    source: str,
    competition: str,
    year: int | None,
    requested_split: Split | None = None,
) -> Split:
    if requested_split:
        validate_training_policy(
            source=source,
            competition=competition,
            year=year,
            split=requested_split,
        )
        return requested_split
    if competition == "HiPhO":
        return "frozen_test"
    if year in FROZEN_TEST_YEARS and competition in FROZEN_COMPETITIONS:
        return "frozen_test"
    return "train"


def validate_training_policy(
    *,
    source: str,
    competition: str,
    year: int | None,
    split: Split,
) -> None:
    if split != "train":
        return
    if source in BLOCKED_TRAIN_SOURCES:
        raise ValueError(f"{source} is blocked from train")
    if competition == "HiPhO":
        raise ValueError("HiPhO is frozen-test only")
    if year is None and competition in FROZEN_COMPETITIONS:
        raise ValueError(f"{competition} rows need an explicit year before train")
    if year is not None and year > TRAIN_YEAR_MAX:
        raise ValueError(f"year {year} is blocked from train")
    if year in FROZEN_TEST_YEARS and competition in FROZEN_COMPETITIONS:
        raise ValueError(f"{competition} {year} is blocked from train")
