"""The TTT config TOMLs parse and launch: service configs validate against
TTTServiceConfig (top-level keys must not hide under [optim]), and the scaleswe base
config's vLLM LoRA sizing covers the TTT service's adapters."""

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from prime_rl.configs.ttt import TTTServiceConfig  # noqa: E402

CONFIGS = Path(__file__).parents[3] / "configs" / "ttt"


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize("name", ["ttt.toml", "scaleswe/ttt_service.toml"])
def test_service_configs_validate(name: str):
    TTTServiceConfig.model_validate(load(CONFIGS / name))


def test_scaleswe_lora_sizing():
    base = load(CONFIGS / "scaleswe" / "base.toml")["inference"]
    service = load(CONFIGS / "scaleswe" / "ttt_service.toml")
    # vLLM requires max_cpu_loras >= max_loras (its default of 100 would abort the launch).
    assert base["max_cpu_loras"] >= base["max_loras"]
    # The engine can only serve adapters whose rank fits max_lora_rank.
    assert base["max_lora_rank"] >= service["lora"]["rank"]
