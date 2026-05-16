from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file

from prime_rl.ttt.lora_engine import HookedLoRAEngine
from prime_rl.ttt.trainer_replay import (
    FrozenLoRAAdapter,
    LoRAModuleState,
    TTTTrainerAdapterManager,
    cleanup_consumed_adapters,
)


def test_materialize_honors_adapter_dir_and_records_stable_metadata(tmp_path: Path):
    engine = HookedLoRAEngine.__new__(HookedLoRAEngine)
    engine.adapter_dir = tmp_path
    engine.rank = 2
    engine.base_step = 17
    engine.load_adapters_into_vllm = False
    engine.materialized = []

    def state_for_kinds(_session, _kinds):
        return {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        }

    engine._state_for_kinds = state_for_kinds
    engine._adapter_config = lambda rank, alpha: {"r": rank, "lora_alpha": alpha}

    session = SimpleNamespace(
        session_id="rollout-123",
        prompt_version=3,
        completion_version=5,
        materialized_adapters=[],
    )

    meta = asyncio.run(
        engine.materialize(
            session,
            name="adapter-a",
            kinds=("prompt", "completion"),
            load_into_vllm=True,
            adapter_kind="combined",
            turn_idx=2,
        )
    )

    adapter_path = tmp_path / "adapter-a"
    assert meta == {
        "adapter_name": "adapter-a",
        "adapter_path": adapter_path.as_posix(),
        "adapter_kind": "combined",
        "loaded_into_vllm": False,
        "rank": 4,
        "base_step": 17,
        "prompt_version": 3,
        "completion_version": 5,
        "session_id": "rollout-123",
        "turn_idx": 2,
    }
    assert session.materialized_adapters == [meta]
    assert (adapter_path / "adapter_model.safetensors").exists()
    assert load_file(adapter_path / "adapter_model.safetensors", device="cpu")
    assert json.loads((adapter_path / "adapter_config.json").read_text()) == {"r": 4, "lora_alpha": 4}


def test_trainer_cleanup_evicts_cache_and_deletes_directories(tmp_path: Path):
    model = nn.Sequential(nn.Linear(3, 4))
    manager = TTTTrainerAdapterManager(model)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    state = LoRAModuleState(a=torch.ones(2, 3), b=torch.ones(4, 2))
    state.to(torch.device("cpu"), torch.float32, cache=True)
    manager.cache[adapter_path.as_posix()] = FrozenLoRAAdapter(
        path=adapter_path.as_posix(),
        modules={"0": state},
    )

    deleted = cleanup_consumed_adapters(manager, {adapter_path.as_posix()}, delete_from_disk=True)

    assert deleted == 1
    assert not adapter_path.exists()
    assert manager.cache == {}
    assert state._device_cache == {}


def test_trainer_cleanup_can_leave_disk_artifacts_while_evicting_cache(tmp_path: Path):
    model = nn.Sequential(nn.Linear(3, 4))
    manager = TTTTrainerAdapterManager(model)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    manager.cache[adapter_path.as_posix()] = FrozenLoRAAdapter(path=adapter_path.as_posix(), modules={})

    deleted = cleanup_consumed_adapters(manager, [adapter_path.as_posix()], delete_from_disk=False)

    assert deleted == 0
    assert adapter_path.exists()
    assert adapter_path.as_posix() not in manager.cache
