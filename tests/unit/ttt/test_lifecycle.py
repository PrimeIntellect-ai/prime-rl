from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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

    def adapter_state(_session):
        return {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        }

    engine._adapter_state = adapter_state
    engine._adapter_config = lambda rank, alpha: {"r": rank, "lora_alpha": alpha}

    session = SimpleNamespace(
        session_id="rollout-123",
        version=5,
        materialized_adapters=[],
        latest_adapter=None,
    )

    meta = asyncio.run(
        engine.materialize(
            session,
            name="adapter-a",
            load_into_vllm=True,
            turn_idx=2,
        )
    )

    adapter_path = tmp_path / "adapter-a"
    assert meta == {
        "adapter_name": "adapter-a",
        "adapter_path": adapter_path.as_posix(),
        "adapter_kind": "snapshot",
        "loaded_into_vllm": False,
        "rank": 2,
        "base_step": 17,
        "version": 5,
        "session_id": "rollout-123",
        "turn_idx": 2,
    }
    assert session.materialized_adapters == [meta]
    assert session.latest_adapter == meta
    assert (adapter_path / "adapter_model.safetensors").exists()
    assert load_file(adapter_path / "adapter_model.safetensors", device="cpu")
    assert json.loads((adapter_path / "adapter_config.json").read_text()) == {"r": 2, "lora_alpha": 2}


def test_get_or_create_session_enforces_max_concurrent_sessions():
    engine = HookedLoRAEngine.__new__(HookedLoRAEngine)
    engine.sessions = {"existing": object()}
    engine.max_concurrent_sessions = 1

    with pytest.raises(RuntimeError, match="max_concurrent_sessions=1"):
        engine.get_or_create_session("new")


def test_append_and_train_with_replay_spans_uses_pre_chunk_adapters():
    engine = HookedLoRAEngine.__new__(HookedLoRAEngine)
    engine.update_every_tokens = 4
    engine.base_step = 7

    def train_chunk(session, token_ids):
        assert len(token_ids) == 4
        return float(session.version)

    async def materialize(session, name, load_into_vllm, turn_idx, *, adapter_kind="snapshot", set_latest=True):
        assert load_into_vllm is False
        meta = {
            "adapter_name": name,
            "adapter_path": f"/tmp/{name}",
            "adapter_kind": adapter_kind,
            "base_step": engine.base_step,
            "version": session.version,
            "turn_idx": turn_idx,
        }
        if set_latest:
            session.latest_adapter = meta
        return meta

    engine._train_chunk = train_chunk
    engine.materialize = materialize
    session = SimpleNamespace(
        session_id="rollout-1234567890",
        version=0,
        pending_token_ids=[],
        latest_adapter=None,
    )

    stats = asyncio.run(
        engine.append_and_train_with_replay_spans(
            session,
            token_ids=list(range(10)),
            replay_mask=[True] * 10,
            turn_idx=3,
        )
    )

    assert stats["trained_chunks"] == 2
    assert stats["trained_token_count"] == 8
    assert stats["pending_token_count"] == 2
    assert session.version == 2
    assert session.pending_token_ids == [8, 9]
    assert stats["prompt_replay_spans"] == [
        {
            "new_start": 0,
            "new_end": 4,
            "adapter_name": None,
            "adapter_path": None,
            "adapter_kind": "base",
            "base_step": 7,
            "adapter_version": 0,
        },
        {
            "new_start": 4,
            "new_end": 8,
            "adapter_name": "ttt-rollout-1234-t3-prompt-v1-b7",
            "adapter_path": "/tmp/ttt-rollout-1234-t3-prompt-v1-b7",
            "adapter_kind": "prompt_replay_snapshot",
            "base_step": 7,
            "adapter_version": 1,
        },
        {
            "new_start": 8,
            "new_end": 10,
            "adapter_name": "ttt-rollout-1234-t3-prompt-v2-b7",
            "adapter_path": "/tmp/ttt-rollout-1234-t3-prompt-v2-b7",
            "adapter_kind": "prompt_replay_snapshot",
            "base_step": 7,
            "adapter_version": 2,
        },
    ]


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
