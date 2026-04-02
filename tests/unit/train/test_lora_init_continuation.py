import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import (
    prepare_init_adapter,
)


class _FakeModel:
    pass


class _PreparedAdapterRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def apply_to_model(self, model, adapter_idx: int = 0) -> None:
        self.calls.append((model, adapter_idx))

    def register_creation_hook(self, model) -> None:
        if getattr(model, "_prime_init_adapter_creation_hook_registered", False):
            return

        def _apply_prepared_init_adapter(idx: int, _run_id: str) -> None:
            self.apply_to_model(model, adapter_idx=idx)

        from prime_rl.trainer.lora import get_multi_run_manager

        get_multi_run_manager().register_creation_hook(_apply_prepared_init_adapter)
        setattr(model, "_prime_init_adapter_creation_hook_registered", True)


def _write_adapter_config(adapter_dir: Path, rank: int = 1, alpha: int = 1) -> None:
    (adapter_dir / "adapter_config.json").write_text(
        f'{{"peft_type":"LORA","r":{rank},"lora_alpha":{alpha},"modules_to_save":null}}'
    )


def test_prepared_init_adapter_registers_creation_hook_for_created_run_slots() -> None:
    model = _FakeModel()
    prepared_adapter = _PreparedAdapterRecorder()
    creation_hooks = []

    class _Manager:
        def register_creation_hook(self, hook):
            creation_hooks.append(hook)

    from unittest.mock import patch

    with patch("prime_rl.trainer.lora.get_multi_run_manager", return_value=_Manager()):
        prepared_adapter.register_creation_hook(model)
        assert len(creation_hooks) == 1
        creation_hooks[0](12, "run_default")

    assert prepared_adapter.calls == [(model, 12)]


def test_prepared_init_adapter_registers_creation_hook_only_once() -> None:
    model = _FakeModel()
    prepared_adapter = _PreparedAdapterRecorder()
    creation_hooks = []

    class _Manager:
        def register_creation_hook(self, hook):
            creation_hooks.append(hook)

    from unittest.mock import patch

    with patch("prime_rl.trainer.lora.get_multi_run_manager", return_value=_Manager()):
        prepared_adapter.register_creation_hook(model)
        prepared_adapter.register_creation_hook(model)

    assert len(creation_hooks) == 1


def test_prepared_init_adapter_creation_hook_does_not_apply_current_slot() -> None:
    model = _FakeModel()
    prepared_adapter = _PreparedAdapterRecorder()
    creation_hooks = []

    class _Manager:
        def register_creation_hook(self, hook):
            creation_hooks.append(hook)

    from unittest.mock import patch

    with patch("prime_rl.trainer.lora.get_multi_run_manager", return_value=_Manager()):
        prepared_adapter.register_creation_hook(model)

    assert prepared_adapter.calls == []
    assert len(creation_hooks) == 1
    creation_hooks[0](7, "run_after_resume")
    assert prepared_adapter.calls == [(model, 7)]


def test_prepare_init_adapter_supports_dtensor_targets_and_preserves_values(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(adapter_dir)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = Path(tmpdir) / "pg_init"
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        try:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import Replicate, distribute_tensor

            mesh = init_device_mesh("cpu", (1,))

            class _Model:
                def __init__(self):
                    self._state = {
                        "model.layers.0.self_attn.q_proj.lora_A.0": distribute_tensor(
                            torch.zeros((1, 2), dtype=torch.float32), mesh, [Replicate()]
                        ),
                        "model.layers.0.self_attn.q_proj.lora_B.0": distribute_tensor(
                            torch.zeros((2, 1), dtype=torch.float32), mesh, [Replicate()]
                        ),
                        "model.layers.0.self_attn.q_proj.lora_A.3": distribute_tensor(
                            torch.zeros((1, 2), dtype=torch.float32), mesh, [Replicate()]
                        ),
                        "model.layers.0.self_attn.q_proj.lora_B.3": distribute_tensor(
                            torch.zeros((2, 1), dtype=torch.float32), mesh, [Replicate()]
                        ),
                    }
                    self.loaded = None

                def state_dict(self):
                    return self._state

                def load_state_dict(self, aligned, strict=False):
                    self.loaded = aligned

            model = _Model()
            prepared = prepare_init_adapter(model, adapter_dir, LoRAConfig(rank=1, alpha=1))
            prepared.apply_to_model(model, adapter_idx=3)

            assert model.loaded is not None
            assert all(hasattr(value, "device_mesh") for value in model.loaded.values())
            assert torch.equal(
                model.loaded["model.layers.0.self_attn.q_proj.lora_A.3"].to_local(),
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            )
            assert torch.equal(
                model.loaded["model.layers.0.self_attn.q_proj.lora_B.3"].to_local(),
                torch.tensor([[3.0], [4.0]], dtype=torch.float32),
            )
        finally:
            dist.destroy_process_group()


def test_prepare_init_adapter_nonzero_adapter_idx_preserves_layer_zero_path(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(adapter_dir)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    class _Model:
        def __init__(self):
            self._state = {
                "model.layers.0.self_attn.q_proj.lora_A.0": torch.zeros((1, 2)),
                "model.layers.0.self_attn.q_proj.lora_B.0": torch.zeros((2, 1)),
                "model.layers.0.self_attn.q_proj.lora_A.12": torch.zeros((1, 2)),
                "model.layers.0.self_attn.q_proj.lora_B.12": torch.zeros((2, 1)),
            }
            self.loaded = None

        def state_dict(self):
            return self._state

        def load_state_dict(self, aligned, strict=False):
            self.loaded = aligned

    model = _Model()
    prepared = prepare_init_adapter(model, adapter_dir, LoRAConfig(rank=1, alpha=1))
    prepared.apply_to_model(model, adapter_idx=12)

    assert model.loaded is not None
    assert set(model.loaded) == {
        "model.layers.0.self_attn.q_proj.lora_A.12",
        "model.layers.0.self_attn.q_proj.lora_B.12",
    }


def test_prepare_init_adapter_nonzero_adapter_idx_preserves_moe_layer_zero_path_and_values(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(adapter_dir)
    save_file(
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": torch.tensor([[5.0, 6.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.1.gate_proj.lora_B.weight": torch.tensor([[7.0], [8.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.tensor([[11.0, 12.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_B.weight": torch.tensor([[13.0], [14.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.1.down_proj.lora_A.weight": torch.tensor([[15.0, 16.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.1.down_proj.lora_B.weight": torch.tensor([[17.0], [18.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight": torch.tensor([[21.0, 22.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_B.weight": torch.tensor([[23.0], [24.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.1.up_proj.lora_A.weight": torch.tensor([[25.0, 26.0]], dtype=torch.float64),
            "base_model.model.model.layers.0.mlp.experts.1.up_proj.lora_B.weight": torch.tensor([[27.0], [28.0]], dtype=torch.float64),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    class _Model:
        def __init__(self):
            stacked = torch.zeros((2, 1, 2), dtype=torch.float32)
            stacked_b = torch.zeros((2, 2, 1), dtype=torch.float32)
            self._state = {
                "model.layers.0.mlp.experts.w1_lora_A.0": stacked.clone(),
                "model.layers.0.mlp.experts.w1_lora_B.0": stacked_b.clone(),
                "model.layers.0.mlp.experts.w2_lora_A.0": stacked.clone(),
                "model.layers.0.mlp.experts.w2_lora_B.0": stacked_b.clone(),
                "model.layers.0.mlp.experts.w3_lora_A.0": stacked.clone(),
                "model.layers.0.mlp.experts.w3_lora_B.0": stacked_b.clone(),
                "model.layers.0.mlp.experts.w1_lora_A.4": stacked.clone(),
                "model.layers.0.mlp.experts.w1_lora_B.4": stacked_b.clone(),
                "model.layers.0.mlp.experts.w2_lora_A.4": stacked.clone(),
                "model.layers.0.mlp.experts.w2_lora_B.4": stacked_b.clone(),
                "model.layers.0.mlp.experts.w3_lora_A.4": stacked.clone(),
                "model.layers.0.mlp.experts.w3_lora_B.4": stacked_b.clone(),
            }
            self.loaded = None

        def state_dict(self):
            return self._state

        def load_state_dict(self, aligned, strict=False):
            self.loaded = aligned

    model = _Model()
    prepared = prepare_init_adapter(model, adapter_dir, LoRAConfig(rank=1, alpha=1))
    prepared.apply_to_model(model, adapter_idx=4)

    assert model.loaded is not None
    assert set(model.loaded) == {
        "model.layers.0.mlp.experts.w1_lora_A.4",
        "model.layers.0.mlp.experts.w1_lora_B.4",
        "model.layers.0.mlp.experts.w2_lora_A.4",
        "model.layers.0.mlp.experts.w2_lora_B.4",
        "model.layers.0.mlp.experts.w3_lora_A.4",
        "model.layers.0.mlp.experts.w3_lora_B.4",
    }
    assert torch.equal(
        model.loaded["model.layers.0.mlp.experts.w1_lora_A.4"],
        torch.tensor([[[1.0, 2.0]], [[5.0, 6.0]]], dtype=torch.float32),
    )
    assert torch.equal(
        model.loaded["model.layers.0.mlp.experts.w2_lora_A.4"],
        torch.tensor([[[11.0, 12.0]], [[15.0, 16.0]]], dtype=torch.float32),
    )
    assert torch.equal(
        model.loaded["model.layers.0.mlp.experts.w3_lora_B.4"],
        torch.tensor([[[23.0], [24.0]], [[27.0], [28.0]]], dtype=torch.float32),
    )


def test_prepared_init_adapter_is_reused_without_reloading_files(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_adapter_config(adapter_dir)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    class _Model:
        def __init__(self):
            self._state = {
                "model.layers.0.self_attn.q_proj.lora_A.0": torch.zeros((1, 2)),
                "model.layers.0.self_attn.q_proj.lora_B.0": torch.zeros((2, 1)),
                "model.layers.0.self_attn.q_proj.lora_A.1": torch.zeros((1, 2)),
                "model.layers.0.self_attn.q_proj.lora_B.1": torch.zeros((2, 1)),
            }
            self.loaded = None

        def state_dict(self):
            return self._state

        def load_state_dict(self, aligned, strict=False):
            self.loaded = aligned

    model = _Model()
    prepared = prepare_init_adapter(model, adapter_dir, LoRAConfig(rank=1, alpha=1))

    from unittest.mock import patch

    with patch("prime_rl.trainer.lora.load_file", side_effect=AssertionError("adapter files should not be re-read")):
        prepared.apply_to_model(model, adapter_idx=0)
        prepared.apply_to_model(model, adapter_idx=1)

    assert model.loaded is not None


def test_prepare_init_adapter_rejects_modules_to_save(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":1,"lora_alpha":1,"modules_to_save":["lm_head"]}'
    )
    save_file({}, str(adapter_dir / "adapter_model.safetensors"))

    class _Model:
        def state_dict(self):
            return {}

    with pytest.raises(ValueError, match="modules_to_save"):
        prepare_init_adapter(_Model(), adapter_dir, LoRAConfig(rank=1, alpha=1))
