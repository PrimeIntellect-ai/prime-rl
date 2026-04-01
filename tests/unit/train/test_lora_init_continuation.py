import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from prime_rl.configs.trainer import LoRAConfig
import pytest

from prime_rl.trainer.lora import (
    _is_model_lora_key_for_adapter,
    _set_adapter_idx_suffix,
    load_init_adapter_weights,
    register_init_adapter_reload_hook,
)


class _FakeModel:
    pass


def test_register_init_adapter_reload_hook_reloads_created_run_slot(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA","r":16,"lora_alpha":32,"modules_to_save":null}')

    config = LoRAConfig(rank=16, alpha=32, init_adapter_path=adapter_dir)
    model = _FakeModel()
    creation_hooks = []
    fake_manager = SimpleNamespace(register_creation_hook=creation_hooks.append)

    with patch("prime_rl.trainer.lora.get_multi_run_manager", return_value=fake_manager), patch(
        "prime_rl.trainer.lora.load_init_adapter_weights"
    ) as load_mock:
        register_init_adapter_reload_hook(model, config)
        assert len(creation_hooks) == 1
        creation_hooks[0](0, "run_default")

    load_mock.assert_called_once_with(model, adapter_dir, config, adapter_idx=0)


def test_register_init_adapter_reload_hook_only_registers_once(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA","r":16,"lora_alpha":32,"modules_to_save":null}')

    config = LoRAConfig(rank=16, alpha=32, init_adapter_path=adapter_dir)
    model = _FakeModel()
    creation_hooks = []
    fake_manager = SimpleNamespace(register_creation_hook=creation_hooks.append)

    with patch("prime_rl.trainer.lora.get_multi_run_manager", return_value=fake_manager):
        register_init_adapter_reload_hook(model, config)
        register_init_adapter_reload_hook(model, config)

    assert len(creation_hooks) == 1


def test_set_adapter_idx_suffix_only_rewrites_adapter_slot_suffix() -> None:
    assert _set_adapter_idx_suffix("model.layers.0.self_attn.q_proj.lora_A.0", 3) == "model.layers.0.self_attn.q_proj.lora_A.3"
    assert _set_adapter_idx_suffix("model.layers.0.mlp.experts.w1_lora_A.0", 2) == "model.layers.0.mlp.experts.w1_lora_A.2"


def test_set_adapter_idx_suffix_rejects_missing_suffix() -> None:
    with pytest.raises(ValueError, match="adapter-slot suffix"):
        _set_adapter_idx_suffix("model.layers.0.self_attn.q_proj.lora_A", 1)


def test_is_model_lora_key_for_adapter_uses_exact_suffix_matching() -> None:
    assert _is_model_lora_key_for_adapter("model.layers.0.self_attn.q_proj.lora_A.10", 10)
    assert not _is_model_lora_key_for_adapter("model.layers.0.self_attn.q_proj.lora_A.10", 1)
    assert _is_model_lora_key_for_adapter("model.layers.0.mlp.experts.w2_lora_B.12", 12)
    assert not _is_model_lora_key_for_adapter("model.layers.0.mlp.experts.w2_lora_B.12", 2)


def test_load_init_adapter_weights_supports_dtensor_targets_and_preserves_values(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA","r":1,"lora_alpha":1,"modules_to_save":null}')
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
            load_init_adapter_weights(model, adapter_dir, LoRAConfig(rank=1, alpha=1), adapter_idx=3)

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


def test_load_init_adapter_weights_nonzero_adapter_idx_preserves_layer_zero_path(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA","r":1,"lora_alpha":1,"modules_to_save":null}')
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
                "model.layers.0.self_attn.q_proj.lora_A.3": torch.zeros((1, 2)),
                "model.layers.0.self_attn.q_proj.lora_B.3": torch.zeros((2, 1)),
            }
            self.loaded = None

        def state_dict(self):
            return self._state

        def load_state_dict(self, aligned, strict=False):
            self.loaded = aligned

    model = _Model()
    load_init_adapter_weights(model, adapter_dir, LoRAConfig(rank=1, alpha=1), adapter_idx=3)

    assert model.loaded is not None
    assert set(model.loaded) == {
        "model.layers.0.self_attn.q_proj.lora_A.3",
        "model.layers.0.self_attn.q_proj.lora_B.3",
    }


def test_load_init_adapter_weights_nonzero_adapter_idx_preserves_moe_layer_zero_path_and_values(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA","r":1,"lora_alpha":1,"modules_to_save":null}')
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
    load_init_adapter_weights(model, adapter_dir, LoRAConfig(rank=1, alpha=1), adapter_idx=4)

    assert model.loaded is not None
    assert set(model.loaded) == set(model._state)
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
