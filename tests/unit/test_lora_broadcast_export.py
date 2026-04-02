import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.ckpt import WeightCheckpointManager
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.runs import get_multi_run_manager, setup_multi_run_manager


def test_get_state_dict_for_run_stays_in_internal_key_space() -> None:
    with tempfile.TemporaryDirectory() as td:
        init_file = Path(td) / "pg_init"
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        try:
            setup_multi_run_manager(Path(td), 1, torch.device("cpu"), LoRAConfig(rank=8, target_modules=["q_proj"]))
            mgr = get_multi_run_manager()
            mod = MultiLoRALinear(nn.Linear(4, 6, bias=False), rank=8, n_adapters=1)
            mgr.register_module("model.layers.0.self_attn.q_proj", mod)

            state = mgr.get_state_dict_for_run(0)

            assert "model.layers.0.self_attn.q_proj.lora_A.weight" in state
            assert "model.layers.0.self_attn.q_proj.lora_B.weight" in state
            assert not any(key.startswith("base_model.model.") for key in state)
        finally:
            dist.destroy_process_group()


def test_get_state_dict_for_run_keeps_moe_keys_internal_and_unprefixed() -> None:
    class _FakeExperts(GroupedExperts):
        def __init__(self):
            nn.Module.__init__(self)
            self.w1 = torch.zeros((2, 4, 4))
            self.w2 = torch.zeros((2, 4, 4))
            self.w3 = torch.zeros((2, 4, 4))
            self.num_experts = 2
            self.hidden_size = 4
            self.intermediate_size = 4

        def forward(self, hidden_states, num_tokens_per_expert):
            raise NotImplementedError

    with tempfile.TemporaryDirectory() as td:
        init_file = Path(td) / "pg_init"
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        try:
            setup_multi_run_manager(Path(td), 1, torch.device("cpu"), LoRAConfig(rank=8, target_modules=["experts"]))
            mgr = get_multi_run_manager()
            mod = MultiLoRAGroupedExperts(_FakeExperts(), rank=8, n_adapters=1)
            mgr.register_module("model.layers.0.mlp.experts", mod)

            state = mgr.get_state_dict_for_run(0)

            assert "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight" in state
            assert "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight" in state
            assert "model.layers.0.mlp.experts.1.up_proj.lora_A.weight" in state
            assert not any(key.startswith("base_model.model.") for key in state)
        finally:
            dist.destroy_process_group()


def test_weight_checkpoint_adapter_export_adds_peft_prefix_exactly_once() -> None:
    with tempfile.TemporaryDirectory() as td:
        init_file = Path(td) / "pg_init"
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        try:
            lora_config = LoRAConfig(rank=8, target_modules=["q_proj"])
            setup_multi_run_manager(Path(td), 1, torch.device("cpu"), lora_config)
            mgr = get_multi_run_manager()
            mod = MultiLoRALinear(nn.Linear(4, 6, bias=False), rank=8, n_adapters=1)
            mgr.register_module("model.layers.0.self_attn.q_proj", mod)

            ckpt_config = type(
                "Cfg",
                (),
                {"save_format": "safetensors", "save_sharded": True, "save_adapter_separately": True},
            )()
            state = WeightCheckpointManager(Path(td), ckpt_config, lora_config).get_run_adapter_state_dict()

            assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in state
            assert "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" in state
            assert not any(key.startswith("base_model.model.base_model.model.") for key in state)
        finally:
            dist.destroy_process_group()
