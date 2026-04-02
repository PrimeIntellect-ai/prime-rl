import torch
from torch import nn

from prime_rl.trainer.lora import merge_lora_state_dict
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens, set_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts


def _set_lora_globals(num_adapters: int, scaling_factors: list[float]) -> None:
    set_lora_num_tokens(torch.ones(num_adapters, dtype=torch.int32), reset_reference=True)
    set_multilora_scaling(torch.tensor(scaling_factors, dtype=torch.bfloat16), reset_reference=True)


def test_merge_lora_state_dict_merges_linear_weights() -> None:
    _set_lora_globals(num_adapters=2, scaling_factors=[0.5, 1.5])

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = MultiLoRALinear(nn.Linear(3, 2, bias=False), rank=2, n_adapters=2)
            self.modules_to_save = nn.Linear(2, 2, bias=False)

    model = Model()
    base_weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    lora_a_run_0 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    lora_b_run_0 = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    lora_a_run_1 = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
        ]
    )
    lora_b_run_1 = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ]
    )
    modules_to_save_weight = torch.tensor(
        [
            [7.0, 8.0],
            [9.0, 10.0],
        ]
    )

    merged_state_dict = merge_lora_state_dict(
        model,
        {
            "linear.base_layer.weight": base_weight,
            "linear.lora_A.0": lora_a_run_0,
            "linear.lora_B.0": lora_b_run_0,
            "linear.lora_A.1": lora_a_run_1,
            "linear.lora_B.1": lora_b_run_1,
            "modules_to_save.weight": modules_to_save_weight,
        },
        run_idx=1,
    )

    expected_weight = base_weight + 1.5 * torch.matmul(lora_b_run_1, lora_a_run_1)
    torch.testing.assert_close(merged_state_dict["linear.weight"], expected_weight)
    torch.testing.assert_close(merged_state_dict["modules_to_save.weight"], modules_to_save_weight)
    assert all("lora_A" not in key and "lora_B" not in key for key in merged_state_dict)
    assert "linear.base_layer.weight" not in merged_state_dict


def test_merge_lora_state_dict_merges_grouped_expert_weights() -> None:
    _set_lora_globals(num_adapters=1, scaling_factors=[0.5])

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = MultiLoRAGroupedExperts(
                GroupedExperts(dim=3, hidden_dim=4, num_experts=2, use_grouped_mm=False),
                rank=2,
                n_adapters=1,
            )

    model = Model()
    base_w1 = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    base_w2 = torch.arange(24, 48, dtype=torch.float32).view(2, 3, 4)
    base_w3 = torch.arange(48, 72, dtype=torch.float32).view(2, 4, 3)

    w1_lora_a = torch.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],
            [[2.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        ]
    )
    w1_lora_b = torch.tensor(
        [
            [[1.0, 2.0], [0.0, 1.0], [2.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0], [1.0, 1.0]],
        ]
    )
    w2_lora_a = torch.tensor(
        [
            [[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
            [[2.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 2.0]],
        ]
    )
    w2_lora_b = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
            [[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]],
        ]
    )
    w3_lora_a = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
            [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
        ]
    )
    w3_lora_b = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]],
        ]
    )

    merged_state_dict = merge_lora_state_dict(
        model,
        {
            "moe.base_layer.w1": base_w1,
            "moe.base_layer.w2": base_w2,
            "moe.base_layer.w3": base_w3,
            "moe.w1_lora_A.0": w1_lora_a,
            "moe.w1_lora_B.0": w1_lora_b,
            "moe.w2_lora_A.0": w2_lora_a,
            "moe.w2_lora_B.0": w2_lora_b,
            "moe.w3_lora_A.0": w3_lora_a,
            "moe.w3_lora_B.0": w3_lora_b,
        },
    )

    torch.testing.assert_close(merged_state_dict["moe.w1"], base_w1 + 0.5 * torch.matmul(w1_lora_b, w1_lora_a))
    torch.testing.assert_close(merged_state_dict["moe.w2"], base_w2 + 0.5 * torch.matmul(w2_lora_b, w2_lora_a))
    torch.testing.assert_close(merged_state_dict["moe.w3"], base_w3 + 0.5 * torch.matmul(w3_lora_b, w3_lora_a))
    assert all("lora_A" not in key and "lora_B" not in key for key in merged_state_dict)
    assert all(".base_layer." not in key for key in merged_state_dict)
