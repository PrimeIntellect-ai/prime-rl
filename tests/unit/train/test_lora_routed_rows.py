"""MultiLoRALinear on routed-MoE rows (e.g. NemotronH fc1/fc2_latent_proj).

Inside the routed experts path the input rows are top-k-expanded and expert-sorted,
so their count doesn't match the per-run token layout. The layer must fall back to
the dominant adapter instead of asserting (the same approximation
MultiLoRAGroupedExperts uses for routed tokens); for single-adapter training this
is exact.
"""

import torch
from torch import nn

from prime_rl.trainer.models.layers.lora import base as lora_base
from prime_rl.trainer.models.layers.lora.base import set_lora_num_tokens, set_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear


def _make_layer(n_adapters: int) -> MultiLoRALinear:
    torch.manual_seed(0)
    # Globals must exist before construction (the layer captures references at init), and
    # both must be cleared together: the setters cross-assert shapes against each other.
    lora_base.LORA_NUM_TOKENS = None
    lora_base.SCALING_FACTORS = None
    set_lora_num_tokens(torch.tensor([6] * n_adapters, dtype=torch.int32), reset_reference=True)
    set_multilora_scaling(torch.full((n_adapters,), 0.5), reset_reference=True)
    base = nn.Linear(16, 8, bias=False)
    return MultiLoRALinear(base, rank=4, n_adapters=n_adapters, alpha=8.0, use_grouped_mm=False)


def test_routed_row_count_uses_dominant_adapter():
    layer = _make_layer(n_adapters=1)
    # 6 tokens * top-3 experts = 18 routed rows != 6 tokens; must not assert.
    x = torch.randn(18, 16)
    with torch.no_grad():
        layer.lora_B[0].normal_()  # nonzero so the adapter actually contributes
        out = layer(x)
        expected = layer.base_layer(x) + 0.5 * (x @ layer.lora_A[0].T) @ layer.lora_B[0].T
    torch.testing.assert_close(out, expected)


def test_matching_row_count_keeps_offset_path():
    layer = _make_layer(n_adapters=1)
    x = torch.randn(6, 16)
    with torch.no_grad():
        layer.lora_B[0].normal_()
        out = layer(x)
        expected = layer.base_layer(x) + 0.5 * (x @ layer.lora_A[0].T) @ layer.lora_B[0].T
    torch.testing.assert_close(out, expected)


def test_dominant_adapter_selection_multi_run():
    layer = _make_layer(n_adapters=2)
    layer._lora_num_tokens = torch.tensor([2, 10], dtype=torch.int32)
    layer._scaling_factors = torch.tensor([0.5, 0.25])
    x = torch.randn(30, 16)  # != 12 tokens -> routed fallback
    with torch.no_grad():
        layer.lora_B[0].normal_()
        layer.lora_B[1].normal_()
        out = layer(x)
        expected = layer.base_layer(x) + 0.25 * (x @ layer.lora_A[1].T) @ layer.lora_B[1].T
    torch.testing.assert_close(out, expected)
