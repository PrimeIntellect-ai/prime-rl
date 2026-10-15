import torch
from torch import nn

from prime_rl.trainer.pretrain.metrics import expert_load_metrics, grad_norms_by_group, param_group


def test_expert_load_metrics_per_layer_summaries():
    counts = torch.tensor([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    metrics = expert_load_metrics(counts, layer_indices=[1, 2])
    assert metrics["expert_load/min_mean/min"] == 0.0
    assert metrics["expert_load/entropy/mean"] == 0.5  # uniform layer 1, one-hot layer 0
    assert metrics["expert_load/layer_2/max_mean"] == 4.0
    assert metrics["expert_load/layer_1/top_share"] == 0.25
    assert metrics["expert_load/layer_2/top_share"] == 1.0


def test_grad_norms_by_group_add_up_to_the_total_norm():
    model = nn.ModuleDict(
        {
            "embed_tokens": nn.Embedding(8, 4),
            "self_attn": nn.Linear(4, 4),
            "input_layernorm": nn.LayerNorm(4),
            "lm_head": nn.Linear(4, 8),
        }
    )
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    norms = grad_norms_by_group(model)
    assert set(norms) == {"embed", "attn", "norm", "lm_head"}
    total = torch.nn.utils.get_total_norm([param.grad for param in model.parameters()])
    assert torch.allclose(torch.stack(list(norms.values())).square().sum().sqrt().float(), total)
    assert param_group("model.layers.3.mlp.router.gate.weight") == "router"
    assert param_group("model.layers.3.mlp.shared_expert.w1") == "shared_experts"
    assert param_group("model.layers.3.mlp.experts.w1") == "experts"
