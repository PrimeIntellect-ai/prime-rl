import pytest
from torch import nn

from prime_rl.trainer.model import freeze_moe_router
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs


class FakeLayer(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp


class FakeLanguageModel(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers


class FakeModel(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.model = language_model


def make_moe_model(num_layers: int = 2, dim: int = 64, hidden_dim: int = 128) -> nn.Module:
    moe_args = MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False, num_shared_experts=0)
    layers = nn.ModuleList([FakeLayer(MoE(moe_args, dim=dim, hidden_dim=hidden_dim)) for _ in range(num_layers)])
    return FakeModel(FakeLanguageModel(layers))


def test_freeze_moe_router_custom_impl():
    model = make_moe_model()

    freeze_moe_router(model)

    for layer in model.model.layers:
        for param in layer.mlp.router.parameters():
            assert not param.requires_grad, "Router parameters should be frozen"
        for param in layer.mlp.experts.parameters():
            assert param.requires_grad, "Expert parameters should remain trainable"


def test_freeze_moe_router_hf_gate():
    """Test freezing with HuggingFace-style gate attribute."""

    class HFMoEMLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.gate = nn.Linear(dim, 8, bias=False)
            self.experts = nn.Linear(dim, dim)

    dim = 64
    layers = nn.ModuleList([FakeLayer(HFMoEMLP(dim)) for _ in range(2)])
    model = FakeModel(FakeLanguageModel(layers))

    freeze_moe_router(model)

    for layer in model.model.layers:
        for param in layer.mlp.gate.parameters():
            assert not param.requires_grad, "Gate parameters should be frozen"
        for param in layer.mlp.experts.parameters():
            assert param.requires_grad, "Expert parameters should remain trainable"


def test_freeze_moe_router_raises_on_dense_model():
    """Test that freeze_moe_router raises when no router parameters are found."""

    class DenseMLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

    layers = nn.ModuleList([FakeLayer(DenseMLP(64)) for _ in range(2)])
    model = FakeModel(FakeLanguageModel(layers))

    with pytest.raises(ValueError, match="No MoE router parameters found"):
        freeze_moe_router(model)
