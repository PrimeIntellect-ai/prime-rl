import torch
import torch.nn as nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import freeze_all_except_lora_and_specified
from prime_rl.trainer.model import unfreeze_vision_encoder


class DummyVisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class DummyInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = DummyVisualEncoder()


class DummyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DummyInnerModel()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 2))


def test_unfreeze_vision_encoder_after_lora_freeze() -> None:
    model = DummyVLM()

    freeze_all_except_lora_and_specified(model, LoRAConfig())
    assert not model.model.visual.proj.weight.requires_grad
    assert not model.base.weight.requires_grad
    assert model.lora_A.requires_grad
    assert model.lora_B.requires_grad

    unfreeze_vision_encoder(model)
    assert model.model.visual.proj.weight.requires_grad
    assert model.model.visual.proj.bias.requires_grad
    assert not model.base.weight.requires_grad
    assert not model.base.bias.requires_grad
