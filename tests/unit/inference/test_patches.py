import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from prime_rl.inference.patches import monkey_patch_fp32_lm_head


def _install_fake_vllm(monkeypatch, *, fp32_enabled: bool, supports_skip_gather: bool):
    class LogitsProcessor:
        def __init__(self):
            self.org_vocab_size = 2
            self.gather_calls = 0

        def _gather_logits(self, logits):
            self.gather_calls += 1
            return logits

    if supports_skip_gather:

        def _get_logits(self, hidden_states, lm_head, embedding_bias, skip_gather=False):
            self.original_call = (hidden_states, lm_head, embedding_bias, skip_gather)
            return "original"

    else:

        def _get_logits(self, hidden_states, lm_head, embedding_bias):
            self.original_call = (hidden_states, lm_head, embedding_bias)
            return "original"

    LogitsProcessor._get_logits = _get_logits

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.config": ModuleType("vllm.config"),
        "vllm.logger": ModuleType("vllm.logger"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.logits_processor": ModuleType("vllm.model_executor.layers.logits_processor"),
    }
    modules["vllm.config"].get_current_vllm_config = lambda: SimpleNamespace(
        additional_config={"fp32_lm_head": fp32_enabled}
    )
    modules["vllm.logger"].init_logger = lambda _name: MagicMock()
    modules["vllm.model_executor.layers.logits_processor"].LogitsProcessor = LogitsProcessor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkey_patch_fp32_lm_head()
    return LogitsProcessor


@pytest.mark.parametrize("supports_skip_gather", [False, True])
def test_fp32_lm_head_patch_delegates_across_vllm_signatures(monkeypatch, supports_skip_gather):
    processor_type = _install_fake_vllm(
        monkeypatch,
        fp32_enabled=False,
        supports_skip_gather=supports_skip_gather,
    )
    processor = processor_type()
    hidden_states = object()
    lm_head = object()
    embedding_bias = object()

    if supports_skip_gather:
        result = processor._get_logits(hidden_states, lm_head, embedding_bias, skip_gather=True)
    else:
        result = processor._get_logits(hidden_states, lm_head, embedding_bias)

    assert result == "original"
    expected = (
        (hidden_states, lm_head, embedding_bias, True)
        if supports_skip_gather
        else (
            hidden_states,
            lm_head,
            embedding_bias,
        )
    )
    assert processor.original_call == expected


@pytest.mark.parametrize("supports_skip_gather", [False, True])
def test_fp32_lm_head_patch_skips_gather_across_vllm_signatures(monkeypatch, supports_skip_gather):
    processor_type = _install_fake_vllm(
        monkeypatch,
        fp32_enabled=True,
        supports_skip_gather=supports_skip_gather,
    )
    processor = processor_type()
    local_logits = torch.tensor([[1.0, 2.0, 3.0]])
    monkeypatch.setattr(torch, "mm", MagicMock(return_value=local_logits))
    hidden_states = torch.ones(1, 2, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=torch.ones(3, 2, dtype=torch.bfloat16), tp_size=2)

    result = processor._get_logits(hidden_states, lm_head, None, skip_gather=True)

    assert result is local_logits
    assert processor.gather_calls == 0


def test_fp32_lm_head_patch_gathers_and_trims_by_default(monkeypatch):
    processor_type = _install_fake_vllm(
        monkeypatch,
        fp32_enabled=True,
        supports_skip_gather=True,
    )
    processor = processor_type()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    monkeypatch.setattr(torch, "mm", MagicMock(return_value=logits))
    hidden_states = torch.ones(1, 2, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=torch.ones(3, 2, dtype=torch.bfloat16), tp_size=2)

    result = processor._get_logits(hidden_states, lm_head, None)

    assert processor.gather_calls == 1
    assert torch.equal(result, logits[..., :2])
