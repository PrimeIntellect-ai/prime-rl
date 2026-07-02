from types import SimpleNamespace

import torch

import prime_rl.inference.vllm.padded_input_scrub as padded_input_scrub
from prime_rl.inference.vllm.padded_input_scrub import (
    _zero_padded_model_inputs,
    monkey_patch_vllm_padded_input_scrub,
)


def test_monkey_patch_preprocess_matches_vllm_preprocess_signature(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    calls = {}

    def fake_preprocess(
        self,
        scheduler_output,
        num_input_tokens,
        intermediate_tensors=None,
    ):
        calls["args"] = (
            self,
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )
        return (
            torch.tensor([10, 11, 12, 991, 992]),
            None,
            torch.tensor([0, 1, 2, 991, 992]),
            intermediate_tensors,
            {},
            None,
        )

    monkeypatch.setattr(padded_input_scrub, "_INSTALLED", False)
    monkeypatch.setattr(GPUModelRunner, "_preprocess", fake_preprocess)
    monkeypatch.setattr(
        GPUModelRunner,
        "_prime_rl_padded_input_scrub",
        False,
        raising=False,
    )

    monkey_patch_vllm_padded_input_scrub()

    self = object()
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=3)
    intermediate_tensors = object()
    result = GPUModelRunner._preprocess(self, scheduler_output, 5, intermediate_tensors)

    assert calls["args"] == (
        self,
        scheduler_output,
        5,
        intermediate_tensors,
    )
    assert result[0].tolist() == [10, 11, 12, 0, 0]
    assert result[2].tolist() == [0, 1, 2, 0, 0]


def test_zero_padded_model_inputs_leaves_scheduled_prefix_intact():
    input_ids = torch.tensor([10, 11, 12, 991, 992])
    positions = torch.tensor([0, 1, 2, 991, 992])
    preprocess_result = (input_ids, None, positions, None, {}, None)

    _zero_padded_model_inputs(
        preprocess_result,
        num_scheduled_tokens=3,
        num_input_tokens=5,
    )

    assert input_ids.tolist() == [10, 11, 12, 0, 0]
    assert positions.tolist() == [0, 1, 2, 0, 0]


def test_zero_padded_model_inputs_zeroes_embeddings_and_mrope_positions():
    inputs_embeds = torch.ones((5, 4))
    inputs_embeds[3:] = 7
    positions = torch.tensor(
        [
            [0, 1, 2, 991, 992],
            [0, 1, 2, 993, 994],
            [0, 1, 2, 995, 996],
        ]
    )
    preprocess_result = (None, inputs_embeds, positions, None, {}, None)

    _zero_padded_model_inputs(
        preprocess_result,
        num_scheduled_tokens=3,
        num_input_tokens=5,
    )

    torch.testing.assert_close(inputs_embeds[:3], torch.ones((3, 4)))
    torch.testing.assert_close(inputs_embeds[3:], torch.zeros((2, 4)))
    assert positions.tolist() == [
        [0, 1, 2, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 1, 2, 0, 0],
    ]


def test_zero_padded_model_inputs_noops_without_padding():
    input_ids = torch.tensor([10, 11, 12])
    positions = torch.tensor([0, 1, 2])
    preprocess_result = (input_ids, None, positions, None, {}, None)

    _zero_padded_model_inputs(
        preprocess_result,
        num_scheduled_tokens=3,
        num_input_tokens=3,
    )

    assert input_ids.tolist() == [10, 11, 12]
    assert positions.tolist() == [0, 1, 2]
