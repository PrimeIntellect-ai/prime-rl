import torch

from prime_rl.inference.vllm.padded_input_scrub import _zero_padded_model_inputs


def test_zero_padded_model_inputs_leaves_scheduled_prefix_intact():
    input_ids = torch.tensor([10, 11, 12, 991, 992])
    positions = torch.tensor([0, 1, 2, 991, 992])
    preprocess_result = (5, input_ids, None, positions)

    _zero_padded_model_inputs(preprocess_result, num_scheduled_tokens=3, num_input_tokens=5)

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
    preprocess_result = (5, None, inputs_embeds, positions)

    _zero_padded_model_inputs(preprocess_result, num_scheduled_tokens=3, num_input_tokens=5)

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
    preprocess_result = (3, input_ids, None, positions)

    _zero_padded_model_inputs(preprocess_result, num_scheduled_tokens=3, num_input_tokens=3)

    assert input_ids.tolist() == [10, 11, 12]
    assert positions.tolist() == [0, 1, 2]
