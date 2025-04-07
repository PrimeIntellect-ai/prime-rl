from zeroband.models import get_model_and_tokenizer

import torch

import pytest


def test_model():
    model, tokenizer = get_model_and_tokenizer("PrimeIntellect/llama-2m-fresh", "flash_attention_2")
    assert model is not None

    BS = 2
    SEQ_LEN = 16

    model = model.to("cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")

        outputs = model(input_ids=inputs_ids).logits

        assert outputs.shape == (BS, SEQ_LEN, len(tokenizer))


def test_model_with_position_ids():
    model, tokenizer = get_model_and_tokenizer("PrimeIntellect/llama-2m-fresh", "flash_attention_2")
    assert model is not None

    BS = 2
    SEQ_LEN = 16

    model = model.to("cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).repeat(BS, 1).to("cuda")

        outputs = model(input_ids=inputs_ids, position_ids=position_ids).logits

        assert outputs.shape == (BS, SEQ_LEN, len(tokenizer))


@pytest.mark.parametrize("correct_position_ids", [True, False])
def test_model_with_sequence_packing(correct_position_ids):
    """
    The goal of this test is to check that the sequence packing works correctly.

    The idea is that is to check that the logits is the same when doing

    [B, seq]  and doing [1, B*seq] with the proper masking.

    """

    model, tokenizer = get_model_and_tokenizer("PrimeIntellect/llama-2m-fresh", "flash_attention_2")
    assert model is not None

    model = model.to("cuda")

    inputs = [0, 1, 2, 3]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.Tensor(inputs).repeat(1, 1).int().to("cuda")
        output_base = model(input_ids=inputs_ids).logits

        assert output_base.shape == (1, len(inputs), len(tokenizer))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.Tensor(inputs + inputs).repeat(1, 1).int().to("cuda")
        if correct_position_ids:
            position_ids = torch.Tensor([0, 1, 2, 3, 0, 1, 2, 3]).repeat(1, 1).int().to("cuda")
            # should work
        else:
            position_ids = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(1, 1).int().to("cuda")
            # should fail
        outputs_packed = model(input_ids=inputs_ids, position_ids=position_ids).logits

        assert outputs_packed.shape == (1, 2 * len(inputs), len(tokenizer))

    output_packed_left = outputs_packed[:, : len(inputs), :]
    output_packed_right = outputs_packed[:, len(inputs) :, :]

    assert output_packed_left.shape == output_base.shape == output_packed_right.shape

    if correct_position_ids:
        torch.testing.assert_close(output_packed_left, output_base)
        torch.testing.assert_close(output_packed_right, output_base)
    else:
        torch.testing.assert_close(output_packed_left, output_base)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(output_packed_right, output_base)
