import torch.nn as nn

from prime_rl.trainer.rl.broadcast.nixl.nixl import get_keep_in_fp32_for_weight_transfer


def test_generic_hugging_face_model_uses_default_weight_transfer_dtype() -> None:
    keep_in_fp32 = get_keep_in_fp32_for_weight_transfer(nn.Linear(2, 2))

    assert keep_in_fp32("weight") is False


def test_prime_model_weight_transfer_dtype_hook_is_preserved() -> None:
    class ModelWithHook(nn.Module):
        @staticmethod
        def keep_in_fp32_for_weight_transfer(name: str) -> bool:
            return name == "keep"

    keep_in_fp32 = get_keep_in_fp32_for_weight_transfer(ModelWithHook())

    assert keep_in_fp32("keep") is True
    assert keep_in_fp32("convert") is False
