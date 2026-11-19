from torch import nn

from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear, replace_linear_with_fp8_blockwise_linear


def test_the_fp8_swap_keeps_biased_linear_layers():
    model = nn.Sequential(nn.Linear(128, 128, bias=False), nn.Linear(128, 128))

    replace_linear_with_fp8_blockwise_linear(model, ignore_modules=[])

    assert type(model[0]) is Float8BlockwiseLinear
    assert type(model[1]) is nn.Linear and model[1].bias is not None
