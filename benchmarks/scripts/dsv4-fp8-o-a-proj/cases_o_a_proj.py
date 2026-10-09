import torch

from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4GroupedLinear
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseGroupedLinear

GROUPS, IN_FEATURES, OUT_PER_GROUP = 8, 4096, 1024
TOKENS_PER_GPU = (4096, 14336)


def _fwd(module, x):
    def run():
        module(x)

    return run


def _fwd_bwd(module, x, grad):
    def run():
        x.grad = None
        module.weight.grad = None
        module(x).backward(grad)

    return run


def cases():
    out = []
    for tokens in TOKENS_PER_GPU:
        bf16 = DeepseekV4GroupedLinear(IN_FEATURES, GROUPS * OUT_PER_GROUP, GROUPS).cuda().to(torch.bfloat16)
        torch.nn.init.normal_(bf16.weight, std=0.02)
        fp8 = Float8BlockwiseGroupedLinear.from_grouped_linear(bf16)
        x = torch.randn(1, tokens, GROUPS, IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(1, tokens, GROUPS, OUT_PER_GROUP, device="cuda", dtype=torch.bfloat16)
        out.append((f"T={tokens} fwd", {"bf16 bmm": _fwd(bf16, x), "fp8 einsum": _fwd(fp8, x)}))
        out.append((f"T={tokens} fwd+bwd", {"bf16 bmm": _fwd_bwd(bf16, x, grad), "fp8 einsum": _fwd_bwd(fp8, x, grad)}))
    return out
