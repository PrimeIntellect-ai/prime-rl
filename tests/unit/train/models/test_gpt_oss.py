from types import SimpleNamespace

import torch

from prime_rl.trainer.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter


def test_gpt_oss_router_counts_are_integer_bincounts():
    router = GptOssTopKRouter(SimpleNamespace(num_local_experts=4, num_experts_per_tok=2, hidden_size=3))
    with torch.no_grad():
        router.weight.zero_()
        router.bias.copy_(torch.tensor([0.0, 3.0, 2.0, 1.0]))

    _, top_indices, num_tokens_per_expert = router(torch.zeros(5, 3))

    assert top_indices.tolist() == [[1, 2]] * 5
    assert num_tokens_per_expert.dtype == torch.int64
    torch.testing.assert_close(num_tokens_per_expert, torch.tensor([0, 5, 5, 0]))
