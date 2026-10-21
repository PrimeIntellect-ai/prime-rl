import torch

from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.prime_moe.moe import PrimeRouter, histogram_quantile_biases, sequence_balance
from prime_rl.trainer.pretrain.train import update_selection_biases


def test_update_selection_biases_moves_towards_underloaded_experts():
    moe = MoE.from_args(MoEArgs(num_experts=4), dim=4, hidden_dim=8, shared_expert=None)
    layer = torch.nn.Module()
    layer.mlp = moe
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])

    update_selection_biases(model, torch.tensor([[9.0, 1.0, 1.0, 1.0]]), rate=0.1)
    update_selection_biases(model, torch.tensor([[9.0, 1.0, 1.0, 1.0]]), rate=0.1)

    # sign(mean - load) = [-1, 1, 1, 1], centred to [-1.5, 0.5, 0.5, 0.5] per step
    torch.testing.assert_close(moe.router.selection_bias, torch.tensor([-0.3, 0.1, 0.1, 0.1]))


def test_sequence_balance():
    scores = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    selected = torch.tensor([[0], [0], [0], [1]])

    # row 0 sends both tokens to expert 0: f = [2, 0], P = [0.9, 0.1]; row 1 is balanced: f = [1, 1], P = [0.5, 0.5]
    torch.testing.assert_close(sequence_balance(scores, selected, num_rows=2), torch.tensor([1.8, 1.0]))
    # the same tokens as one sequence (a packed sample of both rows) hide row 0's imbalance: f = [1.5, 0.5], P = [0.7, 0.3]
    torch.testing.assert_close(sequence_balance(scores, selected, num_rows=1), torch.tensor([1.2]))


def test_quantile_balancing_balances_the_recorded_batch():
    torch.manual_seed(0)
    router = PrimeRouter(
        dim=16, num_experts=6, num_null_experts=2, top_k=2, route_norm=True, route_scale=1.0, bias_on_logits=True
    )
    with torch.no_grad():
        router.gate.weight[0] += 0.5  # expert 0 is preferred
    router.quantile_balance = True
    x = torch.randn(400, 16)
    counts = router(x)[2]
    assert counts.max() / counts.float().mean() > 1.4

    # One quantile update brings every expert, null experts included, close to 400 * 2 / 8 = 100 tokens.
    thresholds = router.quantile_sum / router.quantile_count
    router.selection_bias.copy_(thresholds.mean() - thresholds)
    with torch.no_grad():
        counts = router(x)[2]
    assert counts.max() / counts.float().mean() < 1.15


def test_histogram_quantile_balancing_balances_the_recorded_batch_however_it_is_split():
    torch.manual_seed(0)
    router = PrimeRouter(
        dim=16, num_experts=6, num_null_experts=2, top_k=2, route_norm=True, route_scale=1.0, bias_on_logits=True
    )
    with torch.no_grad():
        router.gate.weight[0] += 0.5  # expert 0 is preferred
    router.quantile_balance = True
    router.quantile_grid = torch.tensor([-4.0, 4.0])
    x = torch.randn(400, 16)
    hists = []
    for chunks in (1, 4):
        router.quantile_hist = torch.zeros(8, 1024)
        router.quantile_range = torch.tensor([float("inf"), float("-inf")])
        for part in x.chunk(chunks):
            router(part)
        hists.append(router.quantile_hist)
    torch.testing.assert_close(hists[0], hists[1])
    # Each token counts once per expert, in the bin of its required bias.
    biased = router.gate(x) + router.selection_bias
    required = biased.topk(router.top_k + 1, dim=1).values[:, -1:] - router.gate(x)
    bins = ((required + 4.0) * (1024 / 8.0)).floor().clamp(0, 1023).long()
    expected = torch.stack([torch.bincount(bins[:, j], minlength=1024) for j in range(8)]).float()
    torch.testing.assert_close(hists[0], expected)

    # One update brings every expert, null experts included, close to 400 * 2 / 8 = 100 tokens.
    router.selection_bias.copy_(histogram_quantile_biases(hists[0], router.quantile_grid, router.top_k))
    with torch.no_grad():
        counts = router(x)[2]
    assert counts.max() / counts.float().mean() < 1.15
