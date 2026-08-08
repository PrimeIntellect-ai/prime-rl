import pytest
import torch
from torch import nn
from torch.optim import AdamW

from prime_rl.trainer.optim import CPUOffloadOptimizer

pytestmark = [pytest.mark.gpu]


class MiniModel(nn.Module):
    """Small transformer-like model with named layers for chunking."""

    def __init__(self, n_layers=4, hidden=32, vocab=100):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Linear(hidden * 4, hidden),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def _make_pair(n_layers=4, hidden=32, vocab=100):
    """Create two identical models + optimizers (plain vs offloaded), same seed."""
    torch.manual_seed(42)
    model_plain = MiniModel(n_layers, hidden, vocab).cuda()

    torch.manual_seed(42)
    model_offloaded = MiniModel(n_layers, hidden, vocab).cuda()

    named_params_plain = list(model_plain.named_parameters())
    named_params_offloaded = list(model_offloaded.named_parameters())

    trainable_plain = [p for _, p in named_params_plain if p.requires_grad]
    trainable_offloaded = [p for _, p in named_params_offloaded if p.requires_grad]

    opt_plain = AdamW(trainable_plain, lr=1e-3, weight_decay=0.01)
    opt_offloaded = CPUOffloadOptimizer(
        AdamW(trainable_offloaded, lr=1e-3, weight_decay=0.01),
        named_params=named_params_offloaded,
    )

    # Verify the two models start identical
    for (n1, p1), (n2, p2) in zip(named_params_plain, named_params_offloaded):
        assert torch.equal(p1, p2), f"Initial mismatch in {n1}"

    return model_plain, model_offloaded, opt_plain, opt_offloaded


def _run_step(model, optimizer, input_ids):
    """Forward → backward → step → zero_grad."""
    out = model(input_ids)
    loss = out.float().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


@pytest.mark.parametrize("n_steps", [1, 5])
def test_param_parity(n_steps):
    """Parameters match between plain AdamW and CPUOffloadOptimizer after N steps."""
    model_plain, model_offloaded, opt_plain, opt_offloaded = _make_pair()

    torch.manual_seed(123)
    for step in range(n_steps):
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        loss_plain = _run_step(model_plain, opt_plain, input_ids)
        loss_offloaded = _run_step(model_offloaded, opt_offloaded, input_ids)

        # Losses should be identical (same model, same input)
        assert torch.allclose(loss_plain, loss_offloaded, atol=1e-5), (
            f"Loss mismatch at step {step}: {loss_plain.item()} vs {loss_offloaded.item()}"
        )

    # Compare every parameter
    for (n, p_plain), (_, p_offloaded) in zip(model_plain.named_parameters(), model_offloaded.named_parameters()):
        assert torch.allclose(p_plain, p_offloaded, atol=1e-6, rtol=1e-5), (
            f"Param mismatch in {n}: max diff {(p_plain - p_offloaded).abs().max().item()}"
        )


@pytest.mark.parametrize("n_steps", [1, 5])
def test_state_parity(n_steps):
    """Optimizer states (exp_avg, exp_avg_sq, step) match after N steps."""
    model_plain, model_offloaded, opt_plain, opt_offloaded = _make_pair()

    torch.manual_seed(123)
    for step in range(n_steps):
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        _run_step(model_plain, opt_plain, input_ids)
        _run_step(model_offloaded, opt_offloaded, input_ids)

    # Move offloaded states to GPU for comparison
    opt_offloaded._move_states("cuda")
    torch.cuda.synchronize()

    # Build a mapping from param id to state for the plain optimizer
    plain_state = opt_plain.state
    offloaded_state = opt_offloaded.state

    # Match params by position (same order in both optimizers)
    params_plain = [p for g in opt_plain.param_groups for p in g["params"]]
    params_offloaded = [p for g in opt_offloaded.param_groups for p in g["params"]]

    assert len(params_plain) == len(params_offloaded), "Different number of params"

    for p_plain, p_offloaded in zip(params_plain, params_offloaded):
        s_plain = plain_state[p_plain]
        s_offloaded = offloaded_state[p_offloaded]
        assert set(s_plain.keys()) == set(s_offloaded.keys()), (
            f"State keys mismatch: {set(s_plain.keys())} vs {set(s_offloaded.keys())}"
        )
        for k in s_plain:
            if isinstance(s_plain[k], torch.Tensor):
                assert torch.allclose(s_plain[k], s_offloaded[k], atol=1e-6, rtol=1e-5), (
                    f"State '{k}' mismatch: max diff {(s_plain[k] - s_offloaded[k]).abs().max().item()}"
                )
            else:
                assert s_plain[k] == s_offloaded[k], f"State '{k}' mismatch: {s_plain[k]} vs {s_offloaded[k]}"

    # Cleanup: move states back to CPU
    opt_offloaded._move_states("cpu")
    torch.cuda.synchronize()


def test_state_dict_roundtrip():
    """state_dict/load_state_dict preserves optimizer state correctly."""
    model_plain, model_offloaded, opt_plain, opt_offloaded = _make_pair()

    torch.manual_seed(123)
    for step in range(3):
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        _run_step(model_plain, opt_plain, input_ids)
        _run_step(model_offloaded, opt_offloaded, input_ids)

    # Save state_dict
    sd = opt_offloaded.state_dict()

    # Create a fresh model with the same initial seed, then copy current weights
    # so the fresh model matches the offloaded model's state after 3 steps
    torch.manual_seed(42)
    model_fresh = MiniModel(4, 32, 100).cuda()
    for (_, p_src), (_, p_dst) in zip(model_offloaded.named_parameters(), model_fresh.named_parameters()):
        p_dst.data.copy_(p_src.data)

    named_params_fresh = list(model_fresh.named_parameters())
    trainable_fresh = [p for _, p in named_params_fresh if p.requires_grad]
    opt_fresh = CPUOffloadOptimizer(
        AdamW(trainable_fresh, lr=1e-3, weight_decay=0.01),
        named_params=named_params_fresh,
    )
    opt_fresh.load_state_dict(sd)

    # Run one more step on both and compare
    input_ids = torch.randint(0, 100, (2, 8)).cuda()
    _run_step(model_offloaded, opt_offloaded, input_ids)
    _run_step(model_fresh, opt_fresh, input_ids)

    for (_, p1), (_, p2) in zip(model_offloaded.named_parameters(), model_fresh.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6, rtol=1e-5), (
            f"Param mismatch after reload: max diff {(p1 - p2).abs().max().item()}"
        )
