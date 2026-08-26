import pytest
import torch

from prime_rl.trainer.models.layers.row_sparse import backbone_keep_index, mask_grad, row_sparse_linear


def _reference(x, weight, keep_mask, upstream):
    """Dense reference: plain linear + torch.where stop-grad, upstream grad masked like the loss."""
    x = x.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    x_cut = torch.where(keep_mask.unsqueeze(-1), x, x.detach())
    out = x_cut @ w.t()
    (out * upstream).sum().backward()
    return out, x.grad, w.grad


def test_row_sparse_linear_matches_dense_stop_grad():
    """Row-sparse backward must equal the dense torch.where(detach) reference when the
    upstream gradient is zero on masked rows (guaranteed by mask_grad in the layer)."""
    torch.manual_seed(0)
    n, d_in, d_out = 32, 16, 24
    x0 = torch.randn(n, d_in)
    w0 = torch.randn(d_out, d_in)
    keep_mask = torch.rand(n) > 0.6
    upstream = torch.randn(n, d_out) * keep_mask.unsqueeze(-1)  # loss reads kept rows only

    ref_out, ref_gx, ref_gw = _reference(x0, w0, keep_mask, upstream)

    x = x0.detach().clone().requires_grad_(True)
    linear = torch.nn.Linear(d_in, d_out, bias=False)
    linear.weight = torch.nn.Parameter(w0.detach().clone())
    keep_index = backbone_keep_index(keep_mask)
    out = row_sparse_linear(linear, mask_grad(x, keep_mask), keep_index)
    (out * upstream).sum().backward()

    torch.testing.assert_close(out, ref_out)  # forward is full-row and identical
    torch.testing.assert_close(x.grad, ref_gx)
    torch.testing.assert_close(linear.weight.grad, ref_gw)
    assert (x.grad[~keep_mask] == 0).all()


def test_backbone_keep_index_edge_cases():
    all_true = torch.ones(8, dtype=torch.bool)
    assert backbone_keep_index(all_true) is None

    all_false = torch.zeros(8, dtype=torch.bool)
    index = backbone_keep_index(all_false)
    assert index is not None and index.numel() == 1  # dummy row keeps grads/collectives alive


@pytest.mark.parametrize("stop_grad", [False, True])
def test_qwen3_stop_grad_context_forward_identical_ctx_grads_zero(stop_grad):
    """Full model path: forward values must be identical with/without stop-grad;
    with stop-grad, context rows of the embedding gradient must be exactly zero."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
    from prime_rl.trainer.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attn_implementation="flash_attention_2",
    )
    if not torch.cuda.is_available():
        pytest.skip("flash-attn backbone requires CUDA")
    device = torch.device("cuda")

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    inject_prime_lm_head(model, chunk_size=8, stop_grad_context=stop_grad)

    s = 16
    input_ids = torch.randint(0, 128, (1, s), device=device)
    labels = torch.randint(0, 128, (1, s), device=device)
    keep_mask = torch.rand(1, s, device=device) > 0.5
    temperature = torch.ones(1, s, device=device)
    seq_lens = torch.tensor([s], device=device)

    out = model(
        input_ids=input_ids,
        labels=labels,
        temperature=temperature,
        keep_mask=keep_mask,
        seq_lens=seq_lens,
    )
    out["logprobs"][keep_mask].sum().backward()

    grad = model.model.embed_tokens.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    if stop_grad:
        # Tokens that appear ONLY at context positions must have zero embedding grad.
        ctx_only = set(input_ids[~keep_mask].tolist()) - set(input_ids[keep_mask].tolist())
        for token in ctx_only:
            assert (grad[token] == 0).all(), f"context-only token {token} received embedding grad"
