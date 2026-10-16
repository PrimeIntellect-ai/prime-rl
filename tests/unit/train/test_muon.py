import pytest
import torch

from prime_rl.trainer.models.fusions import fuse_qkv_projections
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.optim import muon_matrix_partitions

HEAD_DIM, NUM_HEADS, NUM_KV_HEADS = 16, 4, 2
Q_ROWS, KV_ROWS = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM


def build_attention(fused: bool) -> FlashAttention:
    config = AttentionConfig(
        hidden_size=128,
        head_dim=HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        is_causal=True,
        attention_bias=False,
        use_qk_norm=False,
        rms_norm_eps=1e-6,
    )
    attention = FlashAttention(config)
    if fused:
        fuse_qkv_projections(attention)
    return attention


@pytest.mark.parametrize(
    ("heads_per_block", "q_blocks", "kv_blocks"),
    [(None, (Q_ROWS,), (KV_ROWS,)), (1, (HEAD_DIM,) * 4, (HEAD_DIM,) * 2), (2, (2 * HEAD_DIM,) * 2, (2 * HEAD_DIM,))],
)
def test_head_blocks_split_each_projection_by_its_own_heads(heads_per_block, q_blocks, kv_blocks):
    fused = build_attention(fused=True)
    partitions = muon_matrix_partitions(fused, set(fused.parameters()), heads_per_block)
    assert partitions == {fused.qkv_proj.weight: q_blocks + kv_blocks + kv_blocks}

    unfused = build_attention(fused=False)
    partitions = muon_matrix_partitions(unfused, set(unfused.parameters()), heads_per_block)
    expected = {unfused.q_proj.weight: q_blocks, unfused.k_proj.weight: kv_blocks, unfused.v_proj.weight: kv_blocks}
    assert partitions == ({} if heads_per_block is None else expected)


@pytest.mark.parametrize("heads_per_block", [None, 1, 2])
def test_muon_orthogonalizes_each_head_block_on_its_own(heads_per_block):
    muon = pytest.importorskip("dion.muon")
    attention = build_attention(fused=True)
    weight = attention.qkv_proj.weight
    partitions = muon_matrix_partitions(attention, {weight}, heads_per_block)
    blocks = partitions[weight]

    weight.grad = torch.randn_like(weight)
    before = weight.detach().clone()
    optimizer = muon.Muon(
        [weight],
        fsdp_mesh_dim=0,
        world_mesh=None,
        lr=1.0,
        mu=0.0,
        weight_decay=0.0,
        adjust_lr=None,
        matrix_partitions=partitions,
    )
    optimizer.step()
    # With no momentum and no learning-rate adjustment, each block moves by its own orthogonalized gradient
    # (Muon casts the momentum to bfloat16 before Newton-Schulz).
    update = before - weight.detach()
    for block_update, block_grad in zip(update.split(blocks), weight.grad.split(blocks)):
        expected = muon.zeropower_via_newtonschulz5(block_grad.bfloat16()).float()
        torch.testing.assert_close(block_update, expected, atol=1e-6, rtol=0)
