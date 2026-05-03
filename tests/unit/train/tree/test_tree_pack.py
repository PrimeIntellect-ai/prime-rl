import pytest
import torch

from prime_rl.configs.sft import CaterpillarFakeDataConfig, SFTConfig, SFTRawToolCaterpillarDataConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.sft.data import (
    _raw_tool_current_rl_samples_from_messages,
    _raw_tool_selection_metrics_batch,
    _raw_tool_tree_fits_limits,
)
from prime_rl.trainer.tree import (
    Tree,
    TreeNode,
    build_caterpillar,
    pack_tree,
    tree_nll_loss,
    tree_nll_weighted_token_count,
)


def test_simple_y_tree():
    tree = Tree(
        [
            TreeNode(parent=-1, token_ids=[1, 2], loss_mask=[True, True]),
            TreeNode(parent=0, token_ids=[3, 4], loss_mask=[True, False]),
            TreeNode(parent=0, token_ids=[5], loss_mask=[True]),
        ]
    )

    packed = pack_tree(tree)

    assert packed.K == 2
    assert packed.input_ids.tolist() == [1, 2, 3, 4, 5]
    assert packed.position_ids.tolist() == [0, 1, 2, 3, 2]
    assert packed.prev_map.tolist() == [-1, 0, 1, 2, 1]
    assert packed.node_of_token.tolist() == [0, 0, 1, 1, 2]
    assert packed.node_token_range == [(0, 2), (2, 4), (4, 5)]
    torch.testing.assert_close(packed.loss_weights, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.5]))

    expected_mask = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, True],
        ]
    )
    torch.testing.assert_close(packed.attn_mask, expected_mask)

    expected_ancestor_nodes = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ]
    )
    torch.testing.assert_close(packed.is_ancestor_node, expected_ancestor_nodes)


def test_caterpillar_topology():
    turns = [
        ([1, 2], [3], [4, 5]),
        ([6], [7, 8], [9]),
    ]

    tree = build_caterpillar(turns)

    assert len(tree.nodes) == 6
    assert tree.nodes[0].parent == -1
    assert tree.nodes[1].parent == tree.nodes[2].parent == 0
    assert tree.nodes[3].parent == 2
    assert tree.nodes[4].parent == tree.nodes[5].parent == 3
    assert tree.leaves() == [1, 4, 5]
    assert tree.root_path(5) == [0, 2, 3, 5]
    assert pack_tree(tree).K == len(turns) + 1


def test_mask_ancestor_only():
    tree = Tree(
        [
            TreeNode(parent=-1, token_ids=[1], loss_mask=[False]),
            TreeNode(parent=0, token_ids=[2, 3], loss_mask=[True, True]),
            TreeNode(parent=1, token_ids=[4], loss_mask=[True]),
            TreeNode(parent=0, token_ids=[5, 6], loss_mask=[True, True]),
        ]
    )
    packed = pack_tree(tree)

    ancestors = {
        0: {0},
        1: {0, 1},
        2: {0, 1, 2},
        3: {0, 3},
    }
    for node_idx, ancestor_nodes in ancestors.items():
        assert (
            set(torch.nonzero(packed.is_ancestor_node[node_idx], as_tuple=False).flatten().tolist()) == ancestor_nodes
        )

    for query_pos in range(len(packed.input_ids)):
        query_node = packed.node_of_token[query_pos].item()
        for key_pos in range(len(packed.input_ids)):
            key_node = packed.node_of_token[key_pos].item()
            same_node_causal = key_node == query_node and key_pos <= query_pos
            ancestor_block = key_node != query_node and key_node in ancestors[query_node]
            assert packed.attn_mask[query_pos, key_pos].item() == (same_node_causal or ancestor_block)


def test_prev_map():
    tree = build_caterpillar(
        [
            ([10, 11], [12, 13], [14]),
            ([15], [16], [17, 18]),
        ]
    )

    packed = pack_tree(tree)

    assert packed.input_ids.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18]
    assert packed.prev_map.tolist() == [-1, 0, 1, 2, 1, 4, 5, 5, 7]


def test_k_equals_one_reduces_to_sft():
    tree = Tree(
        [
            TreeNode(parent=-1, token_ids=[1, 2], loss_mask=[False, True]),
            TreeNode(parent=0, token_ids=[3, 4], loss_mask=[True, True]),
        ]
    )

    packed = pack_tree(tree)

    assert packed.K == 1
    assert packed.prev_map.tolist() == [-1, 0, 1, 2]
    torch.testing.assert_close(packed.loss_weights, packed.loss_mask.float())

    logits = torch.randn(1, 4, 10, dtype=torch.float64)
    tree_loss = tree_nll_loss(
        logits,
        packed.input_ids.unsqueeze(0),
        packed.prev_map.unsqueeze(0),
        packed.loss_mask.unsqueeze(0),
        packed.loss_weights.unsqueeze(0).to(torch.float64),
    )
    ce = torch.nn.functional.cross_entropy(logits[0, :-1], packed.input_ids[1:], reduction="none")
    sft_loss = (ce * packed.loss_mask[1:].to(torch.float64)).sum()
    torch.testing.assert_close(tree_loss, sft_loss)


def test_tree_nll_token_count_uses_weighted_denominator():
    tree = build_caterpillar(
        [
            ([1, 2], [3], [4, 5]),
            ([6], [7, 8], [9]),
        ]
    )
    packed = pack_tree(tree)

    unweighted_count = (packed.loss_mask & (packed.prev_map >= 0)).sum()
    weighted_count = tree_nll_weighted_token_count(
        packed.prev_map,
        packed.loss_mask,
        packed.loss_weights,
    )

    assert unweighted_count.item() == 6
    torch.testing.assert_close(weighted_count, torch.tensor(8.0 / 3.0))
    assert weighted_count.item() != unweighted_count.item()


def test_raw_tool_tree_limit_allows_packed_tree_larger_than_path_limit():
    tree = Tree(
        [
            TreeNode(parent=-1, token_ids=[1, 2], loss_mask=[False, False]),
            TreeNode(parent=0, token_ids=[3, 4], loss_mask=[True, True]),
            TreeNode(parent=0, token_ids=[5, 6], loss_mask=[True, True]),
        ]
    )

    assert _raw_tool_tree_fits_limits(tree, max_path_tokens=4, max_packed_tokens=6)
    assert not _raw_tool_tree_fits_limits(tree, max_path_tokens=4, max_packed_tokens=4)
    assert not _raw_tool_tree_fits_limits(tree, max_path_tokens=3, max_packed_tokens=6)


def test_raw_tool_config_rejects_packed_limit_below_path_limit():
    with pytest.raises(ValueError, match="max_packed_tokens"):
        SFTRawToolCaterpillarDataConfig(seq_len=8, max_packed_tokens=7)


def test_raw_tool_branching_score_counts_reasoning_turns():
    batch = {
        "prompt": [[{"role": "user", "content": "question"}]],
        "completion": [[{"role": "assistant", "content": "answer", "reasoning_content": "scratchpad"}]],
        "num_turns": [3],
        "token_usage": [{"final_input_tokens": 10, "final_output_tokens": 5}],
    }

    metrics = _raw_tool_selection_metrics_batch(batch, [123])

    assert metrics["row_idx"] == [123]
    assert metrics["assistant_turns"] == [1]
    assert metrics["reasoning_turns"] == [1]
    assert metrics["final_token_estimate"] == [15.0]
    assert metrics["branching_score"] == [6.0]
    assert metrics["cheap_ok"] == [True]


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_raw_tool_current_rl_baseline_breaks_after_reasoning_strip():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
    ]

    samples = _raw_tool_current_rl_samples_from_messages(
        _CharTokenizer(),
        messages,
        seq_len=4096,
        train_response=True,
        train_reasoning=True,
    )

    assert len(samples) == 2
    assert all(len(sample["input_ids"]) == len(sample["target_ids"]) == len(sample["loss_mask"]) for sample in samples)


def test_raw_tool_current_rl_baseline_extends_when_reasoning_not_stripped():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    samples = _raw_tool_current_rl_samples_from_messages(
        _CharTokenizer(),
        messages,
        seq_len=4096,
        train_response=True,
        train_reasoning=True,
    )

    assert len(samples) == 1
    assert len(samples[0]["input_ids"]) == len(samples[0]["target_ids"]) == len(samples[0]["loss_mask"])


@pytest.mark.parametrize(
    ("model_updates", "data_kwargs", "loss_impl", "message"),
    [
        ({"attn": "flash_attention_2"}, {}, "torch", "model.attn"),
        ({"attn": "flash_attention_2", "cp": 2}, {}, "torch", "model.cp"),
        ({"ep": 2}, {}, "torch", "model.ep"),
        ({"impl": "custom"}, {}, "torch", "model.impl"),
        ({}, {}, "liger_fused", "loss_impl"),
        ({}, {"pack_function": "stack"}, "torch", "pack_function"),
        ({}, {"batch_size": 2, "micro_batch_size": 2}, "torch", "micro_batch_size"),
    ],
)
def test_tree_config_rejects_unsupported_combinations(model_updates, data_kwargs, loss_impl, message):
    model = ModelConfig(attn="sdpa", impl="hf")
    for key, value in model_updates.items():
        setattr(model, key, value)
    data = CaterpillarFakeDataConfig(**data_kwargs)

    with pytest.raises(ValueError, match=message):
        SFTConfig(model=model, data=data, loss_impl=loss_impl)


def test_tree_config_accepts_flex_attention():
    SFTConfig(
        model=ModelConfig(attn="flex_attention", impl="hf"),
        data=CaterpillarFakeDataConfig(),
        loss_impl="torch",
    )
