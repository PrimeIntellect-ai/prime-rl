from pathlib import Path

import pytest
import torch

import prime_rl.trainer.models.layers.ngram as ngram_module
from prime_rl.trainer.models.layers.ngram import NGramEmbeddings, owner_counts
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_hasher import NGRAM_HASH_SEED, Qwen38NGramHasher
from prime_rl.utils.cp import CPContext

GOLDEN_PATH = Path(__file__).parents[4] / "benchmarks/scripts/ngram-module/golden/main_1810d5b9a.pt"
CONFIG_NAMES = ["tiny_text", "tiny_composite_4gram", "tiny_truncated"]
CASE_NAMES = ["edge", "edge_eos", "random"]
LAYOUT_BUFFERS = ["layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"]
REAL_CONFIG_FIELDS = [
    "vocab_size",
    "eos_token_id",
    "ngram_vocab_size_base",
    "ngram_size",
    "heads_per_ngram",
    "make_ngram_vocab_size_divisible_by",
]


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(GOLDEN_PATH, weights_only=True)


def _text_config(golden: dict, config_name: str) -> Qwen3_8FlashNextTextConfig:
    return Qwen3_8FlashNextTextConfig(**golden["configs"][config_name]["text_config"])


def _embeddings(config: Qwen3_8FlashNextTextConfig) -> NGramEmbeddings:
    table_decoder_indices = [layer_id - 1 for layer_id in sorted(set(config.ple_layer_ids))]
    head_dim = config.ple_embed_dim // ((config.ngram_size - 1) * config.heads_per_ngram)
    return NGramEmbeddings(Qwen38NGramHasher(config), table_decoder_indices, config.num_hidden_layers, head_dim)


def _hash_case(golden: dict, config_name: str, case_name: str) -> dict:
    (case,) = [case for case in golden["hash_cases"] if (case["config"], case["case"]) == (config_name, case_name)]
    return case


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_hasher_layouts_match_main(golden, config_name):
    hasher = Qwen38NGramHasher(_text_config(golden, config_name))

    for expected in golden["layouts"][config_name]:
        layout = hasher.build_layout(expected["table_index"])
        assert layout == tuple(expected[name].tolist() for name in LAYOUT_BUFFERS)
        assert hasher.table_rows(expected["table_index"]) == expected["num_rows"]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_embeddings_build_main_tables(golden, config_name):
    embeddings = _embeddings(_text_config(golden, config_name))
    expected_layouts = golden["layouts"][config_name]

    assert list(embeddings.tables) == [str(expected["decoder_index"]) for expected in expected_layouts]
    for expected in expected_layouts:
        table = embeddings.tables[str(expected["decoder_index"])]
        assert embeddings.table_indices[expected["decoder_index"]] == expected["table_index"]
        assert table.weight.shape[0] == expected["num_rows"]
        for name in LAYOUT_BUFFERS:
            assert torch.equal(getattr(table, name), expected[name])


@pytest.mark.parametrize("case_name", CASE_NAMES)
@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_hash_matches_main(golden, config_name, case_name):
    case = _hash_case(golden, config_name, case_name)
    embeddings = _embeddings(_text_config(golden, config_name))

    ngram_ids = embeddings.hash(case["input_ids"], case["cu_seqlens"])

    assert ngram_ids.shape[1] == len(case["ngram_ids"])
    for decoder_index, expected in case["ngram_ids"].items():
        assert torch.equal(ngram_ids[:, embeddings.hash_positions[decoder_index]], expected)


def test_real_parameter_hash_matches_main(golden):
    real_hash = golden["real_hash"]
    assert real_hash["params"]["seed"] == NGRAM_HASH_SEED
    config = Qwen3_8FlashNextTextConfig(**{field: real_hash["params"][field] for field in REAL_CONFIG_FIELDS})
    hasher = Qwen38NGramHasher(config)

    layouts = {}
    for expected in real_hash["layouts"]:
        layout = hasher.build_layout(expected["table_index"])
        assert layout == tuple(expected[name].tolist() for name in LAYOUT_BUFFERS)
        assert hasher.table_rows(expected["table_index"]) == expected["num_rows"]
        layouts[expected["table_index"]] = layout

    for case in real_hash["cases"]:
        table_indices = sorted(case["ngram_ids"])
        multipliers, sizes, offsets = (
            torch.tensor([layouts[table_index][buffer] for table_index in table_indices]) for buffer in range(3)
        )
        ngram_ids = hasher(case["input_ids"], case["cu_seqlens"], multipliers, sizes, offsets)
        for position, table_index in enumerate(table_indices):
            assert torch.equal(ngram_ids[:, position], case["ngram_ids"][table_index])


@pytest.mark.parametrize("cp_world_size", [2, 4])
@pytest.mark.parametrize("case_name", CASE_NAMES)
@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_cp_rank_hash_matches_full_row_slice(golden, monkeypatch, config_name, case_name, cp_world_size):
    case = _hash_case(golden, config_name, case_name)
    embeddings = _embeddings(_text_config(golden, config_name))
    full_input_ids = case["input_ids"]
    local_token_count = full_input_ids.shape[1] // cp_world_size

    def gather_full_row(local_input_ids, world_size, group):
        assert world_size == cp_world_size
        return full_input_ids

    monkeypatch.setattr(ngram_module, "gather_for_cp_wo_grad", gather_full_row)
    for cp_rank in range(cp_world_size):
        local_slice = slice(cp_rank * local_token_count, (cp_rank + 1) * local_token_count)
        embeddings.hasher.cp_context = CPContext(None, cp_rank, cp_world_size)

        ngram_ids = embeddings.hash(full_input_ids[:, local_slice], case["cu_seqlens"])

        for decoder_index, expected in case["ngram_ids"].items():
            assert torch.equal(ngram_ids[:, embeddings.hash_positions[decoder_index]], expected[local_slice])


@pytest.mark.parametrize(
    "num_rows, num_ranks, ids",
    [
        pytest.param(9, 4, [0, 2, 2, 3, 5, 6, 8, 8], id="last-rank-owns-no-rows"),
        pytest.param(10, 4, [0, 2, 3, 5, 6, 8, 9, 9], id="last-rank-owns-one-row"),
        pytest.param(10, 4, [9, 9, 9], id="one-owner"),
        pytest.param(5, 8, [0, 1, 4, 4], id="more-ranks-than-rows"),
        pytest.param(9, 4, [], id="no-ids"),
    ],
)
def test_owner_counts_match_bincount(num_rows, num_ranks, ids):
    rows_per_rank = max(1, (num_rows + num_ranks - 1) // num_ranks)
    sorted_ids = torch.tensor(sorted(ids), dtype=torch.long)

    counts = owner_counts(sorted_ids, rows_per_rank, num_ranks)

    assert torch.equal(counts, torch.bincount(sorted_ids // rows_per_rank, minlength=num_ranks))
