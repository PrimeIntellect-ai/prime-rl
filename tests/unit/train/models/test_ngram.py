import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import prime_rl.trainer.models.layers.ngram as ngram_module
import prime_rl.trainer.optim as optim
from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.distributed.ngram_lookup import owner_counts
from prime_rl.trainer.models.layers.ngram import NGramEmbeddings
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.trainer.models.qwen3_8_flash_next.converting_qwen3_8_flash_next import NGramEmbeddingWeights
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_hasher import NGRAM_HASH_SEED, Qwen38NGramHasher
from prime_rl.transports.weights.nccl import filter_state_dict_by_layers, get_max_layer_num, preprocess_layer_checkpoint
from prime_rl.transports.weights.nixl.nixl import LAYER_RE, NIXLWeightSender
from prime_rl.utils.cp import CPContext
from prime_rl.utils.vlm import get_language_model, get_layer_prefix
from prime_rl.utils.weights import _LAYER_RE, get_chunked_conversion_ops, get_consuming_layers, partition_weights

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
MAIN_TABLE_NAME = re.compile(
    r"(.+)\.layers\.(\d+)\.ple\.ple_embedding\."
    r"(ngram_embedding\.weight|layer_multipliers|ngram_heads_vocab_sizes|ngram_heads_offsets)"
)
NO_PARALLELISM = SimpleNamespace(
    ep_enabled=False,
    dp_shard_enabled=False,
    cp_enabled=False,
    dp_replicate_enabled=False,
    world_mesh=None,
)


class RecordingMuon:
    def __init__(self, params, **kwargs) -> None:
        self.param_groups = params


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


def _model(golden: dict, config_name: str, device: str) -> Qwen3_8FlashNextForCausalLM:
    text_config = _text_config(golden, config_name)
    is_composite = golden["configs"][config_name]["composite"]
    config = Qwen3_8FlashNextConfig(text_config=text_config) if is_composite else text_config
    with torch.device(device):
        return Qwen3_8FlashNextForCausalLM(config)


def _branch_name(main_name: str) -> str:
    match = MAIN_TABLE_NAME.fullmatch(main_name)
    if match is None:
        return main_name
    model_prefix, decoder_index, name = match.groups()
    return f"{model_prefix}.ngram.tables.{decoder_index}.{name.removeprefix('ngram_embedding.')}"


def _zero_block(block_input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.zeros_like(block_input)


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


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_model_checkpoint_keys_match_main(golden, config_name):
    model = _model(golden, config_name, "cpu")
    model.init_buffers_post_meta()
    state_dict = model.state_dict()

    assert sorted(state_dict) == sorted(_branch_name(name) for name in golden["prime_keys"][config_name])
    assert sorted(model.convert_to_hf(dict(state_dict))) == golden["hf_keys"][config_name]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_model_exports_main_table_layouts(golden, config_name):
    model = _model(golden, config_name, "cpu")
    model.init_buffers_post_meta()
    model_prefix = "model.language_model" if golden["configs"][config_name]["composite"] else "model"

    hf_state_dict = model.convert_to_hf(model.state_dict())

    for expected in golden["layouts"][config_name]:
        checkpoint_prefix = f"{model_prefix}.layers.{expected['decoder_index']}.ple.ple_embedding"
        for name in LAYOUT_BUFFERS:
            assert torch.equal(hf_state_dict[f"{checkpoint_prefix}.{name}"], expected[name])
        shard_prefix = f"{checkpoint_prefix}.ngram_embedding.shard_"
        shard_rows = sum(value.shape[0] for key, value in hf_state_dict.items() if key.startswith(shard_prefix))
        assert shard_rows == expected["num_rows"]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_model_active_params_match_main(golden, config_name):
    model = _model(golden, config_name, "meta")

    assert model.num_active_params() == golden["num_active_params"][config_name]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_model_optimizer_groups_match_main(golden, monkeypatch, config_name):
    monkeypatch.setattr(optim, "Muon", RecordingMuon)
    monkeypatch.setattr(optim, "_warmup_muon_mesh", lambda mesh: None)
    model = _model(golden, config_name, "meta")
    named_params = list(model.named_parameters())
    names = {id(param): name for name, param in named_params}

    optimizer = optim._create_muon_optimizer(MuonConfig(), named_params, NO_PARALLELISM, model)

    muon_group, *middle_groups, adamw_group = optimizer.param_groups
    groups = {
        "muon": [muon_group],
        "expert": [group for group in middle_groups if "distributed_mesh_name" in group],
        "router": [group for group in middle_groups if "distributed_mesh_name" not in group],
        "adamw": [adamw_group],
    }
    actual = {
        group_name: sorted(names[id(param)] for group in group_list for param in group["params"])
        for group_name, group_list in groups.items()
    }
    expected = {
        group_name: sorted(_branch_name(name) for name in group_names)
        for group_name, group_names in golden["param_groups"][config_name].items()
    }
    assert actual == expected


@pytest.mark.parametrize("case_name", CASE_NAMES)
@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_model_forward_feeds_main_ngram_ids_to_ple(golden, config_name, case_name):
    case = _hash_case(golden, config_name, case_name)
    model = _model(golden, config_name, "cpu")
    model.init_buffers_post_meta()
    language_model = get_language_model(model)
    ple_calls = {}
    ple_hooks = []

    def record_ple_call(decoder_index, module, args, output):
        expanded_states, _, cu_seqlens = args
        ple_calls[decoder_index] = (expanded_states, cu_seqlens, output)

    for decoder_index, layer in enumerate(language_model.layers):
        for block in (getattr(layer, "linear_attn", None), getattr(layer, "self_attn", None), layer.mlp):
            if block is not None:
                block.forward = _zero_block
        if layer.ple is not None:
            torch.nn.init.normal_(layer.ple.conv1d.weight)
            ple_hooks.append(layer.ple.register_forward_hook(partial(record_ple_call, decoder_index)))

    language_model(input_ids=case["input_ids"], seq_lens=case["cu_seqlens"].diff())
    for hook in ple_hooks:
        hook.remove()

    main_ngram_ids = torch.stack([case["ngram_ids"][index] for index in language_model.ngram.hash_positions], dim=1)
    assert torch.equal(language_model.ngram.hash(case["input_ids"], case["cu_seqlens"]), main_ngram_ids)
    assert ple_calls.keys() == case["ngram_ids"].keys()
    for decoder_index, (expanded_states, cu_seqlens, output) in ple_calls.items():
        ple = language_model.layers[decoder_index].ple
        embeddings = language_model.ngram.embed(main_ngram_ids, decoder_index)
        assert torch.equal(output, ple(expanded_states, embeddings, cu_seqlens))


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_nccl_layer_packets_match_main(golden, config_name):
    model = _model(golden, config_name, "meta")
    state_dict = model.state_dict()
    layer_prefix = get_layer_prefix(model.config)
    main_keys = golden["prime_keys"][config_name]
    main_num_layers = get_max_layer_num(dict.fromkeys(main_keys), layer_prefix)
    main_packets = [(-1, sorted(_branch_name(key) for key in main_keys if not key.startswith(layer_prefix)))]
    main_packets += [
        (layer, sorted(_branch_name(key) for key in main_keys if key.startswith(f"{layer_prefix}{layer}.")))
        for layer in range(main_num_layers)
    ]

    packets = filter_state_dict_by_layers(
        state_dict,
        get_max_layer_num(state_dict, layer_prefix),
        layer_prefix,
        get_consuming_layers(model, state_dict),
    )

    assert [(layer, sorted(packet)) for layer, packet in packets] == main_packets
    main_table_weights = [_branch_name(key) for key in main_keys if key.endswith(".ngram_embedding.weight")]
    assert sorted(get_chunked_conversion_ops(model, state_dict)) == sorted(main_table_weights)


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_nccl_layer_packet_sends_main_table_buffers(golden, config_name):
    model = _model(golden, config_name, "cpu")
    model.init_buffers_post_meta()
    state_dict = model.state_dict()
    layer_prefix = get_layer_prefix(model.config)
    chunked_ops = get_chunked_conversion_ops(model, state_dict)

    packets = dict(
        filter_state_dict_by_layers(
            state_dict,
            get_max_layer_num(state_dict, layer_prefix),
            layer_prefix,
            get_consuming_layers(model, state_dict),
        )
    )

    for expected in golden["layouts"][config_name]:
        decoder_index = expected["decoder_index"]
        packet = {key: value for key, value in packets[decoder_index].items() if key not in chunked_ops}
        hf_packet = preprocess_layer_checkpoint(model, packet, decoder_index)
        for name in LAYOUT_BUFFERS:
            assert torch.equal(hf_packet[f"{layer_prefix}{decoder_index}.ple.ple_embedding.{name}"], expected[name])


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_nixl_transfer_groups_match_main(golden, config_name):
    model = _model(golden, config_name, "meta")
    state_dict = model.state_dict()
    main_layers = {}
    for key in golden["prime_keys"][config_name]:
        if state_dict[_branch_name(key)].is_floating_point():
            match = LAYER_RE.search(key)
            main_layers[_branch_name(key)] = None if match is None else int(match.group(1))
    main_layer_numbers = sorted({layer for layer in main_layers.values() if layer is not None})

    groups = NIXLWeightSender.build_transfer_group_index(state_dict, get_consuming_layers(model, state_dict))

    assert groups.group_names == ["non_layer", *(f"layer.{layer}" for layer in main_layer_numbers)]
    assert {
        key: groups.group_names[NIXLWeightSender.find_transfer_group_index(key, groups)] for key in main_layers
    } == {key: "non_layer" if layer is None else f"layer.{layer}" for key, layer in main_layers.items()}


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_partition_units_match_main(golden, config_name):
    model = _model(golden, config_name, "meta")
    state_dict = model.state_dict()
    main_units = defaultdict(set)
    for key in golden["prime_keys"][config_name]:
        match = _LAYER_RE.match(key)
        main_units[match.group(1) if match else ""].add(_branch_name(key))

    owners = partition_weights(
        state_dict,
        len(main_units),
        torch.bfloat16,
        get_layer_prefix(model.config),
        get_consuming_layers(model, state_dict),
    )

    owner_keys = defaultdict(set)
    for key, owner in owners.items():
        owner_keys[owner].add(key)
    assert sorted(map(sorted, owner_keys.values())) == sorted(map(sorted, main_units.values()))


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_table_without_chunked_conversion_raises(golden, monkeypatch, config_name):
    model = _model(golden, config_name, "meta")
    state_dict = model.state_dict()
    unchunked_weight = list(get_chunked_conversion_ops(model, state_dict))[-1]
    chain_without_op = [
        op
        for op in model.conversion_chain(model.config)
        if not (isinstance(op, NGramEmbeddingWeights) and op.runtime_weight == unchunked_weight)
    ]
    monkeypatch.setattr(type(model), "conversion_chain", classmethod(lambda cls, config: chain_without_op))

    with pytest.raises(ValueError, match=re.escape(f"NGramTable weights {[unchunked_weight]} match no ChunkedConvOp")):
        get_chunked_conversion_ops(model, state_dict)


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
