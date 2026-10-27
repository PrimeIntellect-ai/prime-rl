"""Record main's Qwen3.8 n-gram hash ids, table layouts, active params, Muon grouping and checkpoint keys."""

import argparse
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

import prime_rl.trainer.models.qwen3_8_flash_next.ngram_embedding as ngram_embedding
import prime_rl.trainer.optim as optim
from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
from prime_rl.utils.vlm import get_language_model

TINY_TEXT_CONFIG = {
    "vocab_size": 64,
    "bos_token_id": 63,
    "eos_token_id": 63,
    "hidden_size": 16,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "indexer_n_heads": 1,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 1,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "hc_count": 2,
    "hc_lowrank": 4,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "ple_layer_ids": [1, 3],
    "ple_embed_dim": 16,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 17,
    "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 3,
}

CONFIGS = {
    "tiny_text": {"composite": False, "text_config": TINY_TEXT_CONFIG},
    "tiny_composite_4gram": {
        "composite": True,
        "text_config": TINY_TEXT_CONFIG | {"ngram_size": 4, "ple_embed_dim": 24, "ple_layer_ids": [2, 4]},
    },
    "tiny_truncated": {
        "composite": False,
        "text_config": TINY_TEXT_CONFIG | {"num_hidden_layers": 2, "layer_types": TINY_TEXT_CONFIG["layer_types"][:2]},
    },
}

REAL_HASH_PARAMS = {
    "vocab_size": 248320,
    "eos_token_id": 248044,
    "ngram_vocab_size_base": 20_000_000,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "make_ngram_vocab_size_divisible_by": 128,
    "ple_embed_dim": 2560,
    "seed": 1234,
    "table_indices": [0, 1],
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path of the golden .pt file to write")
    return parser.parse_args()


def source_commit() -> str:
    source_dir = Path(ngram_embedding.__file__).parent
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError(f"Tracked files are modified in the source worktree of {source_dir}:\n{status}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, capture_output=True, text=True, check=True
    ).stdout.strip()


def build_model(spec: dict, device: str) -> Qwen3_8FlashNextForCausalLM:
    text_config = Qwen3_8FlashNextTextConfig(**copy.deepcopy(spec["text_config"]))
    config = Qwen3_8FlashNextConfig(text_config=text_config) if spec["composite"] else text_config
    with torch.device(device):
        return Qwen3_8FlashNextForCausalLM(config)


def ngram_embeddings_by_decoder_index(model: Qwen3_8FlashNextForCausalLM) -> dict[int, ngram_embedding.NGramEmbedding]:
    return {
        decoder_index: layer.ple.ple_embedding
        for decoder_index, layer in enumerate(get_language_model(model).layers)
        if layer.ple is not None
    }


def table_layout(embedding: ngram_embedding.NGramEmbedding) -> dict:
    return {
        "table_index": embedding.ngram_layer_index,
        "layer_multipliers": embedding.layer_multipliers.clone(),
        "ngram_heads_vocab_sizes": embedding.ngram_heads_vocab_sizes.clone(),
        "ngram_heads_offsets": embedding.ngram_heads_offsets.clone(),
        "num_rows": embedding.ngram_embedding.num_embeddings,
    }


def non_eos_tokens(count: int, vocab_size: int, eos_token_id: int, generator: torch.Generator) -> torch.Tensor:
    tokens = torch.randint(0, vocab_size - 1, (count,), generator=generator)
    return tokens + (tokens >= eos_token_id).long()


def packed_case(name: str, tokens: torch.Tensor, seq_lens: list[int]) -> dict:
    cu_seqlens, _ = get_cu_seqlens_from_seq_lens(torch.tensor(seq_lens), total_tokens=tokens.numel())
    return {"case": name, "input_ids": tokens.reshape(1, -1).clone(), "cu_seqlens": cu_seqlens}


def edge_case(vocab_size: int, eos_token_id: int) -> dict:
    tokens = non_eos_tokens(32, vocab_size, eos_token_id, torch.Generator().manual_seed(0))
    tokens[[3, 6, 7, 13, 28]] = eos_token_id
    return packed_case("edge", tokens, [10, 1, 2, 3, 7, 9])


def edge_eos_case(vocab_size: int, eos_token_id: int) -> dict:
    tokens = non_eos_tokens(32, vocab_size, eos_token_id, torch.Generator().manual_seed(1))
    tokens[[7, 15, 22]] = eos_token_id
    return packed_case("edge_eos", tokens, [4, 28])


def random_case(vocab_size: int, eos_token_id: int) -> dict:
    generator = torch.Generator().manual_seed(2)
    num_tokens = 256
    tokens = non_eos_tokens(num_tokens, vocab_size, eos_token_id, generator)
    tokens[torch.rand(num_tokens, generator=generator) < 0.05] = eos_token_id
    boundaries = (torch.randperm(num_tokens - 1, generator=generator)[:5] + 1).sort().values
    seq_lens = torch.cat([boundaries, torch.tensor([num_tokens])]).diff(prepend=torch.tensor([0]))
    return packed_case("random", tokens, seq_lens.tolist())


def real_case(vocab_size: int, eos_token_id: int) -> dict:
    tokens = non_eos_tokens(512, vocab_size, eos_token_id, torch.Generator().manual_seed(3))
    tokens[[37, 101, 150, 151, 299, 400]] = eos_token_id
    return packed_case("real", tokens, [100, 1, 199, 212])


@torch.no_grad()
def hash_ids(embeddings: dict[int, ngram_embedding.NGramEmbedding], case: dict) -> dict[int, torch.Tensor]:
    return {
        key: embedding.compute_ngram_ids(case["input_ids"], case["cu_seqlens"]) for key, embedding in embeddings.items()
    }


def real_hash_embedding(table_index: int) -> ngram_embedding.NGramEmbedding:
    params = REAL_HASH_PARAMS
    with torch.device("meta"):
        embedding = ngram_embedding.NGramEmbedding(
            embedding_dim=params["ple_embed_dim"],
            ngram_size=params["ngram_size"],
            heads_per_ngram=params["heads_per_ngram"],
            ngram_vocab_size=params["ngram_vocab_size_base"],
            token_vocab_size=params["vocab_size"],
            eos_token_id=params["eos_token_id"],
            vocab_size_divisor=params["make_ngram_vocab_size_divisible_by"],
            ngram_layer_index=table_index,
            seed=params["seed"],
        )
    for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        setattr(embedding, name, torch.empty_like(getattr(embedding, name), device="cpu"))
    embedding.reset_parameters()
    return embedding


def capture_real_hash() -> dict:
    embeddings = {table_index: real_hash_embedding(table_index) for table_index in REAL_HASH_PARAMS["table_indices"]}
    case = real_case(REAL_HASH_PARAMS["vocab_size"], REAL_HASH_PARAMS["eos_token_id"])
    return {
        "params": REAL_HASH_PARAMS,
        "layouts": [table_layout(embedding) for embedding in embeddings.values()],
        "cases": [case | {"ngram_ids": hash_ids(embeddings, case)}],
    }


def muon_param_groups(model: Qwen3_8FlashNextForCausalLM) -> dict[str, list[str]]:
    named_params = list(model.named_parameters())
    names = {id(param): name for name, param in named_params}
    optimizer = optim._create_muon_optimizer(MuonConfig(), named_params, NO_PARALLELISM, model)
    muon_group, *middle_groups, adamw_group = optimizer.param_groups
    expert_groups = [group for group in middle_groups if "distributed_mesh_name" in group]
    router_groups = [group for group in middle_groups if "distributed_mesh_name" not in group]

    def group_names(groups: list[dict]) -> list[str]:
        return sorted(names[id(param)] for group in groups for param in group["params"])

    return {
        "muon": group_names([muon_group]),
        "expert": group_names(expert_groups),
        "router": group_names(router_groups),
        "adamw": group_names([adamw_group]),
    }


def main() -> None:
    args = parse_args()
    optim.Muon = RecordingMuon
    optim._warmup_muon_mesh = lambda mesh: None
    torch.manual_seed(0)

    golden = {
        "commit": source_commit(),
        "configs": CONFIGS,
        "layouts": {},
        "hash_cases": [],
        "real_hash": capture_real_hash(),
        "num_active_params": {},
        "param_groups": {},
        "hf_keys": {},
        "prime_keys": {},
    }
    for name, spec in CONFIGS.items():
        meta_model = build_model(spec, "meta")
        golden["num_active_params"][name] = meta_model.num_active_params()
        golden["param_groups"][name] = muon_param_groups(meta_model)

        model = build_model(spec, "cpu")
        model.init_buffers_post_meta()
        embeddings = ngram_embeddings_by_decoder_index(model)
        golden["layouts"][name] = [
            {"decoder_index": decoder_index} | table_layout(embedding)
            for decoder_index, embedding in embeddings.items()
        ]
        vocab_size, eos_token_id = spec["text_config"]["vocab_size"], spec["text_config"]["eos_token_id"]
        for case in (
            edge_case(vocab_size, eos_token_id),
            edge_eos_case(vocab_size, eos_token_id),
            random_case(vocab_size, eos_token_id),
        ):
            golden["hash_cases"].append({"config": name} | case | {"ngram_ids": hash_ids(embeddings, case)})

        state_dict = model.state_dict()
        golden["prime_keys"][name] = sorted(state_dict)
        golden["hf_keys"][name] = sorted(model.convert_to_hf(dict(state_dict)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(golden, args.output)
    print(f"Captured {ngram_embedding.__file__} at {golden['commit']} into {args.output}")


if __name__ == "__main__":
    main()
