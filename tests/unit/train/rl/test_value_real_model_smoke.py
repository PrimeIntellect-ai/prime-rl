"""Small Qwen3 MoE critic: packed scoring must match individual scoring on four GPUs."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.model import setup_fsdp
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.rl.value import _score, _score_batch, _train_batch
from prime_rl.transports.batch.types import TrainingSample
from prime_rl.utils.cp import setup_context_parallel


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    dims = ParallelDims(dp_replicate=1, dp_shard=2, cp=2, pp=1, ep=1, world_size=4)
    torch.manual_seed(2026)
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        mlp_only_layers=[],
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"):
        model = Qwen3MoeForCausalLM._from_config(config)
        model.value_head = nn.Linear(64, 1)
    inject_prime_lm_head(model, chunk_size=None)
    model_config = ModelConfig(name="Qwen/Qwen3-0.6B", seq_len=16, cp=2, ep=1, attn="flash_attention_2")
    setup_fsdp(model, model_config, dims)
    setup_context_parallel(model, model_config, dims)

    sequences = [list(range(start, start + length)) for start, length in ((1, 4), (20, 8), (40, 12), (60, 4), (80, 8))]
    packed = _score_batch(model, sequences, dims, 16, "ring", False)
    individual = [_score(model, ids, dims, "ring", False) for ids in sequences]
    if dist.get_rank() == 0:
        assert packed is not None
        for index, (values, bootstrap) in enumerate(individual):
            torch.testing.assert_close(torch.tensor(packed[0][index]), torch.tensor(values), atol=0.02, rtol=0.02)
            torch.testing.assert_close(torch.tensor(packed[1][index]), torch.tensor(bootstrap), atol=0.02, rtol=0.02)

    samples = [
        TrainingSample(
            token_ids=ids,
            mask=[False] * (len(ids) - 1) + [True],
            logprobs=[0.0] * len(ids),
            temperatures=[1.0] * len(ids),
            env_name="smoke",
            advantages=[0.0] * (len(ids) - 1) + [1.0],
            old_values=[0.0] * len(ids),
            value_targets=[0.0] * (len(ids) - 1) + [1.0],
            value_mask=[False] * (len(ids) - 1) + [True],
        )
        for ids in sequences
    ]
    grid = prepare_batch(samples, 16, 2, sum, pad_to_multiple_of=2, for_value=True)
    micro_batches = [DataLoader._micro_batch_to_tensor(None, batch) for batch in grid[dist.get_rank() // 2]]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    value_config = SimpleNamespace(model=model_config, optim=SimpleNamespace(max_norm=None), updates_per_step=1)
    loss = _train_batch(model, optimizer, scheduler, None, micro_batches, dims, value_config, head_only=False)
    assert torch.isfinite(torch.tensor(loss))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
