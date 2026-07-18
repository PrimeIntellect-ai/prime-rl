import torch

from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaPreTrainedModel
from prime_rl.trainer.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoePreTrainedModel
from prime_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast, TrainerShardSource
from prime_rl.weight_transfer.wire import TrainerAgent, TrainerShard, TrainerTable, TrainerTensor


def test_model_weight_transfer_precision_policy():
    assert GlmMoeDsaPreTrainedModel.keep_in_fp32_for_weight_transfer("model.layers.3.mlp.expert_bias")
    assert not GlmMoeDsaPreTrainedModel.keep_in_fp32_for_weight_transfer("model.layers.3.mlp.router.gate.weight")
    assert not Qwen3MoePreTrainedModel.keep_in_fp32_for_weight_transfer("model.layers.3.mlp.router.gate.weight")


def test_trainer_shard_source_refreshes_in_wire_dtype():
    source = torch.tensor([1.003, 2.007], dtype=torch.float32)

    bf16_shard = TrainerShardSource("bf16", (2,), 0, 0, source, torch.bfloat16)
    bf16_arena = torch.empty(2, dtype=torch.bfloat16)
    bf16_shard.bind(bf16_arena, 0)
    bf16_shard.refresh()
    assert torch.equal(bf16_arena, source.bfloat16())

    fp32_shard = TrainerShardSource("fp32", (2,), 0, 0, source, torch.float32)
    fp32_arena = torch.empty(2, dtype=torch.float32)
    fp32_shard.bind(fp32_arena, 0)
    fp32_shard.refresh()
    assert torch.equal(fp32_arena, source)


def test_validate_mixed_wire_table():
    table = TrainerTable(
        agents=[TrainerAgent(name="trainer", metadata=b"")],
        groups=["layer.0"],
        buffer_count=1,
        tensors=[
            TrainerTensor(
                name="bf16",
                master_dtype="float32",
                dtype="bfloat16",
                shape=(2, 3),
                group=0,
                shards=[
                    TrainerShard(
                        agent=0,
                        row_start=0,
                        num_rows=2,
                        addr=0,
                        row_bytes=3 * torch.bfloat16.itemsize,
                        device_id=0,
                    )
                ],
            ),
            TrainerTensor(
                name="fp32",
                master_dtype="float32",
                dtype="float32",
                shape=(2, 3),
                group=0,
                shards=[
                    TrainerShard(
                        agent=0,
                        row_start=0,
                        num_rows=2,
                        addr=0,
                        row_bytes=3 * torch.float32.itemsize,
                        device_id=0,
                    )
                ],
            ),
        ],
    )

    NIXLWeightBroadcast._validate_table(table)
