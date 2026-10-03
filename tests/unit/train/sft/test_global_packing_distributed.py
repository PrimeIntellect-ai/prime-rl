import copy
import json
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from datasets import Dataset
from renderers.base import RenderedTokens
from transformers import Qwen3Config

from prime_rl.configs.sft import GlobalPackingConfig, SFTDataConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.broker import GlobalDataLoader, GlobalPackedDataset
from prime_rl.trainer.sft.data import SFTDataset, cat_collate
from prime_rl.trainer.world import reset_world
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp

pytestmark = pytest.mark.gpu


class VariableRenderer:
    def __call__(self, example):
        return self

    def render(self, messages, **kwargs):
        content = [ord(char) % 50 + 3 for char in messages[-1]["content"]]
        return RenderedTokens(
            token_ids=[0, *content, 1],
            message_indices=[-1, *([len(messages) - 1] * (len(content) + 1))],
            sampled_mask=[False, *([True] * (len(content) + 1))],
        )

    def get_stop_token_ids(self):
        return [1]


def distributed_packing_worker(rank, cp_size, cp_style, model_family, directory):
    world_size = 2 * cp_size
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world_size)
    )
    reset_world()
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{directory}/init",
        timeout=timedelta(seconds=180),
        device_id=torch.device("cuda", rank),
    )
    delivery_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    data_config = SFTDataConfig(seq_len=32, batch_size=4, num_workers=2, global_packing=GlobalPackingConfig())
    raw = Dataset.from_dict({"prompt": [""] * 5, "completion": ["a" * length for length in [27, 18, 7, 22, 8]]})
    source = SFTDataset(raw, VariableRenderer(), shuffle=False, seq_len=32, max_epochs=1)
    dataset = GlobalPackedDataset(source, data_config, dp_size=2) if rank == 0 else None
    loader = GlobalDataLoader(dataset, data_config, cp_size, delivery_group)
    local_rows = list(loader)
    loader.close()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_rows, group=delivery_group)
    for lane in range(2):
        for peer in range(1, cp_size):
            for primary_row, peer_row in zip(gathered[lane * cp_size], gathered[lane * cp_size + peer], strict=True):
                for key in ("input_ids", "target_ids", "position_ids", "seq_lens", "loss_mask"):
                    torch.testing.assert_close(primary_row[key], peer_row[key], rtol=0, atol=0)
    sample_ids = [tuple(sample_id) for rows in gathered[::cp_size] for row in rows for sample_id in row["sample_ids"]]
    assert sorted(sample_ids) == [(0, index) for index in range(5)]
    assert len(local_rows) == 2
    assert any(not row["loss_mask"].any() for rows in gathered for row in rows)

    torch.manual_seed(13)
    config_kwargs = dict(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    if model_family == "glm4":
        model_config = Glm4MoeConfig(
            **config_kwargs,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
        )
        model_class = Glm4MoeForCausalLM
    else:
        model_config = Qwen3Config(**config_kwargs)
        model_class = Qwen3ForCausalLM
    model_config._attn_implementation = "flash_attention_2"
    model = model_class(model_config).to(device="cuda", dtype=torch.bfloat16)
    inject_prime_lm_head(model, chunk_size=None)
    reference = copy.deepcopy(model)
    reference_loss = torch.zeros((), device="cuda")
    token_count = sum(int(row["loss_mask"].sum()) for rows in gathered[::cp_size] for row in rows)

    for example in raw:
        row = cat_collate([source._process(example)])
        inputs = {key: row[key].cuda() for key in ("input_ids", "position_ids", "seq_lens")}
        labels = row["target_ids"].cuda().masked_fill(~row["loss_mask"].cuda(), IGNORE_INDEX)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = reference(**inputs, labels=labels)["loss"]
        reference_loss += loss.detach()
        (loss / token_count).backward()

    runtime = ModelConfig(cp=cp_size, cp_style=cp_style, ep=1, attn="flash_attention_2", compile=None)
    parallel = get_parallel_dims(runtime, data_config.seq_len)
    if cp_size > 1:
        setup_context_parallel(model, runtime, parallel)
    cp_rank = rank % cp_size
    cp_group = parallel.world_mesh["cp"].get_group() if cp_size > 1 else None
    actual_loss = torch.zeros((), device="cuda")
    for row in local_rows:
        input_ids = row["input_ids"].cuda()
        position_ids = row["position_ids"].cuda()
        seq_lens = row["seq_lens"].cuda()
        labels = row["target_ids"].cuda().masked_fill(~row["loss_mask"].cuda(), IGNORE_INDEX)
        if cp_size > 1:
            input_ids, position_ids = setup_cp_params(
                input_ids, position_ids, cp_rank, cp_size, cp_group, seq_lens=seq_lens, cp_style=cp_style
            )
            labels = shard_for_cp(labels, cp_rank, cp_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids, position_ids, seq_lens=seq_lens, labels=labels, seq_lens_are_pre_shard=cp_size > 1)[
                "loss"
            ]
        actual_loss += loss.detach()
        (loss / token_count).backward()
    dist.all_reduce(actual_loss)
    torch.testing.assert_close(actual_loss, reference_loss, atol=0.1, rtol=0.003)
    max_error = 0.0
    for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
        dist.all_reduce(parameter.grad)
        max_error = max(max_error, float((parameter.grad - expected.grad).abs().max()))
        torch.testing.assert_close(parameter.grad, expected.grad, atol=0.003, rtol=0.03)
    if rank == 0:
        with open(f"{directory}/result.json", "w") as output:
            json.dump(
                dict(
                    model=model_family,
                    cp=cp_size,
                    style=cp_style,
                    samples=sample_ids,
                    tokens=token_count,
                    max_grad_error=max_error,
                ),
                output,
            )
    dist.destroy_process_group(delivery_group)
    dist.destroy_process_group()


@pytest.mark.parametrize("cp_size,cp_style", [(1, "ulysses"), (2, "ulysses"), (2, "ring")])
@pytest.mark.parametrize("model_family", ["qwen3", "glm4"])
def test_broker_dp_cp_loss_gradient_parity(tmp_path, cp_size, cp_style, model_family):
    if torch.cuda.device_count() < 2 * cp_size:
        pytest.skip(f"Requires {2 * cp_size} GPUs")
    multiprocessing.spawn(
        distributed_packing_worker, args=(cp_size, cp_style, model_family, str(tmp_path)), nprocs=2 * cp_size
    )
