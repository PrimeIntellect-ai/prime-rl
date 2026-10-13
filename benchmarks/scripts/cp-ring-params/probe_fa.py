"""Probe: does a fullgraph-compiled, AC-wrapped Llama recompile per micro batch under CP, for a given FA backend?

torchrun --nproc-per-node=2 probe_fa.py <flash_attention_2|3|4> <ring|ulysses>
Runs on both the branch and main (detects the old setup_cp_params signature). Run with TORCH_LOGS=recompiles.
"""

import inspect
import os
import sys

import torch
import torch.distributed as dist
from transformers.models.llama.configuration_llama import LlamaConfig

import prime_rl.trainer.model as model_mod
from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.utils.cp import CPContext, setup_cp_params
from prime_rl.utils.utils import default_dtype

attn_impl, cp_style = sys.argv[1], sys.argv[2]

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
group = dist.group.WORLD
world = dist.get_world_size()

config = LlamaConfig(
    hidden_size=512,
    intermediate_size=1024,
    max_position_embeddings=4096,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=2,
    vocab_size=1024,
    rope_theta=10000.0,
)
config._attn_implementation = attn_impl

torch.manual_seed(0)
with torch.device("cuda"), default_dtype(torch.bfloat16):
    model = LlamaForCausalLM._from_config(config)

if cp_style == "ring":
    from prime_rl.trainer.models.layers.attn import substitute_ring_attn

    substitute_ring_attn(group, heads_k_stride=1, attn_impl=attn_impl)
else:
    from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn

    substitute_ulysses_attn(group, attn_impl=attn_impl)
cp_context = CPContext(group, local_rank, world, cp_style)
for module in model.modules():
    if hasattr(module, "cp_context"):
        module.cp_context = cp_context

model_mod.apply_ac(model, ActivationCheckpointConfig())
model_mod.apply_compile(model, CompileConfig(fullgraph=True))

old_signature = "seq_lens" in inspect.signature(setup_cp_params).parameters
batches = [
    [2048],
    [300, 1500, 248],
    [1200, 500, 348],
    [100, 200, 300, 400, 500, 548],
    [700, 700, 648],
    [2000, 48],
    [64] * 32,
    [1030, 1018],
]
for step, lens in enumerate(batches):
    seq_lens = torch.tensor(lens, device="cuda")
    position_ids = torch.cat([torch.arange(n, device="cuda") for n in lens]).unsqueeze(0)
    input_ids = torch.randint(0, config.vocab_size, (1, sum(lens)), device="cuda")
    if old_signature:
        input_ids, position_ids = setup_cp_params(
            input_ids, position_ids, local_rank, world, group, seq_lens=seq_lens, cp_style=cp_style
        )
    else:
        input_ids, position_ids = setup_cp_params(input_ids, position_ids, local_rank, world)
    out = model.model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens, seq_lens_are_pre_shard=True)
    out.last_hidden_state.float().square().mean().backward()
    torch.cuda.synchronize()
    if local_rank == 0:
        print(f"step {step} done lens={lens}", flush=True)

dist.destroy_process_group()
