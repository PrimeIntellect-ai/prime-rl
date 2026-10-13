"""Throwaway probe: does ring CP recompile compiled, AC-wrapped decoder layers per micro batch?

torchrun --nproc-per-node=2 probe.py <llama|gpt_oss> <patterns|nopatterns>
Run with TORCH_LOGS=recompiles and count "Recompiling function" lines per rank.
"""

import os
import sys

import torch
import torch.distributed as dist

import prime_rl.trainer.model as model_mod
from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig
from prime_rl.trainer.models.layers.attn import substitute_ring_attn
from prime_rl.utils.cp import CPContext, setup_cp_params
from prime_rl.utils.utils import default_dtype

arch, mode = sys.argv[1], sys.argv[2]

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
group = dist.group.WORLD
world = dist.get_world_size()

if arch == "llama":
    from transformers.models.llama.configuration_llama import LlamaConfig

    from prime_rl.trainer.models.llama import LlamaForCausalLM as Model

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
    config._attn_implementation = "flash_attention_2"
    attn_impl = "flash_attention_2"
else:
    from prime_rl.trainer.models.gpt_oss import GptOssConfig
    from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as Model

    config = GptOssConfig(
        num_hidden_layers=2,
        num_local_experts=4,
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=128,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        num_experts_per_tok=2,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
        attn_implementation="flash_attention_4",
        use_cache=False,
    )
    attn_impl = "flash_attention_4"

torch.manual_seed(0)
with torch.device("cuda"), default_dtype(torch.bfloat16):
    model = Model._from_config(config)

substitute_ring_attn(group, heads_k_stride=1, attn_impl=attn_impl)
cp_context = CPContext(group, local_rank, world, "ring")
for module in model.modules():
    if hasattr(module, "cp_context"):
        module.cp_context = cp_context

if mode == "nopatterns":
    model_mod.DYNAMIC_INT_ATTRS = ()
model_mod.apply_ac(model, ActivationCheckpointConfig())
model_mod.apply_compile(model, CompileConfig())
if local_rank == 0:
    print("dynamic_sources:", torch.compiler.config.dynamic_sources, flush=True)

# Total length 2048. Rank 1's shard starts at token 1024: its local key range starts at 0 when the first
# document crosses 1024, and at a later document start otherwise. The local document count varies too.
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
inner = model.model
for step, lens in enumerate(batches):
    seq_lens = torch.tensor(lens, device="cuda")
    position_ids = torch.cat([torch.arange(n, device="cuda") for n in lens]).unsqueeze(0)
    input_ids = torch.randint(0, config.vocab_size, (1, sum(lens)), device="cuda")
    input_ids, position_ids = setup_cp_params(input_ids, position_ids, local_rank, world)
    out = inner(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens, seq_lens_are_pre_shard=True)
    out.last_hidden_state.float().square().mean().backward()
    torch.cuda.synchronize()
    if local_rank == 0:
        print(f"step {step} done lens={lens}", flush=True)

dist.destroy_process_group()
