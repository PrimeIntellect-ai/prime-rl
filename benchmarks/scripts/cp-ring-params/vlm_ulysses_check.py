"""Check the Qwen3.5 VLM deferred ulysses CP path against a no-CP reference on the same packed batch.

torchrun --nproc-per-node=2 vlm_ulysses_check.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/unit/train/models/qwen3_5"))
from test_vlm import get_image_inputs, get_mm_token_type_ids, get_model, get_vlm_config  # noqa: E402

from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn  # noqa: E402
from prime_rl.utils.cp import CPContext  # noqa: E402

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
group = dist.group.WORLD
world = dist.get_world_size()

torch.manual_seed(0)
config = get_vlm_config()
config.vision_config.spatial_merge_size = 2
model = get_model(config)
with torch.no_grad():
    for param in model.parameters():
        dist.broadcast(param, src=0)

# Two packed documents, each with one 4x4-patch image (4 merged tokens), lengths 30 and 34.
pixel_values_one, grid_one, n_img = get_image_inputs(config)
grid = torch.tensor([[1, 4, 4], [1, 4, 4]], device="cuda")
patch_dim = pixel_values_one.shape[1]
pixel_values = torch.randn(int(grid.prod(dim=1).sum()), patch_dim, device="cuda", dtype=torch.bfloat16)
n_img = int(grid[0].prod()) // config.vision_config.spatial_merge_size**2
docs = []
for length in (30, 34):
    text = torch.randint(0, 200, (length - n_img,), device="cuda")
    docs.append(torch.cat([text[:6], torch.full((n_img,), config.image_token_id, device="cuda"), text[6:]]))
input_ids = torch.cat(docs).unsqueeze(0)
dist.broadcast(input_ids, src=0)
dist.broadcast(pixel_values, src=0)
seq_lens = torch.tensor([30, 34], device="cuda")
mm_token_type_ids = get_mm_token_type_ids(input_ids, config.image_token_id)
inputs = dict(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_grid_thw=grid,
    mm_token_type_ids=mm_token_type_ids,
    seq_lens=seq_lens,
)

with torch.no_grad():
    reference = model(**inputs)["logits"].float()

substitute_ulysses_attn(group, attn_impl=model.config._attn_implementation)
from prime_rl.trainer.models.layers.attn import FlashAttention  # noqa: E402

ulysses_calls = 0
patched = FlashAttention._compute_attention


def counting_compute_attention(self, *args, **kwargs):
    global ulysses_calls
    ulysses_calls += 1
    return patched(self, *args, **kwargs)


FlashAttention._compute_attention = counting_compute_attention
cp_context = CPContext(group, local_rank, world, "ulysses")
for module in model.modules():
    if hasattr(module, "cp_context"):
        module.cp_context = cp_context

with torch.no_grad():
    sharded = model(**inputs)["logits"].float()

shard = input_ids.shape[1] // world
expected = reference[:, local_rank * shard : (local_rank + 1) * shard]
max_abs = (sharded - expected).abs().max().item()
scale = expected.abs().max().item()
print(
    f"rank {local_rank}: ulysses attention calls={ulysses_calls} local logits {tuple(sharded.shape)} max_abs_diff={max_abs:.3e} ref_max={scale:.3e}",
    flush=True,
)
assert ulysses_calls > 0
torch.testing.assert_close(sharded, expected, rtol=2e-2, atol=2e-2)
print(f"rank {local_rank}: PASS", flush=True)
dist.destroy_process_group()
