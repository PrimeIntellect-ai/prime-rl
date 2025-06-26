# save_ckpt_demo.py
import hashlib
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

from zeroband.utils.models import ModelType


def compute_state_dict_hash(state_dict: dict) -> str:
    """Compute SHA256 hash of a state dict."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode())
        tensor = state_dict[key]
        # Convert bf16 to float32 for numpy compatibility
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        hasher.update(tensor.cpu().numpy().tobytes())
    return hasher.hexdigest()


def apply_fsdp(model: ModelType, reshard_after_forward: bool = False):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def save_ckpt_for_rollout_fast(model, path, dtype=torch.bfloat16):
    rank = dist.get_rank()
    copy_stream = torch.cuda.Stream()  # side stream for copies
    default_stream = torch.cuda.current_stream()  # usually the default stream

    path.mkdir(parents=True, exist_ok=True)
    print(f"[rank={rank}] Saving rollout ckpt at {path}")

    cpu_state = {}
    for k, v in model.state_dict().items():
        # materialise and cast on the default stream
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to(dtype).full_tensor()
        else:
            v = v.to(dtype)

        if rank == 0:
            host_buf = torch.empty_like(v, device="cpu", pin_memory=True)

            # ▸ 1. make the copy stream wait for all prior work on default
            copy_stream.wait_stream(default_stream)

            # ▸ 2. launch the async copy *inside* the copy stream context
            with torch.cuda.stream(copy_stream):
                host_buf.copy_(v, non_blocking=True)
                # ▸ 3. tell the allocator that v is still in use by copy_stream
                v.record_stream(copy_stream)

            cpu_state[k] = host_buf

    # ensure all outstanding copies are finished before we serialise
    copy_stream.synchronize()

    if rank == 0:
        torch.save(cpu_state, path / "model.pt")


def save_ckpt_for_rollout_slow(model, path: Path, dtype: torch.dtype = torch.bfloat16):
    """
    Save the checkpoint for rollout as one unified safetensors file.
    """
    world_info = dist.get_rank(), dist.get_world_size()
    rank, _ = world_info

    path.mkdir(parents=True, exist_ok=True)
    print(f"[rank={rank}] Saving rollout ckpt at {path}")

    cpu_state = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.distributed.tensor.DTensor):
            value = value.to(dtype).full_tensor()  # all-gather
        else:
            value = value.to(dtype)

        if rank == 0:
            host_buf = torch.empty_like(value, device="cpu")
            host_buf.copy_(value)
            cpu_state[key] = host_buf

    if rank == 0:
        torch.save(cpu_state, path / "model.pt")


def hash_ckpt(ckpt_dir: Path):
    # Load checkpoint and compute hash
    ckpt_path = ckpt_dir / "model.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")

    return compute_state_dict_hash(state_dict)


def main():
    # env vars are filled in by torchrun
    rank = int(os.environ["RANK"])
    # world_sz = int(os.environ["WORLD_SIZE"])
    # device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(rank)

    # Load Qwen3 0.6B model
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3-0.6B")

    # Apply FSDP sharding
    apply_fsdp(model, reshard_after_forward=False)

    ckpt_dir = Path("demo_ckpt")
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    save_ckpt_for_rollout_fast(model, ckpt_dir)
    if rank == 0:
        print(f"Fast ckpt hash: {hash_ckpt(ckpt_dir)}")

    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    save_ckpt_for_rollout_slow(model, ckpt_dir)
    if rank == 0:
        print(f"Slow ckpt hash: {hash_ckpt(ckpt_dir)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
