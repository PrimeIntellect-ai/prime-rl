"""Send checkpoint weights through PrimeRL's trainer-side FP8/NCCL path."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops
from prime_rl.trainer.models.glm4_moe.kernel_conversion import convert_glm4_layer_to_vllm_kernel
from prime_rl.transports.weights.nccl import broadcast_integer, broadcast_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    index = json.loads((args.path / "model.safetensors.index.json").read_text())["weight_map"]
    num_layers = json.loads((args.path / "config.json").read_text())["num_hidden_layers"]
    groups = defaultdict(lambda: defaultdict(list))
    for name, filename in index.items():
        layer = int(name.split(".")[2]) if name.startswith("model.layers.") else -1
        if layer >= num_layers:
            continue  # The inference/trainer model does not load the MTP layer.
        groups[layer][filename].append(name)
    pg = StatelessProcessGroup.create(host="127.0.0.1", port=args.port, rank=0, world_size=2, store_timeout=600)
    comm = PyNcclCommunicator(pg, device=0)
    broadcast_integer(len(groups), comm)
    with torch.no_grad():
        for layer, files in sorted(groups.items()):
            state = {}
            for filename, names in files.items():
                with safe_open(args.path / filename, framework="pt", device="cpu") as handle:
                    for name in names:
                        state[name] = handle.get_tensor(name).cuda()
            if layer >= 0:
                apply_hf_to_prime(state, glm_moe_layer_ops(layer))
                state = convert_glm4_layer_to_vllm_kernel(state, layer, quantize_fp8=True)
            print("SENDER_LAYER", layer, "tensors", len(state), flush=True)
            broadcast_state_dict(state, comm)
            del state
    torch.cuda.synchronize()
    print("SENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
