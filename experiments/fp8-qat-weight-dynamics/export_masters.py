"""Export the fp32 master weights of a DCP checkpoint as HF-named safetensors, optionally keeping only the first experts.

bf16 serving weights are exactly ``bf16(master)``, so this export replaces tools/convert_dcp_to_bf16.py for the
analysis. Without ``--experts-per-layer`` every tensor is kept.

Usage (on a GPU node, from the repo root):
    uv run --no-sync torchrun --nproc-per-node 8 experiments/fp8-qat-weight-dynamics/export_masters.py \
        <run>/checkpoints/step_<n> <output_dir> [--experts-per-layer 16]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from convert_dcp_to_bf16 import (  # noqa: E402
    check_not_lora,
    resolve_dcp_dir,
    resolve_run_configs,
    setup_single_process_env,
)

from prime_rl.trainer.ckpt import AppState  # noqa: E402
from prime_rl.trainer.model import setup_model  # noqa: E402
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep  # noqa: E402
from prime_rl.trainer.utils import setup_torch_distributed  # noqa: E402
from prime_rl.trainer.world import get_world  # noqa: E402
from prime_rl.utils.logger import setup_logger  # noqa: E402
from prime_rl.utils.weights import (  # noqa: E402
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict_parallel,
)

EXPERT_INDEX = re.compile(r"\.mlp\.experts\.(\d+)\.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ckpt_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--experts-per-layer", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logger("info")
    dcp_dir = resolve_dcp_dir(args.ckpt_dir)
    model_config, _ = resolve_run_configs(dcp_dir.parent)
    check_not_lora(model_config, dcp_dir)
    setup_single_process_env()
    setup_torch_distributed()
    resolve_ep(model_config)
    model = setup_model(model_config, get_parallel_dims(model_config), loading_from_checkpoint_later=True)

    logger.info(f"Loading DCP checkpoint from {dcp_dir}")
    dcp_load(state_dict={"app": AppState(model, [], None, None)}, checkpoint_id=dcp_dir)
    state_dict = convert_state_dict_to_hf(model, gather_weights_parallel(model, dtype=torch.float32))
    if args.experts_per_layer is not None:
        state_dict = {
            name: tensor
            for name, tensor in state_dict.items()
            if (match := EXPERT_INDEX.search(name)) is None or int(match.group(1)) < args.experts_per_layer
        }
    nonfinite = sum(int((~torch.isfinite(tensor)).sum()) for tensor in state_dict.values())
    counts = torch.tensor([len(state_dict), nonfinite], device="cuda")
    dist.all_reduce(counts)

    logger.info(f"Writing {int(counts[0])} fp32 tensors to {args.output_dir}")
    save_state_dict_parallel(state_dict, args.output_dir)
    if get_world().is_master:
        summary = {"tensors": int(counts[0]), "nonfinite": int(counts[1]), "experts_per_layer": args.experts_per_layer}
        (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Done: {summary}")
        assert summary["nonfinite"] == 0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
