"""Copy the synthetic Qwen3.8 checkpoint and add seeded random vision tower weights, which the VLM class needs."""

import argparse
import re
import shutil
from pathlib import Path

import torch

from prime_rl.trainer.models import get_custom_vlm_cls
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextConfig
from prime_rl.utils.weights import load_state_dict, load_state_dict_keys, save_state_dict

VISUAL_PREFIX = "model.visual."
ATTN_IMPLEMENTATION = "flash_attention_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="HF checkpoint without model.visual.* tensors")
    parser.add_argument("output_dir", type=Path, help="must not exist")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def print_group_stats(state_dict: dict[str, torch.Tensor]) -> None:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        groups.setdefault(re.sub(r"\.blocks\.\d+\.", ".blocks.*.", name), []).append(tensor)
    width = max(len(group) for group in groups)
    for group, tensors in groups.items():
        values = torch.cat([tensor.double().flatten() for tensor in tensors])
        print(
            f"{group:<{width}}  numel={values.numel():>12,}  mean={values.mean().item():+.4e}  "
            f"std={values.std().item() if values.numel() > 1 else 0.0:.4e}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True)

    config = Qwen3_8FlashNextConfig.from_pretrained(args.source_dir, attn_implementation=ATTN_IMPLEMENTATION)
    model_cls = get_custom_vlm_cls(config)
    torch.manual_seed(args.seed)
    model = model_cls(config)
    print(f"built {model_cls.__module__}.{model_cls.__name__} on CPU with torch.manual_seed({args.seed})")
    model_state = {name: tensor.detach() for name, tensor in model.state_dict().items()}
    model.convert_to_hf(model_state)

    source = load_state_dict(args.source_dir)
    missing_from_model = sorted(source.keys() - model_state.keys())
    assert not missing_from_model, missing_from_model
    added = sorted(model_state.keys() - source.keys())
    non_visual = [name for name in added if not name.startswith(VISUAL_PREFIX)]
    assert added and not non_visual, non_visual
    for name, tensor in source.items():
        assert tensor.shape == model_state[name].shape, (name, tensor.shape, model_state[name].shape)

    visual = {
        name: model_state[name].to(torch.bfloat16) if model_state[name].is_floating_point() else model_state[name]
        for name in added
    }
    print_group_stats(visual)
    save_state_dict(source | visual, args.output_dir)
    for path in sorted(args.source_dir.iterdir()):
        if path.suffix != ".safetensors":
            shutil.copyfile(path, args.output_dir / path.name)

    saved = load_state_dict(args.output_dir)
    assert saved.keys() == source.keys() | visual.keys(), sorted(saved.keys() ^ (source.keys() | visual.keys()))
    for name, tensor in (source | visual).items():
        assert saved[name].dtype == tensor.dtype and torch.equal(saved[name], tensor), name
    assert set(load_state_dict_keys(args.output_dir)) == saved.keys()
    visual_numel = sum(tensor.numel() for tensor in visual.values())
    print(f"kept {len(source)} source tensors bit-exact, added {len(visual)} vision tensors ({visual_numel:,} values)")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
