"""Print which Muon param group each parameter of a meta-device model lands in."""

import re
from collections import Counter
from types import SimpleNamespace

import torch
from transformers import AutoConfig

import prime_rl.trainer.optim as optim
from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.models import get_custom_causal_lm_cls


class CapturedParamGroups(Exception):
    def __init__(self, param_groups):
        self.param_groups = param_groups


class CapturingMuon:
    def __init__(self, params, **kwargs):
        raise CapturedParamGroups(params)


def build_meta_model(model_name: str, **config_overrides):
    model_config = AutoConfig.from_pretrained(model_name)
    for key, value in config_overrides.items():
        setattr(model_config, key, value)
    with torch.device("meta"):
        return get_custom_causal_lm_cls(model_config)._from_config(model_config)


def capture_param_groups(model, ep_enabled: bool):
    parallel_dims = SimpleNamespace(
        ep_enabled=ep_enabled,
        dp_shard_enabled=False,
        cp_enabled=False,
        dp_replicate_enabled=False,
        world_mesh=None,
    )
    named_params = list(model.named_parameters())
    optim.Muon = CapturingMuon
    try:
        optim._create_muon_optimizer(MuonConfig(weight_decay=0.0), named_params, parallel_dims, model=None, lr=1e-5)
    except CapturedParamGroups as captured:
        return captured.param_groups, {id(p): n for n, p in named_params}
    raise RuntimeError("Muon was never constructed")


def print_param_groups(label: str, model, ep_enabled: bool):
    param_groups, name_by_id = capture_param_groups(model, ep_enabled)
    print(f"== {label}")
    for group in param_groups:
        layerless_names = Counter(re.sub(r"\.\d+\.", ".*.", name_by_id[id(p)]) for p in group["params"])
        print(f"  {group['algorithm']} mesh={group.get('distributed_mesh_name')} params={len(group['params'])}")
        for name, count in sorted(layerless_names.items()):
            print(f"    {name} x{count}")


def main():
    gpt_oss = build_meta_model("unsloth/gpt-oss-20b-BF16", num_hidden_layers=2)
    print_param_groups("gpt-oss-20b ep=1", gpt_oss, ep_enabled=False)
    print_param_groups("gpt-oss-20b ep=8", gpt_oss, ep_enabled=True)
    flash_next = build_meta_model("Qwen/Qwen3.8-Flash-Next", num_hidden_layers=4, ngram_vocab_size_base=1000)
    print_param_groups("Qwen3.8-Flash-Next ep=8", flash_next, ep_enabled=True)
    nemotron = build_meta_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", mlp_bias=True)
    print_param_groups("Nemotron-3-Nano mlp_bias=True ep=8", nemotron, ep_enabled=True)


if __name__ == "__main__":
    main()
