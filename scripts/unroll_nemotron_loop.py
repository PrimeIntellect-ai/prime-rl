#!/usr/bin/env python3
"""Unroll a PrimeRL looped NemotronH checkpoint into a standard HF NemotronH checkpoint for vLLM."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain
from prime_rl.trainer.models.nemotron_h.loop_unroll import unroll_looped_state_dict
from prime_rl.trainer.weights import load_state_dict, save_state_dict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Source step directory (with config.json & weights)")
    p.add_argument("--dst", type=Path, required=True, help="Output HF directory for unrolled checkpoint")
    p.add_argument("--num-loops", type=int, default=2)
    p.add_argument("--start-layer", type=int, default=6)
    p.add_argument("--end-layer", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    args = p.parse_args()

    cfg_dict = json.loads((args.src / "config.json").read_text())
    loop_cfg = {
        "num_loops": args.num_loops,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }

    sd = load_state_dict(args.src)
    unrolled_prime_sd, new_cfg_dict = unroll_looped_state_dict(sd, cfg_dict, loop_cfg)
    del sd

    # Convert Prime format -> standard HF NemotronH format for vLLM
    new_cfg = NemotronHConfig.from_dict(new_cfg_dict)
    for op in reversed(conversion_chain(new_cfg)):
        op.to_hf(unrolled_prime_sd)

    args.dst.mkdir(parents=True, exist_ok=True)
    save_state_dict(unrolled_prime_sd, args.dst)
    (args.dst / "config.json").write_text(json.dumps(new_cfg_dict, indent=2) + "\n")
    for fname in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        src_f = args.src / fname
        if src_f.is_file():
            shutil.copy2(src_f, args.dst / fname)
    print(f"Unrolled {cfg_dict['num_hidden_layers']} -> {new_cfg_dict['num_hidden_layers']} layers into {args.dst}")


if __name__ == "__main__":
    main()
