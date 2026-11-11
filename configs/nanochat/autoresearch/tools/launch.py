#!/usr/bin/env python
"""Launch one nanochat pretraining run on Model Factory.

python launch.py --depth 12 --residual '{"type":"plain"}' --flops 1e18 --seed 0 --tag iso1e18 [--volume auto]

Computes params/FLOPs on the meta device with the branch's model code, derives steps, weight decay (nanochat
ff46300 rule, B fixed at 2^19), AdamW LR scaling (1/sqrt(d/768)), writes model config + TOML, puts the model config
on the volume, validates, submits with the commit SHA and appends the run to ledger.jsonl.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cpu_import  # noqa: E402
import ledger as L  # noqa: E402

TEAM = os.environ.get("PRIME_TEAM_ID", "")
if TEAM:
    os.environ["PRIME_TEAM_ID"] = TEAM
B = 2**19
SEQ = 2048
POOL = [v for v in os.environ.get("VOLUME_POOL", "").split(",") if v]  # one volume per concurrent run
BOS = 32759
VAL_ROWS = 10240  # 40 * 2^19 tokens
D12_SCALING = 110101632
D_REF = 10.5 * D12_SCALING
BPB_FACTOR = 0.3042629655704041  # masked val loss -> bpb on the first 10240 val rows (see journal)
TERMINAL = {"COMPLETED", "FAILED", "STOPPED", "CANCELLED", "ERROR", "TERMINATED", "CRASHED"}


def sh(cmd, check=True, retries=0, retry_on=("HTTP 403", "locally compiled config")):
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        if p.returncode == 0 or not any(s in out for s in retry_on) or attempt == retries:
            break
        time.sleep(20)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out}")
    return out


def busy_volumes():
    busy = set()
    for r in L.load().values():
        if r.get("status") not in TERMINAL and r.get("volume"):
            busy.add(r["volume"])
        elif r.get("ended_at") and time.time() - r["ended_at"] < 170 and r.get("volume"):
            busy.add(r["volume"])
    return busy


def pick_volume(depth_tokens):
    busy = busy_volumes()
    for v in POOL:
        if v not in busy:
            return v
    return None


def val_interval_and_steps(steps):
    """A val interval dividing ``steps`` (so the last val sees ~final weights), between steps/16 and steps/5;
    nudges steps by at most +-3 when needed."""
    for delta in (0, 1, -1, 2, -2, 3, -3):
        s = steps + delta
        for k in range(8, 17):
            if s % k == 0:
                return s // k, s
        for k in range(5, 8):
            if s % k == 0:
                return s // k, s
    return max(1, steps // 8), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--residual", default='{"type":"plain"}')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--flops", type=float)
    g.add_argument("--tokens", type=float)
    g.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int, default=0, help="0 -> init_seed 42 (branch default); k -> init_seed 42+k")
    ap.add_argument("--data-seed", action="store_true", help="also vary the data order with the seed (needs code)")
    ap.add_argument("--volume", default="auto")
    ap.add_argument("--ref", default=None, help="commit SHA (default: HEAD of the worktree, must be pushed)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default=None, help="short variant name for the ledger")
    ap.add_argument("--optim", action="append", default=[], help="extra optim key=value (TOML literal)")
    ap.add_argument("--extra", action="append", default=[], help="extra top-level TOML lines, e.g. 'val.core=true'")
    ap.add_argument("--warmdown", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=0.2, help="reference weight decay before scaling")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--code-eval", action="store_true", help="use the in-trainer bpb/CORE eval (needs code)")
    ap.add_argument("--core-max-per-task", type=int, default=-1)
    ap.add_argument("--gpu", default="H200_141GB")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    residual = json.loads(a.residual)
    init_seed = 42 + a.seed
    model = cpu_import.build(a.depth, residual)
    c = cpu_import.counts(model)
    fpt = c["flops_per_token"]
    if a.flops:
        steps = round(a.flops / (fpt * B))
    elif a.tokens:
        steps = round(a.tokens / B)
    else:
        steps = a.steps
    interval, steps = val_interval_and_steps(steps)
    tokens = steps * B
    wd = a.wd * D_REF / tokens
    dim = model.config.hidden_size
    lr_scale = (dim / 768) ** -0.5
    variant = a.variant or residual["type"]
    name = f"d{a.depth}-{variant}-{a.tag}-s{a.seed}".replace("_", "-")
    ref = a.ref or subprocess.run(["git", "-C", cpu_import.PRIME, "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()

    volume = a.volume if a.volume != "auto" else pick_volume(tokens)
    if volume is None:
        print("NO_FREE_VOLUME")
        sys.exit(3)

    # model config: content-addressed folder on the volume
    mcfg = {"model_type": "nanochat", "architectures": ["NanochatForCausalLM"], "vocab_size": 32768,
            "hidden_size": dim, "num_hidden_layers": a.depth, "num_attention_heads": dim // 128,
            "num_key_value_heads": dim // 128, "max_position_embeddings": SEQ, "window_pattern": "SSSL",
            "logit_softcap": 15.0, "tie_word_embeddings": False, "init_seed": init_seed, "residual": residual}
    import hashlib
    h = hashlib.sha1(json.dumps(mcfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg_name = f"d{a.depth}-{residual['type']}-{h}".replace("_", "-")
    cfg_dir = ROOT / "configs" / "models" / "cfg" / cfg_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(mcfg, indent=2))

    optim = {"type": '"nanochat_muon"', "total_steps": steps, "weight_decay": f"{wd:.6g}",
             "embedding_lr": f"{0.3 * lr_scale:.6g}", "unembedding_lr": f"{0.004 * lr_scale:.6g}"}
    for kv in a.optim:
        k, v = kv.split("=", 1)
        optim[k] = v
    val_data_extra = f"mask_target_id = {BOS}" if not a.code_eval else ""
    lines = [
        'mode = "pretrain"',
        f"max_steps = {steps}",
        f'gpu_type = "{a.gpu}"',
        f'volume = "{volume}"',
        "",
        "# derived (launch.py): " + json.dumps(dict(depth=a.depth, flops_per_token=fpt, scaling_params=c["scaling"],
                                                     total_params=c["total"], tokens=tokens,
                                                     total_flops=fpt * tokens, weight_decay=wd,
                                                     adamw_lr_scale=lr_scale)),
        "[model]",
        f'name = "/volume/models/cfg/{cfg_name}"',
        'offload = "none"',
        'ac = "None"',
        'ac_offloading = "None"',
        "dp_replicate = 8",
        "",
        "[tokenizer]",
        'name = "/volume/models/tokenizer-rustbpe32k"',
        "",
        "[data]",
        'paths = ["datasets/fineweb100b-packed/train"]',
        f"seq_len = {SEQ}",
        "batch_size = 256",
        f"micro_batch_size = {a.micro_batch}",
        "packed_rows = true",
    ]
    if a.data_seed:
        lines.append(f"seed = {a.seed}")
    lines += [
        "",
        "[val]",
        f"interval = {interval}",
        f"num_samples = {VAL_ROWS}",
    ]
    if a.code_eval:
        lines += ["bpb = true", "final_bpb = true", "core = true", f"core_max_per_task = {a.core_max_per_task}",
                  'eval_bundle = "/volume/eval_bundle"']
    lines += [
        "",
        "[val.data]",
        'paths = ["datasets/fineweb100b-packed/val"]',
        f"seq_len = {SEQ}",
        "packed_rows = true",
        f"micro_batch_size = {a.micro_batch}",
    ]
    if val_data_extra:
        lines.append(val_data_extra)
    lines += ["", "[optim]"] + [f"{k} = {v}" for k, v in optim.items()]
    lines += ["", "[scheduler]", 'type = "warmdown"', f"warmdown_ratio = {a.warmdown}", "",
              "[deployment]", 'type = "single_node"', "num_train_gpus = 8"]
    for e in a.extra:
        lines.append(e)
    toml = "\n".join(lines) + "\n"
    run_dir = ROOT / "configs" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    toml_path = run_dir / f"{name}-{time.strftime('%H%M%S')}.toml"
    toml_path.write_text(toml)
    print(toml)
    print(f"steps={steps} tokens={tokens/1e9:.3f}B flops={fpt*tokens:.3e} fpt={fpt} wd={wd:.4f} vol={volume}")
    if a.dry:
        return

    out = sh(["prime", "--plain", "volumes", "put", volume, str(cfg_dir) + "/", f"models/cfg/{cfg_name}"],
             retries=2)
    out = sh(["prime", "--plain", "train", "validate", str(toml_path), "--ref", ref], retries=3, check=False)
    if "error" in out.lower() and "valid" not in out.lower():
        print(out)
        raise SystemExit("validate failed")
    print(out[-800:])
    out = sh(["prime", "--plain", "train", str(toml_path), "--ref", ref, "-y", "-o", "json"], retries=3)
    m = re.search(r'"id":\s*"([a-z0-9]{20,})"', out) or re.search(r"\b([a-z0-9]{24})\b", out)
    if not m:
        print(out)
        raise SystemExit("could not parse run id")
    run_id = m.group(1)
    rec = dict(run_id=run_id, url=None, sha=ref,
               config=str(toml_path.relative_to(ROOT)), model_config=str((cfg_dir / "config.json").relative_to(ROOT)),
               depth=a.depth, variant=variant, residual=residual, seed=a.seed, init_seed=init_seed,
               data_seed=a.seed if a.data_seed else None, tag=a.tag,
               params_total=c["total"], params_scaling=c["scaling"], flops_per_token=fpt,
               total_flops=fpt * tokens, tokens=tokens, steps=steps, weight_decay=wd, val_interval=interval,
               volume=volume, gpu=a.gpu, status="SUBMITTED", launched_at=time.time(),
               eval_mode="code" if a.code_eval else "masked_loss", note=a.note)
    L.upsert(rec)
    print(f"LAUNCHED {run_id} {name} on {volume}")


if __name__ == "__main__":
    main()
