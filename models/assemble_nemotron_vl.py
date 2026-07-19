"""Assemble the Nemotron-VL composite checkpoint (vision graft).

Combines three sources into one HF-style checkpoint directory:

  1. LM backbone   — NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (frozen during training)
  2. Vision tower  — CRADIO v4-H, extracted from Nano Omni's `vision_model.*` keys
                     (the exact encoder NVIDIA grafted onto a NemotronH backbone)
  3. Projector     — InternVL-style `mlp1`: RMSNorm(5120) -> Linear(5120, 20480)
                     -> ReLU^2 -> Linear(20480, d_lm), all bias-free (matches Nano
                     Omni's modeling.py). The norm and first matrix are warm-started
                     from Nano Omni (target-independent); the output matrix
                     (20480 -> d_lm) is freshly initialized since d_lm differs
                     (Nano 2688, Super 4096, Ultra 8192).

Output layout follows Nano Omni's key convention so the eventual prime-rl
NemotronVL class and any vLLM plugin see a familiar shape:

  language_model.backbone.*   language_model.lm_head.weight   (LM, mtp.* dropped)
  vision_model.radio_model.*                                  (CRADIO, verbatim)
  mlp1.{0,1,3}.weight                                         (projector)

LM shards are large (~240GB for Super), so by default they are hardlinked
into the output dir and only re-keyed via the index... which safetensors does
NOT allow (key names live inside the shard files). Therefore two modes:

  --mode hardlink  (default) LM shards are hardlinked unmodified; their keys KEEP
                   the original text-checkpoint names (`backbone.*`, `lm_head.weight`,
                   `mtp.*`). Only vision/projector tensors get a new shard. The
                   composite index maps both conventions; the NemotronVL loader
                   must treat un-prefixed keys as language-model keys. Fast, ~0
                   extra disk.
  --mode rewrite   Every LM shard is streamed and rewritten with the
                   `language_model.` prefix (and `mtp.*` dropped) for a fully
                   Omni-consistent layout. Costs a full copy of the LM (~240GB
                   disk + I/O time).

Tokenizer: Super and Omni share the same 131072-token Nemotron vocab with 1000
reserved special slots; Omni simply renamed ids 18/19/20 from <SPECIAL_18/19/20>
to <image>/<img>/</img>. We apply the same renames to Super's tokenizer files —
no vocab resize, no embedding surgery. NOTE: rows 18/19/20 of Super's (frozen)
embedding matrix were never trained as image markers; <image> (id 18) doesn't
matter (replaced by projector outputs), but <img>/</img> enter the LM as
untrained embeddings. If SFT stalls, making those two rows trainable is a cheap
follow-up ablation.

Usage:
    uv run python models/assemble_nemotron_vl.py \
        --super models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
        --omni  models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        --output models/Nemotron-3-Super-VL-graft
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

VISION_PREFIXES = ("vision_model.", "mlp1.")
LM_DROP_PREFIXES = ("mtp.",)
# Omni fields that define the image pipeline (tiling, pixel shuffle, markup).
# Copied verbatim into the composite config; the renderer and model class read
# them from here as the single source of truth.
OMNI_IMAGE_FIELDS = [
    "downsample_ratio",
    "force_image_size",
    "patch_size",
    "use_thumbnail",
    "ps_version",
    "image_tag_type",
    "img_context_token_id",
    "img_context_token",
    "img_start_token",
    "img_end_token",
    "vit_hidden_size",
    "projector_hidden_size",
    "norm_mean",
    "norm_std",
]
# Reserved Nemotron special-token slots that Omni repurposed for image markup.
TOKEN_RENAMES = {"<SPECIAL_18>": "<image>", "<SPECIAL_19>": "<img>", "<SPECIAL_20>": "</img>"}
IMG_START_TOKEN_ID = 19
IMG_END_TOKEN_ID = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--super", dest="super_dir", type=Path, required=True, help="Local Super checkpoint dir")
    parser.add_argument("--omni", dest="omni_dir", type=Path, required=True, help="Local Nano Omni checkpoint dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir for the composite checkpoint")
    parser.add_argument("--mode", choices=["hardlink", "rewrite"], default="hardlink")
    parser.add_argument(
        "--projector-init",
        choices=["warmstart", "random"],
        default="warmstart",
        help="warmstart: mlp1.0/mlp1.1 from Nano Omni, mlp1.3 fresh. random: all three fresh.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output dir")
    return parser.parse_args()


def load_index(ckpt_dir: Path) -> dict:
    return json.loads((ckpt_dir / "model.safetensors.index.json").read_text())


def extract_vision_and_projector(omni_dir: Path) -> dict[str, torch.Tensor]:
    """Load all vision_model.* and mlp1.* tensors from the Omni checkpoint."""
    weight_map = load_index(omni_dir)["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        if key.startswith(VISION_PREFIXES):
            shards.setdefault(fname, []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for fname, keys in sorted(shards.items()):
        with safe_open(omni_dir / fname, framework="pt") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    n_vision = sum(k.startswith("vision_model.") for k in tensors)
    n_proj = sum(k.startswith("mlp1.") for k in tensors)
    print(f"Extracted {n_vision} vision tensors + {n_proj} projector tensors from {len(shards)} Omni shard(s)")
    return tensors


def build_projector(
    omni_tensors: dict[str, torch.Tensor],
    vit_hidden_size: int,
    projector_hidden_size: int,
    lm_hidden_size: int,
    init: str,
    initializer_range: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Projector tensors in Omni naming: mlp1 = Sequential(RMSNorm, Linear, ReLU^2, Linear).

    All modules are bias-free, matching Omni (only .weight keys exist).
    """
    in_dim = vit_hidden_size * 4  # 2x2 pixel-shuffle concatenation
    generator = torch.Generator().manual_seed(seed)

    def fresh(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float32).normal_(0.0, initializer_range, generator=generator)

    if init == "warmstart":
        ln = omni_tensors["mlp1.0.weight"]
        fc1 = omni_tensors["mlp1.1.weight"]
        assert ln.shape == (in_dim,), f"mlp1.0 shape {ln.shape} != ({in_dim},)"
        assert fc1.shape == (projector_hidden_size, in_dim), f"mlp1.1 shape {fc1.shape}"
        print(f"Projector: warm-starting LN({in_dim}) and Linear({in_dim} -> {projector_hidden_size}) from Nano Omni")
    else:
        ln = torch.ones(in_dim, dtype=torch.float32)
        fc1 = fresh((projector_hidden_size, in_dim))
        print(f"Projector: random init for LN and Linear({in_dim} -> {projector_hidden_size})")

    fc2 = fresh((lm_hidden_size, projector_hidden_size))
    print(f"Projector: fresh Linear({projector_hidden_size} -> {lm_hidden_size}), std={initializer_range}")

    projector = {
        "mlp1.0.weight": ln.to(torch.bfloat16),
        "mlp1.1.weight": fc1.to(torch.bfloat16),
        "mlp1.3.weight": fc2.to(torch.bfloat16),
    }
    n_params = sum(t.numel() for t in projector.values())
    print(f"Projector total: {n_params / 1e6:.1f}M params")
    return projector


def write_lm_shards(super_dir: Path, output: Path, mode: str) -> dict[str, str]:
    """Place LM shards into the output dir; return their weight_map entries."""
    index = load_index(super_dir)
    weight_map: dict[str, str] = {}
    shard_names = sorted(set(index["weight_map"].values()))

    if mode == "hardlink":
        for fname in shard_names:
            target = output / fname
            if not target.exists():
                try:
                    target.hardlink_to(super_dir / fname)
                except OSError:  # cross-device: fall back to copy
                    shutil.copy2(super_dir / fname, target)
        for key, fname in index["weight_map"].items():
            if key.startswith(LM_DROP_PREFIXES):
                continue  # mtp keys stay inside the linked shards but are omitted from the index
            weight_map[key] = fname
        print(f"Hardlinked {len(shard_names)} LM shards (keys keep original names, mtp.* dropped from index)")
        return weight_map

    # rewrite mode: stream each shard, prefix keys, drop mtp.*
    for i, fname in enumerate(shard_names):
        out_name = fname.replace("model-", "model-lm-")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(super_dir / fname, framework="pt") as f:
            for key in f.keys():
                if key.startswith(LM_DROP_PREFIXES):
                    continue
                new_key = f"language_model.{key}"
                tensors[new_key] = f.get_tensor(key)
                weight_map[new_key] = out_name
        save_file(tensors, output / out_name, metadata={"format": "pt"})
        print(f"  rewrote shard {i + 1}/{len(shard_names)}: {out_name} ({len(tensors)} tensors)")
    return weight_map


def build_config(super_dir: Path, omni_dir: Path, mode: str) -> dict:
    super_config = json.loads((super_dir / "config.json").read_text())
    omni_config = json.loads((omni_dir / "config.json").read_text())

    # Clean vision config matching prime-rl's RadioViTConfig; the raw RADIO config
    # (timm args blob) is preserved alongside as radio_config_reference.json.
    vision_config = {
        "model_type": "radio_vit",
        "hidden_size": omni_config["vit_hidden_size"],
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "intermediate_size": 4 * omni_config["vit_hidden_size"],
        "patch_size": omni_config["patch_size"],
        "max_img_size": omni_config["vision_config"]["max_resolution"],
        "num_cls_tokens": 10,
        "layer_norm_eps": 1e-6,
        "qkv_bias": True,
    }

    text_config = dict(super_config)
    text_config.pop("auto_map", None)

    config = {
        "architectures": ["NemotronVLForCausalLM"],
        "model_type": "nemotron_vl",
        "text_config": text_config,
        "vision_config": vision_config,
        # min/max_num_patches govern dynamic tiling; Omni keeps them in
        # preprocessor_config.json (copied alongside), mirrored here for the renderer.
        "min_num_patches": 1024,
        "max_num_patches": 13312,
        "img_start_token_id": IMG_START_TOKEN_ID,
        "img_end_token_id": IMG_END_TOKEN_ID,
        # hardlink mode keeps original text-checkpoint key names for LM shards
        "lm_weights_prefixed": mode == "rewrite",
        "torch_dtype": "bfloat16",
    }
    for field in OMNI_IMAGE_FIELDS:
        config[field] = omni_config[field]
    return config


def write_tokenizer(super_dir: Path, output: Path) -> None:
    """Copy Super tokenizer files, renaming reserved slots 18/19/20 to image tokens."""
    tok_config = json.loads((super_dir / "tokenizer_config.json").read_text())
    renamed = 0
    for entry in tok_config.get("added_tokens_decoder", {}).values():
        if entry["content"] in TOKEN_RENAMES:
            entry["content"] = TOKEN_RENAMES[entry["content"]]
            renamed += 1
    (output / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2, ensure_ascii=False))

    tokenizer = json.loads((super_dir / "tokenizer.json").read_text())
    for entry in tokenizer.get("added_tokens", []):
        if entry["content"] in TOKEN_RENAMES:
            entry["content"] = TOKEN_RENAMES[entry["content"]]
            renamed += 1
    vocab = tokenizer.get("model", {}).get("vocab", {})
    for old, new in TOKEN_RENAMES.items():
        if old in vocab:
            vocab[new] = vocab.pop(old)
            renamed += 1
    (output / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False))

    shutil.copy2(super_dir / "special_tokens_map.json", output / "special_tokens_map.json")
    shutil.copy2(super_dir / "chat_template.jinja", output / "chat_template.jinja")
    print(
        f"Tokenizer: copied from Super with {renamed} token renames "
        f"({', '.join(f'{o} -> {n}' for o, n in TOKEN_RENAMES.items())})"
    )


def main() -> None:
    args = parse_args()
    output: Path = args.output
    if output.exists():
        if not args.force:
            raise SystemExit(f"{output} already exists; pass --force to overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    omni_config = json.loads((args.omni_dir / "config.json").read_text())
    super_config = json.loads((args.super_dir / "config.json").read_text())
    lm_hidden_size = super_config["hidden_size"]

    # 1. Vision tower + projector
    omni_tensors = extract_vision_and_projector(args.omni_dir)
    vision_tensors = {k: v for k, v in omni_tensors.items() if k.startswith("vision_model.")}
    projector_tensors = build_projector(
        omni_tensors,
        vit_hidden_size=omni_config["vit_hidden_size"],
        projector_hidden_size=omni_config["projector_hidden_size"],
        lm_hidden_size=lm_hidden_size,
        init=args.projector_init,
        initializer_range=super_config.get("initializer_range", 0.02),
        seed=args.seed,
    )
    for key, tensor in vision_tensors.items():
        assert tensor.dtype in (torch.bfloat16, torch.float32), f"{key}: unexpected dtype {tensor.dtype}"
    vision_shard = "model-vision.safetensors"
    save_file({**vision_tensors, **projector_tensors}, output / vision_shard, metadata={"format": "pt"})
    print(f"Wrote {vision_shard} ({len(vision_tensors) + len(projector_tensors)} tensors)")

    # 2. LM shards
    weight_map = write_lm_shards(args.super_dir, output, args.mode)
    for key in list(vision_tensors) + list(projector_tensors):
        weight_map[key] = vision_shard

    # 3. Index
    total_size = 0
    for fname in sorted(set(weight_map.values())):
        total_size += (output / fname).stat().st_size
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"Index: {len(weight_map)} tensors, {total_size / 1e9:.1f} GB across {len(set(weight_map.values()))} shards")

    # 4. Config, tokenizer, aux files
    config = build_config(args.super_dir, args.omni_dir, args.mode)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    write_tokenizer(args.super_dir, output)
    shutil.copy2(args.super_dir / "generation_config.json", output / "generation_config.json")
    # Omni's image tiling/preprocessing reference for the renderer implementation.
    shutil.copy2(args.omni_dir / "preprocessor_config.json", output / "preprocessor_config.json")
    shutil.copy2(args.omni_dir / "image_processing.py", output / "omni_image_processing_reference.py")
    (output / "radio_config_reference.json").write_text(json.dumps(omni_config["vision_config"], indent=2))

    # 5. Sanity checks
    with safe_open(output / vision_shard, framework="pt") as f:
        fc2 = f.get_tensor("mlp1.3.weight")
        assert fc2.shape == (lm_hidden_size, omni_config["projector_hidden_size"])
        assert fc2.dtype == torch.bfloat16
        assert not fc2.isnan().any()
        std = fc2.float().std().item()
        assert math.isfinite(std) and 0 < std < 1, f"suspicious mlp1.3 std: {std}"
    print(f"Sanity: mlp1.3 shape {tuple(fc2.shape)}, std {std:.4f}")
    print(f"\nDone: {output}")


if __name__ == "__main__":
    main()
