"""Diff our vendored vision path against NVIDIA's own code, weight-for-weight.

Two stages (separate envs — the reference needs timm/open_clip, which conflict with the
project venv's pinned torch/flash-attn; our side is self-contained and needs no timm):

Stage "ref" (isolated env, NO prime_rl imports) — executes NVIDIA's code verbatim:
  - RADIO tower: `AutoModel.from_config(omni_config.vision_config, trust_remote_code=True)`
    -> nvidia/C-RADIOv4-H hub code, exactly how Nano Omni builds it.
  - pixel_shuffle / _extract_feature_single: unbound methods of the Nano Omni model class,
    called on a shim namespace so the 30B LM is never instantiated.
  - mlp1: the Omni module's RMSNorm/SquaredReLU classes with the ORIGINAL 20480->2688 output.
  Dumps pixel_values + fp32 tower features + bf16 end-to-end outputs per resolution.

Stage "ours" (project venv): loads the dump, runs RadioVisionModel +
NemotronVLModel.extract_feature (unbound, same shim trick) + ProjectorRMSNorm projector on
the SAME pixel tensors and weights, and compares:
  - fp32 vision tower features (structural check, tight tolerance, no bf16 noise)
  - bf16 end-to-end extract_feature output (training-realistic)

Non-square and extreme-aspect resolutions catch h/w transposition bugs in
reshape/pos-embed/pixel-shuffle.

Run from the prime-rl repo root:
  uv run --no-project --with torch --with transformers --with safetensors --with timm \
      --with einops --with open_clip_torch --with torchvision \
      python models/validation/diff_vision_path.py --stage ref --dump /tmp/ref_vision.pt
  uv run --no-sync python models/validation/diff_vision_path.py --stage ours --dump /tmp/ref_vision.pt
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OMNI_DIR = REPO_ROOT / "models" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
GRAFT_ROOT = REPO_ROOT / "models" / "Nemotron-3-Super-VL-graft"

RESOLUTIONS = [(512, 512), (640, 480), (3000, 150)]


def load_omni_vision_state() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    index = json.loads((OMNI_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {index[k] for k in index if k.startswith(("vision_model.", "mlp1."))}
    vision, mlp1 = {}, {}
    for shard in shards:
        with safe_open(OMNI_DIR / shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith("vision_model."):
                    vision[key[len("vision_model.") :]] = f.get_tensor(key)
                elif key.startswith("mlp1."):
                    mlp1[key[len("mlp1.") :]] = f.get_tensor(key)
    return vision, mlp1


def make_pixel_values(width: int, height: int) -> torch.Tensor:
    spec = importlib.util.spec_from_file_location(
        "omni_image_processing", GRAFT_ROOT / "omni_image_processing_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((GRAFT_ROOT / "preprocessor_config.json").read_text())
    proc = getattr(module, cfg["image_processor_type"])(
        **{k: v for k, v in cfg.items() if k not in ("image_processor_type", "auto_map")}
    )
    from PIL import Image

    g = torch.Generator().manual_seed(width * 100_000 + height)
    arr = (torch.rand(height, width, 3, generator=g) * 255).to(torch.uint8).numpy()
    out = proc._preprocess([proc._process_image(Image.fromarray(arr, mode="RGB"))])
    pv = out["pixel_values"]
    return (pv if isinstance(pv, torch.Tensor) else pv[0].unsqueeze(0)).float()


def build_projector(norm_cls, act_cls, mlp1_sd) -> torch.nn.Sequential:
    hidden = mlp1_sd["1.weight"].shape[1]
    proj_hidden, llm_hidden = mlp1_sd["3.weight"].shape[1], mlp1_sd["3.weight"].shape[0]
    mlp1 = torch.nn.Sequential(
        norm_cls(hidden, eps=1e-5),
        torch.nn.Linear(hidden, proj_hidden, bias=False),
        act_cls(),
        torch.nn.Linear(proj_hidden, llm_hidden, bias=False),
    )
    mlp1.load_state_dict(mlp1_sd)
    return mlp1


@torch.no_grad()
def run_ref(dump_path: Path) -> None:
    from transformers import AutoConfig, AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    vision_sd, mlp1_sd = load_omni_vision_state()
    print(f"loaded {len(vision_sd)} vision tensors, mlp1 shapes: {[tuple(t.shape) for t in mlp1_sd.values()]}")

    omni_cfg = AutoConfig.from_pretrained(OMNI_DIR, trust_remote_code=True)
    radio = AutoModel.from_config(omni_cfg.vision_config, trust_remote_code=True)
    missing, unexpected = radio.load_state_dict(vision_sd, strict=False)
    print(f"reference RADIO load: missing={missing} unexpected={unexpected}")
    # summary_idxs is a hub-code buffer absent from the Omni checkpoint; video_embedder is the
    # Omni-only video patch embedder (dropped in our conversion as well).
    assert set(missing) <= {"radio_model.summary_idxs"}, missing
    assert set(unexpected) <= {"radio_model.model.patch_generator.video_embedder.weight"}, unexpected

    omni_cls = get_class_from_dynamic_module("modeling.NemotronH_Nano_Omni_Reasoning_V3", str(OMNI_DIR))
    omni_module = sys.modules[omni_cls.__module__]
    mlp1 = build_projector(omni_module.RMSNorm, omni_module.SquaredReLU, mlp1_sd)

    shim = SimpleNamespace(
        vision_model=radio,
        mlp1=mlp1,
        downsample_ratio=omni_cfg.downsample_ratio,
        ps_version=getattr(omni_cfg, "ps_version", "v2"),
    )
    shim.pixel_shuffle = MethodType(omni_cls.pixel_shuffle, shim)
    print(f"reference: downsample={shim.downsample_ratio} ps_version={shim.ps_version}")

    # Omni's __init__ replaces the RADIO input conditioner with nn.Identity (normalization
    # lives in the image processor); the checkpoint's conditioner buffers (CLIP stats) are
    # dead weight. Mirror that call — it is what our vendored no-op conditioner replicates.
    cond = radio.radio_model.input_conditioner
    print(
        f"conditioner buffers (pre-detach): mean={cond.norm_mean.flatten().tolist()} std={cond.norm_std.flatten().tolist()}"
    )
    radio.radio_model.make_preprocessor_external()

    dump = {"torch_version": torch.__version__}
    for width, height in RESOLUTIONS:
        pv = make_pixel_values(width, height)
        radio.float().eval()
        tower_fp32 = radio(pv).features

        radio.to(torch.bfloat16)
        radio.config.torch_dtype = torch.bfloat16
        mlp1.to(torch.bfloat16).eval()
        e2e_bf16 = MethodType(omni_cls._extract_feature_single, shim)(pv)

        dump[f"{width}x{height}"] = {"pv": pv, "tower_fp32": tower_fp32, "e2e_bf16": e2e_bf16}
        print(
            f"  {width}x{height}: tile {tuple(pv.shape)} -> tower {tuple(tower_fp32.shape)} e2e {tuple(e2e_bf16.shape)}"
        )

    torch.save(dump, dump_path)
    print(f"reference dump written to {dump_path}")


def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, str]:
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).min().item()
    rel = (diff.max() / a.abs().max()).item()
    return (
        rel,
        cos,
        f"max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} scale={a.abs().max():.3e} rel={rel:.3e} min_cos={cos:.6f}",
    )


@torch.no_grad()
def run_ours(dump_path: Path) -> None:
    from prime_rl.trainer.models.nemotron_vl.configuration_nemotron_vl import RadioViTConfig
    from prime_rl.trainer.models.nemotron_vl.modeling_nemotron_vl import (
        NemotronVLModel,
        ProjectorRMSNorm,
        SquaredReLU,
    )
    from prime_rl.trainer.models.nemotron_vl.modeling_radio import RadioVisionModel

    vision_sd, mlp1_sd = load_omni_vision_state()
    vision_sd = {k: v for k, v in vision_sd.items() if "video_embedder" not in k}

    graft_cfg = json.loads((GRAFT_ROOT / "config.json").read_text())
    visual = RadioVisionModel(RadioViTConfig(**graft_cfg["vision_config"]))
    visual.load_state_dict(vision_sd, strict=True)
    mlp1 = build_projector(ProjectorRMSNorm, SquaredReLU, mlp1_sd)

    shim = SimpleNamespace(
        visual=visual,
        mlp1=mlp1,
        config=SimpleNamespace(
            downsample_ratio=graft_cfg["downsample_ratio"],
            vision_config=SimpleNamespace(patch_size=graft_cfg["vision_config"]["patch_size"]),
        ),
    )
    shim.pixel_shuffle = MethodType(NemotronVLModel.pixel_shuffle, shim)
    extract = MethodType(NemotronVLModel.extract_feature, shim)

    dump = torch.load(dump_path, weights_only=False)
    print(f"comparing against reference dump (torch {dump['torch_version']}; ours {torch.__version__})")

    failed = False
    for width, height in RESOLUTIONS:
        ref = dump[f"{width}x{height}"]
        pv = ref["pv"]
        print(f"\n== {width}x{height} tile {tuple(pv.shape)} ==")

        visual.float().eval()
        our_tower = visual(pv)
        assert our_tower.shape == ref["tower_fp32"].shape, (our_tower.shape, ref["tower_fp32"].shape)
        rel, _, line = stats(ref["tower_fp32"], our_tower)
        print(f"  tower fp32: {line}")
        failed |= rel > 1e-4

        visual.to(torch.bfloat16)
        mlp1.to(torch.bfloat16).eval()
        our_e2e = extract(pv)
        assert our_e2e.shape == ref["e2e_bf16"].shape, (our_e2e.shape, ref["e2e_bf16"].shape)
        _, cos, line = stats(ref["e2e_bf16"], our_e2e)
        print(f"  e2e   bf16: {line}")
        failed |= cos < 0.999

    print("\nFAIL: divergence above tolerance (tower fp32 rel > 1e-4 or e2e min_cos < 0.999)" if failed else "\nPASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ref", "ours"], required=True)
    parser.add_argument("--dump", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "ref":
        run_ref(args.dump)
    else:
        run_ours(args.dump)


if __name__ == "__main__":
    main()
