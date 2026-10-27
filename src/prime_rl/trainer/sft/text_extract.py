"""Extract the text-only checkpoints of the TB4 SFT run and push them to Hugging Face.

On this branch the SFT entrypoint runs this job instead of training (a hosted run can only start the
trainer). Global rank 0 alone does the work; there is no model, no GPU work and no collective.

For each checkpoint it runs extract_nemotron_backbone.py (next to this file, unchanged) on the VL export
that the TB4 VL export run (VL_EXPORTS) wrote to the volume, writes the result under the run's directory
([ckpt] output_dir/text-export/<name>), checks every tensor's name, dtype and shape against the text model
the SFT runs trained from, adds a model card, and pushes the folder to a private repo. Failures are logged
and the files stay on the volume.
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

HF_ORG = "PrimeIntellect"
EXTRACT_SCRIPT = Path(__file__).with_name("extract_nemotron_backbone.py")
VL_EXPORTS = Path("/volume/runs/urip6n8rxgsen8fkveitdjtt/hf-export")
# The text model the SFT runs trained from (the release's language model): the reference layout.
TRAINING_BASE = Path("/volume/models/nemotron-3.5-super-text")
BASE_REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
BASE_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"


@dataclass(frozen=True)
class Checkpoint:
    name: str
    repo_name: str
    run_id: str
    step: int
    epochs: float
    val_loss: float


CHECKPOINTS = [
    Checkpoint("tb4-step144", "Nemotron-3.5-Super-tb4-sft-ep3.5", "eivawxtikvt75p1ietxg4lok", 144, 3.5, 0.6056),
    Checkpoint("tb4-step288", "Nemotron-3.5-Super-tb4-sft-ep7", "eivawxtikvt75p1ietxg4lok", 288, 7, 0.6468),
]


def headers(model_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """name -> (dtype, shape) for every tensor of a sharded safetensors checkpoint."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    out: dict[str, tuple[str, list[int]]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(model_dir / shard, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name != "__metadata__":
                out[name] = (info["dtype"], info["shape"])
    if set(out) != set(index["weight_map"]):
        raise ValueError(f"{model_dir}: shard headers and index disagree")
    return out


def check_against_training_base(out_dir: Path) -> int:
    expected, actual = headers(TRAINING_BASE), headers(out_dir)
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"tensor names differ from {TRAINING_BASE}: missing {missing[:5]}, extra {extra[:5]}")
    wrong = [name for name in expected if expected[name] != actual[name]]
    if wrong:
        raise ValueError(f"dtype/shape differ from {TRAINING_BASE}: {[(n, expected[n], actual[n]) for n in wrong[:5]]}")
    return len(actual)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def model_card(ckpt: Checkpoint, num_tensors: int) -> str:
    return f"""---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: transformers
pipeline_tag: text-generation
base_model: {BASE_REPO}
---

# {ckpt.repo_name}

The language model of [{BASE_REPO}](https://huggingface.co/{BASE_REPO}) (revision `{BASE_REVISION[:8]}`)
after {ckpt.epochs:g} epochs of supervised fine-tuning on TB4 (Terminal-Bench-style) agent traces
(mini-swe-agent 2.4.6 harness, distilled from GLM-5.3), trained with prime-rl. Text-only `NemotronHForCausalLM`: this is the checkpoint
that was trained and the one to evaluate.

- Checkpoint: hosted run `{ckpt.run_id}`, step {ckpt.step}; validation loss {ckpt.val_loss} on 44 held-out
  tasks (base model 0.6698).
- Training: 1M-token (1,048,576) packed sequences, AdamW, learning rate warmed up from 1e-6 to 1e-5 over
  4 steps, then cosine decay to 1e-6 at step 288 (7 epochs).
- {num_tensors} tensors, the same names, dtypes and shapes as the release's language model. The
  multi-token-prediction head (`mtp.*`) was not trained and is the release's.
- Produced by `extract_nemotron_backbone.py` from the VL export of the same checkpoint.

Training rendered past assistant turns with their reasoning kept. To match it at inference, pass
`chat_template_kwargs={{"truncate_history_thinking": false}}`.
"""


def push(ckpt: Checkpoint, out_dir: Path) -> None:
    from huggingface_hub import HfApi

    logger = get_logger()
    api = HfApi()
    repo_id = f"{HF_ORG}/{ckpt.repo_name}"
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=out_dir, commit_message=f"Text-only checkpoint ({ckpt.name})")
    files = sorted(p.name for p in out_dir.glob("*.safetensors"))
    with ThreadPoolExecutor(8) as pool:
        local = dict(zip(files, pool.map(lambda name: sha256(out_dir / name), files)))
    remote = {i.path: i.lfs.sha256 for i in api.get_paths_info(repo_id, files, expand=True) if getattr(i, "lfs", None)}
    mismatched = sorted(name for name in files if remote.get(name) != local[name])
    if mismatched:
        raise ValueError(f"{repo_id}: {len(mismatched)} shards missing or different: {mismatched[:5]}")
    logger.success(f"[{ckpt.name}] {repo_id}: all {len(files)} shards match the local sha256")


def run(config: SFTConfig) -> None:
    logger = get_logger()
    if get_world().rank != 0:
        return
    if config.ckpt is None or config.ckpt.output_dir is None:
        raise ValueError("The job writes under [ckpt] output_dir (the run's directory on the volume), which is unset")
    out_root = config.ckpt.output_dir / "text-export"
    can_push = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not can_push:
        logger.error("No HF_TOKEN in the environment: the checkpoints stay on the volume only")
    for ckpt in CHECKPOINTS:
        src, out_dir = VL_EXPORTS / ckpt.name, out_root / ckpt.name
        logger.info(f"[{ckpt.name}] extracting {src} -> {out_dir}")
        try:
            subprocess.run([sys.executable, "-I", str(EXTRACT_SCRIPT), str(src), str(out_dir)], check=True)
            num_tensors = check_against_training_base(out_dir)
            logger.success(f"[{ckpt.name}] {num_tensors} tensors match {TRAINING_BASE} in name, dtype and shape")
            (out_dir / "README.md").write_text(model_card(ckpt, num_tensors))
            if can_push:
                push(ckpt, out_dir)
        except Exception as error:
            logger.error(f"[{ckpt.name}] FAILED: {error!r}")
    logger.success(f"Text extraction finished under {out_root}")
