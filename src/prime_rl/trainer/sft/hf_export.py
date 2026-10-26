"""Export the swe-marathon SFT checkpoints of Nemotron 3.5 Super into the release's VL Hugging Face layout.

On this branch the SFT entrypoint runs this export instead of training, because a hosted run can only
start the trainer. For each checkpoint, the export:
1. loads the model weights (not the optimizer) from the DCP checkpoint;
2. converts them to HF names;
3. rewrites the release's 64 safetensors files with the fine-tuned language-model tensors, cast to the
   release's dtypes. Every other tensor is kept byte-for-byte: the vision tower, the projector, and the
   MTP head, which SFT never trained.

Each rank rewrites the files assigned to it. The files go under the run's directory on the volume
(``[ckpt] output_dir``). They are checked against the release (keys, shapes, dtypes, how far each tensor
moved) and then pushed to private Hugging Face repos.
"""

import hashlib
import json
import os
import shutil
import socket
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.tensor import DTensor

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.utils import get_dcp_group
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.weights import resolve_fqn, resolve_wire_dtype

HF_ORG = "PrimeIntellect"

BASE_REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
BASE_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
# The release as downloaded at revision 1f8953f7. Its safetensors files are byte-identical to BASE_REVISION's
# (RELEASE_SHA256 is checked against every file read); only three code/processor files changed since.
BASE_DIR = Path("/volume/models/omni-src")
# Non-weight files of the release, fetched at BASE_REVISION. README.md is replaced by a model card.
RELEASE_FILES = [
    "__init__.py",
    "bias.md",
    "chat_template.jinja",
    "config.json",
    "configuration_nemotron_h_omni.py",
    "configuration_radio.py",
    "explainability.md",
    "generation_config.json",
    "image_processing_nemotron_h_omni.py",
    "model.safetensors.index.json",
    "modeling_nemotron_h_omni.py",
    "modeling_radio.py",
    "preprocessor_config.json",
    "privacy.md",
    "processing_nemotron_h_omni.py",
    "processor_config.json",
    "safety.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "ultra_v3_reasoning_parser.py",
]
LM_PREFIX = "language_model."
MTP_PREFIX = "language_model.mtp."
# A fine-tuned tensor this far (relative L2) from the release means a wrong key mapping, not training.
MAX_RELATIVE_CHANGE = 0.2
GLOO_TIMEOUT = timedelta(hours=12)

# sha256 of every safetensors file of BASE_REPO at BASE_REVISION (the hub's LFS object ids).
RELEASE_SHA256 = {
    "model-00001-of-00063.safetensors": "9e414e02442b2564263b2d9458ac3573da5cec63db0f7398cecca647bedc6476",
    "model-00002-of-00063.safetensors": "75f5474a48fec78b855df47c5c71e78647414d65a095e51fe7f5ce2367153e5b",
    "model-00003-of-00063.safetensors": "0637d9d923a39bd0304e7b42ccfa666b65825b0d4379b13b8f09827b091f1950",
    "model-00004-of-00063.safetensors": "9613d25fb2476c5fa03f324e7b9eefddc3dfe6eb72cfbcda46d023596b0b9c4d",
    "model-00005-of-00063.safetensors": "b46aeda71a3b4831fb5f37784844ad18f31037b57e0ebcef6f40e0ce33a95aab",
    "model-00006-of-00063.safetensors": "e3df88c895542a107eb2e74041edde5eb8227f9e941fb1b0056c5e7ba8979d20",
    "model-00007-of-00063.safetensors": "7bf0aecf7e14a2aee6d4217d1177443511568541ec0ba032745f80393ea3afbe",
    "model-00008-of-00063.safetensors": "cbda59736f89e335f28ff42d700b986e72a8363a9ee2d887f78f03918bc57d36",
    "model-00009-of-00063.safetensors": "24c0c91e19450cab83acb4fce7c50c9d74f8439b1011bab60f4c0b3cde08b7bc",
    "model-00010-of-00063.safetensors": "b7f7d05e38d14d7f93e8e98883b66af95eea64bdcd44d14cab360bb8f7e4cd60",
    "model-00011-of-00063.safetensors": "97252aba070b0dd910b46f2fbeeb15dbdf3d87209bf8a58c4f80ecab4b817f56",
    "model-00012-of-00063.safetensors": "afad5bbaef982a1ae3d9848293ae3373f43aa266ca549bf518333e26eb5526fd",
    "model-00013-of-00063.safetensors": "3e400dc5d16f1ddbe4d91d9d6a003c3b781fb76173f54bde03a004e282bd74a3",
    "model-00014-of-00063.safetensors": "d877e906b45fe2e89a6676741412430fb01c553fac94510fc56f082406107044",
    "model-00015-of-00063.safetensors": "eb332a8471edbf884b59260d8dcbdcf15b38c475c32bf0e3b6760bd0b1141487",
    "model-00016-of-00063.safetensors": "5e860db2d2c947375445f99382952951430f876afd4b0ee62424f5183ff0f586",
    "model-00017-of-00063.safetensors": "e562f1d2d0e0e1820b8da731db7c815a110481a830c34dcc53dbc67d7f3bc947",
    "model-00018-of-00063.safetensors": "350c734db75b445d2928c8e9665610cf35fb788d7678bb9f5b6102cfe73be667",
    "model-00019-of-00063.safetensors": "9909d2a3029aac18424f00a317cb943492b9e69cbbc3b007d59d0d31ec1ecde6",
    "model-00020-of-00063.safetensors": "20f063e35f7490896fbdfb53358b315fe7e3fe5f2f8f511f22287d81e14acbeb",
    "model-00021-of-00063.safetensors": "9d987fad5424a053a10ec3a83cbdd1949454a25580dfd9c871a38fba884aba52",
    "model-00022-of-00063.safetensors": "56d0de2a7d7c8608318d98a2794afdcab45b3593cb9e887277bf037b4327dc6b",
    "model-00023-of-00063.safetensors": "0fa6c3687fd11e8417c1cdfe1ce6ec518e9bc827e405b6211e2518e9c0849d11",
    "model-00024-of-00063.safetensors": "0a755baa853e565a8b1a0c5223cb04c5ebce4d71389ba9e483fa6a52c8e8a489",
    "model-00025-of-00063.safetensors": "9c57dd168025be808092bb92360647a932919ed17624fc253399241f427d9f60",
    "model-00026-of-00063.safetensors": "06bc8ce5e7f4e79b5d9f729e7fd98454dfedc915aa7214dfde96b69806b34082",
    "model-00027-of-00063.safetensors": "cd10a292f9dc435f70a2e868f70c1b0d5e7a16b784f03a9a7534e663983a00ba",
    "model-00028-of-00063.safetensors": "8b2d17cbf64a49f133cd649103430c9bcc888c617d5b4ba1c572cff6b54daae3",
    "model-00029-of-00063.safetensors": "143218156b4f88f0ee9cc810fc1f4ac7b4fefa435b7672610e9b54f0119e467f",
    "model-00030-of-00063.safetensors": "08acadc9137ae77874ed851f631379b2a8c2a97a4a1a0865ed96737b1de6cfb6",
    "model-00031-of-00063.safetensors": "e97bbb1692b572ea708005ffb46d10baa225b0af8b72c1356848b346f456582b",
    "model-00032-of-00063.safetensors": "95439dc712c347d839caaba25e6c027b640706c6277e57246bb6f8110d301ce4",
    "model-00033-of-00063.safetensors": "4a06850dc8edac8474a286ba03a62b777f4c7f1e1c44fc7f9badd40d8d90171d",
    "model-00034-of-00063.safetensors": "cbd4c9ff97fe26264d6a1c44c240b3f68e90faa537fde0149e6c7b07a7a058d0",
    "model-00035-of-00063.safetensors": "58708f7a7af9e55f1267f4ff08a68b9191f07df817a7b4399fe69c0f0a32f46c",
    "model-00036-of-00063.safetensors": "fa9bbbeb1ff56374f13d54c78e63e3cf8f76c4b34fc3cc730e65a26a54841d84",
    "model-00037-of-00063.safetensors": "d592cc8fca5428d317fc4d3a905986c9fbe78291cd2230b749693badf1c9c985",
    "model-00038-of-00063.safetensors": "38390d86e22b76a0699c21ffccebb0d65db2b46b66a18ac271c0f46b2a1adb42",
    "model-00039-of-00063.safetensors": "d46ed93cd2606156df4039584d31080cf855ca20f1f98f1dadc02ed9677cc522",
    "model-00040-of-00063.safetensors": "fc10d57dc17509dae80b0e7f7a48e92d538c5d13057403f51963a09bd2ea7548",
    "model-00041-of-00063.safetensors": "fa31548d31978868a0ad3c796a6ed0931260da53228e9cb95262570a90ad38a7",
    "model-00042-of-00063.safetensors": "0586b248623c0520b04dd1710be5ef2405cfdc382217062b984883151b21f831",
    "model-00043-of-00063.safetensors": "bc1cd809c380ac0573e5b1b4db3c50ca7d42bad640fcea38e9938f3aad522762",
    "model-00044-of-00063.safetensors": "3ce17aa3cb8fe63f3995c56e11824cbba6f4c81c64f53d7218969a0be577cd6a",
    "model-00045-of-00063.safetensors": "7e73845b5575bec5157ad220646e31471d41f7f9e5f462cd8d7a722d268a3c57",
    "model-00046-of-00063.safetensors": "6d23de2e2b6b27c97204fd57736700fca59b7214f56470da276c8a243f2505ed",
    "model-00047-of-00063.safetensors": "05ef2d60c2e2e0dc90c1f4cb5f6aa9a85e42d41e556be5dadc366cc4896e7f4b",
    "model-00048-of-00063.safetensors": "e247651c2cd58c77a61705b15a97a6d4cadfcc24e3608531b20678aa03dd70ad",
    "model-00049-of-00063.safetensors": "70499df23ed928f648bf29b4036b2e1b771df2cd7ca22e61005a9d3e1d30e7ac",
    "model-00050-of-00063.safetensors": "98438893f58367ae0be9d0a16eff6931911d25c32fbd621e8e56972fb6067df3",
    "model-00051-of-00063.safetensors": "3aa496901e9f0a8d9d51ba5ac325062e6179a5ed66250431600e1835e0dcdc74",
    "model-00052-of-00063.safetensors": "875a1bcbadf27ee2ba27b8f6e01124d5ad96c254f64cb562da305f1b3061118d",
    "model-00053-of-00063.safetensors": "81624cee5848beb165d7bf607ba1eb070237e6389b00c430a0887f0a7b64e502",
    "model-00054-of-00063.safetensors": "a546a0b998b16a4e4514da4e890704f5925e7c82f962fbdff86861ed1873ce0d",
    "model-00055-of-00063.safetensors": "9af8efb9ff76e67ed5b47c08890ab2773dc86ad325c3132efb651491baa77171",
    "model-00056-of-00063.safetensors": "20d7938ea6aa169d93727f89bb699345c77296ba1ba429e7f883b6c685158051",
    "model-00057-of-00063.safetensors": "f00a655a4c7566857aa9715ce17a645311c5a45037e14b9d2160cdca37953636",
    "model-00058-of-00063.safetensors": "6e4140669f095da4297ae1c27ac06dd2f31799a26571962f8d988551fc6f09e2",
    "model-00059-of-00063.safetensors": "960e17039b0a42537ee2f6607b0262bf2e9b5a8ed8d8e2f1f54069987a27f4d3",
    "model-00060-of-00063.safetensors": "20fa189576d2f51750fc65c1bcf63fb7533294df23a402bd0564b78d56471b4e",
    "model-00061-of-00063.safetensors": "1fc55c6a5b6ed9747e30a0ac694bb4fcbe692d7def09d532e77aead1b9bb7f0e",
    "model-00062-of-00063.safetensors": "b1ea8a4a9774ce830ba96fe9867d800a34c5271e2dbbe2c4e871cd90d3122018",
    "model-00063-of-00063.safetensors": "f9d59883028fdeb7aaf2957c3751652dc9ba84517706698ff23c71313cd9cc9c",
    "model-summary-idxs.safetensors": "ea20914783af0cab7dba87c31dd7bae23568c3ccb96e6c1f7aad8f8c3276ff51",
}


@dataclass(frozen=True)
class Export:
    name: str
    dcp_dir: Path
    repo_name: str
    run_id: str
    step: int
    epochs: int
    val_loss: float


EXPORTS = [
    Export(
        name="ep2-step104",
        dcp_dir=Path("/volume/runs/hnp5pwsw3fx9p5b7r6bjmnnx/checkpoints/step_104/trainer"),
        repo_name="Nemotron-3.5-Super-VL-swe-marathon-sft-ep2",
        run_id="hnp5pwsw3fx9p5b7r6bjmnnx",
        step=104,
        epochs=2,
        val_loss=0.4274,
    ),
    Export(
        name="ep3-step156",
        dcp_dir=Path("/volume/runs/siuskw4we4txi09583x5x807/checkpoints/step_156/trainer"),
        repo_name="Nemotron-3.5-Super-VL-swe-marathon-sft-ep3",
        run_id="siuskw4we4txi09583x5x807",
        step=156,
        epochs=3,
        val_loss=0.4256,
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def assign_files(sizes: dict[str, int], num_ranks: int) -> list[list[str]]:
    """Spread files over ranks by size, largest first onto the least-loaded rank (deterministic)."""
    loads = [0] * num_ranks
    assigned: list[list[str]] = [[] for _ in range(num_ranks)]
    for name in sorted(sizes, key=lambda name: (-sizes[name], name)):
        rank = min(range(num_ranks), key=lambda r: (loads[r], r))
        assigned[rank].append(name)
        loads[rank] += sizes[name]
    return assigned


def make_dir(path: Path, group) -> None:
    """Create ``path`` on every node (the volume may be staged per node): master first, then one rank per
    node if it does not see it yet, since concurrent mkdirs of one path can fail on a parallel FS."""
    world = get_world()
    if world.is_master:
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=group)
    if world.local_rank == 0 and not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=group)


def is_fine_tuned(key: str) -> bool:
    return key.startswith(LM_PREFIX) and not key.startswith(MTP_PREFIX)


def collect_hf_tensors(model: torch.nn.Module, wanted: set[str]) -> dict[str, Tensor]:
    """Gather every weight (a collective, so all ranks walk all keys) and keep the HF tensors in ``wanted`` on CPU."""
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    kept: dict[str, Tensor] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value = value.to(resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)).full_tensor()
        for hf_key, tensor in model.convert_to_hf({resolve_fqn(model, key): value}).items():
            if hf_key in wanted:
                kept[hf_key] = tensor.to("cpu", copy=True).contiguous()
    return kept


def write_file(src: Path, dst: Path, fine_tuned: dict[str, Tensor]) -> dict[str, float]:
    """Write ``src`` to ``dst`` with its fine-tuned keys swapped in (cast to the source dtype).

    Consumes the swapped-in entries of ``fine_tuned`` and returns each one's relative L2 change.
    """
    tensors: dict[str, Tensor] = {}
    changes: dict[str, float] = {}
    with safe_open(src, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            base = f.get_tensor(key)
            if not is_fine_tuned(key):
                tensors[key] = base
                continue
            new = fine_tuned.pop(key[len(LM_PREFIX) :])
            if not torch.isfinite(new).all():
                raise ValueError(f"{key}: non-finite values")
            if new.shape != base.shape:
                raise ValueError(f"{key}: fine-tuned shape {tuple(new.shape)} != release shape {tuple(base.shape)}")
            new = new.to(base.dtype)
            base_norm = base.float().norm().item()
            delta_norm = (new.float() - base.float()).norm().item()
            changes[key] = delta_norm / base_norm if base_norm > 0 else delta_norm
            tensors[key] = new
    save_file(tensors, dst, metadata=metadata)
    return changes


def check_layout(src: Path, dst: Path) -> None:
    """``dst`` holds exactly ``src``'s keys with the same dtypes and shapes."""
    with safe_open(src, framework="pt") as a, safe_open(dst, framework="pt") as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError(f"{dst.name}: key set differs from the release")
        for key in a.keys():
            sa, sb = a.get_slice(key), b.get_slice(key)
            if (sa.get_dtype(), sa.get_shape()) != (sb.get_dtype(), sb.get_shape()):
                raise ValueError(
                    f"{dst.name}:{key}: {sb.get_dtype()} {sb.get_shape()} != {sa.get_dtype()} {sa.get_shape()}"
                )


def summarize_changes(changes: dict[str, float]) -> dict:
    groups: dict[str, list[float]] = {}
    for key, change in changes.items():
        if ".mixer.experts." in key:
            group = "routed_experts"
        elif ".mixer.shared_experts." in key:
            group = "shared_experts"
        elif ".mixer.gate." in key:
            group = "router"
        elif "latent_proj" in key:
            group = "latent_proj"
        elif any(f".mixer.{name}" in key for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            group = "attention"
        elif ".mixer." in key:
            group = "mamba"
        else:
            group = key[len(LM_PREFIX) :] if "layers." not in key else "layer_norms"
        groups.setdefault(group, []).append(change)

    def quantiles(values: list[float]) -> dict:
        values = sorted(values)
        pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
        return {
            "count": len(values),
            "zero": sum(v == 0 for v in values),
            "min": values[0],
            "p50": pick(0.5),
            "p99": pick(0.99),
            "max": values[-1],
        }

    largest = sorted(changes.items(), key=lambda item: -item[1])[:20]
    return {
        "all": quantiles(list(changes.values())),
        "groups": {g: quantiles(v) for g, v in sorted(groups.items())},
        "largest": largest,
    }


def model_card(export: Export) -> str:
    return f"""---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: transformers
pipeline_tag: image-text-to-text
base_model: {BASE_REPO}
---

# {export.repo_name}

[{BASE_REPO}](https://huggingface.co/{BASE_REPO}) (revision `{BASE_REVISION[:8]}`) after
{export.epochs} epochs of supervised fine-tuning of its language model on swe-marathon agent traces
(OpenCode harness, distilled from GLM-5.3), trained with prime-rl.

- Checkpoint: hosted run `{export.run_id}`, step {export.step}; validation loss {export.val_loss}
  on 50 held-out tasks (base model 0.5086).
- Training: 640K-token packed sequences, AdamW, peak learning rate 1e-5 with cosine decay to 1e-6
  (epoch 3: re-warmed to 3e-6).
- Fine-tuned: the language model (`language_model.backbone.*`, `language_model.lm_head.weight`).
- Unchanged from the release: the vision tower and projector (text-only training), and the
  multi-token-prediction head (`language_model.mtp.*`), which was not trained and is copied as is.
  MTP speculative decoding therefore drafts with the base model's head.
- Same files, shard layout, dtypes, config, code and tokenizer as the release; load it the same way.

Training rendered past assistant turns with their reasoning kept. To match it at inference, pass
`chat_template_kwargs={{"truncate_history_thinking": false}}`.

`export_report.json` records, for every fine-tuned tensor, its relative change from the release.
"""


def fetch_release_files(dest: Path) -> str:
    """Release files at BASE_REVISION; falls back to BASE_DIR's copies (revision 1f8953f7) without network."""
    logger = get_logger()
    try:
        from huggingface_hub import hf_hub_download

        for name in RELEASE_FILES:
            hf_hub_download(BASE_REPO, name, revision=BASE_REVISION, local_dir=dest)
        return BASE_REVISION
    except Exception as error:
        logger.warning(f"Could not fetch release files at {BASE_REVISION[:8]} ({error!r}); using {BASE_DIR}")
        for name in RELEASE_FILES:
            shutil.copyfile(BASE_DIR / name, dest / name)
        return "1f8953f7 (offline fallback)"


def upload(export: Export, out_dir: Path, files_by_host: dict[str, list[str]], group) -> None:
    """Push an export to its private repo: release files first, then one commit per node for its shards.

    Uploads run inside try blocks on purpose: a failed push must not fail the run, because the files on the
    volume are only kept when the run ends cleanly, and they can be pushed again from there.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    logger = get_logger()
    world = get_world()
    api = HfApi()
    host = socket.gethostname()
    host_order = sorted(files_by_host)
    is_host_leader = world.local_rank == 0

    repo_id = f"{HF_ORG}/{export.repo_name}"
    ok = [True]
    if world.is_master:
        try:
            api.create_repo(repo_id, private=True, exist_ok=True)
            small = [p.name for p in sorted(out_dir.iterdir()) if not p.name.endswith(".safetensors")]
            api.create_commit(
                repo_id,
                operations=[CommitOperationAdd(name, out_dir / name) for name in small],
                commit_message=f"Model card, config, code and tokenizer ({export.name})",
            )
        except Exception as error:
            logger.error(f"[{export.name}] repo setup on {repo_id} failed: {error!r}")
            ok = [False]
    dist.broadcast_object_list(ok, src=0, group=group)
    if not ok[0]:
        return

    # Every node pushes its own shards' bytes concurrently; the commits that reference them go one by one.
    operations: list = []
    error_text = None
    if is_host_leader:
        try:
            operations = [CommitOperationAdd(name, out_dir / name) for name in files_by_host[host]]
            api.preupload_lfs_files(repo_id, additions=operations)
        except Exception as error:
            error_text = repr(error)
            logger.error(f"[{export.name}] upload from {host} failed: {error_text}")
    for turn in host_order:
        if is_host_leader and turn == host and error_text is None:
            try:
                api.create_commit(
                    repo_id,
                    operations=operations,
                    commit_message=f"Weights {len(operations)} shards ({export.name}, {host})",
                )
            except Exception as error:
                error_text = repr(error)
                logger.error(f"[{export.name}] commit from {host} failed: {error_text}")
        dist.barrier(group=group)

    if world.is_master:
        try:
            expected = json.loads((out_dir / "export_report.json").read_text())["sha256"]
            remote = {
                info.path: info.lfs.sha256
                for info in api.get_paths_info(repo_id, list(expected), expand=True)
                if getattr(info, "lfs", None) is not None
            }
            missing = sorted(name for name, digest in expected.items() if remote.get(name) != digest)
            if missing:
                logger.error(f"[{export.name}] {len(missing)} shards missing or different on {repo_id}: {missing[:5]}")
            else:
                logger.success(f"[{export.name}] all {len(expected)} shards on {repo_id} match the local sha256")
        except Exception as error:
            logger.error(f"[{export.name}] could not verify {repo_id}: {error!r}")
    dist.barrier(group=group)


def run(config: SFTConfig, parallel_dims: ParallelDims) -> None:
    logger = get_logger()
    world = get_world()
    if not HF_ORG:
        raise ValueError("hf_export.HF_ORG is not set")
    if config.ckpt is None or config.ckpt.output_dir is None:
        raise ValueError(
            "The export writes under [ckpt] output_dir (the run's directory on the volume), which is unset"
        )
    out_root = config.ckpt.output_dir / "hf-export"
    group = dist.new_group(backend="gloo", timeout=GLOO_TIMEOUT)

    missing = [str(export.dcp_dir) for export in EXPORTS if not (export.dcp_dir / ".metadata").is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoints not found: {missing}")
    # Every rank must take the same upload branch (it issues collectives), so the master decides.
    do_upload = [bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))]
    dist.broadcast_object_list(do_upload, src=0, group=group)
    if not do_upload[0]:
        logger.error("No HF_TOKEN in the environment: the exports stay on the volume only")

    if world.is_master:
        (out_root / "release").mkdir(parents=True, exist_ok=True)
        release_revision = fetch_release_files(out_root / "release")
        logger.info(f"Release files from revision {release_revision}")
    dist.barrier(group=group)

    weight_map: dict[str, str] = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(weight_map.values()))
    if set(files) != set(RELEASE_SHA256):
        raise ValueError(f"{BASE_DIR} index lists other files than the release")
    assignment = assign_files({name: (BASE_DIR / name).stat().st_size for name in files}, world.world_size)
    my_files = assignment[world.rank]
    for name in my_files:
        if sha256_file(BASE_DIR / name) != RELEASE_SHA256[name]:
            raise ValueError(f"{BASE_DIR / name} differs from {BASE_REPO}@{BASE_REVISION[:8]}")
    wanted = {key[len(LM_PREFIX) :] for key, name in weight_map.items() if name in my_files and is_fine_tuned(key)}
    num_fine_tuned = sum(is_fine_tuned(key) for key in weight_map)
    logger.info(
        f"Rank {world.rank}: files {my_files} verified against the release; {len(wanted)} fine-tuned keys to fill"
    )

    hosts: list[str | None] = [None] * world.world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    files_by_host: dict[str, list[str]] = {}
    for rank, names in enumerate(assignment):
        files_by_host.setdefault(hosts[rank], []).extend(names)

    logger.info(f"Building the model ({config.model.name}) without loading HF weights")
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=True)

    exports_written: list[str] = []
    for export in EXPORTS:
        out_dir = out_root / export.name
        logger.info(f"[{export.name}] loading model weights from {export.dcp_dir}")
        dcp_load(
            state_dict={"app": AppState(model, [], None, None)},
            checkpoint_id=export.dcp_dir,
            process_group=get_dcp_group(),
        )

        logger.info(f"[{export.name}] gathering and converting to HF names")
        fine_tuned = collect_hf_tensors(model, wanted)
        if set(fine_tuned) != wanted:
            raise ValueError(
                f"[{export.name}] rank {world.rank}: missing converted keys {sorted(wanted - set(fine_tuned))[:5]}"
            )
        torch.cuda.empty_cache()

        make_dir(out_dir, group)
        changes: dict[str, float] = {}
        digests: dict[str, str] = {}
        for name in my_files:
            changes.update(write_file(BASE_DIR / name, out_dir / name, fine_tuned))
            check_layout(BASE_DIR / name, out_dir / name)
            digests[name] = sha256_file(out_dir / name)
        if fine_tuned:
            raise ValueError(
                f"[{export.name}] rank {world.rank}: converted keys not in the release: {sorted(fine_tuned)[:5]}"
            )
        logger.info(
            f"[{export.name}] rank {world.rank}: wrote {len(my_files)} files ({len(changes)} fine-tuned tensors)"
        )

        gathered: list = [None] * world.world_size
        dist.all_gather_object(gathered, (changes, digests), group=group)
        ok = [True]
        if world.is_master:
            all_changes = {key: value for rank_changes, _ in gathered for key, value in rank_changes.items()}
            all_digests = {name: digest for _, rank_digests in gathered for name, digest in rank_digests.items()}
            summary = summarize_changes(all_changes)
            report = {
                "export": export.name,
                "source": {"run_id": export.run_id, "step": export.step, "dcp_dir": str(export.dcp_dir)},
                "base": {"repo": BASE_REPO, "revision": BASE_REVISION, "release_files_revision": release_revision},
                "fine_tuned_tensors": len(all_changes),
                "expected_fine_tuned_tensors": num_fine_tuned,
                "relative_change": summary,
                "relative_change_per_tensor": dict(sorted(all_changes.items())),
                "sha256": all_digests,
            }
            for name in RELEASE_FILES:
                shutil.copyfile(out_root / "release" / name, out_dir / name)
            (out_dir / "README.md").write_text(model_card(export))
            (out_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
            all_stats = summary["all"]
            problems = []
            if len(all_changes) != num_fine_tuned or len(all_digests) != len(files):
                problems.append(f"{len(all_changes)}/{num_fine_tuned} tensors, {len(all_digests)}/{len(files)} files")
            if all_stats["max"] > MAX_RELATIVE_CHANGE:
                problems.append(f"largest relative change {summary['largest'][0]} > {MAX_RELATIVE_CHANGE}")
            if all_stats["zero"] > all_stats["count"] // 2:
                problems.append(
                    f"{all_stats['zero']}/{all_stats['count']} tensors equal the release (checkpoint not loaded?)"
                )
            logger.info(f"[{export.name}] relative change from the release: {json.dumps(summary['groups'])}")
            if problems:
                logger.error(f"[{export.name}] export check FAILED, not uploading: {problems}")
                ok = [False]
            else:
                logger.success(
                    f"[{export.name}] {len(all_changes)} fine-tuned tensors in {len(files)} files written to {out_dir}"
                )
        dist.broadcast_object_list(ok, src=0, group=group)
        if ok[0]:
            exports_written.append(export.name)
            # Push before loading the next checkpoint, so a later failure cannot cost this export its upload.
            if do_upload[0]:
                upload(export, out_dir, files_by_host, group)

    dist.barrier(group=group)
    logger.success(f"HF export finished: {exports_written} under {out_root}")
