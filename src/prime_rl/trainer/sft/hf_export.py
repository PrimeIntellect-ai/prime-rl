"""End-of-run Hugging Face export of an SFT run's checkpoints (``[hf_export]``).

Runs after the final step and the final checkpoint, before the trainer exits. ``last`` is exported from the weights
in memory; ``best`` is loaded from the ``[val] keep_best`` snapshot first (unless it is the final step). Each export
rewrites the safetensors files of ``base_dir`` (the HF checkpoint the run started from, e.g. the text-only
``NemotronHForCausalLM`` extract of Nemotron 3.5 Super):

1. every rank gathers the weights (a collective) and keeps, on CPU, the HF tensors of the base files assigned to it;
2. it rewrites those files with the fine-tuned tensors cast to the base file's dtypes. Tensors the model does not
   train (``keep_base_prefixes``, e.g. the MTP head, which prime-rl's NemotronH drops) are copied byte-for-byte, and
   so are the file names, the shard layout, the index, config and tokenizer;
3. it checks its files against the base (key sets, dtypes, shapes) and reports how far each tensor moved;
4. global rank 0 checks the union against every base header, writes ``export_report.json`` and a model card, and,
   with a repo and ``HF_TOKEN``, the files are pushed: small files by rank 0, the shards by one rank per node
   (preupload, then one commit per node in turn), then the remote sha256 is checked against the report.

Every rank writes only its own files and every node uploads only what its ranks wrote, so nothing depends on
files another node wrote during the run (hosted runs stage the volume per node and sync it after the run).
Failures are logged and never raised: the run must still end cleanly so that its checkpoints are kept.

Adapted from the one-off export jobs of exp/nemotron-sft-hf-export (19cabb8ae, TEXT_LAYOUT) and
exp/nemotron-sft-text-extract (039ecaaaf, check_against_training_base).
"""

import hashlib
import json
import os
import shutil
import socket
import struct
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

EXPORT_DIR_NAME = "hf-export"
# Non-weight files of the base checkpoint carried over (those that exist). README.md is replaced by a model card.
BASE_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
REQUIRED_BASE_FILES = ["config.json", "model.safetensors.index.json"]
GLOO_TIMEOUT = timedelta(hours=12)


# File logic (no collectives; unit-tested on CPU)


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


def is_fine_tuned(key: str, keep_base_prefixes: tuple[str, ...]) -> bool:
    return not key.startswith(keep_base_prefixes)


def write_file(
    src: Path, dst: Path, fine_tuned: dict[str, Tensor], keep_base_prefixes: tuple[str, ...]
) -> dict[str, float]:
    """Write ``src`` to ``dst`` with its fine-tuned keys swapped in (cast to the source dtype).

    Consumes the swapped-in entries of ``fine_tuned`` and returns each one's relative L2 change.
    """
    tensors: dict[str, Tensor] = {}
    changes: dict[str, float] = {}
    with safe_open(src, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            base = f.get_tensor(key)
            if not is_fine_tuned(key, keep_base_prefixes):
                tensors[key] = base
                continue
            if key not in fine_tuned:
                raise ValueError(f"{key}: the model produced no tensor for it")
            new = fine_tuned.pop(key)
            if not torch.isfinite(new).all():
                raise ValueError(f"{key}: non-finite values")
            if new.shape != base.shape:
                raise ValueError(f"{key}: fine-tuned shape {tuple(new.shape)} != base shape {tuple(base.shape)}")
            new = new.to(base.dtype)
            base_norm = base.float().norm().item()
            delta_norm = (new.float() - base.float()).norm().item()
            changes[key] = delta_norm / base_norm if base_norm > 0 else delta_norm
            tensors[key] = new
    tmp = dst.with_name(dst.name + ".tmp")
    save_file(tensors, tmp, metadata=metadata)
    os.replace(tmp, dst)
    return changes


def read_header(path: Path) -> dict[str, tuple[str, list[int]]]:
    """name -> (dtype, shape) of every tensor in a safetensors file, from its header only."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return {name: (info["dtype"], info["shape"]) for name, info in header.items() if name != "__metadata__"}


def check_layout(src: Path, dst: Path) -> None:
    """``dst`` holds exactly ``src``'s keys with the same dtypes and shapes."""
    a, b = read_header(src), read_header(dst)
    if set(a) != set(b):
        raise ValueError(f"{dst.name}: key set differs from the base")
    wrong = [key for key in a if a[key] != b[key]]
    if wrong:
        raise ValueError(f"{dst.name}: {[(k, b[k], a[k]) for k in wrong[:5]]} (written, base)")


def check_against_base(base_dir: Path, written: dict[str, dict[str, tuple[str, list[int]]]]) -> int:
    """Every tensor of the base checkpoint is in the written files, in the same file, with its dtype and shape.

    ``written`` maps file name -> its header as read back by the rank that wrote it."""
    weight_map: dict[str, str] = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    actual = {key: (name, info) for name, header in written.items() for key, info in header.items()}
    missing, extra = sorted(set(weight_map) - set(actual)), sorted(set(actual) - set(weight_map))
    if missing or extra:
        raise ValueError(f"tensor names differ from {base_dir}: missing {missing[:5]}, extra {extra[:5]}")
    misplaced = [key for key, name in weight_map.items() if actual[key][0] != name]
    if misplaced:
        raise ValueError(f"tensors in other files than in {base_dir}: {misplaced[:5]}")
    expected: dict[str, tuple[str, list[int]]] = {}
    for name in sorted(set(weight_map.values())):
        expected.update(read_header(base_dir / name))
    wrong = [key for key in expected if expected[key] != actual[key][1]]
    if wrong:
        raise ValueError(f"dtype/shape differ from {base_dir}: {[(k, expected[k], actual[k][1]) for k in wrong[:5]]}")
    return len(actual)


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
            group = key if "layers." not in key else "layer_norms"
        groups.setdefault(group, []).append(change)

    def quantiles(values: list[float]) -> dict:
        values = sorted(values)
        pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
        return {
            "count": len(values),
            "zero": sum(v == 0 for v in values),
            "min": values[0],
            "p50": pick(0.5),
            "p99": pick(0.99),
            "max": values[-1],
        }

    if not changes:
        return {
            "all": {"count": 0, "zero": 0, "min": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0},
            "groups": {},
            "largest": [],
        }
    largest = sorted(changes.items(), key=lambda item: -item[1])[:20]
    return {
        "all": quantiles(list(changes.values())),
        "groups": {g: quantiles(v) for g, v in sorted(groups.items())},
        "largest": largest,
    }


def find_problems(
    summary: dict, num_fine_tuned: int, num_files: int, num_digests: int, max_relative_change: float
) -> list[str]:
    stats = summary["all"]
    problems = []
    if stats["count"] != num_fine_tuned or num_digests != num_files:
        problems.append(f"{stats['count']}/{num_fine_tuned} tensors, {num_digests}/{num_files} files")
    if stats["max"] > max_relative_change:
        problems.append(f"largest relative change {summary['largest'][0]} > {max_relative_change}")
    if stats["zero"] > stats["count"] // 2:
        problems.append(f"{stats['zero']}/{stats['count']} tensors equal the base (weights not trained or not loaded?)")
    return problems


@dataclass
class ExportInfo:
    name: str  # "best" or "last"
    repo_id: str | None
    step: int
    max_steps: int | None
    val_loss: float | None
    run_id: str
    extra: dict = field(default_factory=dict)


def model_card(info: ExportInfo, base_dir: Path, base_config: dict, description: str, num_tensors: int) -> str:
    architectures = ", ".join(base_config.get("architectures", [])) or "the base architecture"
    val = f"{info.val_loss:.4f}" if info.val_loss is not None else "not measured"
    which = "the lowest validation loss of the run" if info.name == "best" else "the final weights of the run"
    title = info.repo_id.split("/")[-1] if info.repo_id else f"{info.run_id}-{info.name}"
    return f"""---
library_name: transformers
pipeline_tag: text-generation
---

# {title}

Supervised fine-tuning checkpoint trained with prime-rl: {which}.

- Run `{info.run_id}`, checkpoint `{info.name}`: weights after {info.step} optimizer steps{f" of {info.max_steps}" if info.max_steps else ""}; validation loss {val}.
- Layout: `{architectures}`, the same files, tensor names, dtypes, shapes, config and tokenizer as the checkpoint
  training started from (`{base_dir}`), {num_tensors} tensors. Tensors the trainer does not hold (e.g. an MTP
  head) are that checkpoint's, unchanged.
{description.strip()}

`export_report.json` records, for every fine-tuned tensor, its relative change from the starting checkpoint, and
the sha256 of every shard.
"""


# Distributed steps


def make_dir(path: Path, group) -> None:
    """Create ``path`` on every node (the volume may be staged per node): master first, then one rank per node."""
    world = get_world()
    if world.is_master:
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=group)
    if world.local_rank == 0 and not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=group)


def gather(obj: Any, group) -> list[Any]:
    out: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


def from_master(obj: Any, group) -> Any:
    box = [obj]
    dist.broadcast_object_list(box, src=0, group=group)
    return box[0]


@torch.no_grad()
def collect_hf_tensors(model: torch.nn.Module, wanted: set[str]) -> tuple[dict[str, Tensor], set[str]]:
    """Gather every weight (a collective, so all ranks walk all keys) and keep the HF tensors in ``wanted`` on CPU.

    Also returns every HF key the model produced, to check the mapping against the base."""
    from prime_rl.utils.weights import resolve_fqn, resolve_wire_dtype

    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    kept: dict[str, Tensor] = {}
    produced: set[str] = set()
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value = value.to(resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)).full_tensor()
        for hf_key, tensor in model.convert_to_hf({resolve_fqn(model, key): value}).items():
            produced.add(hf_key)
            if hf_key in wanted:
                kept[hf_key] = tensor.to("cpu", copy=True).contiguous()
        del value
    return kept, produced


@dataclass
class ExportContext:
    base_dir: Path
    out_root: Path
    keep_base_prefixes: tuple[str, ...]
    max_relative_change: float
    private: bool
    description: str
    weight_map: dict[str, str]
    files: list[str]
    my_files: list[str]
    files_by_host: dict[str, list[str]]
    wanted: set[str]
    num_fine_tuned: int
    base_config: dict
    do_upload: bool
    group: Any


def prepare(config, out_root: Path) -> ExportContext:
    logger = get_logger()
    world = get_world()
    cfg = config.hf_export
    base_dir = Path(cfg.base_dir) if cfg.base_dir is not None else Path(config.model.name)
    group = dist.new_group(backend="gloo", timeout=GLOO_TIMEOUT)
    missing = [name for name in REQUIRED_BASE_FILES if not (base_dir / name).is_file()]
    if not _all_true(not missing, group):
        raise FileNotFoundError(f"{base_dir} is not a local HF checkpoint (missing {missing})")
    weight_map: dict[str, str] = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(weight_map.values()))
    assignment = assign_files({name: (base_dir / name).stat().st_size for name in files}, world.world_size)
    my_files = assignment[world.rank]
    prefixes = tuple(cfg.keep_base_prefixes)
    wanted = {key for key, name in weight_map.items() if name in my_files and is_fine_tuned(key, prefixes)}
    hosts = gather(socket.gethostname(), group)
    files_by_host: dict[str, list[str]] = {}
    for rank, names in enumerate(assignment):
        files_by_host.setdefault(hosts[rank], []).extend(names)
    # Every rank must take the same upload branch (it issues collectives), so the master decides.
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    do_upload = bool(from_master(has_token, group))
    if not do_upload:
        logger.warning("HF export: no HF_TOKEN in the environment, the exports stay on the volume only")
    logger.info(f"HF export: base {base_dir}, {len(files)} files, rank {world.rank} writes {my_files}")
    return ExportContext(
        base_dir=base_dir,
        out_root=out_root,
        keep_base_prefixes=prefixes,
        max_relative_change=cfg.max_relative_change,
        private=cfg.private,
        description=cfg.model_card,
        weight_map=weight_map,
        files=files,
        my_files=my_files,
        files_by_host=files_by_host,
        wanted=wanted,
        num_fine_tuned=sum(is_fine_tuned(key, prefixes) for key in weight_map),
        base_config=json.loads((base_dir / "config.json").read_text()),
        do_upload=do_upload,
        group=group,
    )


def _all_true(flag: bool, group) -> bool:
    return all(gather(bool(flag), group))


def export_weights(model: torch.nn.Module, info: ExportInfo, ctx: ExportContext) -> Path | None:
    """Write (and check) one export from the model's current weights. Returns its directory if it passed."""
    logger = get_logger()
    world = get_world()
    out_dir = ctx.out_root / info.name
    start = time.perf_counter()
    logger.info(f"[{info.name}] gathering the weights after step {info.step} and converting them to HF names")
    fine_tuned, produced = collect_hf_tensors(model, ctx.wanted)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    make_dir(out_dir, ctx.group)
    changes: dict[str, float] = {}
    digests: dict[str, str] = {}
    headers: dict[str, dict] = {}
    error = None
    try:
        expected = {key for key in ctx.weight_map if is_fine_tuned(key, ctx.keep_base_prefixes)}
        if world.is_master and not produced <= set(ctx.weight_map):
            raise ValueError(f"the model produced keys the base lacks: {sorted(produced - set(ctx.weight_map))[:5]}")
        if not expected <= produced:
            raise ValueError(f"the model produced none of: {sorted(expected - produced)[:5]}")
        for name in ctx.my_files:
            changes.update(write_file(ctx.base_dir / name, out_dir / name, fine_tuned, ctx.keep_base_prefixes))
            check_layout(ctx.base_dir / name, out_dir / name)
            headers[name] = read_header(out_dir / name)
            digests[name] = sha256_file(out_dir / name)
        if fine_tuned:
            raise ValueError(f"converted keys not in this rank's files: {sorted(fine_tuned)[:5]}")
    except Exception as exc:
        error = f"rank {world.rank}: {exc!r}"
        logger.error(f"[{info.name}] {error}")
    del fine_tuned

    gathered = gather((changes, digests, headers, error), ctx.group)
    ok = True
    if world.is_master:
        errors = [e for *_, e in gathered if e is not None]
        all_changes = {k: v for rank_changes, *_ in gathered for k, v in rank_changes.items()}
        all_digests = {n: d for _, rank_digests, *_ in gathered for n, d in rank_digests.items()}
        all_headers = {n: h for _, _, rank_headers, _ in gathered for n, h in rank_headers.items()}
        summary = summarize_changes(all_changes)
        problems = list(errors)
        num_tensors = 0
        if not errors:
            try:
                num_tensors = check_against_base(ctx.base_dir, all_headers)
            except Exception as exc:
                problems.append(repr(exc))
            problems += find_problems(
                summary, ctx.num_fine_tuned, len(ctx.files), len(all_digests), ctx.max_relative_change
            )
        report = {
            "export": info.name,
            "source": {
                "run_id": info.run_id,
                "step": info.step,
                "max_steps": info.max_steps,
                "val_loss": info.val_loss,
                **info.extra,
            },
            "base": str(ctx.base_dir),
            "tensors": num_tensors,
            "fine_tuned_tensors": len(all_changes),
            "expected_fine_tuned_tensors": ctx.num_fine_tuned,
            "kept_from_base_prefixes": list(ctx.keep_base_prefixes),
            "problems": problems,
            "relative_change": summary,
            "relative_change_per_tensor": dict(sorted(all_changes.items())),
            "sha256": all_digests,
        }
        try:
            for name in BASE_FILES:
                if (ctx.base_dir / name).is_file():
                    shutil.copyfile(ctx.base_dir / name, out_dir / name)
            (out_dir / "README.md").write_text(
                model_card(info, ctx.base_dir, ctx.base_config, ctx.description, num_tensors)
            )
            (out_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        except Exception as exc:
            problems.append(repr(exc))
        logger.info(f"[{info.name}] relative change from the base: {json.dumps(summary['groups'])}")
        if problems:
            logger.error(f"[{info.name}] export check FAILED, not uploading: {problems[:5]}")
            ok = False
        else:
            logger.success(
                f"[{info.name}] {num_tensors} tensors ({len(all_changes)} fine-tuned, max relative change "
                f"{summary['all']['max']:.4f}) in {len(ctx.files)} files written to {out_dir} "
                f"in {time.perf_counter() - start:.0f}s"
            )
    ok = bool(from_master(ok, ctx.group))
    return out_dir if ok else None


def upload(info: ExportInfo, out_dir: Path, ctx: ExportContext) -> None:
    """Push an export: small files first (rank 0), then each node's shards (preupload in parallel, commits in turn)."""
    from huggingface_hub import CommitOperationAdd, HfApi

    logger = get_logger()
    world = get_world()
    group = ctx.group
    api = HfApi()
    host = socket.gethostname()
    is_host_leader = world.local_rank == 0
    repo_id = info.repo_id

    ok = True
    if world.is_master:
        try:
            api.create_repo(repo_id, private=ctx.private, exist_ok=True)
            small = [
                p.name
                for p in sorted(out_dir.iterdir())
                if p.is_file() and not p.name.endswith((".safetensors", ".tmp"))
            ]
            api.create_commit(
                repo_id,
                operations=[CommitOperationAdd(name, out_dir / name) for name in small],
                commit_message=f"Model card, config, tokenizer and export report ({info.name}, step {info.step})",
            )
        except Exception as error:
            logger.error(f"[{info.name}] repo setup on {repo_id} failed: {error!r}")
            ok = False
    if not from_master(ok, group):
        return

    operations: list = []
    error_text = None
    if is_host_leader:
        try:
            operations = [CommitOperationAdd(name, out_dir / name) for name in ctx.files_by_host[host]]
            api.preupload_lfs_files(repo_id, additions=operations)
        except Exception as error:
            error_text = repr(error)
            logger.error(f"[{info.name}] upload from {host} failed: {error_text}")
    for turn in sorted(ctx.files_by_host):
        if is_host_leader and turn == host and error_text is None and operations:
            try:
                api.create_commit(
                    repo_id,
                    operations=operations,
                    commit_message=f"Weights: {len(operations)} shards ({info.name}, {host})",
                )
            except Exception as error:
                error_text = repr(error)
                logger.error(f"[{info.name}] commit from {host} failed: {error_text}")
        dist.barrier(group=group)

    if world.is_master:
        try:
            expected = json.loads((out_dir / "export_report.json").read_text())["sha256"]
            remote = {
                item.path: item.lfs.sha256
                for item in api.get_paths_info(repo_id, list(expected), expand=True)
                if getattr(item, "lfs", None) is not None
            }
            missing = sorted(name for name, digest in expected.items() if remote.get(name) != digest)
            if missing:
                logger.error(f"[{info.name}] {len(missing)} shards missing or different on {repo_id}: {missing[:5]}")
            else:
                logger.success(f"[{info.name}] all {len(expected)} shards on {repo_id} match the local sha256")
        except Exception as error:
            logger.error(f"[{info.name}] could not verify {repo_id}: {error!r}")
    dist.barrier(group=group)


def run(config, model: torch.nn.Module, best, final_step: int, ckpt_root: Path, extra: dict | None = None) -> None:
    """Export ``config.hf_export.checkpoints`` (last first, from memory; then best). Never raises."""
    from prime_rl.trainer.model import reshard_module

    logger = get_logger()
    cfg = config.hf_export
    out_root = ckpt_root / EXPORT_DIR_NAME
    run_id = ckpt_root.name
    start = time.perf_counter()
    try:
        ctx = prepare(config, out_root)
    except Exception as exc:
        logger.error(f"HF export: setup failed, nothing exported: {exc!r}")
        return
    history = best.history if best is not None else {}
    done: list[str] = []
    for name in [n for n in ("last", "best") if n in cfg.checkpoints]:
        try:
            if name == "last":
                step = final_step
            else:
                if best is None or best.best is None:
                    logger.error("HF export: no best checkpoint was recorded, skipping 'best'")
                    continue
                step = best.best["step"]
                if step != final_step and not best.restore(model):
                    logger.error(f"HF export: could not load the best checkpoint (step {step}), skipping 'best'")
                    continue
            reshard_module(model)
            info = ExportInfo(
                name=name,
                repo_id=cfg.best_repo if name == "best" else cfg.last_repo,
                step=step,
                max_steps=config.max_steps,
                val_loss=history.get(step),
                run_id=run_id,
                extra=dict(extra or {}),
            )
            out_dir = export_weights(model, info, ctx)
            if out_dir is None:
                continue
            done.append(name)
            if info.repo_id is None:
                logger.info(f"[{name}] no repo configured, kept on the volume at {out_dir}")
            elif ctx.do_upload:
                upload(info, out_dir, ctx)
        except Exception as exc:
            logger.error(f"HF export of '{name}' failed: {exc!r}")
    logger.success(f"HF export finished in {time.perf_counter() - start:.0f}s: {done} under {out_root}")
