"""End-of-run Hugging Face export of an SFT run's checkpoints (``[hf_export]``).

Runs after the final step and the final checkpoint, before the trainer exits. ``last`` is exported from the weights
in memory; ``best`` is loaded from the ``[val] keep_best`` snapshot first, or, when the best is the final step, copied
from the ``last`` export (every rank links or copies only the files it wrote itself). Each export
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

Nodes are ``rank // local_world_size`` (torchrun's contiguous ranks per node), the same split as ``local_rank == 0``
picking each node's uploader.

Every rank writes only its own files and every node uploads only what its ranks wrote, so nothing depends on
files another node wrote during the run. Every file is written to a temp file of its writer's own (host, pid, random
suffix) and renamed onto its path atomically, and the checks read that temp file: this holds whether the run dir is
staged per node or shared by all nodes (as the hosted volume behaves: a delete on one node removes other nodes'
files too). Failures are logged and never raised: the run must still end cleanly so that its checkpoints are kept.

With ``full_per_node`` (the in-run eval serves the export with vLLM on every node) the files are spread over the
ranks of each node instead of over all ranks, so every node writes a complete copy from the same gather (no extra
traffic). On a shared run dir the nodes write the same paths at once: each writer renames its own complete temp file,
so a path only ever holds one complete copy. The copies are byte-identical (no reduction in the gather, a fixed cast
and writer), which the master checks from the sha256 each writer took of its own temp file; uploads still take each
file from one node only.

A restart of a finished run (``training_done``) reuses an export whose ``export_report.json`` (written last, after
every check passed) is there for the same step, and pushes it again only if the repo does not have it.

Every rank hands a file to its writer thread as soon as all of the file's tensors are gathered: the writes overlap
the gather, and host memory holds the files not written yet instead of all of the rank's files.

``run`` returns what it exported (``ExportResult`` per checkpoint) and, per ``[hf_export.cleanup]``, deletes what the
exports made redundant once they are verified on Hugging Face (the best snapshot, the final checkpoint, the exports).

Adapted from the one-off export jobs of exp/nemotron-sft-hf-export (19cabb8ae, TEXT_LAYOUT) and
exp/nemotron-sft-text-extract (039ecaaaf, check_against_training_base).
"""

import filecmp
import hashlib
import json
import os
import shutil
import socket
import struct
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

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


def training_done(checkpoint_step: int | None, max_steps: int | None) -> bool:
    """The resumed checkpoint is at (or past) the last step: a restart of a finished run (e.g. its export or its
    in-run eval died). It must train nothing (no step past ``max_steps``, no new final checkpoint, no new best) and
    go straight to the export, which reuses what the earlier attempt exported."""
    return checkpoint_step is not None and max_steps is not None and checkpoint_step >= max_steps


def temp_path(dst: Path) -> Path:
    """A temp name next to ``dst`` no other writer uses (ending in ``.tmp``, which uploads skip)."""
    return dst.with_name(f"{dst.name}.{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")


def replace_atomically(dst: Path, write: Callable[[Path], Any]) -> Any:
    """``write`` this writer's own temp file, then rename it onto ``dst`` (atomic): ``dst`` never holds a partial
    file, even when other nodes write the same path at the same time. Returns what ``write`` returned."""
    tmp = temp_path(dst)
    try:
        out = write(tmp)
        os.replace(tmp, dst)
        return out
    finally:
        tmp.unlink(missing_ok=True)


def copy_atomically(src: Path, dst: Path) -> None:
    replace_atomically(dst, lambda tmp: shutil.copyfile(src, tmp))


def write_file(
    src: Path, dst: Path, fine_tuned: dict[str, Tensor], keep_base_prefixes: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, tuple[str, list[int]]], str]:
    """Write ``src`` to ``dst`` with its fine-tuned keys swapped in (cast to the source dtype).

    Consumes the swapped-in entries of ``fine_tuned``. Returns each one's relative L2 change, and the header and
    sha256 of the written file, read back (and checked against ``src``) from this writer's temp file before the
    rename: another node may rename its copy onto ``dst`` at any time.
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

    def save(tmp: Path) -> tuple[dict, str]:
        save_file(tensors, tmp, metadata=metadata)
        check_layout(src, tmp)
        return read_header(tmp), sha256_file(tmp)

    header, digest = replace_atomically(dst, save)
    return changes, header, digest


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


def files_by_node(assignment: list[list[str]], local_world_size: int) -> dict[int, list[str]]:
    """The files each node's ranks write, by node index (``rank // local_world_size``)."""
    out: dict[int, list[str]] = {}
    for rank, names in enumerate(assignment):
        out.setdefault(rank // local_world_size, []).extend(names)
    return out


def make_dir(path: Path, group) -> str | None:
    """Create ``path`` on every node (the volume may be staged per node): master first, then one rank per node.

    Returns this rank's error, if any; every rank reaches both barriers either way."""
    world = get_world()
    error = None
    try:
        if world.is_master:
            path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        error = f"rank {world.rank}: {exc!r}"
    dist.barrier(group=group)
    try:
        if error is None and world.local_rank == 0 and not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        error = f"rank {world.rank}: {exc!r}"
    dist.barrier(group=group)
    return error


def gather(obj: Any, group) -> list[Any]:
    out: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


def from_master(obj: Any, group) -> Any:
    box = [obj]
    dist.broadcast_object_list(box, src=0, group=group)
    return box[0]


@torch.no_grad()
def collect_hf_tensors(
    model: torch.nn.Module, wanted: set[str], on_tensor: Callable[[str, Tensor], None] | None = None
) -> tuple[dict[str, Tensor], set[str]]:
    """Gather every weight (a collective, so all ranks walk all keys) and keep the HF tensors in ``wanted`` on CPU.

    With ``on_tensor``, each wanted tensor is handed to it instead of being kept (the returned dict is then empty).
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
                tensor = tensor.to("cpu", copy=True).contiguous()
                if on_tensor is None:
                    kept[hf_key] = tensor
                else:
                    on_tensor(hf_key, tensor)
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
    files_by_node: dict[int, list[str]]
    wanted: set[str]
    num_fine_tuned: int
    base_config: dict
    do_upload: bool
    group: Any
    full_per_node: bool = False
    num_nodes: int = 1


def prepare(config, out_root: Path, full_per_node: bool = False) -> ExportContext:
    """Collective. ``full_per_node``: every node writes all files (spread over its own ranks), see the module docs."""
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
    sizes = {name: (base_dir / name).stat().st_size for name in files}
    # Uploads always take each file from the one node whose ranks hold it in the world-wide split.
    assignment = assign_files(sizes, world.world_size)
    my_files = (
        assign_files(sizes, world.local_world_size)[world.local_rank] if full_per_node else assignment[world.rank]
    )
    prefixes = tuple(cfg.keep_base_prefixes)
    wanted = {key for key, name in weight_map.items() if name in my_files and is_fine_tuned(key, prefixes)}
    # Every rank must take the same upload branch (it issues collectives), so the master decides.
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    do_upload = bool(from_master(has_token, group))
    if not do_upload:
        logger.warning("HF export: no HF_TOKEN in the environment, the exports stay on the volume only")
    logger.info(
        f"HF export: base {base_dir}, {len(files)} files{' (a full copy per node)' if full_per_node else ''}, "
        f"rank {world.rank} writes {my_files}"
    )
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
        files_by_node=files_by_node(assignment, world.local_world_size),
        wanted=wanted,
        num_fine_tuned=sum(is_fine_tuned(key, prefixes) for key in weight_map),
        base_config=json.loads((base_dir / "config.json").read_text()),
        do_upload=do_upload,
        group=group,
        full_per_node=full_per_node,
        num_nodes=world.num_nodes,
    )


def _all_true(flag: bool, group) -> bool:
    return all(gather(bool(flag), group))


def export_weights(model: torch.nn.Module, info: ExportInfo, ctx: ExportContext) -> Path | None:
    """Write (and check) one export from the model's current weights. Returns its directory if it passed."""
    logger = get_logger()
    world = get_world()
    out_dir = ctx.out_root / info.name
    start = time.perf_counter()

    changes: dict[str, float] = {}
    digests: dict[str, str] = {}
    headers: dict[str, dict] = {}
    error: str | None = None
    drop_report(out_dir)
    mkdir_error = make_dir(out_dir, ctx.group)
    if mkdir_error is not None:
        error = f"rank {world.rank}: {OSError(f'could not create {out_dir}: {mkdir_error}')!r}"

    # Each of this rank's files is written as soon as all its fine-tuned tensors are gathered, by one writer thread,
    # so the writes overlap the gather (a write on the main thread would hold every rank's next collective) and host
    # memory holds the files not written yet rather than all of them.
    fine_tuned: dict[str, Tensor] = {}
    file_keys = {name: set() for name in ctx.my_files}
    for key in ctx.wanted:
        file_keys[ctx.weight_map[key]].add(key)
    pending = {name: set(keys) for name, keys in file_keys.items()}
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-export-writer")
    writes: dict[str, Future] = {}

    def write(name: str, tensors: dict[str, Tensor]) -> tuple[dict[str, float], dict, str]:
        return write_file(ctx.base_dir / name, out_dir / name, tensors, ctx.keep_base_prefixes)

    def submit(names: list[str]) -> None:
        for name in names:
            del pending[name]
            tensors = {key: fine_tuned.pop(key) for key in file_keys[name] if key in fine_tuned}
            if error is None:
                writes[name] = writer.submit(write, name, tensors)

    def on_tensor(hf_key: str, tensor: Tensor) -> None:
        # Never raises: every rank must keep walking the collective gather.
        nonlocal error
        if error is not None:
            return  # drop the tensor
        try:
            name = ctx.weight_map[hf_key]
            keys = pending.get(name)
            if keys is None or hf_key not in keys:
                raise ValueError(f"{hf_key}: produced again after it was written")
            fine_tuned[hf_key] = tensor
            keys.discard(hf_key)
            if not keys:
                submit([name])
        except Exception as exc:
            error = f"rank {world.rank}: {exc!r}"
            fine_tuned.clear()

    # Files of only base-kept tensors (e.g. an MTP-only shard) need nothing from the model.
    submit([name for name, keys in pending.items() if not keys])
    logger.info(f"[{info.name}] gathering the weights after step {info.step} and converting them to HF names")
    try:
        _, produced = collect_hf_tensors(model, ctx.wanted, on_tensor)
    finally:
        writer.shutdown(wait=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        if error is not None:
            raise _RankFailed(error)
        expected = {key for key in ctx.weight_map if is_fine_tuned(key, ctx.keep_base_prefixes)}
        if world.is_master and not produced <= set(ctx.weight_map):
            raise ValueError(f"the model produced keys the base lacks: {sorted(produced - set(ctx.weight_map))[:5]}")
        if not expected <= produced:
            raise ValueError(f"the model produced none of: {sorted(expected - produced)[:5]}")
        for name in sorted(writes):
            file_changes, headers[name], digests[name] = writes[name].result()
            changes.update(file_changes)
        # Files whose tensors never all arrived: write_file names the missing one.
        for name in sorted(pending):
            tensors = {key: fine_tuned.pop(key) for key in file_keys[name] if key in fine_tuned}
            file_changes, headers[name], digests[name] = write(name, tensors)
            changes.update(file_changes)
        if fine_tuned:
            raise ValueError(f"converted keys not in this rank's files: {sorted(fine_tuned)[:5]}")
    except _RankFailed:
        logger.error(f"[{info.name}] {error}")
    except Exception as exc:
        error = f"rank {world.rank}: {exc!r}"
        logger.error(f"[{info.name}] {error}")
    fine_tuned.clear()

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
            if ctx.full_per_node:
                problems += check_node_copies(
                    [rank_digests for _, rank_digests, *_ in gathered], world.local_world_size, ctx.files
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
            "full_copy_per_node": ctx.full_per_node,
            "problems": problems,
            "relative_change": summary,
            "relative_change_per_tensor": dict(sorted(all_changes.items())),
            "sha256": all_digests,
        }
        try:
            write_small_files(info, out_dir, ctx, report)
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
    if ok:
        copy_base_files_on_node(out_dir, ctx)
    return out_dir if ok else None


class _RankFailed(Exception):
    """This rank's error is already recorded."""


def drop_report(out_dir: Path) -> None:
    """Master, before an export (re)writes ``out_dir``: ``export_report.json`` marks a complete, checked export (a
    restart reuses it), so an earlier one must not stay next to files being rewritten."""
    if not get_world().is_master:
        return
    try:
        (out_dir / "export_report.json").unlink(missing_ok=True)
    except OSError as exc:
        get_logger().warning(f"could not remove the old {out_dir / 'export_report.json'}: {exc!r}")


def check_node_copies(rank_digests: list[dict[str, str]], local_world_size: int, files: list[str]) -> list[str]:
    """With a full copy per node: every node wrote every file, and all copies of a file are identical."""
    num_nodes = len(rank_digests) // local_world_size
    copies: dict[str, dict[int, str]] = {}
    for rank, digests in enumerate(rank_digests):
        for name, digest in digests.items():
            copies.setdefault(name, {})[rank // local_world_size] = digest
    problems = []
    for name in files:
        got = copies.get(name, {})
        if len(got) != num_nodes:
            problems.append(f"{name}: written on nodes {sorted(got)} of {num_nodes}")
        elif len(set(got.values())) != 1:
            problems.append(f"{name}: the node copies differ ({sorted(set(got.values()))[:2]})")
    return problems


def copy_base_files_on_node(out_dir: Path, ctx: ExportContext) -> None:
    """With a full copy per node, the other nodes' ``local_rank 0`` also put the base's config and tokenizer next to
    their copy (the master wrote its own with the report): enough to serve it. On a shared run dir they are there
    already (left alone); otherwise each is written by atomic rename, so the master's upload never reads a partial
    one. Best effort, logged."""
    world = get_world()
    if not ctx.full_per_node or world.is_master or world.local_rank != 0:
        return
    try:
        for name in BASE_FILES:
            src, dst = ctx.base_dir / name, out_dir / name
            if src.is_file() and not (dst.is_file() and filecmp.cmp(src, dst, shallow=False)):
                copy_atomically(src, dst)
    except Exception as exc:
        get_logger().error(f"copying the base config/tokenizer next to {out_dir} failed: {exc!r}")


def write_small_files(info: ExportInfo, out_dir: Path, ctx: ExportContext, report: dict) -> None:
    """Master: the base's config and tokenizer files, the model card and, last, the export report."""
    for name in BASE_FILES:
        if (ctx.base_dir / name).is_file():
            copy_atomically(ctx.base_dir / name, out_dir / name)
    card = model_card(info, ctx.base_dir, ctx.base_config, ctx.description, report["tensors"])
    replace_atomically(out_dir / "README.md", lambda tmp: tmp.write_text(card))
    text = json.dumps(report, indent=2) + "\n"
    replace_atomically(out_dir / "export_report.json", lambda tmp: tmp.write_text(text))


def link_or_copy(src: Path, dst: Path) -> None:
    """A hard link where the filesystem allows it (exports are only ever replaced by rename, never rewritten in
    place), else a copy; through this writer's own temp file."""

    def make(tmp: Path) -> None:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)

    replace_atomically(dst, make)


def copy_export(src_info: ExportInfo, src_dir: Path, info: ExportInfo, ctx: ExportContext) -> Path | None:
    """Make export ``info`` from ``src_info``'s, which holds the same weights (the best is the final step), without
    gathering or writing the weights again. Every rank links or copies only the shards it wrote into ``src_dir``, and
    the master its own small files with a new model card and report. Returns the directory if it passed."""
    logger = get_logger()
    world = get_world()
    out_dir = ctx.out_root / info.name
    drop_report(out_dir)
    error = make_dir(out_dir, ctx.group)
    try:
        if error is not None:
            raise OSError(f"could not create {out_dir}: {error}")
        for name in ctx.my_files:
            link_or_copy(src_dir / name, out_dir / name)
            if (out_dir / name).stat().st_size != (src_dir / name).stat().st_size:
                raise OSError(f"{out_dir / name}: size differs from {src_dir / name}")
    except Exception as exc:
        error = f"rank {world.rank}: {exc!r}"
        logger.error(f"[{info.name}] {error}")
    errors = [e for e in gather(error, ctx.group) if e is not None]
    ok = True
    if world.is_master:
        problems = list(errors)
        if not problems:
            try:
                report = json.loads((src_dir / "export_report.json").read_text())
                report["export"] = info.name
                report["source"] = {
                    "run_id": info.run_id,
                    "step": info.step,
                    "max_steps": info.max_steps,
                    "val_loss": info.val_loss,
                    **info.extra,
                }
                report["copied_from"] = src_info.name
                write_small_files(info, out_dir, ctx, report)
            except Exception as exc:
                problems.append(repr(exc))
        if problems:
            logger.error(f"[{info.name}] copying the '{src_info.name}' export FAILED, not uploading: {problems[:5]}")
            ok = False
        else:
            logger.success(
                f"[{info.name}] step {info.step} is also '{src_info.name}': its {len(ctx.files)} files copied to "
                f"{out_dir}"
            )
    ok = bool(from_master(ok, ctx.group))
    if ok:
        copy_base_files_on_node(out_dir, ctx)
    return out_dir if ok else None


def remote_mismatches(api, repo_id: str, expected: dict[str, str]) -> list[str]:
    """The files of ``expected`` (name -> sha256) that ``repo_id`` lacks or holds with another sha256."""
    remote = {
        item.path: item.lfs.sha256
        for item in api.get_paths_info(repo_id, list(expected), expand=True)
        if getattr(item, "lfs", None) is not None
    }
    return sorted(name for name, digest in expected.items() if remote.get(name) != digest)


def reusable_export(out_dir: Path, base_dir: Path, step: int | None) -> dict | None:
    """The report of a complete export in ``out_dir`` that passed its checks, for ``step`` (None: any step), else
    None. The report is written last, only after every check passed; the files it lists must still be there with the
    base's layout (headers only: cheap)."""
    try:
        report = json.loads((out_dir / "export_report.json").read_text())
        if report.get("problems") != [] or (step is not None and report["source"]["step"] != step):
            return None
        files = sorted(set(json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"].values()))
        if sorted(report["sha256"]) != files:
            return None
        for name in REQUIRED_BASE_FILES:
            if not (out_dir / name).is_file():
                return None
        for name in files:
            check_layout(base_dir / name, out_dir / name)
    except Exception:
        return None
    return report


def on_hub(repo_id: str, expected: dict[str, str], group) -> bool:
    """Collective: whether ``repo_id`` holds every file of ``expected`` with its sha256 (the master asks)."""
    found = False
    if get_world().is_master:
        try:
            from huggingface_hub import HfApi

            found = not remote_mismatches(HfApi(), repo_id, expected)
        except Exception as exc:
            get_logger().warning(f"could not check {repo_id}: {exc!r}")
    return bool(from_master(found, group))


def upload(info: ExportInfo, out_dir: Path, ctx: ExportContext) -> bool:
    """Push an export: small files first (rank 0), then each node's shards (preupload in parallel, commits in turn).

    Returns (on every rank) whether every shard on the repo was verified against the local sha256."""
    from huggingface_hub import CommitOperationAdd, HfApi

    logger = get_logger()
    world = get_world()
    group = ctx.group
    api = HfApi()
    host = socket.gethostname()
    node = world.rank // world.local_world_size
    is_node_leader = world.local_rank == 0
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
        return False

    operations: list = []
    error_text = None
    if is_node_leader:
        try:
            operations = [CommitOperationAdd(name, out_dir / name) for name in ctx.files_by_node.get(node, [])]
            api.preupload_lfs_files(repo_id, additions=operations)
        except Exception as error:
            error_text = repr(error)
            logger.error(f"[{info.name}] upload from node {node} ({host}) failed: {error_text}")
    for turn in sorted(ctx.files_by_node):
        if is_node_leader and turn == node and error_text is None and operations:
            try:
                api.create_commit(
                    repo_id,
                    operations=operations,
                    commit_message=f"Weights: {len(operations)} shards ({info.name}, node {node})",
                )
            except Exception as error:
                error_text = repr(error)
                logger.error(f"[{info.name}] commit from node {node} ({host}) failed: {error_text}")
        dist.barrier(group=group)

    verified = False
    if world.is_master:
        try:
            expected = json.loads((out_dir / "export_report.json").read_text())["sha256"]
            missing = remote_mismatches(api, repo_id, expected)
            if missing:
                logger.error(f"[{info.name}] {len(missing)} shards missing or different on {repo_id}: {missing[:5]}")
            else:
                verified = True
                logger.success(f"[{info.name}] all {len(expected)} shards on {repo_id} match the local sha256")
        except Exception as error:
            logger.error(f"[{info.name}] could not verify {repo_id}: {error!r}")
    return bool(from_master(verified, group))


@dataclass
class ExportResult:
    """What ``run`` made of one checkpoint (the same on every rank)."""

    name: str
    ok: bool = False
    dir: str | None = None
    step: int | None = None
    val_loss: float | None = None
    repo_id: str | None = None
    pushed: bool = False
    """The upload was attempted (a repo and HF_TOKEN were set)."""
    verified: bool = False
    """Every shard on the repo matches the local sha256."""
    same_as: str | None = None
    """'last' when 'best' is the final step (its files are a copy of the 'last' export)."""

    def to_dict(self) -> dict:
        return asdict(self)


def run(
    config,
    model: torch.nn.Module,
    best,
    final_step: int,
    ckpt_root: Path,
    extra: dict | None = None,
    full_per_node: bool = False,
    defer_export_cleanup: bool = False,
    resumed_finished: bool = False,
) -> dict[str, ExportResult]:
    """Export ``config.hf_export.checkpoints`` (last first, from memory; then best). Never raises.

    ``full_per_node``: every node writes a complete copy (for the in-run eval). ``defer_export_cleanup``: leave the
    exports in place (the in-run eval deletes them after it is done, per ``[hf_export.cleanup] pushed_exports``).
    An export already complete for the same step (``reusable_export``) is kept, not written again, and pushed only
    if its repo lacks it. ``resumed_finished`` (``training_done``: nothing was trained): without a recorded best (its
    snapshot was deleted after the 'best' export was verified), the 'best' export there is reused whatever its step."""
    from prime_rl.trainer.model import reshard_module

    logger = get_logger()
    cfg = config.hf_export
    out_root = ckpt_root / EXPORT_DIR_NAME
    run_id = ckpt_root.name
    start = time.perf_counter()
    results: dict[str, ExportResult] = {name: ExportResult(name=name) for name in cfg.checkpoints}
    try:
        ctx = prepare(config, out_root, full_per_node=full_per_node)
    except Exception as exc:
        logger.error(f"HF export: setup failed, nothing exported: {exc!r}")
        return results
    history = best.history if best is not None else {}
    done: list[str] = []
    # The passed export of the final weights (in memory), if any: 'best' at the final step is copied from it.
    final_export: tuple[ExportInfo, Path] | None = None

    def reuse(name: str, step: int | None) -> tuple[ExportInfo, Path] | None:
        """Collective: keep an export that is already complete for ``step``, and make sure its repo has it."""
        out_dir = out_root / name
        report = from_master(reusable_export(out_dir, ctx.base_dir, step) if get_world().is_master else None, ctx.group)
        if report is None:
            return None
        source = report["source"]
        info = ExportInfo(
            name=name,
            repo_id=cfg.best_repo if name == "best" else cfg.last_repo,
            step=source["step"],
            max_steps=source.get("max_steps"),
            val_loss=source.get("val_loss"),
            run_id=run_id,
            extra=dict(extra or {}),
        )
        result = results[name]
        result.ok, result.dir, result.step, result.val_loss = True, str(out_dir), info.step, info.val_loss
        result.repo_id, result.same_as = info.repo_id, report.get("copied_from")
        logger.info(f"[{name}] the export of step {info.step} in {out_dir} is complete and checked: kept as it is")
        if info.repo_id is not None and ctx.do_upload:
            result.pushed = True
            result.verified = on_hub(info.repo_id, report["sha256"], ctx.group)
            if result.verified:
                logger.info(f"[{name}] {info.repo_id} already has it (sha256 checked)")
            else:
                result.verified = upload(info, out_dir, ctx)
        return info, out_dir

    for name in [n for n in ("last", "best") if n in cfg.checkpoints]:
        result = results[name]
        try:
            if name == "last":
                step, val_loss = final_step, history.get(final_step)
            elif best is not None and best.best is not None:
                # The loss measured on the snapshot's weights (a replayed validation never replaces it).
                step, val_loss = best.best["step"], best.best["val_loss"]
            elif resumed_finished:
                step, val_loss = None, None
            else:
                logger.error("HF export: no best checkpoint was recorded, skipping 'best'")
                continue
            reused = reuse(name, step)
            if reused is not None:
                done.append(name)
                if name == "last":
                    final_export = reused
                continue
            if step is None:
                logger.error("HF export: no best checkpoint was recorded and no 'best' export to reuse, skipping it")
                continue
            if name == "best":
                if step != final_step and not best.restore(model):
                    logger.error(f"HF export: could not load the best checkpoint (step {step}), skipping 'best'")
                    continue
            info = ExportInfo(
                name=name,
                repo_id=cfg.best_repo if name == "best" else cfg.last_repo,
                step=step,
                max_steps=config.max_steps,
                val_loss=val_loss,
                run_id=run_id,
                extra=dict(extra or {}),
            )
            result.step, result.val_loss, result.repo_id = step, val_loss, info.repo_id
            if final_export is not None and step == final_step:
                out_dir = copy_export(*final_export, info, ctx)
                result.same_as = final_export[0].name
            else:
                reshard_module(model)
                out_dir = export_weights(model, info, ctx)
                if out_dir is not None and step == final_step:
                    final_export = (info, out_dir)
            if out_dir is None:
                continue
            result.ok, result.dir = True, str(out_dir)
            done.append(name)
            if info.repo_id is None:
                logger.info(f"[{name}] no repo configured, kept on the volume at {out_dir}")
            elif ctx.do_upload:
                result.pushed = True
                result.verified = upload(info, out_dir, ctx)
        except Exception as exc:
            logger.error(f"HF export of '{name}' failed: {exc!r}")
    logger.success(f"HF export finished in {time.perf_counter() - start:.0f}s: {done} under {out_root}")
    try:
        cleanup_staging(cfg.cleanup, results, best, ckpt_root, final_step, ctx.group, defer_export_cleanup)
    except Exception as exc:
        logger.error(f"HF export: staging cleanup failed: {exc!r}")
    return results


def staging_deletions(
    cleanup, results: dict[str, ExportResult], best_root: Path | None, ckpt_root: Path, final_step: int, exports: bool
) -> list[Path]:
    """What ``[hf_export.cleanup]`` allows deleting now, given the export results (pure; unit-tested). Every deletion
    waits until the export that replaces the data is verified on Hugging Face: the run dir is shared by the nodes, so
    nothing deleted here survives anywhere else on the volume."""
    out: list[Path] = []
    best, last = results.get("best"), results.get("last")
    if cleanup.best_snapshot and best_root is not None and best is not None and best.ok and best.verified:
        out.append(best_root)
    if cleanup.final_checkpoint and last is not None and last.ok and last.verified:
        # prime_rl.utils.pathing: get_step_path(get_ckpt_dir(run dir), step)
        out.append(ckpt_root / "checkpoints" / f"step_{final_step}")
    if exports and cleanup.pushed_exports:
        out += [Path(r.dir) for r in results.values() if r.ok and r.verified and r.dir is not None]
    return out


def cleanup_staging(
    cleanup, results: dict[str, ExportResult], best, ckpt_root: Path, final_step: int, group, defer_exports: bool
) -> None:
    """Collective: after every rank is done reading, each node's ``local_rank 0`` deletes the targets (the same paths
    on every node: whichever node gets there first removes them, the others find nothing left)."""
    logger = get_logger()
    world = get_world()
    targets = staging_deletions(
        cleanup, results, best.root if best is not None else None, ckpt_root, final_step, exports=not defer_exports
    )
    dist.barrier(group=group)
    if world.local_rank == 0:
        for path in targets:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Deleted {path} ([hf_export.cleanup])")
    dist.barrier(group=group)
