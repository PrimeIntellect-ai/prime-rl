"""Export GeneBench RL checkpoints of Nemotron 3.5 Super to text-only Hugging Face repos (no training).

On this branch the SFT entrypoint runs this export instead of training, because a hosted run can only start the
trainer. The watcher picks the steps and passes them in GB_EXPORT_* variables, written into the config's
[env_vars] table (see rl_export_plan.py; `prime train -e` only forwards HF_TOKEN). A step's checkpoint counts as
complete only when every data file its .metadata references is fully on the volume. For each step, the export:
1. loads the model weights (not the optimizer) from the RL trainer's DCP checkpoint,
   /volume/runs/$GB_EXPORT_RUN_ID/checkpoints/step_N/trainer. Its app.model entries are the same 763 fp32
   tensors under the same canonical names as an SFT checkpoint, so the same AppState template loads them;
2. gathers them in fp32 and converts them to HF names;
3. rewrites every safetensors file of the base text model (/volume/models/nemotron-3.5-super-text, the model
   that was trained) with the trained tensors cast to that file's dtypes. The 1,040 mtp.* tensors are not in the
   checkpoint (prime-rl's NemotronH drops them) and are copied from the base unchanged.

The result is the base model's exact layout: the same tensor names, dtypes, shapes and shard files, plus its
config, tokenizer and chat template. Each rank writes the files assigned to it under the run's directory
([ckpt] output_dir/hf-export/step_N; it is synced to the volume after the run). Files are checked against the
base (key sets, dtypes, shapes, how far each tensor moved), pushed to the step's private repo and verified there
by sha256. Then the run's export_report.json is updated, and the step's GB_EXPORT_DONE line is logged.
Any failure raises on every rank, so the run exits non-zero.
"""

import json
import os
import shutil
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.tensor import DTensor

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.sft import rl_export_plan as plan
from prime_rl.trainer.sft.rl_export_files import assign_files, is_trained, write_my_files
from prime_rl.trainer.utils import get_dcp_group
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.weights import resolve_fqn

BASE_REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
BASE_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
# The release's language model as its own NemotronHForCausalLM checkpoint (keys without "language_model.",
# config = the release's llm_config): what the RL run trained from and what prime-rl's trainer and inference load.
BASE_DIR = Path("/volume/models/nemotron-3.5-super-text")
BASE_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
EXPECTED_TENSORS = 42683
EXPECTED_UNTRAINED = 1040
# A trained tensor this far (relative L2) from the base means a wrong key mapping, not training.
MAX_RELATIVE_CHANGE = 0.2
GLOO_TIMEOUT = timedelta(hours=12)
ALLOWED_EXTRA_REPO_FILES = {".gitattributes"}


def raise_if_failed(error: str | None, what: str, group) -> None:
    """Collective: every rank raises if any rank failed, so the run exits non-zero instead of hanging."""
    errors: list = [None] * get_world().world_size
    dist.all_gather_object(errors, error, group=group)
    failed = [(rank, err) for rank, err in enumerate(errors) if err is not None]
    if failed:
        raise RuntimeError(f"{what} failed on {len(failed)} rank(s): {failed[:4]}")


def run_guarded(what: str, group, fn, *args, **kwargs):
    """Run ``fn`` on every rank and fail all ranks together if it raised on any (``fn`` must not use collectives)."""
    result, error = None, None
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        get_logger().error(f"{what} failed on rank {get_world().rank}: {error}")
    raise_if_failed(error, what, group)
    return result


def make_dir(path: Path, group) -> None:
    """Create ``path`` on every node: master first, then one rank per node if it does not see it yet, since
    concurrent mkdirs of one path can fail on a parallel FS."""
    world = get_world()
    if world.is_master:
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=group)
    if world.local_rank == 0 and not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=group)


def collect_hf_tensors(model: torch.nn.Module, wanted: set[str]) -> dict[str, Tensor]:
    """Gather every weight in fp32 (a collective, so all ranks walk all keys) and keep the HF tensors in
    ``wanted`` on CPU. fp32 keeps the trainer's exact weights, so each is rounded once, to its file's dtype."""
    kept: dict[str, Tensor] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value = value.full_tensor()
        for hf_key, tensor in model.convert_to_hf({resolve_fqn(model, key): value.float()}).items():
            if hf_key in wanted:
                kept[hf_key] = tensor.to("cpu", copy=True).contiguous()
    return kept


def hub(what: str, fn, *args, **kwargs):
    """A Hugging Face call, retried on transient errors (all of the export's calls are safe to repeat)."""
    return plan.retry(lambda: fn(*args, **kwargs), what, log=get_logger().warning)


def dcp_file_ends(dcp_dir: Path) -> dict[str, int]:
    """Data file -> largest offset + length that the checkpoint's .metadata references in it."""
    metadata = FileSystemReader(dcp_dir).read_metadata()
    ends: dict[str, int] = {}
    for info in (metadata.storage_data or {}).values():
        ends[info.relative_path] = max(ends.get(info.relative_path, 0), info.offset + info.length)
    return ends


def dcp_is_complete(dcp_dir: Path) -> bool:
    """Every data file the checkpoint's .metadata references exists at its full size."""
    try:
        problems = plan.storage_problems(dcp_dir, dcp_file_ends(dcp_dir))
    except Exception as error:
        problems = [f"cannot read {dcp_dir}/.metadata: {type(error).__name__}: {error}"]
    if problems:
        get_logger().warning(f"{dcp_dir} is not a complete checkpoint: {problems[:3]} ({len(problems)} problems)")
    return not problems


def preflight_repos(specs: list[plan.ExportSpec]) -> None:
    """Master only: the token works, and every repo exists (created if missing) and is private."""
    from huggingface_hub import HfApi

    api = HfApi()
    user = hub("whoami", api.whoami).get("name")
    get_logger().info(f"Hugging Face token of {user}")
    for spec in specs:
        hub(f"create_repo {spec.repo_id}", api.create_repo, spec.repo_id, private=True, exist_ok=True)
        if not hub(f"repo_info {spec.repo_id}", api.repo_info, spec.repo_id).private:
            raise ValueError(f"{spec.repo_id} exists and is PUBLIC; refusing to upload")


def upload(spec: plan.ExportSpec, out_dir: Path, files_by_host: dict[str, list[str]], group) -> str:
    """Push a step to its private repo: the small files first, then one commit per node for its shards. Verifies
    the repo's files and every shard's sha256 against the local files; returns the head commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    world = get_world()
    api = HfApi()
    host = socket.gethostname()
    is_host_leader = world.local_rank == 0

    def push_small() -> None:
        if world.is_master:
            small = [p.name for p in sorted(out_dir.iterdir()) if not p.name.endswith(".safetensors")]
            hub(
                f"small-file commit to {spec.repo_id}",
                api.create_commit,
                spec.repo_id,
                operations=[CommitOperationAdd(name, out_dir / name) for name in small],
                commit_message=f"Model card, config, tokenizer and export report ({spec.name}, {spec.role})",
            )

    run_guarded(f"[{spec.name}] small-file commit to {spec.repo_id}", group, push_small)

    # Every node pushes its own shards' bytes concurrently; the commits that reference them go one by one.
    operations: list = []

    def preupload() -> None:
        if is_host_leader:
            operations.extend(CommitOperationAdd(name, out_dir / name) for name in files_by_host[host])
            hub(f"shard upload to {spec.repo_id}", api.preupload_lfs_files, spec.repo_id, additions=operations)

    run_guarded(f"[{spec.name}] shard upload", group, preupload)
    for turn in sorted(files_by_host):

        def commit() -> None:
            if is_host_leader and turn == host:
                hub(
                    f"shard commit to {spec.repo_id}",
                    api.create_commit,
                    spec.repo_id,
                    operations=operations,
                    commit_message=f"Weights: {len(operations)} shards ({spec.name}, {host})",
                )

        run_guarded(f"[{spec.name}] shard commit from {turn}", group, commit)

    head = [None]

    def verify() -> None:
        if not world.is_master:
            return
        expected = json.loads((out_dir / "export_report.json").read_text())["sha256"]
        local_small = {p.name for p in out_dir.iterdir() if not p.name.endswith(".safetensors")}
        info = hub(f"repo_info {spec.repo_id}", api.repo_info, spec.repo_id, files_metadata=False)
        remote_files = {s.rfilename for s in info.siblings or []}
        missing_files = sorted((set(expected) | local_small) - remote_files)
        extra_files = sorted(remote_files - set(expected) - local_small - ALLOWED_EXTRA_REPO_FILES)
        if missing_files or extra_files:
            raise ValueError(f"{spec.repo_id}: files missing {missing_files[:5]}, unexpected {extra_files[:5]}")
        remote = {
            path_info.path: path_info.lfs.sha256
            for path_info in hub(
                f"get_paths_info {spec.repo_id}", api.get_paths_info, spec.repo_id, list(expected), expand=True
            )
            if getattr(path_info, "lfs", None) is not None
        }
        wrong = sorted(name for name, digest in expected.items() if remote.get(name) != digest)
        if wrong:
            raise ValueError(f"{spec.repo_id}: {len(wrong)} shards missing or different: {wrong[:5]}")
        get_logger().success(f"[{spec.name}] all {len(expected)} shards on {spec.repo_id} match the local sha256")
        head[0] = info.sha

    run_guarded(f"[{spec.name}] verification of {spec.repo_id}", group, verify)
    dist.broadcast_object_list(head, src=0, group=group)
    return head[0]


def run(config: SFTConfig, parallel_dims: ParallelDims) -> None:
    logger = get_logger()
    world = get_world()
    group = dist.new_group(backend="gloo", timeout=GLOO_TIMEOUT)

    # Fail before building anything: the request, the checkpoints, the token, the repos and the base model.
    def read_request() -> list[plan.ExportSpec]:
        values, sources = plan.resolve_export_env(os.environ, getattr(config, "env_vars", None))
        if world.is_master:
            logger.info(f"GB_EXPORT_* request: {values} (from {sources})")
        return plan.parse_export_env(values)

    specs = run_guarded("GB_EXPORT_* check", group, read_request)
    if config.ckpt is None or config.ckpt.output_dir is None:
        raise ValueError("The export writes under [ckpt] output_dir (the run's directory), which is unset")

    def check_token() -> None:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            raise ValueError("No HF_TOKEN in the environment (pass -e ~/.config/prime-export/hf.env)")

    run_guarded("HF_TOKEN check", group, check_token)
    run_id = specs[0].run_id
    logger.info(f"Exporting run {run_id}: {[(s.step, s.role, s.repo_id) for s in specs]}")
    # Master only (each check reads a ~13 MB .metadata from the volume); the result goes to every rank.
    complete_box = [
        run_guarded(
            "checkpoint check",
            group,
            lambda: plan.check_checkpoints(specs, is_complete=dcp_is_complete) if world.is_master else None,
        )
    ]
    dist.broadcast_object_list(complete_box, src=0, group=group)
    complete = complete_box[0]
    logger.info(f"Complete checkpoints of {run_id}: {complete}")

    def check_base() -> dict[str, str]:
        weight_map: dict[str, str] = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())["weight_map"]
        untrained = sum(not is_trained(key) for key in weight_map)
        if (len(weight_map), untrained) != (EXPECTED_TENSORS, EXPECTED_UNTRAINED):
            raise ValueError(f"{BASE_DIR}: {len(weight_map)} tensors / {untrained} mtp.*, expected 42683 / 1040")
        for name in BASE_FILES + sorted(set(weight_map.values())):
            if not (BASE_DIR / name).is_file():
                raise FileNotFoundError(BASE_DIR / name)
        return weight_map

    weight_map = run_guarded("base model check", group, check_base)
    run_guarded("Hugging Face preflight", group, lambda: preflight_repos(specs) if world.is_master else None)

    out_root = config.ckpt.output_dir / "hf-export"
    report_path = config.ckpt.output_dir / "export_report.json"
    metrics_path = plan.RUNS_ROOT / run_id / "monitors" / "file" / "metrics.jsonl"
    evals: dict = {}
    if world.is_master:
        try:  # best effort: the evals only go into the report and the model card
            evals = plan.heldout_evals(metrics_path)
        except Exception as error:
            logger.warning(f"Could not read held-out evals from {metrics_path}: {error!r}")
        logger.info(f"Held-out evals found in {metrics_path}: {json.dumps(evals, sort_keys=True)}")
        best = plan.best_step(evals, complete)
        for spec in specs:
            if "best" in spec.role.split("+") and best is not None and best != spec.step:
                logger.warning(f"Step {spec.step} is exported as best, but the run's metrics rank step {best} first")

    files = sorted(set(weight_map.values()))
    sizes = {name: (BASE_DIR / name).stat().st_size for name in files}
    assignment = assign_files(sizes, world.world_size)
    my_files = assignment[world.rank]
    wanted = {key for key, name in weight_map.items() if name in my_files and is_trained(key)}
    num_trained = sum(is_trained(key) for key in weight_map)
    hosts: list[str | None] = [None] * world.world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    files_by_host: dict[str, list[str]] = {}
    for rank, names in enumerate(assignment):
        files_by_host.setdefault(hosts[rank], []).extend(names)
    make_dir(config.ckpt.output_dir, group)
    if world.local_rank == 0:
        # Logged, not enforced: the staging dir may be a mount whose reported free space means little.
        need = sum(sizes[name] for name in files_by_host[socket.gethostname()]) * len(specs)
        try:
            free = shutil.disk_usage(config.ckpt.output_dir).free
        except OSError as error:
            free = float("nan")
            logger.warning(f"disk_usage({config.ckpt.output_dir}) failed: {error!r}")
        level = logger.warning if free < need * 1.05 else logger.info
        level(
            f"{socket.gethostname()}: {need / 1e9:.1f} GB to write under {config.ckpt.output_dir}, {free / 1e9:.1f} GB free"
        )
    logger.info(f"Rank {world.rank}: files {my_files}; {len(wanted)} trained keys to fill")

    logger.info(f"Building the model ({config.model.name}) without loading HF weights")
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=True)

    report = {
        "run_id": run_id,
        "base": {"dir": str(BASE_DIR), "repo": BASE_REPO, "revision": BASE_REVISION},
        "requested": [{"step": s.step, "role": s.role, "repo": s.repo_id} for s in specs],
        "complete_checkpoints": complete,
        "heldout_evals": {str(step): metrics for step, metrics in sorted(evals.items())},
        "exports": [],
        "all_done": False,
    }
    run_guarded("report", group, lambda: plan.write_json(report_path, report) if world.is_master else None)

    for spec in specs:
        out_dir = out_root / spec.name
        dcp_dir = spec.dcp_dir()
        logger.info(f"[{spec.name}] loading model weights from {dcp_dir}")
        dcp_load(
            state_dict={"app": AppState(model, [], None, None)},
            checkpoint_id=dcp_dir,
            process_group=get_dcp_group(),
        )

        logger.info(f"[{spec.name}] gathering (fp32) and converting to HF names")
        trained = collect_hf_tensors(model, wanted)
        torch.cuda.empty_cache()

        def check_converted() -> None:
            if set(trained) != wanted:
                raise ValueError(f"missing converted keys {sorted(wanted - set(trained))[:5]}")

        run_guarded(f"[{spec.name}] conversion", group, check_converted)

        make_dir(out_dir, group)
        exported, fp32, copied, digests = run_guarded(
            f"[{spec.name}] writing shards", group, write_my_files, BASE_DIR, my_files, out_dir, trained
        )
        logger.info(f"[{spec.name}] rank {world.rank}: wrote {len(my_files)} files ({len(exported)} trained tensors)")

        gathered: list = [None] * world.world_size
        dist.all_gather_object(gathered, (exported, fp32, copied, digests), group=group)
        entry = None

        def finish_step() -> None:
            nonlocal entry
            if not world.is_master:
                return
            all_exported = {k: v for e, _, _, _ in gathered for k, v in e.items()}
            all_fp32 = {k: v for _, f, _, _ in gathered for k, v in f.items()}
            all_copied = sum(c for _, _, c, _ in gathered)
            all_digests = {name: d for _, _, _, ds in gathered for name, d in ds.items()}
            problems = plan.check_changes(all_exported, all_fp32, num_trained, MAX_RELATIVE_CHANGE)
            if all_copied != EXPECTED_UNTRAINED or len(all_digests) != len(files):
                problems.append(
                    f"{all_copied}/{EXPECTED_UNTRAINED} copied tensors, {len(all_digests)}/{len(files)} files"
                )
            heldout = evals.get(spec.step)
            entry = plan.step_entry(
                spec,
                commit=None,
                tensor_count=len(all_exported) + all_copied,
                swapped=len(all_exported),
                copied=all_copied,
                sha256=all_digests,
                exported_changes=all_exported,
                trained_changes=all_fp32,
                heldout=heldout,
            )
            detail = dict(entry, relative_change_per_tensor=dict(sorted(all_exported.items())))
            detail["relative_change_fp32_per_tensor"] = dict(sorted(all_fp32.items()))
            for name in BASE_FILES:
                shutil.copyfile(BASE_DIR / name, out_dir / name)
            (out_dir / "README.md").write_text(plan.model_card(spec, heldout, BASE_REPO, BASE_REVISION))
            plan.write_json(out_dir / "export_report.json", detail)
            logger.info(f"[{spec.name}] relative change (exported): {json.dumps(entry['relative_change']['groups'])}")
            logger.info(f"[{spec.name}] relative change (fp32): {json.dumps(entry['relative_change_fp32']['groups'])}")
            if problems:
                raise ValueError(f"export check FAILED, not uploading: {problems}")
            logger.success(f"[{spec.name}] {len(all_exported)} trained + {all_copied} copied tensors in {out_dir}")

        run_guarded(f"[{spec.name}] export check", group, finish_step)
        # Push before loading the next checkpoint, so a later failure cannot cost this step its upload.
        commit = upload(spec, out_dir, files_by_host, group)

        def record() -> None:
            if world.is_master:
                entry["commit"] = commit
                report["exports"].append(entry)
                plan.write_json(report_path, report)
                logger.success(plan.done_line(entry))
                # The same line without the logger's prefix and color codes, for whoever parses the log.
                print(plan.done_line(entry), flush=True)

        run_guarded(f"[{spec.name}] report", group, record)

    def finish() -> None:
        if world.is_master:
            report["all_done"] = True
            plan.write_json(report_path, report)
            logger.success(plan.ALL_DONE_MARKER)
            print(plan.ALL_DONE_MARKER, flush=True)

    run_guarded("final report", group, finish)
