"""Every model compiles each decoder layer as one graph (`fullgraph=True`) under the trainer's setup on 2 GPUs.

Each case builds a tiny seeded model with `setup_model` on its layout (activation checkpointing with offloading, then
torch.compile per layer, then FSDP, CP or EP) and runs one forward and backward. A case passes when the loss and grad
norm are finite, Dynamo counted no graph break, and Dynamo compiled exactly one graph per kind of decoder layer, so no
kind of layer runs eagerly or compiles twice (as PR #4609 fixed for layers whose inputs differed in their view base).
One `torchrun` launch runs every selected case, and each case reports as its own test. A case that crashes or hangs the
launch fails, and a relaunch runs the cases it did not reach.

Run cases without pytest by case id, or every case if none is given (each rank writes `rank<N>.json` around each case):

    uv run torchrun --standalone --nproc-per-node=2 tests/unit/train/test_compile.py <output_dir> <case>...
"""

import functools
import json
import math
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tiny_models import (
    TINY_MODELS,
    TINY_PRIME_RL_MODELS,
    TOKENIZER,
    Case,
    compute_loss,
    load_rows,
    model_config,
    save_tiny_model,
)
from torch._dynamo.utils import clear_compilation_metrics, counters, get_compilation_metrics

import prime_rl._compat  # noqa: F401
from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import setup_model, setup_tokenizer
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims
from prime_rl.trainer.utils import clip_grad_norm_, setup_torch_distributed
from prime_rl.trainer.vlm import get_language_model
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel
from prime_rl.utils.logger import setup_logger

NUM_RANKS = 2
PROCESS_GROUP_TIMEOUT = timedelta(minutes=5)
LAUNCH_SECONDS = 5 * 60
"""A launch's budget for imports, process group setup, building the models and loading the rows."""
CASE_SECONDS = 3 * 60
"""Each case's budget for a cold compile and its step. The cases of a launch share their budgets."""
TINY_CONFIGS = TINY_MODELS | TINY_PRIME_RL_MODELS


def needs_sm100() -> str | None:
    return None if torch.cuda.get_device_capability()[0] == 10 else "MXFP8 needs an SM100 GPU"


def needs_deepgemm_gpu() -> str | None:
    return None if torch.cuda.get_device_capability()[0] in (9, 10) else "DeepGEMM FP8 needs an SM90 or SM100 GPU"


@dataclass(frozen=True)
class CompileCase(Case):
    compile: bool = True
    fullgraph: bool = True
    requirement: Callable[[], str | None] | None = None
    """Returns why this GPU or environment cannot run the case, or None if it can."""

    def skip_reason(self) -> str | None:
        if self.name in KNOWN_FAILURES:
            return f"fails fullgraph today: {KNOWN_FAILURES[self.name]}"
        return None if self.requirement is None else self.requirement()


CP_MODELS = [
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_8_flash_next",
    "glm5_next",
    "glm_moe_dsa",
    "deepseek_v4",
    "gpt_oss",
    "nemotron_h",
    "afmoe",
]
CASES = [
    *(CompileCase(f"{model}-dp2", model) for model in TINY_CONFIGS),
    CompileCase("nemotron_h-dp2-token_chunk", "nemotron_h", overrides={"ac": {"token_chunk_size": 64}}),
    *(CompileCase(f"{model}-cp2", model, cp=2) for model in CP_MODELS),
    CompileCase("qwen3_moe-ep2-torch-bf16", "qwen3_moe", ep=2),
    CompileCase(
        "qwen3_moe-ep2-torch-sonicmoe",
        "qwen3_moe",
        ep=2,
        overrides={"moe": {"compute": {"type": "bf16", "backend": "sonicmoe"}}},
    ),
    CompileCase(
        "qwen3_moe-ep2-torch-deepgemm_fp8",
        "qwen3_moe",
        ep=2,
        overrides={"moe": {"compute": {"type": "deepgemm_fp8"}}},
        requirement=needs_deepgemm_gpu,
    ),
    CompileCase(
        "qwen3_moe-ep2-torch-mxfp8",
        "qwen3_moe",
        ep=2,
        overrides={"moe": {"compute": {"type": "mxfp8"}}},
        requirement=needs_sm100,
    ),
    CompileCase(
        "qwen3_moe-ep2-torch_mxfp8-mxfp8",
        "qwen3_moe",
        ep=2,
        overrides={"moe": {"dispatch": {"type": "torch", "transport": "mxfp8"}, "compute": {"type": "mxfp8"}}},
        requirement=needs_sm100,
    ),
    CompileCase(
        "qwen3_moe-ep2-deepep-bf16",
        "qwen3_moe",
        ep=2,
        overrides={"moe": {"dispatch": {"type": "deepep"}}},
    ),
]


INKLING2_LAYER_KINDS_UNCOUNTED = "layer_kinds counts 2 kinds for Inkling 2's 4 distinct decoder layer graphs"
KNOWN_FAILURES = {"inkling2-dp2": INKLING2_LAYER_KINDS_UNCOUNTED}


def isolated(run, *args) -> dict:
    """The result of `run`, or the error it raised, so that one failing case does not hide the others."""
    try:
        return run(*args)
    except Exception as error:
        traceback.print_exc()
        return {"error": "".join(traceback.format_exception_only(error)).strip()}


def layer_kinds(model: torch.nn.Module) -> set[tuple[tuple[str, ...], str | None]]:
    """The decoder layers' distinct pairs of module classes and config layer type. Layers of different kinds cannot
    share a compiled graph."""
    language_model = get_language_model(model)
    layers = [getattr(layer, "_checkpoint_wrapped_module", layer) for layer in language_model.layers]
    layer_types = getattr(language_model.config, "layer_types", None) or [None] * len(layers)
    return {
        (tuple(sorted({type(module).__name__ for module in layer.modules()})), layer_type)
        for layer, layer_type in zip(layers, layer_types, strict=True)
    }


def run_step(config: ModelConfig, parallel_dims: ParallelDims, rows: list[dict], progress: dict) -> dict:
    """One forward and backward of this rank's row through the trainer's model, with its summed loss and grad norm.
    `progress` holds the stage reached and the number of decoder layer kinds, also when the step raises."""
    progress["stage"] = "setup"
    model = setup_model(config, parallel_dims)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config, parallel_dims)
    progress["layer_kinds"] = len(layer_kinds(model))
    progress["stage"] = "step"
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if parallel_dims.cp_enabled else 0
    row = rows[parallel_dims.get_mesh("dp").get_local_rank()]
    with maybe_activation_offloading(config.ac_offloading, model):
        loss, _ = compute_loss(model, row, cp_rank, parallel_dims.cp)
    loss.backward()
    grad_norm = clip_grad_norm_(None, model, max_norm=math.inf)
    return {"loss_sum": loss.item(), "grad_norm": grad_norm.item()}


def failures(result: dict) -> list[str]:
    if "skipped" in result:
        return []
    found = [result["error"]] if "error" in result else []
    for metric in ("loss_sum", "grad_norm"):
        if metric in result and not math.isfinite(result[metric]):
            found.append(f"{metric} is {result[metric]}")
    num_graphs = sum(bool(frame["ops"]) for frame in result.get("frames", []))
    if "error" not in result and num_graphs != result["layer_kinds"]:
        found.append(f"{num_graphs} compiled graphs for {result['layer_kinds']} decoder layer kinds")
    if result.get("graph_breaks"):
        found.append(f"graph breaks: {result['graph_breaks']}")
    return found


def compile_case(config: ModelConfig, parallel_dims: ParallelDims, rows: list[dict]) -> dict:
    torch._dynamo.reset()
    counters.clear()
    clear_compilation_metrics()
    start = time.perf_counter()
    progress = {}
    result = isolated(run_step, config, parallel_dims, rows, progress)
    result.update(progress)
    result["seconds"] = time.perf_counter() - start
    result["graph_breaks"] = dict(counters["graph_break"])
    frame_compiles = [
        metrics
        for metrics in get_compilation_metrics()
        if metrics.co_name is not None and metrics.is_forward is not False
    ]
    result["frames"] = [
        {
            "code": f"{metrics.co_name} ({Path(metrics.co_filename).name}:{metrics.co_firstlineno})",
            "ops": metrics.graph_op_count,
        }
        for metrics in frame_compiles
    ]
    result["recompiles"] = [metrics.recompile_reason for metrics in frame_compiles if metrics.recompile_reason]
    torch.cuda.empty_cache()
    return result


def select_cases(names: list[str]) -> list[CompileCase]:
    by_name = {case.name: case for case in CASES}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown case ids: {unknown}")
    return [by_name[name] for name in names] if names else CASES


def in_run_order(cases: list[CompileCase]) -> list[CompileCase]:
    """CP cases run last, since CP setup patches attention classes for the rest of the process."""
    return sorted(cases, key=lambda case: case.cp)


def save_results(path: Path, results: dict) -> None:
    partial_path = path.with_suffix(".partial")
    partial_path.write_text(json.dumps(results, indent=2))
    partial_path.replace(path)


def main(output_dir: Path, cases: list[CompileCase]) -> None:
    setup_logger("warning")
    models_dir = output_dir / "models"
    configs = {case.name: model_config(case, models_dir) for case in cases}
    skip_reasons = {case.name: case.skip_reason() for case in cases}
    runnable = [case for case in cases if skip_reasons[case.name] is None]
    setup_torch_distributed(timeout=PROCESS_GROUP_TIMEOUT)
    built = {name: isolated(save_tiny_model, name, models_dir) for name in sorted({case.model for case in runnable})}
    dist.barrier()
    rows = load_rows(setup_tokenizer(TokenizerConfig(name=TOKENIZER), model_name=TOKENIZER))

    results_path = output_dir / f"rank{get_world().rank}.json"
    results, layouts = {}, {}
    for case in in_run_order(cases):
        results[case.name] = {"stage": "started"}
        save_results(results_path, results)
        if skip_reasons[case.name] is not None:
            result = {"skipped": skip_reasons[case.name]}
        elif "error" in built[case.model]:
            result = {**built[case.model], "stage": "build"}
        else:
            if (case.cp, case.ep) not in layouts:
                layouts[case.cp, case.ep] = get_parallel_dims(configs[case.name])
            result = compile_case(configs[case.name], layouts[case.cp, case.ep], rows)
        result["failures"] = failures(result)
        status = "SKIP" if "skipped" in result else "FAIL" if result["failures"] else "PASS"
        print(f"[rank {get_world().rank}] {status} {case.name}: {json.dumps(result)}", flush=True)
        results[case.name] = result
        save_results(results_path, results)
    dist.barrier()
    if get_world().is_master:
        shutil.rmtree(models_dir)


def launch(names: list[str], run_process, output_dir: Path) -> dict:
    """Each rank's results. A case left started when a launch exits crashed or hung it: it fails, and a relaunch runs
    the cases without a result, so every relaunch has at least one case fewer."""
    cmd = ["uv", "run", "torchrun", "--standalone", f"--nproc-per-node={NUM_RANKS}", __file__, output_dir.as_posix()]
    ranks = [{} for _ in range(NUM_RANKS)]
    while names:
        timeout = LAUNCH_SECONDS + CASE_SECONDS * len(names)
        start = time.monotonic()
        returncode = run_process([*cmd, *names], timeout=timeout).returncode
        exit_cause = (
            f"timed out after {timeout} s" if time.monotonic() - start >= timeout else f"exited with code {returncode}"
        )
        for rank, results in enumerate(ranks):
            rank_path = output_dir / f"rank{rank}.json"
            if rank_path.exists():
                results.update(json.loads(rank_path.read_text()))
                rank_path.unlink()
        started = [name for name in names if any(results.get(name) == {"stage": "started"} for results in ranks)]
        if not started:
            break
        for results in ranks:
            if "failures" not in results.get(started[0], {}):
                results[started[0]] = {"failures": [f"the launch {exit_cause} during this case"]}
        names = [name for name in names if not all("failures" in results.get(name, {}) for results in ranks)]
    models_dir = output_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    return {"returncode": returncode, "ranks": ranks}


@pytest.fixture(scope="module")
def launch_results(request: pytest.FixtureRequest, run_process, tmp_path_factory) -> Callable[[], dict]:
    """Launches the selected, runnable cases on the first request."""
    callspecs = [getattr(item, "callspec", None) for item in request.session.items]
    selected = {
        callspec.params["case"].name
        for callspec in callspecs
        if callspec is not None and isinstance(callspec.params.get("case"), CompileCase)
    }
    names = [case.name for case in in_run_order(CASES) if case.name in selected and case.skip_reason() is None]
    return functools.cache(lambda: launch(names, run_process, tmp_path_factory.mktemp("compile")))


def case_failures(launch_result: dict, case: CompileCase) -> list[str]:
    found = []
    for rank, results in enumerate(launch_result["ranks"]):
        if case.name not in results:
            found.append(f"rank {rank}: not run, the launch exited with code {launch_result['returncode']} first")
        else:
            found += [f"rank {rank}: {failure}" for failure in results[case.name]["failures"]]
    return found


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_compiles_fullgraph(launch_results: Callable[[], dict], case: CompileCase):
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"needs {NUM_RANKS} GPUs")
    skip_reason = case.skip_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)
    found = case_failures(launch_results(), case)
    assert not found, "\n".join(found)


if __name__ == "__main__":
    main(Path(sys.argv[1]), select_cases(sys.argv[2:]))
