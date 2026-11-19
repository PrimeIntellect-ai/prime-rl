"""Every model compiles each decoder layer as one graph (`fullgraph=True`) under the trainer's setup on 2 GPUs.

Each case builds a tiny seeded model with `setup_model` on its layout (activation checkpointing with offloading, then
torch.compile per layer, then FSDP, CP or EP) and runs one forward and backward. A case passes when the loss and grad
norm are finite, Dynamo counted no graph break, and Dynamo compiled at least one graph per kind of decoder layer, so no
kind of layer runs eagerly. One `torchrun` launch runs every selected case of a group; each case reports as its own
test.

Run cases without pytest, by case id or group name (each rank writes `rank<N>.json` after every case):

    uv run torchrun --standalone --nproc-per-node=2 tests/unit/train/test_compile.py <output_dir> <case or group>...
"""

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
from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims
from prime_rl.trainer.utils import clip_grad_norm_, setup_torch_distributed
from prime_rl.trainer.vlm import get_language_model
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import setup_context_parallel
from prime_rl.utils.logger import setup_logger

NUM_RANKS = 2
PROCESS_GROUP_TIMEOUT = timedelta(minutes=5)
TINY_CONFIGS = TINY_MODELS | TINY_PRIME_RL_MODELS
WRAPPER_MODEL_TYPES = {"glm5_next", "muse_glimmer", "nemotron_h_omni", "qwen3_5", "qwen3_5_moe", "qwen4_exp"}


def needs_sm100() -> str | None:
    return None if torch.cuda.get_device_capability()[0] == 10 else "MXFP8 needs an SM100 GPU"


def needs_deepgemm_gpu() -> str | None:
    return None if torch.cuda.get_device_capability()[0] in (9, 10) else "DeepGEMM FP8 needs an SM90 or SM100 GPU"


@dataclass(frozen=True)
class CompileCase(Case):
    compile: bool = True
    fullgraph: bool = True
    group: str = "dp2"
    """Cases of one group share a torchrun launch."""
    requirement: Callable[[], str | None] | None = None
    """Returns why this GPU or environment cannot run the case, or None if it can."""

    def skip_reason(self) -> str | None:
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
    *(CompileCase(f"{model}-cp2", model, cp=2, group="cp2") for model in CP_MODELS),
    CompileCase("qwen3_moe-ep2-torch-bf16", "qwen3_moe", ep=2, group="ep2_torch"),
    CompileCase(
        "qwen3_moe-ep2-torch-sonicmoe",
        "qwen3_moe",
        ep=2,
        group="ep2_torch",
        overrides={"moe": {"compute": {"type": "bf16", "backend": "sonicmoe"}}},
    ),
    CompileCase(
        "qwen3_moe-ep2-torch-deepgemm_fp8",
        "qwen3_moe",
        ep=2,
        group="ep2_torch",
        overrides={"moe": {"compute": {"type": "deepgemm_fp8"}}},
        requirement=needs_deepgemm_gpu,
    ),
    CompileCase(
        "qwen3_moe-ep2-torch-mxfp8",
        "qwen3_moe",
        ep=2,
        group="ep2_torch",
        overrides={"moe": {"compute": {"type": "mxfp8"}}},
        requirement=needs_sm100,
    ),
    CompileCase(
        "qwen3_moe-ep2-torch_mxfp8-mxfp8",
        "qwen3_moe",
        ep=2,
        group="ep2_torch",
        overrides={"moe": {"dispatch": {"type": "torch", "transport": "mxfp8"}, "compute": {"type": "mxfp8"}}},
        requirement=needs_sm100,
    ),
    CompileCase(
        "qwen3_moe-ep2-deepep-bf16",
        "qwen3_moe",
        ep=2,
        group="ep2_deepep",
        overrides={"moe": {"dispatch": {"type": "deepep"}}},
    ),
]
GROUPS = {
    group: [case for case in CASES if case.group == group] for group in dict.fromkeys(case.group for case in CASES)
}
TIMEOUTS = {"dp2": 30 * 60, "cp2": 20 * 60, "ep2_torch": 15 * 60, "ep2_deepep": 10 * 60}


def isolated(run, *args) -> dict:
    """The result of `run`, or the error it raised, so that one failing case does not hide the others."""
    try:
        return run(*args)
    except Exception as error:
        traceback.print_exc()
        return {"error": "".join(traceback.format_exception_only(error)).strip()}


def layer_kinds(model: torch.nn.Module) -> set[tuple[str, ...]]:
    """The decoder layers' distinct sets of module classes. Layers of different kinds cannot share a compiled graph."""
    layers = [getattr(layer, "_checkpoint_wrapped_module", layer) for layer in get_language_model(model).layers]
    return {tuple(sorted({type(module).__name__ for module in layer.modules()})) for layer in layers}


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
    if "error" not in result and num_graphs < result["layer_kinds"]:
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
    by_name = {case.name: [case] for case in CASES} | GROUPS
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown case ids or groups: {unknown}. Groups: {list(GROUPS)}")
    return [case for name in names for case in by_name[name]]


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
    cp_cases_last = sorted(cases, key=lambda case: case.cp)
    for case in cp_cases_last:
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
        partial_path = results_path.with_suffix(".partial")
        partial_path.write_text(json.dumps(results, indent=2))
        partial_path.replace(results_path)
    dist.barrier()
    if get_world().is_master:
        shutil.rmtree(models_dir)


def launch(group: str, names: list[str], run_process, tmp_path_factory: pytest.TempPathFactory) -> dict:
    output_dir = tmp_path_factory.mktemp(f"compile-{group}")
    cmd = ["uv", "run", "torchrun", "--standalone", f"--nproc-per-node={NUM_RANKS}", __file__, output_dir.as_posix()]
    returncode = run_process([*cmd, *names], timeout=TIMEOUTS[group]).returncode
    rank_paths = [output_dir / f"rank{rank}.json" for rank in range(NUM_RANKS)]
    return {
        "returncode": returncode,
        "ranks": [json.loads(path.read_text()) if path.exists() else {} for path in rank_paths],
    }


@pytest.fixture(scope="module")
def group_results(request: pytest.FixtureRequest, run_process, tmp_path_factory) -> Callable[[str], dict]:
    """Launches a group's selected, runnable cases on the first request for that group."""
    callspecs = [getattr(item, "callspec", None) for item in request.session.items]
    selected = {
        callspec.params["case"].name
        for callspec in callspecs
        if callspec is not None and isinstance(callspec.params.get("case"), CompileCase)
    }
    launches = {}

    def results(group: str) -> dict:
        if group not in launches:
            names = [case.name for case in GROUPS[group] if case.name in selected and case.skip_reason() is None]
            launches[group] = launch(group, names, run_process, tmp_path_factory)
        return launches[group]

    return results


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
def test_compiles_fullgraph(group_results: Callable[[str], dict], case: CompileCase):
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"needs {NUM_RANKS} GPUs")
    skip_reason = case.skip_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)
    found = case_failures(group_results(case.group), case)
    assert not found, "\n".join(found)


def test_every_text_architecture_has_a_case():
    registered = set(_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE) - WRAPPER_MODEL_TYPES
    covered = {TINY_CONFIGS[case.model].model_type for case in GROUPS["dp2"]}
    missing = sorted(registered - covered)
    assert not missing, (
        f"Missing tiny config for model types {missing}: add one to tests/unit/train/tiny_models.py, which gives it "
        "a dp2 case here, and a cp2 case if the model has its own context-parallel code path."
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), select_cases(sys.argv[2:]))
