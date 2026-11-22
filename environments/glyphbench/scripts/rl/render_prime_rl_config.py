#!/usr/bin/env python3
"""Materialize a run-specific Prime-RL v0.9 GlyphBench config."""

from __future__ import annotations

import argparse
import copy
import re
import tomllib
from pathlib import Path

import tomli_w
from prime_rl.configs.rl import RLConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/rl/qwen35-4b-glyphbench"
SINGLE_CONFIG = CONFIG_DIR / "single_task.toml"
MULTITASK_CONFIG = CONFIG_DIR / "multitask.toml"
PILOT_TASKS = CONFIG_DIR / "multitask_pilot_tasks.txt"
GROUP_SIZES = (8, 16, 32, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("single", "multitask"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--eval-examples",
        type=int,
        help="held-out examples (single) or fixed seeds per panel task (multitask)",
    )
    parser.add_argument("--task", default="glyphbench/classics-snake-easy-v0")
    parser.add_argument(
        "--task-file",
        type=Path,
        help="custom multitask training manifest; evaluation uses the same tasks",
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--capture-traces",
        action="store_true",
        help="enable Prime-RL's high-volume local episode JSONL monitor",
    )
    parser.add_argument(
        "--raw-inference-metrics",
        action="store_true",
        help="export the full per-replica vLLM Prometheus surface",
    )
    parser.add_argument("--group-size", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="total multitask rollouts per optimizer step (must divide by group size)",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--max-inflight",
        type=int,
        help="cap adaptive rollout concurrency (must be at least group size)",
    )
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="disable periodic and final training-state checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="optimizer-step interval for resume-capable checkpoints",
    )
    parser.add_argument("--infer-nodes", type=int)
    parser.add_argument("--train-nodes", type=int)
    parser.add_argument(
        "--single-node",
        action="store_true",
        help="render a local single-node config instead of a Slurm multi-node config",
    )
    parser.add_argument("--train-gpus", type=int, default=4)
    parser.add_argument("--infer-gpus", type=int, default=4)
    parser.add_argument("--slurm-time")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tag", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_size is None:
        args.group_size = 512 if args.mode == "single" else 64
    if args.group_size not in GROUP_SIZES:
        choices = ", ".join(str(size) for size in GROUP_SIZES)
        raise ValueError(f"--group-size for {args.mode} must be one of: {choices}")
    if args.no_eval and (args.eval_examples is not None or args.eval_interval is not None):
        raise ValueError("--no-eval cannot be combined with eval overrides")
    if args.no_checkpoint and args.checkpoint_interval is not None:
        raise ValueError("--no-checkpoint cannot be combined with --checkpoint-interval")
    if args.task_file is not None and args.mode != "multitask":
        raise ValueError("--task-file is only valid for multitask")
    if args.task_file is not None and args.pilot:
        raise ValueError("--task-file cannot be combined with --pilot")
    if args.batch_size is not None and args.mode != "multitask":
        raise ValueError("--batch-size is only valid for multitask")
    if args.single_node and any(
        value is not None for value in (args.infer_nodes, args.train_nodes, args.slurm_time)
    ):
        raise ValueError("--single-node cannot be combined with node-count or Slurm overrides")
    if args.no_wandb and any(
        value is not None for value in (args.wandb_project, args.wandb_group)
    ):
        raise ValueError("--no-wandb cannot be combined with W&B overrides")
    base = SINGLE_CONFIG if args.mode == "single" else MULTITASK_CONFIG
    with base.open("rb") as handle:
        config = tomllib.load(handle)

    config["run"]["name"] = args.run_name
    config["slurm"]["job_name"] = _slurm_name(args.run_name)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("max_steps must be positive")
        config["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        config["trainer"]["optim"]["lr"] = args.learning_rate
    if args.max_inflight is not None:
        if args.max_inflight < args.group_size:
            raise ValueError("--max-inflight must be at least --group-size")
        config["orchestrator"].setdefault("concurrency", {})["max_inflight"] = args.max_inflight
    if args.eval_interval is not None:
        if args.eval_interval < 1:
            raise ValueError("eval_interval must be positive")
        config["orchestrator"]["eval"]["interval"] = args.eval_interval
    if args.no_eval:
        config["orchestrator"].pop("eval")
    if args.no_checkpoint:
        config.pop("ckpt", None)
    elif args.checkpoint_interval is not None:
        if args.checkpoint_interval < 1:
            raise ValueError("--checkpoint-interval must be positive")
        config["ckpt"]["interval"] = args.checkpoint_interval
    if args.capture_traces:
        config["trainer"]["monitors"]["file"] = {}
        config["orchestrator"]["monitors"]["file"] = {}
    if args.raw_inference_metrics:
        config["orchestrator"]["collect_inference_metrics"] = True
    deployment = config["deployment"]
    if args.single_node:
        if args.train_gpus < 1 or args.infer_gpus < 1:
            raise ValueError("--train-gpus and --infer-gpus must be positive")
        deployment = config["deployment"] = {
            "type": "single_node",
            "gpus_per_node": args.train_gpus + args.infer_gpus,
            "num_train_gpus": args.train_gpus,
            "num_infer_gpus": args.infer_gpus,
        }
        config.pop("slurm")
        inference = config["inference"]["vllm"]
        inference["data_parallel_size"] = args.infer_gpus
        inference["data_parallel_size_local"] = args.infer_gpus
        inference["api_server_count"] = args.infer_gpus
    elif args.infer_nodes is not None:
        if args.infer_nodes < 1:
            raise ValueError("--infer-nodes must be positive")
        deployment["num_infer_nodes"] = args.infer_nodes
    if args.train_nodes is not None:
        if args.train_nodes < 1:
            raise ValueError("--train-nodes must be positive")
        deployment["num_train_nodes"] = args.train_nodes
    if args.slurm_time is not None:
        config["slurm"]["time"] = args.slurm_time
    if args.no_wandb:
        config["monitors"].pop("wandb", None)
    elif args.wandb_project is not None:
        config["monitors"]["wandb"]["project"] = args.wandb_project
    if args.wandb_group is not None:
        config["monitors"]["wandb"]["group"] = args.wandb_group
    if not args.no_wandb:
        config["monitors"]["wandb"]["tags"].extend(args.wandb_tag)

    train = config["orchestrator"]["train"]["source"]
    evaluation = config["orchestrator"]["eval"]["source"] if not args.no_eval else None
    if args.mode == "single":
        if args.pilot:
            raise ValueError("--pilot is only valid for multitask")
        config["orchestrator"]["batch_size"] = args.group_size
        config["orchestrator"]["group_size"] = args.group_size
        train[0]["name"] = f"train-{_task_slug(args.task)}"
        train[0]["env"]["taskset"]["tasks"] = [args.task]
        if evaluation is not None:
            evaluation[0]["name"] = f"eval-{_task_slug(args.task)}-fixed"
            evaluation[0]["env"]["taskset"]["tasks"] = [args.task]
        if not args.no_wandb:
            config["monitors"]["wandb"]["tags"].append(_task_slug(args.task))
        topology = f"1x{args.group_size}"
    else:
        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            config["orchestrator"]["batch_size"] = args.batch_size
        config["orchestrator"]["group_size"] = args.group_size
        train[0]["group_size"] = args.group_size
        batch_size = config["orchestrator"]["batch_size"]
        if batch_size % args.group_size:
            raise ValueError("--batch-size must be divisible by --group-size")
        topology = f"{batch_size // args.group_size}x{args.group_size}"
        if args.pilot:
            train[0]["name"] = "train-pilot"
            train[0]["env"]["taskset"]["task_file"] = str(PILOT_TASKS.relative_to(ROOT))
            if not args.no_wandb:
                config["monitors"]["wandb"]["tags"].remove("craftax100")
                config["monitors"]["wandb"]["tags"].append("pilot")
        elif args.task_file is not None:
            task_file = _repo_relative_path(args.task_file)
            task_slug = _task_slug(task_file.stem)
            train[0]["name"] = f"train-{task_slug}"
            train[0]["env"]["taskset"]["task_file"] = str(task_file)
            if not args.no_wandb:
                config["monitors"]["wandb"]["tags"] = [
                    tag for tag in config["monitors"]["wandb"]["tags"] if tag != "craftax100"
                ]
                config["monitors"]["wandb"]["tags"].append(task_slug)
            if evaluation is not None:
                evaluation[0]["name"] = f"eval-{task_slug}-fixed"
                evaluation[0]["env"]["taskset"]["task_file"] = str(task_file)

    if args.eval_examples is not None:
        if args.eval_examples < 1:
            raise ValueError("eval_examples must be positive")
        evaluation[0]["env"]["taskset"]["num_seeds"] = args.eval_examples
    if evaluation is not None:
        eval_taskset = evaluation[0]["env"]["taskset"]
        if args.mode == "single":
            multiplier = 1
        else:
            eval_task_file = ROOT / eval_taskset["task_file"]
            multiplier = _manifest_size(eval_task_file)
        config["orchestrator"]["eval"]["num_examples"] = eval_taskset["num_seeds"] * multiplier

    if not args.no_wandb:
        config["monitors"]["wandb"]["tags"] = [
            tag
            for tag in config["monitors"]["wandb"]["tags"]
            if not re.fullmatch(r"\d+x\d+", tag) and not re.fullmatch(r"\d+-step", tag)
        ]
        config["monitors"]["wandb"]["tags"].extend(
            [topology, f"{config['max_steps']}-step"]
        )
        config["monitors"]["wandb"]["tags"] = [
            tag
            for tag in config["monitors"]["wandb"]["tags"]
            if not re.fullmatch(r"(?:\d+i\d+t|\d+igpu\d+tgpu)", tag)
        ]
        hardware = (
            f"{deployment['num_infer_gpus']}igpu{deployment['num_train_gpus']}tgpu"
            if args.single_node
            else f"{deployment['num_infer_nodes']}i{deployment['num_train_nodes']}t"
        )
        config["monitors"]["wandb"]["tags"].append(hardware)

    _assert_training_contract(
        config,
        expected_seq_len=65_536,
        multitask=args.mode == "multitask",
        allow_raw_inference_metrics=args.raw_inference_metrics,
        expect_eval=not args.no_eval,
        single_node=args.single_node,
    )
    # Prime's config validators normalize the TOML "None" sentinel in place.
    # Validate a copy so the hand-written TOML remains serializable.
    RLConfig.model_validate(copy.deepcopy(config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tomli_w.dumps(config))
    print(args.output)


def _assert_training_contract(
    config: dict,
    *,
    expected_seq_len: int,
    multitask: bool,
    allow_raw_inference_metrics: bool = False,
    expect_eval: bool = True,
    single_node: bool = False,
) -> None:
    assert config["seq_len"] == expected_seq_len
    assert config["trainer"]["loss"]["type"] == "ipo"
    expected_ac = (
        {"mode": "full", "freq": 1}
        if multitask
        else {
            "mode": "selective",
            "freq": 1,
            "targets": ["norm", "attn_proj", "mlp", "linear_attn"],
        }
    )
    assert config["trainer"]["model"]["ac"] == expected_ac
    assert config["trainer"]["model"]["compile"] == "None"
    assert config["orchestrator"]["renderer"] == {
        "name": "qwen3.5",
        "enable_thinking": True,
        "thinking_retention": "all",
    }
    assert config["orchestrator"]["train"]["sampling"]["max_completion_tokens"] == 4096
    assert config["inference"]["vllm"]["max_model_len"] == expected_seq_len
    if "wandb" in config["monitors"]:
        assert config["monitors"]["wandb"]["offline"] is False
    assert config["orchestrator"]["collect_inference_metrics"] is allow_raw_inference_metrics
    # Full token-level traces can exceed 500 MB per 512-rollout step. They are
    # opt-in through --capture-traces for bounded diagnostics, never a
    # production default.
    for component in ("trainer", "orchestrator"):
        assert config[component]["monitors"]["file"] in ("None", {})
    if single_node:
        assert config["deployment"]["type"] == "single_node"
        assert config["deployment"]["num_train_gpus"] >= 1
        assert config["deployment"]["num_infer_gpus"] >= 1
        assert "slurm" not in config
    else:
        assert config["deployment"]["num_train_nodes"] >= 1
        assert config["deployment"]["num_infer_nodes"] >= 1
        assert config["deployment"]["num_infer_replicas"] == 1
        assert config["deployment"]["orchestrator_on_inference"] is True
    batch_size = config["orchestrator"]["batch_size"]
    assert batch_size is not None
    assert batch_size % config["orchestrator"]["group_size"] == 0
    if not multitask:
        assert batch_size == config["orchestrator"]["group_size"]
    assert config["orchestrator"]["max_off_policy_steps"] <= 3
    if not single_node:
        assert config["slurm"]["cleanup_grace_period"] == 120
    assert ("eval" in config["orchestrator"]) is expect_eval
    phases = ("train", "eval") if expect_eval else ("train",)
    for phase in phases:
        for source in config["orchestrator"][phase]["source"]:
            taskset = source["env"]["taskset"]
            assert taskset["use_memory"] is False
            assert "n_frames" not in taskset
            assert taskset["max_output_tokens"] == 4096
            player = source["env"]["player"]
            assert player["harness"]["id"] == "glyphbench"
            assert "max_turns" not in player
            assert "max_output_tokens" not in player
            if multitask:
                # Stop between turns one token before the engine boundary. The
                # resulting max_total_tokens stop condition is a truncation,
                # rather than a superficially successful native-horizon exit.
                assert player["max_total_tokens"] == expected_seq_len - 1
            else:
                assert "max_total_tokens" not in player
    assert all(
        source["env"]["taskset"]["mixed_seeds"] is True
        for source in config["orchestrator"]["train"]["source"]
    )


def _task_slug(task_id: str) -> str:
    name = task_id.rsplit("/", 1)[-1].removesuffix("-v0")
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


def _manifest_size(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text().splitlines()
    )


def _repo_relative_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ValueError(f"task manifest does not exist: {resolved}")
    try:
        return resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"task manifest must be inside the repository: {resolved}") from exc


def _slurm_name(run_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name)[:80]


if __name__ == "__main__":
    main()
