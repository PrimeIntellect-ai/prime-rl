"""Asynchronous eval worker for pretraining runs: on its own GPUs, it scores the newest model snapshot of a
running training run (`[eval_snapshots]` in the run's config) on the eval suite, logs the results into the run's
W&B run and `metrics.jsonl` at the snapshot's step, deletes the snapshot and every older one, and repeats until
it has scored the final step. Started with torchrun on the eval GPUs:

    torchrun --nproc-per-node 8 -m prime_rl.trainer.pretrain.eval_worker --run-dir <run dir> --data-dir <sets> --sets <names>

The model is built once with the trainer's code (`setup_model` from the run's resolved config) and kept resident;
each round loads only the weights. Rows of every set are scored data-parallel over the worker's GPUs."""

import asyncio
import json
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from transformers import AutoTokenizer

from prime_rl import monitors
from prime_rl.configs.pretrain import EvalWorkerConfig, PretrainConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.pretrain.snapshots import delete_snapshots_up_to, finished_snapshot_steps, snapshot_dir
from prime_rl.trainer.pretrain.suite import prepare_set, score_requests, summarize, use_scoring_head
from prime_rl.trainer.utils import get_dcp_group, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


def run_config(config: EvalWorkerConfig) -> PretrainConfig:
    """The training run's resolved config (waiting for the launcher to write it), with the parallelism of the
    worker: plain FSDP over its GPUs."""
    path = config.run_dir / "configs/latest/resolved/pretrain.json"
    while not path.exists():
        time.sleep(config.poll_seconds)
    run = PretrainConfig.model_validate_json(path.read_text())
    run.model.compile = None  # packed rows of varying length would recompile at every forward
    run.model.ep = 1
    run.model.cp = 1
    run.model.dp_replicate = 1
    return run


def broadcast(value: int) -> int:
    values = [value]
    dist.broadcast_object_list(values, src=0, group=get_dcp_group())
    return values[0]


@clean_exit
def main_loop(config: EvalWorkerConfig) -> None:
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    run = run_config(config)
    setup_torch_distributed(timeout=timedelta(seconds=run.dist_timeout_seconds))
    torch.set_float32_matmul_precision(run.matmul_precision)

    asyncio.run(
        monitors.setup(
            producer="eval",
            wandb=run.monitors.wandb,
            file=run.monitors.file,
            output_dir=config.run_dir,
            run_config=run,
            overview_flavor="sft",
        )
    )

    model = setup_model(run.model, get_parallel_dims(run.model), loading_from_checkpoint_later=True)
    model.eval()
    use_scoring_head(model)
    tokenizer = AutoTokenizer.from_pretrained(run.tokenizer.name, trust_remote_code=run.tokenizer.trust_remote_code)
    start = time.perf_counter()
    eval_sets = [prepare_set(name, config.data_dir, tokenizer, config.seq_len) for name in config.sets]
    num_tokens = sum(len(ids) - begin for eval_set in eval_sets for ids, begin in eval_set.requests)
    logger.info(f"Prepared {len(eval_sets)} sets ({num_tokens:,} scored tokens) in {time.perf_counter() - start:.0f}s")

    last_step = -1
    while True:
        newest = broadcast(max(finished_snapshot_steps(config.run_dir), default=-1) if world.is_master else 0)
        if newest <= last_step:
            if run.max_steps is not None and last_step >= run.max_steps:
                break
            time.sleep(config.poll_seconds)
            continue

        round_start = time.perf_counter()
        dcp_load(
            {"app": AppState(model, [], None, None)},
            checkpoint_id=snapshot_dir(config.run_dir) / f"step_{newest}",
            process_group=get_dcp_group(),
        )
        load_time = time.perf_counter() - round_start
        metrics = {}
        for eval_set in eval_sets:
            scored = score_requests(model, eval_set.requests, config.token_budget, get_dcp_group())
            if not world.is_master:
                continue
            summary, arrays = summarize(eval_set, *scored)
            metrics |= {
                f"eval/{eval_set.name}/{key}": value for key, value in summary.items() if isinstance(value, float)
            }
            if config.save_arrays:
                out = config.run_dir / "evals" / f"step_{newest}"
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out / f"{eval_set.name}.npz", **arrays)
                (out / f"{eval_set.name}.json").write_text(
                    json.dumps(summary | {"data_sha256": eval_set.sha256}, default=float)
                )
        if world.is_master:
            round_time = time.perf_counter() - round_start
            metrics |= {"time/eval_round": round_time, "time/eval_load": load_time, "step": newest}
            asyncio.run(monitors.log(metrics, step=newest))
            delete_snapshots_up_to(config.run_dir, newest)
            logger.success(f"Evals | Step {newest} | {round_time:.0f}s | {len(eval_sets)} sets")
        last_step = newest

    asyncio.run(monitors.finalize())


def main():
    set_proc_title("PretrainEvalWorker")
    main_loop(cli(EvalWorkerConfig))


if __name__ == "__main__":
    main()
