"""Separate PPO critic: token scoring and online value updates."""

import prime_rl._compat  # noqa: F401

# ruff: noqa: I001

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.distributed as dist
import verifiers.v1 as vf

from prime_rl.configs.value import ValueConfig
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.lora import get_lora_state
from prime_rl.trainer.model import get_full_offload_dtype_policy, setup_model
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.trainer.optim import FullCPUOffloadOptimizer, setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.rl.data import DataLoader, TensorMicroBatch
from prime_rl.trainer.rl.loss import shift_tensor_right
from prime_rl.trainer.rl.policy_sync import latest_ready_step, load_policy_backbone, snapshot_path
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import (
    begin_backward,
    clip_grad_norm_,
    finish_backward,
    prepare_gradient_offload,
    scale_gradients_,
    setup_full_cpu_optimizer_offload,
    setup_torch_distributed,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.cp import gather_for_cp, gather_for_cp_wo_grad, setup_context_parallel, setup_cp_params
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import resolve_latest_ckpt_step


@dataclass
class ScoreRequest:
    batched_token_ids: list[list[int]]
    single: bool = False
    enqueued_at: float = field(default_factory=time.perf_counter)
    done: threading.Event = field(default_factory=threading.Event)
    values: list[list[float]] | None = None
    bootstrap_values: list[float] | None = None
    error: BaseException | None = None


@dataclass
class ScoreBin:
    indices: list[int]
    token_ids: list[int]
    lengths: list[int]


def pack_score_requests(batched_token_ids: list[list[int]], seq_len: int, dp: int, cp: int) -> list[list[ScoreBin]]:
    """Pack whole scoring sequences, then give every DP replica the same number of bins."""
    if not batched_token_ids or any(not ids or len(ids) > seq_len for ids in batched_token_ids):
        raise ValueError("Scoring requires nonempty sequences within value.model.seq_len")
    divisor = 2 * cp if cp > 1 else 1
    if seq_len % divisor:
        raise ValueError("value.model.seq_len must be divisible by the CP score divisor")
    bins: list[ScoreBin] = []
    for index in sorted(range(len(batched_token_ids)), key=lambda i: len(batched_token_ids[i]), reverse=True):
        tokens = batched_token_ids[index]
        target = next((b for b in bins if len(b.token_ids) + len(tokens) <= seq_len), None)
        if target is None:
            target = ScoreBin([], [], [])
            bins.append(target)
        target.indices.append(index)
        target.token_ids.extend(tokens)
        target.lengths.append(len(tokens))
    ranks: list[list[ScoreBin]] = [[] for _ in range(dp)]
    loads = [0] * dp
    for score_bin in sorted(bins, key=lambda b: len(b.token_ids), reverse=True):
        rank = min(range(dp), key=lambda i: (loads[i], len(ranks[i])))
        ranks[rank].append(score_bin)
        loads[rank] += len(score_bin.token_ids)
    steps = max(map(len, ranks))
    for rank_bins in ranks:
        rank_bins.extend(ScoreBin([], [], []) for _ in range(steps - len(rank_bins)))
    return ranks


@dataclass
class ServiceState:
    requests: queue.Queue[ScoreRequest] = field(default_factory=queue.Queue)
    completed_step: int = 0
    completed_metrics: dict[str, float] = field(default_factory=dict)


def _server_handler(state: ServiceState, max_seq_len: int | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                return
            if self.path != "/status":
                self.send_error(404)
                return
            payload = json.dumps({"completed_step": state.completed_step, "metrics": state.completed_metrics}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != "/score":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            data = json.loads(body)
            single = "token_ids" in data
            batched_token_ids = [data["token_ids"]] if single else data.get("batched_token_ids")
            if (
                not isinstance(batched_token_ids, list)
                or not batched_token_ids
                or any(
                    not isinstance(ids, list) or not ids or any(not isinstance(token, int) for token in ids)
                    for ids in batched_token_ids
                )
            ):
                self.send_error(400, "batched_token_ids must be a nonempty list of nonempty integer lists")
                return
            if max_seq_len is not None and any(len(ids) > max_seq_len for ids in batched_token_ids):
                self.send_error(400, "scoring sequence exceeds value.model.seq_len")
                return
            request = ScoreRequest(batched_token_ids, single=single)
            state.requests.put(request)
            if not request.done.wait(timeout=3600):
                self.send_error(504, "value scoring timed out")
                return
            if request.error is not None:
                self.send_error(500, str(request.error))
                return
            if single:
                payload = json.dumps(
                    {"values": request.values[0], "bootstrap_value": request.bootstrap_values[0]}
                ).encode()
            else:
                payload = json.dumps({"values": request.values, "bootstrap_values": request.bootstrap_values}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    return Handler


def _forward_values(
    model, input_ids, position_ids, seq_lens, parallel_dims, cp_style, *, lora_enabled=False, head_only=False
):
    cp_enabled = parallel_dims.cp_enabled
    if cp_enabled:
        cp_mesh = parallel_dims.get_mesh("cp")
        input_ids, position_ids = setup_cp_params(
            input_ids,
            position_ids,
            cp_mesh.get_local_rank(),
            parallel_dims.cp,
            cp_mesh.get_group(),
            seq_lens=seq_lens,
            cp_style=cp_style,
        )
    if lora_enabled:
        set_lora_num_tokens(torch.tensor([input_ids.shape[1]], device="cuda", dtype=torch.int32))
    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        seq_lens=seq_lens,
        seq_lens_are_pre_shard=cp_enabled,
        return_values=True,
        head_only=head_only,
    )
    return output["values"]


def _score(model, token_ids: list[int], parallel_dims, cp_style: str, lora_enabled: bool) -> tuple[list[float], float]:
    original_length = len(token_ids)
    padding = (-original_length) % parallel_dims.cp
    padded = token_ids + [0] * padding
    input_ids = torch.tensor(padded, device="cuda", dtype=torch.long).unsqueeze(0)
    position_ids = torch.arange(len(padded), device="cuda", dtype=torch.long).unsqueeze(0)
    seq_lens = torch.tensor([len(padded)], device="cuda", dtype=torch.long)
    model.eval()
    with torch.no_grad():
        values = _forward_values(
            model, input_ids, position_ids, seq_lens, parallel_dims, cp_style, lora_enabled=lora_enabled
        )
        if parallel_dims.cp_enabled:
            values = gather_for_cp_wo_grad(values, parallel_dims.cp, parallel_dims.get_mesh("cp").get_group())
    return (
        shift_tensor_right(values)[0, :original_length].float().cpu().tolist(),
        values[0, original_length - 1].float().item(),
    )


def _score_batch(
    model, batched_token_ids: list[list[int]], parallel_dims, seq_len: int, cp_style: str, lora_enabled: bool
) -> tuple[list[list[float]], list[float]] | None:
    dp = parallel_dims.get_mesh("dp").size()
    cp = parallel_dims.cp
    rank = dist.get_rank()
    dp_rank = rank // parallel_dims.non_data_parallel_size
    bins_by_rank = pack_score_requests(batched_token_ids, seq_len, dp, cp)
    local_results: list[tuple[int, list[float], float]] = []
    model.eval()
    with torch.no_grad():
        for step in range(len(bins_by_rank[0])):
            score_bin = bins_by_rank[dp_rank][step]
            length = len(score_bin.token_ids)
            max_length = max(len(bins[step].token_ids) for bins in bins_by_rank)
            divisor = 2 * cp if cp > 1 else 1
            padded_length = ((max_length + divisor - 1) // divisor) * divisor
            padding = padded_length - length
            tokens = score_bin.token_ids + [0] * padding
            positions = [position for n in score_bin.lengths for position in range(n)] + list(range(padding))
            segment_lengths = score_bin.lengths + ([padding] if padding else [])
            input_ids = torch.tensor(tokens, device="cuda", dtype=torch.long).unsqueeze(0)
            position_ids = torch.tensor(positions, device="cuda", dtype=torch.long).unsqueeze(0)
            seq_lens = torch.tensor(segment_lengths, device="cuda", dtype=torch.long)
            raw_values = _forward_values(
                model, input_ids, position_ids, seq_lens, parallel_dims, cp_style, lora_enabled=lora_enabled
            )
            if parallel_dims.cp_enabled:
                raw_values = gather_for_cp_wo_grad(raw_values, cp, parallel_dims.get_mesh("cp").get_group())
            if rank % parallel_dims.non_data_parallel_size == 0:
                offset = 0
                for index, n in zip(score_bin.indices, score_bin.lengths, strict=True):
                    segment = raw_values[:, offset : offset + n]
                    local_results.append(
                        (index, shift_tensor_right(segment)[0].float().cpu().tolist(), segment[0, -1].float().item())
                    )
                    offset += n
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_results, gathered, dst=0)
    if rank != 0:
        return None
    values: list[list[float] | None] = [None] * len(batched_token_ids)
    bootstraps: list[float | None] = [None] * len(batched_token_ids)
    for rank_results in gathered:
        for index, token_values, bootstrap in rank_results:
            values[index] = token_values
            bootstraps[index] = bootstrap
    if any(value is None for value in values) or any(value is None for value in bootstraps):
        raise RuntimeError("Value scoring failed to return every sequence")
    return values, bootstraps


def _train_batch(
    model, optimizer, scheduler, gradient_manager, micro_batches, parallel_dims, config, head_only, updates=None
):
    model.train()
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    total_count = torch.zeros((), device="cuda", dtype=torch.long)
    for batch in micro_batches:
        mask = batch["value_mask"]
        if mask is None:
            raise ValueError("PPO critic received a batch without value targets")
        mask = mask.to("cuda")
        total_count += mask.sum()
    dist.all_reduce(total_count, group=dp_cp_group)
    denominator = total_count.clamp_min(1)

    last_loss = 0.0
    forward_backward_seconds = 0.0
    optimizer_seconds = 0.0
    offload_materialize_seconds = 0.0
    offload_optimizer_kernel_seconds = 0.0
    for _ in range(updates if updates is not None else config.updates_per_step):
        update_start = time.perf_counter()
        update_loss = 0.0
        prepare_gradient_offload(gradient_manager, parallel_dims.fsdp_gradient_divide_factor, overlap_optimizer=True)
        for micro_step, batch in enumerate(micro_batches):
            input_ids = batch["input_ids"].to("cuda")
            position_ids = batch["position_ids"].to("cuda")
            seq_lens = batch["seq_lens"].to("cuda")
            targets = batch["value_targets"].to("cuda")
            mask = batch["value_mask"].to("cuda")
            values = _forward_values(
                model,
                input_ids,
                position_ids,
                seq_lens,
                parallel_dims,
                config.model.cp_style,
                lora_enabled=config.model.lora is not None,
                head_only=head_only,
            )
            if parallel_dims.cp_enabled:
                values = gather_for_cp(values, parallel_dims.get_mesh("cp").get_group())
            values = shift_tensor_right(values)
            loss = (((values.float() - targets.float()) ** 2) * mask).sum() / denominator
            begin_backward(gradient_manager, final_backward=micro_step == len(micro_batches) - 1)
            loss.backward()
            finish_backward(gradient_manager)
            update_loss += loss.detach().item()
        forward_backward_seconds += time.perf_counter() - update_start
        optimizer_start = time.perf_counter()
        if gradient_manager is None:
            scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor)
        if config.optim.max_norm is not None:
            clip_grad_norm_(gradient_manager, model, config.optim.max_norm, parallel_dims.ep_enabled)
        optimizer.step()
        if isinstance(optimizer, FullCPUOffloadOptimizer):
            offload_materialize_seconds += optimizer.last_step_timings.get("materialize", 0.0)
            offload_optimizer_kernel_seconds += optimizer.last_step_timings.get("optimizer_kernel", 0.0)
        optimizer.zero_grad()
        scheduler.step()
        optimizer_seconds += time.perf_counter() - optimizer_start
        last_loss = update_loss
    return last_loss, {
        "forward_backward_seconds": forward_backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "offload_materialize_seconds": offload_materialize_seconds,
        "offload_optimizer_kernel_seconds": offload_optimizer_kernel_seconds,
    }


def _pretrain(model, optimizer, scheduler, gradient_manager, parallel_dims, config):
    if config.pretrain_data is None or config.pretrain_steps == 0:
        return
    with Path(config.pretrain_data).open() as records:
        for step in range(config.pretrain_steps):
            line = records.readline()
            if not line:
                raise ValueError(f"value.pretrain_data has fewer than {config.pretrain_steps} records")
            item = json.loads(line)
            trace = item.get("trace", item)
            if "token_ids" in trace:
                streams = [(trace["token_ids"], trace["action_mask"])]
                reward = float(item["reward"])
            else:
                native_trace = vf.Trace.model_validate(trace)
                streams = [(sample.token_ids, sample.mask) for sample in trace_to_samples(native_trace)]
                reward = float(item.get("reward", native_trace.reward))
            if not streams:
                raise ValueError("Pretraining trace has no trainable branches")
            batches = []
            for token_ids, action_mask in streams:
                if len(token_ids) != len(action_mask):
                    raise ValueError("Pretraining token_ids and action_mask lengths differ")
                if len(token_ids) > config.model.seq_len:
                    raise ValueError("Pretraining trace exceeds value.model.seq_len")
                padding = (-len(token_ids)) % parallel_dims.cp
                n = len(token_ids) + padding
                mask = action_mask + [False] * padding
                target = [reward if sampled else 0.0 for sampled in mask]
                batches.append(
                    {
                        "input_ids": torch.tensor(token_ids + [0] * padding).unsqueeze(0),
                        "position_ids": torch.arange(n).unsqueeze(0),
                        "seq_lens": torch.tensor([n]),
                        "value_targets": torch.tensor(target, dtype=torch.float).unsqueeze(0),
                        "value_mask": torch.tensor(mask, dtype=torch.bool).unsqueeze(0),
                    }
                )
            _train_batch(
                model,
                optimizer,
                scheduler,
                gradient_manager,
                batches,
                parallel_dims,
                config,
                head_only=step < config.head_warmup_steps,
            )


def train(config: ValueConfig):
    world = get_world()
    logger = setup_logger("info")
    setup_torch_distributed(
        timeout=timedelta(seconds=3600),
        enable_gloo=config.model.fsdp_cpu_offload or config.model.full_offload is not None,
    )
    if config.model.full_offload is not None:
        setup_full_cpu_optimizer_offload(config.model.full_offload)
    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt, resume=config.resume)
    checkpoint_step = None
    if config.resume is not None:
        checkpoint_step = (
            config.resume.step
            or (config.resume.dir_step if config.resume.dir is not None else None)
            or resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        )
    model = setup_model(
        config.model,
        parallel_dims,
        checkpoint_step is not None,
        value_model=True,
        freeze_attention=config.freeze_attention,
    )
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config.model, parallel_dims)
    if config.model.lora is not None:
        get_lora_state().reset_adapter_parameters()
    optimizer, gradient_manager = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        cpu_offload=config.model.optim_cpu_offload,
        full_offload_config=config.model.full_offload,
        model=model,
        full_offload_dtype_policy=(
            get_full_offload_dtype_policy(model, config.model) if config.model.full_offload is not None else None
        ),
    )
    scheduler_steps = None
    if config.max_steps is not None:
        scheduler_steps = (config.max_steps + config.pretrain_steps) * config.updates_per_step
        if config.policy_sync_interval is not None:
            first_actor_step = max(0, config.head_warmup_steps - config.pretrain_steps) + 1
            syncs = config.max_steps // config.policy_sync_interval
            syncs -= (first_actor_step - 1) // config.policy_sync_interval
            scheduler_steps += syncs * config.policy_sync_retune_updates
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)
    progress = Progress()
    if checkpoint_step is not None:
        resume_path = config.resume.dir / "trainer" if config.resume and config.resume.dir is not None else None
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress, path=resume_path)
        progress.step += 1

    if checkpoint_step is None:
        _pretrain(model, optimizer, scheduler, gradient_manager, parallel_dims, config)
    dataloader = DataLoader(
        config.rollout_dir, progress.step, parallel_dims.get_mesh("dp").size(), config.rollout_transport
    )
    dist.barrier()

    state = ServiceState(completed_step=progress.step - 1)
    server = None
    server_thread = None
    if world.is_master:
        server = ThreadingHTTPServer(
            (config.service_host, config.service_port), _server_handler(state, config.model.seq_len)
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        logger.info(f"Value service ready on {config.service_host}:{config.service_port}")

    try:
        final_step = None
        last_batch = None
        last_sync_step = progress.step - 1
        score_requests = 0
        score_tokens = 0
        score_queue_seconds = 0.0
        score_compute_seconds = 0.0
        while True:
            request = None
            if world.is_master:
                sync_step = (
                    latest_ready_step(config.policy_sync_dir, last_sync_step, final_step or progress.step - 1)
                    if config.policy_sync_dir is not None and last_batch is not None
                    else None
                )
                if sync_step is not None:
                    command = ("sync", sync_step)
                elif dataloader.receiver.can_receive():
                    if final_step is not None:
                        raise RuntimeError("Value trainer received a batch after max_steps")
                    command = ("train", None)
                else:
                    try:
                        request = state.requests.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    command = ("score", request.batched_token_ids)
            else:
                command = None
            objects = [command]
            dist.broadcast_object_list(objects, src=0, device=torch.device("cuda"))
            action, payload = objects[0]
            if action == "sync":
                assert config.policy_sync_dir is not None and last_batch is not None
                load_policy_backbone(model, config.policy_sync_dir, payload)
                _train_batch(
                    model,
                    optimizer,
                    scheduler,
                    gradient_manager,
                    last_batch,
                    parallel_dims,
                    config,
                    head_only=False,
                    updates=config.policy_sync_retune_updates,
                )
                last_sync_step = payload
                if world.is_master:
                    (snapshot_path(config.policy_sync_dir, payload) / ".applied").touch()
                    logger.info(f"Loaded policy backbone at step {payload} and retuned critic LoRA")
                continue
            if action == "score":
                if request is not None:
                    score_requests += 1
                    score_tokens += sum(map(len, request.batched_token_ids))
                    score_queue_seconds += time.perf_counter() - request.enqueued_at
                score_start = time.perf_counter()
                try:
                    result = _score_batch(
                        model,
                        payload,
                        parallel_dims,
                        config.model.seq_len,
                        config.model.cp_style,
                        config.model.lora is not None,
                    )
                    if request is not None:
                        request.values, request.bootstrap_values = result
                except BaseException as error:
                    if request is not None:
                        request.error = error
                    raise
                finally:
                    if request is not None:
                        score_compute_seconds += time.perf_counter() - score_start
                        request.done.set()
                continue

            micro_batches: list[TensorMicroBatch] = dataloader.get_batch()
            t0 = time.perf_counter()
            loss, train_metrics = _train_batch(
                model,
                optimizer,
                scheduler,
                gradient_manager,
                micro_batches,
                parallel_dims,
                config,
                head_only=progress.step <= max(0, config.head_warmup_steps - config.pretrain_steps),
            )
            last_batch = micro_batches
            if world.is_master:
                train_seconds = time.perf_counter() - t0
                state.completed_metrics = {
                    "train_seconds": train_seconds,
                    "score_queue_seconds": score_queue_seconds,
                    "score_compute_seconds": score_compute_seconds,
                    **train_metrics,
                }
                logger.info(
                    f"Value step {progress.step} | loss={loss:.5f} | time={train_seconds:.1f}s "
                    f"forward_backward={train_metrics['forward_backward_seconds']:.1f}s "
                    f"optimizer={train_metrics['optimizer_seconds']:.1f}s "
                    f"offload_materialize={train_metrics['offload_materialize_seconds']:.1f}s "
                    f"offload_kernel={train_metrics['offload_optimizer_kernel_seconds']:.1f}s | "
                    f"score_requests={score_requests} score_tokens={score_tokens} "
                    f"score_queue={score_queue_seconds:.1f}s score_compute={score_compute_seconds:.1f}s"
                )
                score_requests = 0
                score_tokens = 0
                score_queue_seconds = 0.0
                score_compute_seconds = 0.0
            is_last_step = config.max_steps is not None and progress.step >= config.max_steps
            if (
                config.ckpt is not None
                and config.ckpt.interval
                and progress.step % config.ckpt.interval == 0
                and not is_last_step
            ):
                ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
                ckpt_manager.maybe_clean()
            if is_last_step:
                if config.ckpt is not None:
                    ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
                final_step = progress.step
            else:
                progress.step += 1
            state.completed_step = final_step or progress.step - 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join()
        if gradient_manager is not None:
            gradient_manager.close()
    logger.info("Value trainer finished")


def main():
    set_proc_title("Value Trainer")
    train(cli(ValueConfig))


if __name__ == "__main__":
    main()
