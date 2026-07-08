"""TTT trainer v2 — the prime-rl trainer stack (FSDP + MultiLoRA slots) as the TTT engine.

The lightweight v1 trainer (`trainer.py`: single-device HF + PEFT, one resident adapter,
CPU swap in/out) is right for small models; at 100B-MoE scale it can't hold the model, and
per-rollout swap+forward serialization can't keep up with a fleet of concurrent rollouts.
v2 reuses the multi-tenant LoRA machinery the RL trainer already has:

- **Model**: `setup_model` — the same custom modeling stack the RL trainer runs (glm4_moe
  etc.), FSDP2 across the TTT node(s), CP for long single sequences, AC, fused LM head.
- **Adapters**: `apply_lora_to_model` inside `setup_model` swaps the target modules for
  `MultiLoRALinear`/`MultiLoRAGroupedExperts` with `max_slots` resident adapter slots; a
  TTT rollout claims a free slot on first update (`reset_parameters(idx)` → B=0,
  base-identical — the same zero-init semantics as v1) and frees it on release.
- **Batched updates**: the segmented `lora_num_tokens` layout means sequences belonging to
  DIFFERENT adapters pack into ONE forward — the service drains its queue into packed
  micro-batches (sorted by slot; each sequence's tokens contiguous), so N concurrent
  rollouts' updates cost ~N/pack_size forwards instead of N.
- **Per-slot optimizers**: one optimizer per slot over
  `runs.get_named_parameters_for_run(idx)`, state dropped on release.
- **Checkpoints**: the existing slot-sliced adapter export (`get_state_dict_for_run` +
  `convert_adapter_to_hf` + `save_lora_config`) — the PEFT/vLLM on-disk format, identical
  to v1's, so the rollout hook, vLLM loads, and the RL replay manager are unchanged.

Multi-rank: rank 0 runs the HTTP server and broadcasts work orders
(`torch.distributed` broadcast_object_list); all ranks execute the same
forward/backward/step; rank 0 writes checkpoints and replies. Same-numerics bonus: the
service shares modeling code with the RL trainer's frozen-adapter replay.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.qa_render import assert_prefix_stable_template, render_qa_pair


@dataclass
class SlotState:
    """One rollout's slot assignment + bookkeeping (adapter weights live in the model's
    slot; optimizer state in `optimizer`)."""

    rollout_id: str
    adapter_name: str
    idx: int
    version: int = 0
    optimizer: torch.optim.Optimizer | None = None


@dataclass
class UpdateJob:
    rollout_id: str
    adapter_name: str
    token_ids: list[int]
    loss_mask: list[bool]
    seq_no: int
    qa_pairs: list[dict] | None = None
    train_rollout: bool = True
    system_prompt: str | None = None
    tools: list[dict] | None = None
    # Filled during execution
    sequences: list[tuple[list[int], list[bool]]] = field(default_factory=list)


class TTTTrainerV2:
    """The FSDP/MultiLoRA TTT engine. Construct on every rank (under torchrun); drive via
    `update_batch` / `release` — same per-job semantics as v1's `TTTTrainer.update`, plus
    cross-rollout batching."""

    def __init__(self, config: TTTServiceConfig):
        assert config.engine.type == "fsdp"
        model_config = config.engine.to_model_config(config.lora)
        if getattr(model_config, "cp", 1) > 1:
            # update_batch feeds full packed sequences to every rank; CP would need the RL
            # train loop's per-rank sequence sharding (setup_cp_params/shard_for_cp), which
            # this engine deliberately doesn't do — updates are many short-ish sequences
            # packed together, so FSDP across ranks is the right axis. Fail at startup, not
            # with silent numerics.
            raise ValueError(
                "TTT fsdp engine does not support context parallelism (engine.model.cp > 1): "
                "update forwards are packed multi-sequence batches, not one long sequence. "
                "Use FSDP sharding (more ranks) and lower max_tokens_per_forward instead."
            )

        from prime_rl.trainer.model import setup_model, setup_tokenizer
        from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
        from prime_rl.trainer.runs import MultiRunManager, setup_multi_run_manager
        from prime_rl.trainer.world import get_world

        self.config = config
        self.logger = get_logger()
        self.world = get_world()
        self.device = torch.device("cuda", self.world.local_rank)
        self.ckpt_root = config.output_dir / "ttt"
        self.max_slots = config.engine.max_slots

        # The slot registry: we use MultiRunManager purely as the module/param registry
        # (register_module / *_for_run / reset_run_parameters) — its run-discovery /
        # eviction machinery is multi-tenant-platform-shaped and stays unused. max_runs
        # sizes the resident adapter slots inside apply_lora_to_model.
        self.runs: MultiRunManager = setup_multi_run_manager(config.output_dir, self.max_slots, self.device, None)
        # All slots share one alpha/rank ⇒ one scaling factor everywhere.
        self.runs.scaling_factors.fill_(config.lora.alpha / config.lora.rank)

        resolve_ep(model_config)
        self.parallel_dims = get_parallel_dims(model_config)
        self.logger.info(f"TTT v2: initializing model ({model_config.name}) on {self.world}")
        self.model = setup_model(model_config, self.parallel_dims)
        self.model.train()
        self.model_config = model_config
        self._tokenizer = None
        self._tokenizer_setup = setup_tokenizer
        # Startup canary: fail the service at launch when the chat template isn't
        # prefix-stable (Q&A masking would otherwise silently skip every pair).
        assert_prefix_stable_template(self.tokenizer)

        self.slots: dict[str, SlotState] = {}
        self.free_idxs = set(range(self.max_slots))
        self.logger.info(
            f"TTT v2 ready: {self.max_slots} adapter slots, rank={config.lora.rank}, "
            f"targets={config.lora.target_modules}"
        )

    # -- slot registry -----------------------------------------------------------------------

    def _claim(self, rollout_id: str, adapter_name: str) -> SlotState:
        state = self.slots.get(rollout_id)
        if state is not None:
            return state
        if not self.free_idxs:
            raise RuntimeError(
                f"no free TTT adapter slots (max_slots={self.max_slots}); "
                "raise [engine].max_slots or lower rollout concurrency"
            )
        idx = min(self.free_idxs)
        self.free_idxs.discard(idx)
        self.runs.reset_run_parameters(idx)  # B=0 → base-identical until the first step
        state = SlotState(rollout_id=rollout_id, adapter_name=adapter_name, idx=idx)
        self.slots[rollout_id] = state
        return state

    def release(self, rollout_id: str) -> SlotState | None:
        state = self.slots.pop(rollout_id, None)
        if state is not None:
            state.optimizer = None
            self.free_idxs.add(state.idx)
        if not self.config.keep_checkpoints:
            shutil.rmtree(self.ckpt_root / rollout_id, ignore_errors=True)
        return state

    def _optimizer(self, state: SlotState) -> torch.optim.Optimizer:
        if state.optimizer is None:
            params = [p for _, p in self.runs.get_named_parameters_for_run(state.idx)]
            cfg = self.config.optim
            if cfg.type == "sgd":
                state.optimizer = torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            else:
                state.optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        return state.optimizer

    # -- sequence prep (shared semantics with v1) ----------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from prime_rl.configs.trainer import TokenizerConfig

            self._tokenizer = self._tokenizer_setup(TokenizerConfig(name=self.config.model.name))
        return self._tokenizer

    def _tokenize_qa(
        self,
        qa_pairs: list[dict],
        system_prompt: str | None,
        tools: list[dict] | None,
    ) -> list[tuple[list[int], list[bool]]]:
        """Identical contract to v1 (`TTTTrainer._tokenize_qa`): standalone [system, Q, A]
        rendering with the chat template's `tools=`, loss on the answer only."""

        template_kwargs: dict = {"tools": tools} if tools else {}
        head = [{"role": "system", "content": system_prompt}] if system_prompt else []
        sequences: list[tuple[list[int], list[bool]]] = []
        for pair in qa_pairs:
            if not str(pair.get("answer", "")).strip():
                continue
            conversation = [
                *head,
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]},
            ]
            rendered = render_qa_pair(self.tokenizer, conversation, template_kwargs)
            if rendered is None:
                continue  # non-prefix-stable render: skip rather than train on the full render
            full, prompt_len = rendered
            if len(full) - prompt_len < 1:
                continue
            sequences.append((full, [False] * prompt_len + [True] * (len(full) - prompt_len)))
        return sequences

    def validate_job(self, job: UpdateJob) -> None:
        """Pure validation (no slot claim, no state mutation) — safe to run on rank 0 only,
        before a job is broadcast. Raises ValueError on a malformed or out-of-order job."""
        if len(job.token_ids) != len(job.loss_mask):
            raise ValueError(f"token_ids ({len(job.token_ids)}) and loss_mask ({len(job.loss_mask)}) must align")
        if len(job.token_ids) < 2:
            raise ValueError("need at least 2 tokens to form a next-token target")
        state = self.slots.get(job.rollout_id)
        expected = (state.version if state is not None else 0) + 1
        if job.seq_no != expected:
            raise ValueError(f"out-of-order update for {job.rollout_id}: expected seq_no {expected}, got {job.seq_no}")
        if state is None and not self.free_idxs:
            raise ValueError(
                f"no free TTT adapter slots (max_slots={self.max_slots}); "
                "raise [engine].max_slots or lower rollout concurrency"
            )
        if not job.train_rollout and not job.qa_pairs:
            raise ValueError("no trainable sequences (train_rollout=false and no qa_pairs)")

    def prepare_job(self, job: UpdateJob) -> SlotState:
        """Claim the job's slot and materialize its training sequences. MUTATING — must run
        identically on every rank (jobs arrive in the same order via broadcast, and slot
        claim is deterministic: lowest free index)."""
        self.validate_job(job)
        state = self._claim(job.rollout_id, job.adapter_name)
        sequences: list[tuple[list[int], list[bool]]] = []
        if job.train_rollout:
            sequences.append((job.token_ids, job.loss_mask))
        if job.qa_pairs:
            sequences.extend(self._tokenize_qa(job.qa_pairs, job.system_prompt, job.tools))
        sequences = [(ids, mask) for ids, mask in sequences if len(ids) >= 2 and any(mask[1:])]
        if not sequences:
            raise ValueError("no trainable sequences (empty loss masks / QA rendered empty)")
        job.sequences = sequences
        return state

    # -- the batched update ---------------------------------------------------------------------

    def _pack(self, jobs: list[UpdateJob]) -> list[list[UpdateJob]]:
        """Greedy first-fit packing of whole jobs into forward-sized bins. Jobs stay atomic
        (one job's sequences share a slot and an optimizer step); bins bound the packed
        token count by `engine.max_tokens_per_forward`."""
        cap = self.config.engine.max_tokens_per_forward

        def job_tokens(job: UpdateJob) -> int:
            return sum(len(ids) for ids, _ in job.sequences)

        bins: list[tuple[int, list[UpdateJob]]] = []
        for job in sorted(jobs, key=job_tokens, reverse=True):
            tokens = job_tokens(job)
            for i, (used, members) in enumerate(bins):
                if used + tokens <= cap:
                    bins[i] = (used + tokens, [*members, job])
                    break
            else:
                bins.append((tokens, [job]))
        return [members for _, members in bins]

    def update_batch(self, jobs: list[UpdateJob]) -> dict[str, dict]:
        """Run a batch of prepared update jobs (possibly for different rollouts) and return
        `{rollout_id: result}`. Jobs are packed into forwards sorted by slot index — the
        segmented `lora_num_tokens` layout MultiLoRA dispatches on. Each job takes
        `steps_per_update` optimizer steps on ITS OWN slot params; the packed forward is
        shared, the backward populates each slot's grads independently (token segments)."""
        from prime_rl.trainer.models.layers.lora import set_lora_num_tokens

        states = {job.rollout_id: self.prepare_job(job) for job in jobs}
        results: dict[str, dict] = {}
        for bin_jobs in self._pack(jobs):
            # Sort by slot: MultiLoRA's segment layout requires slot-ordered tokens.
            bin_jobs = sorted(bin_jobs, key=lambda j: states[j.rollout_id].idx)
            optimizers = [self._optimizer(states[j.rollout_id]) for j in bin_jobs]

            flat_ids: list[int] = []
            flat_pos: list[int] = []
            flat_labels: list[int] = []
            flat_target: list[bool] = []
            lora_num_tokens = torch.zeros(self.max_slots, dtype=torch.int32, device=self.device)
            # Per-job denominators: each job's loss is normalized by its own loss-token
            # count (matching v1's per-update semantics, independent of co-packed jobs).
            job_spans: list[tuple[int, int, int]] = []  # (start, end, num_loss_tokens)
            for job in bin_jobs:
                idx = states[job.rollout_id].idx
                start = len(flat_ids)
                for ids, mask in job.sequences:
                    flat_ids.extend(ids)
                    flat_pos.extend(range(len(ids)))
                    # next-token: label[i] = ids[i+1]; last position gets an ignore label 0
                    # and is never targeted (mask shifted left, last position False).
                    flat_labels.extend([*ids[1:], 0])
                    flat_target.extend([*mask[1:], False])
                    lora_num_tokens[idx] += len(ids)
                loss_count = sum(sum(mask[1:]) for _, mask in job.sequences)
                job_spans.append((start, len(flat_ids), max(loss_count, 1)))

            input_ids = torch.tensor([flat_ids], dtype=torch.long, device=self.device)
            position_ids = torch.tensor([flat_pos], dtype=torch.long, device=self.device)
            labels = torch.tensor([flat_labels], dtype=torch.long, device=self.device)
            target = torch.tensor([flat_target], dtype=torch.bool, device=self.device)

            job_losses = [0.0] * len(bin_jobs)
            for _ in range(self.config.steps_per_update):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                set_lora_num_tokens(lora_num_tokens)
                logprobs = self._forward_logprobs(input_ids, position_ids, labels)
                # Sum of per-job mean NLLs: each job's gradient lands only on its own
                # slot's params (token segments), so summing job losses is exact.
                total = None
                for j, (start, end, denom) in enumerate(job_spans):
                    span_target = target[0, start:end]
                    span_nll = -(logprobs[0, start:end] * span_target).sum() / denom
                    job_losses[j] = float(span_nll.detach())
                    total = span_nll if total is None else total + span_nll
                total.backward()
                if self.config.optim.max_norm is not None:
                    for job, optimizer in zip(bin_jobs, optimizers):
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group["params"]],
                            self.config.optim.max_norm,
                        )
                for optimizer in optimizers:
                    optimizer.step()

            for job, loss in zip(bin_jobs, job_losses):
                state = states[job.rollout_id]
                state.version = job.seq_no
                ckpt_path = self.save_checkpoint(state)
                num_loss = sum(sum(mask[1:]) for _, mask in job.sequences)
                results[job.rollout_id] = {
                    "version": state.version,
                    "loss": loss,
                    "ckpt_path": str(ckpt_path),
                    "num_loss_tokens": num_loss,
                }
                self.logger.info(
                    f"ttt v2 update {job.rollout_id} v{job.seq_no}: loss={loss:.4f} "
                    f"({num_loss} loss tokens, slot {state.idx})"
                )
        return results

    def _forward_logprobs(self, input_ids, position_ids, labels):
        """One packed forward -> per-token logprobs of `labels`. Uses the trainer stack's
        `forward`; the fused LM head computes logprobs directly (it requires labels and
        per-token temperatures — 1.0 everywhere: TTT is plain NLL, no sampling
        temperature), the vanilla head falls back to selective log-softmax."""
        from prime_rl.trainer.model import forward
        from prime_rl.trainer.rl.loss import selective_log_softmax

        temperature = torch.ones_like(input_ids, dtype=torch.float32)
        out = forward(self.model, input_ids, position_ids, labels=labels, temperature=temperature)
        if out.get("logprobs") is not None:
            return out["logprobs"]
        return selective_log_softmax(out["logits"].float(), labels)

    # -- checkpointing ----------------------------------------------------------------------------

    def save_checkpoint(self, state: SlotState) -> Path:
        """Slot-sliced adapter export in the PEFT/vLLM format (same on-disk contract as
        v1). `get_state_dict_for_run` gathers DTensor shards; only rank 0 writes."""
        import safetensors.torch
        from torch.distributed.tensor import DTensor

        from prime_rl.trainer.lora import save_lora_config

        raw = self.runs.get_state_dict_for_run(state.idx)
        tensors = {
            f"base_model.model.{key}": (value.full_tensor() if isinstance(value, DTensor) else value).to("cpu")
            for key, value in raw.items()
        }
        path = self.ckpt_root / state.rollout_id / f"v{state.version}"
        if self.world.is_master:
            tmp = path.with_name(f"{path.name}.tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            safetensors.torch.save_file(tensors, tmp / "adapter_model.safetensors")
            save_lora_config(
                self.model,
                tmp,
                rank=self.config.lora.rank,
                alpha=self.config.lora.alpha,
                dropout=self.config.lora.dropout,
            )
            shutil.rmtree(path, ignore_errors=True)
            tmp.rename(path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return path
