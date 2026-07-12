"""TTT trainer v2 — the prime-rl trainer stack (FSDP + MultiLoRA slots) as the TTT engine.

The lightweight v1 trainer (`trainer.py`: single-device HF + PEFT, one resident adapter,
CPU swap in/out) is right for small models; at 100B-MoE scale it can't hold the model, and
per-rollout swap+forward serialization can't keep up with a fleet of concurrent rollouts.
v2 reuses the multi-tenant LoRA machinery the RL trainer already has:

- **Model**: `setup_model` — the same custom modeling stack the RL trainer runs (glm4_moe
  etc.), FSDP2 across the TTT node(s), AC, and a chunked/fused LM head.
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
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import torch

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.ttt.identity import (
    ReleaseResult,
    checkpoint_rollout_dir,
    update_fingerprint,
    validate_adapter_name,
    validate_rollout_id,
)
from prime_rl.ttt.validation import model_vocab_size, validate_token_ids
from prime_rl.utils.logger import get_logger
from prime_rl.utils.qa_render import assert_prefix_stable_template, render_qa_pair

if TYPE_CHECKING:
    from prime_rl.trainer.optim import CPUOffloadOptimizer

_RELEASE_TOMBSTONES = 4096


class _FrozenWeightLogProbEntropyFn:
    """Lazily builds the TTT-only frozen-weight autograd function.

    Importing the trainer LM-head module at file import time pulls in the full trainer
    model stack. Keep that import lazy, as the rest of this module does, so registry-only
    clients and tests do not pay that cost.
    """

    _function = None

    @classmethod
    def get(cls):
        if cls._function is not None:
            return cls._function

        from prime_rl.trainer.models.layers.lm_head import _SequenceChunkedLogProbEntropyFn

        class FrozenWeightLogProbEntropyFn(_SequenceChunkedLogProbEntropyFn):
            """The normal chunked forward, with backward only for hidden states."""

            @staticmethod
            def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
                assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
                    "Backward through entropy is not implemented in FusedOutputLinear"
                )

                hidden, weight, labels, inv_temperature, logz = ctx.saved_tensors
                chunk_size: int = ctx.chunk_size

                n = hidden.shape[0]
                vocab = weight.shape[0]
                vocab_chunk_size = min(vocab, 8192)
                grad_hidden = torch.zeros_like(hidden)

                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    hidden_chunk = hidden[start:end]
                    labels_chunk = labels[start:end]
                    grad_chunk = grad_logprobs[start:end].to(torch.float32)
                    inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
                    logz_chunk = logz[start:end]

                    for vocab_start in range(0, vocab, vocab_chunk_size):
                        vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                        weight_chunk = weight[vocab_start:vocab_end]
                        logits_chunk = hidden_chunk @ weight_chunk.t()
                        scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                        probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                        grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                        mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                        if torch.any(mask):
                            idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                            grad_logits[mask, idx] += grad_chunk[mask]
                        grad_logits = grad_logits * inv_t_chunk

                        grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)

                # TTT freezes the base model, including the vocabulary-sized output
                # weight. Returning None avoids materializing a useless [V, H] gradient.
                return grad_hidden, None, None, None, None

        cls._function = FrozenWeightLogProbEntropyFn
        return cls._function


def _frozen_fused_output_forward(self, hidden_states, labels=None, temperature=None):
    """FusedOutputLinear.forward for a frozen TTT output weight."""
    from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput

    assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"
    assert temperature is not None, "FusedOutputLinear requires per-token temperatures"

    b, s, h = hidden_states.shape
    hidden_states = hidden_states.reshape(b * s, h).contiguous()
    labels = labels.reshape(b * s).contiguous()
    inv_t = 1.0 / temperature.reshape(b * s).contiguous()

    function = _FrozenWeightLogProbEntropyFn.get()
    logprobs, entropy = function.apply(hidden_states, self.weight, labels, inv_t, self.chunk_size)
    return PrimeLmOutput(logprobs=logprobs.reshape(b, s), entropy=entropy.reshape(b, s))


def _patch_frozen_fused_lm_head(model: torch.nn.Module) -> bool:
    """Use a hidden-gradient-only backward for TTT's frozen fused LM head.

    ``setup_model`` has already injected and FSDP-wrapped the head. Patch only its
    forward method: replacing the module or parameter here would invalidate FSDP's
    sharding metadata. The trainer compiler only compiles transformer layers, not the
    output head, so this post-setup instance patch does not invalidate a compiled graph.
    """
    from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear

    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, FusedOutputLinear):
        return False
    if lm_head.weight.requires_grad:
        raise ValueError("TTT's fused LM head must remain frozen")
    if getattr(lm_head, "_ttt_frozen_weight_backward", False):
        return True

    lm_head.forward = MethodType(_frozen_fused_output_forward, lm_head)
    lm_head._ttt_frozen_weight_backward = True
    return True


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _locally_finite(values) -> bool:
    for tensor in _iter_tensors(values):
        local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        if not bool(torch.isfinite(local).all().item()):
            return False
    return True


@dataclass
class SlotState:
    """One rollout's slot assignment + bookkeeping (adapter weights live in the model's
    slot; optimizer state in `optimizer`)."""

    rollout_id: str
    adapter_name: str
    idx: int
    version: int = 0
    optimizer: torch.optim.Optimizer | CPUOffloadOptimizer | None = None
    last_result: dict | None = None
    """The successful UpdateResponse payload of the last applied update, so an exact
    replay (client retry after a lost response) can be answered without re-training."""
    last_fingerprint: str | None = None
    loaded_version: int = 0
    metadata_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update_snapshot(self) -> tuple[int, dict | None, str | None]:
        with self.metadata_lock:
            return self.version, self.last_result, self.last_fingerprint

    def commit_update(self, version: int, result: dict, fingerprint: str) -> None:
        """Publish replay metadata and version as one reader-visible transaction."""
        with self.metadata_lock:
            self.last_result = result
            self.last_fingerprint = fingerprint
            self.version = version


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
        # This process is independent from the policy trainer; apply the same FP32/TF32
        # contract before importing/building any model or adapter state.
        torch.set_float32_matmul_precision(config.engine.matmul_precision)

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
        _patch_frozen_fused_lm_head(self.model)
        self.vocab_size = model_vocab_size(self.model)
        self._reject_grouped_expert_lora(self.model)
        self._reject_output_head_lora(self.model)
        self._reject_missing_lora_targets()
        self.model.train()
        self.model_config = model_config
        self._tokenizer = None
        self._tokenizer_setup = setup_tokenizer

        self.slots: dict[str, SlotState] = {}
        self.adapter_names: dict[str, str] = {}
        self.released: OrderedDict[str, str] = OrderedDict()
        self.free_idxs = set(range(self.max_slots))
        # Optional gloo group for control-plane collectives (set by run_server_v2): the
        # checkpoint barrier rides it so it shares the watchdog-free control plane.
        self.ctrl_pg = None
        self.logger.info(
            f"TTT v2 ready: {self.max_slots} adapter slots, rank={config.lora.rank}, "
            f"targets={config.lora.target_modules}"
        )

    def _require_finite(self, values, label: str) -> None:
        """Reject a non-finite value coherently across every distributed rank."""
        finite = torch.tensor(int(_locally_finite(values)), dtype=torch.int32, device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not bool(finite.item()):
            raise FloatingPointError(f"non-finite TTT v2 {label}")

    # -- slot registry -----------------------------------------------------------------------

    def _claim(self, rollout_id: str, adapter_name: str) -> SlotState:
        validate_rollout_id(rollout_id)
        validate_adapter_name(adapter_name)
        state = self.slots.get(rollout_id)
        if state is not None:
            if state.adapter_name != adapter_name:
                raise ValueError(
                    f"rollout {rollout_id!r} is bound to adapter {state.adapter_name!r}, not {adapter_name!r}"
                )
            return state
        owner = self.adapter_names.get(adapter_name)
        if owner is not None:
            raise ValueError(f"adapter {adapter_name!r} is already bound to rollout {owner!r}")
        if adapter_name in self.released.values():
            raise ValueError(f"adapter {adapter_name!r} is retained for an idempotent release retry")
        if rollout_id in self.released:
            raise ValueError(f"rollout {rollout_id!r} was already released and cannot be reused")
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
        self.adapter_names[adapter_name] = rollout_id
        return state

    def validate_release(self, rollout_id: str, adapter_name: str) -> ReleaseResult:
        validate_rollout_id(rollout_id)
        validate_adapter_name(adapter_name)
        state = self.slots.get(rollout_id)
        if state is not None:
            if state.adapter_name != adapter_name:
                raise ValueError(
                    f"rollout {rollout_id!r} is bound to adapter {state.adapter_name!r}, not {adapter_name!r}"
                )
            return ReleaseResult(adapter_name=state.adapter_name, released=True)
        prior_name = self.released.get(rollout_id)
        if prior_name is None:
            owner = self.adapter_names.get(adapter_name)
            if owner is not None:
                raise ValueError(f"adapter {adapter_name!r} belongs to rollout {owner!r}")
            raise ValueError(f"unknown or expired rollout {rollout_id!r}")
        if prior_name != adapter_name:
            raise ValueError(f"rollout {rollout_id!r} was released as adapter {prior_name!r}, not {adapter_name!r}")
        return ReleaseResult(adapter_name=prior_name, released=False)

    def release(self, rollout_id: str, adapter_name: str) -> ReleaseResult:
        result = self.validate_release(rollout_id, adapter_name)
        if not result.released:
            self.released.move_to_end(rollout_id)
            return result
        state = self.slots.pop(rollout_id)
        state.optimizer = None
        self.free_idxs.add(state.idx)
        self.adapter_names.pop(state.adapter_name, None)
        self.released[rollout_id] = state.adapter_name
        self.released.move_to_end(rollout_id)
        while len(self.released) > _RELEASE_TOMBSTONES:
            self.released.popitem(last=False)
        # Only rank 0 owns checkpoint I/O. Other ranks can differ in loaded_version because
        # adapter loading is an HTTP-side rank-0 operation, but their slot lifecycle remains
        # identical.
        if getattr(getattr(self, "world", None), "is_master", True):
            with state.metadata_lock:
                loaded_version = state.loaded_version
                version = state.version
            rollout_dir = checkpoint_rollout_dir(self.ckpt_root, rollout_id)
            if not self.config.keep_checkpoints:
                shutil.rmtree(rollout_dir, ignore_errors=True)
            else:
                for unacknowledged_version in range(loaded_version + 1, version + 1):
                    shutil.rmtree(rollout_dir / f"v{unacknowledged_version}", ignore_errors=True)
        return result

    def mark_loaded(self, rollout_id: str, adapter_name: str, version: int) -> None:
        state = self.slots.get(rollout_id)
        if state is None:
            raise ValueError(f"unknown rollout {rollout_id!r}")
        if state.adapter_name != adapter_name:
            raise ValueError(f"rollout {rollout_id!r} is bound to adapter {state.adapter_name!r}, not {adapter_name!r}")
        with state.metadata_lock:
            if version != state.version:
                raise ValueError(
                    f"cannot mark rollout {rollout_id!r} version {version} loaded; current version is {state.version}"
                )
            state.loaded_version = max(state.loaded_version, version)

    @staticmethod
    def _reject_grouped_expert_lora(model: torch.nn.Module) -> None:
        """Reject grouped-expert adapters until their multi-slot semantics are real.

        The current grouped-expert wrappers select ``argmax(lora_num_tokens)`` and thus
        support only one active adapter per forward. V2 deliberately packs several slots,
        and replay cannot resolve their virtual per-expert export paths either. Linear
        targets (including attention projections) do have segmented multi-slot routing.
        """
        from prime_rl.trainer.models.layers.lora.multi_moe import (
            MultiLoRAGptOssGroupedExperts,
            MultiLoRAGroupedExperts,
            MultiLoRANonGatedGroupedExperts,
        )

        grouped_types = (MultiLoRAGroupedExperts, MultiLoRANonGatedGroupedExperts, MultiLoRAGptOssGroupedExperts)
        names = [name for name, module in model.named_modules() if isinstance(module, grouped_types)]
        if names:
            preview = ", ".join(names[:5])
            raise ValueError(
                "TTT FSDP does not support grouped-expert LoRA targets in packed updates; "
                f"matched {preview}. Keep lora.target_modules on nn.Linear projections "
                "(attention-only for MoE deployments)."
            )

    @staticmethod
    def _reject_output_head_lora(model: torch.nn.Module) -> None:
        """The fused output head cannot execute an adapted wrapper."""
        from prime_rl.trainer.models.layers.lora import MultiLoRAModule

        output_modules = {getattr(model, "lm_head", None)}
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            output_modules.add(get_output_embeddings())
        names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, MultiLoRAModule)
            and (module in output_modules or "lm_head" in name.replace("_checkpoint_wrapped_module.", "").split("."))
        ]
        if names:
            raise ValueError(
                "TTT FSDP does not support LoRA on the causal LM output head; "
                f"realized adapted target(s): {', '.join(names[:5])}. Keep lora.target_modules "
                "on replay-supported hidden nn.Linear projections."
            )

    def _reject_missing_lora_targets(self) -> None:
        """Fail at startup when target patterns matched no supported module."""
        if not self.runs._modules:
            raise ValueError(
                "TTT FSDP lora.target_modules matched no model modules; fix the target "
                "names/patterns before starting the service."
            )

    def _optimizer(self, state: SlotState) -> torch.optim.Optimizer | CPUOffloadOptimizer:
        if state.optimizer is None:
            params = [p for _, p in self.runs.get_named_parameters_for_run(state.idx)]
            cfg = self.config.optim
            if cfg.type == "sgd":
                optimizer = torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            else:
                optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
            if self.config.engine.model.optim_cpu_offload:
                from prime_rl.trainer.optim import CPUOffloadOptimizer

                state.optimizer = CPUOffloadOptimizer(optimizer)
            else:
                state.optimizer = optimizer
        return state.optimizer

    # -- sequence prep (shared semantics with v1) ----------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tokenizer = self._tokenizer_setup(self.config.tokenizer)
            assert_prefix_stable_template(tokenizer)
            self._tokenizer = tokenizer
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

    def validate_job(self, job: UpdateJob) -> dict | None:
        """Pure validation (no slot claim, no state mutation) — safe to run on rank 0 only,
        before a job is broadcast. Raises ValueError on a malformed or out-of-order job.
        Returns the cached result dict when the job is an exact replay of the last applied
        update (a client retry after a lost response) — duplicate-ok, else None."""
        validate_rollout_id(job.rollout_id)
        validate_adapter_name(job.adapter_name)
        if len(job.token_ids) != len(job.loss_mask):
            raise ValueError(f"token_ids ({len(job.token_ids)}) and loss_mask ({len(job.loss_mask)}) must align")
        if len(job.token_ids) < 2:
            raise ValueError("need at least 2 tokens to form a next-token target")
        validate_token_ids(job.token_ids, self.vocab_size)
        if len(job.token_ids) > self.config.engine.max_tokens_per_forward:
            # A single job that can't fit one forward would fail packing on every retry —
            # reject it up-front instead of poisoning batches.
            raise ValueError(
                f"job too large: {len(job.token_ids)} tokens > "
                f"max_tokens_per_forward ({self.config.engine.max_tokens_per_forward})"
            )
        state = self.slots.get(job.rollout_id)
        if state is not None and state.adapter_name != job.adapter_name:
            raise ValueError(
                f"rollout {job.rollout_id!r} is bound to adapter {state.adapter_name!r}, not {job.adapter_name!r}"
            )
        owner = self.adapter_names.get(job.adapter_name)
        if owner is not None and owner != job.rollout_id:
            raise ValueError(f"adapter {job.adapter_name!r} is already bound to rollout {owner!r}")
        if state is None and job.adapter_name in self.released.values():
            raise ValueError(f"adapter {job.adapter_name!r} is retained for an idempotent release retry")
        if state is None and job.rollout_id in self.released:
            raise ValueError(f"rollout {job.rollout_id!r} was already released and cannot be reused")
        fingerprint = update_fingerprint(
            rollout_id=job.rollout_id,
            adapter_name=job.adapter_name,
            token_ids=job.token_ids,
            loss_mask=job.loss_mask,
            seq_no=job.seq_no,
            qa_pairs=job.qa_pairs,
            train_rollout=job.train_rollout,
            system_prompt=job.system_prompt,
            tools=job.tools,
        )
        if state is not None:
            state_version, last_result, last_fingerprint = state.update_snapshot()
        else:
            state_version, last_result, last_fingerprint = 0, None, None
        if state is not None and job.seq_no == state_version and last_result is not None:
            # Replay of the already-applied update (retry after a 503/502 lost the
            # response): the training already happened — answer from cache, don't 409.
            if fingerprint != last_fingerprint:
                raise ValueError(
                    f"seq_no {job.seq_no} for rollout {job.rollout_id!r} does not match the cached request"
                )
            return last_result
        expected = state_version + 1
        if job.seq_no != expected:
            raise ValueError(f"out-of-order update for {job.rollout_id}: expected seq_no {expected}, got {job.seq_no}")
        if job.qa_pairs:
            # Shape-check here so a malformed pair 409s its caller: past this point a
            # KeyError inside `_tokenize_qa` would escape `update_batch`'s per-job
            # ValueError isolation and hit the work loop's fail-fast — one bad request
            # must never take the whole service down.
            for i, pair in enumerate(job.qa_pairs):
                if not isinstance(pair, dict) or not isinstance(pair.get("question"), str):
                    raise ValueError(
                        f"malformed qa_pairs[{i}]: expected a dict with a string 'question' "
                        f"(and 'answer'), got {type(pair).__name__}"
                    )
                if not isinstance(pair.get("answer", ""), str):
                    raise ValueError(f"malformed qa_pairs[{i}]: 'answer' must be a string when present")
        if not job.train_rollout and not job.qa_pairs:
            raise ValueError("no trainable sequences (train_rollout=false and no qa_pairs)")
        if job.train_rollout and not any(job.loss_mask[1:]) and not job.qa_pairs:
            # Mirrors the late "no trainable sequences" error but at rank-0 validation
            # time, so an all-context mask 409s its caller before any slot claim.
            raise ValueError("loss mask selects no trainable target tokens")

    def prepare_job(self, job: UpdateJob) -> None:
        """Materialize the job's training sequences. PURE CPU work (validate + tokenize QA
        + build sequences + oversize check), deterministic across ranks — the GPU-mutating
        slot claim lives in `_claim`, called by `update_batch` AFTER the per-job error net
        so a CUDA fault there fail-fasts instead of being swallowed as a per-job error."""
        self.validate_job(job)
        sequences: list[tuple[list[int], list[bool]]] = []
        if job.train_rollout:
            sequences.append((job.token_ids, job.loss_mask))
        if job.qa_pairs:
            sequences.extend(self._tokenize_qa(job.qa_pairs, job.system_prompt, job.tools))
        sequences = [(ids, mask) for ids, mask in sequences if len(ids) >= 2 and any(mask[1:])]
        if not sequences:
            raise ValueError("no trainable sequences (empty loss masks / QA rendered empty)")
        for index, (ids, _) in enumerate(sequences):
            validate_token_ids(ids, self.vocab_size, source=f"sequences[{index}].token_ids")
        # QA pairs are only tokenized here, so the oversize check in validate_job can't
        # see them — re-check the packed total now that all sequences are materialized.
        total_tokens = sum(len(ids) for ids, _ in sequences)
        if total_tokens > self.config.engine.max_tokens_per_forward:
            raise ValueError(
                f"job too large after QA tokenization: {total_tokens} tokens > "
                f"max_tokens_per_forward ({self.config.engine.max_tokens_per_forward})"
            )
        job.sequences = sequences

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
        from torchtitan.distributed.utils import clip_grad_norm_

        from prime_rl.trainer.models.layers.lora import set_lora_num_tokens

        assert len({job.rollout_id for job in jobs}) == len(jobs), "duplicate rollout_ids in one batch"
        # In the service path, rank 0 has already materialized ``job.sequences`` and the
        # broadcast carries those exact token lists to every worker. That is important:
        # tokenizer/template code is external, and a rank-local preparation failure would
        # make workers enter different packed forwards/collectives. Direct unit callers may
        # still pass an unprepared job; only deterministic ValueError rejections are isolated
        # here. Unexpected faults escape to the work loop's fail-fast path.
        prepared: list[UpdateJob] = []
        results: dict[str, dict] = {}
        # Capacity is checked here, in authoritative work-queue order. New rollouts already
        # admitted in this batch count against the same free-slot snapshot.
        new_claims = 0
        new_adapter_owners: dict[str, str] = {}
        for job in jobs:
            try:
                cached = self.validate_job(job)
                if cached is not None:
                    # Exact replay of the last applied update: answer from cache — no
                    # forward, no optimizer step, no version bump. last_result is set on
                    # every rank, so the short-circuit is deterministic across ranks.
                    results[job.rollout_id] = cached
                    continue
                is_new = job.rollout_id not in self.slots
                owner = new_adapter_owners.get(job.adapter_name)
                if is_new and owner is not None and owner != job.rollout_id:
                    raise ValueError(
                        f"adapter {job.adapter_name!r} is already claimed in this batch by rollout {owner!r}"
                    )
                if is_new and new_claims >= len(self.free_idxs):
                    raise ValueError(
                        f"no free TTT adapter slots (max_slots={self.max_slots}); "
                        "raise [engine].max_slots or lower rollout concurrency"
                    )
                if not job.sequences:
                    self.prepare_job(job)
                prepared.append(job)
                if is_new:
                    new_adapter_owners[job.adapter_name] = job.rollout_id
                new_claims += is_new  # only count jobs that will actually claim
            except ValueError as e:
                results[job.rollout_id] = {"error": f"{type(e).__name__}: {e}"}
        if not prepared:
            return results  # everything failed validation: no forward to run
        # Claim slots for the surviving jobs — outside any broad catch (see above).
        states: dict[str, SlotState] = {
            job.rollout_id: self._claim(job.rollout_id, job.adapter_name) for job in prepared
        }
        jobs = prepared
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
                span_losses = []
                for start, end, denom in job_spans:
                    span_target = target[0, start:end]
                    span_nll = -(logprobs[0, start:end] * span_target).sum() / denom
                    span_losses.append(span_nll)
                    total = span_nll if total is None else total + span_nll
                self._require_finite(span_losses, "loss")
                for j, span_nll in enumerate(span_losses):
                    job_losses[j] = float(span_nll.detach())
                total.backward()
                params = [p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
                self._require_finite([p.grad for p in params if p.grad is not None], "gradients")
                if self.config.optim.max_norm is not None:
                    for job, optimizer in zip(bin_jobs, optimizers):
                        clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group["params"]],
                            max_norm=self.config.optim.max_norm,
                            ep_enabled=self.parallel_dims.ep_enabled,
                        )
                    self._require_finite([p.grad for p in params if p.grad is not None], "clipped gradients")
                for optimizer in optimizers:
                    optimizer.step()
                self._require_finite(params, "updated adapter parameters")
                # CPUOffloadOptimizer.state_dict() temporarily rehydrates its state on
                # CUDA, synchronizes, then offloads it again.  Finiteness only needs the
                # already-resident state mapping, so inspect that in place and avoid an
                # extra GPU transfer/sync for every packed slot on every step.
                self._require_finite([optimizer.state for optimizer in optimizers], "optimizer state")

            for job, loss in zip(bin_jobs, job_losses):
                state = states[job.rollout_id]
                # Export against an explicit staged version. The live state's version and
                # replay cache remain at the prior coherent snapshot until checkpointing
                # succeeds and ``commit_update`` publishes all three metadata fields.
                staged = SlotState(job.rollout_id, job.adapter_name, state.idx, version=job.seq_no)
                ckpt_path = self.save_checkpoint(staged)
                num_loss = sum(sum(mask[1:]) for _, mask in job.sequences)
                result = {
                    "version": job.seq_no,
                    "loss": loss,
                    "ckpt_path": str(ckpt_path),
                    "num_loss_tokens": num_loss,
                }
                fingerprint = update_fingerprint(
                    rollout_id=job.rollout_id,
                    adapter_name=job.adapter_name,
                    token_ids=job.token_ids,
                    loss_mask=job.loss_mask,
                    seq_no=job.seq_no,
                    qa_pairs=job.qa_pairs,
                    train_rollout=job.train_rollout,
                    system_prompt=job.system_prompt,
                    tools=job.tools,
                )
                # Cache the payload so an exact replay of this seq_no (client retry after
                # a lost response) can be answered without re-training.
                state.commit_update(job.seq_no, result, fingerprint)
                results[job.rollout_id] = result
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
        if not _locally_finite(tensors):
            raise FloatingPointError("non-finite TTT v2 checkpoint tensors")
        path = checkpoint_rollout_dir(self.ckpt_root, state.rollout_id) / f"v{state.version}"
        if self.world.is_master:
            tmp = path.with_name(f"{path.name}.tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            try:
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
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=self.ctrl_pg)
        return path
