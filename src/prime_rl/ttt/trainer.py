"""The TTT training core: one base model, one PEFT LoRA adapter per rollout.

`TTTTrainer` owns the base model copy and a registry of live adapters. An update swaps the
rollout's adapter weights onto the shared PEFT wrapper, runs `steps_per_update` gradient
steps on the branch's flat token sequence (CE loss on the masked positions), and writes a
PEFT-format checkpoint — the exact format vLLM's `/load_lora_adapter` consumes. Optimizer
state persists per rollout across its updates (the adapter learns continuously) and is
dropped on release.

One adapter set is *resident* in the PEFT wrapper at a time; per-rollout states are swapped
in/out around each update (cheap: rank-r matrices). Concurrency is therefore serialized in
the trainer itself; the server layer bounds queuing.

No tokenizer anywhere: the service consumes the exact token ids the inference engine saw.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.utils.logger import get_logger


@dataclass
class AdapterState:
    """One rollout's training state between updates."""

    rollout_id: str
    adapter_name: str
    version: int = 0
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    """The adapter's lora_A/lora_B tensors (CPU), swapped onto the model per update."""
    optim_state: dict | None = None
    """The optimizer state dict, restored per update so momentum carries across updates."""


class TTTTrainer:
    def __init__(self, config: TTTServiceConfig):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        self.config = config
        self.logger = get_logger()
        self.device = torch.device(config.model.device)
        self.ckpt_root = config.output_dir / "ttt"

        self.logger.info(f"Loading base model {config.model.name} on {self.device}")
        model = AutoModelForCausalLM.from_pretrained(config.model.name, dtype=torch.bfloat16)
        model.to(self.device)
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # With frozen embeddings, checkpointed blocks see no-grad inputs and backward
            # never reaches the adapters — require grads on the embedding output.
            model.enable_input_require_grads()
            model.config.use_cache = False
        self.peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, self.peft_config)
        self.model.train()
        # Zero-init template: a fresh adapter's tensors (B=0 ⇒ identity), cloned per rollout.
        self._template = {
            name: param.detach().to("cpu", copy=True)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.adapters: dict[str, AdapterState] = {}
        self.logger.info(
            f"TTT trainer ready (rank={config.lora.rank}, alpha={config.lora.alpha}, "
            f"targets={config.lora.target_modules}, {len(self._template)} adapter tensors)"
        )

    # -- adapter registry -------------------------------------------------------------------

    def _get_or_create(self, rollout_id: str, adapter_name: str) -> AdapterState:
        state = self.adapters.get(rollout_id)
        if state is None:
            state = AdapterState(
                rollout_id=rollout_id,
                adapter_name=adapter_name,
                tensors={name: t.clone() for name, t in self._template.items()},
            )
            self.adapters[rollout_id] = state
        return state

    def release(self, rollout_id: str) -> AdapterState | None:
        """Drop a rollout's training state; optionally delete its checkpoints."""
        state = self.adapters.pop(rollout_id, None)
        if not self.config.keep_checkpoints:
            shutil.rmtree(self.ckpt_root / rollout_id, ignore_errors=True)
        return state

    def _swap_in(self, state: AdapterState) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(state.tensors[name].to(self.device))

    def _swap_out(self, state: AdapterState) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    state.tensors[name] = param.detach().to("cpu", copy=True)

    def _optimizer(self, state: AdapterState) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        cfg = self.config.optim
        if cfg.type == "sgd":
            optim: torch.optim.Optimizer = torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            optim = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        if state.optim_state is not None:
            optim.load_state_dict(state.optim_state)
        return optim

    # -- the update -------------------------------------------------------------------------

    def update(
        self,
        rollout_id: str,
        adapter_name: str,
        token_ids: list[int],
        loss_mask: list[bool],
        seq_no: int,
    ) -> dict:
        """One TTT update: `steps_per_update` gradient steps on the branch's sequence, then
        a versioned PEFT checkpoint. Returns `{version, loss, ckpt_path, num_loss_tokens}`.

        `seq_no` must be exactly `state.version + 1` — the rollout side blocks per update,
        so a mismatch means a lost/duplicated update and the state can't be trusted."""
        if len(token_ids) != len(loss_mask):
            raise ValueError(f"token_ids ({len(token_ids)}) and loss_mask ({len(loss_mask)}) must align")
        if len(token_ids) < 2:
            raise ValueError("need at least 2 tokens to form a next-token target")
        state = self._get_or_create(rollout_id, adapter_name)
        if seq_no != state.version + 1:
            raise ValueError(f"out-of-order update for {rollout_id}: expected seq_no {state.version + 1}, got {seq_no}")

        start = time.perf_counter()
        self._swap_in(state)
        optim = self._optimizer(state)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        # Next-token loss: position i predicts token i+1, so shift the mask left — the
        # loss lands on the *predicting* positions of masked targets.
        labels = torch.tensor([token_ids[1:]], dtype=torch.long, device=self.device)
        target_mask = torch.tensor([loss_mask[1:]], dtype=torch.bool, device=self.device)
        num_loss_tokens = int(target_mask.sum())
        if num_loss_tokens == 0:
            raise ValueError("loss_mask selects no target tokens")

        loss_value = 0.0
        for _ in range(self.config.steps_per_update):
            optim.zero_grad(set_to_none=True)
            logits = self.model(input_ids=input_ids).logits[:, :-1]
            losses = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                reduction="none",
            )
            loss = (losses * target_mask.reshape(-1)).sum() / num_loss_tokens
            loss.backward()
            if self.config.optim.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.config.optim.max_norm,
                )
            optim.step()
            loss_value = float(loss.detach())

        state.optim_state = optim.state_dict()
        self._swap_out(state)
        state.version = seq_no
        ckpt_path = self.save_checkpoint(state)
        seconds = time.perf_counter() - start
        self.logger.info(
            f"ttt update {rollout_id} v{seq_no}: loss={loss_value:.4f} "
            f"({num_loss_tokens}/{len(token_ids)} loss tokens, {seconds:.2f}s)"
        )
        return {
            "version": state.version,
            "loss": loss_value,
            "ckpt_path": str(ckpt_path),
            "num_loss_tokens": num_loss_tokens,
        }

    def save_checkpoint(self, state: AdapterState) -> Path:
        """Write the adapter as a PEFT checkpoint (`adapter_config.json` +
        `adapter_model.safetensors`) — the format vLLM's LoRA loader consumes. Atomic:
        written to a tmp dir, then renamed."""
        import safetensors.torch

        path = self.ckpt_root / state.rollout_id / f"v{state.version}"
        tmp = path.with_name(f"{path.name}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        # PEFT save-format keys: strip the wrapper prefix and the active-adapter infix
        # ("...lora_A.default.weight" -> "...lora_A.weight"), keep "base_model.model.".
        tensors = {name.replace(".default.", "."): tensor for name, tensor in state.tensors.items()}
        safetensors.torch.save_file(tensors, tmp / "adapter_model.safetensors")
        self.peft_config.save_pretrained(str(tmp))
        shutil.rmtree(path, ignore_errors=True)
        tmp.rename(path)
        return path
