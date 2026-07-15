"""The TTT v1 training core: one base model, one PEFT LoRA adapter per rollout.

An update swaps the rollout's adapter onto the shared PEFT wrapper, runs
`steps_per_update` gradient steps (CE on the masked positions of the raw branch tokens
and/or rendered Q&A pairs), and writes a PEFT-format checkpoint that vLLM's
`/load_lora_adapter` consumes. One adapter resident at a time — the trainer serializes.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.ttt.identity import (
    checkpoint_rollout_dir,
    update_fingerprint,
    validate_adapter_name,
    validate_rollout_id,
)
from prime_rl.ttt.validation import validate_qa_pairs
from prime_rl.utils.logger import get_logger
from prime_rl.utils.qa_render import assert_prefix_stable_template, tokenize_qa_pairs


def _load_qa_tokenizer(config: TTTServiceConfig):
    """Load the service's configured Q&A tokenizer (template override, pad=eos) via the
    trainer's setup_tokenizer — same contract v2 uses."""
    from prime_rl.trainer.model import setup_tokenizer

    assert config.tokenizer.name is not None  # TTTServiceConfig fills this from model.name
    return setup_tokenizer(config.tokenizer)


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
    last_result: dict | None = None
    """The successful result payload of the last applied update."""
    last_fingerprint: str | None = None
    """Semantic fingerprint proving that a same-sequence retry is exact."""


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
        self._tokenizer = None  # lazy — only Q&A training needs it
        # Startup canary: a chat template that isn't prefix-stable would silently skip every
        # Q&A pair — fail the service at launch instead. A configured tokenizer that cannot
        # load stays lazy (Q&A requests will then fail loudly on first use).
        try:
            tokenizer = _load_qa_tokenizer(config)
        except Exception:
            pass
        else:
            assert_prefix_stable_template(tokenizer)
            self._tokenizer = tokenizer
        self.logger.info(
            f"TTT trainer ready (rank={config.lora.rank}, alpha={config.lora.alpha}, "
            f"targets={config.lora.target_modules}, {len(self._template)} adapter tensors)"
        )

    # -- adapter registry -------------------------------------------------------------------

    def _get_or_create(self, rollout_id: str, adapter_name: str) -> AdapterState:
        validate_rollout_id(rollout_id)
        validate_adapter_name(adapter_name)
        state = self.adapters.get(rollout_id)
        if state is not None and state.adapter_name != adapter_name:
            raise ValueError(
                f"rollout {rollout_id!r} is bound to adapter {state.adapter_name!r}, not {adapter_name!r}"
            )
        if state is None:
            state = AdapterState(
                rollout_id=rollout_id,
                adapter_name=adapter_name,
                tensors={name: t.clone() for name, t in self._template.items()},
            )
            self.adapters[rollout_id] = state
        return state

    def release(self, rollout_id: str, adapter_name: str | None = None) -> AdapterState | None:
        """Drop a rollout's training state; optionally delete its checkpoints."""
        rollout_dir = checkpoint_rollout_dir(self.ckpt_root, rollout_id)
        state = self.adapters.get(rollout_id)
        if state is not None and adapter_name is not None and state.adapter_name != adapter_name:
            raise ValueError(
                f"rollout {rollout_id!r} is bound to adapter {state.adapter_name!r}, not {adapter_name!r}"
            )
        state = self.adapters.pop(rollout_id, None)
        if not self.config.keep_checkpoints:
            shutil.rmtree(rollout_dir, ignore_errors=True)
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

    def _tokenize_qa(self, qa_pairs, system_prompt=None, tools=None):
        if self._tokenizer is None:
            self._tokenizer = _load_qa_tokenizer(self.config)
        return tokenize_qa_pairs(self._tokenizer, qa_pairs, system_prompt, tools)

    def update(
        self,
        rollout_id: str,
        adapter_name: str,
        token_ids: list[int],
        loss_mask: list[bool],
        seq_no: int,
        qa_pairs: list[dict] | None = None,
        train_rollout: bool = True,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """One TTT update: `steps_per_update` gradient steps, then a versioned PEFT
        checkpoint. The training set is the branch's raw sequence (`train_rollout`), the
        rendered Q&A pairs (`qa_pairs`), or both; each gradient step accumulates over every
        sequence, normalized by the step's total loss-token count. Returns
        `{version, loss, ckpt_path, num_loss_tokens}`.

        `seq_no` must be exactly `state.version + 1` — the rollout side blocks per update,
        so a mismatch means a lost/duplicated update and the state can't be trusted."""
        if len(token_ids) != len(loss_mask):
            raise ValueError(f"token_ids ({len(token_ids)}) and loss_mask ({len(loss_mask)}) must align")
        if len(token_ids) < 2:
            raise ValueError("need at least 2 tokens to form a next-token target")
        # Validate seq_no BEFORE allocating: a bad first update must not leave a fresh
        # (never-trained) AdapterState behind that skews later expected seq_nos.
        validate_rollout_id(rollout_id)
        validate_adapter_name(adapter_name)
        fingerprint = update_fingerprint(
            rollout_id=rollout_id,
            adapter_name=adapter_name,
            token_ids=token_ids,
            loss_mask=loss_mask,
            seq_no=seq_no,
            qa_pairs=qa_pairs,
            train_rollout=train_rollout,
            system_prompt=system_prompt,
            tools=tools,
        )
        existing = self.adapters.get(rollout_id)
        if existing is not None and existing.adapter_name != adapter_name:
            raise ValueError(
                f"rollout {rollout_id!r} is bound to adapter {existing.adapter_name!r}, not {adapter_name!r}"
            )
        if existing is not None and seq_no == existing.version and existing.last_result is not None:
            if fingerprint != existing.last_fingerprint:
                raise ValueError(f"replayed update for {rollout_id} does not match the cached request")
            return existing.last_result
        expected = (existing.version if existing is not None else 0) + 1
        if seq_no != expected:
            raise ValueError(f"out-of-order update for {rollout_id}: expected seq_no {expected}, got {seq_no}")
        if qa_pairs:
            validate_qa_pairs(qa_pairs)  # 409 before any state mutation, not a KeyError-500
        state = self._get_or_create(rollout_id, adapter_name)

        sequences: list[tuple[list[int], list[bool]]] = []
        if train_rollout:
            sequences.append((token_ids, loss_mask))
        if qa_pairs:
            sequences.extend(self._tokenize_qa(qa_pairs, system_prompt, tools))
        sequences = [(ids, mask) for ids, mask in sequences if len(ids) >= 2 and any(mask[1:])]
        if not sequences:
            raise ValueError("no trainable sequences (empty loss masks / QA rendered empty)")

        start = time.perf_counter()
        self._swap_in(state)
        optim = self._optimizer(state)

        # Next-token loss per sequence: position i predicts token i+1, so shift the mask
        # left — the loss lands on the *predicting* positions of masked targets.
        tensors = []
        for ids, mask in sequences:
            tensors.append(
                (
                    torch.tensor([ids], dtype=torch.long, device=self.device),
                    torch.tensor([ids[1:]], dtype=torch.long, device=self.device),
                    torch.tensor([mask[1:]], dtype=torch.bool, device=self.device),
                )
            )
        num_loss_tokens = int(sum(m.sum() for _, _, m in tensors))

        loss_value = 0.0
        for _ in range(self.config.steps_per_update):
            optim.zero_grad(set_to_none=True)
            step_loss = 0.0
            for input_ids, labels, target_mask in tensors:
                logits = self.model(input_ids=input_ids).logits[:, :-1]
                losses = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    labels.reshape(-1),
                    reduction="none",
                )
                loss = (losses * target_mask.reshape(-1)).sum() / num_loss_tokens
                loss.backward()
                step_loss += float(loss.detach())
            if self.config.optim.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.config.optim.max_norm,
                )
            optim.step()
            loss_value = step_loss

        state.optim_state = optim.state_dict()
        self._swap_out(state)
        state.version = seq_no
        ckpt_path = self.save_checkpoint(state)
        seconds = time.perf_counter() - start
        self.logger.info(
            f"ttt update {rollout_id} v{seq_no}: loss={loss_value:.4f} "
            f"({num_loss_tokens} loss tokens over {len(sequences)} sequence(s), {seconds:.2f}s)"
        )
        # Cache the payload so an exact replay of this seq_no (client retry after a lost
        # response) can be answered without re-training.
        state.last_result = {
            "version": state.version,
            "loss": loss_value,
            "ckpt_path": str(ckpt_path),
            "num_loss_tokens": num_loss_tokens,
        }
        state.last_fingerprint = fingerprint
        return state.last_result

    def save_checkpoint(self, state: AdapterState) -> Path:
        """Write the adapter as a PEFT checkpoint (`adapter_config.json` +
        `adapter_model.safetensors`) — the format vLLM's LoRA loader consumes. Atomic:
        written to a tmp dir, then renamed."""
        import safetensors.torch

        path = checkpoint_rollout_dir(self.ckpt_root, state.rollout_id) / f"v{state.version}"
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
