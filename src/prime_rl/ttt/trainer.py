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

The raw-branch path needs no tokenizer: it consumes the exact token ids the inference
engine saw. Q&A training is the one exception — the pairs arrive as text (they are trained
*standalone*, without the branch context they were generated under, so the engine's
context-tokenization is meaningless for them) and are rendered with the base model's own
chat template, loss on the answer tokens. The tokenizer is lazy-loaded on first Q&A use.
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
from prime_rl.utils.qa_render import assert_prefix_stable_template, render_qa_pair


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
        self._tokenizer = None  # lazy — only Q&A training needs it
        # Startup canary: a chat template that isn't prefix-stable would silently skip every
        # Q&A pair — fail the service at launch instead. A model dir without tokenizer files
        # stays lazy (Q&A requests will then fail loudly on first use).
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.model.name)
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

    def _tokenize_qa(
        self,
        qa_pairs: list[dict],
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> list[tuple[list[int], list[bool]]]:
        """Render each Q&A pair standalone with the base model's chat template — no branch
        context (that's the point: the knowledge must come from the weights), but
        conditioned on the rollout's system prompt and tool schemas (chat templates render
        tools into the system block), so tool lessons are learned next to the tool
        descriptions. Loss on the answer tokens only; the system/question prefix is
        context. Returns `(token_ids, loss_mask)` per pair; pairs whose answer renders to
        nothing are skipped."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)

        template_kwargs: dict = {"tools": tools} if tools else {}
        head = [{"role": "system", "content": system_prompt}] if system_prompt else []
        sequences: list[tuple[list[int], list[bool]]] = []
        for pair in qa_pairs:
            if not str(pair.get("answer", "")).strip():
                continue  # a blank answer renders only template scaffold — nothing to learn
            conversation = [
                *head,
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]},
            ]
            rendered = render_qa_pair(self._tokenizer, conversation, template_kwargs)
            if rendered is None:
                continue  # non-prefix-stable render: skip rather than train on the full render
            full, prompt_len = rendered
            if len(full) - prompt_len < 1:
                continue
            mask = [False] * prompt_len + [True] * (len(full) - prompt_len)
            sequences.append((full, mask))
        return sequences

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
        existing = self.adapters.get(rollout_id)
        expected = (existing.version if existing is not None else 0) + 1
        if seq_no != expected:
            raise ValueError(f"out-of-order update for {rollout_id}: expected seq_no {expected}, got {seq_no}")
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
