from __future__ import annotations

import contextlib
import contextvars
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoModelForCausalLM

from prime_rl.trainer.weights import load_state_dict

ActiveAdapters = tuple[str, tuple[Literal["prompt", "completion"], ...]] | None
_ACTIVE_ADAPTERS: contextvars.ContextVar[ActiveAdapters] = contextvars.ContextVar("ttt_active_adapters", default=None)


@dataclass
class LoRAWeights:
    a: nn.Parameter
    b: nn.Parameter

    def to(self, device: torch.device) -> None:
        for param in (self.a, self.b):
            param.data = param.data.to(device=device, non_blocking=True)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device=device, non_blocking=True)


@dataclass
class TTTSession:
    session_id: str
    prompt: dict[str, LoRAWeights]
    completion: dict[str, LoRAWeights]
    prompt_optimizer: torch.optim.Optimizer
    completion_optimizer: torch.optim.Optimizer
    prompt_version: int = 0
    completion_version: int = 0
    turn_idx: int = 0
    final_prompt_adapter: dict[str, Any] | None = None
    materialized_adapters: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default=None, repr=False)

    def to_device(self, device: torch.device) -> None:
        for weights in [*self.prompt.values(), *self.completion.values()]:
            weights.to(device)
        for optimizer in (self.prompt_optimizer, self.completion_optimizer):
            for state in optimizer.state.values():
                self._move_optimizer_state(state, device)

    def to_cpu(self) -> None:
        self.to_device(torch.device("cpu"))

    def combined_adapters(self) -> list[dict[str, Any]]:
        return [adapter for adapter in self.materialized_adapters if adapter.get("adapter_kind") == "combined"]

    @classmethod
    def _move_optimizer_state(cls, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = cls._move_optimizer_state(item, device)
            return value
        if isinstance(value, list):
            for idx, item in enumerate(value):
                value[idx] = cls._move_optimizer_state(item, device)
            return value
        if isinstance(value, tuple):
            return tuple(cls._move_optimizer_state(item, device) for item in value)
        return value


class HookedLoRAEngine:
    """Small experimental LoRA engine for rollout-time TTT.

    This intentionally avoids Prime-RL trainer/FSDP internals. It keeps a
    single HF model replica in the learner process and injects rollout-local
    LoRA deltas with module hooks selected by a context variable.
    """

    def __init__(
        self,
        model_name: str,
        adapter_dir: Path,
        target_modules: list[str],
        rank: int,
        alpha: float,
        dropout: float,
        lr: float,
        weight_decay: float,
        steps_per_update: int,
        max_grad_norm: float | None,
        prompt_loss_weight: float,
        completion_loss_weight: float,
        device: str,
        dtype: torch.dtype,
        vllm_admin_base_urls: list[str],
        load_adapters_into_vllm: bool = True,
        unload_vllm_adapters: bool = True,
    ) -> None:
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_p = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.steps_per_update = steps_per_update
        self.max_grad_norm = max_grad_norm
        self.prompt_loss_weight = prompt_loss_weight
        self.completion_loss_weight = completion_loss_weight
        self.device = torch.device(device)
        self.dtype = dtype
        self.vllm_admin_base_urls = [url.rstrip("/") for url in vllm_admin_base_urls]
        self.load_adapters_into_vllm = load_adapters_into_vllm
        self.unload_vllm_adapters = unload_vllm_adapters
        self.base_step = 0
        self.sessions: dict[str, TTTSession] = {}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False

        self.modules: dict[str, nn.Linear] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._matches_target(name):
                self.modules[name] = module
                module.register_forward_hook(self._make_hook(name))
        if not self.modules:
            raise ValueError(f"No linear TTT LoRA target modules matched {target_modules!r}.")

        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    def _matches_target(self, module_name: str) -> bool:
        parts = module_name.split(".")
        return any(pattern in parts or module_name.endswith(pattern) for pattern in self.target_modules)

    def _make_hook(self, module_name: str):
        def hook(module: nn.Linear, inputs: tuple[Any, ...], output: Tensor) -> Tensor:
            active = _ACTIVE_ADAPTERS.get()
            if active is None:
                return output
            session_id, kinds = active
            session = self.sessions.get(session_id)
            if session is None:
                return output
            x = inputs[0]
            delta = None
            for kind in kinds:
                weights = getattr(session, kind).get(module_name)
                if weights is None:
                    continue
                x2 = torch.matmul(x.to(torch.float32), weights.a.to(torch.float32).transpose(0, 1))
                part = torch.matmul(x2, weights.b.to(torch.float32).transpose(0, 1)) * self.scaling
                delta = part if delta is None else delta + part
            if delta is None:
                return output
            return output + delta.to(dtype=output.dtype, device=output.device)

        return hook

    def _new_lora(self, module: nn.Linear) -> LoRAWeights:
        a = nn.Parameter(torch.empty(self.rank, module.in_features, device=self.device, dtype=self.dtype))
        b = nn.Parameter(torch.zeros(module.out_features, self.rank, device=self.device, dtype=self.dtype))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        return LoRAWeights(a=a, b=b)

    def _new_lora_dict(self) -> dict[str, LoRAWeights]:
        return {name: self._new_lora(module) for name, module in self.modules.items()}

    def get_or_create_session(self, session_id: str) -> TTTSession:
        session = self.sessions.get(session_id)
        if session is not None:
            return session
        prompt = self._new_lora_dict()
        completion = self._new_lora_dict()
        prompt_params = [p for weights in prompt.values() for p in (weights.a, weights.b)]
        completion_params = [p for weights in completion.values() for p in (weights.a, weights.b)]
        session = TTTSession(
            session_id=session_id,
            prompt=prompt,
            completion=completion,
            prompt_optimizer=torch.optim.AdamW(prompt_params, lr=self.lr, weight_decay=self.weight_decay),
            completion_optimizer=torch.optim.AdamW(completion_params, lr=self.lr, weight_decay=self.weight_decay),
        )
        self.sessions[session_id] = session
        return session

    def load_base_weights(self, weight_dir: Path | None, step: int) -> None:
        if weight_dir is not None and weight_dir.exists():
            state = load_state_dict(weight_dir)
            self.model.load_state_dict(state, strict=False)
        self.base_step = step
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.train()

    @contextlib.contextmanager
    def active(self, session_id: str, kinds: tuple[Literal["prompt", "completion"], ...]):
        token = _ACTIVE_ADAPTERS.set((session_id, kinds))
        try:
            yield
        finally:
            _ACTIVE_ADAPTERS.reset(token)

    def train_tokens(self, session: TTTSession, kind: Literal["prompt", "completion"], token_ids: list[int]) -> float:
        if len(token_ids) < 2:
            return 0.0
        loss_weight = self.prompt_loss_weight if kind == "prompt" else self.completion_loss_weight
        if loss_weight <= 0:
            return 0.0
        optimizer = session.prompt_optimizer if kind == "prompt" else session.completion_optimizer
        loss_value = 0.0
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        active_kinds: tuple[Literal["prompt", "completion"], ...] = ("prompt", "completion")
        for _ in range(self.steps_per_update):
            session.prompt_optimizer.zero_grad(set_to_none=True)
            session.completion_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            with self.active(session.session_id, active_kinds):
                out = self.model(input_ids=input_ids, labels=input_ids)
                loss = out.loss
            (loss * loss_weight).backward()
            if self.max_grad_norm is not None:
                params = [p for weights in getattr(session, kind).values() for p in (weights.a, weights.b)]
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            optimizer.step()
            session.prompt_optimizer.zero_grad(set_to_none=True)
            session.completion_optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
        if kind == "prompt":
            session.prompt_version += 1
        else:
            session.completion_version += 1
        return loss_value

    def _adapter_config(self, rank: int, alpha: int) -> dict[str, Any]:
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": self.model_name,
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted({name.split(".")[-1] for name in self.modules}),
            "modules_to_save": None,
        }

    def _state_for_kinds(
        self, session: TTTSession, kinds: tuple[Literal["prompt", "completion"], ...]
    ) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for module_name in self.modules:
            a_parts = []
            b_parts = []
            for kind in kinds:
                weights = getattr(session, kind)[module_name]
                a_parts.append(weights.a.detach().to("cpu"))
                b_parts.append((weights.b.detach().to("cpu") * self.scaling))
            a = torch.cat(a_parts, dim=0).contiguous()
            b = torch.cat(b_parts, dim=1).contiguous()
            prefix = f"base_model.model.{module_name}"
            state[f"{prefix}.lora_A.weight"] = a
            state[f"{prefix}.lora_B.weight"] = b
        return state

    async def materialize(
        self,
        session: TTTSession,
        name: str,
        kinds: tuple[Literal["prompt", "completion"], ...],
        load_into_vllm: bool,
        adapter_kind: Literal["combined", "final_prompt"],
        turn_idx: int,
    ) -> dict[str, Any]:
        path = self.adapter_dir / name
        path.mkdir(parents=True, exist_ok=True)
        rank = self.rank * len(kinds)
        save_file(self._state_for_kinds(session, kinds), path / "adapter_model.safetensors", metadata={"format": "pt"})
        with open(path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(self._adapter_config(rank=rank, alpha=rank), f, indent=2)
        meta = {
            "adapter_name": name,
            "adapter_path": path.as_posix(),
            "adapter_kind": adapter_kind,
            "loaded_into_vllm": bool(load_into_vllm and self.load_adapters_into_vllm),
            "rank": rank,
            "base_step": self.base_step,
            "prompt_version": session.prompt_version,
            "completion_version": session.completion_version,
            "session_id": session.session_id,
            "turn_idx": turn_idx,
        }
        if load_into_vllm and self.load_adapters_into_vllm:
            await self._load_vllm_adapter(name, path)
        session.materialized_adapters.append(meta)
        return meta

    async def _load_vllm_adapter(self, name: str, path: Path) -> None:
        if not self.vllm_admin_base_urls:
            return
        payload = {"lora_name": name, "lora_path": path.as_posix()}
        async with httpx.AsyncClient(timeout=120.0) as client:
            import asyncio

            results = await asyncio.gather(
                *(client.post(f"{url}/load_lora_adapter", json=payload) for url in self.vllm_admin_base_urls)
            )
            for response in results:
                response.raise_for_status()

    async def unload_vllm_adapter(self, name: str) -> None:
        if not (self.unload_vllm_adapters and self.vllm_admin_base_urls):
            return
        payload = {"lora_name": name}
        async with httpx.AsyncClient(timeout=120.0) as client:
            import asyncio

            responses = await asyncio.gather(
                *(client.post(f"{url}/v1/unload_lora_adapter", json=payload) for url in self.vllm_admin_base_urls),
                return_exceptions=True,
            )
            for response in responses:
                if isinstance(response, Exception):
                    continue
                if response.status_code not in (200, 404):
                    response.raise_for_status()

    async def unload_session_combined_adapters(self, session: TTTSession | None) -> None:
        if session is None:
            return
        for adapter in session.combined_adapters():
            if adapter.get("loaded_into_vllm"):
                await self.unload_vllm_adapter(str(adapter["adapter_name"]))
