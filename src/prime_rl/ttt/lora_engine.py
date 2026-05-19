from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoModelForCausalLM

from prime_rl.trainer.weights import load_state_dict
from prime_rl.utils.logger import get_logger

ActiveAdapters = str | None
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
    adapter: dict[str, LoRAWeights]
    optimizer: torch.optim.Optimizer
    version: int = 0
    turn_idx: int = 0
    pending_token_ids: list[int] = field(default_factory=list)
    latest_adapter: dict[str, Any] | None = None
    materialized_adapters: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default=None, repr=False)

    def to_device(self, device: torch.device) -> None:
        for weights in self.adapter.values():
            weights.to(device)
        for state in self.optimizer.state.values():
            self._move_optimizer_state(state, device)

    def to_cpu(self) -> None:
        self.to_device(torch.device("cpu"))

    def loaded_adapters(self) -> list[dict[str, Any]]:
        return [adapter for adapter in self.materialized_adapters if adapter.get("loaded_into_vllm")]

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
        update_every_tokens: int,
        max_grad_norm: float | None,
        device: str,
        dtype: torch.dtype,
        vllm_admin_base_urls: list[str],
        max_concurrent_sessions: int = 64,
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
        self.update_every_tokens = update_every_tokens
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.dtype = dtype
        self.vllm_admin_base_urls = [url.rstrip("/") for url in vllm_admin_base_urls]
        self.max_concurrent_sessions = max_concurrent_sessions
        self.load_adapters_into_vllm = load_adapters_into_vllm
        self.unload_vllm_adapters = unload_vllm_adapters
        self.base_step = 0
        self.sessions: dict[str, TTTSession] = {}

        logger = get_logger()
        logger.info(f"Loading TTT learner base model from {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        ).to(self.device)
        logger.info(f"TTT learner base model loaded on {self.device}")
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
        logger.info(f"Registered TTT LoRA hooks for {len(self.modules)} linear modules")

        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    def _matches_target(self, module_name: str) -> bool:
        parts = module_name.split(".")
        return any(pattern in parts or module_name.endswith(pattern) for pattern in self.target_modules)

    def _make_hook(self, module_name: str):
        def hook(module: nn.Linear, inputs: tuple[Any, ...], output: Tensor) -> Tensor:
            active = _ACTIVE_ADAPTERS.get()
            if active is None:
                return output
            session = self.sessions.get(active)
            if session is None:
                return output
            x = inputs[0]
            weights = session.adapter.get(module_name)
            if weights is None:
                return output
            x2 = torch.matmul(x.to(torch.float32), weights.a.to(torch.float32).transpose(0, 1))
            delta = torch.matmul(x2, weights.b.to(torch.float32).transpose(0, 1)) * self.scaling
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
        if len(self.sessions) >= self.max_concurrent_sessions:
            raise RuntimeError(
                f"TTT learner has {len(self.sessions)} active sessions, "
                f"which reaches max_concurrent_sessions={self.max_concurrent_sessions}."
            )
        adapter = self._new_lora_dict()
        params = [p for weights in adapter.values() for p in (weights.a, weights.b)]
        session = TTTSession(
            session_id=session_id,
            adapter=adapter,
            optimizer=torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay),
        )
        self.sessions[session_id] = session
        return session

    def load_base_weights(self, weight_dir: Path | None, step: int) -> None:
        start = time.perf_counter()
        if weight_dir is not None and weight_dir.exists():
            state = load_state_dict(weight_dir)
            self.model.load_state_dict(state, strict=False)
        self.base_step = step
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.train()
        get_logger().info(
            f"TTT timing load_base_weights step={step} weight_dir={weight_dir} elapsed={time.perf_counter() - start:.3f}s"
        )

    @contextlib.contextmanager
    def active(self, session_id: str):
        token = _ACTIVE_ADAPTERS.set(session_id)
        try:
            yield
        finally:
            _ACTIVE_ADAPTERS.reset(token)

    def append_and_train(self, session: TTTSession, token_ids: list[int]) -> dict[str, Any]:
        start = time.perf_counter()
        session.pending_token_ids.extend(int(token_id) for token_id in token_ids)
        losses: list[float] = []
        trained_token_count = 0
        initial_pending = len(session.pending_token_ids)
        while len(session.pending_token_ids) >= self.update_every_tokens:
            chunk = session.pending_token_ids[: self.update_every_tokens]
            del session.pending_token_ids[: self.update_every_tokens]
            losses.append(self._train_chunk(session, chunk))
            trained_token_count += len(chunk)
            session.version += 1
            session.latest_adapter = None
        elapsed = time.perf_counter() - start
        if losses or elapsed > 1.0:
            get_logger().info(
                f"TTT timing append_and_train session={session.session_id} input_tokens={len(token_ids)} "
                f"initial_pending={initial_pending} trained_chunks={len(losses)} "
                f"trained_tokens={trained_token_count} pending_tokens={len(session.pending_token_ids)} "
                f"version={session.version} elapsed={elapsed:.3f}s loss={(losses[-1] if losses else 0.0):.6f}"
            )
        return {
            "trained_chunks": len(losses),
            "trained_token_count": trained_token_count,
            "pending_token_count": len(session.pending_token_ids),
            "loss": losses[-1] if losses else 0.0,
        }

    async def append_and_train_with_replay_spans(
        self,
        session: TTTSession,
        token_ids: list[int],
        replay_mask: list[bool] | None,
        turn_idx: int,
    ) -> dict[str, Any]:
        """Append prompt tokens and emit causal adapter spans for selected replay tokens.

        The replay span adapter is always the LoRA state *before* the chunk
        containing those prompt tokens is learned. This preserves online TTT
        causality for prompt-side auxiliary losses.
        """
        if replay_mask is not None and len(replay_mask) != len(token_ids):
            raise ValueError(f"replay_mask length {len(replay_mask)} does not match token_ids length {len(token_ids)}.")

        start = time.perf_counter()
        pending_items: list[tuple[int, int | None]] = [(int(token_id), None) for token_id in session.pending_token_ids]
        pending_items.extend((int(token_id), idx) for idx, token_id in enumerate(token_ids))
        initial_pending = len(session.pending_token_ids)
        losses: list[float] = []
        trained_token_count = 0
        replay_spans: list[dict[str, Any]] = []

        while len(pending_items) >= self.update_every_tokens:
            chunk_items = pending_items[: self.update_every_tokens]
            del pending_items[: self.update_every_tokens]
            replay_indices = self._selected_new_indices(chunk_items, replay_mask)
            if replay_indices:
                meta = await self._current_replay_adapter(session, turn_idx, set_latest=False)
                replay_spans.append(self._replay_span(replay_indices, meta))

            chunk = [token_id for token_id, _ in chunk_items]
            losses.append(await asyncio.to_thread(self._train_chunk, session, chunk))
            trained_token_count += len(chunk)
            session.version += 1
            session.latest_adapter = None

        tail_replay_indices = self._selected_new_indices(pending_items, replay_mask)
        if tail_replay_indices:
            meta = await self._current_replay_adapter(session, turn_idx, set_latest=True)
            replay_spans.append(self._replay_span(tail_replay_indices, meta))

        session.pending_token_ids = [token_id for token_id, _ in pending_items]
        elapsed = time.perf_counter() - start
        if losses or replay_spans or elapsed > 1.0:
            get_logger().info(
                f"TTT timing append_and_train_with_replay session={session.session_id} input_tokens={len(token_ids)} "
                f"initial_pending={initial_pending} trained_chunks={len(losses)} "
                f"trained_tokens={trained_token_count} pending_tokens={len(session.pending_token_ids)} "
                f"replay_spans={len(replay_spans)} version={session.version} elapsed={elapsed:.3f}s "
                f"loss={(losses[-1] if losses else 0.0):.6f}"
            )
        return {
            "trained_chunks": len(losses),
            "trained_token_count": trained_token_count,
            "pending_token_count": len(session.pending_token_ids),
            "loss": losses[-1] if losses else 0.0,
            "prompt_replay_spans": replay_spans,
        }

    def _selected_new_indices(
        self,
        items: list[tuple[int, int | None]],
        replay_mask: list[bool] | None,
    ) -> list[int]:
        if replay_mask is None:
            return []
        return [idx for _, idx in items if idx is not None and replay_mask[idx]]

    def _replay_span(self, replay_indices: list[int], meta: dict[str, Any] | None) -> dict[str, Any]:
        span = {
            "new_start": min(replay_indices),
            "new_end": max(replay_indices) + 1,
            "adapter_name": None,
            "adapter_path": None,
            "adapter_kind": "base",
            "base_step": self.base_step,
            "adapter_version": 0,
        }
        if meta:
            span.update(
                {
                    "adapter_name": meta.get("adapter_name"),
                    "adapter_path": meta.get("adapter_path"),
                    "adapter_kind": meta.get("adapter_kind"),
                    "base_step": meta.get("base_step"),
                    "adapter_version": meta.get("version"),
                }
            )
        return span

    async def _current_replay_adapter(
        self,
        session: TTTSession,
        turn_idx: int,
        *,
        set_latest: bool,
    ) -> dict[str, Any] | None:
        if session.version <= 0:
            return None
        if session.latest_adapter is not None and session.latest_adapter.get("version") == session.version:
            return session.latest_adapter
        name = f"ttt-{session.session_id[:12]}-t{turn_idx}-prompt-v{session.version}-b{self.base_step}"
        return await self.materialize(
            session,
            name=name,
            load_into_vllm=False,
            turn_idx=turn_idx,
            adapter_kind="prompt_replay_snapshot",
            set_latest=set_latest,
        )

    def _train_chunk(self, session: TTTSession, token_ids: list[int]) -> float:
        if len(token_ids) < 2:
            return 0.0
        start = time.perf_counter()
        loss_value = 0.0
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        for _ in range(self.steps_per_update):
            session.optimizer.zero_grad(set_to_none=True)
            with self.active(session.session_id):
                out = self.model(input_ids=input_ids, labels=input_ids)
                loss = out.loss
            loss.backward()
            if self.max_grad_norm is not None:
                params = [p for weights in session.adapter.values() for p in (weights.a, weights.b)]
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            session.optimizer.step()
            session.optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
        get_logger().info(
            f"TTT timing train_chunk session={session.session_id} tokens={len(token_ids)} "
            f"steps={self.steps_per_update} elapsed={time.perf_counter() - start:.3f}s loss={loss_value:.6f}"
        )
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

    def _adapter_state(self, session: TTTSession) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for module_name in self.modules:
            weights = session.adapter[module_name]
            a = weights.a.detach().to("cpu").contiguous()
            b = (weights.b.detach().to("cpu") * self.scaling).contiguous()
            prefix = f"base_model.model.{module_name}"
            state[f"{prefix}.lora_A.weight"] = a
            state[f"{prefix}.lora_B.weight"] = b
        return state

    async def materialize(
        self,
        session: TTTSession,
        name: str,
        load_into_vllm: bool,
        turn_idx: int,
        *,
        adapter_kind: str = "snapshot",
        set_latest: bool = True,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        path = self.adapter_dir / name
        path.mkdir(parents=True, exist_ok=True)
        state_start = time.perf_counter()
        save_file(self._adapter_state(session), path / "adapter_model.safetensors", metadata={"format": "pt"})
        state_elapsed = time.perf_counter() - state_start
        with open(path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(self._adapter_config(rank=self.rank, alpha=self.rank), f, indent=2)
        meta = {
            "adapter_name": name,
            "adapter_path": path.as_posix(),
            "adapter_kind": adapter_kind,
            "loaded_into_vllm": bool(load_into_vllm and self.load_adapters_into_vllm),
            "rank": self.rank,
            "base_step": self.base_step,
            "version": session.version,
            "session_id": session.session_id,
            "turn_idx": turn_idx,
        }
        if load_into_vllm and self.load_adapters_into_vllm:
            load_start = time.perf_counter()
            await self._load_vllm_adapter(name, path)
            load_elapsed = time.perf_counter() - load_start
        else:
            load_elapsed = 0.0
        session.materialized_adapters.append(meta)
        if set_latest:
            session.latest_adapter = meta
        get_logger().info(
            f"TTT timing materialize session={session.session_id} turn={turn_idx} adapter={name} "
            f"version={session.version} path={path} save={state_elapsed:.3f}s "
            f"vllm_load={load_elapsed:.3f}s total={time.perf_counter() - start:.3f}s"
        )
        return meta

    async def _load_vllm_adapter(self, name: str, path: Path) -> None:
        if not self.vllm_admin_base_urls:
            return
        start = time.perf_counter()
        payload = {"lora_name": name, "lora_path": path.as_posix()}
        async with httpx.AsyncClient(timeout=120.0) as client:
            import asyncio

            results = await asyncio.gather(
                *(client.post(f"{url}/load_lora_adapter", json=payload) for url in self.vllm_admin_base_urls)
            )
            for response in results:
                response.raise_for_status()
        get_logger().info(
            f"TTT timing vllm_load adapter={name} urls={len(self.vllm_admin_base_urls)} "
            f"elapsed={time.perf_counter() - start:.3f}s"
        )

    async def ensure_vllm_loaded(self, meta: dict[str, Any] | None) -> None:
        if not meta or meta.get("loaded_into_vllm") or not self.load_adapters_into_vllm:
            return
        name = meta.get("adapter_name")
        path = meta.get("adapter_path")
        if not name or not path:
            return
        await self._load_vllm_adapter(str(name), Path(str(path)))
        meta["loaded_into_vllm"] = True

    async def unload_vllm_adapter(self, name: str) -> None:
        if not (self.unload_vllm_adapters and self.vllm_admin_base_urls):
            return
        start = time.perf_counter()
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
        get_logger().info(
            f"TTT timing vllm_unload adapter={name} urls={len(self.vllm_admin_base_urls)} "
            f"elapsed={time.perf_counter() - start:.3f}s"
        )

    def mark_vllm_unloaded(self, session: TTTSession, name: str) -> None:
        for adapter in session.materialized_adapters:
            if adapter.get("adapter_name") == name:
                adapter["loaded_into_vllm"] = False
        if session.latest_adapter and session.latest_adapter.get("adapter_name") == name:
            session.latest_adapter["loaded_into_vllm"] = False

    async def unload_session_loaded_adapters(self, session: TTTSession | None) -> None:
        if session is None:
            return
        for adapter in session.loaded_adapters():
            if adapter.get("loaded_into_vllm"):
                await self.unload_vllm_adapter(str(adapter["adapter_name"]))
