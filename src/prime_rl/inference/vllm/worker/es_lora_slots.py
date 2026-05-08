from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.lora.model_manager import LoRAModelManager


@dataclass(frozen=True)
class ESLoRATensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name.replace("torch.", ""))


def _noise_like(theta: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=theta.device)
    generator.manual_seed(seed)
    return torch.randn(theta.shape, generator=generator, device=theta.device, dtype=torch.float32)


def _normalize_rewards(rewards: list[float], mode: str) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32)
    if mode == "none":
        return values
    centered = values - values.mean()
    if mode == "centered":
        return centered
    if mode != "zscore":
        raise ValueError(f"Unsupported reward normalization: {mode}")
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return torch.zeros_like(values)
    return centered / std


def _load_specs(payload: list[dict[str, Any]]) -> list[ESLoRATensorSpec]:
    return [
        ESLoRATensorSpec(
            name=str(item["name"]),
            shape=tuple(int(v) for v in item["shape"]),
            dtype=str(item["dtype"]),
            numel=int(item["numel"]),
        )
        for item in payload
    ]


def _unflatten_lora_tensors(
    flat: torch.Tensor,
    specs: list[ESLoRATensorSpec],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    offset = 0
    for spec in specs:
        dtype = _dtype_from_name(spec.dtype)
        tensors[spec.name] = flat[offset : offset + spec.numel].reshape(spec.shape).to(dtype=dtype)
        offset += spec.numel
    return tensors


class ESLoRASlotWorkerMixin:
    """vLLM worker extension mixin for ES-managed persistent LoRA slots."""

    def es_init_lora_slots(
        self,
        theta_path: str,
        specs_payload: list[dict[str, Any]],
        adapter_config: dict[str, Any],
        slots: list[dict[str, Any]],
    ) -> None:
        lora_manager = self.model_runner.lora_manager
        adapter_manager: LoRAModelManager = lora_manager._adapter_manager

        state = torch.load(Path(theta_path), map_location="cpu", weights_only=False)
        theta = state["theta"] if isinstance(state, dict) else state
        self.es_lora_theta = theta.to(device=self.device, dtype=torch.float32)
        self.es_lora_specs = _load_specs(specs_payload)
        self.es_lora_adapter_config = dict(adapter_config)
        max_position_embeddings = getattr(self, "max_position_embeddings", None)
        self.es_lora_peft_helper = PEFTHelper.from_dict(
            self.es_lora_adapter_config | {"vllm_max_position_embeddings": max_position_embeddings}
        )
        self.es_lora_peft_helper.validate_legal(lora_manager.lora_config)

        self.es_lora_slots = {int(slot["lora_int_id"]): str(slot["lora_name"]) for slot in slots}
        rank = int(self.es_lora_adapter_config["r"])
        for lora_int_id, lora_name in self.es_lora_slots.items():
            request = LoRARequest(lora_name=lora_name, lora_int_id=lora_int_id, lora_path="/prime_rl_es_slot")
            if lora_int_id not in lora_manager.list_adapters():
                lora_manager.add_dummy_lora(request, rank=rank)
            adapter_manager.activate_adapter(lora_int_id)

    def _es_lora_model_from_flat(self, lora_int_id: int, flat: torch.Tensor) -> LoRAModel:
        model = self.model_runner.lora_manager._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)
        tensors = _unflatten_lora_tensors(flat, self.es_lora_specs)
        return LoRAModel.from_lora_tensors(
            lora_model_id=lora_int_id,
            tensors=tensors,
            peft_helper=self.es_lora_peft_helper,
            device=str(self.device),
            dtype=self.model_runner.lora_manager.lora_config.lora_dtype,
            model_vocab_size=self.model_runner.lora_manager.vocab_size,
            weights_mapper=hf_to_vllm_mapper,
            skip_prefixes=lora_skip_prefixes,
        )

    @torch.no_grad()
    def es_materialize_lora_slots(self, slot_payload: list[dict[str, Any]], sigma: float) -> None:
        adapter_manager: LoRAModelManager = self.model_runner.lora_manager._adapter_manager
        for item in slot_payload:
            lora_int_id = int(item["lora_int_id"])
            seed = int(item["seed"])
            sign = int(item.get("sign", 1))
            if lora_int_id not in self.es_lora_slots:
                raise ValueError(f"Unknown ES LoRA slot id: {lora_int_id}")
            slot_index = adapter_manager.lora_index_to_id.index(lora_int_id)
            candidate_theta = self.es_lora_theta + sign * float(sigma) * _noise_like(self.es_lora_theta, seed)
            lora_model = self._es_lora_model_from_flat(lora_int_id, candidate_theta)
            adapter_manager._create_merged_loras_inplace(lora_model)
            for module_name, module in adapter_manager.modules.items():
                module_lora = adapter_manager._get_lora_layer_weights(lora_model, module_name)
                if module_lora is None:
                    module.reset_lora(slot_index)
                    continue
                module.set_lora(slot_index, module_lora.lora_a, module_lora.lora_b)
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def es_update_lora_theta(
        self,
        candidates_payload: list[dict[str, Any]],
        rewards: list[float],
        lr: float,
        normalization: str,
        mirrored: bool,
        sigma: float,
    ) -> None:
        if mirrored:
            by_seed: dict[int, dict[int, float]] = {}
            for item, reward in zip(candidates_payload, rewards, strict=True):
                by_seed.setdefault(int(item["seed"]), {})[int(item.get("sign", 1))] = float(reward)
            grad = torch.zeros_like(self.es_lora_theta)
            used_pairs = 0
            for seed, pair in by_seed.items():
                if 1 not in pair or -1 not in pair:
                    continue
                grad.add_(_noise_like(self.es_lora_theta, seed), alpha=(pair[1] - pair[-1]) / (2.0 * float(sigma)))
                used_pairs += 1
            if used_pairs:
                grad.div_(float(used_pairs))
        else:
            scores = _normalize_rewards(rewards, normalization).tolist()
            grad = torch.zeros_like(self.es_lora_theta)
            for item, score in zip(candidates_payload, scores, strict=True):
                grad.add_(_noise_like(self.es_lora_theta, int(item["seed"])), alpha=float(score))
            if candidates_payload:
                grad.div_(float(len(candidates_payload)))
        self.es_lora_theta.add_(grad, alpha=float(lr))
        torch.cuda.synchronize(self.device)
