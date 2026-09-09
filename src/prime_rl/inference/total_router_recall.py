"""Experimental vLLM 0.28 TRR capture over the existing routed-experts transport.

The int32 payload has shape [tokens, layers, 2 * top_k]: logical expert IDs
followed by the bit patterns of the FP32 combine weights. Both halves therefore
undergo exactly the same request slicing, truncation, and trainer packing.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def pack_routing(ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights.dtype != torch.float32 or weights.shape != ids.shape:
        raise ValueError("TRR requires FP32 weights with the same shape as expert IDs")
    return torch.cat((ids.to(torch.int32), weights.contiguous().view(torch.int32)), dim=-1)


def enable_total_router_capture():
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
        RoutedExpertsManager,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    if getattr(BaseRouter, "_prime_trr", False):
        return

    original_capture_init = RoutedExpertsCapturer.__init__
    original_manager_init = RoutedExpertsManager.__init__

    def check_config(config):
        if config.model_config.hf_text_config.model_type not in ("qwen3_moe", "glm4_moe"):
            raise ValueError("Experimental TRR supports Qwen3 MoE and GLM4 MoE only")
        if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.enable_expert_parallel:
            raise ValueError("Experimental TRR requires TP1 and EP1")
        if not config.model_config.enforce_eager:
            raise ValueError("Experimental TRR requires eager inference")

    def capture_init(self, max_num_batched_tokens, vllm_config, kv_cache_config):
        check_config(vllm_config)
        original_capture_init(self, max_num_batched_tokens, vllm_config, kv_cache_config)
        shape = (*self.device_buffer.shape[:-1], self.device_buffer.shape[-1] * 2)
        self.device_buffer = torch.zeros(shape, dtype=torch.int32, device=self.device_buffer.device)
        logger.info("TRR capture enabled: %s int32 IDs + FP32 weight bits", shape)

    def manager_init(self, vllm_config, kv_cache_config):
        check_config(vllm_config)
        original_manager_init(self, vllm_config, kv_cache_config)
        shape = (*self.routed_experts_by_slot.shape[:-1], self.routed_experts_by_slot.shape[-1] * 2)
        self.routed_experts_by_slot = np.zeros(shape, dtype=np.int32)
        logger.info("TRR scheduler storage: %.3f GB, shape %s", self.routed_experts_by_slot.nbytes / 1e9, shape)

    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        self._validate_eplb_state()
        weights, ids = self._compute_routing(hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)
        if self.capture_fn is not None:
            self.capture_fn(pack_routing(ids, weights))
        ids = self._apply_eplb_mapping(ids)
        return weights, self._convert_indices_dtype(ids, topk_indices_dtype)

    RoutedExpertsCapturer.__init__ = capture_init
    RoutedExpertsManager.__init__ = manager_init
    BaseRouter._select_experts = select_experts
    BaseRouter._prime_trr = True
