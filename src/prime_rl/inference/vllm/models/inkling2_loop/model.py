"""Looped Inkling 2 (Inkling 2 Mini) for RL rollouts: BF16 routed experts and weight updates loaded in place.

vLLM serves the release's NVFP4 / MXFP8 experts, takes a checkpoint with exactly the release's dtypes, and after
loading derives further tensors from the weights. The trainer holds the experts in BF16 and pushes BF16 weights
while the server runs. ``Inkling2LoopForRL`` is vLLM's model with what that takes:

- BF16 routed experts (``Inkling2BF16Experts``), built from vLLM's own kernels: its Hopper router, activation and
  top-k combine, and the grouped GEMM of its expert LoRA path.
- A ``load_weights`` that serves both the cold start and every weight update. It writes into the tensors the
  model already serves and refreshes the derived ones in place, since captured CUDA graphs hold their addresses.
- The trainer's values for the depth-mix and loop-message queries. The checkpoint stores them in FP32; the
  trainer computes with their BF16 rounding (FSDP casts every parameter to BF16), so the server rounds them too.

The BF16 experts, the in-place refresh of the derived tensors and the class names are adapted from the Inkling 2
vLLM plugin developed for SkyRL (Apache-2.0):
https://github.com/erictang000/SkyRL/blob/faae9039fb0f4b6be27c02dc0045c1de63e3b888/skyrl/backends/skyrl_train/patches/vllm/inkling2/model.py
``load_weights`` is adapted from ``Inkling2ForConditionalGeneration.load_weights`` of vLLM (Apache-2.0; added by
vllm-project/vllm#67325, read at commit b426b46e6).
"""

import re
from collections.abc import Callable, Iterable, Iterator, Sequence

import torch
import vllm.envs as envs
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.inkling2.nvidia.attention import Inkling2Attention
from vllm.models.inkling2.nvidia.loop import Inkling2LoopForConditionalGeneration
from vllm.models.inkling2.nvidia.moe import Inkling2Gate, Inkling2MoE, Inkling2RoutingCapture
from vllm.models.inkling2.nvidia.ops.grouped_gemm import grouped_gemm
from vllm.models.inkling2.nvidia.ops.moe import combine
from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts, swiglu
from vllm.models.inkling2.nvidia.ops.sconv import shift_rel_proj

# Depth-mix and loop-message queries: FP32 in the checkpoint, parameters of the trainer.
TRAINED_FP32_TENSOR_RE = re.compile(r"\.attn_res\.|\.middle_cycle_cross_pass_(kv_)?queries$")
PACKED_EXPERT_DTYPES = (torch.uint8, torch.float8_e4m3fn)
_LAYER_INDEX_RE = re.compile(r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)")


class Inkling2BF16Experts(nn.Module):
    """One layer's routed experts with BF16 weights in the checkpoint's layout: ``w13_weight [E, 2I, H]`` holds the
    gate projection in its even rows and the up projection in its odd rows, ``w2_weight [E, H, I]`` projects down."""

    def __init__(self, config) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        num_experts, hidden, intermediate = config.n_routed_experts, config.hidden_size, config.intermediate_size
        self.w13_weight = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.empty(num_experts, hidden, intermediate, dtype=torch.bfloat16), requires_grad=False
        )
        self.register_buffer("expert_bounds", torch.arange(num_experts + 1, dtype=torch.int32), persistent=False)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[T, H]`` BF16.
            ids: ``[T, topk]`` int32 expert of every routing slot, -1 for the slots of a padding token.
            weights: ``[T * topk]`` FP32 routing weights.

        Returns:
            ``[T, H]`` BF16. The routing weight scales the activation, before the down projection, and a token's
            slots are summed in FP32 and rounded once, as in vLLM's own experts.
        """
        num_rows = ids.numel()
        flat_ids = ids.flatten()
        # The grouped GEMM takes rows grouped by expert: row `r` is slot `order[r]`, of token `order[r] // topk`.
        sorted_ids, order = torch.sort(flat_ids, stable=True)
        offsets = torch.searchsorted(sorted_ids, self.expert_bounds, out_int32=True)
        gate_up = grouped_gemm(x, self.w13_weight, order, offsets, num_rows, self.topk, gather=True)
        activated = swiglu(gate_up, weights[order], sorted_ids, self.gate_cap, self.up_cap)
        down = grouped_gemm(activated, self.w2_weight, order, offsets, num_rows, self.topk, gather=False)
        row_of_slot = torch.empty(num_rows, dtype=torch.int32, device=x.device)
        row_of_slot[order] = torch.arange(num_rows, dtype=torch.int32, device=x.device)
        return combine(down, torch.where(flat_ids >= 0, row_of_slot, -1), self.topk, x.dtype)


class Inkling2BF16MoE(Inkling2MoE):
    """vLLM's Inkling 2 MoE (its gate and its routed-experts capture rows) on ``Inkling2BF16Experts``."""

    def __init__(self, config, visit_ids: Sequence[int]) -> None:
        # Not Inkling2MoE.__init__: it builds the packed experts.
        nn.Module.__init__(self)
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2BF16Experts(config)
        self.routing_captures = nn.ModuleList(Inkling2RoutingCapture(visit) for visit in visit_ids)
        self.register_buffer(
            "all_experts", torch.tensor([0, config.n_routed_experts], dtype=torch.int32), persistent=False
        )

    def forward(self, x: torch.Tensor, reduce: bool = True, push=None, lora=None, pass_idx: int = 0) -> torch.Tensor:
        if not reduce or push is not None or lora is not None:
            raise NotImplementedError("The BF16 Inkling 2 experts run on one GPU, without LoRA")
        x = x.contiguous()
        gate = self.gate
        ids, weights, _ = select_experts(x, gate.weight, gate.bias, gate.topk, gate.route_scale, self.all_experts)
        capture_fn = self.routing_captures[pass_idx].capture_fn
        if capture_fn is not None:
            capture_fn(ids)
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                ids = torch.where(is_padding.unsqueeze(1), -1, ids)
        return self.experts(x, ids, weights)


class Inkling2LoopForRL(Inkling2LoopForConditionalGeneration):
    # Weight updates go straight through `load_weights` (see `worker/weight_transfer.py`).
    loads_weights_in_place = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                "Inkling 2 with BF16 experts runs one replica per GPU: use data_parallel_size instead of "
                "tensor_parallel_size"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        text_config = self.text_config
        for layer_id, layer in enumerate(self.model.layers):
            layer.mlp = Inkling2BF16MoE(text_config, text_config.layer_visit_ids(layer_id))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a checkpoint, or a weight update, in checkpoint names. Every tensor of the language model must be
        there, with the checkpoint's dtypes and BF16 experts. The vision tower may be left out: the trainer does
        not hold it."""
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        layer_limits = {"layers": num_layers, "attn_queries": num_layers - 1, "mlp_queries": num_layers}
        loaded: set[str] = set()
        shards: dict[str, set[int]] = {}
        visual_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in self._trainer_values(weights):
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name, shard_id = name.replace(old, new), shard
                    break
            if name not in params:
                index = _LAYER_INDEX_RE.search(name)
                if index is not None and int(index.group(2)) >= layer_limits[index.group(1)]:
                    continue  # a model truncated to fewer layers
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}")
            loader: Callable[..., None] = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                shards.setdefault(name, set()).add(shard_id)
        for name, shard_ids in shards.items():
            fused = next(new for new, _ in self._shards.values() if new in name)
            if shard_ids == {shard for new, shard in self._shards.values() if new == fused}:
                loaded.add(name)
        if visual_weights:
            loaded |= {f"visual.{name}" for name in self.visual.load_weights(visual_weights)}
        if missing := {name for name in params if not name.startswith("visual.")} - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        self._refresh_derived()
        return loaded

    def _trainer_values(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            if ".mlp.experts." in name and weight.dtype in PACKED_EXPERT_DTYPES:
                raise ValueError(
                    f"{name} holds packed {weight.dtype} experts. RL serving needs them in BF16: export the "
                    "checkpoint with tools/convert_inkling2_loop_to_bf16.py"
                )
            if TRAINED_FP32_TENSOR_RE.search(name) and weight.dtype == torch.float32:
                weight = weight.to(torch.bfloat16).to(torch.float32)
            yield name, weight

    def _refresh_derived(self) -> None:
        """Recompute, in place, the tensors vLLM derives from the weights after the cold start."""
        attentions = [module for module in self.modules() if isinstance(module, Inkling2Attention)]
        if not hasattr(attentions[0], "rel_proj_shifted"):
            return  # The cold start: vLLM derives them once this load returns.
        for attention in attentions:
            attention.rel_proj_shifted.copy_(shift_rel_proj(attention.rel_logits_proj, attention.is_local))
        # Copies the scaled, stacked loop-message queries into their buffer.
        self.model.process_weights_after_loading()
