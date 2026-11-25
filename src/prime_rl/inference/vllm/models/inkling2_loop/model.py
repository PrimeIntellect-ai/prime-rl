"""Looped Inkling 2 (Inkling 2 Mini) for RL rollouts: routed experts in BF16 or FP8, weight updates loaded in place.

vLLM serves the release's NVFP4 / MXFP8 experts, takes a checkpoint with exactly the release's dtypes, and after
loading derives further tensors from the weights. The trainer holds the experts in BF16 and pushes BF16 weights
while the server runs. ``Inkling2LoopForRL`` is vLLM's model with what that takes:

- BF16 routed experts (``Inkling2BF16Experts``), built from vLLM's own kernels: its Hopper router, activation and
  top-k combine, and the grouped GEMM of its expert LoRA path.
- FP8 routed experts (``Inkling2FP8Experts``), selected with ``inference.vllm.quantization = "fp8_per_block"``:
  the format and the kernels of vLLM's online ``fp8_per_block`` experts, with the routing weights applied after
  the second product and summed in FP32, as the trainer's ``model.moe.compute.type = "deepgemm_fp8"`` computes
  them. They take BF16 weights and round them as they load.
- A ``load_weights`` that serves both the cold start and every weight update. It writes into the tensors the
  model already serves and refreshes the derived ones in place, since captured CUDA graphs hold their addresses.
- The trainer's values for the depth-mix and loop-message queries. The checkpoint stores them in FP32; the
  trainer computes with their BF16 rounding (FSDP casts every parameter to BF16), so the server rounds them too.

The BF16 experts, the in-place refresh of the derived tensors and the class names are adapted from the Inkling 2
vLLM plugin developed for SkyRL (Apache-2.0):
https://github.com/erictang000/SkyRL/blob/faae9039fb0f4b6be27c02dc0045c1de63e3b888/skyrl/backends/skyrl_train/patches/vllm/inkling2/model.py
``load_weights`` is adapted from ``Inkling2ForConditionalGeneration.load_weights`` of vLLM (Apache-2.0; added by
vllm-project/vllm#67325, read at commit b426b46e6).
The FP8 experts follow vLLM as well (Apache-2.0, read at the same commit).
``Inkling2FP8Experts.load_bf16_weight`` rounds with the call of
``Fp8PerBlockOnlineMoEMethod.process_weights_after_loading``
(``vllm/model_executor/layers/quantization/online/fp8.py``). ``Inkling2FP8Experts.forward`` runs the kernels of
``fused_experts_impl`` (``vllm/model_executor/layers/fused_moe/fused_moe.py``) in the order of steps of
``TritonExperts.apply`` for block-wise FP8 (``vllm/model_executor/layers/fused_moe/experts/triton_moe.py``), with
the routing weights applied in the final sum. Rounding each expert in place as it loads, at the cold start and on
every weight update, follows ``Inkling2MXFP8Experts`` of the same SkyRL plugin.
"""

import re
from collections.abc import Callable, Iterable, Iterator, Sequence

import torch
import vllm.envs as envs
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.fused_moe import dispatch_fused_moe_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.inkling2.nvidia.attention import Inkling2Attention
from vllm.models.inkling2.nvidia.loop import Inkling2LoopForConditionalGeneration
from vllm.models.inkling2.nvidia.moe import Inkling2Gate, Inkling2MoE, Inkling2RoutingCapture
from vllm.models.inkling2.nvidia.ops.grouped_gemm import grouped_gemm
from vllm.models.inkling2.nvidia.ops.moe import combine
from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts, swiglu
from vllm.models.inkling2.nvidia.ops.sconv import shift_rel_proj
from vllm.triton_utils import tl
from vllm.utils.deep_gemm import per_block_cast_to_fp8

from prime_rl.inference.patches import INKLING2_LOOP_EXPERT_FORMAT_ATTRIBUTE

# Depth-mix and loop-message queries: FP32 in the checkpoint, parameters of the trainer.
TRAINED_FP32_TENSOR_RE = re.compile(r"\.attn_res\.|\.middle_cycle_cross_pass_(kv_)?queries$")
PACKED_EXPERT_DTYPES = (torch.uint8, torch.float8_e4m3fn)
EXPERT_FORMATS = ("bf16", "fp8_per_block")
# vLLM's `fp8_per_block`: one FP32 scale per 128 x 128 block of an expert matrix, per token per 128 channels of an
# activation.
FP8_BLOCK_SHAPE = [128, 128]
_LAYER_INDEX_RE = re.compile(r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)")


class Inkling2BF16Experts(nn.Module):
    """One layer's routed experts with BF16 weights in the checkpoint's layout: ``w13_weight [E, 2I, H]`` holds the
    gate projection in its even rows and the up projection in its odd rows, ``w2_weight [E, H, I]`` projects down."""

    # vLLM puts LoRA adapters on an MoE's experts unless they say otherwise. These take none.
    supports_lora = False

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


class Inkling2RLMoE(Inkling2MoE):
    """vLLM's Inkling 2 MoE (its gate and its routed-experts capture rows) on experts that take BF16 weight
    updates: ``Inkling2BF16Experts`` or ``Inkling2FP8Experts``."""

    def __init__(self, config, visit_ids: Sequence[int], experts: nn.Module) -> None:
        # Not Inkling2MoE.__init__: it builds the packed experts.
        nn.Module.__init__(self)
        self.gate = Inkling2Gate(config)
        self.experts = experts
        self.routing_captures = nn.ModuleList(Inkling2RoutingCapture(visit) for visit in visit_ids)
        self.register_buffer(
            "all_experts", torch.tensor([0, config.n_routed_experts], dtype=torch.int32), persistent=False
        )

    def forward(self, x: torch.Tensor, reduce: bool = True, push=None, lora=None, pass_idx: int = 0) -> torch.Tensor:
        if not reduce or push is not None or lora is not None:
            raise NotImplementedError("The RL Inkling 2 experts run on one GPU, without LoRA")
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


class Inkling2FP8Experts(nn.Module):
    """One layer's routed experts in vLLM's online ``fp8_per_block`` format: E4M3 weights with one FP32 scale per
    128 x 128 block, and activations rounded to E4M3 with one FP32 scale per token per 128 channels on their way
    into both projections.

    ``w13_weight [E, 2I, H]`` holds the gate projection in its first ``I`` rows and the up projection in the rest,
    not interleaved as in the checkpoint: a block of 128 rows then holds what the trainer rounds as one block of
    ``gate_proj`` or of ``up_proj``. ``w2_weight [E, H, I]`` projects down.
    """

    # vLLM puts LoRA adapters on an MoE's experts unless they say otherwise. These take none.
    supports_lora = False

    def __init__(self, config) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        num_experts, hidden, intermediate = config.n_routed_experts, config.hidden_size, config.intermediate_size
        block_rows, block_columns = FP8_BLOCK_SHAPE
        if hidden % block_rows or hidden % block_columns or intermediate % block_rows or intermediate % block_columns:
            raise ValueError(
                f"FP8 experts need sizes that are multiples of {FP8_BLOCK_SHAPE}: {hidden}, {intermediate}"
            )
        self.num_experts, self.intermediate_size = num_experts, intermediate
        for name, rows, columns in (("w13_weight", 2 * intermediate, hidden), ("w2_weight", hidden, intermediate)):
            weight = torch.empty(num_experts, rows, columns, dtype=torch.float8_e4m3fn)
            scale = torch.empty(num_experts, rows // block_rows, columns // block_columns, dtype=torch.float32)
            self.register_parameter(name, nn.Parameter(weight, requires_grad=False))
            self.register_parameter(f"{name}_scale", nn.Parameter(scale, requires_grad=False))

    @torch.no_grad()
    def load_bf16_weight(self, name: str, weight: torch.Tensor) -> None:
        """Round a BF16 expert weight in the checkpoint's layout into the served ``name`` and its scales, in place.

        ``per_block_cast_to_fp8`` is the function vLLM's online ``fp8_per_block`` rounds expert weights with."""
        codes, scales = getattr(self, name), getattr(self, f"{name}_scale")
        if weight.dtype != torch.bfloat16 or weight.shape != codes.shape:
            raise ValueError(f"{name}: expected BF16 {tuple(codes.shape)}, got {weight.dtype} {tuple(weight.shape)}")
        for expert in range(self.num_experts):
            # Rounded on the GPU that serves it, also at the cold start, when the checkpoint arrives on the CPU.
            matrix = weight[expert].to(codes.device)
            if name == "w13_weight":
                matrix = torch.cat([matrix[0::2], matrix[1::2]])
            expert_codes, expert_scales = per_block_cast_to_fp8(matrix, block_size=FP8_BLOCK_SHAPE, use_ue8m0=False)
            codes[expert].copy_(expert_codes)
            scales[expert].copy_(expert_scales)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[T, H]`` BF16.
            ids: ``[T, topk]`` int32 expert of every routing slot, -1 for the slots of a padding token.
            weights: ``[T * topk]`` FP32 routing weights.

        Returns:
            ``[T, H]`` BF16. The steps are those of vLLM's FP8 experts, in the trainer's order: both projections
            take FP8 activations and return BF16, the activation runs on the BF16 gate/up product, and a token's
            slots are scaled by their routing weights and summed in FP32, then rounded once.
        """
        tokens, topk = ids.shape
        intermediate = self.intermediate_size
        config = try_get_optimal_moe_config(
            self.w13_weight.shape, self.w2_weight.shape, topk, "fp8_w8a8", tokens, block_shape=FP8_BLOCK_SHAPE
        )
        # Slots grouped by expert in blocks of BLOCK_SIZE_M rows; the slots of a padding token are left out.
        sorted_slots, block_experts, num_slots_padded = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], self.num_experts, ignore_invalid_experts=True
        )[:3]
        fp8_kernel = dict(
            sorted_token_ids=sorted_slots,
            expert_ids=block_experts,
            num_tokens_post_padded=num_slots_padded,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=FP8_BLOCK_SHAPE,
        )
        # FP32 scales that are not rounded to powers of two, as the trainer computes them on SM90.
        x_fp8, x_scale = per_token_group_quant_fp8(x, FP8_BLOCK_SHAPE[1], use_ue8m0=False)
        gate_up = torch.empty((tokens, topk, 2 * intermediate), dtype=x.dtype, device=x.device)
        dispatch_fused_moe_kernel(
            x_fp8, self.w13_weight, gate_up, x_scale, self.w13_weight_scale, None, None,
            mul_routed_weight=False, top_k=topk, **fp8_kernel,
        )  # fmt: skip
        # Softcapped SwiGLU of [gate | up] in FP32, rounded once. The rows of padding slots hold nothing that is read.
        activated = torch.empty((tokens * topk, intermediate), dtype=x.dtype, device=x.device)
        torch.ops._C.situ_and_mul(activated, gate_up.view(tokens * topk, 2 * intermediate), self.gate_cap, self.up_cap)
        activated_fp8, activated_scale = per_token_group_quant_fp8(activated, FP8_BLOCK_SHAPE[1], use_ue8m0=False)
        down = torch.empty((tokens, topk, x.shape[1]), dtype=x.dtype, device=x.device)
        dispatch_fused_moe_kernel(
            activated_fp8, self.w2_weight, down, activated_scale, self.w2_weight_scale, None, None,
            mul_routed_weight=False, top_k=1, num_input_tokens=tokens, **fp8_kernel,
        )  # fmt: skip
        # Padding tokens keep the zeros: the kernel leaves a token without a slot untouched.
        out = x.new_zeros((tokens, x.shape[1]))
        return moe_fused_mul_sum(down, weights.view(tokens, topk), out, ids)


class Inkling2LoopForRL(Inkling2LoopForConditionalGeneration):
    # Weight updates go straight through `load_weights` (see `worker/weight_transfer.py`).
    loads_weights_in_place = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                "Inkling 2 for RL runs one replica per GPU: use data_parallel_size instead of tensor_parallel_size"
            )
        expert_format = getattr(vllm_config.model_config.hf_config, INKLING2_LOOP_EXPERT_FORMAT_ATTRIBUTE, "bf16")
        if expert_format not in EXPERT_FORMATS:
            raise ValueError(f"Inkling 2 for RL serves its experts as one of {EXPERT_FORMATS}, not {expert_format!r}")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        text_config = self.text_config
        experts_cls = Inkling2FP8Experts if expert_format == "fp8_per_block" else Inkling2BF16Experts
        for layer_id, layer in enumerate(self.model.layers):
            layer.mlp = Inkling2RLMoE(text_config, text_config.layer_visit_ids(layer_id), experts_cls(text_config))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a checkpoint, or a weight update, in checkpoint names. Every tensor of the language model must be
        there, with the checkpoint's dtypes and BF16 experts. The vision tower may be left out: the trainer does
        not hold it. FP8 experts round their BF16 weights as they load."""
        params = dict(self.named_parameters())
        fp8_weights = {
            f"{prefix}.{attribute}": (module, attribute)
            for prefix, module in self.named_modules()
            if isinstance(module, Inkling2FP8Experts)
            for attribute in ("w13_weight", "w2_weight")
        }
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
            if name in fp8_weights:
                experts, attribute = fp8_weights[name]
                experts.load_bf16_weight(attribute, weight)
                loaded.update((name, f"{name}_scale"))
                continue
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
