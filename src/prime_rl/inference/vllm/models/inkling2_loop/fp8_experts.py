"""FP8 routed experts of the looped Inkling 2 (Inkling 2 Mini) for RL rollouts: ``Inkling2FP8Experts``, which
``Inkling2LoopForRL`` (``model.py``) builds with ``inference.vllm.quantization = "fp8_per_block"``.

The module needs none of ``vllm.models.inkling2``, so it also imports on the vLLM release prime-rl pins, and
``model.py`` imports it only when FP8 experts are selected: a BF16 server does not depend on vLLM's FP8 kernels.

The experts follow vLLM (Apache-2.0, read at commit b426b46e6). ``Inkling2FP8Experts.load_bf16_weight`` rounds with
the call of ``Fp8PerBlockOnlineMoEMethod.process_weights_after_loading``
(``vllm/model_executor/layers/quantization/online/fp8.py``). ``Inkling2FP8Experts.forward`` runs the kernels of
``fused_experts_impl`` (``vllm/model_executor/layers/fused_moe/fused_moe.py``) in the order of steps of
``TritonExperts.apply`` for block-wise FP8 (``vllm/model_executor/layers/fused_moe/experts/triton_moe.py``), with
the routing weights applied in the final sum. Rounding each expert in place as it loads, at the cold start and on
every weight update, follows ``Inkling2MXFP8Experts`` of the Inkling 2 vLLM plugin developed for SkyRL (Apache-2.0):
https://github.com/erictang000/SkyRL/blob/faae9039fb0f4b6be27c02dc0045c1de63e3b888/skyrl/backends/skyrl_train/patches/vllm/inkling2/model.py
"""

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.fused_moe import dispatch_fused_moe_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.triton_utils import tl
from vllm.utils.deep_gemm import per_block_cast_to_fp8

# vLLM's `fp8_per_block`: one FP32 scale per 128 x 128 block of an expert matrix, per token per 128 channels of an
# activation.
FP8_BLOCK_SHAPE = [128, 128]


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
