"""Dispatch per-token NVFP4 activations through FlashInfer's one-sided all2all.

vLLM's online ``nvfp4_per_token`` experts quantize their input after the all2all, so under DP+EP with
``all2all_backend = "flashinfer_nvlink_one_sided"`` every token crosses NVLink as bf16 (2 B per element)
and each receiving rank quantizes it again. Here the sending rank quantizes once and dispatches the FP4
values and the E4M3 block scales with the FP32 per-token scale appended to each token's scale row: 0.28x the
bf16 bytes and 0.55x those of MXFP8 dispatch. Each token's quantization depends only
on that token, so the experts receive exactly the inputs they would have computed after receipt.
"""

import torch

PER_TOKEN_SCALE_BYTES = torch.float32.itemsize


def patch_nvfp4_per_token_dispatch():
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_ep_group
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.experts import trtllm_nvfp4_moe
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsModular
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as nvfp4_oracle
    from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided import (
        FlashInferNVLinkOneSidedPrepareAndFinalize,
        get_local_sizes,
    )
    from vllm.model_executor.layers.quantization.online import nvfp4 as online_nvfp4
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import quantize_nvfp4_per_token_input

    logger = init_logger(__name__)

    class Nvfp4PerTokenDispatchPrepareAndFinalize(FlashInferNVLinkOneSidedPrepareAndFinalize):
        def prepare(
            self,
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=False,
        ):
            assert not defer_input_quant
            if apply_router_weight_on_input:
                assert topk_ids.size(1) == 1, "apply_router_weight_on_input is only implemented for topk=1"
                a1.mul_(topk_weights.to(a1.dtype))

            global_num_tokens_cpu = get_local_sizes()
            self.runtime_max_tokens_per_rank = (
                max(global_num_tokens_cpu) if global_num_tokens_cpu is not None else a1.shape[0]
            )
            x_q, block_scale, per_token_scale = quantize_nvfp4_per_token_input(a1)
            # The wrapper dispatches one scale tensor per token: carry the FP32 per-token scale as 4 more bytes
            # of each token's E4M3 block-scale row and split them again on receipt.
            block_scale = block_scale.view(torch.uint8).reshape(x_q.shape[0], -1)
            scale_width = block_scale.shape[1]
            token_scale = per_token_scale.reshape(-1, 1).view(torch.uint8)
            received = self.all2all_manager.get_communication().dispatch(
                x_q.view(torch.uint8),
                topk_ids,
                topk_weights,
                hidden_states_scale=torch.cat([block_scale, token_scale], dim=1),
                runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            )
            recv_scales = received.hidden_states_scale
            self.received_per_token_scale = recv_scales[:, scale_width:].contiguous().view(torch.float32).squeeze(1)
            scales = recv_scales[:, :scale_width].contiguous()
            return received.hidden_states.view(x_q.dtype), scales, None, received.topk_ids, received.topk_weights

    class TrtLlmNvFp4PrequantizedExperts(TrtLlmNvFp4ExpertsModular):
        """Per-token NVFP4 experts whose input arrives quantized; the dispatcher holds the per-token scales."""

        dispatcher: Nvfp4PerTokenDispatchPrepareAndFinalize

        @property
        def expects_unquantized_inputs(self) -> bool:
            return False

        def _invoke_kernel(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            a1q_scale,
        ):
            # Mirrors TrtLlmNvFp4ExpertsModular._invoke_kernel; only the input scales differ.
            import flashinfer

            per_token_scale = self.dispatcher.received_per_token_scale
            # The received batch always fits one launch (chunking starts at 300K tokens), so no slicing is needed.
            assert per_token_scale.shape[0] == hidden_states.shape[0], (per_token_scale.shape, hidden_states.shape)
            block_scale = a1q_scale.view(torch.float8_e4m3fn)
            defer = output is not None and self._uses_post_gemm2_hook(topk_ids.shape[0])
            flashinfer_output = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
                topk_ids=(topk_ids, topk_weights),
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=block_scale.reshape(*hidden_states.shape[:-1], -1),
                gemm1_weights=w1,
                gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=self.gemm1_alpha,
                gemm1_beta=self.gemm1_beta,
                gemm1_clamp_limit=self.gemm1_clamp_limit,
                gemm2_weights=w2,
                gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=self.g1_scale_c,
                output1_scale_gate_scalar=self.quant_config.g1_alphas,
                output2_scale_scalar=self.quant_config.g2_alphas,
                num_experts=global_num_experts,
                top_k=self.topk,
                n_group=0,
                topk_group=0,
                intermediate_size=self.intermediate_size_per_partition,
                local_expert_offset=self.ep_rank * self.local_num_experts,
                local_num_experts=self.local_num_experts,
                routed_scaling_factor=None,
                routing_method_type=1,  # not used
                do_finalize=output is not None and not defer,
                activation_type=trtllm_nvfp4_moe.activation_to_flashinfer_int(activation),
                per_token_scale=per_token_scale,
                output=None if defer else output,
                tune_max_num_tokens=min(
                    trtllm_nvfp4_moe.fi_moe_largest_bucket(self.moe_config), self._get_chunk_size()
                ),
            )
            if defer:
                self._run_post_gemm2_hook(
                    trtllm_nvfp4_moe.convert_flashinfer_moe_output(
                        flashinfer_output, do_finalize=False, num_tokens=topk_ids.shape[0], top_k=self.topk
                    ),
                    output,
                )
                return None
            if output is not None:
                return None
            return trtllm_nvfp4_moe.convert_flashinfer_moe_output(
                flashinfer_output, do_finalize=False, num_tokens=hidden_states.shape[0], top_k=self.topk
            )

    original_make_kernel = nvfp4_oracle.make_nvfp4_moe_kernel

    def make_nvfp4_moe_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        backend,
        routing_tables=None,
        per_token_activation=False,
    ):
        if not (
            per_token_activation
            and backend == nvfp4_oracle.NvFp4MoeBackend.FLASHINFER_TRTLLM
            and moe_config.use_fi_nvl_one_sided_kernels
            and issubclass(experts_cls, TrtLlmNvFp4ExpertsModular)
        ):
            return original_make_kernel(
                moe_quant_config, moe_config, experts_cls, backend, routing_tables, per_token_activation
            )
        from flashinfer.fused_moe import QuantFormat

        prepare_finalize = Nvfp4PerTokenDispatchPrepareAndFinalize(
            max_num_tokens=get_current_vllm_config().scheduler_config.max_num_batched_tokens,
            top_k=moe_config.experts_per_token,
            num_experts=moe_config.num_experts,
            hidden_size=moe_config.hidden_dim,
            dispatch_format=QuantFormat.NVFP4,
            extra_payload_bytes_per_token=PER_TOKEN_SCALE_BYTES,
            num_dispatchers=get_ep_group().device_communicator.all2all_manager.world_size,
        )
        logger.info_once("Dispatching per-token NVFP4 MoE activations quantized (FP4 + E4M3 + FP32 scale per token)")
        experts = TrtLlmNvFp4PrequantizedExperts(
            moe_config=moe_config, quant_config=moe_quant_config, per_token_activation=True
        )
        experts.dispatcher = prepare_finalize
        return mk.FusedMoEKernel(prepare_finalize, experts)

    nvfp4_oracle.make_nvfp4_moe_kernel = make_nvfp4_moe_kernel
    online_nvfp4.make_nvfp4_moe_kernel = make_nvfp4_moe_kernel
