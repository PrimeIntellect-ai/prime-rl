from collections.abc import Iterable

import torch


def patch_gpt_oss_weight_loading() -> None:
    """Route GPT-OSS BF16 expert and sink weights through vLLM loaders.

    Remove this patch when the pinned vLLM version uses weight loaders for these
    prepared tensors in ``GptOssModel._load_weights_other``.
    """
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
        maybe_remap_moe_expert_param_name,
    )
    from vllm.model_executor.models import gpt_oss

    original_load_weights = gpt_oss.GptOssModel._load_weights_other
    if getattr(original_load_weights, "_prime_rl_uses_weight_loaders", False):
        return

    original_expert_weight_loader = RoutedExperts.weight_loader

    def expert_weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if expert_id is None:
            if shard_id is not None:
                raise ValueError("A combined expert tensor must already be sharded")
            if param.shape != loaded_weight.shape:
                raise ValueError(
                    f"Combined expert tensor shape {tuple(loaded_weight.shape)} does not match "
                    f"parameter shape {tuple(param.shape)}"
                )
            param.copy_(loaded_weight)
            return True if return_success else None
        return original_expert_weight_loader(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success,
        )

    RoutedExperts.weight_loader = expert_weight_loader

    def load_weights(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded_params: set[str] = set()
        use_ep = self.parallel_config.enable_expert_parallel
        tp_size, tp_rank = gpt_oss.FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=gpt_oss.get_tensor_model_parallel_world_size(),
            dp_size=gpt_oss.get_dp_group().world_size,
            dp_rank=gpt_oss.get_dp_group().rank_in_group,
            pcp_size=gpt_oss.get_pcp_group().world_size,
            pcp_rank=gpt_oss.get_pcp_group().rank_in_group,
        )
        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = gpt_oss.cdiv(intermediate_size, tp_size)
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        def remaining_weights():
            for name, weight in weights:
                if not (
                    name.endswith((".w13_weight", ".w2_weight", ".w13_bias", ".w2_bias")) or name.endswith(".sinks")
                ):
                    yield name, weight
                    continue

                name = maybe_remap_moe_expert_param_name(name, params)
                if gpt_oss.is_pp_missing_parameter(name, self):
                    continue

                param = params[name]
                if name.endswith(".w13_weight"):
                    if use_ep:
                        weight = weight[ep_rank_start:ep_rank_end]
                    else:
                        weight = weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]
                    weight = weight.permute(0, 2, 1).contiguous()
                elif name.endswith(".w2_weight"):
                    if use_ep:
                        weight = weight[ep_rank_start:ep_rank_end]
                    else:
                        weight = weight[:, tp_rank_start:tp_rank_end]
                    weight = weight.permute(0, 2, 1).contiguous()
                elif name.endswith(".w13_bias"):
                    if use_ep:
                        weight = weight[ep_rank_start:ep_rank_end]
                    else:
                        weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
                elif name.endswith(".w2_bias"):
                    if use_ep:
                        weight = weight[ep_rank_start:ep_rank_end]
                    elif tp_rank != 0:
                        weight = weight.zero_()
                else:
                    weight = weight.narrow(0, head_start, heads_per_rank)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, weight)
                    loaded_params.add(name)
                    continue

                param.weight_loader(
                    param,
                    weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)

        loaded_params.update(
            original_load_weights(
                self,
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                remaining_weights(),
                stacked_params_mapping,
            )
        )
        return loaded_params

    load_weights._prime_rl_uses_weight_loaders = True
    gpt_oss.GptOssModel._load_weights_other = load_weights
