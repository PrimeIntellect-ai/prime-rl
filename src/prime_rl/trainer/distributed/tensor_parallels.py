import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from prime_rl.utils.logger import get_logger


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = False,
    enable_float8_tensorwise_tp: bool = False,
    enable_async_tp: bool = False,
):
    parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),  # TODO: This would shard on seq dim?
            ),
            "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel as rowwise_parallel,
        )
        from torchao.float8.float8_tensor_parallel import (
            Float8RowwiseParallel as colwise_parallel,
        )
        from torchao.float8.float8_tensor_parallel import (
            PrepareFloat8ModuleInput as prepare_module_input,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.model.layers:
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            # TODO: can we do kwargs? otherwise, we HAVE to rewrite the model impls :(
            "self_attn": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "self_attn.q_norm": SequenceParallel(),
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_norm": SequenceParallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            "self_attn.o_proj": rowwise_parallel(output_layouts=Shard(1)),
            "post_attention_layernorm": SequenceParallel(),
        }
        if not hasattr(transformer_block.mlp, "experts"):
            layer_plan.update(
                {
                    "mlp": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.up_proj": colwise_parallel(),
                    "mlp.down_proj": rowwise_parallel(output_layouts=Shard(1)),
                    "mlp.gate_proj": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    get_logger().info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )
