import pickle
from typing import TYPE_CHECKING, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_state_dict(communicator: PyNcclCommunicator) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    communicator.broadcast(state_tensor, src=0)

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        communicator.broadcast(concatenated, src=0)

        # Split concatenated tensor back into individual tensors
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        del concatenated


class NCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
    ):
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(self.communicator):
                yield key, value


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(
        self,
        host: str,
        port: int,
        server_rank: int,
        num_inference_server: int,
        timeout: int,
        use_vllm_format_transfer: bool = False,
    ) -> None:
        """Initialize the NCCL broadcast receiver."""
        self.use_vllm_format_transfer = use_vllm_format_transfer
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        global_rank_inference = (server_rank * tp_size * dp_size) + (dp_rank * tp_size) + tp_rank
        global_inference_world_size = num_inference_server * tp_size * dp_size

        logger.info(
            f"Worker [tp={tp_rank} dp={dp_rank} server_rank={server_rank}] -> [global_rank={global_rank_inference} global_world_size={global_inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=global_inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
        )

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        if self.use_vllm_format_transfer:
            self._load_kernel_format(model, state_iter)
            self._update_mla_absorbed_weights(model)
            return

        model.load_weights(state_iter)  # type: ignore
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)

    @torch.no_grad()
    def _update_mla_absorbed_weights(self, model: Module) -> None:
        """Recompute MLA absorbed KV weights after in-place kv_b_proj updates."""
        from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights

        for name, module in model.named_modules():
            has_absorbed_weights = hasattr(module, "W_UV") or hasattr(module, "W_UK_T")
            if not has_absorbed_weights or not hasattr(module, "kv_b_proj"):
                continue

            if hasattr(module, "W_UV"):
                out_dtype = module.W_UV.dtype
            else:
                out_dtype = torch.bfloat16

            kv_b_proj_weight = get_and_maybe_dequant_weights(module.kv_b_proj, out_dtype=out_dtype).T
            kv_b_proj_weight = kv_b_proj_weight.view(
                module.kv_lora_rank,
                module.num_heads,
                module.qk_nope_head_dim + module.v_head_dim,
            )
            w_uk, w_uv = kv_b_proj_weight.split([module.qk_nope_head_dim, module.v_head_dim], dim=-1)

            if hasattr(module, "W_UV"):
                module.W_UV.copy_(w_uv.transpose(0, 1))
            if hasattr(module, "W_UK_T"):
                module.W_UK_T.copy_(w_uk.permute(1, 2, 0))

            logger.debug(f"Updated MLA absorbed weights for module {name}")

    def _build_expert_map(self, model: Module) -> dict[str, torch.Tensor]:
        """Map FusedMoE module names to global expert indices local to this worker."""
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        expert_slices: dict[str, torch.Tensor] = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, FusedMoE):
                continue
            if module._expert_map is None:
                continue

            global_indices = torch.where(module._expert_map >= 0)[0]
            local_indices = module._expert_map[global_indices]
            global_indices = global_indices[local_indices.argsort()]
            expert_slices[module_name] = global_indices
        return expert_slices

    @torch.no_grad()
    def _load_kernel_format(self, model: Module, state_iter: Generator[tuple[str, torch.Tensor], None, None]) -> None:
        """Load vLLM kernel-format tensors using in-place copy_ updates."""
        params = dict(model.named_parameters())
        expert_slices = self._build_expert_map(model)

        loaded = 0
        skipped: list[str] = []
        shape_mismatches: list[str] = []

        for name, tensor in state_iter:
            if name not in params:
                skipped.append(name)
                continue

            param = params[name]
            if param.shape != tensor.shape:
                sliced = False
                for module_name, global_indices in expert_slices.items():
                    if not name.startswith(f"{module_name}."):
                        continue
                    tensor = tensor[global_indices]
                    sliced = True
                    break

                if not sliced or param.shape != tensor.shape:
                    shape_mismatches.append(
                        f"{name}: param={list(param.shape)} != received={list(tensor.shape)}"
                    )
                    continue

            param.copy_(tensor)
            loaded += 1

        if shape_mismatches:
            logger.error(
                f"Kernel weight transfer had {len(shape_mismatches)} shape mismatches: {shape_mismatches}"
            )
        if skipped:
            logger.warning(f"Kernel weight transfer skipped {len(skipped)} weights not found in model: {skipped}")
        logger.info(f"Kernel weight transfer copied {loaded} weights in-place")
