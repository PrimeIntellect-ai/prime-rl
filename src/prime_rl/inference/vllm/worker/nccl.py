import pickle
import threading
from typing import TYPE_CHECKING, Any, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint_layerwise,
    load_weights_kernel,
    update_mla_absorbed_weights,
)
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")

# NemotronH mixer.D is dropped by vLLM 0.22's layerwise online-reload path (left uninitialized).
_RELOAD_CORRUPTED_SUFFIXES = (".mixer.D",)


def _restore_reload_corrupted_params(model: Module, received: dict[str, torch.Tensor]) -> None:
    """Work around a vLLM 0.22 layerwise-reload bug for NemotronH.

    The online reload drops the weight load for every Mamba layer's ``mixer.D`` (the SSD skip
    connection), leaving it as uninitialized ``empty_strided`` memory -- it reads back as garbage
    (NaN/inf) and the logits go NaN after a weight update. The received broadcast value is correct,
    so restore D from it via the param's own ``weight_loader``. Remove once the upstream bug is fixed.
    """

    def _layer_key(name: str) -> str:
        index = name.find("layers.")
        return name[index:] if index >= 0 else name

    received_by_key = {_layer_key(name): tensor for name, tensor in received.items()}
    for name, param in model.named_parameters():
        if not name.endswith(_RELOAD_CORRUPTED_SUFFIXES):
            continue
        tensor = received_by_key.get(_layer_key(name))
        if tensor is None:
            continue
        tensor = tensor.to(device=param.device)
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None:
            weight_loader(param, tensor)
        elif tensor.shape == param.shape:
            param.data.copy_(tensor.to(param.dtype))


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_object(communicator: PyNcclCommunicator) -> object:
    """Receive a small pickled Python payload from the trainer master rank."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    communicator.broadcast(state_tensor, src=0)
    return pickle.loads(bytes(state_tensor.cpu().numpy()))


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
        disable_nccl_p2p_if_unavailable()

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

    def arm_lora_receive(self, step: int, header_expectation: dict | None = None) -> None:
        """Arm a background LoRA receive and return after the receiver thread is live.

        This method is intentionally fast: the API server awaits this worker RPC
        before returning 202 to the orchestrator. The blocking NCCL receive runs
        in the background and is joined by ``wait_lora_receive``.
        """
        thread = getattr(self, "_lora_receive_thread", None)
        if thread is not None and thread.is_alive():
            state = getattr(self, "_lora_receive_state", {})
            raise RuntimeError(f"LoRA receive for step {state.get('step')} is still in flight")

        self._lora_receive_state = {
            "step": step,
            "status": "armed",
            "error": None,
        }
        self._lora_receive_error = None
        thread = threading.Thread(
            target=self._run_lora_receive_thread,
            args=(step, header_expectation),
            name=f"lora-receive-step-{step}",
            daemon=True,
        )
        self._lora_receive_thread = thread
        thread.start()

    def _run_lora_receive_thread(self, step: int, header_expectation: dict | None) -> None:
        self._lora_receive_state["status"] = "receiving"
        try:
            self.receive_lora_update(step, header_expectation)
        except BaseException as exc:
            self._lora_receive_error = exc
            self._lora_receive_state["status"] = "error"
            self._lora_receive_state["error"] = repr(exc)
            return
        self._lora_receive_state["status"] = "ok"

    def receive_lora_update(self, step: int, header_expectation: dict | None) -> None:
        """Receive and apply an NCCL LoRA update."""
        receiver = getattr(self, "nccl_broadcast_receiver", None)
        if receiver is None:
            raise RuntimeError("NCCL broadcast receiver is not initialized")

        header = receive_object(receiver.communicator)
        adapter = self._validate_lora_header(step, header, header_expectation)
        tensors = dict(receive_state_dict(receiver.communicator))
        self._commit_lora_adapter(adapter, tensors)

    def _validate_lora_header(self, step: int, header: object, header_expectation: dict | None) -> dict:
        if not isinstance(header, dict):
            raise RuntimeError(f"LoRA update header must be a dict, got {type(header).__name__}")
        if header.get("step") != step:
            raise RuntimeError(f"LoRA update header step {header.get('step')} did not match armed step {step}")

        adapters = header.get("adapters")
        if not isinstance(adapters, list) or len(adapters) != 1 or not isinstance(adapters[0], dict):
            raise RuntimeError("NCCL LoRA currently expects exactly one adapter in the update header")

        adapter = adapters[0]
        if header_expectation is not None:
            expected_step = header_expectation.get("step")
            if expected_step is not None and expected_step != step:
                raise RuntimeError(f"LoRA update was armed for step {expected_step}, not {step}")
            expected_adapters = header_expectation.get("adapters", [])
            if len(expected_adapters) != 1 or not isinstance(expected_adapters[0], dict):
                raise RuntimeError("LoRA update expectation must contain exactly one adapter")
            expected_adapter = expected_adapters[0]
            for key in ("lora_name", "lora_int_id"):
                if key in expected_adapter and adapter.get(key) != expected_adapter[key]:
                    raise RuntimeError(
                        f"LoRA update header {key}={adapter.get(key)!r} did not match expected "
                        f"{expected_adapter[key]!r}"
                    )

        for key in ("lora_name", "lora_int_id", "peft_config"):
            if key not in adapter:
                raise RuntimeError(f"LoRA update adapter header is missing {key!r}")
        return adapter

    def _commit_lora_adapter(self, adapter: dict[str, Any], tensors: dict[str, torch.Tensor]) -> None:
        if not tensors:
            raise RuntimeError("LoRA update contained no tensors")

        model_runner = getattr(self, "model_runner", None)
        lora_manager = getattr(model_runner, "lora_manager", None)
        if lora_manager is None:
            raise RuntimeError("LoRA is not enabled in the vLLM model runner")

        adapter_manager = getattr(lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("vLLM LoRA adapter manager is not initialized")

        from vllm.lora.peft_helper import PEFTHelper

        peft_config = dict(adapter["peft_config"])
        peft_helper = PEFTHelper.from_dict(peft_config)
        peft_helper.validate_legal(lora_manager.lora_config)

        model = adapter_manager.model
        lora_id = int(adapter["lora_int_id"])
        lora_model = lora_manager._lora_model_cls.from_lora_tensors(
            lora_model_id=lora_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
            dtype=getattr(lora_manager.lora_config, "lora_dtype", None),
            model_vocab_size=getattr(lora_manager, "vocab_size", None),
            weights_mapper=getattr(model, "hf_to_vllm_mapper", None),
            skip_prefixes=getattr(model, "lora_skip_prefixes", None),
        )
        lora_model.is_3d_lora_weight = bool(adapter.get("is_3d_lora_weight", False))

        adapter_manager.remove_adapter(lora_id)
        if (
            hasattr(adapter_manager, "capacity")
            and hasattr(adapter_manager, "remove_oldest_adapter")
            and len(adapter_manager) + 1 > adapter_manager.capacity
        ):
            adapter_manager.remove_oldest_adapter()
        adapter_manager.add_adapter(lora_model)
        adapter_manager.activate_adapter(lora_id)

    def wait_lora_receive(self, step: int) -> dict:
        state = getattr(self, "_lora_receive_state", None)
        if state is None:
            raise RuntimeError(f"No LoRA receive has been armed for step {step}")
        if state["step"] != step:
            raise RuntimeError(f"LoRA receive armed for step {state['step']}, not {step}")

        thread = getattr(self, "_lora_receive_thread", None)
        if thread is None:
            raise RuntimeError(f"LoRA receive thread missing for step {step}")
        thread.join()

        error = getattr(self, "_lora_receive_error", None)
        if error is not None:
            raise RuntimeError(f"LoRA receive failed for step {step}: {error!r}") from error
        return {"status": state["status"], "step": step}

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
        """
        # Use the worker's device index directly as the local rank.
        # The previous dp_group-based computation broke in vLLM v1 multiprocess
        # DP mode where each worker is a separate process with a singleton
        # DP group (rank_in_group is always 0).
        local_rank = self.device.index
        global_rank_inference = rank_offset + local_rank
        requested_config = {
            "host": host,
            "port": port,
            "rank_offset": rank_offset,
            "inference_world_size": inference_world_size,
            "timeout": timeout,
            "quantize_in_weight_transfer": quantize_in_weight_transfer,
            "local_rank": local_rank,
        }
        current_config = getattr(self, "_nccl_broadcaster_config", None)
        if current_config is not None:
            if current_config == requested_config:
                logger.info("NCCL broadcast receiver already initialized with matching config; skipping")
                self.quantize_in_weight_transfer = quantize_in_weight_transfer
                return
            raise RuntimeError(
                f"NCCL broadcast receiver already initialized with {current_config}; requested {requested_config}"
            )
        if hasattr(self, "nccl_broadcast_receiver"):
            raise RuntimeError("NCCL broadcast receiver already initialized without a recorded config")

        self.quantize_in_weight_transfer = quantize_in_weight_transfer

        logger.info(
            f"Worker [local_rank={local_rank} rank_offset={rank_offset}] "
            f"-> [global_rank={global_rank_inference} inference_world_size={inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
        )
        self._nccl_broadcaster_config = requested_config

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        if self.quantize_in_weight_transfer:
            load_weights_kernel(model, state_iter)
            update_mla_absorbed_weights(model)
            return

        # vLLM 0.22's layerwise reload drops NemotronH mixer.D's weight load (see
        # _restore_reload_corrupted_params). Capture the correct received value to restore after.
        received_reload_fix: dict[str, torch.Tensor] = {}

        def _capture_reload_fix(weights):
            for name, tensor in weights:
                if name.endswith(_RELOAD_CORRUPTED_SUFFIXES):
                    received_reload_fix[name] = tensor.detach().to("cpu", copy=True)
                yield name, tensor

        load_weights_checkpoint_layerwise(
            model,
            _capture_reload_fix(state_iter),
            self.model_runner.model_config,
            self.vllm_config,
        )
        _restore_reload_corrupted_params(model, received_reload_fix)
