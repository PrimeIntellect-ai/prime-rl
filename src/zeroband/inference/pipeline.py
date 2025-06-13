import time
from functools import partial
from typing import Tuple

import msgspec
import torch
import torch.nn as nn
from prime_iroh import Node
from safetensors.torch import load, save
from vllm import LLM
from vllm.distributed.parallel_state import get_tp_group
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.model_executor.layers.sampler import SamplerOutput

from zeroband.inference.config import PipelineParallelConfig
from zeroband.inference.utils import rgetattr
from zeroband.utils.logger import get_logger


def serialize_tensors(tensor_dict: dict[str, torch.Tensor]) -> bytes:
    """Safely serializes a dictionary of tensors to bytes."""
    return save(tensor_dict)


def deserialize_tensors(data: bytes, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    """Safely deserializes a dictionary of tensors from bytes."""
    tensor_dict = load(data)
    if device is not None:
        return {key: tensor.to(device) for key, tensor in tensor_dict.items()}
    return tensor_dict


def serialize_sampler_output(output: SamplerOutput) -> bytes:
    """Safely serializes a vLLM SamplerOutput object"""
    return msgspec.json.encode(output)


def deserialize_sampler_output(data: bytes) -> SamplerOutput:
    """Safely deserializes a vLLM SamplerOutput object"""
    return msgspec.json.decode(data, type=SamplerOutput)


def setup_comm(config: PipelineParallelConfig) -> Node | None:
    """
    Setup P2P communication via using `prime-iroh` nodes. Forms a ring topology
    between the model shards with unidirectional communication flow.

    Args:
        config: The pipeline configuration

    Returns:
        The node if world_size > 1, otherwise None
    """
    if not config.is_enabled:
        return None

    logger = get_logger("INFER")
    logger.info(f"Setting up pipeline parallel communication ({config})")
    start_time = time.time()

    # Setup node (with or without seed)
    if config.iroh_seed is not None:
        logger.debug(f"Using seed: {config.iroh_seed}")
        # If seed is provided, create a new node with the seed
        node = Node.with_seed(num_streams=1, seed=config.iroh_seed)
    else:
        # If no seed, create a new node
        node = Node(num_streams=1)
    logger.info(f"Initialized node ({node.node_id()})")

    # Connect to peer
    if config.iroh_peer_id is None:
        config.iroh_peer_id = input("Enter peer address: ").strip()
    logger.info(f"Setting up outgoing connection to {config.iroh_peer_id}")
    node.connect(config.iroh_peer_id, num_retries=config.connection_num_retries)
    logger.info(f"Outgoing connection to {config.iroh_peer_id} successful!")

    # Wait for connection to sender and receiver to be established
    # Note: This requires the PP communication loop to be closed, e.g. for 4 stages:
    # 0 -> 1 -> 2 -> 3 -> 0
    logger.info("Waiting for incoming connection")
    while not node.is_ready():
        time.sleep(0.1)
    logger.info("Incoming connection successful!")
    logger.info(f"Pipeline parallel communication setup in {time.time() - start_time:.2f}s")

    return node


def patch_model_load(config: PipelineParallelConfig) -> None:
    """
    Patch the vLLM model load to only load the correct model shard.
    """
    import vllm.model_executor.models.utils as model_utils
    from vllm.model_executor.models.utils import LayerFn, PPMissingLayer, maybe_offload_to_cpu

    # Skip patching if world_size == 1
    if not config.is_enabled:
        return

    def _patched_make_layers(num_hidden_layers: int, layer_fn: LayerFn, prefix: str) -> Tuple[int, int, torch.nn.ModuleList]:
        """
        This is a patched version of the `make_layers` function in vLLM which is
        called when PP is used internally. It returns the index of the first and
        last layer for the current shard. The only difference to the original
        function is that we pass the PP rank and world size directly to the
        `get_pp_indices` function, instead of getting them from the PP
        torch.distributed group (vLLM default).

        Args:
            num_hidden_layers: The total number of hidden layers in the model
            layer_fn: The function to create a layer
            prefix: The prefix to use for the layer

        Returns:
            The index of the first and last layer for the current shard, and the nn.ModuleList of the layers
        """
        from vllm.distributed.utils import get_pp_indices

        start_layer, end_layer = get_pp_indices(num_hidden_layers, config.rank, config.world_size)
        modules = torch.nn.ModuleList(
            [PPMissingLayer() for _ in range(start_layer)]
            + [maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}")) for idx in range(start_layer, end_layer)]
            + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
        )
        return start_layer, end_layer, modules

    # Monkey patch the function
    get_logger("INFER").info(f"Patching model init for pp.rank={config.rank} in pp.world_size={config.world_size}")
    model_utils.make_layers = _patched_make_layers


def setup_hooks(
    llm: LLM,
    config: PipelineParallelConfig,
    node: Node | None,
    start_layer_key: str = "model.start_layer",
    end_layer_key: str = "model.end_layer",
    model_layers_key: str = "model.layers",
) -> None:
    """
    Sets up hooks to enable pipeline parallel inference. Directly sets up the
    main hooks on the driver worker (current process) to receive and relay
    intermediate states and outputs. For non-driver workers, we setup a hook
    with a blocking receive to receive intermediate states from the driver worker.
    This is only needed when TP and PP are enabled.

    NB:
    - We assume that the vLLM model class (i.e. the nn.Module implmenting the
    forward pass) has an attribute `start_layer` and `end_layer` that indicates
    index of first and last layer (+1) of the current model shard. It can be
    accessed by the `start_layer_key` and `end_layer_key` arguments.
    - We assume that the vLLM model class has an attribute `layers` that is a
    list of layers. It can be accessed via the `model_layers_key` argument.

    Args:
        llm: The vLLM instance
        config: The pipeline configuration
        node: The node class instances for communication (None if world_size == 1)
        start_layer_key: The key to the start layer in the model (e.g. "model.start_layer")
        end_layer_key: The key to the end layer in the model (e.g. "model.end_layer")
        model_layers_key: The key to the layers in the model (e.g. "model.layers")
    """
    # Setup driver hooks
    driver_worker = llm.llm_engine.model_executor.driver_worker
    setup_hooks_driver(driver_worker, config, node, start_layer_key, end_layer_key, model_layers_key)

    # Setup non-driver hooks
    model_executor = llm.llm_engine.model_executor

    if isinstance(model_executor, MultiprocessingDistributedExecutor):
        model_executor.collective_rpc(
            lambda worker: setup_hooks_non_driver(worker, config, start_layer_key, model_layers_key),
            kwargs=dict(async_run_tensor_parallel_workers_only=True),
        )


def setup_hooks_driver(
    worker,
    config: PipelineParallelConfig,
    node: Node | None,
    start_layer_key: str,
    end_layer_key: str,
    model_layers_key: str,
) -> None:
    """
    Setup hooks on the driver worker (current process) for pipeline parallel
    communication. Installs different hooks depending on the stage of the pipeline:

    1. First stage:
    - Receive sample outputs from last stage
    - Send intermediate states to next stage

    2. Last stage:
    - Receive intermediate states from previous stage
    - Broadcast intermediate states within TP group
    - Send sample outputs to first stage

    3. Intermediate stages:
    - Receive intermediate states from previous stage
    - Broadcast intermediate states within TP group
    - Send intermediate states to next stage
    - Receive and relay sample outputs to the next stage (unless next stage is the last stage)

    Note that the order in which the hooks of the same type are registered is
    important. For example, the non-first stages need to first receive the
    intermediate states before broadcasting them within the TP group.

    Args:
        worker: The worker object (passed by collective_rpc)
        config: The pipeline configuration
        node: The node class instances for communication (None if world_size == 1)
        start_layer_key: The key to the start layer in the model (e.g. "model.start_layer")
        end_layer_key: The key to the end layer in the model (e.g. "model.end_layer")
        model_layers_key: The key to the layers in the model (e.g. "model.layers")
    """
    if not config.is_enabled:
        assert node is None, "Node should be None if pipeline is disabled"
        return

    # Model runner owns sampler, model owns layers
    model_runner = worker.model_runner
    model = worker.get_model()

    # Extract first and last layers (pre/post-hook to recv/send intermediate states)
    first_layer_idx = rgetattr(model, start_layer_key)
    last_layer_idx = rgetattr(model, end_layer_key) - 1
    first_layer: nn.Module = rgetattr(model, model_layers_key)[first_layer_idx]
    last_layer: nn.Module = rgetattr(model, model_layers_key)[last_layer_idx]

    # Extract sampler (post-hook to recv/send outputs)
    sampler: nn.Module = model_runner.sampler

    # Don't relay outputs from stage with index -2->-1
    relay = config.rank != config.world_size - 2

    # Get logger
    logger = get_logger("INFER")

    if config.is_first_stage:  # First stage
        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.debug("Registered post-hook send_intermediate_states on last layer")

        # Receive outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, node=node, relay=relay))
        logger.debug("Registered post-hook recv_output on sampler")
    elif config.is_last_stage:  # Last stage
        # Receive intermediate states from previous stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.debug("Registered pre-hook recv_intermediate_states on first layer")

        # Broadcast intermediate states within TP group (pre-hook)
        first_layer.register_forward_pre_hook(broadcast_intermediate_states)
        logger.debug("Registered pre-hook broadcast_intermediate_states on first layer")

        # Send outputs to first  stage (post-hook)
        sampler.register_forward_hook(partial(send_output, node=node))
        logger.debug("Registered post-hook send_output on sampler")
    else:
        # Receive intermediate states from previous stage and send positions to next stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.debug("Registered pre-hook recv_intermediate_states on first layer")

        # Broadcast intermediate states within TP group (pre-hook)
        first_layer.register_forward_pre_hook(broadcast_intermediate_states)
        logger.debug("Registered pre-hook broadcast_intermediate_states on first layer")

        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.debug("Registered post-hook send_intermediate_states on last layer")

        # Receive and relay outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, node=node, relay=relay))
        get_logger("INFER").debug("Registered post-hook recv_output on sampler")


def setup_hooks_non_driver(
    worker,
    config: PipelineParallelConfig,
    start_layer_key: str,
    model_layers_key: str,
) -> None:
    """
    Setup hooks on non-driver worker via remote RPC call (called by vLLM model
    executor). For all but the first stage, we need to receive the intermediate
    states from the previous stage by listening to a broadcast from the driver
    worker. This is only required when TP and PP are enabled.

    Note that all arguments to this function must be pickleable.

    Args:
        worker: The worker object (passed by collective_rpc)
        config: The pipeline configuration
        node: The node class instances for communication (None if world_size == 1)
        start_layer_key: The key to the start layer in the model (e.g. "model.start_layer")
        end_layer_key: The key to the end layer in the model (e.g. "model.end_layer")
        model_layers_key: The key to the layers in the model (e.g. "model.layers")
    """
    if not config.is_enabled:
        return

    # Model owns layers
    model = worker.get_model()

    # Extract first and last layers (pre/post-hook to recv/send intermediate states)
    first_layer_idx = rgetattr(model, start_layer_key)
    first_layer: nn.Module = rgetattr(model, model_layers_key)[first_layer_idx]

    if not config.is_first_stage:  # Not first stage
        # Receive intermediate states from TP driver worker (pre-hook)
        first_layer.register_forward_pre_hook(broadcast_intermediate_states)
        get_logger("INFER").debug("Registered pre-hook broadcast_intermediate_states on first layer")


def broadcast_intermediate_states(_, input: Tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A pre-hook that broadcasts the hidden states and residual from TP rank 0 to
    all other ranks. Note that if the TP size is 1, this is a no-op.

    Args:
        _: The module that is being hooked
        input: The input to the module (here the positions, hidden states and residual of a decoder layer)

    Returns:
        The modified input tuple (positions, hidden_states, residual)
    """
    positions, hidden_states, residual = input
    if residual is None:
        residual = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
    get_tp_group().broadcast(hidden_states)
    get_tp_group().broadcast(residual)
    # logger.debug("Broadcasted hidden_states and residual")

    return positions, hidden_states, residual


def send_intermediate_states(_, __, output: Tuple, node: Node) -> None:
    """
    A post-hook that sends the hidden states and residual of the last decoder
    layer to the next stage.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        output: The output of the module (here the decoder layer output)
        node: The node that is being hooked
    """
    hidden_states, residual = output
    serialized_tensors = serialize_tensors({"hidden_states": hidden_states, "residual": residual})
    node.isend(serialized_tensors, tag=0, latency=None).wait()
    # get_logger("INFER").debug(f"Sent hidden_states and residual ({hidden_states.shape}, {residual.shape}) ({len(serialized_tensors)} bytes)")


def recv_intermediate_states(_, input: Tuple, node: Node) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A pre-hook that receives the hidden states and residual from the previous
    stage at the first layer of the current node.

    Args:
        _: The module that is being hooked
        input: The input to the module (here the positions, hidden states and residual of the previous node's last layer)
        node: The node class instances for communication

    Returns:
        The modified input tuple (positions, hidden_states, residual)
    """
    positions, _, _ = input
    device = positions.device
    serialized_tensors = node.irecv(tag=0).wait()
    deserialized_tensors = deserialize_tensors(serialized_tensors, device)
    hidden_states = deserialized_tensors["hidden_states"]
    residuals = deserialized_tensors["residual"]
    # get_logger("INFER").debug(f"Received hidden_states and residuals ({hidden_states.shape}, {residuals.shape}) ({len(serialized_tensors)} bytes)")

    return positions, hidden_states, residuals


def recv_output(_, __, output, node: Node, relay=False) -> SamplerOutput:
    """
    A post-hook that receives sampling outputs from the last stage node and
    optionally relays them to the next stage node.  For a pipeline with 4
    stages, this hook should be registered as follows:

    Rank 1: Receive output + relay
    Rank 2: Receive output + relay
    Rank 3: Receive output
    Rank 4: *Do not register hook* (use the `send_output` hook)

    Receiving and relaying the outputs is necessary for the schedulers to be
    synchronized across stages.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        ____: The outputs of the module
        node: The node class instances for communication
        relay: Whether to relay the outputs to the next stage node

    Returns:
        The sampling outputs
    """
    serialized_output = node.irecv(tag=0).wait()
    # get_logger("INFER").debug(f"Received outputs ({len(serialized_output)} bytes)")
    if relay:
        node.isend(serialized_output, tag=0, latency=None).wait()
        # get_logger("INFER").debug(f"Sent outputs ({len(serialized_output)} bytes)")
    output = deserialize_sampler_output(serialized_output)
    return output


def send_output(_, __, output: SamplerOutput, node: Node) -> None:
    """
    A post-hook that sends the sampling outputs from the last stage node to the
    first stage node.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        output: The outputs of the module
        node: The node class instances for communication

    Returns:
        None
    """
    serialized_output = serialize_sampler_output(output)
    node.isend(serialized_output, tag=0, latency=None).wait()
    # get_logger("INFER").debug(f"Sent outputs ({len(serialized_output)} bytes)")


def all_reduce(node: Node, tensor: torch.Tensor, config: PipelineParallelConfig, op: callable = torch.add) -> torch.Tensor:
    """
    Performs a ring all-reduce operation on tensors with a custom reduction operation.

    Args:
        node: The communication node (already organized in a ring topology)
        tensor: The tensor value to reduce
        config: Inference config containing world_size
        op: Custom reduction operation (e.g., torch.add, torch.max, torch.min)

    Returns:
        The reduced tensor after applying the operation across all nodes in the ring
    """
    # No communication needed for single node
    if not config.is_enabled:
        return tensor

    result_tensor = tensor.clone()
    current_tensor = tensor.clone()
    get_logger("INFER").debug(f"Initial tensor: {current_tensor}")

    # Ring all-reduce: each node sends/receives for (world_size - 1) iterations
    for _ in range(config.world_size - 1):
        # Serialize current tensor for transmission
        tensor_dict = {"data": current_tensor}
        send_data = serialize_tensors(tensor_dict)
        # get_logger("INFER").debug(f"Sending {current_tensor} ({len(send_data)} bytes) to next node")
        send_future = node.isend(send_data, tag=0, latency=None)

        # Receive tensor from previous node
        recv_future = node.irecv(tag=0)

        # Wait for both operations to complete
        send_future.wait()
        recv_data = recv_future.wait()

        # Deserialize received tensor and apply reduction operation
        received_tensors = deserialize_tensors(recv_data)
        # logger.debug(f"Received {received_tensors['data']} ({len(recv_data)} bytes) from previous node")
        current_tensor = received_tensors["data"]

        # Apply the custom reduction operation
        result_tensor = op(result_tensor, current_tensor)

    get_logger("INFER").debug(f"All-reduced tensor: {result_tensor}")

    return result_tensor
