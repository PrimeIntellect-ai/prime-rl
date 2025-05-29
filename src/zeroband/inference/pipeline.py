import time
from functools import partial
from typing import Tuple

import msgspec
import torch
import torch.nn as nn
from prime_iroh import Node
from pydantic_config import BaseConfig
from safetensors.torch import load, save
from vllm import LLM
from vllm.model_executor.layers.sampler import SamplerOutput

from zeroband.utils.logger import get_logger

# Global logger
logger = get_logger("INFER")


class PipelineConfig(BaseConfig):
    # The rank of the current node in the pipeline
    rank: int = 0

    # The total number of nodes in the pipeline (e.g. the number of PP model shards)
    world_size: int = 1

    # The seed used to create the public node address (optional, will lead to deterministic connection strings)
    iroh_seed: int | None = None

    # The peer ID to connect to (optional, if not provided, the user will be prompted to enter it)
    iroh_peer_id: str | None = None

    # How many times to retry connection to peer (each retry takes ~30s)
    connection_num_retries: int = 10  # Each retry takes ~30s, so 10 retries is ~300s (5min)


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


def setup_pipeline(config: PipelineConfig, llm: LLM) -> Node:
    """
    Setup PRIME-IROH communication and hooks for pipeline parallel inference.

    Args:
        llm: The LLM model shard instance
        rank: The rank of the current process (this is equivalent to the model shard index)
        world_size: The total number of stages
        iroh_seed: The seed for the PRIME-IROH node (optional, will lead to deterministic connection strings)
        iroh_peer_id: The peer ID for the PRIME-IROH node (optional)
    """
    logger.info(f"Setting up pipeline parallelism (pp.rank={config.rank}, pp.world_size={config.world_size})")
    node = setup_comm(config=config)
    setup_hooks(config=config, llm=llm, node=node)
    return node


def setup_comm(config: PipelineConfig) -> Node:
    """
    Setup communication via PRIME-IROH. Forms a ring topology between the model shards
    with unidirectional communication flow.

    Args:
        world_size: The total number of model shards
        iroh_seed: The seed for the PRIME-IROH node (optional, will lead to deterministic connection strings)
        iroh_peer_id: The peer ID for the PRIME-IROH node (optional)
    """
    assert config.world_size > 1, "Pipeline parallel inference requires at least 2 stages"

    # Setup node (with or without seed)
    if config.iroh_seed is not None:
        logger.debug(f"Using IROH seed: {config.iroh_seed}")
        # If seed is provided, create a new node with the seed
        node = Node.with_seed(num_streams=1, seed=config.iroh_seed)
    else:
        # If no seed, create a new node
        node = Node(num_streams=1)
    logger.info(f"Created node (ID={node.node_id()})")

    # Connect to peer
    if config.iroh_peer_id is None:
        config.iroh_peer_id = input("Enter Peer ID: ").strip()
    logger.info(f"Setting up outgoing connection to {config.iroh_peer_id}")
    node.connect(config.iroh_peer_id, num_retries=config.connection_num_retries)
    logger.info(f"Outgoing connection to {config.iroh_peer_id} successful!")

    # Wait for connection to sender and receiver to be established
    # Note: This requires the PP communication loop to be closed, e.g. for 4 stages:
    # 0 -> 1 -> 2 -> 3 -> 0
    logger.info("Waiting for incoming connection...")
    while not node.is_ready():
        time.sleep(0.1)
    logger.info("All connections successful!")

    return node


def setup_hooks(config: PipelineConfig, llm: LLM, node: Node) -> None:
    """
    Setup hooks to enable pipeline parallel inference based on pipeline topology.

    Args:
        rank: The stage index of the current process
        world_size: The total number of stages
        llm: The LLM model shard instance
        node: The node class instances for communication
    """
    assert config.world_size > 1, "Pipeline parallel inference requires at least 2 stages"

    # Model runner owns sampler, model owns layers
    model_runner: nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner
    model: nn.Module = model_runner.model

    # Extract first and last layers (pre/post-hook to recv/send intermediate states)
    first_layer: nn.Module = model.model.layers[0]
    last_layer: nn.Module = model.model.layers[-1]

    # Extract sampler (post-hook to recv/send outputs)
    sampler: nn.Module = model_runner.sampler

    # Don't relay outputs from stage with index -2->-1
    relay = config.rank != config.world_size - 2

    if config.rank == 0:  # First stage
        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.debug("Registered post-hook send_intermediate_states on last layer")

        # Receive outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, node=node, relay=relay))
        logger.debug("Registered post-hook recv_output on sampler")
    elif config.rank == config.world_size - 1:  # Last stage
        # Receive intermediate states from previous stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.debug("Registered pre-hook recv_intermediate_states on first layer")

        # Send outputs to first  stage (post-hook)
        sampler.register_forward_hook(partial(send_output, node=node))
        logger.debug("Registered post-hook send_output on sampler")
    else:
        # Receive intermediate states from previous stage and send positions to next stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.debug("Registered pre-hook recv_intermediate_states on first layer")

        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.debug("Registered post-hook send_intermediate_states on last layer")

        # Receive and relay outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, node=node, relay=relay))
        logger.debug("Registered post-hook recv_output on sampler")


# TODO: Outputs of decoder blocks look different for vLLM implementations and HF-based implementations. The implementation currently breaks for HF-based implementations.
def send_intermediate_states(_, __, output: Tuple, node: Node) -> None:
    """
    A post-hook that sends the hidden states and residual of the last decoder layer to the next stage node's first layer.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        output: The output of the module (here the decoder layer output)
        node: The node that is being hooked
    """
    hidden_states, residual = output
    serialized_tensors = serialize_tensors({"hidden_states": hidden_states, "residual": residual})
    node.isend(serialized_tensors, tag=0, latency=None).wait()
    logger.debug(f"Sent hidden_states and residual ({hidden_states.shape}, {residual.shape}) ({len(serialized_tensors)} bytes)")


def recv_intermediate_states(_, input: Tuple, node: Node) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A pre-hook that receives the hidden states and residual from the previous stage node's last layer at the first layer of the current node.

    Assumes the node is correctly set up to receive hidden states and residual from the previous node.

    Args:
        _: The module that is being hooked
        input: The input to the module (here the positions, hidden states and residual of the previous node's last layer)
        node: The node class instances for communication
    """
    positions, _, _ = input
    device = positions.device
    serialized_tensors = node.irecv(tag=0).wait()
    deserialized_tensors = deserialize_tensors(serialized_tensors, device)
    hidden_states = deserialized_tensors["hidden_states"]
    residuals = deserialized_tensors["residual"]
    logger.debug(f"Got hidden_states and residuals ({hidden_states.shape}, {residuals.shape}) ({len(serialized_tensors)} bytes)")

    return positions, hidden_states, residuals


def recv_output(_, __, output, node: Node, relay=False) -> SamplerOutput:
    """
    A post-hook that receives sampling outputs from the last stage node and optionally relays them to the next stage node.
    For a pipeline with 4 stages, this hook should be registered as follows:

    Rank 1: Receive output + relay
    Rank 2: Receive output + relay
    Rank 3: Receive output
    Rank 4: *Do not register hook* (use the `send_output` hook)

    Receiving and relaying the outputs is necessary for the schedulers to be synchronized across stages.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        ____: The outputs of the module
        node: The node class instances for communication
        relay: Whether to relay the outputs to the next stage node
    """
    serialized_output = node.irecv(tag=0).wait()
    logger.debug(f"Received outputs ({len(serialized_output)} bytes)")
    if relay:
        node.isend(serialized_output, tag=0, latency=None).wait()
        logger.debug(f"Sent outputs ({len(serialized_output)} bytes)")
    output = deserialize_sampler_output(serialized_output)
    return output


def send_output(_, __, output: SamplerOutput, node: Node) -> None:
    """
    A post-hook that sends the sampling outputs from the last stage node to the first stage node.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        output: The outputs of the module
        node: The node class instances for communication
    """
    serialized_output = serialize_sampler_output(output)
    node.isend(serialized_output, tag=0, latency=None).wait()
    logger.debug(f"Sent outputs ({len(serialized_output)} bytes)")


def all_reduce(node: Node, tensor: torch.Tensor, config: PipelineConfig, op: callable = torch.add) -> torch.Tensor:
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
    if config.world_size == 1:
        logger.debug("No communication needed to all-reduce tensor with world_size=1")
        return tensor

    result_tensor = tensor.clone()
    current_tensor = tensor.clone()
    logger.debug(f"Initial tensor: {current_tensor}")

    # Ring all-reduce: each node sends/receives for (world_size - 1) iterations
    for _ in range(config.world_size - 1):
        # Serialize current tensor for transmission
        tensor_dict = {"data": current_tensor}
        send_data = serialize_tensors(tensor_dict)
        logger.debug(f"Sending {current_tensor} ({len(send_data)} bytes) to next node")
        send_future = node.isend(send_data, tag=0, latency=None)

        # Receive tensor from previous node
        recv_future = node.irecv(tag=0)

        # Wait for both operations to complete
        send_future.wait()
        recv_data = recv_future.wait()

        # Deserialize received tensor and apply reduction operation
        received_tensors = deserialize_tensors(recv_data)
        logger.debug(f"Received {received_tensors['data']} ({len(recv_data)} bytes) from previous node")
        current_tensor = received_tensors["data"]

        # Apply the custom reduction operation
        result_tensor = op(result_tensor, current_tensor)

    logger.debug(f"All-reduced tensor: {result_tensor}")

    return result_tensor
