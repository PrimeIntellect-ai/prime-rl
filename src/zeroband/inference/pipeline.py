import time
import pickle
from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
from vllm.model_executor.layers.sampler import SamplerOutput
from pydantic_config import BaseConfig
from vllm import LLM

from prime_iroh import Node
from zeroband.utils.logger import get_logger

# Global logger
logger = get_logger("INFER")


class PipelineConfig(BaseConfig):
    rank: int = 0
    world_size: int = 1
    iroh_seed: int | None = None
    iroh_peer_id: str | None = None


def setup_pipeline(llm: LLM, rank: int, world_size: int, iroh_seed: int | None = None, iroh_peer_id: str | None = None) -> Node:
    """
    Setup PRIME-IROH communication and hooks for pipeline parallel inference.

    Args:
        llm: The LLM model shard instance
        rank: The rank of the current process (this is equivalent to the model shard index)
        world_size: The total number of stages
        iroh_seed: The seed for the PRIME-IROH node (optional, will lead to deterministic connection strings)
        iroh_peer_id: The peer ID for the PRIME-IROH node (optional, will connect to a random peer if not provided)
    """
    node = setup_comm(world_size=world_size, iroh_seed=iroh_seed, iroh_peer_id=iroh_peer_id)
    setup_hooks(rank=rank, world_size=world_size, llm=llm, node=node)


def setup_comm(world_size: int, iroh_seed: int | None, iroh_peer_id: str | None) -> Node:
    """
    Setup communication via PRIME-IROH. Forms a ring topology between the model shards
    with unidirectional communication flow.

    Args:
        world_size: The total number of model shards
        iroh_seed: The seed for the PRIME-IROH node (optional, will lead to deterministic connection strings)
        iroh_peer_id: The peer ID for the PRIME-IROH node (optional, will connect to a random peer if not provided)
    """
    assert world_size > 1, "Pipeline parallel inference requires at least 2 stages"

    # Setup node (with or without seed)
    if iroh_seed is not None:
        # If seed is provided, create a new node with the seed
        node = Node.with_seed(num_streams=1, seed=iroh_seed)
    else:
        # If no seed, create a new node
        node = Node(num_streams=1)
    logger.info(f"Created node (ID={node.node_id()})")

    # Connect to peer
    if iroh_peer_id is None:
        iroh_peer_id = input("Enter Peer ID: ").strip()
    logger.info(f"Setting up outgoing connection to {iroh_peer_id}")
    node.connect(iroh_peer_id)
    logger.info(f"Outgoing connection to {iroh_peer_id} successful!")

    # Wait for connection to sender and receiver to be established
    # Note: This requires the PP communication loop to be closed, e.g. for 4 stages:
    # 0 -> 1 -> 2 -> 3 -> 0
    logger.info("Waiting for incoming connection...")
    while not node.is_ready():
        time.sleep(0.1)
    logger.info("All connections successful!")

    return node


def setup_hooks(rank: int, world_size: int, llm: LLM, node: Node) -> None:
    """
    Setup hooks to enable pipeline parallel inference based on pipeline topology.

    Args:
        rank: The stage index of the current process
        world_size: The total number of stages
        llm: The LLM model shard instance
        node: The node class instances for communication
    """
    assert world_size > 1, "Pipeline parallel inference requires at least 2 stages"

    # Model runner owns sampler, model owns layers
    model_runner: nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner
    model: nn.Module = model_runner.model

    # Extract first and last layers (pre/post-hook to recv/send intermediate states)
    first_layer: nn.Module = model.model.layers[0]
    last_layer: nn.Module = model.model.layers[-1]

    # Extract sampler (post-hook to recv/send outputs)
    sampler: nn.Module = model_runner.sampler

    # Don't relay outputs from stage with index -2->-1
    relay = rank != world_size - 2

    if rank == 0:  # First stage
        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.info("Registered post-hook send_intermediate_states on last layer")

        # Receive outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, node=node, relay=relay))
        logger.info("Registered post-hook recv_output on sampler")
    elif rank == world_size - 1:  # Last stage
        # Receive intermediate states from previous stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.info("Registered pre-hook recv_intermediate_states on first layer")

        # Send outputs to first  stage (post-hook)
        sampler.register_forward_hook(partial(send_output, node=node))
        logger.info("Registered post-hook send_output on sampler")
    else:
        # Receive intermediate states from previous stage and send positions to next stage (pre-hook)
        first_layer.register_forward_pre_hook(partial(recv_intermediate_states, node=node))
        logger.info("Registered pre-hook recv_intermediate_states on first layer")

        # Send intermediate states to next stage (post-hook)
        last_layer.register_forward_hook(partial(send_intermediate_states, node=node))
        logger.info("Registered post-hook send_intermediate_states on last layer")

        # Receive and relay outputs from last stage (post-hook)
        sampler.register_forward_hook(partial(recv_output, relay=relay))
        logger.info("Registered post-hook recv_output on sampler")


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
    serialized_hidden_states = pickle.dumps(hidden_states.to("cpu"))
    serialized_residual = pickle.dumps(residual.to("cpu"))
    node.isend(serialized_hidden_states, tag=0, latency=None).wait()
    node.isend(serialized_residual, tag=0, latency=None).wait()
    logger.debug(
        f"Sent hidden_states and residual ({hidden_states.shape}, {residual.shape}) ({len(serialized_hidden_states) + len(serialized_residual)} bytes)"
    )


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
    serialized_hidden_states = node.irecv(tag=0).wait()
    serialized_residual = node.irecv(tag=0).wait()
    hidden_states = pickle.loads(serialized_hidden_states).to(device)
    residuals = pickle.loads(serialized_residual).to(device)
    logger.debug(
        f"Got hidden_states and residuals ({hidden_states.shape}, {residuals.shape}) ({len(serialized_hidden_states) + len(serialized_residual)} bytes)"
    )

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
    output = pickle.loads(serialized_output)
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
    serialized_output = pickle.dumps(output)
    node.isend(serialized_output, tag=0, latency=None).wait()
    logger.debug(f"Sent outputs ({len(serialized_output)} bytes)")
