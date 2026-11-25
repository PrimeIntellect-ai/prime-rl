"""Training annotations and transport loss routing."""

from __future__ import annotations

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.orchestrator.credit import ADVANTAGE_KEY, RL_WEIGHT_KEY, trace_advantage
from prime_rl.transports.batch import TrainingSample


def trainable_nodes(trace: vf.Trace) -> list[vf.MessageNode]:
    """Nodes with sampled (mask-True) tokens on the trace's trainable branches, in graph order."""
    on_trainable_branch = {id(node) for branch in trace.branches if branch.trainable for node in branch.nodes}
    return [node for node in trace.nodes if id(node) in on_trainable_branch and np.any(node.mask)]


def assign_advantage(trace: vf.Trace, value: float) -> None:
    """Credit every sampled token on the trace's trainable branches with ``value``. A trace
    with no such token gets no credit."""
    if trainable_nodes(trace):
        trace.info[ADVANTAGE_KEY] = float(value)


def assign_rl_weight(trace: vf.Trace, value: float) -> None:
    """Weight the rl loss of every sampled token on the trace's trainable branches by ``value``."""
    trace.info[RL_WEIGHT_KEY] = float(value)


def assign_reference_logprobs(branch: vf.Branch, values: list[float] | np.ndarray) -> None:
    """Project branch-aligned reference logprobs onto sampled graph nodes."""
    if len(values) != len(branch.token_ids):
        raise ValueError(
            f"reference logprobs must align with branch tokens: got {len(values)}, expected {len(branch.token_ids)}"
        )
    values = np.asarray(values, dtype=np.float32)
    offset = 0
    for node in branch.nodes:
        end = offset + len(node.token_ids)
        mask = np.asarray(node.mask, dtype=bool)
        if mask.any() and node.reference_logprobs is None:
            node.reference_logprobs = values[offset:end][mask]
        offset = end


def is_trainable(trace: vf.Trace) -> bool:
    """Whether the trace's sampled tokens carry nonzero RL credit."""
    return bool(trace_advantage(trace))


def stamp_loss_routing(sample: TrainingSample, action_loss_type: ActionLossType) -> None:
    """Route action tokens into the algorithm's declared loss component."""
    if action_loss_type == "rl":
        return

    seq_len = len(sample.token_ids)
    sample.rl_weights = np.zeros(seq_len, dtype=np.float32)
    action_weights = (
        np.array(sample.ce_weights, dtype=np.float32)
        if action_loss_type == "ce" and sample.ce_weights is not None
        else np.zeros(seq_len, dtype=np.float32)
    )
    action_weights[np.asarray(sample.mask, dtype=bool)] = 1.0
    if action_loss_type == "ce":
        sample.ce_weights = action_weights
    else:
        assert action_loss_type == "ref_kl"
        sample.ref_kl_weights = action_weights
