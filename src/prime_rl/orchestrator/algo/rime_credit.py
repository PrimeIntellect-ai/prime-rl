"""RIME credit assignment. Arrays are [sibling, task position]."""

from dataclasses import dataclass

import numpy as np


def matrix(value, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or min(out.shape) < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite, nonempty [G,T] matrix")
    return out


def normalize_siblings(value, epsilon=1e-8):
    """Population standard deviation; constant groups receive zero advantage."""
    value = matrix(value, "rewards")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    centered = value - value.mean(axis=0, keepdims=True)
    std = value.std(axis=0, keepdims=True)
    return np.divide(centered, std, out=np.zeros_like(value), where=std > epsilon)


@dataclass(frozen=True)
class Credit:
    actor: np.ndarray
    writer: np.ndarray
    writer_reward: np.ndarray
    writer_mask: np.ndarray


def assign_suffix_credit(task, exploration, well_formed):
    """WRITE after position t receives only outcomes at positions greater than t.

    Format validity is diagnostic: invalid writes retain old memory and receive
    the same task-return rule. Terminal writes have no future task credit.
    """
    task = matrix(task, "task")
    exploration = matrix(exploration, "exploration")
    gates = matrix(well_formed, "well_formed")
    if task.shape != exploration.shape or task.shape != gates.shape:
        raise ValueError("All matrices must have the same [G,T] shape")
    if task.shape[0] < 2 or task.shape[1] < 2:
        raise ValueError("Writer suffix credit needs at least two siblings and tasks")
    if np.any(exploration != 0) or not np.isin(gates, [0, 1]).all():
        raise ValueError("Writer suffix credit forbids exploration scores and requires binary diagnostics")
    reward = np.zeros_like(task)
    reward[:, :-1] = np.cumsum(task[:, :0:-1], axis=1)[:, ::-1]
    mask = np.ones_like(task, dtype=bool)
    mask[:, -1] = False
    return Credit(np.zeros_like(task), normalize_siblings(reward), reward, mask)


def assign_credit(task, exploration, well_formed, *, horizon, beta):
    """Past window includes current task; future window starts at the next task.

    Boundary convention (explicit reproduction choice): shorten windows at the
    ends of the stream. Invalid WRITE outputs have reward zero BEFORE group
    normalization, not a removed sample. Final WRITE has no downstream credit.
    """
    task, exploration, gate = (
        matrix(x, n) for x, n in ((task, "task"), (exploration, "exploration"), (well_formed, "well_formed"))
    )
    if task.shape != exploration.shape or task.shape != gate.shape:
        raise ValueError("All reward matrices must have the same [G,T] shape")
    if task.shape[0] < 2 or task.shape[1] < 2:
        raise ValueError("Joint training needs at least two siblings and two tasks")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if ((exploration < 0) | (exploration > 1)).any() or not np.isin(gate, [0, 1]).all():
        raise ValueError("Exploration must be in [0,1] and format gates must be binary")
    reward = np.zeros_like(task)
    for t in range(task.shape[1] - 1):
        past = task[:, max(0, t - horizon + 1) : t + 1].mean(axis=1)
        future = task[:, t + 1 : t + 1 + horizon].mean(axis=1)
        reward[:, t] = gate[:, t] * (future - past)
    mask = np.ones_like(task, dtype=bool)
    mask[:, -1] = False
    return Credit(
        normalize_siblings(task) + beta * normalize_siblings(exploration), normalize_siblings(reward), reward, mask
    )
