"""GEPA adapter for disjoint General Agent train and validation task sets."""

from __future__ import annotations

from general_agent.solver.local.env import load_environment as load_general_agent


def load_environment(train_tasks_dir: str, eval_tasks_dir: str, **kwargs):
    train_env = load_general_agent(tasks_dir=train_tasks_dir, **kwargs)
    eval_env = load_general_agent(tasks_dir=eval_tasks_dir, **kwargs)
    train_env.eval_dataset = eval_env.dataset
    return train_env
