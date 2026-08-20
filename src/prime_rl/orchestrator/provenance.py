"""Strict access to provenance required by the orchestrator."""

from typing import Any

import verifiers.v1 as vf


def episode_env_name(episode: vf.Episode[Any, Any, Any]) -> str:
    name = episode.env.name
    if name is None:
        raise ValueError("Orchestrated episode is missing its environment name")
    return name


def episode_group_id(episode: vf.Episode[Any, Any, Any]) -> str:
    group = episode.group
    if group is None:
        raise ValueError("Orchestrated episode is missing its rollout group")
    return group.id


def train_work(episode: vf.Episode[Any, Any, Any]) -> vf.TrainWorkInfo:
    run = episode.run
    if not isinstance(run, vf.TrainRunInfo) or not isinstance(run.work, vf.TrainWorkInfo):
        raise ValueError("Train episode is missing training-work provenance")
    return run.work


def eval_work(episode: vf.Episode[Any, Any, Any]) -> vf.EvalWorkInfo:
    run = episode.run
    if not isinstance(run, vf.TrainRunInfo) or not isinstance(run.work, vf.EvalWorkInfo):
        raise ValueError("Eval episode is missing evaluation-work provenance")
    return run.work
