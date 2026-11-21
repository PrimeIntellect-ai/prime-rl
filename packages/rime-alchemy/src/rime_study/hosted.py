"""Verifiers-native stream environment for the RIME prime-rl algorithm.

Each episode is ONE sibling's complete ten-trial stream. prime-rl groups episodes
for the same source task, then RIMEAlgorithm assigns cross-sibling advantages.
"""

import json

import verifiers.v1 as vf
from rime_alchemy.configs import RIMEAlchemyEnvConfig, RIMEAlchemyTasksetConfig, StreamData, StreamTask

from rime_study.alchemy import AlchemyStream
from rime_study.data import ROOT, training_streams, verify_sources
from rime_study.decision_judge import MemoryDecisionJudge
from rime_study.rollout import ACT, WRITE, ProtocolError


class TurnTask(vf.Task[vf.TaskData]):
    pass


class RIMEAlchemyTaskset(vf.Taskset[StreamTask, RIMEAlchemyTasksetConfig]):
    def load(self):
        verify_sources()
        split = json.loads((ROOT / "data/development_splits.json").read_text())
        if self.config.split not in {"train", "validation"}:
            raise ValueError("RIME supports only the audited development train/validation pools")
        allowed = set(split[f"{self.config.split}_episode_indices"])
        for row in training_streams():
            index = row["metadata"]["episode_index"]
            if index in allowed:
                yield StreamTask(
                    StreamData(name=f"alchemy-{index}", prompt="Run the Alchemy stream", episode_index=index),
                    self.config.task,
                )


class RIMEAlchemyEnv(vf.Env[RIMEAlchemyEnvConfig]):
    async def setup(self, agents: vf.Agents):
        agents.actor.trainable = True
        agents.writer.trainable = True
        judge_config = self.config.exploration_judge
        if self.config.judge_local:
            client = agents.actor.config.client
            if client is None:
                raise ValueError("Local Clef needs the resolved actor inference client")
            judge_config = judge_config.model_copy(update={
                "base_url": client.base_url.rstrip("/") + "/rime/judge",
                "api_key_var": client.api_key_var,
                "headers": client.headers,
                "headers_from_env": client.headers_from_env,
            })
        self.memory_judge = MemoryDecisionJudge(
            judge_config,
            max_concurrent=self.config.judge_max_concurrent,
            max_retries=self.config.judge_max_retries,
            retry_timeout=self.config.judge_retry_timeout,
            transport_model=agents.actor.config.model if self.config.judge_local else None,
        )

    async def run(self, task: StreamTask, agents: vf.Agents):
        env = AlchemyStream(task.data.episode_index)
        memory = ""
        rewards, scores, gates, training = [], [], [], []

        async def turn(agent, instruction, payload):
            trace = await agent.run(TurnTask(vf.TaskData(prompt=json.dumps(payload), system_prompt=instruction)))
            if trace.has_error or trace.is_format_violation or trace.is_truncated or trace.num_turns == 0:
                raise ProtocolError("Incomplete or truncated RIME model turn")
            return trace, trace.last_reply or ""

        for t in range(env.num_trials):
            if not self.config.carry_memory:
                memory = ""
            history = []
            initial_observation = env.observation()
            invalid = 0
            actor_task = TurnTask(
                vf.TaskData(
                    system_prompt=ACT, prompt=json.dumps({"memory": memory, "initial_observation": initial_observation})
                )
            )
            # One trace per complete ACT trajectory keeps whole stream cohorts
            # together when the train sink counts samples by trace.
            async with agents.actor.interaction(actor_task) as interaction:
                feedback = None
                for _ in range(env.max_steps + 2):
                    segment = await interaction.turn(feedback)
                    if segment.terminated or interaction.trace.is_truncated:
                        raise ProtocolError("Actor terminated before finishing the trial")
                    reply = segment.last_reply
                    try:
                        result = env.step(json.loads(reply))
                    except (ValueError, TypeError, KeyError) as exc:
                        invalid += 1
                        if invalid > 1:
                            raise ProtocolError("Actor exhausted action repair budget") from exc
                        result = {"error": str(exc), "observation": env.observation()}
                    history.append({"actor": reply, "result": result})
                    if result.get("trial_done"):
                        rewards.append(result["trial_reward"])
                        break
                    feedback = json.dumps(result)
                else:
                    raise ProtocolError("Actor exceeded action budget")
            actor_trace = interaction.trace
            if actor_trace.has_error or actor_trace.is_format_violation or actor_trace.is_truncated:
                raise ProtocolError("Incomplete actor trajectory")
            # Character cap is explicit for this adapter; no tokenizer/model
            # download is needed in the environment server.
            writer_trace, reply = await turn(
                agents.writer,
                WRITE.replace("token budget", "character budget"),
                {
                    "previous_memory": memory,
                    "trajectory": {"initial_observation": initial_observation, "turns": history},
                    "memory_character_budget": self.config.memory_chars,
                },
            )
            try:
                parsed = json.loads(reply)
                proposed = parsed["memory"]
                valid = (
                    set(parsed) == {"memory"}
                    and isinstance(proposed, str)
                    and len(proposed) <= self.config.memory_chars
                )
            except (ValueError, TypeError, KeyError):
                valid = False
            score = 0.0
            if valid:
                score = await self.memory_judge.score_memories(memory, proposed, trace=writer_trace)
                memory = proposed
            gates.append(int(valid))
            scores.append(float(score))
            training.append((actor_trace, "actor", t))
            actor_trace.record_reward("task", rewards[-1])
            actor_trace.record_metric("rime/task_raw_reward", rewards[-1])
            actor_trace.record_metric("rime/exploration_reward", float(score))
            writer_trace.record_metric("rime/writer_format_valid", float(valid))
            writer_trace.record_metric("rime/memory_chars", float(len(memory)))
            if t < env.num_trials - 1:
                training.append((writer_trace, "writer", t))
                env.advance_after_write()
            else:
                writer_trace.drop("Terminal memory has no future-task credit")
        midpoint = len(rewards) // 2
        early = sum(rewards[:midpoint]) / midpoint
        late = sum(rewards[midpoint:]) / (len(rewards) - midpoint)
        actor_trace.record_metric("rime/late_minus_early_raw_reward", late - early)
        actor_trace.record_metric("rime/stream_raw_reward", sum(rewards))
        for trace, role, position in training:
            trace.info["rime"] = {
                "episode_index": task.data.episode_index,
                "role": role,
                "position": position,
                "task_rewards": rewards,
                "exploration_rewards": scores,
                "format_gates": gates,
            }
