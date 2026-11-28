"""Verifiers-native stream environment for the RIME prime-rl algorithm.

Each episode is ONE sibling's complete ten-trial stream. prime-rl groups episodes
for the same source task, then RIMEAlgorithm assigns cross-sibling advantages.
"""

import hashlib
import json
from dataclasses import replace

import verifiers.v1 as vf
from rime_alchemy.configs import RIMEAlchemyEnvConfig, RIMEAlchemyTasksetConfig, StreamData, StreamTask
from verifiers.v1.configs.client import EvalClientConfig, TrainClientConfig
from verifiers.v1.clients.client import ModelRoute

from rime_study.alchemy import AlchemyStream
from rime_study.data import ROOT, chemistry_fingerprint, prebuilt_episodes, training_streams, verify_sources
from rime_study.decision_judge import MemoryDecisionJudge
from rime_study.legal_action_schema import action_schema
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
    def _check_role_contract(self, agents):
        if self.config.action_decoding != "visible_legal_schema":
            return
        if not isinstance(agents.actor.ctx.client, EvalClientConfig) or agents.actor.trainable:
            raise ValueError("Legal ACT must use an immutable evaluation client")
        writer_client = EvalClientConfig if self.config.eval_untrainable else TrainClientConfig
        if not isinstance(agents.writer.ctx.client, writer_client):
            raise ValueError("WRITE client does not match the training/evaluation contract")
        expected = not self.config.eval_untrainable and not self.config.writer_frozen
        if agents.writer.trainable != expected:
            raise ValueError("WRITE trainability differs from configured role")
        if (agents.writer.ctx.sampling.model_extra or {}).get("response_format") is not None:
            raise ValueError("Writer must remain unconstrained")

    async def setup(self, agents: vf.Agents):
        agents.actor.trainable = not self.config.actor_frozen and not self.config.eval_untrainable
        agents.writer.trainable = (
            not self.config.writer_frozen and not self.config.eval_untrainable
            and self.config.memory_mode == "written"
        )
        policy_client = agents.writer.config.client
        if (self.config.actor_frozen or self.config.writer_frozen) and policy_client is None:
            raise ValueError("Frozen roles need the resolved policy client")
        for role, frozen in (("actor", self.config.actor_frozen), ("writer", self.config.writer_frozen)):
            if not frozen:
                continue
            agent = getattr(agents, role)
            client = EvalClientConfig(
                base_url=policy_client.base_url.rstrip("/") + "/rime/writer",
                api_key_var=policy_client.api_key_var,
                headers=policy_client.headers,
                headers_from_env=policy_client.headers_from_env,
            )
            sampling = agent.config.sampling.model_copy(
                update={"chat_template_kwargs": {"enable_thinking": True}},
            )
            agent.config = agent.config.model_copy(update={"client": client, "sampling": sampling})
            agent.ctx = replace(agent.ctx, client=client, sampling=sampling)
        self._check_role_contract(agents)
        self.memory_judge = None
        if not self.config.score_exploration:
            return
        judge_config = self.config.exploration_judge
        if self.config.judge_local and self.config.score_exploration:
            client = agents.actor.config.client
            if client is None:
                raise ValueError("Local Clef needs the resolved actor inference client")
            judge_config = judge_config.model_copy(
                update={
                    "base_url": client.base_url.rstrip("/") + "/rime/judge",
                    "api_key_var": client.api_key_var,
                    "headers": client.headers,
                    "headers_from_env": client.headers_from_env,
                }
            )
        self.memory_judge = MemoryDecisionJudge(
            judge_config,
            max_concurrent=self.config.judge_max_concurrent,
            max_retries=self.config.judge_max_retries,
            retry_timeout=self.config.judge_retry_timeout,
            transport_model=agents.actor.config.model if self.config.judge_local else None,
        )

    async def run(self, task: StreamTask, agents: vf.Agents, *, _donor=False):
        self._check_role_contract(agents)
        if self.config.memory_mode == "fork":
            from rime_study.memory_forks import run_memory_forks

            return await run_memory_forks(self.config, task, agents)
        env = AlchemyStream(task.data.episode_index)
        memory = ""
        rewards, scores, gates, training = [], [], [], []
        memories = []
        donor_memories = None
        if self.config.memory_mode == "shuffled":
            episodes = prebuilt_episodes()
            fingerprint = chemistry_fingerprint(episodes[task.data.episode_index][0])
            pool = json.loads((ROOT / "data/development_splits.json").read_text())["validation_episode_indices"]
            candidates = [index for index in pool if chemistry_fingerprint(episodes[index][0]) != fingerprint]
            if not candidates:
                raise ValueError("Shuffled memory needs another held-out chemistry")
            donor_index = candidates[task.data.episode_index % len(candidates)]
            donor = RIMEAlchemyEnv(
                self.config.model_copy(
                    update={
                        "memory_mode": "written",
                        "score_exploration": False,
                        "carry_memory": True,
                    }
                )
            )
            donor_task = StreamTask(task.data.model_copy(update={"episode_index": donor_index}), task.config)
            donor_memories = await donor.run(donor_task, agents, _donor=True)

        async def turn(agent, instruction, payload):
            trace = await agent.run(TurnTask(vf.TaskData(prompt=json.dumps(payload), system_prompt=instruction)))
            if trace.has_error or trace.is_format_violation or trace.is_truncated or trace.num_turns == 0:
                raise ProtocolError("Incomplete or truncated RIME model turn")
            return trace, trace.last_reply or ""

        for t in range(env.num_trials):
            if not self.config.carry_memory:
                memory = ""
            elif donor_memories is not None:
                memory = donor_memories[t - 1] if t else ""
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
            actor_route = ModelRoute(agents.actor.ctx) if self.config.action_decoding == "visible_legal_schema" else None
            route_kwargs = {"route": actor_route} if actor_route is not None else {}
            async with agents.actor.interaction(actor_task, **route_kwargs) as interaction:
                feedback = None
                for turn_index in range(env.max_steps + 2):
                    self._check_role_contract(agents)
                    if actor_route is not None:
                        schema = action_schema(env.observation())
                        sampling = actor_route.ctx.sampling.model_copy(update={
                            "response_format": {"type": "json_schema", "json_schema": {
                                "name": "visible_action", "strict": True, "schema": schema,
                            }},
                        })
                        await actor_route.update(sampling=sampling)
                        interaction.trace.info.setdefault("rime_legal_action_schema", []).append({
                            "turn": turn_index + 1,
                            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                        })
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
            actor_trace.record_metric("rime/trial_position", t + 1)
            actor_trace.record_metric("rime/is_target_trial", int(not _donor))
            if self.config.memory_mode in {"history", "shuffled"}:
                if self.config.memory_mode == "history":
                    memory = (
                        memory
                        + "\n"
                        + json.dumps(
                            {
                                "initial_observation": initial_observation,
                                "turns": history,
                            }
                        )
                    )[-self.config.memory_chars :]
                actor_trace.record_reward("task", rewards[-1])
                actor_trace.record_metric("rime/task_raw_reward", rewards[-1])
                if t < env.num_trials - 1:
                    env.advance_after_write()
                continue
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
                if self.config.score_exploration:
                    score = await self.memory_judge.score_memories(memory, proposed, trace=writer_trace)
                memory = proposed
            memories.append(memory)
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
            if _donor:
                actor_trace.drop("Unrelated-chemistry memory donor; excluded from target scores")
                writer_trace.drop("Unrelated-chemistry memory donor; excluded from target scores")
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
        return memories
