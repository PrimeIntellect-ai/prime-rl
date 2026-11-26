"""Read-time memory interventions with exactly replayed visible experience."""

import hashlib
import json
import random

import verifiers.v1 as vf

from rime_study.alchemy import AlchemyStream
from rime_study.rollout import ACT, WRITE, ProtocolError


class ObservationLedger:
    def __init__(self, cap):
        self.cap = cap
        self.facts = {}

    def observe(self, before, action, result):
        if "error" in result:
            return
        stone = next((s for s in before["stones"] if s["id"] == action.get("stone")), None)
        if stone is None:
            return
        if action["op"] == "cauldron":
            key = ("reward", stone["value"])
            fact = f"Observed cauldron reward for current value {stone['value']}: {result['reward']}."
        elif action["op"] == "potion" and "observation" in result:
            potion = next(p for p in before["potions"] if p["id"] == action["potion"])
            after = next(s for s in result["observation"]["stones"] if s["id"] == stone["id"])
            key = ("potion", potion["color"], tuple(stone["properties"]))
            fact = (
                f"Observed potion color {potion['color']} on properties {stone['properties']} "
                f"(value {stone['value']}) -> {after['properties']} (value {after['value']})."
            )
        else:
            return
        self.facts.pop(key, None)
        self.facts[key] = fact

    def text(self):
        text = "Verified past visible observations; applicability beyond observed states is uncertain.\n"
        for fact in reversed(list(self.facts.values())):
            if len(text) + len(fact) + 1 <= self.cap:
                text += fact + "\n"
        return text.rstrip()[: self.cap] if self.facts else ""


def replay_prefix(index, prefix):
    """Reconstruct from fixed task inputs, checking all visible transitions."""
    env = AlchemyStream(index)
    for trial in prefix:
        if env.observation() != trial["initial_observation"]:
            raise RuntimeError("Memory-fork prefix initial state mismatch")
        for turn in trial["turns"]:
            try:
                result = env.step(json.loads(turn["actor"]))
            except (ValueError, TypeError, KeyError) as exc:
                result = {"error": str(exc), "observation": env.observation()}
            if result != turn["result"]:
                raise RuntimeError("Memory-fork prefix transition mismatch")
        env.advance_after_write()
    return env


async def run_memory_forks(config, task, agents):
    from rime_study.hosted import TurnTask

    async def actor_trial(env, memory, ledger=None):
        initial = env.observation()
        history = []
        invalid = 0
        actor_task = TurnTask(
            vf.TaskData(system_prompt=ACT, prompt=json.dumps({"memory": memory, "initial_observation": initial}))
        )
        async with agents.actor.interaction(actor_task) as interaction:
            feedback = None
            for _ in range(env.max_steps + 2):
                segment = await interaction.turn(feedback)
                if segment.terminated or interaction.trace.is_truncated:
                    raise ProtocolError("Actor terminated before finishing the trial")
                before = env.observation()
                reply = segment.last_reply
                try:
                    action = json.loads(reply)
                    result = env.step(action)
                except (ValueError, TypeError, KeyError) as exc:
                    invalid += 1
                    if invalid > 1:
                        raise ProtocolError("Actor exhausted action repair budget") from exc
                    result = {"error": str(exc), "observation": env.observation()}
                history.append({"actor": reply, "result": result})
                if ledger is not None and "error" not in result:
                    ledger.observe(before, action, result)
                if result.get("trial_done"):
                    break
                feedback = json.dumps(result)
            else:
                raise ProtocolError("Actor exceeded action budget")
        trace = interaction.trace
        if trace.has_error or trace.is_format_violation or trace.is_truncated:
            raise ProtocolError("Incomplete actor trajectory")
        trace.record_reward("task", result["trial_reward"])
        return {"initial_observation": initial, "turns": history, "reward": result["trial_reward"]}, trace

    index = task.data.episode_index
    env = AlchemyStream(index)
    if config.fork_trial + config.fork_horizon > env.num_trials:
        raise ValueError("Memory fork exceeds stream length")
    ledger = ObservationLedger(config.memory_chars)
    memory = ""
    prefix = []
    for _ in range(config.fork_trial):
        row, trace = await actor_trial(env, memory, ledger)
        prefix.append(row)
        old_memory = memory
        writer = await agents.writer.run(
            TurnTask(
                vf.TaskData(
                    system_prompt=WRITE.replace("token budget", "character budget"),
                    prompt=json.dumps(
                        {
                            "previous_memory": memory,
                            "trajectory": {"initial_observation": row["initial_observation"], "turns": row["turns"]},
                            "memory_character_budget": config.memory_chars,
                        }
                    ),
                )
            )
        )
        if writer.has_error or writer.is_truncated or writer.is_format_violation or writer.num_turns == 0:
            raise ProtocolError("Incomplete memory-fork writer turn")
        valid = False
        try:
            parsed = json.loads(writer.last_reply or "")
            valid = (
                set(parsed) == {"memory"}
                and isinstance(parsed["memory"], str)
                and len(parsed["memory"]) <= config.memory_chars
            )
            if valid:
                memory = parsed["memory"]
        except (ValueError, TypeError, KeyError):
            pass
        writer.record_metric("rime/writer_format_valid", float(valid))
        env.advance_after_write()
    memories = {"old": old_memory, "replacement": memory, "verified": ledger.text(), "empty": ""}
    expected = env.observation()
    fingerprint = hashlib.sha256(
        json.dumps({"prefix": prefix, "next_observation": expected}, sort_keys=True).encode()
    ).hexdigest()
    jobs = [(arm, repetition) for repetition in range(config.fork_repetitions) for arm in memories]
    random.Random(index).shuffle(jobs)
    outcomes = []
    for arm, repetition in jobs:
        branch = replay_prefix(index, prefix)
        if branch.observation() != expected:
            raise RuntimeError("Memory-fork continuation initial state mismatch")
        result = {"arm": arm, "repetition": repetition, "rewards": [], "has_error": False}
        try:
            for position in range(config.fork_horizon):
                row, trace = await actor_trial(branch, memories[arm])
                trace.record_metric("rime/fork_trial_position", position + 1)
                trace.info["rime_fork"] = {"arm": arm, "repetition": repetition, "prefix_sha256": fingerprint}
                result["rewards"].append(row["reward"])
                if position + 1 < config.fork_horizon:
                    branch.advance_after_write()
        except ProtocolError as exc:
            result.update(has_error=True, error=str(exc))
        result["completed_trials"] = len(result["rewards"])
        result["observed_reward"] = sum(result["rewards"])
        result["complete_mean_reward"] = (
            sum(result["rewards"]) / config.fork_horizon if not result["has_error"] else None
        )
        outcomes.append(result)
        # One summary per attempt, including failed continuations; never impute
        # unobserved rewards or let a failed arm cancel its matched siblings.
        writer.record_metric(f"rime/fork/{arm}/rep{repetition}/has_error", float(result["has_error"]))
        writer.record_metric(f"rime/fork/{arm}/rep{repetition}/completed_trials", result["completed_trials"])
        for position, reward in enumerate(result["rewards"], start=1):
            writer.record_metric(f"rime/fork/{arm}/rep{repetition}/trial{position}_reward", reward)
        if result["complete_mean_reward"] is not None:
            writer.record_metric(f"rime/fork/{arm}/rep{repetition}/mean_reward", result["complete_mean_reward"])
    writer.info["rime_forks"] = {
        "episode_index": index,
        "prefix_sha256": fingerprint,
        "prefix": prefix,
        "memories": memories,
        "replacement_format_valid": valid,
        "past_window_mean": sum(r["reward"] for r in prefix[-config.fork_horizon :]) / config.fork_horizon,
        "outcomes": outcomes,
        "horizon": config.fork_horizon,
        "note": "Fixed note over future window; stochastic continuations, no exact RNG pairing claim.",
    }
