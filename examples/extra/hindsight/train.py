"""A two-step goal-conditioned parity actor. Install the project and run with uv run python."""

import random

import torch
import verifiers.v1 as vf

from prime_rl.orchestrator.hindsight import FixedHorizonWorld, make_hindsight_batch
from prime_rl.trainer.rl.hindsight import hindsight_loss

ACTIONS = (10, 11)


def world_for(goal):
    def transition(state, token):
        return ((state[0] + ACTIONS.index(token)) % 2,)

    def prefixes(g, actions):
        return [(99, g[0], *actions[:i]) for i in range(len(actions))]

    return FixedHorizonWorld(
        "parity", (0,), (goal,), frozenset({(0,), (1,)}), ACTIONS, 2, "finished", transition, prefixes
    )


def policy_step(actor, prefix):
    _, goal, *previous = prefix
    parity = sum(ACTIONS.index(t) for t in previous) % 2
    return actor[goal, len(previous), parity]


@torch.no_grad()
def sample_episode(actor, w, index, rng):
    actions = []
    prompt = (99, w.requested[0])
    for _ in range(w.horizon):
        probs = policy_step(actor, (*prompt, *actions)).softmax(0)
        action = torch.multinomial(probs, 1, generator=rng).item()
        actions.append(ACTIONS[action])
    end = w.initial
    for token in actions:
        end = w.transition(end, token)
    task = vf.TraceTask(type="Parity", data=vf.TaskData(idx=index, prompt=str(w.requested)))
    root = vf.MessageNode(message=vf.UserMessage(content=str(w.requested)), token_ids=prompt, mask=[False, False])
    answer = vf.MessageNode(
        message=vf.AssistantMessage(content=str(actions)), token_ids=actions, mask=[True, True], parent=0, sampled=True
    )
    trace = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[root, answer],
        calls=[vf.ModelCall(node=1, model="parity-actor")],
        rewards={"reward": vf.Reward(score=float(end == w.requested))},
        ok=True,
        is_completed=True,
        stop_condition=w.stop_condition,
    )
    return vf.Episode(env=vf.EnvInfo(id=w.env_id), task=task, traces=[trace], ok=True)


def main():
    rng = torch.Generator().manual_seed(32)
    actor = torch.nn.Parameter(torch.zeros(2, 2, 2, 2))
    optimizer = torch.optim.SGD([actor], lr=0.1)
    goals = random.Random(11)
    worlds = [world_for(goals.randrange(2)) for _ in range(32)]
    episodes = [sample_episode(actor, w, i, rng) for i, w in enumerate(worlds)]
    metrics = {}
    batch = make_hindsight_batch(episodes, worlds, metrics=metrics)
    used = [t for t in batch if t is not None]
    logits = torch.stack([torch.stack([policy_step(actor, p) for p in t.prefixes]) for t in used])
    columns = torch.tensor([[ACTIONS.index(a) for a in t.actions] for t in used])
    # Per-decision mean. The original episode count still includes all omitted goals.
    loss = hindsight_loss(logits, columns, len(episodes)) / 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(dict(**metrics, loss=loss.item(), max_parameter_change=actor.detach().abs().max().item()))


if __name__ == "__main__":
    main()
