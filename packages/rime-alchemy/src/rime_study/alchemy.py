"""A visible-state-only adapter around pinned Symbolic Alchemy.

This defines a reproduction harness, not the authors' unreleased prompt format.
Trial transitions are explicit: the last observation of a task never contains
the next task, so a writer can run before the next actor is given its input.
"""

from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import utils

from rime_study.data import prebuilt_episodes


class AlchemyStream:
    def __init__(self, episode_index: int, max_steps_per_trial: int = 20):
        chemistry, items = prebuilt_episodes()[episode_index]
        self.num_trials = items.num_trials
        self.max_steps = max_steps_per_trial
        self._env = symbolic_alchemy.get_symbolic_alchemy_fixed(
            items,
            chemistry,
            end_trial_action=True,
            max_steps_per_trial=max_steps_per_trial,
        )
        self._env.reset()
        self.trial_index = 0
        self.steps = 0
        self.trial_reward = 0.0
        self.awaiting_write = False
        self.done = False

    def _visible_state(self) -> dict:
        stones, potions = [], []
        for stone in self._env.game_state.existing_stones():
            visible = self._env.perceived_stone(stone)
            slot = self._env.game_state.get_stone_ind(stone_inst=stone.idx)
            stones.append(
                {"id": int(slot), "properties": list(map(int, visible.perceived_coords)), "value": int(visible.reward)}
            )
        for potion in self._env.game_state.existing_potions():
            slot = self._env.game_state.get_potion_ind(potion_inst=potion.idx)
            visible = self._env.perceived_potion(potion)
            potions.append({"id": int(slot), "color": int(visible.index())})
        return {
            "trial": self.trial_index + 1,
            "steps_remaining": self.max_steps - self.steps,
            "stones": sorted(stones, key=lambda x: x["id"]),
            "potions": sorted(potions, key=lambda x: x["id"]),
        }

    def observation(self) -> dict:
        if self.awaiting_write or self.done:
            raise RuntimeError("No actor observation between trials; write memory first")
        return self._visible_state()

    def step(self, action: dict) -> dict:
        if self.awaiting_write or self.done:
            raise RuntimeError("Trial is closed; finish the memory update before acting")
        if not isinstance(action, dict):
            raise ValueError("Action must be a JSON object")
        visible = self.observation()
        op = action.get("op")
        if op in {"potion", "cauldron"}:
            if type(action.get("stone")) is not int or action["stone"] not in {x["id"] for x in visible["stones"]}:
                raise ValueError("Stone must be an existing visible slot")
            if op == "potion" and (
                type(action.get("potion")) is not int or action["potion"] not in {x["id"] for x in visible["potions"]}
            ):
                raise ValueError("Potion must be an existing visible slot")
            upstream = utils.SlotBasedAction(
                stone_ind=action["stone"],
                cauldron=op == "cauldron",
                potion_ind=action.get("potion") if op == "potion" else None,
            )
        elif op == "end_trial":
            upstream = utils.SlotBasedAction(end_trial=True)
        else:
            raise ValueError("Allowed operations: potion, cauldron, end_trial")
        timestep = self._env.step_slot_based_action(upstream)
        reward = float(timestep.reward or 0.0)
        self.trial_reward += reward
        self.steps += 1
        self.done = bool(timestep.last())
        self.awaiting_write = self.done or self._env.is_new_trial()
        result = {
            "reward": reward,
            "trial_reward": self.trial_reward,
            "trial_done": self.awaiting_write,
            "stream_done": self.done,
        }
        # Upstream advances eagerly. Hide the next task until WRITE has ended.
        if not self.awaiting_write:
            result["observation"] = self.observation()
        return result

    def advance_after_write(self) -> dict:
        if not self.awaiting_write or self.done:
            raise RuntimeError("Only a nonterminal closed trial can advance")
        self.trial_index += 1
        if self._env.trial_number != self.trial_index:
            raise RuntimeError("Upstream trial transition is inconsistent")
        self.steps = 0
        self.trial_reward = 0.0
        self.awaiting_write = False
        return self.observation()
