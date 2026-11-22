"""Example: build a custom agent loop with the GlyphBench environment interface.

This shows how to plug any LLM (or rule-based agent) into the env loop.
"""

from glyphbench.core import make_env


def my_agent(obs: str, action_names: list[str]) -> int:
    """A trivial rule-based agent that always picks the first action.

    Replace this with your LLM call:
        response = client.chat(messages=[
            {"role": "system", "content": env.system_prompt()},
            {"role": "user", "content": obs},
        ])
        action_name = parse_action(response.text)
        return action_names.index(action_name)
    """
    return 0


def main() -> None:
    env = make_env("glyphbench/minigrid-doorkey-6x6-v0", max_turns=100)
    obs, info = env.reset(seed=42)

    # The system prompt describes the game rules to an LLM agent
    print("=== System Prompt (first 300 chars) ===")
    print(env.system_prompt()[:300])
    print("...\n")

    action_names = env.action_spec.names
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        action = my_agent(obs, action_names)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

    print(f"Episode finished: steps={step}, return={total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
