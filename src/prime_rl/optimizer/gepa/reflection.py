from __future__ import annotations

import random
from typing import Sequence

from prime_rl.optimizer.gepa.config import OperatorsConfig
from openai import AsyncOpenAI
from prime_rl.orchestrator.config import ModelConfig as OrchestratorModelConfig, ClientConfig


REFLECT_TEMPLATE = (
    "You are optimizing a system instruction for a model on a benchmark.\n"
    "Given a list of failure examples (short summaries), propose concise edits to the instruction\n"
    "that will improve accuracy without adding verbosity or vagueness.\n"
    "Return only the edited instruction text. Keep it under the specified character limit."
)


def reflect(prompt: str, failures: Sequence[str], cfg: OperatorsConfig, rng: random.Random) -> str:
    # Placeholder: lightweight heuristic reflection combining failures into a short clause
    if not failures:
        return prompt
    clause = "; ".join(failures[:3])
    edited = (
        prompt
        + "\nConstraints: Avoid previous mistakes such as: "
        + clause
        + ". Be explicit, deterministic, and adhere to required answer format."
    )
    if len(edited) > cfg.max_prompt_chars:
        edited = edited[: cfg.max_prompt_chars]
    return edited


async def reflect_llm(
    client_cfg: ClientConfig,
    model_cfg: OrchestratorModelConfig,
    current_prompt: str,
    feedbacks: list[str],
    max_chars: int,
) -> str:
    """Use the model to propose an edited instruction given feedback examples."""
    client = AsyncOpenAI(base_url=f"http://{client_cfg.host}:{client_cfg.port}/v1", api_key=client_cfg.api_key)

    system = (
        "You are an expert instruction engineer.\n"
        "Given the current system instruction and a few failure summaries, propose an improved instruction.\n"
        "Constraints: keep it concise, deterministic, preserve required tags/formatting, and stay under the character limit."
    )
    fb_text = "\n- ".join([f for f in feedbacks[:5]]) if feedbacks else "(no feedback provided)"
    user = (
        f"Current instruction:\n" 
        f"-----\n{current_prompt}\n-----\n"
        f"Failures:\n- {fb_text}\n"
        f"Return only the full edited instruction (no commentary), max {max_chars} characters."
    )
    resp = await client.chat.completions.create(
        model=model_cfg.name,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=512,
    )
    text = resp.choices[0].message.content or current_prompt
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


