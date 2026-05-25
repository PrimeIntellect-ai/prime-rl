"""System prompt loading and Turn 0 message construction.

All prompts are loaded verbatim from articraft's generated files — no edits,
no translation, no simplification.  This ensures RL-trained models transfer
cleanly back to articraft inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Re-use articraft's own helpers for Turn 0 message construction.
from agent.tools import build_first_turn_messages

PROVIDER_FOR_SYSTEM_PROMPT = "openrouter"
_SYSTEM_PROMPT_TEMPLATE = (
    "agent/prompts/generated/designer_system_prompt_{provider}.txt"
)


def load_system_prompt(articraft_root: Path, *, provider: str | None = None) -> str:
    """Load the generated system prompt for *provider* verbatim."""
    provider = provider or PROVIDER_FOR_SYSTEM_PROMPT
    path = articraft_root / _SYSTEM_PROMPT_TEMPLATE.format(provider=provider)
    if not path.is_file():
        raise FileNotFoundError(
            f"System prompt not found: {path}  "
            f"(available: {list((articraft_root / 'agent/prompts/generated').glob('*.txt'))})"
        )
    return path.read_text(encoding="utf-8")


def load_scaffold_text(articraft_root: Path) -> str:
    """Load articraft's default scaffold.py content."""
    path = articraft_root / "scaffold.py"
    if not path.is_file():
        raise FileNotFoundError(f"scaffold.py not found: {path}")
    return path.read_text(encoding="utf-8")


def build_turn0_messages(
    prompt_text: str,
    *,
    sdk_docs_context: str,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Build the Turn 0 user messages (SDK docs + runtime guidance + task prompt).

    Directly reuses ``agent.tools.build_first_turn_messages`` so the message
    structure is identical to articraft inference.
    """
    provider = provider or PROVIDER_FOR_SYSTEM_PROMPT
    return build_first_turn_messages(
        prompt_text,
        sdk_docs_context=sdk_docs_context,
        provider=provider,
    )
