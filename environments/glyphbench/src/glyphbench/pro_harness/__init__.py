"""Standalone GlyphBench Pro harness for long-horizon CraftaxFull / NetHack.

A self-contained agent harness — independent of the verifiers/``prime eval``
path — that drives a GlyphBench env directly and makes its own LLM calls. It
ports the strong features of the original hand-built Craftax harness into a
glyphbench-native module:

  * Pluggable inference backend (vLLM/OpenAI-compatible OR Azure Responses
    API direct) — swap with one config field.
  * Pluggable game (CraftaxFull / NetHack) via env adapters — swap with the
    ``task_id``.
  * Two calls per environment step: an ACTION turn (reason + commit one
    action) and a MEMORY turn (update the scratchpad/plan and the spatial
    landmark map after seeing the reward + next observation).
  * Persistent scratchpad + queryable spatial landmark memory.
  * Per-turn recent-action history, current-area strategic focus, and a
    feedback channel so the model recovers from parse failures.
  * Robust to Azure content-filter refusals (benign-reworded retry) and
    transient errors (backoff); never truncates the reasoning budget.

Typical use is via the CLI: ``python -m glyphbench.pro_harness --help``.
Programmatic use::

    from glyphbench.pro_harness import ProConfig, ProHarness
    cfg = ProConfig(task_id="glyphbench/craftaxfull-v0", backend="openai",
                    base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-4B")
    ProHarness(cfg).run()
"""

from glyphbench.pro_harness.adapters import EnvAdapter, get_adapter
from glyphbench.pro_harness.clients import (
    AzureResponsesClient,
    CompletionResult,
    LLMClient,
    OpenAIChatClient,
    build_client,
)
from glyphbench.pro_harness.config import SUPPORTED_TASK_IDS, ProConfig
from glyphbench.pro_harness.harness import EpisodeResult, ProHarness
from glyphbench.pro_harness.memory import Scratchpad, SpatialMemory, apply_memory
from glyphbench.pro_harness.parser import parse_action

__all__ = [
    "ProConfig",
    "ProHarness",
    "EpisodeResult",
    "SUPPORTED_TASK_IDS",
    "LLMClient",
    "OpenAIChatClient",
    "AzureResponsesClient",
    "CompletionResult",
    "build_client",
    "EnvAdapter",
    "get_adapter",
    "Scratchpad",
    "SpatialMemory",
    "apply_memory",
    "parse_action",
]
