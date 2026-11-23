"""Task hooks for the length penalty ablations, plugged through `env.taskset.task.*`."""

import verifiers.v1 as vf

# The rlm harness gives the model exactly these native tools; everything else (the rlm API,
# MCP tools) is Python inside an `ipython` cell.
RLM_NATIVE_TOOLS = {"ipython", "wait"}


async def unknown_native_tool(response: vf.Response) -> bool:
    """Stop when the model calls a native tool the harness does not have, e.g. `rlm.tools.search`,
    an MCP tool name, or a misspelled `ipython`."""
    return any(call.name not in RLM_NATIVE_TOOLS for call in response.message.tool_calls or [])
