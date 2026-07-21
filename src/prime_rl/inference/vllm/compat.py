"""Narrow compatibility surface for vLLM's token-in/token-out API move.

vLLM moved these types from ``entrypoints.serve.disagg`` to
``entrypoints.scale_out.token_in_token_out`` without a compatibility alias.
Keep the version branch in one module so Prime's serving and client code do not
grow parallel implementations.
"""

try:
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
        GenerateRequest,
        GenerateResponse,
        GenerateResponseChoice,
    )
    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("vllm.entrypoints.scale_out"):
        raise
    from vllm.entrypoints.serve.disagg.protocol import (
        GenerateRequest,
        GenerateResponse,
        GenerateResponseChoice,
    )
    from vllm.entrypoints.serve.disagg.serving import ServingTokens



def serving_renderer(serving_tokens: ServingTokens):
    """Return the renderer without coupling it to the import package layout."""
    for attribute in ("online_renderer", "openai_serving_render"):
        if hasattr(serving_tokens, attribute):
            return getattr(serving_tokens, attribute)
    raise AttributeError("vLLM ServingTokens exposes no supported renderer attribute")

__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "GenerateResponseChoice",
    "ServingTokens",
    "serving_renderer",
]
