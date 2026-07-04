"""Track which policy weight version sampled each output token.

Async RL keeps requests running across in-flight weight updates (``/pause`` →
``/update_weights`` → ``/resume``), so one completion can span several policy
versions. The trainer masks tokens that are too off-policy, which requires
knowing, per token, the version that sampled it. That boundary is only
observable inside the engine core: the scheduler knows every live request's
output length at the moment the weights change.

Mechanics:
- ``Scheduler.add_request`` records a base segment ``[current_version, 0]``.
- ``EngineCore.set_prime_weight_version(version)`` — invoked by the server's
  ``/update_weights`` handler via the utility RPC (fans out to every engine
  core, including DP) while the engine is paused — appends
  ``[version, num_output_tokens]`` to every live request.
- ``Scheduler._free_request`` pops the segments and merges them into the
  finish ``kv_transfer_params`` dict under ``"prime_weight_versions"``, which
  vLLM already threads through ``EngineCoreOutput`` → ``RequestOutput`` →
  the ``/inference/v1/generate`` response. ``PrimeRlServingTokens`` strips the
  key back out of ``kv_transfer_params`` and surfaces it as the choice's
  ``weight_versions``.

A segment list ``[[v0, 0], [v1, n1], ...]`` means output tokens ``[0, n1)``
were sampled by version ``v0``, ``[n1, n2)`` by ``v1``, and so on. Versions
are trainer checkpoint steps; ``0`` means "the weights the server started
with" (the orchestrator resolves that to its resume step).
"""

from vllm.logger import init_logger

PRIME_WEIGHT_VERSIONS_KEY = "prime_weight_versions"


def monkey_patch_prime_weight_version_tracking():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore

    logger = init_logger(__name__)

    if getattr(Scheduler, "_prime_tracks_weight_versions", False):
        return

    _original_add_request = Scheduler.add_request
    _original_free_request = Scheduler._free_request

    def _segments(scheduler) -> dict:
        segments = getattr(scheduler, "_prime_weight_segments", None)
        if segments is None:
            segments = scheduler._prime_weight_segments = {}
        return segments

    def add_request(self, request):
        _original_add_request(self, request)
        # Streaming-input sessions re-enter add_request under an existing id;
        # only start tracking on first admission.
        if request.request_id in self.requests:
            _segments(self).setdefault(
                request.request_id, [[getattr(self, "_prime_weight_version", 0), 0]]
            )

    def _free_request(self, request, delay_free_blocks: bool = False):
        segments = _segments(self).pop(request.request_id, None)
        kv_xfer_params = _original_free_request(self, request, delay_free_blocks)
        if segments is None:
            return kv_xfer_params
        kv_xfer_params = kv_xfer_params if kv_xfer_params is not None else {}
        kv_xfer_params[PRIME_WEIGHT_VERSIONS_KEY] = segments
        return kv_xfer_params

    def prime_set_weight_version(self, version: int) -> None:
        self._prime_weight_version = version
        for request_id, request in self.requests.items():
            segments = _segments(self).get(request_id)
            if segments is None:
                continue
            offset = request.num_output_tokens
            if segments[-1][1] == offset:
                # No tokens sampled under the previous segment; supersede it.
                segments[-1][0] = version
            else:
                segments.append([version, offset])

    Scheduler.add_request = add_request
    Scheduler._free_request = _free_request
    Scheduler.prime_set_weight_version = prime_set_weight_version
    Scheduler._prime_tracks_weight_versions = True

    def set_prime_weight_version(self, version: int) -> None:
        """Utility RPC target (see the server's ``/update_weights`` handler)."""
        self.scheduler.prime_set_weight_version(int(version))

    EngineCore.set_prime_weight_version = set_prime_weight_version
    logger.info("Installed prime weight-version tracking (per-request output-token segments).")


def monkey_patch_strip_weight_versions_from_chat():
    """Drop the weight-version segments from chat-completions responses.

    Weight versions are only consumed via the ``/inference/v1/generate``
    (serving_tokens) path, which pops the key out of ``kv_transfer_params``.
    The chat path (evals) would otherwise forward it inside the response's
    ``kv_transfer_params``, where the PD router expects only NIXL handshake
    keys — same failure mode as routed_experts on the chat path (see
    ``monkey_patch_strip_routed_experts_from_chat``)."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    logger = init_logger(__name__)

    if getattr(OpenAIServingChat.chat_completion_full_generator, "_prime_rl_strips_weight_versions", False):
        return

    _original = OpenAIServingChat.chat_completion_full_generator

    async def _strip(result_generator):
        async for res in result_generator:
            if res.kv_transfer_params is not None:
                res.kv_transfer_params.pop(PRIME_WEIGHT_VERSIONS_KEY, None)
                if not res.kv_transfer_params:
                    res.kv_transfer_params = None
            yield res

    async def _patched(self, request, result_generator, *args, **kwargs):
        return await _original(self, request, _strip(result_generator), *args, **kwargs)

    _patched._prime_rl_strips_weight_versions = True
    OpenAIServingChat.chat_completion_full_generator = _patched
    logger.info("Stripped prime weight-version segments from chat-completions responses.")
