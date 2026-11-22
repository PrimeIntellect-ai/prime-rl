"""Configuration for the standalone GlyphBench Pro harness.

``ProConfig`` is the single source of truth for a run: which game to play,
which inference server to talk to (an OpenAI-compatible endpoint *or* the
Azure Responses API directly), the sampling/budget settings, and the
robustness knobs. It is deliberately transport-agnostic so swapping
vLLM <-> Azure or Craftax <-> NetHack is a one-line change.

The harness supports:
  * Easy server swap: ``backend="openai"`` for any OpenAI-compatible server
    (vLLM, OpenAI, Prime, or the Azure chat-completions proxy);
    ``backend="azure"`` to hit the Azure Responses API directly.
  * Easy env swap: ``task_id`` selects any GlyphBench env (the harness is
    only *supported* on craftaxfull / nethack, but it will drive any
    ``BaseGlyphEnv``).
  * No reasoning-budget truncation: ``max_output_tokens=None`` means "send
    no output cap at all" so the model is never cut off mid-reasoning. The
    harness also never trims the prompt to fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from glyphbench.azure_openai import redact_endpoint_url

# The harness is intended for these two long-horizon games. We do not hard
# block other ids (it will drive any BaseGlyphEnv), but we warn.
SUPPORTED_TASK_IDS: frozenset[str] = frozenset(
    {
        "glyphbench/craftaxfull-v0",
        "glyphbench/nethack-full-v0",
    }
)


@dataclass
class ProConfig:
    """All settings for one pro-harness run."""

    # --- What to play -----------------------------------------------------
    task_id: str = "glyphbench/craftaxfull-v0"

    # --- Which server ("openai" = OpenAI-compatible chat; "azure" = Azure
    #     Responses API direct) ------------------------------------------
    backend: str = "openai"

    # OpenAI-compatible chat completions (vLLM / OpenAI / Prime / proxy).
    base_url: str | None = None
    model: str | None = None
    api_key: str = field(default="EMPTY", repr=False)

    # Azure Responses API (direct; for GPT-5.x "pro"/reasoning deployments).
    azure_endpoint: str | None = None
    azure_api_key: str | None = field(default=None, repr=False)
    azure_policy_id: str | None = None
    azure_auth_mode: str = "auto"  # auto | bearer (Foundry v1) | api-key (legacy)

    # --- Sampling / reasoning budget -------------------------------------
    # None => no output-token cap is sent (the model is never truncated
    # mid-reasoning). Set an int only if you deliberately want a cap.
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # top_k is non-standard OpenAI; forwarded to vLLM via extra_body. Needed for
    # correct Qwen3.x sampling (recommended top_k=20). presence_penalty is
    # standard. Both ignored by the Azure Responses backend.
    top_k: int | None = None
    presence_penalty: float | None = None
    # Forwarded to OpenAI-compatible chat templates (not Azure). This keeps
    # Qwen thinking/non-thinking evals explicit and reproducible.
    enable_thinking: bool | None = None
    # low | medium | high | None. Forwarded to the Responses API
    # `reasoning.effort` and the chat `reasoning_effort` param.
    reasoning_effort: str | None = None
    # GPT-5.6 Responses API execution mode. ``None`` keeps Azure's standard
    # default; ``pro`` asks the model to spend additional work on difficult
    # decisions. This is independent from reasoning_effort.
    reasoning_mode: str | None = None

    # ``normal`` is the canonical GlyphBench action protocol with linear
    # conversation aggregation. ``pro`` adds model-maintained scratchpad,
    # spatial memory, floor focus, and grounded memory-update calls.
    harness_variant: str = "pro"

    # Primary environment observation delivered to the model. ``text`` is the
    # existing GlyphBench glyph-grid mode; ``native_text`` is the upstream
    # Craftax coordinate-by-coordinate language renderer. ``pixels`` is
    # intentionally a narrowly scoped CraftaxFull/Azure/Pro research mode: the
    # native renderer frame replaces the textual grid on both action and
    # memory turns, while all other Pro scaffolding remains unchanged.
    observation_mode: str = "text"
    image_detail: str = "original"

    # --- Rollout ----------------------------------------------------------
    num_episodes: int = 1
    # Parallel episode workers share one thread-safe inference client so a
    # local vLLM server can batch turns across seeds and reuse prompt prefixes.
    max_concurrent_episodes: int = 1
    seed: int = 42
    # None => use each env's native max_turns.
    max_turns: int | None = None
    # "noop"  => on an unrecoverable parse failure, apply the env's noop
    #            action (dynamics advance, faithful to the original Craftax
    #            harness which applied action 0).
    # "freeze"=> advance only the turn counter, leave env state unchanged.
    forfeit_mode: str = "noop"

    # --- Robustness -------------------------------------------------------
    request_timeout: float = 600.0
    # Retries for transient transport errors (429 / 5xx / timeouts).
    max_retries: int = 4
    retry_backoff: float = 2.0
    # Retries that re-issue the request with a softened, benign system
    # framing after an Azure content-filter / refusal (the original's trick).
    content_filter_retries: int = 2
    # Focused low-effort re-ask when the action could not be parsed.
    parse_retries: int = 1

    # --- Memory / history rendered to the model --------------------------
    recent_actions_window: int = 8
    show_floor_focus: bool = True
    # Pro memory is always refreshed after rewards, achievements, parse
    # failures, truncations, and area changes. This interval controls the
    # additional periodic refreshes. 1 preserves the original every-turn
    # protocol; 5 is substantially more efficient for long Craftax runs.
    memory_update_every: int = 1
    # Observation-only navigation aid: remembers visible tiles, visited
    # coordinates, frontiers, and durable features without reading hidden env
    # state. This is scaffolding, not an action policy.
    show_exploration_aid: bool = True

    # --- Output / display -------------------------------------------------
    output_dir: str | None = None
    render_terminal: bool = False
    verbose: bool = True
    save_results: bool = True
    # Stream every native pixel frame to an MP4 as the episode runs. Streaming
    # avoids holding a 10,001-frame CraftaxFull episode in memory and removes
    # the old silent 4,000-frame GIF cap.
    save_video: bool = True
    video_fps: int = 8
    video_max_frames: int | None = None
    # Flush transcript + a progress snapshot to disk every N env turns
    # mid-episode, so a long uncapped run survives a kill / 24h limit. 0 = only
    # write at episode end.
    checkpoint_every: int = 0

    # Free-form tag stored in the run metadata (e.g. an experiment name).
    run_tag: str = ""

    # Optional W&B tracking. When unset, the harness remains dependency-light
    # and writes only local results. Secrets are never persisted in config.
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"openai", "azure"}:
            raise ValueError(f"backend must be 'openai' or 'azure', got {self.backend!r}")
        if self.forfeit_mode not in {"noop", "freeze"}:
            raise ValueError(f"forfeit_mode must be 'noop' or 'freeze', got {self.forfeit_mode!r}")
        if self.azure_auth_mode not in {"auto", "bearer", "api-key"}:
            raise ValueError(
                "azure_auth_mode must be one of auto/bearer/api-key, "
                f"got {self.azure_auth_mode!r}"
            )
        # Note: "none" is a DISTINCT literal effort (0 reasoning tokens),
        # different from None (omit the param → the model's default effort).
        # We keep "none" literal so it can be sent as reasoning.effort="none".
        if self.harness_variant not in {"normal", "pro"}:
            raise ValueError(
                f"harness_variant must be 'normal' or 'pro', got {self.harness_variant!r}"
            )
        if self.observation_mode not in {"text", "native_text", "pixels"}:
            raise ValueError(
                "observation_mode must be 'text', 'native_text', or 'pixels', "
                f"got {self.observation_mode!r}"
            )
        if self.image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError(
                "image_detail must be low/high/original/auto, "
                f"got {self.image_detail!r}"
            )
        if self.observation_mode == "pixels":
            if self.harness_variant != "pro":
                raise ValueError("pixel observations are supported only by the Pro harness")
            if self.task_id != "glyphbench/craftaxfull-v0":
                raise ValueError("pixel observations currently require craftaxfull-v0")
            if self.backend != "azure":
                raise ValueError("pixel observations currently require Azure Responses")
        if self.observation_mode == "native_text":
            if self.harness_variant != "pro":
                raise ValueError("native-text observations are supported only by the Pro harness")
            if self.task_id != "glyphbench/craftaxfull-v0":
                raise ValueError("native-text observations currently require craftaxfull-v0")
        if self.reasoning_mode not in {None, "standard", "pro"}:
            raise ValueError(
                f"reasoning_mode must be 'standard', 'pro', or None; got {self.reasoning_mode!r}"
            )
        _VALID_EFFORTS = {None, "minimal", "low", "medium", "high", "xhigh", "max", "none"}
        if self.reasoning_effort not in _VALID_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(e for e in _VALID_EFFORTS if e)} "
                f"or None; got {self.reasoning_effort!r}"
            )
        for name in (
            "num_episodes", "max_concurrent_episodes", "memory_update_every",
            "video_fps", "request_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_turns", "max_output_tokens", "video_max_frames"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        for name in (
            "max_retries", "content_filter_retries", "parse_retries",
            "recent_actions_window", "checkpoint_every", "retry_backoff",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def public_dict(self) -> dict[str, object]:
        """Configuration suitable for result files and tracking services."""
        config = asdict(self)
        config.pop("api_key")
        config.pop("azure_api_key")
        for key in ("base_url", "azure_endpoint"):
            config[key] = redact_endpoint_url(config[key])
        return config

    def describe(self) -> str:
        if self.backend == "azure":
            where = f"azure-responses:{redact_endpoint_url(self.azure_endpoint)}"
        else:
            where = f"openai:{redact_endpoint_url(self.base_url)}"
        cap = "uncapped" if self.max_output_tokens is None else str(self.max_output_tokens)
        return (
            f"task={self.task_id} backend={self.backend} model={self.model} "
            f"endpoint={where} max_output_tokens={cap} "
            f"harness={self.harness_variant} "
            f"observation_mode={self.observation_mode} "
            f"image_detail={self.image_detail} "
            f"reasoning_effort={self.reasoning_effort} "
            f"reasoning_mode={self.reasoning_mode or 'standard'} "
            f"episodes={self.num_episodes} "
            f"max_concurrent_episodes={self.max_concurrent_episodes}"
        )
