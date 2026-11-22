"""CLI entry point for the Pro harness.

Examples
--------
vLLM / any OpenAI-compatible server::

    python -m glyphbench.pro_harness \\
        --task glyphbench/craftaxfull-v0 \\
        --backend openai --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3.5-4B --api-key-env OPENAI_API_KEY_LOCAL \\
        --num-episodes 3 --seed 42

Azure Responses API (GPT-5.x reasoning), direct — content-filter robust::

    python -m glyphbench.pro_harness \\
        --task glyphbench/nethack-full-v0 \\
        --backend azure \\
        --azure-endpoint "$AZURE_OPENAI_RESPONSES_ENDPOINT" \\
        --azure-model gpt-5.4 --reasoning-effort high \\
        --num-episodes 1

Swapping the inference server is just ``--backend``/``--base-url``; swapping
the game is just ``--task``. Reasoning is never truncated unless you pass an
explicit ``--max-output-tokens``.
"""

from __future__ import annotations

import argparse
import os
import sys

from glyphbench.pro_harness.config import ProConfig
from glyphbench.pro_harness.harness import ProHarness


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m glyphbench.pro_harness",
        description="Standalone GlyphBench Pro harness for CraftaxFull / NetHack.",
    )
    p.add_argument(
        "--task",
        default="glyphbench/craftaxfull-v0",
        help="GlyphBench env id (craftaxfull / nethack-full).",
    )
    p.add_argument("--backend", choices=["openai", "azure"], default="openai")
    p.add_argument(
        "--harness-variant",
        choices=["normal", "pro"],
        default="pro",
        help="Canonical rolling-history harness or scratchpad/spatial-memory Pro.",
    )
    p.add_argument(
        "--observation-mode",
        choices=["text", "native_text", "pixels"],
        default="text",
        help=(
            "Primary model observation. 'text' is the GlyphBench glyph grid; "
            "'native_text' uses Craftax's upstream coordinate-list renderer; "
            "'pixels' sends the native Craftax frame through Azure Responses. "
            "The latter two are supported only for CraftaxFull Pro."
        ),
    )
    p.add_argument(
        "--image-detail",
        choices=["low", "high", "original", "auto"],
        default="original",
        help="Responses API detail for --observation-mode pixels.",
    )

    # OpenAI-compatible.
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    p.add_argument("--model", default=None, help="Model id (HF or deployment).")
    p.add_argument("--api-key-env", default=None, help="Name of the env var holding the API key.")
    p.add_argument("--api-key", default=None, help="API key literal (overrides env).")

    # Azure Responses.
    p.add_argument(
        "--azure-endpoint",
        default=None,
        help="Full Azure Responses URL (.../responses?api-version=...).",
    )
    p.add_argument("--azure-model", default=None, help="Azure deployment/model name.")
    p.add_argument("--azure-api-key-env", default="AZURE_OPENAI_API_KEY")
    p.add_argument("--azure-policy-id", default=None)
    p.add_argument(
        "--azure-auth-mode",
        choices=["auto", "bearer", "api-key"],
        default=os.environ.get("AZURE_OPENAI_AUTH_MODE", "auto"),
        help="Azure auth style. Auto detects Foundry /openai/v1 bearer auth.",
    )

    # Sampling / budget.
    # "none" = explicit no-reasoning (distinct from omitting). GPT-5.6 supports
    # none/low/medium/high/xhigh/max; minimal is intentionally excluded.
    p.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    p.add_argument(
        "--reasoning-mode",
        choices=["standard", "pro"],
        default=None,
        help="Optional GPT-5.6 execution mode, independent of effort.",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Output cap. Omit (or <=0) for NO cap (no reasoning truncation).",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument(
        "--top-k", type=int, default=None, help="top_k (vLLM via extra_body; e.g. 20 for Qwen3.x)."
    )
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set chat_template_kwargs.enable_thinking for Qwen-style vLLM models.",
    )

    # Rollout.
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument(
        "--max-concurrent-episodes",
        type=int,
        default=1,
        help="Episode workers sharing one inference server (default: sequential).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Per-episode turn cap. Omit for the env's native budget.",
    )
    p.add_argument("--forfeit-mode", choices=["noop", "freeze"], default="noop")

    # Robustness.
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--content-filter-retries", type=int, default=2)
    p.add_argument("--parse-retries", type=int, default=1)
    p.add_argument("--no-floor-focus", action="store_true")
    p.add_argument("--recent-actions-window", type=int, default=8)
    p.add_argument(
        "--memory-update-every",
        type=int,
        default=1,
        help="Pro periodic memory refresh interval; important outcomes still refresh immediately.",
    )
    p.add_argument(
        "--no-exploration-aid",
        action="store_true",
        help="Disable the observation-only visited/frontier navigation summary.",
    )

    # Output.
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--no-video", action="store_true", help="Disable native per-episode MP4 recording."
    )
    p.add_argument("--video-fps", type=int, default=8)
    p.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Optional recording cap. Omit or <=0 to record every native frame.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Flush transcript+progress to disk every N turns "
        "(0 = only at episode end). Use for long uncapped runs.",
    )
    p.add_argument("--render", action="store_true", help="Per-step terminal status.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-tag", default="")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-run-id", default=None)
    return p


def config_from_args(args: argparse.Namespace) -> ProConfig:
    max_out = args.max_output_tokens
    if max_out is not None and max_out <= 0:
        max_out = None
    video_max_frames = args.video_max_frames
    if video_max_frames is not None and video_max_frames <= 0:
        video_max_frames = None

    shared = dict(
        task_id=args.task,
        model=args.model,
        max_output_tokens=max_out,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        reasoning_mode=args.reasoning_mode,
        harness_variant=args.harness_variant,
        observation_mode=args.observation_mode,
        image_detail=args.image_detail,
        num_episodes=args.num_episodes,
        max_concurrent_episodes=args.max_concurrent_episodes,
        seed=args.seed,
        max_turns=args.max_turns,
        forfeit_mode=args.forfeit_mode,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        content_filter_retries=args.content_filter_retries,
        parse_retries=args.parse_retries,
        recent_actions_window=args.recent_actions_window,
        show_floor_focus=not args.no_floor_focus,
        memory_update_every=args.memory_update_every,
        show_exploration_aid=not args.no_exploration_aid,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        save_video=not args.no_video,
        video_fps=args.video_fps,
        video_max_frames=video_max_frames,
        checkpoint_every=args.checkpoint_every,
        render_terminal=args.render,
        verbose=not args.quiet,
        run_tag=args.run_tag,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_run_id=args.wandb_run_id,
    )

    if args.backend == "openai":
        api_key = args.api_key
        if api_key is None and args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
        if api_key is None:
            api_key = (
                os.environ.get("OPENAI_API_KEY_LOCAL")
                or os.environ.get("OPENAI_API_KEY")
                or "EMPTY"
            )
        base_url = args.base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1"
        model = args.model or os.environ.get("MODEL")
        shared["model"] = model
        return ProConfig(backend="openai", base_url=base_url, api_key=api_key, **shared)

    # Azure Responses.
    endpoint = (
        args.azure_endpoint
        or os.environ.get("AZURE_OPENAI_RESPONSES_ENDPOINT")
        or os.environ.get("AZURE_ENDPOINT")
    )
    model = (
        args.azure_model
        or args.model
        or os.environ.get("AZURE_OPENAI_MODEL")
        or os.environ.get("AZURE_MODEL_NAME")
        or "gpt-5.4"
    )
    key = os.environ.get(args.azure_api_key_env) or os.environ.get("AZURE_API_KEY")
    shared["model"] = model
    return ProConfig(
        backend="azure",
        azure_endpoint=endpoint,
        azure_api_key=key,
        azure_policy_id=args.azure_policy_id,
        azure_auth_mode=args.azure_auth_mode,
        **shared,
    )


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = _build_parser().parse_args(argv)
    cfg = config_from_args(args)

    # Validate required wiring up front with clear errors.
    if cfg.backend == "openai" and not cfg.model:
        print("error: --model is required for --backend openai", file=sys.stderr)
        return 2
    if cfg.backend == "azure":
        if not cfg.azure_endpoint:
            print(
                "error: --azure-endpoint (or AZURE_OPENAI_RESPONSES_ENDPOINT) "
                "is required for --backend azure",
                file=sys.stderr,
            )
            return 2
        if not cfg.azure_api_key:
            print(
                "error: Azure API key not found (set AZURE_OPENAI_API_KEY or --azure-api-key-env)",
                file=sys.stderr,
            )
            return 2

    harness = ProHarness(cfg)
    harness.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
