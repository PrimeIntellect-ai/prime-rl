from pathlib import Path
from typing import Literal

from pydantic import Field

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig


class TTTOptimizerConfig(BaseConfig):
    type: Literal["adamw", "sgd"] = "adamw"
    """Optimizer for the per-rollout adapter updates. State is kept per rollout across its
    updates and dropped on release."""

    lr: float = Field(1e-4, gt=0)
    """Learning rate."""

    betas: tuple[float, float] = (0.9, 0.999)
    """AdamW betas (ignored for sgd)."""

    weight_decay: float = 0.0
    """Weight decay."""

    max_norm: float | None = 1.0
    """Gradient-norm clip (None = no clipping)."""


class TTTLoRAConfig(BaseConfig):
    rank: int = Field(16, ge=1)
    """Rank of the per-rollout adapters. The inference config's ``max_lora_rank`` must
    cover it."""

    alpha: float = Field(32.0, ge=0)
    """LoRA scaling parameter."""

    dropout: float = Field(0.0, ge=0, le=1)
    """LoRA dropout (0 is the sane default for single-sequence memorization updates)."""

    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    """Module names to adapt (PEFT semantics). Must be modules vLLM can serve LoRA for."""


class TTTModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    """Base model the adapters train against — must match the model the inference server
    serves (updates consume the engine's exact token ids)."""

    device: str = "cuda"
    """Device for the base model (one copy; adapters are swapped onto it per update)."""

    gradient_checkpointing: bool = True
    """Trade compute for memory on the long single-sequence updates."""


class TTTServiceConfig(BaseConfig):
    """The TTT service: a fourth process type (alongside inference / orchestrator /
    trainer) that owns test-time training. It holds one copy of the base model, trains one
    LoRA adapter per rollout on demand (`POST /update` with exact token ids + loss mask),
    saves a PEFT-format checkpoint per version, and (re)loads the adapter into the vLLM
    deployment. See `verifiers.v1.ttt` for the rollout side."""

    model: TTTModelConfig = TTTModelConfig()

    lora: TTTLoRAConfig = TTTLoRAConfig()

    optim: TTTOptimizerConfig = TTTOptimizerConfig()

    steps_per_update: int = Field(1, ge=1)
    """Gradient steps per `/update` call on the same payload (Cartridges-style multi-step
    is `> 1`)."""

    host: str = "0.0.0.0"
    """Host to bind to."""

    port: int = 8092
    """Port to bind to."""

    inference_admin_urls: list[str] = ["http://localhost:8000"]
    """Admin URLs of every vLLM server in the deployment (root, without ``/v1``); the
    service posts ``/load_lora_adapter`` / ``/v1/unload_lora_adapter`` to each after every
    update. Empty list = train + checkpoint only (no engine loads) — useful for tests."""

    output_dir: Path = Path("outputs")
    """Run output directory; adapter checkpoints land under ``<output_dir>/ttt/<rollout_id>/v<k>/``."""

    max_concurrent_updates: int = Field(1, ge=1)
    """Distinct rollouts allowed to run an update concurrently (same-rollout updates are
    always serialized). Bound by GPU memory for concurrent long-sequence backwards."""

    keep_checkpoints: bool = True
    """Keep every adapter version on disk after the rollout releases (the RL replay
    artifacts). False deletes the rollout's checkpoint dir on `/release` (eval-only runs
    that will never replay)."""

    log: LogConfig = LogConfig()
