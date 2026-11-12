from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_config import BaseConfig


class SpeculatorConfig(BaseConfig):
    name: str
    """Speculators checkpoint to train alongside the policy."""

    revision: str | None = None

    freeze_backbone: bool = False
    """Train only the draft while keeping the target policy fixed."""

    loss_weight: float = Field(1.0, gt=0)

    lr: float | None = Field(None, gt=0)
    """Draft learning rate. None shares the policy optimizer's learning rate and schedule."""

    attn: Literal["sdpa", "eager", "simple_flex_attention"] = "sdpa"

    gradient_checkpointing: bool = False

    training: dict[str, Any] = {}
    """Flat upstream loss arguments, e.g. loss_fn, max_anchors, and confidence_head_alpha."""


class SpecDecodeConfig(BaseConfig):
    """Standalone Speculators training with prime-rl's TOML configuration loader."""

    train: dict[str, Any]
    """Upstream Speculators training groups, validated by its configuration resolver."""

    output_dir: Path = Path("outputs/specdecode")
    """Resolved configuration and upstream checkpoints."""

    num_gpus: int = Field(1, ge=1)
    """Local training processes. Run inside a compute allocation."""

    dry_run: bool = False
    """Validate and write the resolved configuration without loading models."""
