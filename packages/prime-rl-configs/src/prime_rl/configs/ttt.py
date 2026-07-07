from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

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
    """Device for the base model (peft engine only; the fsdp engine places per rank)."""

    gradient_checkpointing: bool = True
    """Trade compute for memory on the long single-sequence updates (peft engine only; the
    fsdp engine configures AC via ``engine.model``)."""


class PeftEngineConfig(BaseConfig):
    """The lightweight engine (v1): single-device HF + PEFT, one resident adapter swapped
    in/out per update. Right for small models (≤~8B) and CPU tests; can't hold large MoEs
    and serializes updates."""

    type: Literal["peft"] = "peft"


class FSDPEngineConfig(BaseConfig):
    """The trainer-stack engine (v2): the prime-rl custom modeling stack (FSDP2 / EP / CP /
    AC / fused LM head) with ``max_slots`` resident MultiLoRA adapter slots — one slot per
    concurrent TTT rollout, claimed on first update and freed on release. Updates for
    different rollouts pack into shared forwards (the segmented ``lora_num_tokens``
    layout), so throughput scales with tokens, not update count. Launch under torchrun
    across the TTT node(s)."""

    type: Literal["fsdp"] = "fsdp"

    max_slots: int = Field(64, ge=1)
    """Resident adapter slots = max concurrent TTT rollouts this service instance can hold.
    Rank-r attention adapters are tens of MB each; size against orchestrator
    ``max_inflight_rollouts`` (and the engines' ``max_loras``)."""

    max_tokens_per_forward: int = Field(65536, ge=1024)
    """Token budget per packed update forward (activation-memory bound, like the RL
    trainer's ``seq_len``). Jobs are packed whole into forwards under this cap."""

    max_batch_wait_seconds: float = Field(0.25, ge=0)
    """How long the server collects queued update jobs into one batch before running it.
    Small values favor latency (rollouts block per update); larger values favor packing."""

    model: dict = Field(default_factory=dict)
    """Trainer ``ModelConfig`` overrides for the TTT model stack (e.g. ``impl``, ``attn``,
    ``cp``, ``cp_style``, ``ac``, ``fused_lm_head_token_chunk_size``,
    ``optimization_dtype``). ``name`` and ``lora`` are filled from the TTT config; pick the
    same modeling settings as the RL trainer for numerics parity."""

    def to_model_config(self, lora: "TTTLoRAConfig"):
        """Build the trainer ``ModelConfig`` for the TTT stack: user overrides + the TTT
        model name + the slotted LoRA config (rank/alpha/targets from ``[lora]``)."""
        from prime_rl.configs.trainer import LoRAConfig as TrainerLoRAConfig
        from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig

        data = dict(self.model)
        if "name" not in data and self._model_name is not None:
            data["name"] = self._model_name
        data["lora"] = TrainerLoRAConfig(
            rank=lora.rank,
            alpha=lora.alpha,
            dropout=lora.dropout,
            target_modules=lora.target_modules,
        ).model_dump()
        return TrainerModelConfig.model_validate(data)

    # Set by TTTServiceConfig validation so the engine renders the right base model.
    _model_name: str | None = PrivateAttr(default=None)


TTTEngineConfig = PeftEngineConfig | FSDPEngineConfig


class TTTServiceConfig(BaseConfig):
    """The TTT service: a fourth process type (alongside inference / orchestrator /
    trainer) that owns test-time training. It holds one copy of the base model, trains one
    LoRA adapter per rollout on demand (`POST /update` with exact token ids + loss mask),
    saves a PEFT-format checkpoint per version, and (re)loads the adapter into the vLLM
    deployment. See `verifiers.v1.ttt` for the rollout side."""

    model: TTTModelConfig = TTTModelConfig()

    engine: TTTEngineConfig = Field(default_factory=PeftEngineConfig, discriminator="type")
    """The training engine: ``peft`` (v1 — single-device HF+PEFT, small models) or ``fsdp``
    (v2 — the prime-rl trainer stack with MultiLoRA slots, large models / high throughput;
    launch under torchrun)."""

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

    @model_validator(mode="after")
    def thread_model_name_into_engine(self):
        if isinstance(self.engine, FSDPEngineConfig):
            self.engine._model_name = self.model.name
        return self
