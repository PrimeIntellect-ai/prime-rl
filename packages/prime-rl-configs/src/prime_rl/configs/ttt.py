import re
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from prime_rl.configs.shared import LogConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.utils.config import BaseConfig

_REGEX_METACHARS = frozenset(".*+?^$[]{}|()\\")


def _targets_module(pattern: str, module_name: str) -> bool:
    if any(char in pattern for char in _REGEX_METACHARS):
        return re.search(pattern, module_name) is not None
    return pattern in module_name.split(".")


class TTTOptimizerConfig(BaseConfig):
    type: Literal["adamw", "sgd"] = "adamw"
    """Optimizer for the per-rollout adapter updates. State is kept per rollout across its
    updates and dropped on release."""

    lr: float = Field(1e-4, gt=0, allow_inf_nan=False)
    """Learning rate."""

    betas: tuple[float, float] = (0.9, 0.999)
    """AdamW betas (ignored for sgd)."""

    weight_decay: float = Field(0.0, ge=0, allow_inf_nan=False)
    """Weight decay."""

    max_norm: float | None = Field(1.0, gt=0, allow_inf_nan=False)
    """Gradient-norm clip (None = no clipping)."""

    @model_validator(mode="after")
    def validate_betas(self):
        if any(not isfinite(beta) or not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("optimizer betas must be finite and in [0, 1)")
        return self


class TTTLoRAConfig(BaseConfig):
    rank: int = Field(16, ge=1)
    """Rank of the per-rollout adapters. The inference config's ``max_lora_rank`` must
    cover it."""

    alpha: float = Field(32.0, ge=0, allow_inf_nan=False)
    """LoRA scaling parameter."""

    dropout: float = Field(0.0, ge=0, le=1, allow_inf_nan=False)
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

    @model_validator(mode="after")
    def validate_target_modules(self):
        if not self.target_modules or any(not target.strip() for target in self.target_modules):
            raise ValueError("lora.target_modules must contain at least one non-blank module pattern")
        for target in self.target_modules:
            try:
                targets_lm_head = any(
                    _targets_module(target, name) for name in ("lm_head", "model.lm_head", "language_model.lm_head")
                )
                targets_grouped_experts = _targets_module(target, "model.layers.0.mlp.experts")
            except re.error as exc:
                raise ValueError(f"lora.target_modules contains invalid regex {target!r}: {exc}") from exc
            if targets_lm_head:
                raise ValueError(f"lora.target_modules pattern {target!r} targets unsupported lm_head")
            if targets_grouped_experts:
                raise ValueError(f"lora.target_modules pattern {target!r} targets unsupported grouped experts")
        return self


class TTTModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    """Base model the adapters train against — must match the model the inference server
    serves (updates consume the engine's exact token ids)."""

    device: str = "cuda"
    """Device for the base model (peft engine only; the fsdp engine places per rank)."""

    gradient_checkpointing: bool = True
    """Trade compute for memory on the long single-sequence updates (peft engine only; the
    fsdp engine configures AC via ``engine.model``)."""


class TTTFSDPModelConfig(TrainerModelConfig):
    """Trainer model settings that are safe for packed TTT updates."""

    @field_validator("fused_lm_head_token_chunk_size", mode="before")
    @classmethod
    def require_fused_lm_head_chunking(cls, value):
        if type(value) is not int or value < 1:
            raise ValueError("TTT FSDP requires fused_lm_head_token_chunk_size >= 1")
        return value

    @model_validator(mode="after")
    def validate_packed_forward(self):
        if self.vlm is not None:
            raise ValueError("TTT FSDP does not support VLM inputs")
        if self.attn not in ("flash_attention_2", "flash_attention_3", "fa4"):
            raise ValueError("TTT FSDP packed updates require flash attention")
        return self


class PeftEngineConfig(BaseConfig):
    """The lightweight engine (v1): single-device HF + PEFT, one resident adapter swapped
    in/out per update. Right for small models (≤~8B) and CPU tests; can't hold large MoEs
    and serializes updates."""

    type: Literal["peft"] = "peft"


class FSDPEngineConfig(BaseConfig):
    """The trainer-stack engine (v2): the prime-rl custom modeling stack (FSDP2 / EP / AC /
    fused LM head) with ``max_slots`` resident MultiLoRA adapter slots — one slot per
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

    max_batch_wait_seconds: float = Field(0.25, ge=0, allow_inf_nan=False)
    """How long the server collects queued update jobs into one batch before running it.
    Small values favor latency (rollouts block per update); larger values favor packing."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Float32 matrix-multiplication precision used by the service process."""

    model: TTTFSDPModelConfig = Field(default_factory=TTTFSDPModelConfig)
    """Trainer ``ModelConfig`` overrides for the TTT model stack (e.g. ``impl``, ``attn``,
    ``cp``, ``cp_style``, ``ac``, ``fused_lm_head_token_chunk_size``,
    ``optimization_dtype``). ``name`` and ``lora`` are filled from the TTT config; pick the
    same modeling settings as the RL trainer for numerics parity."""

    def to_model_config(self, lora: "TTTLoRAConfig"):
        """Build the trainer ``ModelConfig`` for the TTT stack: user overrides + the TTT
        model name (threaded into ``model.name`` by TTTServiceConfig validation) + the
        slotted LoRA config (rank/alpha/targets from ``[lora]``)."""
        from prime_rl.configs.trainer import LoRAConfig as TrainerLoRAConfig

        data = self.model.model_dump()
        data["lora"] = TrainerLoRAConfig(
            rank=lora.rank,
            alpha=lora.alpha,
            dropout=lora.dropout,
            target_modules=lora.target_modules,
        ).model_dump()
        return TrainerModelConfig.model_validate(data)


TTTEngineConfig = PeftEngineConfig | FSDPEngineConfig


class TTTServiceConfig(BaseConfig):
    """The TTT service: a fourth process type (alongside inference / orchestrator /
    trainer) that owns test-time training. It holds one copy of the base model, trains one
    LoRA adapter per rollout on demand (`POST /update` with exact token ids + loss mask),
    saves a PEFT-format checkpoint per version, and (re)loads the adapter into the vLLM
    deployment. See `verifiers.v1.ttt` for the rollout side."""

    model: TTTModelConfig = TTTModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()
    """Tokenizer/chat template used for Q&A updates."""

    engine: TTTEngineConfig = Field(default_factory=PeftEngineConfig, discriminator="type")
    """The training engine: ``peft`` (v1 — single-device HF+PEFT, small models) or ``fsdp``
    (v2 — the prime-rl trainer stack with MultiLoRA slots, large models / high throughput;
    launch under torchrun)."""

    lora: TTTLoRAConfig = TTTLoRAConfig()

    optim: TTTOptimizerConfig = TTTOptimizerConfig()

    steps_per_update: int = Field(1, ge=1)
    """Gradient steps per `/update` call on the same payload (Cartridges-style multi-step
    is `> 1`)."""

    adapter_prefix: str = Field("ttt", min_length=1, max_length=127, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    """Rollout adapter namespace; managed envs must use the same prefix."""

    host: str = "0.0.0.0"
    """Host to bind to."""

    port: int = Field(8092, ge=1, le=65535)
    """Port to bind to."""

    inference_admin_urls: list[str] = ["http://localhost:8000"]
    """Admin URLs of every vLLM server in the deployment (root, without ``/v1``); the
    service posts ``/load_lora_adapter`` / ``/v1/unload_lora_adapter`` to each after every
    update. Empty list = train + checkpoint only (no engine loads) — useful for tests."""

    admin_timeout_seconds: float = Field(120.0, gt=0, allow_inf_nan=False)
    """Deadline for each inference adapter load/unload request."""

    output_dir: Path = Path("outputs")
    """Run output directory; adapter checkpoints land under ``<output_dir>/ttt/<rollout_id>/v<k>/``."""

    max_concurrent_updates: int = Field(1, ge=1)
    """Distinct rollouts allowed to queue an update concurrently. The PEFT trainer still
    executes one update at a time because its adapters share one mutable wrapper."""

    startup_timeout_seconds: int = Field(1800, ge=1)
    """How long a managed launcher waits for service readiness."""

    keep_checkpoints: bool = True
    """Keep every adapter version on disk after the rollout releases (the RL replay
    artifacts). False deletes the rollout's checkpoint dir on `/release` (eval-only runs
    that will never replay)."""

    log: LogConfig = LogConfig()

    @model_validator(mode="after")
    def thread_model_name_into_engine(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if isinstance(self.engine, FSDPEngineConfig):
            self.engine.model.name = self.model.name
        return self
