from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.utils.pydantic_config import BaseConfig

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2", "flash_attention_3"]

MOE_MODEL_MAPS = {
    "Qwen/Qwen3-30B-A3B": "Jackmin108/Qwen3-30B-A3B-Fast",
    "moonshotai/Moonlight-16B-A3B-Instruct": "Jackmin108/Moonlight-16B-A3B-Instruct-Fast",
}


class ActivationCheckpointConfig(BaseConfig):
    """Configures activation checkpointing."""

    freq: Annotated[
        int,
        Field(
            ge=1,
            description="Applies activation checkpointing to every `freq` layers. Defaults to 1, which will is full activation checkpointing.",
        ),
    ] = 1


class ActivationOffloadingConfig(BaseConfig):
    """Configures the activation offloading."""

    pin_memory: Annotated[bool, Field(description="Whether to pin the offloaded activations to CPU memory.")] = True

    max_inflight_activations: Annotated[
        int,
        Field(
            ge=1,
            description="The maximum number of activations to keep in while offloading further. (More activations means smoother overlap, but more gpu memory usage)",
        ),
    ] = 5


class CompileConfig(BaseConfig):
    """Configures model compilation."""

    fullgraph: Annotated[
        bool,
        Field(description="Whether to compile the transformer blocks with fullgraph."),
    ] = False


class BenchConfig(BaseConfig):
    """Configures benchmark mode."""

    output_json: Annotated[
        Path | None,
        Field(description="Path to write benchmark results as JSON. If not set, only prints to console."),
    ] = None


class DebugModelConfig(BaseConfig):
    """Debugging feature around model and distributed training."""

    num_layers: Annotated[
        int | None,
        Field(description="The number of layers in the model."),
    ] = None

    random_init: Annotated[
        bool,
        Field(
            description="Whether to random initialize the model.",
        ),
    ] = False


class LoRAConfig(BaseConfig):
    """Configuration for LoRA (Low-Rank Adaptation)."""

    rank: Annotated[
        int,
        Field(
            ge=1,
            description="Rank of the low-rank decomposition matrices.",
        ),
    ] = 16

    alpha: Annotated[
        float,
        Field(
            ge=0,
            description="LoRA scaling parameter.",
        ),
    ] = 32.0

    dropout: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="LoRA dropout rate.",
        ),
    ] = 0.0

    target_modules: Annotated[
        list[str],
        Field(
            description="Module names or regex patterns for modules to apply LoRA to. Simple names (e.g., 'q_proj') match any component in the module path. Regex patterns match anywhere in the name.",
        ),
    ] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "experts",
    ]

    modules_to_save: Annotated[
        list[str],
        Field(
            description="Module names or regex patterns for modules to keep fully trainable (not freeze). Simple names match any component in the module path. Regex patterns match anywhere in the name.",
        ),
    ] = []


class ModelConfig(BaseConfig):
    """Configures the model for training."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

    seq_len: Annotated[int, Field(description="The sequence length to use for the model.")] = 2048

    attn: Annotated[AttnImplementation, Field(description="The attention implementation to use.")] = "flash_attention_2"

    compile: Annotated[
        CompileConfig | None,
        Field(
            description="Whether to compile the model using `torch.compile`.",
        ),
    ] = None

    ac: Annotated[
        ActivationCheckpointConfig | None,
        Field(
            description="Whether to apply activation checkpointing to the model. If None, will not apply activation checkpointing.",
        ),
    ] = None

    ac_offloading: Annotated[
        ActivationOffloadingConfig | None,
        Field(
            description="Whether to apply activation offloading to the model. If None, will not apply activation offloading.",
        ),
    ] = None

    fsdp_cpu_offload: Annotated[
        bool,
        Field(
            description="Whether to enable FSDP CPU offloading for parameters, gradients, and optimizer states. When enabled, uses pinned memory for efficient CPU-GPU transfers.",
        ),
    ] = False

    optim_cpu_offload: Annotated[
        bool,
        Field(
            description="Whether to enable optimizer state CPU offloading. Unlike fsdp_cpu_offload, this only moves optimizer states (momentum, variance) to CPU, keeping weights on GPU. This avoids the H2D all-gather overhead while still saving GPU memory.",
        ),
    ] = False

    reshard_after_forward: Annotated[
        bool, Field(description="Whether to reshard the model after each forward pass.")
    ] = True

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for model and tokenizer initialization.",
        ),
    ] = False

    dp_replicate: Annotated[
        int,
        Field(
            description="The data parallel dim where model weights are replicated.",
        ),
    ] = 1

    ep: Annotated[
        int,
        Field(
            description="The expert parallelism to use if the model has MoE layers. If 1, then no EP will be used.",
        ),
    ] = 1

    tp: Annotated[
        int,
        Field(
            description="The tensor parallelism size to use. If 1, then no TP will be used.",
        ),
    ] = 1

    cp: Annotated[
        int,
        Field(
            description="The context parallelism size to use. If 1, then no CP will be used.",
        ),
    ] = 1

    impl: Annotated[
        Literal["hf", "custom", "auto"],
        Field(
            description=(
                "Model implementation to use. 'auto' (default) selects 'custom' if supported by the model, "
                "otherwise 'hf'."
            ),
        ),
    ] = "auto"

    optimization_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model optimization.",
        ),
    ] = "float32"

    reduce_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model reduce.",
        ),
    ] = "float32"

    moe_use_grouped_mm: Annotated[
        bool,
        Field(
            description="Whether to use grouped mm for the MoE layers. Require compute capability >= 9.0",
        ),
    ] = True

    lora: Annotated[
        LoRAConfig | None,
        Field(
            description="Whether to apply LoRA to the model. If None, will not apply LoRA.",
        ),
    ] = None

    debug: Annotated[
        DebugModelConfig,
        Field(
            description="Debugging feature around model and distributed training.",
        ),
    ] = DebugModelConfig()

    fused_lm_head_chunk_size: Annotated[
        int | Literal["auto", "disabled"],
        Field(
            description=(
                "The chunk size to use for the fused LM head. "
                "Three behaviors: "
                "(1) int >= 512: explicitly set chunk size for fused LM head; "
                "(2) 'auto': auto-enable (RL training auto-sets to 2048); "
                "(3) 'disabled': explicitly disable fused LM head (use vanilla). "
                "Explicitly setting an integer value for this feature isn't supported for SFT training."
            ),
        ),
    ] = "auto"

    @model_validator(mode="after")
    def _map_model_name_for_moe(self):
        """Map model name if it exists in MOE_MODEL_MAPS."""
        if self.name in MOE_MODEL_MAPS:
            self.name = MOE_MODEL_MAPS[self.name]
        return self

    @model_validator(mode="after")
    def trust_remote_code_only_with_hf(self):
        """Trust remote code only if the model is from HF."""
        if self.trust_remote_code:
            if self.impl not in ("hf", "auto"):
                raise ValueError("Trust remote code is only supported with the HF implementation or auto mode.")
        return self

    @model_validator(mode="after")
    def cp_only_with_flash_attn(self):
        if self.cp > 1 and self.attn not in ["flash_attention_2", "flash_attention_3"]:
            raise ValueError("CP is only supported with flash attention 2 or flash attention 3")
        return self

    @model_validator(mode="after")
    def ac_offloading_requires_ac(self):
        """Automatically enable activation checkpointing when activation offloading is enabled."""
        if self.ac_offloading is not None and self.ac is None:
            self.ac = ActivationCheckpointConfig()
        return self

    @model_validator(mode="after")
    def cpu_offload_mutual_exclusion(self):
        if self.fsdp_cpu_offload and self.optim_cpu_offload:
            raise ValueError("Cannot enable both fsdp_cpu_offload and optim_cpu_offload. Use one or the other.")
        return self

    @model_validator(mode="after")
    def fused_lm_head_chunk_size_is_valid(self):
        if isinstance(self.fused_lm_head_chunk_size, int):
            low = 512
            if self.fused_lm_head_chunk_size < low:
                raise ValueError(
                    f"Fused LM head chunk size must be at least {low}, got {self.fused_lm_head_chunk_size}"
                )
        return self


class TokenizerConfig(BaseConfig):
    """Configuration for the tokenizer."""

    name: Annotated[
        str | None,
        Field(description="The name or path of the tokenizer to use. If None, will use the model's default tokenizer."),
    ] = None

    trust_remote_code: Annotated[
        bool | None,
        Field(
            description="Whether to trust remote code for tokenizer initialization. If None, will use the model's default trust remote code setting.",
        ),
    ] = None

    chat_template: Annotated[
        str | None,
        Field(
            description="The chat template to use for the tokenizer. If None, will use the tokenizer's default chat template."
        ),
    ] = None


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseModel):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear"] = "linear"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = (
        10
    )

    decay_steps: Annotated[
        int,
        Field(
            ge=0,
            description="Number of steps to decay the learning rate during the final portion of training.",
        ),
    ] = 10

    min_lr: Annotated[float, Field(ge=0, description="Minimum learning rate to converge to.")] = 0.0


class CosineSchedulerConfig(BaseModel):
    """Configuration for cosine learning rate scheduler."""

    type: Literal["cosine"] = "cosine"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = (
        10
    )

    min_lr: Annotated[float, Field(ge=0, description="Minimum learning rate to converge to.")] = 0.0


SchedulerConfigType: TypeAlias = ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig


class BaseOptimizerConfig(BaseModel):
    lr: Annotated[float, Field(ge=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    max_norm: Annotated[float, Field(ge=0, description="Maximum gradient norm to clip.")] = 1.0


class SGDConfig(BaseOptimizerConfig):
    type: Literal["sgd"] = "sgd"
    nesterov: bool = True
    momentum: float = 0.9


class AdamWConfig(BaseOptimizerConfig):
    type: Literal["adamw"] = "adamw"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


class MuonConfig(BaseOptimizerConfig):
    type: Literal["muon"] = "muon"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


OptimizerConfigType: TypeAlias = SGDConfig | AdamWConfig | MuonConfig


class WeightCheckpointConfig(BaseConfig):
    """Configures saving HF-compatible weight checkpoints."""

    save_sharded: Annotated[
        bool,
        Field(
            description="Whether to save the weight checkpoint in sharded format.",
        ),
    ] = True

    save_format: Annotated[
        Literal["safetensors", "torch"],
        Field(
            description="The format to save the weight checkpoint in.",
        ),
    ] = "safetensors"

    save_adapter_separately: Annotated[
        bool,
        Field(
            description="Whether to save LoRA adapters separately before merging into full model weights.",
        ),
    ] = False


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the training checkpoint. If None, will only checkpoint at the end of training.",
        ),
    ] = None

    weights: WeightCheckpointConfig | None = WeightCheckpointConfig()

    resume_step: Annotated[
        int | None,
        Field(
            ge=-1,
            description="Step to resume training from. If None, will start from scratch. If -1, will restart from latest checkpoint available.",
        ),
    ] = None

    keep_last: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency.",
        ),
    ] = None

    keep_interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep checkpoints at every N steps permanently (e.g., keep_interval=100 keeps step 100, 200, ...). If None, no interval-based keeping.",
        ),
    ] = None

    skip_progress: Annotated[
        bool,
        Field(
            description="Whether to skip loading the progress from checkpoint.",
        ),
    ] = False

    skip_scheduler: Annotated[
        bool,
        Field(
            description="Whether to skip loading the scheduler from checkpoint.",
        ),
    ] = False

    skip_dataloader: Annotated[
        bool,
        Field(
            description="Whether to skip loading the dataloader from checkpoint.",
        ),
    ] = False
