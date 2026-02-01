from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    BenchConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)
from prime_rl.utils.config import HeartbeatConfig, LogConfig, WandbConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class ValidationConfig(BaseConfig):
    """Configures online validation during SFT training."""

    enabled: Annotated[bool, Field(description="Enable online validation during training.")] = False
    interval: Annotated[int, Field(ge=1, description="Run validation every N training steps.")] = 100
    split: Annotated[str, Field(description="HuggingFace dataset split to use for validation.")] = "validation"
    max_samples: Annotated[int | None, Field(ge=1, description="Maximum number of validation samples to use.")] = None


class BaseDataConfig(BaseModel):
    """Base config for SFT data."""

    batch_size: Annotated[int, Field(ge=1)] = 128
    seq_len: Annotated[int, Field(ge=1)] = 128
    pack_function: Literal["cat", "stack", "single"] = "cat"
    micro_batch_size: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class FakeDataConfig(BaseDataConfig):
    """Configures fake data used for debugging."""

    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    input_ids: Literal["increasing", "random"] = "increasing"


class LossMaskConfig(BaseConfig):
    """Configures which message types contribute to the loss. If True, the loss_mask will be True and the message type will contribute to the loss."""

    system: Annotated[bool, Field(description="Whether system messages contribute to the loss.")] = False
    user: Annotated[bool, Field(description="Whether user messages contribute to the loss.")] = False
    assistant: Annotated[bool, Field(description="Whether assistant messages contribute to the loss.")] = True
    tool: Annotated[bool, Field(description="Whether tool messages contribute to the loss.")] = False


class SFTDataConfig(BaseDataConfig):
    """Configures the data used for training."""

    type: Literal["sft"] = "sft"

    name: Annotated[str, Field(description="Name or path of the HF dataset to use.")] = (
        "PrimeIntellect/Reverse-Text-SFT"
    )
    subsets: Annotated[list[str] | None, Field(description="Subsets to use from the HF dataset.")] = None
    splits: Annotated[list[str] | None, Field(description="Splits to use from the HF dataset.")] = None
    probabilities: Annotated[list[float] | None, Field(description="Probabilities to use for each subset/split.")] = (
        None
    )
    stopping_strategy: Annotated[
        Literal["first_exhausted", "all_exhausted"],
        Field(description=""),
    ] = "all_exhausted"
    shuffle: Annotated[bool, Field(description="Whether to shuffle the dataset at the beginning of each epoch.")] = True
    seed: Annotated[
        int,
        Field(
            description="Random seed to use for shuffling the dataset. We also shuffle at the end of each epoch by adding epoch count to the seed."
        ),
    ] = 0

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        if self.subsets is not None or self.splits is not None:
            if self.subsets is not None and self.splits is not None:
                if len(self.subsets) != len(self.splits):
                    raise ValueError(
                        "Number of subsets must be equal to number of splits. Please specify which split to load for each subset."
                    )
            if self.subsets is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.subsets):
                    raise ValueError(
                        "Number of probabilities must be equal to number of subsets. Please specify a probability for each subset."
                    )
            if self.splits is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.splits):
                    raise ValueError(
                        "Number of probabilities must be equal to number of splits. Please specify a probability for each split."
                    )
        return self


DataConfigType: TypeAlias = FakeDataConfig | SFTDataConfig


class SFTTrainerConfig(BaseSettings):
    """Configures the SFT trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The tokenizer configuration
    tokenizer: TokenizerConfig = TokenizerConfig()

    # The data configuration
    data: Annotated[DataConfigType, Field(discriminator="type")] = SFTDataConfig()

    # The validation configuration
    validation: ValidationConfig = ValidationConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(description="Maximum number of steps to run training for. If None, will run indefinitely."),
    ] = None

    memory_profiler_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    bench: Annotated[
        BenchConfig | None,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 4 and use fake data.",
        ),
    ] = None

    trace_path: Annotated[Path | None, Field(description="Path to write pytorch profiler trace to.")] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(
            description="Timeout in seconds for torch distributed ops. Defaults to 600 seconds.",
        ),
    ] = 600

    loss_impl: Annotated[
        Literal["liger", "torch"], Field(description="Implementation of the cross entropy loss function to use.")
    ] = "torch"

    heartbeat: Annotated[
        HeartbeatConfig | None, Field(description="The heartbeat config for monitoring training progress.")
    ] = None

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench is not None:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def validate_pack_function(self):
        if self.model.cp > 1 and self.data.pack_function != "cat":
            raise ValueError("Packing function must be 'cat' when CP is enabled")
        return self

    @model_validator(mode="after")
    def validate_cp_seq_len(self):
        if self.model.cp > 1 and self.data.seq_len % self.model.cp != 0:
            raise ValueError("Sequence length must be divisible by CP degree")
        return self

    @model_validator(mode="after")
    def validate_cp_micro_batch_size(self):
        if self.model.cp > 1 and self.data.micro_batch_size != 1:
            raise ValueError("Micro batch size must be 1 when CP is enabled")
        return self

    @model_validator(mode="after")
    def validate_seq_len(self):
        if self.data.pack_function == "stack":
            if self.data.seq_len % 256 != 0:
                raise ValueError("The sequence length must be divisible by 256 when using pack function stack")
        return self

    @model_validator(mode="after")
    def dont_do_massive_traces(self):
        if self.trace_path:
            if self.max_steps is None:
                raise ValueError("Must specify max_steps when tracing")
            if self.max_steps >= 10:
                raise ValueError(
                    "Tracing more than 10 steps is not recommended as your trace will be massive. Remove this line if you really want to trace more steps."
                )
        return self

    @model_validator(mode="after")
    def validate_lora_adapter_saving(self):
        if self.ckpt and self.ckpt.weights and self.ckpt.weights.save_adapter_separately:
            lora_enabled = self.model and self.model.lora
            if not lora_enabled:
                raise ValueError(
                    "save_adapter_separately=True requires LoRA to be enabled. "
                    "Set model.lora or disable save_adapter_separately."
                )
        return self

    @model_validator(mode="after")
    def validate_opt_and_fsdp_offload(self):
        if self.optim.type == "muon" and self.model.fsdp_cpu_offload:
            raise ValueError("Muon optimizer does not support FSDP CPU offload")
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def validate_and_disable_chunked_loss(self):
        if isinstance(self.model.fused_lm_head_chunk_size, int):
            raise ValueError(
                "Chunked loss is not supported for SFT training yet, please set `model.fused_lm_head_chunk_size` to 'disabled'"
            )

        self.model.fused_lm_head_chunk_size = "disabled"
        return self

    @model_validator(mode="after")
    def ep_only_with_custom_impl(self):
        if self.model.ep > 1 and self.model.impl not in ("custom", "auto"):
            raise ValueError("EP is only supported with the custom implementation or auto mode")

        return self
