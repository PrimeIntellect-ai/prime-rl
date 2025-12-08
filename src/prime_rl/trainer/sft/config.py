from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator


from typing import Dict, Optional
from typing_extensions import Annotated, Literal
from pydantic import Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    HeartbeatConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)
from prime_rl.utils.config import LogConfig, WandbConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class BaseDataConfig(BaseModel):
    """Base config for SFT data."""

    batch_size: Annotated[int, Field(ge=1)] = 128
    seq_len: Annotated[int, Field(ge=1)] = 128
    pack_function: Literal["cat", "stack"] = "cat"
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

    name: Annotated[
        str,
        Field(description="Name or path of the HF dataset to use."),
    ] = "PrimeIntellect/Reverse-Text-SFT"

    # NEW: proper nested structure for complex multi-subset data loading
    subsets: Annotated[
        Optional[Dict[str, Dict[str, float]]],
        Field(
            description="Mapping from subset name → its own split ratios. "
                        "Example: {'wiki': {'train': 0.8, 'test': 0.2}, 'news': {'train': 0.95}}"
        ),
    ] = None

    # Keep old flat fields for backward compatibility (deprecated)
    old_subsets: Annotated[
        Optional[list[str]],
        Field(description="DEPRECATED: use nested `subsets` instead"),
    ] = None
    splits: Annotated[
        Optional[list[str]],
        Field(description="DEPRECATED: use nested `subsets` instead"),
    ] = None
    probabilities: Annotated[
        Optional[list[float]],
        Field(description="DEPRECATED: use nested `subsets` instead"),
    ] = None

    stopping_strategy: Annotated[
        Literal["first_exhausted", "all_exhausted"],
        Field(description="Stop when first subset is exhausted or when all are."),
    ] = "all_exhausted"

    shuffle: Annotated[
        bool,
        Field(description="Whether to shuffle the dataset at the beginning of each epoch."),
    ] = True

    seed: Annotated[
        int,
        Field(description="Random seed for shuffling (epoch count is added each epoch)."),
    ] = 0

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        # Warn if someone still uses the old flat style
        if self.old_subsets is not None or self.splits is not None or self.probabilities is not None:
            print(
                "Warning: old_subsets/splits/probabilities are deprecated → "
                "use the new nested `subsets` dictionary instead."
            )

        # Validate the new nested format
        if self.subsets is not None:
            # Check that subsets is not empty
            if not self.subsets:
                raise ValueError("subsets dictionary cannot be empty. Provide at least one subset with splits.")
            
            for subset_name, split_ratios in self.subsets.items():
                if not split_ratios:
                    raise ValueError(f"Subset '{subset_name}' must have at least one split defined")
                total = sum(split_ratios.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"Subset '{subset_name}' split ratios must sum to 1.0 (got {total:.6f})"
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
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5 and use fake data.",
        ),
    ] = False

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
        if self.bench:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def validate_pack_function(self):
        if self.model.cp > 1 and self.data.pack_function != "stack":
            raise ValueError("Packing function must be 'stack' when CP is enabled")
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
            lora_enabled = self.model and self.model.experimental and self.model.experimental.lora
            if not lora_enabled:
                raise ValueError(
                    "save_adapter_separately=True requires LoRA to be enabled. "
                    "Set model.experimental.lora or disable save_adapter_separately."
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
