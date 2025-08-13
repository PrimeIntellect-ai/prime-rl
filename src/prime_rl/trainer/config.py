from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from prime_rl.utils.pydantic_config import BaseConfig

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2"]


class ModelConfig(BaseConfig):
    """Configures the model for training."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

    attn: Annotated[AttnImplementation, Field(description="The attention implementation to use.")] = "flash_attention_2"

    compile: Annotated[
        bool,
        Field(
            description="Whether to compile the model using `torch.compile`. Currently discouraged because it was found to destabilize training.",
        ),
    ] = False

    ac: Annotated[
        bool,
        Field(
            description="Whether to apply activation checkpointing to the model.",
        ),
    ] = False

    reshard_after_forward: Annotated[
        bool, Field(description="Whether to reshard the model after each forward pass.")
    ] = True


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseModel):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear"] = "linear"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = 0

    decay_steps: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of steps to decay the learning rate during the final portion of training. If None, will use remaining steps after warmup.",
        ),
    ] = None


class CosineSchedulerConfig(BaseModel):
    """Configuration for cosine learning rate scheduler."""

    type: Literal["cosine"] = "cosine"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = 0

    decay_steps: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of steps to decay the learning rate during the final portion of training. If None, will use remaining steps after warmup.",
        ),
    ] = None


SchedulerConfigType: TypeAlias = ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig


class BaseOptimizerConfig(BaseModel):
    lr: Annotated[float, Field(ge=0)] = 4e-4
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


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the checkpoint. If None, will only checkpoint at the end of training.",
        ),
    ] = None

    save_async: Annotated[
        bool,
        Field(
            description="Whether to save the checkpoint asynchronously.",
        ),
    ] = False

    resume_step: Annotated[
        int | None,
        Field(
            ge=1,
            description="Step to resume training from. If None, will start from scratch.",
        ),
    ] = None

    keep: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints.",
        ),
    ] = None


class WeightCheckpointConfig(BaseConfig):
    """Configures checkpointing the model weights for updating the inference engines (RL trainer) or continued post-training (on SFT trainer)."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the weights. If None, will only keep necessary weight checkpoints for resuming training.",
        ),
    ] = None

    save_async: Annotated[
        bool,
        Field(
            description="Whether to save the weights asynchronously.",
        ),
    ] = True
