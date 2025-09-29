from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.utils.pydantic_config import BaseConfig

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2"]

MOE_MODEL_MAPS = {
    "Qwen/Qwen3-30B-A3B": "Jackmin108/Qwen3-30B-A3B-Fast",
    "moonshotai/Moonlight-16B-A3B-Instruct": "Jackmin108/Moonlight-16B-A3B-Instruct-Fast",
}


class ActivationCheckpointConfig(BaseModel):
    """Configures activation checkpointing."""

    freq: Annotated[
        int,
        Field(
            ge=1,
            description="Applies activation checkpointing to every `freq` layers. Defaults to 1, which will is full activation checkpointing.",
        ),
    ] = 1


class CompileConfig(BaseModel):
    """Configures model compilation."""

    fullgraph: Annotated[
        bool,
        Field(description="Whether to compile the transformer blocks with fullgraph."),
    ] = False


class DebugModelConfig(BaseModel):
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
        CompileConfig | None,
        Field(
            description="Whether to compile the model using `torch.compile`. Currently discouraged because it was found to destabilize training.",
        ),
    ] = None

    ac: Annotated[
        ActivationCheckpointConfig | None,
        Field(
            description="Whether to apply activation checkpointing to the model. If None, will not apply activation checkpointing.",
        ),
    ] = None

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
        Literal["hf", "liger_kernel", "custom"],
        Field(
            description="Whether to use Liger Kernel.",
        ),
    ] = "hf"

    log_signature: Annotated[
        bool,
        Field(
            description="Whether to log the model signature after loading the model.",
        ),
    ] = False

    load_using_meta: Annotated[
        bool,
        Field(
            description="Whether to load the model using meta device then load from HF ckpt.",
        ),
    ] = False

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

    debug: Annotated[
        DebugModelConfig,
        Field(
            description="Debugging feature around model and distributed training.",
        ),
    ] = DebugModelConfig()

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
            if self.impl != "hf":
                raise ValueError("Trust remote code is only supported with the HF implementation.")
        return self

    @model_validator(mode="after")
    def random_init_only_with_meta(self):
        """Random initialize is only supported with the custom implementation."""
        if self.debug.random_init and not self.load_using_meta:
            raise ValueError("Random initialize is only supported when loading with meta.")
        return self


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


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the training checkpoint. If None, will only checkpoint at the end of training.",
        ),
    ] = None

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
    
    skip_dataloader: Annotated[
        bool,
        Field(
            description="Whether to skip checkpointing the dataloader. If True, will not checkpoint the dataloader.",
        ),
    ] = False


class WeightCheckpointConfig(BaseConfig):
    """Configures checkpointing the model weights for updating the inference engines (RL trainer) or continued post-training (on SFT trainer)."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save weight checkpoint. If None, will save all necessary weight checkpoints on RL trainer and only final weight checkpoint on SFT trainer.",
        ),
    ] = None

    save_async: Annotated[
        bool,
        Field(
            description="Whether to save the weight checkpoint asynchronously.",
        ),
    ] = True
