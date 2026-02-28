from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prime_rl.configs.shared import SlurmConfig
from prime_rl.utils.pydantic_config import BaseSettings

# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


class VLLMConfig(BaseModel):
    """Configures vLLM. Arguments must match exactly with vLLM's CLI arguments."""

    model_config = ConfigDict(extra="allow")

    model_name: Annotated[str, Field(description="The name of the model to use.")] = "Qwen/Qwen3-0.6B"
    tool_call_parser: Annotated[str | None, Field(description="The tool call parser to use.")] = None
    reasoning_parser: Annotated[str | None, Field(description="Parser reasoning parser to use.")] = None
    data_parallel_size: Annotated[int, Field(description="The data parallel size to use.")] = 1
    data_parallel_size_local: Annotated[
        int | None, Field(description="The data parallel size to use on this node.")
    ] = None
    tensor_parallel_size: Annotated[int, Field(description="The tensor parallel size to use.")] = 1
    enable_lora: Annotated[bool, Field(description="Whether to enable LoRA.")] = False
    max_lora_rank: Annotated[int | None, Field(description="The maximum LoRA rank to use.")] = None
    max_loras: Annotated[
        int,
        Field(
            description="The maximum number of LoRAs to use.",
        ),
    ] = 8
    max_cpu_loras: Annotated[
        int,
        Field(
            description="The maximum number of LoRAs to use on CPU.",
        ),
    ] = 100
    api_server_count: Annotated[int, Field(description="The number of API servers to use.")] = 1

    @model_validator(mode="after")
    def validate_valid_vllm_arg(self):
        # TODO
        return self

    @model_validator(mode="after")
    def auto_setup_tool_call_parser(self):
        # TODO
        return self

    @model_validator(mode="after")
    def auto_setup_reasoning_parser(self):
        # TODO
        return self

    @model_validator(mode="after")
    def auto_setup_max_lora_rank(self):
        """
        Auto-setup max_lora_rank by rounding up to the nearest valid vLLM value.

        vLLM only accepts specific values for max_lora_rank: (1, 8, 16, 32, 64, 128, 256, 320, 512).
        This validator ensures that any configured rank is rounded up to the minimum valid value
        that can serve adapters of the requested rank.
        """
        if self.max_lora_rank is not None:
            original_rank = self.max_lora_rank
            for valid_rank in VALID_VLLM_LORA_RANKS:
                if valid_rank >= self.max_lora_rank:
                    self.max_lora_rank = valid_rank
                    break
            else:
                raise ValueError(f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}")
        return self

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        """
        Ensures that we have at least as many API servers as data parallel
        size. Unless LoRA is enabled, in which case only one API server is
        supported (vLLM limitation).
        """
        if "api_server_count" not in self.model_fields_set:
            min_api_server_count = self.data_parallel_size_local or self.data_parallel_size
            if self.api_server_count < min_api_server_count:
                self.api_server_count = min_api_server_count

        if self.enable_lora:
            self.api_server_count = 1  # LoRA requires only one API server
        return self


class WeightBroadcastConfig(BaseSettings):
    """Configures weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


class BaseInferenceDeploymentConfig(BaseModel):
    """Base deployment config for inference."""

    model_config = ConfigDict(extra="forbid")

    gpus_per_node: Annotated[int, Field(description="Number of GPUs per node.")] = 8


class SingleNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a single-node inference deployment."""

    type: Literal["single_node"] = "single_node"


class MultiNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    """Configures a multi-node inference deployment. Each node runs an independent vLLM replica."""

    type: Literal["multi_node"] = "multi_node"

    num_nodes: Annotated[int, Field(ge=1, description="Number of inference nodes.")] = 1


InferenceDeploymentConfig: TypeAlias = Annotated[
    SingleNodeInferenceDeploymentConfig | MultiNodeInferenceDeploymentConfig, Field(discriminator="type")
]


class InferenceConfig(BaseSettings):
    """Configures inference."""

    vllm: VLLMConfig = VLLMConfig()

    weight_broadcast: Annotated[WeightBroadcastConfig, Field(description="The weight broadcast config.")] = (
        WeightBroadcastConfig()
    )

    deployment: Annotated[
        InferenceDeploymentConfig,
        Field(
            description="Deployment configuration for inference.",
        ),
    ] = SingleNodeInferenceDeploymentConfig()

    slurm: Annotated[
        SlurmConfig | None,
        Field(
            description="SLURM configuration. If set, the run will be submitted as a SLURM job instead of running locally.",
        ),
    ] = None

    output_dir: Annotated[Path, Field(description="Directory for SLURM logs and generated scripts.")] = Path("outputs")

    dry_run: Annotated[bool, Field(description="Only validate and dump resolved configs and exit early.")] = False

    @model_validator(mode="after")
    def validate_multi_node_requires_slurm(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            import prime_rl

            templates_dir = Path(prime_rl.__file__).parent / "templates"
            self.slurm.template_path = templates_dir / "inference.sbatch.j2"
        return self
