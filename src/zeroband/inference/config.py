import sys
from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from zeroband.inference.pipeline import PipelineConfig
from zeroband.inference.rewards import RewardsConfig
from zeroband.utils.monitor import MultiMonitorConfig

# Dynamically extract paths to config files from CLI to pass to pydantic-settings TOML source as `toml_file` argument
# This is a hacky workaround from https://github.com/pydantic/pydantic-settings/issues/259
TOML_FILE_PATHS = [arg.replace("@", "").strip() for arg in sys.argv if arg.startswith("@") and arg.endswith(".toml")]
sys.argv = [arg for arg in sys.argv if not arg.startswith("@")]


class SamplingConfig(BaseConfig):
    """Parameters for sampling tokens from logits. Follows the vLLM sampling parameters."""

    # Number of output sequences to return for the given prompt.
    n: Annotated[int, Field(default=8, ge=1)]

    # Penalizes new tokens based on whether they appear in the generated text so far (Values >0 penalize/ <0 reward repeated tokens)
    presence_penalty: Annotated[float, Field(default=0)]

    # Penalizes new tokens based on their frequency in the generated text so far (Values >0 penalize/ <0 reward repeated tokens)
    frequency_penalty: Annotated[float, Field(default=0)]

    # Scales the output probability distribution to control the randomness of the sampling. Lower values lead to more deterministic sampling while higher values make the model more random. If 0, sampling will be greedy.
    temperature: Annotated[float, Field(default=0.6, ge=0)]

    # The cumulative probability of the top tokens to consider. For example, if 0.9, then the minimum set of tokens whose cumulative probability is at least 0.9 is considered. If 1, all tokens are considered.
    top_p: Annotated[float, Field(default=1, gt=0, le=1)]

    # The number of tokens with highest probability to consider. If -1, all tokens are considered.
    top_k: Annotated[int, Field(default=-1, ge=-1)]

    # The minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.
    min_p: Annotated[float, Field(default=0.0, ge=0)]

    # The number of log probabilities to return per output token. When set to None, no probability is returned. If set to a non-None value, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens (e.g. 0 returns only the logprob of the chosen token)
    logprobs: Annotated[int | None, Field(default=0)]

    # Whether to ignore the EOS token. If True, the model will generate until the end of the sequence. If False, the model will generate until the EOS token is reached.
    ignore_eos: Annotated[bool, Field(default=False)]

    # Maximum number of output tokens to generate per sequence. If None, the model will generate until EOS or maximum context length is reached.
    max_tokens: Annotated[int | None, Field(default=None, ge=1)]

    # Minimum number of output tokens to generate per sequence.
    min_tokens: Annotated[int, Field(default=0, ge=0)]

    # Random seed to use for generation. If None, sampling will be random.
    seed: int | None = None

    @model_validator(mode="after")
    def convert_negative_logprobs_to_none(self):
        """Convert negative logprobs values to None to disable logprobs calculation."""
        if self.logprobs is not None and self.logprobs < 0:
            self.logprobs = None
        return self


class ParallelConfig(BaseConfig):
    """
    Configurations for multi-node and multi-GPU setups. By default, inference
    runs on a single GPU. We support tensor parallelism via vLLM, data
    parallelism via multi-processing and pipeline parallelism over public IP via
    custom hooks.  All combinations of parallelism are supported except for DP
    and PP together.
    """

    # The TP world size, i.e. the number of local GPUs to use for tensor parallelism within vLLM. This argument is directly passed to vLLM as `tensor_parallel_size`. If "auto", will be set to the number of local GPUs available.
    tp: Annotated[int | Literal["auto"], Field(default=1, ge=1)]

    # The DP world size, i.e. the number of local GPUs use for data parallelism. This argument is used to spawn multiple processes running vLLM instance independently.
    dp: Annotated[int, Field(default=1, ge=1)]

    # The pipeline parallelism configuration
    pp: Annotated[PipelineConfig, Field(default=PipelineConfig())]

    @model_validator(mode="after")
    def enforce_eager_for_pp(self):
        if self.pp.world_size > 1:
            self.enforce_eager = True
        return self

    @model_validator(mode="after")
    def assert_valid_parallelism(self):
        assert not (self.dp > 1 and self.pp.world_size > 1), "Cannot use PP and DP together"
        return self

    def __str__(self) -> str:
        pp_str = f"pp.rank={self.pp.rank}, pp.world_size={self.pp.world_size}"
        return f"tp={self.tp} dp={self.dp} {pp_str}"


class ModelConfig(BaseConfig):
    """Configurations for the model to be used for inference. Most arguments are passed directly to the vLLM LLM class as engine arguments."""

    # The name or path of the HF model to use
    name: Annotated[str, Field(default="Qwen/Qwen3-0.6B", alias="model")]

    # Data type for model weights and activations. Defaults to "auto" which will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models
    dtype: Annotated[Literal["auto", "float16", "bfloat16", "float32"], Field(default="auto")]

    # Data type for the KV cache. Defaults to "auto" which will use the model data type.
    kv_cache_dtype: Annotated[Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"], Field(default="auto")]

    # The maximum model context length. Defaults to None which will use the maximum context length as specified in the model config. Note, that the model will stop generation if the number of input and output tokens exceeds this value.
    max_model_len: Annotated[int | None, Field(default=None)]

    # Method used to quantize the weights. Defaults to None which will applies the default quantization (if any) as specified in the model config.
    quantization: Annotated[Literal["awq", "gguf", "gptq", "bitsandbytes", "fp8"] | None, Field(default=None)]

    # Whether to enforce PyTorch eager mode for the model. Defaults to False, which uses PyTorch eager and cuda graphs in hybrid for maximal performance.
    enforce_eager: Annotated[bool, Field(default=False)]

    # The device to use for inference. Defaults to "auto".
    device: Annotated[Literal["auto", "cuda", "cpu"], Field(default="auto")]

    # Whether to enable thinking for the model. Used by the `format_prompts` function to prepend a thinking prompt
    enable_thinking: Annotated[bool, Field(default=True)]


class DifficultyFilteringConfig(BaseConfig):
    """Configuration that controls offline difficulty filtering of the dataset."""

    # The field in the dataset that contains the solve rate
    solve_rate_field: Annotated[str, Field(default="solve_rate_qwen_r1_distill_7b")]

    # The minimum solve rate to filter by
    min_solve_rate: Annotated[float, Field(default=0.0, ge=0, le=1)]

    # The maximum solve rate to filter by
    max_solve_rate: Annotated[float, Field(default=0.5, ge=0, le=1)]


class DataConfig(BaseConfig):
    """Configurations for the dataset to be used for inference."""

    # The name of the HF dataset to use
    name: Annotated[str, Field(default="PrimeIntellect/INTELLECT-2-RL-Dataset", alias="dataset")]

    # The split of the dataset to use
    split: Annotated[str, Field(default="train")]

    # The maximum number of input tokens. If set, filters out all samples with more than this number of input tokens. Defaults to None, which means no filtering.
    max_prompt_len: Annotated[int | None, Field(default=None)]

    # Configuration that controls offline difficulty filtering of the dataset
    difficulty_filtering: Annotated[DifficultyFilteringConfig | None, Field(default=None)]

    def __str__(self) -> str:
        max_prompt_len_str = "disabled" if self.max_prompt_len is None else self.max_prompt_len
        difficult_filter_str = "disabled" if self.difficulty_filtering is None else self.difficulty_filtering
        return f"name={self.name} split={self.split} max_prompt_len={max_prompt_len_str} difficulty_filtering={difficult_filter_str}"


class RLConfig(BaseConfig):
    """Configuration if inference is run in an RL setting"""

    # An API endpoint that returns the current step during an RL run. Defaults to None, which means that the local inference step counter is used.
    step_endpoint: Annotated[str | None, Field(default=None)]

    # The path to the checkpoint to start from. Defaults to None, which means that the base model specified in `--model.name` is used.
    ckpt_start_path: Annotated[str | None, Field(default=None)]

    # The path to read new checkpoints from. Only relevant for RL training when the inference model is updated during the run. Defaults to "checkpoints", which means that checkpoints will be read from subdirectories for each step in the "checkpoints" folder at the root of the project directory.
    ckpt_path: Annotated[str, Field(default="checkpoints")]

    # Whether to clean the checkpoint path at the start of the inference. Useful for debugging. Defaults to False.
    clean_ckpt_path: Annotated[bool, Field(default=False)]

    # The maximum number of steps that inference can be ahead of training. Defaults to 2, which means that inference can be 2 steps ahead of training.
    max_async: Annotated[int, Field(default=2)]


class Config(BaseSettings):
    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The data configuration
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The parallel configuration
    parallel: Annotated[ParallelConfig, Field(default=ParallelConfig())]

    # The reward configuration
    rewards: Annotated[RewardsConfig, Field(default=RewardsConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The RL configuration. If None, inference will run in a non-RL setting.
    rl: Annotated[RLConfig | None, Field(default=None)]

    # Whether to produce TOPLOC proofs for the inference outputs. Defaults to False. This is required in production to ensure that the inference outputs are can be verified.
    toploc: Annotated[bool, Field(default=False)]

    # The maximum number of of sequences to decode in parallel. Defaults to "auto", which automatically computes the maximum batch size based on the model's context length and available KV cache.
    max_batch_size: Annotated[int | Literal["auto"], Field(default="auto")]

    # The step to start from. Defaults to 0, which means that inference will start from the beginning of the dataset.
    start_step: Annotated[int, Field(default=0, ge=0)]

    # The maximum number of steps to run. Defaults to None, which means the inference will run indefinitely.
    max_steps: Annotated[int | None, Field(default=None)]

    # The path to write inference outputs (rollouts) to. The folder will be automatically created and populated with subdirectories for each step. Defaults to writing to "rollouts" at the root of the project directory.
    rollout_path: Annotated[str, Field(default="rollouts")]

    # Whether to clean the rollout path at the start of the inference. Useful for debugging. Defaults to False.
    clean_rollout_path: Annotated[bool, Field(default=False)]

    # Random seed for reproducible outputs. Is used across inference components, such as the model, sampling and batching. Should only be used for debugging. Defaults to None, which skips seeding.
    seed: Annotated[int | None, Field(default=None)]

    toploc2: bool = True

    @model_validator(mode="after")
    def disable_toploc_for_fp32(self):
        if self.model.dtype == "float32":
            self.toploc = False
        return self

    # Pydantic settings configuration
    model_config = SettingsConfigDict(env_prefix="PRIME_", cli_parse_args=True, cli_kebab_case=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # This is a hacky way to dynamically load TOML file paths from CLI
        # https://github.com/pydantic/pydantic-settings/issues/259
        global TOML_FILE_PATHS
        return (
            TomlConfigSettingsSource(settings_cls, toml_file=TOML_FILE_PATHS),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
