import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from zeroband.utils.config import BaseConfig, MultiMonitorConfig

# These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
TOML_PATHS: list[str] = []


def set_toml_paths(toml_paths: list[str]) -> None:
    global TOML_PATHS
    TOML_PATHS = toml_paths


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model. Largely follows the vLLM sampling parameters (https://docs.vllm.ai/en/latest/api/vllm.sampling_params.html)."""

    n: Annotated[int, Field(default=16, ge=1, description="Number of output sequences to return for the given prompt.")]

    presence_penalty: Annotated[
        float,
        Field(
            default=0,
            description="Penalizes new tokens based on whether they appear in the generated text so far. Values >0 => penalize, Values <0 => reward repeated tokens",
        ),
    ]

    frequency_penalty: Annotated[
        float,
        Field(
            default=0,
            description="Penalizes new tokens based on their frequency in the generated text so far. Values <0 => penalize repetition, Values >0 => reward repetition",
        ),
    ]

    temperature: Annotated[
        float,
        Field(
            default=1.0,
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ]

    top_p: Annotated[
        float,
        Field(
            default=1,
            gt=0,
            le=1,
            description="Cumulative probability of the top tokens to consider. If 1, all tokens are considered.",
        ),
    ]

    top_k: Annotated[
        int,
        Field(default=-1, ge=-1, description="Number of top tokens to consider. If -1, all tokens are considered."),
    ]

    min_p: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.",
        ),
    ]

    logprobs: Annotated[
        int | None,
        Field(
            default=0,
            description="Number of tokens to return log probabilities for. If None, no probability is returned. For all other values, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens (e.g. 0 returns only the logprob of the chosen token)",
        ),
    ]

    max_tokens: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of output tokens to generate per sequence. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ]
    min_tokens: Annotated[int, Field(default=0, ge=0, description="Minimum number of output tokens to generate per sequence.")]

    @model_validator(mode="after")
    def convert_negative_logprobs_to_none(self):
        """Convert negative logprobs values to None to disable logprobs calculation."""
        if self.logprobs is not None and self.logprobs < 0:
            self.logprobs = None
        return self


class PipelineParallelConfig(BaseConfig):
    """Configures pipeline parallel inference."""

    rank: Annotated[int, Field(default=0, ge=0, description="Rank of the current node in the pipeline")]

    world_size: Annotated[int, Field(default=1, ge=1, description="Total number of pipeline stages.")]

    iroh_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Seed used to create the public node address. If None, a random seed will be used.",
        ),
    ]

    iroh_peer_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Peer address to connect to. If None, the user will be prompted to enter it.",
        ),
    ]

    # Each retry takes ~30s, so 10 retries is ~300s (5min)
    connection_num_retries: Annotated[
        int,
        Field(default=10, ge=0, description="How many times to retry connection to peer. Each retry takes ~30s."),
    ]

    @property
    def is_enabled(self) -> bool:
        """Returns True if pipeline parallelism is enabled (world_size > 1)."""
        return self.world_size > 1

    @property
    def is_first_stage(self) -> bool:
        """Returns True if the current rank is the first rank."""
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        """Returns True if the current rank is the last rank."""
        return self.rank == self.world_size - 1


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    tp: Annotated[
        int | Literal["auto"],
        Field(
            default=1,
            description="Number of local GPUs to use for tensor parallelism. It is directly passed to vLLM. If 'auto', will be set to all available local GPUs.",
        ),
    ]

    dp: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Number of local GPUs to use for data parallelism. It is used to spawn multiple processes running vLLM instances independently.",
        ),
    ]

    # The pipeline parallelism configuration
    pp: Annotated[PipelineParallelConfig, Field(default=PipelineParallelConfig())]

    @model_validator(mode="after")
    def assert_valid_parallelism(self):
        assert not (self.dp > 1 and self.pp.world_size > 1), "Cannot use PP and DP together"
        return self

    def __str__(self) -> str:
        pp_str = f"pp.rank={self.pp.rank}, pp.world_size={self.pp.world_size}"
        return f"tp={self.tp} dp={self.dp} {pp_str}"


class LenRewardsConfig(BaseConfig):
    """Configures length reward."""

    reward_type: Annotated[Literal["exact", "max", "clip"], Field(default="max")]
    target_length_sampling: Annotated[Literal["discrete", "range"], Field(default="discrete")]
    length_prompt_location: Annotated[Literal["system_prompt", "instruction"], Field(default="system_prompt")]

    # applicable if target_length_sampling == "range"
    min_length: Annotated[int, Field(default=1000)]
    max_length: Annotated[int, Field(default=24000)]

    # applicable if target_length_sampling == "discrete"
    target_lengths: Annotated[list[float], Field(default=[500, 1000, 2000, 3000])]

    # applicable for reward_type max and exact
    reward_coef: Annotated[float, Field(default=0.0003)]

    # only applicable for reward_type == "max"
    max_reward_delta: Annotated[float, Field(default=0.5)]


class RewardsConfig(BaseConfig):
    """Configures rewards compuation"""

    len_reward: Annotated[LenRewardsConfig | None, Field(default=None)]
    advantage_estimation_method: Annotated[Literal["grpo", "dr_grpo", "opo"], Field(default="dr_grpo")]
    compute_reward: Annotated[bool, Field(default=True, description="Whether to compute the reward. If not set, will set reward to 0.")]

    def __str__(self) -> str:
        len_reward_str = "disabled" if self.len_reward is None else self.len_reward
        return f"len_reward={len_reward_str} advantage_estimation_method={self.advantage_estimation_method}"


class ModelConfig(BaseConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html)."""

    name: Annotated[str, Field(default="Qwen/Qwen3-0.6B", description="Name or path of the HF model to use.")]

    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            default="auto",
            description="Data type for model weights and activations. If 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.",
        ),
    ]

    kv_cache_dtype: Annotated[
        Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        Field(default="auto", description="Data type for the KV cache. If 'auto' will use the model data type."),
    ]

    max_model_len: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum model context length. If None, will use the maximum context length from model config.",
        ),
    ]

    quantization: Annotated[
        Literal["awq", "gguf", "gptq", "bitsandbytes", "fp8"] | None,
        Field(
            default=None,
            description="Method used to quantize the weights. If None, will apply the default quantization (if any) from model config.",
        ),
    ]

    enforce_eager: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance.",
        ),
    ]

    device: Annotated[Literal["auto", "cuda", "cpu"], Field(default="auto", description="Device to use for inference.")]

    enable_thinking: Annotated[
        bool,
        Field(default=True, description="Whether to enable thinking. Used by the `format_prompts` function to prepend a thinking prompt."),
    ]


class DifficultyFilteringConfig(BaseConfig):
    """Configures filtering of the dataset by difficulty. If None, no filtering is applied."""

    solve_rate_field: Annotated[
        str, Field(default="solve_rate_qwen_r1_distill_7b", description="Dataset field in the dataset that contains the solve rate.")
    ]
    min_solve_rate: Annotated[float, Field(default=0.0, ge=0, le=1, description="Minimum solve rate to include.")]
    max_solve_rate: Annotated[float, Field(default=0.5, ge=0, le=1, description="Maximum solve rate to include.")]


class DataConfig(BaseConfig):
    """Configures the data to be used for inference."""

    name: Annotated[
        str,
        Field(
            default="PrimeIntellect/INTELLECT-2-RL-Dataset",
            description="Name of the HF dataset to use.",
        ),
    ]

    split: Annotated[str, Field(default="train", description="Split of the dataset to use.")]

    max_prompt_len: Annotated[
        int | None,
        Field(
            default=None,
            description="If set, filters out all samples with more than this number of input tokens. If None, no filtering is applied.",
        ),
    ]

    difficulty_filtering: Annotated[DifficultyFilteringConfig | None, Field(default=None)]

    def __str__(self) -> str:
        max_prompt_len_str = "disabled" if self.max_prompt_len is None else self.max_prompt_len
        difficult_filter_str = "disabled" if self.difficulty_filtering is None else self.difficulty_filtering
        return f"name={self.name} split={self.split} max_prompt_len={max_prompt_len_str} difficulty_filtering={difficult_filter_str}"


class RLConfig(BaseConfig):
    """Configures inference when used in conjunction with a RL trainer."""

    step_endpoint: Annotated[
        str | None,
        Field(
            default=None,
            description="An API endpoint that returns the current step during an RL run. Defaults to None, which means that the local inference step counter is used.",
        ),
    ]

    ckpt_start_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to the checkpoint to start from. Defaults to None, which means that the base model specified in `--model.name` is used.",
        ),
    ]

    ckpt_path: Annotated[Path, Field(default=Path("checkpoints"), description="Path to read new checkpoints from.")]

    clean_ckpt_path: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to clean the checkpoint path at the start of the inference.",
        ),
    ]

    max_async: Annotated[
        int,
        Field(
            default=2,
            description="Maximum number of steps that inference can be ahead of training.",
        ),
    ]


class Config(BaseSettings):
    """Configures inference."""

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

    toploc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to produce TOPLOC proofs for the inference outputs.",
        ),
    ]

    toploc2: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to use the TOPLOC2 Sampler",
        ),
    ]

    max_batch_size: Annotated[
        int | Literal["auto"],
        Field(
            default="auto",
            description="Maximum number of of sequences to decode in parallel. If 'auto', the maximum batch size is automatically computed.",
        ),
    ]

    start_step: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Inference step to start from.",
        ),
    ]

    max_steps: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of inference steps to run. If None, will run indefinitely.",
        ),
    ]

    rollout_path: Annotated[
        Path,
        Field(
            default=Path("rollouts"),
            description="Path to write inference outputs to. The folder will be automatically created and populated with subdirectories for each step.",
        ),
    ]

    clean_rollout_path: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to clean the rollout path at the start of the inference.",
        ),
    ]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed used across inference components. If None, no seeding is used.",
        ),
    ]

    log_level: Annotated[
        Literal["debug", "info", "warning", "critical"],
        Field(
            default="info",
            description="Logging level for the inference run.",
        ),
    ]

    task_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Task ID for the inference run. Set in production by protocol worker via environment variable. Not necessary for local runs.",
        ),
    ]

    group_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Group ID for the inference run. Set in production by protocol worker via environment variable. Not necessary for local runs.",
        ),
    ]

    @model_validator(mode="after")
    def set_log_level(self):
        os.environ["PRIME_LOG_LEVEL"] = self.log_level
        return self

    @model_validator(mode="after")
    def enforce_eager_for_pp(self):
        if self.parallel.pp.world_size > 1:
            self.model.enforce_eager = True
        return self

    @model_validator(mode="after")
    def disable_toploc_for_fp32(self):
        if self.model.dtype == "float32":
            self.toploc = False
        return self

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_prefix="PRIME_",
        env_nested_delimiter="__",
        # By default, we do not parse CLI. To activate, set `_cli_parse_args` to true or a list of arguments at init time.
        cli_parse_args=False,
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_implicit_flags=True,
        cli_use_class_docs_for_groups=True,
    )

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
        global TOML_PATHS
        return (
            TomlConfigSettingsSource(settings_cls, toml_file=TOML_PATHS),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
