from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig

from zeroband.inference.pipeline import PipelineConfig
from zeroband.inference.rewards import RewardsConfig
from zeroband.utils.monitor import MultiMonitorConfig


class SamplingConfig(BaseConfig):
    """Parameters for sampling tokens from logits. Follows the vLLM sampling parameters."""

    # Number of output sequences to return for the given prompt.
    n: int = 8

    # Penalizes new tokens based on whether they appear in the generated text so far (Values >0 penalize/ <0 reward repeated tokens)
    presence_penalty: float = 0

    # Penalizes new tokens based on their frequency in the generated text so far (Values >0 penalize/ <0 reward repeated tokens)
    frequency_penalty: float = 0

    # Scales the output probability distribution to control the randomness of the sampling. Lower values lead to more deterministic sampling while higher values make the model more random. If 0, sampling will be greedy.
    temperature: float = 0.6

    # The cumulative probability of the top tokens to consider. For example, if 0.9, then the minimum set of tokens whose cumulative probability is at least 0.9 is considered. If 1, all tokens are considered.
    top_p: float = 1

    # The number of tokens with highest probability to consider. If -1, all tokens are considered.
    top_k: int = -1

    # The minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.
    min_p: float = 0.0

    # The number of log probabilities to return per output token. When set to None, no probability is returned. If set to a non-None value, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens (e.g. 0 returns only the logprob of the chosen token)
    logprobs: int | None = 0

    # Whether to ignore the EOS token. If True, the model will generate until the end of the sequence. If False, the model will generate until the EOS token is reached.
    ignore_eos: bool = False

    # Maximum number of output tokens to generate per sequence. If None, the model will generate until EOS or maximum context length is reached.
    max_tokens: int | None = None

    # Minimum number of output tokens to generate per sequence.
    min_tokens: int | None = None

    # Random seed to use for generation. If None, sampling will be random.
    seed: int | None = None

    @model_validator(mode="after")
    def convert_negative_logprobs_to_none(self):
        """Convert negative logprobs values to None to disable logprobs calculation."""
        if self.logprobs is not None and self.logprobs < 0:
            self.logprobs = None
        return self


class DifficultyFilteringConfig(BaseConfig):
    solve_rate_field: str = "solve_rate_qwen_r1_distill_7b"
    min_solve_rate: float = 0.0
    max_solve_rate: float = 0.5


class Config(BaseConfig):
    model_name: str
    dataset: str = "PrimeIntellect/INTELLECT-2-RL-Dataset"

    # The maximum number of of sequences to decode in parallel (if None, will be computed automatically)
    batch_size: int | Literal["auto"] = "auto"

    # The step to start from (if None, will start from 0)
    start_step: int | None = None

    output_path: str = "outputs"
    clean_output_path: bool = False  # if true, the output path will be cleaned up before running the inference

    total_step: int | None = None
    rollout_path: str | None = None
    step_endpoint: str | None = None
    download_dir: str | None = None

    quant: Literal["fp8"] | None = None

    sampling: SamplingConfig = SamplingConfig()

    # Whether to enable thinking for the model. Used by the `format_prompts` function to prepend a thinking prompt
    enable_thinking: bool = True

    enforce_eager: bool = False
    max_model_len: int | None = None

    async_level: int = 2  # the amount of step for which we can be in advance

    # Parallelism
    tp: int | Literal["auto"] = 1
    dp: int = 1
    pp: PipelineConfig = PipelineConfig()

    # Monitoring (performance, progress, system metrics, etc.)
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    gpus_ids: list[int] | None = None
    prime_log_freq: int | None = None

    seed: int | None = None  # THIS ARG FOR TESTING PURPOSES ONLY

    dtype: Literal["fp32", "bf16"] = "bf16"

    ckpt_start_path: str | None = None

    toploc: bool = False
    toploc2: bool = True

    rewards: RewardsConfig = RewardsConfig()
    difficulty_filtering: DifficultyFilteringConfig | None = None

    max_prompt_len: int | None = None

    @model_validator(mode="after")
    def disable_toploc_for_fp32(self):
        if self.dtype == "fp32":
            self.toploc = False
        return self

    @model_validator(mode="after")
    def enforce_eager_for_tp(self):
        if self.pp.world_size > 1:
            self.enforce_eager = True
        return self

    @model_validator(mode="after")
    def assert_valid_parallelism(self):
        assert not (self.dp > 1 and self.pp.world_size > 1), "Cannot use PP and DP together"
        return self
