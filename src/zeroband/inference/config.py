from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig

from zeroband.inference.pipeline import PipelineConfig
from zeroband.inference.rewards import RewardsConfig
from zeroband.utils.monitor import MultiMonitorConfig


class SamplingParamConfig(BaseConfig):
    temperature: float = 0.6
    max_tokens: int | None = None
    ignore_eos: bool = False
    top_p: float = 1
    n: int = 8
    logprobs: int = 0  # 0 mean 1 logprob here
    top_k: int = -1
    seed: int | None = None


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

    sampling: SamplingParamConfig = SamplingParamConfig()

    # Whether to enable thinking for the model. Used by the `format_prompts` function to prepend a thinking prompt
    enable_thinking: bool = True

    # Whether to calculate and save logprobs in the output
    enable_logprobs: bool = False

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
