from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig

from zeroband.inference.pipeline import PipelineConfig
from zeroband.inference.rewards import LenRewardsConfig
from zeroband.inference.toploc import TopLocConfig
from zeroband.utils.models import ModelName


class SamplingConfig(BaseConfig):
    temperature: float = 0.6
    max_tokens: int | None = None
    ignore_eos: bool = False
    top_k: int = -1
    top_p: float = 1
    n: int = 8
    logprobs: int = 0  # 0 mean 1 logprob here


class ModelConfig(BaseConfig):
    # Model name (HF compativle)
    name: ModelName

    # vLLM LLMEngine argument
    enforce_eager: bool = False
    dtype: Literal["fp32", "bf16"] = "bf16"
    quant: Literal["fp8"] | None = None
    max_model_len: int | None = None


class DataFilteringConfig(BaseConfig):
    solve_rate_field: str = "solve_rate_qwen_r1_distill_7b"
    min_solve_rate: float = 0.0
    max_solve_rate: float = 0.5


class DataConfig(BaseConfig):
    # Dataset
    name: str

    # Offline difficulty filtering settings
    filtering: DataFilteringConfig | None = None


class ParallelConfig(BaseConfig):
    # Tensor parallelism size (passed directly to vLLM)
    tp: int | Literal["auto"] = 1

    # Data parallelism size (spawns dp processes)
    dp: int = 1

    # Pipeline parallelism config (specifies rank and size of pipeline parallelism)
    pp: PipelineConfig = PipelineConfig()


class IOConfig(BaseConfig):
    # Path for writing generation output files
    data_dir: str = "outputs"

    # Path to read model checkpoints from
    checkpoint_dir: str | None = None

    # Path to read initial model checkpoint from
    checkpoint_start_dir: str | None = None

    # Path for caching data/ model downloads (e.g. vLLM, HF, ... cache)
    cache_dir: str | None = None

    # If true, the data_dir will be cleaned before running inference
    cleanup: bool = False


class Config(BaseConfig):
    model: ModelConfig
    data: DataConfig
    sampling: SamplingConfig = SamplingConfig()
    io: IOConfig = IOConfig()
    parallel: ParallelConfig = ParallelConfig()
    toploc: TopLocConfig = TopLocConfig()
    len_reward: LenRewardsConfig | None = None

    # How many samples to process in one iteration
    batch_size: int = 32

    # How many steps to run (one step means processing one batch)
    max_steps: int | None = None

    # How many samples to process in total
    max_samples: int | None = None

    # Seed for reproducibility (only use for testing purposes)
    seed: int | None = None

    async_level: int = 2  # the amount of step for which we can be in advance

    # Miscellaneous (find better place for these later on)
    step_endpoint: str | None = None
    ckpt_start_path: str | None = None
    prime_log_freq: int | None = None

    @model_validator(mode="after")
    def disable_toploc_for_fp32(self):
        if self.model.dtype == "fp32":
            self.toploc = False
        return self

    @model_validator(mode="after")
    def enforce_eager_for_tp(self):
        if self.parallel.pp.world_size > 1:
            self.model.enforce_eager = True
        return self

    @model_validator(mode="after")
    def assert_valid_parallelism(self):
        assert not (self.parallel.dp > 1 and self.parallel.pp.world_size > 1), "Cannot use PP and DP together"
        return self
