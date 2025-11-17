from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.utils.pydantic_config import BaseConfig


class ModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[str, Field(description="Name or path of the HF model to use.")] = "Qwen/Qwen3-0.6B"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for tokenizer initialization.",
        ),
    ] = False


ServerType = Literal["vllm", "openai"]


class ClientConfig(BaseConfig):
    """Configures the OAI client."""

    timeout: Annotated[
        int,
        Field(
            description="Timeout in seconds. By default, it is set to 1200 seconds.",
        ),
    ] = 1200

    base_url: Annotated[
        list[str],
        Field(
            description="Base URLs to use for the OpenAI API. By default, it is set to a single server on localhost at port 8000 which matches the default local vLLM server configuration. If you specify more than one URL, the client will round-robin (chat) completion requests across all servers.",
        ),
    ] = ["http://localhost:8000/v1"]

    api_key_var: Annotated[
        str,
        Field(
            description="Name of environment variable containing the API key to use for the OpenAI API. Will parse using `os.getenv(client_config.api_key_var)`. Can be set to an arbitrary string if the inference server is not protected by an API key. If multiple URLs are specified, the same API key will be used for all servers.",
        ),
    ] = "OPENAI_API_KEY"

    server_type: Annotated[
        ServerType,
        Field(
            description="Type of inference server that the client is connected to. Can be 'vllm' or 'openai'. Defaults to vLLM, which is our default client for training.",
        ),
    ] = "vllm"

    @model_validator(mode="after")
    def auto_setup_server_type(self):
        if any(base_url == "https://api.openai.com/v1" for base_url in self.base_url):
            self.server_type = "openai"
        return self


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        str,
        Field(description="Logging level for the process. Will determine the logging verbosity and format."),
    ] = "info"

    vf_level: Annotated[
        str,
        Field(description="Logging level for the verifiers package. Will determine the logging verbosity and format."),
    ] = "warn"

    file: Annotated[
        bool,
        Field(
            description="Whether to log to a file. If True, will log to a file in the output directory.",
        ),
    ] = True

    log_data: Annotated[
        bool,
        Field(
            description="Whether to log the first data sample to the logger.",
        ),
    ] = False


class LogExtrasConfig(BaseConfig):
    """Configures extra logging for W&B tables."""

    samples: Annotated[
        bool,
        Field(
            description="Whether to log prompt/response samples to W&B tables.",
        ),
    ] = True

    distributions: Annotated[
        bool,
        Field(
            description="Whether to log distributions (like rewards, advantages, etc.) to W&B tables.",
        ),
    ] = True

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Step interval at which to log extras to W&B table.",
        ),
    ] = 10


class PlatformUploadConfig(BaseConfig):
    """Configures uploading to Prime platform."""

    enabled: Annotated[
        bool,
        Field(description="Whether to upload metrics to the Prime platform."),
    ] = False

    api_key: Annotated[
        str | None,
        Field(description="API key for Prime platform. If None, will read from PRIME_API_KEY env var."),
    ] = None

    api_url: Annotated[
        str,
        Field(description="Base URL for Prime platform API."),
    ] = "https://api.primeintellect.ai"

    upload_rollouts: Annotated[
        bool,
        Field(description="Whether to upload sample rollouts to the platform."),
    ] = True

    upload_interval: Annotated[
        int,
        Field(description="Upload metrics every N steps. Set to 1 to upload every step.", ge=1),
    ] = 1

    max_rollouts_per_step: Annotated[
        int,
        Field(description="Maximum number of rollouts to upload per step.", ge=1),
    ] = 10


class WandbMonitorConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: Annotated[str, Field(description="The W&B project to log to.")] = "prime-rl"

    name: Annotated[
        str | None,
        Field(
            description="The W&B name to to use for logging.",
        ),
    ] = None

    offline: Annotated[bool, Field(description="Whether to run W&B in offline mode.")] = False

    # Individual configs (can only be specified on trainer or orchestrator)
    id: Annotated[
        str | None,
        Field(
            description="The W&B run ID to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.",
        ),
    ] = None

    log_extras: Annotated[
        LogExtrasConfig | None,
        Field(
            description="Configuration for logging extras to W&B tables. If None, no extras are logged.",
        ),
    ] = None

    platform: Annotated[
        PlatformUploadConfig | None,
        Field(
            description="Configuration for uploading to Prime platform. If None, platform upload is disabled.",
        ),
    ] = None
