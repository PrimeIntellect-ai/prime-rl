from pydantic import BaseModel, model_validator

from prime_rl.utils.pydantic_config import BaseSettings, parse_argv


class EnvConfig(BaseModel):
    """Configures the verifiers environment."""

    dod: dict[str, dict] = {}
    lod: list[dict] = [{}]
    l: list[str] = []
    d: dict = {}

    @model_validator(mode="before")
    @classmethod
    def parse_dynamic_env_configs(cls, values):
        print(values)
        return values


class EvalConfig(BaseSettings):
    env: dict = {}

    @model_validator(mode="before")
    @classmethod
    def parse_dynamic_env_configs(cls, values):
        print(values)
        return values


print(parse_argv(EvalConfig))
