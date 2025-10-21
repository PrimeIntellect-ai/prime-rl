from typing import Annotated, List

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from prime_rl.utils.pydantic_config import BaseSettings, parse_argv


def parse_dict_from_str(s: str) -> dict:
    # Accept str like "a=1,b=two" and turn into {"a":"1","b":"two"}
    d = {}
    d["args"] = {}
    for kv in [kv.strip() for kv in s.strip().split(";")]:
        if "=" not in kv:
            raise ValueError(f"Expected key=value, got: {kv!r}")
        k, val = kv.split("=", 1)
        if k in ["id", "name"]:
            d[k.strip()] = val.strip()
        else:
            d["args"][k.strip()] = val.strip()
    return d


class EnvConfig(BaseModel):
    id: str
    name: str
    args: dict = {}

    @model_validator(mode="before")
    @classmethod
    def auto_set_id(cls, values):
        if "id" not in values:
            values["id"] = values["name"]
        return values


# Reusable “string to dict” adapter
EnvCLIConfig = Annotated[EnvConfig, BeforeValidator(parse_dict_from_str)]


class AppSettings(BaseSettings):
    env: List[EnvCLIConfig] = Field(default_factory=list)


print(parse_argv(AppSettings).env)
