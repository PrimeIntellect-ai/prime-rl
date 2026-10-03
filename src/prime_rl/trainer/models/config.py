from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

AttnImplementation = Literal["flash_attention_2", "flash_attention_3", "flash_attention_4"]


class PrimeModelConfig(BaseModel):
    """Architecture hyperparameters, validated from a checkpoint's ``config.json``.

    Field names match the ``config.json`` keys. Keys an architecture does not declare are dropped.
    ``attn_implementation`` is not read from ``config.json``: the trainer sets it, and it propagates
    to nested sub-configs (e.g. a VLM's ``text_config`` and ``vision_config``).
    """

    model_config = ConfigDict(extra="ignore")

    model_type: ClassVar[str]

    attn_implementation: AttnImplementation = "flash_attention_2"
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    @field_validator("pad_token_id", mode="before")
    @classmethod
    def _unwrap_pad_token_id(cls, value: int | list[int] | None) -> int | None:
        # Some configs (e.g. Llama 3.2) store a list of token ids.
        return value[0] if isinstance(value, list) else value

    @model_validator(mode="after")
    def _propagate_attn_implementation(self) -> "PrimeModelConfig":
        for name in type(self).model_fields:
            sub_config = getattr(self, name)
            if isinstance(sub_config, PrimeModelConfig):
                sub_config.attn_implementation = self.attn_implementation
        return self
