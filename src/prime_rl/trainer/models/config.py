from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

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
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_tied_word_embeddings(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("tie_word_embeddings"):
            raise ValueError(
                f"{cls.__name__}: the checkpoint ties its LM head to the input embeddings "
                "(tie_word_embeddings=true), which PrimeRL does not support."
            )
        return data

    @model_validator(mode="after")
    def _propagate_attn_implementation(self) -> "PrimeModelConfig":
        for name in type(self).model_fields:
            sub_config = getattr(self, name)
            if isinstance(sub_config, PrimeModelConfig):
                sub_config.attn_implementation = self.attn_implementation
        return self
