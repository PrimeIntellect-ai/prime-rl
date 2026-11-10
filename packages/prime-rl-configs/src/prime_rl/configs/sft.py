from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator
from renderers import AutoRendererConfig, DefaultRendererConfig, RendererConfig
from renderers.base import MODEL_RENDERER_MAP

from prime_rl.configs.supervised import BaseDataConfig, BaseValConfig, SupervisedTrainerConfig
from prime_rl.utils.config import BaseConfig


class FakeDataConfig(BaseDataConfig):
    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    """Use fixed-length samples or variable-length samples."""

    input_ids: Literal["increasing", "random"] = "increasing"
    """Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids."""

    seed: int = 0
    """Seed for the per-rank packing/token generator, combined with the data rank."""


class LossMaskConfig(BaseConfig):
    system: bool = False
    """System messages contribute to the loss."""

    user: bool = False
    """User messages contribute to the loss."""

    assistant: bool = True
    """Assistant messages contribute to the loss."""

    tool: bool = False
    """Tool messages contribute to the loss."""


class SFTColumnsConfig(BaseConfig):
    """Dataset columns the trainer reads. Set a field to read a column under another name."""

    messages: str = "messages"
    """Column with the whole chat as a list of OpenAI chat messages."""

    prompt: str = "prompt"
    """Column with the prompt as a list of OpenAI chat messages, read with ``completion`` when a row has no messages."""

    completion: str = "completion"
    """Column with the completion as a list of OpenAI chat messages."""

    tools: str = "tools"
    """Column with the tool schemas in OpenAI function-calling format."""

    message_loss_mask: str = "message_loss_mask"
    """Optional column of booleans or integer 0/1 flags, one per resolved message (prompt + completion for split rows). Zero excludes a message from loss; one preserves its normal supervision. Missing or null masks leave supervision unchanged."""

    renderer: dict[str, str] = {"reasoning_effort": "reasoning_effort"}
    """Per-sample renderer arguments as ``renderer field = dataset column``, e.g. ``reasoning_effort = "effort"``. A row's non-null value overrides the ``[renderer]`` setting; rows and datasets without the column use it unchanged."""


class SFTDataConfig(BaseDataConfig):
    type: Literal["sft"] = "sft"

    name: str = "PrimeIntellect/Reverse-Text-SFT"
    """HF dataset name or path."""

    revision: str | None = None
    """HF dataset revision to load. Ignored for a local path."""

    subsets: list[str] | None = None
    """Subsets to load from the HF dataset."""

    splits: list[str] | None = None
    """Splits to load from the HF dataset."""

    probabilities: list[float] | None = None
    """Sampling probabilities for each subset/split."""

    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted"
    """Stopping strategy when interleaving multiple subsets/splits."""

    shuffle: bool = True
    """Shuffle the dataset at the start of each epoch."""

    seed: int = 0
    """Random seed for shuffling. Re-shuffled per epoch by adding the epoch count to the seed."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns that carry per-sample renderer arguments."""

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        if self.subsets is not None or self.splits is not None:
            if self.subsets is not None and self.splits is not None:
                if len(self.subsets) != len(self.splits):
                    raise ValueError(
                        "Number of subsets must be equal to number of splits. Please specify which split to load for each subset."
                    )
            if self.subsets is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.subsets):
                    raise ValueError(
                        "Number of probabilities must be equal to number of subsets. Please specify a probability for each subset."
                    )
            if self.splits is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.splits):
                    raise ValueError(
                        "Number of probabilities must be equal to number of splits. Please specify a probability for each split."
                    )
        return self


class SFTValConfig(BaseValConfig):
    after_optimizer_step: bool = False
    """Run the interval validations after the step's optimizer update instead of before it. ``val/loss`` at
    step N is then the loss of the weights a checkpoint of step N holds, and the final step is validated as well
    (also when ``max_steps`` is not a multiple of ``interval``). ``eval_on_start`` still runs before the first
    update. Off: ``val/loss`` at step N is the loss of the weights after N - 1 updates."""

    keep_best: bool = False
    """Keep a weights-only snapshot (model parameters and buffers, no optimizer state) of the lowest validation
    loss so far under ``<checkpoint dir>/best-weights``, with ``best.json`` recording its step and loss. Each rank
    writes and reads only its own shard file, so the snapshot only reloads into the same world size and sharding.
    The ``eval_on_start`` validation is not a candidate. A new best replaces the previous snapshot."""

    data: SFTDataConfig


class SFTHFExportConfig(BaseConfig):
    """Export checkpoints as Hugging Face models before the trainer exits.

    The export rewrites the safetensors files of ``base_dir`` (the HF checkpoint the run started from) with the
    fine-tuned tensors, cast to the base file's dtypes. Every other tensor (e.g. an untrained MTP head) and the
    shard layout, index, config and tokenizer stay the base's. The result goes to
    ``<checkpoint dir>/hf-export/<name>``, is checked against the base (keys, dtypes, shapes, relative change of
    each tensor) and, if a repo is set and ``HF_TOKEN`` is in the environment, pushed. A failed export or push
    is logged and never fails the run."""

    checkpoints: list[Literal["best", "last"]] = ["best", "last"]
    """Which checkpoints to export: ``last`` is the final weights, ``best`` the ``[val] keep_best`` snapshot."""

    best_repo: str | None = None
    """Hugging Face repo id (``org/name``) for the best checkpoint. None keeps it on the volume only."""

    last_repo: str | None = None
    """Hugging Face repo id (``org/name``) for the last checkpoint. None keeps it on the volume only."""

    private: bool = True
    """Create the repos as private."""

    base_dir: Path | None = None
    """Local HF checkpoint whose files are rewritten. Defaults to ``model.name``, which must then be a local
    directory."""

    keep_base_prefixes: list[str] = ["mtp."]
    """Base tensors under these name prefixes are not held by the trainer (prime-rl's NemotronH drops the MTP
    head) and are copied from the base unchanged. Every other base tensor must come from the trained model."""

    max_relative_change: float = Field(0.2, gt=0)
    """Refuse to push when any fine-tuned tensor moved further than this (relative L2) from the base: that
    points to a wrong key mapping, not training."""

    model_card: str = ""
    """Free text added to the generated README.md (what was trained on, how to run it)."""

    @model_validator(mode="after")
    def validate_checkpoints(self):
        if not self.checkpoints or len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("hf_export.checkpoints must be a non-empty list without duplicates")
        for repo in (self.best_repo, self.last_repo):
            if repo is not None and repo.count("/") != 1:
                raise ValueError(f"hf_export repo {repo!r} must be 'org/name'")
        return self


DataConfig: TypeAlias = Annotated[FakeDataConfig | SFTDataConfig, Field(discriminator="type")]


class SFTConfig(SupervisedTrainerConfig):
    mode: Literal["sft"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run sft`` accepts only
    "sft" (or no mode)."""

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer config. Defaults to auto-selecting from the tokenizer model name."""

    data: DataConfig = SFTDataConfig()

    val: SFTValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    hf_export: SFTHFExportConfig | None = None
    """Export the best and/or last checkpoint as Hugging Face models at the end of the run. If None, no export."""

    @model_validator(mode="after")
    def validate_best_and_hf_export(self):
        if self.hf_export is None:
            return self
        if "best" in self.hf_export.checkpoints and (self.val is None or not self.val.keep_best):
            raise ValueError("hf_export of the 'best' checkpoint needs [val] keep_best = true")
        if self.model.lora is not None:
            raise ValueError("hf_export writes full weights and does not support LoRA")
        if self.model.vlm is not None:
            raise ValueError("hf_export does not support VLM training")
        return self

    @model_validator(mode="after")
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        if self.data.type == "fake" and self.val is None:
            return self

        model_id = self.tokenizer.name or self.model.name
        if isinstance(self.renderer, AutoRendererConfig):
            if model_id in MODEL_RENDERER_MAP:
                return self
            reason = f"no typed renderer is registered for {model_id!r}"
        elif isinstance(self.renderer, DefaultRendererConfig):
            reason = "renderer.name='default' selects DefaultRenderer"
        else:
            return self

        raise ValueError(
            f"SFT requires a typed renderer with sampled-token and content attribution, but {reason}. "
            "Implement and register the renderer in the renderers package, or explicitly select an existing "
            "typed renderer only when its template is verified to match."
        )

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires data.micro_batch_size = 1.")
        if self.val is not None and self.val.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires val.data.micro_batch_size = 1.")
        return self
