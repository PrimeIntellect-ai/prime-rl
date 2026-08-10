"""DatasetSFTSource: trainer-bound training samples straight from a HF SFT dataset.

The ``dataset_sft`` run-level algorithm replaces env rollouts entirely: the
orchestrator renders dataset examples (same schema as the native SFT trainer —
``messages`` or ``prompt``/``completion`` columns) through the policy renderer
into ce-routed :class:`TrainingSample` payloads and ships them through the
regular pack → send pipeline. Epochs cycle indefinitely (reshuffled per epoch),
mirroring env-based training; ``max_steps`` bounds the run.

The data position (``{epoch, cursor}``) round-trips through the orchestrator
checkpoint like ``TrainSource``'s does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import DatasetSFTAlgoConfig
from prime_rl.configs.sft import SFTDataConfig
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from datasets import Dataset
    from renderers.base import Renderer


class DatasetSFTSource:
    """Renders one HF SFT dataset into per-step ``TrainingSample`` batches.

    ``load()`` and ``next_batch()`` block on dataset I/O / tokenization — call
    them off the event loop. Batching matches the orchestrator's train sink:
    ``batch_size`` counts samples per step, ``token_batch_size`` accumulates
    payload tokens past the threshold."""

    def __init__(
        self,
        config: DatasetSFTAlgoConfig,
        *,
        renderer: Renderer,
        seq_len: int,
        batch_size: int | None,
        token_batch_size: int | None,
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.renderer = renderer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.epoch = 0
        self.cursor = 0
        self.num_examples = 0
        self._dataset: Dataset | None = None
        self._epoch_view: Dataset | None = None

    def load(self) -> None:
        """Load (and interleave) the raw HF dataset. Blocking I/O."""
        from prime_rl.trainer.sft.data import load_sft_dataset

        dataset_config = self.config.dataset
        self._dataset = load_sft_dataset(
            SFTDataConfig(
                name=dataset_config.name,
                subsets=dataset_config.subsets,
                splits=dataset_config.splits,
                probabilities=dataset_config.probabilities,
                stopping_strategy=dataset_config.stopping_strategy,
            )
        )
        self.num_examples = len(self._dataset)
        self._epoch_view = self._shuffle()
        get_logger().info(f"Loaded SFT dataset {dataset_config.name} ({self.num_examples} examples)")

    def _shuffle(self) -> Dataset:
        assert self._dataset is not None
        if not self.config.dataset.shuffle:
            return self._dataset
        return self._dataset.shuffle(seed=self.epoch + self.config.dataset.seed)

    def state_dict(self) -> dict:
        # The empty ``envs`` table keeps the checkpoint payload compatible with
        # the manager's ``TrainSource`` resume logging.
        return {"envs": {}, "dataset": {"epoch": self.epoch, "cursor": self.cursor}}

    def load_state_dict(self, state_dict: dict) -> None:
        position = state_dict["dataset"]
        self.epoch = position["epoch"]
        self.cursor = position["cursor"]
        self._epoch_view = self._shuffle()
        get_logger().info(f"Resumed dataset position - epoch={self.epoch}, cursor={self.cursor}/{self.num_examples}")

    def _next_example(self) -> dict:
        assert self._epoch_view is not None, "call load() first"
        if self.cursor >= self.num_examples:
            self.epoch += 1
            self.cursor = 0
            self._epoch_view = self._shuffle()
        example = self._epoch_view[self.cursor]
        self.cursor += 1
        return example

    def _render(self, example: dict) -> TrainingSample | None:
        """Render one example into a ce-routed ``TrainingSample``; ``None``
        when it carries no trainable tokens within ``seq_len``."""
        from prime_rl.trainer.sft.data import render_example

        rendered = render_example(self.renderer, example, self.config.dataset.loss_mask)
        if rendered.multi_modal_data is not None and rendered.multi_modal_data.mm_items:
            raise ValueError("dataset_sft does not support multimodal datasets")
        token_ids = list(rendered.token_ids)[: self.seq_len]
        mask = list(rendered.loss_mask)[: self.seq_len]
        if not any(mask):
            get_logger().warning(
                f"Skipping example because no trainable tokens were found within the context window ({self.seq_len})"
            )
            return None
        sample = TrainingSample(
            token_ids=token_ids,
            mask=mask,
            logprobs=[0.0] * len(token_ids),
            temperatures=[1.0] * len(token_ids),
            env_name=self.config.dataset.name,
        )
        stamp_loss_routing(sample, "ce")
        return sample

    def next_batch(self) -> list[TrainingSample]:
        """Render the next step's batch. Blocking (tokenizes examples)."""
        samples: list[TrainingSample] = []
        num_tokens = 0
        while True:
            if self.batch_size is not None:
                if len(samples) >= self.batch_size:
                    return samples
            elif num_tokens >= (self.token_batch_size or 0):
                return samples
            sample = self._render(self._next_example())
            if sample is None:
                continue
            samples.append(sample)
            num_tokens += len(sample.token_ids)
