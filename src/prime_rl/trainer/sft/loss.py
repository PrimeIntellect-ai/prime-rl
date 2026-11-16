"""Custom SFT losses (``[loss]``): a loss function in place of token cross-entropy, with the rest of the SFT trainer
unchanged."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from prime_rl.configs.sft import SFTCustomLossConfig, SFTDecisionLossConfig
from prime_rl.trainer.data import Batch
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.lm_head import token_logprobs
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing the loss on one sample, as packing put it in the micro-batch's row, without the row's
    padding."""

    model: nn.Module
    hidden_states: Float[Tensor, "seq hidden"]
    """The backbone's final (post-norm) hidden states of the sample's tokens, without the vocabulary projection."""
    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    target_ids: Int[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]
    truncated: bool
    """Whether packing cut the sample's tail: only a sample longer than a whole row, which is then alone in it."""
    sample_data: Any = None
    """What the dataset attached to the sample (``Sample.sample_data``), such as a System One record's questions and
    targets; None for chat data."""
    _target_logprobs: Callable[[], Tensor] | None = field(default=None, repr=False)

    def target_logprobs(self) -> Float[Tensor, "seq"]:
        """log p(target_ids) of the sample's tokens through the model's LM head. In training, the LM head runs once per
        micro-batch, on all of its samples, at the first call on any of them."""
        if self._target_logprobs is not None:
            return self._target_logprobs()
        return token_logprobs(self.model.lm_head, self.hidden_states[None], self.target_ids[None])[0]


@dataclass
class LossOutputs:
    """Outputs from computing the loss on one sample."""

    loss: Float[Tensor, ""]
    """Summed loss. The trainer divides it by the global ``count`` over all samples, ranks and micro-batches of the
    step."""
    count: int | Tensor
    """What the loss is averaged over (e.g. the sample, or its tokens)."""
    metrics: dict[str, Tensor] = field(default_factory=dict)
    """Values to log, any shape, one value per item (e.g. the sample, or each of its tokens). Each name is logged with
    the mean, std, min and max of its values over the step's samples and ranks, as ``loss/<name>/{mean,std,min,max}``
    in training and ``val/<name>/...`` in validation."""


LossFn = Callable[..., LossOutputs]
"""A custom loss: ``my_loss(inputs: LossInputs, **kwargs) -> LossOutputs``, called with ``[loss.kwargs]``."""


def setup_sft_loss_fn(config: SFTCustomLossConfig | SFTDecisionLossConfig) -> Callable[[LossInputs], LossOutputs]:
    if isinstance(config, SFTDecisionLossConfig):
        # Imported here: the System One loss module imports this one.
        from prime_rl.trainer.sft.system_one.loss import decision_loss

        return partial(decision_loss, brier_weight=config.brier_weight)
    fn: LossFn = import_object(config.import_path)
    return partial(fn, **config.kwargs)


def run_loss_fn(loss_fn: Callable[[LossInputs], LossOutputs], model: nn.Module, batch: Batch) -> LossOutputs:
    """The custom loss of one micro-batch: ``loss_fn`` called on each of its samples, inside the model's forward pass on
    the backbone's final hidden states, where stock SFT computes its cross-entropy: FSDP keeps the root unit gathered
    for it and treats the modules it calls, such as the LM head, as part of the forward. The samples' losses and counts
    are summed, and each metric's values concatenated. ``batch["sample_lens"]`` and ``batch["truncated"]`` are read
    on the host."""
    sample_lens, truncated = batch["sample_lens"].tolist(), batch["truncated"].tolist()
    sample_data = batch.get("sample_data")
    if sample_data is None:
        sample_data = [None] * len(sample_lens)
    num_tokens = sum(sample_lens)
    input_ids, position_ids, target_ids, loss_mask = (
        batch[key][0, :num_tokens] for key in ("input_ids", "position_ids", "target_ids", "loss_mask")
    )

    def loss(hidden_states: Tensor) -> LossOutputs:
        hidden_states = hidden_states[0, :num_tokens]
        row_logprobs = cache(lambda: token_logprobs(model.lm_head, hidden_states[None], target_ids[None])[0])

        def sample_logprobs(span: slice) -> Tensor:
            return row_logprobs()[span]

        outputs, start = [], 0
        for sample_len, cut, data in zip(sample_lens, truncated, sample_data, strict=True):
            span = slice(start, start + sample_len)
            inputs = LossInputs(
                model=model,
                hidden_states=hidden_states[span],
                input_ids=input_ids[span],
                position_ids=position_ids[span],
                target_ids=target_ids[span],
                loss_mask=loss_mask[span],
                truncated=cut,
                sample_data=data,
                _target_logprobs=partial(sample_logprobs, span),
            )
            outputs.append(loss_fn(inputs))
            start += sample_len
        metrics: dict[str, list[Tensor]] = {}
        for output in outputs:
            for name, values in output.metrics.items():
                metrics.setdefault(name, []).append(values.detach().reshape(-1))
        return LossOutputs(
            loss=sum(output.loss for output in outputs),
            count=sum(output.count for output in outputs),
            metrics={name: torch.cat(values) for name, values in metrics.items()},
        )

    return forward(model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], hidden_states_fn=loss)
