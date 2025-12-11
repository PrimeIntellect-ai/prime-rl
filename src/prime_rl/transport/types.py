import msgspec


# Orchestrator -> Packer
class TrainingExample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    advantage: float | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingExample]
    temperature: float
    seq_len: int
    step: int


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int] | None = None
