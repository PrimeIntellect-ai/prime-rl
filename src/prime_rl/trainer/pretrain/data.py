from collections.abc import Iterable, Iterator
from typing import cast

from datasets import Dataset, load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.pretrain import TextDataConfig
from prime_rl.trainer.sft.data import Sample, StatefulIterableDataset, cat_collate, setup_and_interleave_datasets
from prime_rl.utils.logger import get_logger

LIST_KEYS = ("input_ids", "target_ids", "loss_mask", "position_ids", "seq_lens")


def tokenize_document(tokenizer: PreTrainedTokenizer, text: str) -> list[int]:
    """Token ids of one document, with the tokenizer's BOS (if it adds one) and a closing EOS."""
    token_ids = tokenizer(text)["input_ids"]
    if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
        token_ids.append(tokenizer.eos_token_id)
    return token_ids


def cut_row(tokens: list[int], doc_starts: list[int], seq_len: int, document_masking: bool) -> Sample:
    """One training row from ``seq_len + 1`` tokens: inputs are the first ``seq_len``, targets the
    last ``seq_len``. ``doc_starts`` are the offsets in ``tokens`` where a document begins. With
    document masking, every document (and the continued document at offset 0) is its own segment
    with positions starting at 0."""
    assert len(tokens) == seq_len + 1
    if document_masking:
        boundaries = sorted({0, *(start for start in doc_starts if 0 < start < seq_len)})
        seq_lens = [end - start for start, end in zip(boundaries, [*boundaries[1:], seq_len])]
    else:
        seq_lens = [seq_len]
    return {
        "input_ids": tokens[:-1],
        "target_ids": tokens[1:],
        "loss_mask": [True] * seq_len,
        "position_ids": [position for length in seq_lens for position in range(length)],
        "seq_lens": seq_lens,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


class TextDataset(StatefulIterableDataset):
    """Tokenized documents of this data rank, reshuffled every epoch. Documents are addressed by
    their global index, so resuming jumps straight to the saved position."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        shuffle: bool = True,
        seed: int = 0,
        non_dp_size: int = 1,
    ):
        super().__init__(non_dp_size)
        self.dataset = dataset
        self.num_examples = len(dataset)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.shuffle = shuffle
        self.seed = seed

    def epoch_dataset(self) -> Dataset:
        return self.dataset.shuffle(seed=self.seed + self.epoch) if self.shuffle else self.dataset

    def __iter__(self) -> Iterator[list[int]]:
        self._setup_world_info()
        dataset = self.epoch_dataset()
        while True:
            self.step += 1
            epoch = (self.step - 1) // self.num_examples
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.epoch_dataset()
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue
            example = cast(dict, dataset[(self.step - 1) % self.num_examples])
            token_ids = tokenize_document(self.tokenizer, example[self.text_column])
            source = example.get("__subset") or example.get("__split")
            self.num_samples[source] += 1
            self.num_tokens[source] += len(token_ids)
            yield token_ids


class StreamPacker(StatefulIterableDataset):
    """Cuts a document stream into rows of exactly ``seq_len`` tokens (see ``cut_row``) and
    yields ``rows_per_sample`` consecutive rows concatenated as one packed sample. Documents
    continue across rows, so nothing is truncated or padded, but attention never crosses a
    row. The tokens not yet emitted are part of the checkpoint state. Any iterable of
    documents works as the source; checkpointing needs a ``TextDataset``."""

    def __init__(
        self,
        dataset: TextDataset | Iterable[list[int]],
        seq_len: int,
        document_masking: bool,
        rows_per_sample: int = 1,
    ):
        self.dataset = dataset
        self.seq_len = seq_len
        self.document_masking = document_masking
        self.rows_per_sample = rows_per_sample
        self.buffer: list[int] = []
        self.doc_starts: list[int] = []

    def state_dict(self) -> dict:
        return {
            "dataset": self.dataset.state_dict(),
            "progress": {"num_samples": dict(self.dataset.num_samples), "num_tokens": dict(self.dataset.num_tokens)},
            "buffer": self.buffer,
            "doc_starts": self.doc_starts,
        }

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.dataset.num_samples.update(state_dict["progress"]["num_samples"])
        self.dataset.num_tokens.update(state_dict["progress"]["num_tokens"])
        self.buffer = state_dict["buffer"]
        self.doc_starts = state_dict["doc_starts"]

    def next_row(self) -> Sample:
        row = cut_row(self.buffer[: self.seq_len + 1], self.doc_starts, self.seq_len, self.document_masking)
        self.buffer = self.buffer[self.seq_len :]
        self.doc_starts = [start - self.seq_len for start in self.doc_starts if start >= self.seq_len]
        return row

    def __iter__(self) -> Iterator[Sample]:
        for document in self.dataset:
            self.doc_starts.append(len(self.buffer))
            self.buffer.extend(document)
            while len(self.buffer) >= self.seq_len * self.rows_per_sample + 1:
                rows = [self.next_row() for _ in range(self.rows_per_sample)]
                sample = {key: [value for row in rows for value in row[key]] for key in LIST_KEYS}
                yield {**sample, "mm_kwargs": None, "mm_token_type_ids": None}


def load_text_dataset(config: TextDataConfig) -> Dataset:
    """Load (and interleave) the configured subsets/splits. Only their files are downloaded."""
    splits = config.splits or ["train"] * len(config.subsets or [None])
    subsets = config.subsets or [None] * len(splits)
    if len(splits) == 1:
        dataset = load_dataset(config.name, subsets[0], split=splits[0], revision=config.revision)
        return cast(Dataset, dataset).select_columns([config.text_column])
    dataset = setup_and_interleave_datasets(
        dataset_name=config.name,
        subsets_and_splits=list(zip(subsets, splits)),
        probabilities=config.probabilities,
        stopping_strategy=config.stopping_strategy,
        seed=config.seed,
        revision=config.revision,
    )
    return dataset.select_columns([config.text_column, "__subset", "__split"])


def split_holdout(dataset: Dataset, holdout_docs: int) -> tuple[Dataset, Dataset | None]:
    """Split off the last ``holdout_docs`` documents (in dataset order) as a held-out set."""
    if holdout_docs == 0:
        return dataset, None
    num_train = len(dataset) - holdout_docs
    assert num_train > 0, f"data.holdout_docs ({holdout_docs}) must be smaller than the dataset ({len(dataset)})"
    return dataset.select(range(num_train)), dataset.select(range(num_train, len(dataset)))


def load_eval_documents(
    dataset: str, subset: str | None, split: str, revision: str | None, text_column: str
) -> Dataset:
    return cast(Dataset, load_dataset(dataset, subset, split=split, revision=revision)).select_columns([text_column])


def setup_text_dataloader(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, config: TextDataConfig, non_dp_size: int
) -> StatefulDataLoader:
    get_logger().info(f"Training on {len(dataset):,} documents from {config.name}")
    text_dataset = TextDataset(
        dataset, tokenizer, config.text_column, shuffle=config.shuffle, seed=config.seed, non_dp_size=non_dp_size
    )
    packer = StreamPacker(text_dataset, config.seq_len, config.document_masking, config.micro_batch_size)
    return StatefulDataLoader(
        packer, batch_size=1, collate_fn=cat_collate, num_workers=config.num_workers, pin_memory=True
    )
