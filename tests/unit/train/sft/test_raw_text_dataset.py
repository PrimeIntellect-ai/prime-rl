from datasets import Dataset

from prime_rl.trainer.sft.data import RawTextDataset


class _DummyTokenizer:
    """Tokenizes each character to its ordinal, so expected token ids are trivial to check."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [ord(c) for c in text]}


def test_raw_text_dataset_yields_shifted_next_token_samples():
    dataset = Dataset.from_list([{"text": "abcd"}, {"text": "wxyz"}])
    raw_text_dataset = RawTextDataset(dataset, _DummyTokenizer(), shuffle=False)

    sample = next(iter(raw_text_dataset))
    expected_ids = [ord(c) for c in "abcd"]
    assert sample["input_ids"] == expected_ids[:-1]
    assert sample["target_ids"] == expected_ids[1:]
    assert sample["position_ids"] == list(range(len(expected_ids) - 1))
    assert all(sample["loss_mask"])
    assert len(sample["loss_mask"]) == len(expected_ids) - 1


def test_raw_text_dataset_skips_empty_and_single_token_rows():
    dataset = Dataset.from_list([{"text": ""}, {"text": "a"}, {"text": "bc"}])
    raw_text_dataset = RawTextDataset(dataset, _DummyTokenizer(), shuffle=False, max_epochs=1)

    samples = list(raw_text_dataset)
    assert len(samples) == 1
    assert samples[0]["input_ids"] == [ord("b")]
    assert samples[0]["target_ids"] == [ord("c")]


def test_raw_text_dataset_state():
    dataset = Dataset.from_list([{"text": "abcd"}])
    raw_text_dataset = RawTextDataset(dataset, _DummyTokenizer(), shuffle=False)
    dataiter = iter(raw_text_dataset)

    assert raw_text_dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert raw_text_dataset.state_dict() == {"step": 1, "epoch": 0}
