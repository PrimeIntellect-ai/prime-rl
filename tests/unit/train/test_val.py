import json
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from renderers import AutoRendererConfig
from transformers import AutoTokenizer

from prime_rl.configs.supervised import ChatDataConfig
from prime_rl.configs.val import ChatValSourceConfig, DocumentsValSourceConfig
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.sft.data import setup_sft_dataset
from prime_rl.trainer.val import chat_source, contrast_summary, documents_source, rank_rows, token_byte_lengths

# Documents in two files of one folder, with an empty one. A token's id is its position in the folder.
DOCUMENTS = [[range(0, 10), range(10, 30), range(30, 30), range(30, 60)], [range(60, 103)]]


@pytest.mark.parametrize("num_documents", [None, 4])
def test_documents_source_scores_every_document_on_its_own(tmp_path: Path, num_documents: int | None):
    """A documents source scores every document (or the first num_documents) as if it were alone: windows of seq_len
    scored tokens, none with a token before its own first one, and every token after the document's first scored
    once."""
    for index, documents in enumerate(DOCUMENTS):
        tokens = [token for document in documents for token in document]
        np.array(tokens, dtype="<u4").tofile(tmp_path / f"{index:02d}.ds")
        np.cumsum([len(document) for document in documents], dtype="<u8").tofile(tmp_path / f"{index:02d}.ds.index")
        (tmp_path / f"{index:02d}.ds.metadata").write_text("tokenizer|4\n")
    seq_len = 8
    expected, items = [], []
    documents = [list(document) for documents in DOCUMENTS for document in documents][:num_documents]
    for item, document in enumerate(documents):
        for start in range(0, len(document) - 1, seq_len):
            window = document[start : start + seq_len + 1]
            expected.append((window[:-1], window[1:]))
            items.append(item)

    val_source = documents_source(
        DocumentsValSourceConfig(name="held-out", path=tmp_path, seq_len=seq_len), num_documents
    )
    assert [(inputs.tolist(), labels.tolist()) for inputs, labels in val_source.requests] == expected
    assert val_source.items.tolist() == items
    assert val_source.num_items == len(documents)


@pytest.mark.parametrize(("data_world_size", "cp_size"), [(1, 1), (3, 1), (2, 2)])
def test_ranks_score_every_token_once_in_equal_rows(data_world_size: int, cp_size: int):
    """Every (data, CP) rank runs the same number of rows, each its shard of a micro_batch_tokens row, and over all
    ranks every scored label is scored once and attributed to its request."""
    rng = np.random.default_rng(0)
    requests = []
    for length in rng.integers(1, 17, size=20):
        labels = rng.integers(0, 1000, size=length)
        requests.append((rng.integers(0, 1000, size=length), np.where(rng.random(length) < 0.7, labels, IGNORE_INDEX)))
    micro_batch_tokens = 32
    num_rows, scored = set(), []
    for data_rank in range(data_world_size):
        for cp_rank in range(cp_size):
            rows = list(rank_rows(requests, micro_batch_tokens, data_rank, data_world_size, cp_rank, cp_size))
            num_rows.add(len(rows))
            for inputs, positions, labels, lengths, owners in rows:
                assert len(inputs) == len(positions) == len(labels) == micro_batch_tokens // cp_size
                assert sum(lengths) == micro_batch_tokens
                if owners is not None:
                    scored += zip(owners.tolist(), labels[labels != IGNORE_INDEX].tolist())
    assert len(num_rows) == 1
    expected = [
        (index, label)
        for index, (_, labels) in enumerate(requests)
        for label in labels[labels != IGNORE_INDEX].tolist()
    ]
    assert sorted(scored) == sorted(expected)


def test_chat_source_scores_the_tokens_sft_trains_on(tmp_path: Path):
    """A chat source scores exactly the tokens SFT trains on in the same row, and none of the role scaffolding."""
    messages = [
        {"role": "system", "content": "Reverse the text."},
        {"role": "user", "content": "abc def"},
        {"role": "assistant", "content": "fed cba"},
        {"role": "user", "content": "ghi"},
        {"role": "assistant", "content": "ihg"},
    ]
    (tmp_path / "train.jsonl").write_text(json.dumps({"messages": messages}) + "\n")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    renderer = AutoRendererConfig()
    data = ChatDataConfig(source=[{"name": "chat", "dataset": str(tmp_path)}], seq_len=1024, shuffle=False)
    (sample,) = islice(setup_sft_dataset(tokenizer, data, dp_rank=0, dp_size=1, renderer_config=renderer), 1)

    config = ChatValSourceConfig(name="chat", dataset=str(tmp_path), seq_len=1024, loss_mask=data.loss_mask)
    config.resolve_columns(data.columns)
    # num_samples above the split's size scores every row.
    ((inputs, labels),) = chat_source(config, 5, tokenizer, renderer, ()).requests
    assert inputs.tolist() == sample["input_ids"].tolist()
    assert labels.tolist() == np.where(sample["loss_mask"], sample["target_ids"], IGNORE_INDEX).tolist()
    assert tokenizer.convert_tokens_to_ids("<|im_start|>") not in labels


def test_token_byte_lengths_add_up_to_the_text_bytes():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Tokens holding part of a multi-byte character count their own bytes; <think> is added but not special.
    text = "<think>Grüße aus Zürich!</think> 東京で会いましょう 🙂👍🏽 naïve façade"
    byte_lengths = token_byte_lengths(tokenizer)
    assert byte_lengths[tokenizer.encode(text)].sum() == len(text.encode())
    assert byte_lengths[tokenizer.eos_token_id] == 0


def test_contrast_summary_compares_item_losses():
    """A contrast compares the item losses (mean loss per scored token) of two sources, without the items with no scored
    token: the gap of the means, its standard error, and with pair keys the share of keys at which a's loss is lower."""
    # Item losses: a [2, 1, -, 2] with keys x, y, (z), x; b [3, 0.5, 1] with keys x, y, z.
    a = {"nll": np.array([2.0, 3.0, 0.0, 4.0]), "tokens": np.array([1, 3, 0, 2])}
    b = {"nll": np.array([6.0, 0.5, 1.0]), "tokens": np.array([2, 1, 1])}
    summary = contrast_summary(a, b, (["x", "y", "z", "x"], ["x", "y", "z"]))
    # Sample variances 1/3 (a) and 7/4 (b), over 3 items each.
    assert summary == pytest.approx({"gap": 1.5 - 5 / 3, "gap_se": 5 / 6, "win_rate": 0.5})
    assert contrast_summary(a, b).keys() == {"gap", "gap_se"}
