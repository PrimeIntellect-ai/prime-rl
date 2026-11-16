from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from prime_rl.configs.supervised import RawValSetConfig
from prime_rl.trainer.val import raw_set, token_byte_lengths

# Documents in two files of one folder, with an empty one. A token's id is its position in the folder.
DOCUMENTS = [[range(0, 10), range(10, 30), range(30, 30), range(30, 60)], [range(60, 103)]]


@pytest.mark.parametrize("num_documents", [None, 4])
def test_raw_set_scores_every_document_on_its_own(tmp_path: Path, num_documents: int | None):
    """A raw set scores every document (or the first num_documents) as if it were alone: windows of seq_len scored
    tokens, none with a token before its own first one, and every token after the document's first scored once."""
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

    val_set = raw_set(RawValSetConfig(name="held-out", path=tmp_path, seq_len=seq_len), num_documents)
    assert [(inputs.tolist(), labels.tolist()) for inputs, labels in val_set.requests] == expected
    assert val_set.items.tolist() == items
    assert val_set.num_items == len(documents)


def test_token_byte_lengths_add_up_to_the_text_bytes():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Tokens holding part of a multi-byte character count their own bytes; <think> is added but not special.
    text = "<think>Grüße aus Zürich!</think> 東京で会いましょう 🙂👍🏽 naïve façade"
    byte_lengths = token_byte_lengths(tokenizer)
    assert byte_lengths[tokenizer.encode(text)].sum() == len(text.encode())
    assert byte_lengths[tokenizer.eos_token_id] == 0
