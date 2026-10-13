from datasets import Dataset
from transformers import AutoTokenizer

from prime_rl.trainer.pretrain.data import StreamPacker, TextDataset, cut_row, tokenize_document


def test_cut_row_segments_documents():
    tokens = list(range(9))
    row = cut_row(tokens, doc_starts=[-2, 3, 5], seq_len=8, document_masking=True)
    assert row["input_ids"] == tokens[:8]
    assert row["target_ids"] == tokens[1:]
    assert row["seq_lens"] == [3, 2, 3]
    assert row["position_ids"] == [0, 1, 2, 0, 1, 0, 1, 2]
    unmasked = cut_row(tokens, doc_starts=[3, 5], seq_len=8, document_masking=False)
    assert unmasked["seq_lens"] == [8]
    assert unmasked["position_ids"] == list(range(8))


def test_stream_packer_keeps_every_token():
    documents = [list(range(start, start + length)) for start, length in [(0, 5), (100, 9), (200, 4), (300, 30)]]
    rows = list(StreamPacker(documents, seq_len=8, document_masking=True))
    stream = [token for document in documents for token in document]
    assert len(rows) == (len(stream) - 1) // 8
    assert [token for row in rows for token in row["input_ids"]] == stream[: len(rows) * 8]
    assert all(row["target_ids"][-1] == next_row["input_ids"][0] for row, next_row in zip(rows, rows[1:]))
    assert rows[0]["seq_lens"] == [5, 3]
    assert rows[1]["seq_lens"] == [6, 2]

    samples = list(StreamPacker(documents, seq_len=8, document_masking=False, rows_per_sample=2))
    assert [sample["seq_lens"] for sample in samples] == [[8, 8]] * len(samples)
    assert samples[0]["input_ids"] == rows[0]["input_ids"] + rows[1]["input_ids"]


def test_stream_packer_resumes_mid_document():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    texts = [f"Document {i}: " + "word " * (7 * i + 3) for i in range(40)]
    dataset = Dataset.from_list([{"text": text} for text in texts])

    def packer():
        return StreamPacker(TextDataset(dataset, tokenizer, shuffle=True, seed=1), seq_len=16, document_masking=True)

    reference = packer()
    reference_rows = iter(reference)
    for _ in range(7):
        next(reference_rows)
    state = reference.state_dict()
    expected = [next(reference_rows) for _ in range(20)]

    resumed = packer()
    resumed.load_state_dict(state)
    assert [row for row, _ in zip(resumed, range(20))] == expected


def test_tokenize_document_appends_eos():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    token_ids = tokenize_document(tokenizer, "Hello world")
    assert token_ids[-1] == tokenizer.eos_token_id
    assert token_ids.count(tokenizer.eos_token_id) == 1
