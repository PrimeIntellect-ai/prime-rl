import numpy as np
import pytest

from prime_rl.configs.sft import TokensDataConfig, TokenSourceConfig
from prime_rl.trainer.sft.data import TokenDataset, TokenShards, setup_dataloader


def write_shards(folder, docs_per_file: list[list[list[int]]]) -> list[int]:
    """Write datatrove-format shards and return the token stream."""
    folder.mkdir()
    stream = []
    for i, docs in enumerate(docs_per_file):
        file = folder / f"{i:05d}.ds"
        np.array([t for doc in docs for t in doc], dtype="<u2").tofile(file)
        np.cumsum([len(doc) for doc in docs]).astype("<u8").tofile(f"{file}.index")
        (folder / f"{i:05d}.ds.metadata").write_text("tok|2\n0\n0")
        stream += [t for doc in docs for t in doc]
    return stream


@pytest.fixture
def sources(tmp_path):
    rng = np.random.default_rng(0)

    def docs(n, offset):
        return [list(offset + rng.integers(0, 100, rng.integers(1, 20))) for _ in range(n)]

    write_shards(tmp_path / "a", [docs(30, 0), docs(30, 0)])
    write_shards(tmp_path / "b", [docs(50, 1000)])
    return {name: TokenSourceConfig(path=tmp_path / name) for name in ("a", "b")}


def test_token_shards_read_crosses_files_and_wraps(tmp_path):
    stream = write_shards(tmp_path / "s", [[[1, 2, 3], [4, 5]], [[6, 7, 8, 9]]])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "s"))
    tokens, doc_starts = shards.read(1, 12)
    assert tokens.tolist() == (stream * 2)[1:13]
    # [4, 5] starts at 2, [6, 7, 8, 9] at 4 (next file), [1, 2, 3] at 8 (wrap) and [4, 5] at 11
    assert doc_starts == [2, 4, 8, 11]


def test_token_dataset_resumes_exactly(sources):
    config = TokensDataConfig(
        mixture={
            "sources": sources,
            "phases": [{"weights": {"a": 0.7, "b": 0.3}}, {"start": 0.5, "weights": {"b": 1.0}}],
        },
        seq_len=16,
        batch_size=4,
        micro_batch_size=2,
    )

    def loader():
        dataset = TokenDataset(
            {name: TokenShards(source) for name, source in sources.items()},
            config.mixture.phases,
            seq_len=16,
            batch_size=4,
            max_steps=10,
        )
        return setup_dataloader(dataset, config)

    it = iter(loader())
    batches = [next(it) for _ in range(20)]
    for batch in batches:
        assert batch["seq_lens"].sum() == 32
        starts = np.cumsum([0, *batch["seq_lens"].tolist()[:-1]])
        assert (batch["position_ids"][0, starts] == 0).all()
    # Two micro batches per step: steps 5-9 are in the second phase, which reads only source b
    assert all((batch["input_ids"] >= 1000).all() for batch in batches[10:])

    dataloader = loader()
    it = iter(dataloader)
    for _ in range(3):
        next(it)
    state = dataloader.state_dict()
    resumed = loader()
    resumed.load_state_dict(state)
    for expected, batch in zip(batches[3:], iter(resumed)):
        assert batch["input_ids"].equal(expected["input_ids"]) and batch["position_ids"].equal(expected["position_ids"])
