from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import prime_rl.trainer.pretrain.data as pretrain_data
from prime_rl.configs.pretrain import DatatroveDataConfig, PretrainDataConfig
from prime_rl.trainer.data import setup_dataloader
from prime_rl.trainer.pretrain.data import (
    TokenShards,
    detect_separator,
    document_edges,
    permute,
    setup_document_dataset,
    setup_token_dataset,
    warn_if_no_document_separator,
)
from prime_rl.utils.bin_packing import fill_rows

# Documents of the pool, in two folders, with an empty one. A token's id is its position in the pool.
DOCUMENTS = [[range(0, 10), range(10, 30), range(30, 30), range(30, 60)], [range(60, 103)]]


@pytest.fixture
def folders(tmp_path: Path) -> list[Path]:
    folders = []
    for index, documents in enumerate(DOCUMENTS):
        folder = tmp_path / str(index)
        folder.mkdir()
        np.array([token for document in documents for token in document], dtype="<u4").tofile(folder / "00.ds")
        np.cumsum([len(document) for document in documents], dtype="<u8").tofile(folder / "00.ds.index")
        (folder / "00.ds.metadata").write_text("tokenizer|4\n")
        folders.append(folder)
    return folders


def write_shard(path: Path, documents: list[list[int]]) -> None:
    """A datatrove DocumentTokenizer shard: uint32 tokens, uint64 cumulative document ends, metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path)
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(f"{path}.index")
    Path(f"{path}.metadata").write_text("tokenizer|4\n0\n0")


def consecutive_documents(lengths: list[int], first: int = 0) -> list[list[int]]:
    """Documents of the given lengths whose tokens are their positions in the stream, from ``first`` on."""
    starts = (first + np.cumsum([0, *lengths])).tolist()
    return [list(range(start, start + length)) for start, length in zip(starts, lengths)]


def data_config(tmp_path: Path, sources: dict[str, dict], **data) -> PretrainDataConfig:
    """Sources of the folders ``tmp_path / name``, with their settings (weight, repetitions)."""
    return PretrainDataConfig(
        sources=[{"name": name, "path": tmp_path / name, **settings} for name, settings in sources.items()],
        **{"seq_len": 16, "batch_size": 4, **data},
    )


def take(dataset, num_rows: int) -> list[dict]:
    rows = iter(dataset)
    return [next(rows) for _ in range(num_rows)]


def plan(sampler, steps: int) -> tuple[list, list[dict]]:
    """The first ``steps`` steps and the cursors after them."""
    planned, cursors, cursor = [], [], sampler.start()
    for _ in range(steps):
        step, cursor = sampler.plan(cursor)
        planned.append(step)
        cursors.append(cursor)
    return planned, cursors


# Long documents (cut into pieces), an empty one, and two files.
LENGTHS = [5, 40, 0, 7, 16, 17, 3, 33, 1, 9, 12, 20, 2, 27, 0, 11]


def write_source(folder: Path, lengths: list[int], first: int = 0) -> None:
    documents = consecutive_documents(lengths, first)
    write_shard(folder / "00.ds", documents[: len(documents) // 2])
    write_shard(folder / "01.ds", documents[len(documents) // 2 :])


def test_rows_keep_every_prediction_of_a_pass(tmp_path):
    """The source holds exactly 4 steps of tokens (and an empty file). Its rows are full, positions restart at every
    segment, and they score every prediction within a document exactly once; the other scored targets are document
    starts after document ends. The pieces of the long documents land in several steps, and the next pass reads the
    units in another order."""
    lengths = [*LENGTHS, 4 * 4 * 16 - sum(LENGTHS)]
    write_source(tmp_path / "a", lengths)
    write_shard(tmp_path / "a" / "02.ds", [[], []])
    dataset = setup_token_dataset(data_config(tmp_path, {"a": {}}, seed=3))
    rows = take(dataset, 4 * 4)
    starts = np.cumsum([0, *lengths])[:-1]
    within = {(t, t + 1) for start, n in zip(starts, lengths) for t in range(start, start + n - 1)}
    ends, firsts = {s + n - 1 for s, n in zip(starts, lengths) if n}, {s for s, n in zip(starts, lengths) if n}
    scored, other = Counter(), []
    for row in rows:
        assert len(row["input_ids"]) == sum(row["seq_lens"]) == 16
        assert row["position_ids"] == [p for n in row["seq_lens"] for p in range(n)]
        for token, target, score in zip(row["input_ids"], row["target_ids"], row["loss_mask"]):
            if score and (token, target) in within:
                scored[token, target] += 1
            elif score:
                other.append((token, target))
    assert set(scored) == within and set(scored.values()) == {1}
    assert all(token in ends and target in firsts for token, target in other)
    # The 40-token document, tokens 5-44, is cut into 3 pieces.
    steps = {i // 4 for i, row in enumerate(rows) if any(5 <= token < 45 for token in row["input_ids"])}
    assert len(steps) > 1
    planned = plan(dataset.sampler, 8)[0]
    passes = [
        [
            (file, t)
            for step in planned[k : k + 4]
            for _, file, begin, end, _, _ in step.units
            for t in range(begin, end)
        ]
        for k in (0, 4)
    ]
    assert passes[0] != passes[1] and sorted(passes[0]) == sorted(passes[1])


def test_every_step_holds_the_mixture(tmp_path):
    """Without weights, a and b get their natural shares, by their tokens; with weights, theirs, and from step 11 on a
    phase's, which stops b. After every step, every source's tokens in the phase are within seq_len of its share,
    the rest of a unit cut at a step's end included."""
    write_source(tmp_path / "a", LENGTHS * 20)
    write_source(tmp_path / "b", [6, 30, 9, 2, 18, 4] * 30, 10**6)
    natural = setup_token_dataset(data_config(tmp_path, {"a": {}, "b": {}}, batch_size=8)).sampler
    shares = natural.num_tokens / natural.num_tokens.sum()
    for step, cursor in enumerate(plan(natural, 20)[1]):
        assert np.abs(np.array(cursor["phase_tokens"]) - shares * (step + 1) * 128).max() <= 16
    phases = [{"start_step": 11, "weights": {"a": 1.0, "b": 0.0}}]
    config = data_config(tmp_path, {"a": {"weight": 0.25}, "b": {"weight": 0.75}}, phases=phases, batch_size=8)
    for step, cursor in enumerate(plan(setup_token_dataset(config).sampler, 20)[1]):
        steps, shares = (step + 1, [0.25, 0.75]) if step < 10 else (step - 9, [1.0, 0.0])
        assert np.abs(np.array(cursor["phase_tokens"]) - np.array(shares) * steps * 128).max() <= 16


def test_repetitions_read_a_subset_that_many_times(tmp_path):
    """With 2.5 repetitions over 12 steps, a reads a subset of a fifth of its tokens (weight 0.5): every token of it
    two or three times, no other token, and exactly its share of the run. A larger budget's subset holds the smaller
    one's. The metrics count tokens, unique tokens, repetitions and the mean piece length in a step's rows."""
    write_source(tmp_path / "a", [5, 40, 7, 16, 17, 3, 33, 9, 12, 20, 27, 11] * 10)
    write_shard(tmp_path / "b" / "00.ds", consecutive_documents([8] * 200, 10**6))
    sources = {"a": {"weight": 0.5, "repetitions": 2.5}, "b": {"weight": 0.5}}

    def read(max_steps: int) -> tuple[Counter, object, dict]:
        sampler = setup_token_dataset(data_config(tmp_path, sources, batch_size=8), max_steps=max_steps).sampler
        planned, cursors = plan(sampler, max_steps)
        tokens = Counter(
            (file, t)
            for step in planned
            for source, file, begin, end, _, _ in step.units
            if source == 0
            for t in range(begin, end)
        )
        return tokens, sampler, cursors[-1]

    tokens, sampler, cursor = read(12)
    assert sampler.set_tokens[0] >= 12 * 128 * 0.5 / 2.5 > sampler.set_tokens[0] - 16
    assert len(tokens) == sampler.set_tokens[0] and set(tokens.values()) == {2, 3}
    assert abs(cursor["tokens"][0] - 12 * 128 * 0.5) <= 16
    metrics = sampler.progress(cursor)
    assert metrics["progress/a/unique_tokens"] == sampler.set_tokens[0]
    assert metrics["progress/a/repetitions"] == cursor["tokens"][0] / sampler.set_tokens[0]
    assert metrics["progress/mean_segment_length"] == 128 / sum(cursor["pieces"])
    assert set(read(6)[0]) <= set(read(24)[0])


def test_subsets_and_resumes_fail_clearly(tmp_path):
    """A subset larger than its source fails at startup. A resume fails when the data settings differ, except for
    max_steps when no source has repetitions."""
    write_shard(tmp_path / "a" / "00.ds", consecutive_documents([8] * 100))
    with pytest.raises(ValueError, match="needs 1,600 unique tokens but has 800"):
        setup_token_dataset(data_config(tmp_path, {"a": {"repetitions": 1.0}}, batch_size=10), max_steps=10)
    state = setup_token_dataset(data_config(tmp_path, {"a": {}}), max_steps=10).state_dict()
    setup_token_dataset(data_config(tmp_path, {"a": {}}), max_steps=20).load_state_dict(state)
    with pytest.raises(ValueError, match="in seed"):
        setup_token_dataset(data_config(tmp_path, {"a": {}}, seed=1), max_steps=10).load_state_dict(state)
    state = setup_token_dataset(data_config(tmp_path, {"a": {"repetitions": 2.0}}), max_steps=10).state_dict()
    with pytest.raises(ValueError, match="max_steps"):
        setup_token_dataset(data_config(tmp_path, {"a": {"repetitions": 2.0}}), max_steps=11).load_state_dict(state)


def global_rows(config: PretrainDataConfig, gpus: int, steps: int, states=None) -> tuple[list[list], list[dict]]:
    """Every step's rows over all GPUs (sorted), and every GPU's dataloader state after the last step, with two
    dataloader workers per GPU."""
    rows, new_states = [[] for _ in range(steps)], []
    for rank in range(gpus):
        loader = setup_dataloader(setup_token_dataset(config, dp_rank=rank, dp_size=gpus), config, cp=1)
        if states is not None:
            # As the checkpoint manager: a rank the saved run did not have loads rank 0's state.
            loader.load_state_dict(states[rank] if rank < len(states) else states[0])
        batches = iter(loader)
        for step in range(steps):
            for _ in range(config.batch_size // (gpus * config.micro_batch_size)):
                batch = next(batches)
                rows[step] += batch["input_ids"].view(-1, config.seq_len).tolist()
        new_states.append(loader.state_dict())
    return [sorted(step_rows) for step_rows in rows], new_states


@pytest.mark.parametrize(("gpus_before", "gpus_after"), [(2, 2), (4, 2), (2, 4)])
def test_resume_continues_the_stream(tmp_path, gpus_before, gpus_after):
    """A run stopped after step 3 (mid-pass) or after the step where b starts its second pass, and resumed, also on
    another number of GPUs, reads the rows of an uninterrupted run."""
    write_source(tmp_path / "a", LENGTHS)
    write_source(tmp_path / "b", [6, 30, 9, 2, 18, 4], 10**6)
    sources = {"a": {"weight": 0.4}, "b": {"weight": 0.6}}
    config = data_config(tmp_path, sources, micro_batch_size=1, num_workers=2, batch_size=8)
    sampler = setup_token_dataset(config).sampler
    cursor, wrap = sampler.start(), None
    while wrap is None:
        cursor = sampler.plan(cursor)[1]
        wrap = cursor["step"] if cursor["units"][1] >= sampler.num_units[1] else None
    uninterrupted, _ = global_rows(config, gpus_before, wrap + 3)
    for stop in (3, wrap):
        before, states = global_rows(config, gpus_before, stop)
        after, _ = global_rows(config, gpus_after, wrap + 3 - stop, states)
        assert before + after == uninterrupted


def test_eod_mask_loss(tmp_path):
    """With the separator, an EOS input has no loss and every other position keeps its loss, and the last word of a
    document still learns to predict EOS."""
    eos = 999
    words = consecutive_documents([2, 6, 1, 11, 3, 4, 9, 5, 2, 7] * 6)
    write_shard(tmp_path / "a" / "00.ds", [[*document, eos] for document in words])
    config = data_config(tmp_path, {"a": {}})
    off = take(setup_token_dataset(config), 40)
    on = take(setup_token_dataset(config, eod_mask_token=eos), 40)
    eos_inputs = word_to_eos = 0
    for plain, masked in zip(off, on):
        assert {k: v for k, v in plain.items() if k != "loss_mask"} == {
            k: v for k, v in masked.items() if k != "loss_mask"
        }
        for token, target, before, after in zip(
            masked["input_ids"], masked["target_ids"], plain["loss_mask"], masked["loss_mask"]
        ):
            assert after == (before and token != eos)
            eos_inputs += before and token == eos
            if before and target == eos and token != eos:
                word_to_eos += 1
    assert eos_inputs > 0 and word_to_eos > 0


def test_long_documents_are_cached(tmp_path):
    """A second start maps the table of long documents from the cache; a changed folder gets a new one."""
    write_source(tmp_path / "a", LENGTHS)
    config = data_config(tmp_path, {"a": {}})
    sampler = setup_token_dataset(config, cache_dir=tmp_path / "cache").sampler
    assert (sampler.long_documents.tolist(), sampler.pieces_before.tolist()) == ([1, 5, 7, 11, 13], [0, 2, 3, 5, 6, 7])
    cached = setup_token_dataset(config, cache_dir=tmp_path / "cache").sampler
    assert isinstance(cached.long_documents, np.memmap) and cached.pieces_before.tolist() == [0, 2, 3, 5, 6, 7]
    write_source(tmp_path / "a", [40, *LENGTHS[1:], 3])
    assert setup_token_dataset(config, cache_dir=tmp_path / "cache").sampler.long_documents.tolist()[0] == 0
    assert len(list((tmp_path / "cache").iterdir())) == 2


def test_fill_rows():
    """Every row is full. Best fit keeps every item whole where cutting their concatenation every 16 tokens would cut
    the second; only an item that fits nowhere is cut."""
    assert fill_rows([9, 16, 7], num_rows=2, capacity=16).tolist() == [[0, 1, 0, 16], [1, 0, 0, 9], [1, 2, 0, 7]]
    pieces = fill_rows([10, 10, 10, 2], num_rows=2, capacity=16)
    assert np.bincount(pieces[:, 0], pieces[:, 3]).tolist() == [16, 16]
    assert sorted(map(tuple, pieces[:, 1:3].tolist())) == [(0, 0), (1, 0), (2, 0), (2, 6), (3, 0)]


def test_unit_order_needs_no_table():
    for n in [1, 2, 7, 250, 4097]:
        assert sorted(permute(np.arange(n), n, key=3).tolist()) == list(range(n))
    assert permute(np.arange(250), 250, key=3).tolist() != permute(np.arange(250), 250, key=4).tolist()
    # Per index domains and keys: two permutations at once.
    both = permute(np.r_[np.arange(7), np.arange(250)], np.r_[[7] * 7, [250] * 250], np.r_[[3] * 7, [4] * 250])
    assert both.tolist() == [*permute(np.arange(7), 7, 3).tolist(), *permute(np.arange(250), 250, 4).tolist()]
    # 10^15 units: any unit is computed directly, without a permutation table.
    units = permute(np.arange(4), 10**15, key=3)
    assert len(set(units.tolist())) == 4 and (units < 10**15).all()


def test_token_shards_keep_few_files_open(tmp_path, monkeypatch):
    """The limit holds for all shards of a process together, e.g. the sources of a mixture."""
    for folder in ("a", "b"):
        for i in range(6):
            write_shard(tmp_path / folder / f"{i:02d}.ds", [list(range(100 * i, 100 * i + 40))])
    monkeypatch.setattr(pretrain_data, "MAX_OPEN_FILES", 2)
    monkeypatch.setattr(pretrain_data, "OPEN_FILES", OrderedDict())
    for folder in ("a", "b"):
        shards = TokenShards([tmp_path / folder])
        for i in range(6):
            assert shards.open(i)[0][:3].tolist() == [100 * i, 100 * i + 1, 100 * i + 2]
    assert len(pretrain_data.OPEN_FILES) == 2


def scored_segments(batch: dict) -> list[tuple[list[int], list[int], list[int]]]:
    """Inputs, targets and positions of every scored segment of a micro-batch row."""
    segments, start = [], 0
    for length in batch["seq_lens"].tolist():
        end = start + length
        if batch["loss_mask"][0, start:end].all():
            segments.append(
                tuple(batch[key][0, start:end].tolist() for key in ("input_ids", "target_ids", "position_ids"))
            )
        else:
            assert not batch["loss_mask"][0, start:end].any()
        start = end
    return segments


@pytest.mark.parametrize("num_documents", [None, 4])
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_validation_scores_every_document_on_its_own(folders: list[Path], world_size: int, num_documents: int | None):
    """The data ranks together score every document (or the first num_documents) as if it were alone: windows of
    seq_len scored tokens, each with positions from 0 and no token before its own first one, every token after the
    document's first scored once. Every rank runs the same number of micro-batches."""
    config = DatatroveDataConfig(paths=folders, seq_len=8, batch_size=2, micro_batch_size=2)
    expected = []
    for document in [list(document) for documents in DOCUMENTS for document in documents][:num_documents]:
        for start in range(0, len(document) - 1, config.seq_len):
            window = document[start : start + config.seq_len + 1]
            expected.append((window[:-1], window[1:], list(range(len(window) - 1))))

    batches_by_rank = []
    for rank in range(world_size):
        dataset = setup_document_dataset(config, dp_rank=rank, dp_size=world_size, num_documents=num_documents)
        batches_by_rank.append(list(setup_dataloader(dataset, config, cp=1)))

    assert len({len(batches) for batches in batches_by_rank}) == 1
    for batches in batches_by_rank:
        assert all(batch["input_ids"].shape == (1, config.seq_len * config.micro_batch_size) for batch in batches)
    segments = [segment for batches in batches_by_rank for batch in batches for segment in scored_segments(batch)]
    assert sorted(segments) == sorted(expected)


def test_separator_check_skips_empty_files_and_documents(tmp_path: Path, monkeypatch):
    """Both folders start with an empty file and an empty document. Only the one whose first real document has no
    separator warns, and only real documents are read: an empty first document does not wrap around to the last
    token."""
    eos = 1
    for name, document in (("separated", [5, 6, eos]), ("run-together", [5, 6, 7])):
        for file, documents in (("00.ds", []), ("01.ds", [[], document])):
            write_shard(tmp_path / name / file, documents)
    warnings = []
    monkeypatch.setattr(pretrain_data, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    tokenizer = SimpleNamespace(all_special_ids=[eos], added_tokens_decoder={}, bos_token_id=None, decode=str)
    shards = TokenShards([tmp_path / "separated", tmp_path / "run-together"])
    warn_if_no_document_separator(shards, tokenizer)
    assert len(warnings) == 1
    assert "run-together" in warnings[0]
    edges = {folder.name: edges for folder, edges in document_edges(shards, 16).items()}
    assert edges == {"separated": ([5], [eos]), "run-together": ([5], [7])}
    with pytest.raises(ValueError, match=r"run-together: .*\(7\)"):
        detect_separator(TokenShards([tmp_path / "run-together"]), tokenizer)


EOS = 999  # every document ends with it, as the tokenizer writes the shards


def test_eod_mask_loss_detects_the_shared_separator(tmp_path):
    """The separator is the special token all checked documents of all folders end with."""
    for name, separator in (("a", EOS), ("b", EOS), ("c", EOS + 1)):
        documents = [[*document, separator] for document in consecutive_documents([3, 5, 2] * 8)]
        write_shard(tmp_path / name / "00.ds", documents)
    # EOS + 1 is special only as an added token, like Qwen's <|endoftext|> once the trainer sets pad to EOS.
    added_tokens_decoder = {EOS + 1: SimpleNamespace(special=True)}
    tokenizer = SimpleNamespace(
        all_special_ids=[EOS], added_tokens_decoder=added_tokens_decoder, bos_token_id=None, decode=str
    )
    assert detect_separator(TokenShards([tmp_path / "a", tmp_path / "b"]), tokenizer) == EOS
    assert detect_separator(TokenShards([tmp_path / "c"]), tokenizer) == EOS + 1
    with pytest.raises(ValueError, match=r"/a: .*\(999\); .*/c: .*\(1000\)"):
        detect_separator(TokenShards([tmp_path / "a", tmp_path / "c"]), tokenizer)


def test_eod_mask_loss_masks_nothing_on_bos_separated_shards(tmp_path, monkeypatch):
    """Documents that start with BOS and end with ordinary tokens have nothing to mask: BOS predicts each document's
    first token."""
    bos = 1
    write_shard(tmp_path / "a" / "00.ds", [[bos, *document] for document in consecutive_documents([3, 5, 2] * 8)])
    tokenizer = SimpleNamespace(all_special_ids=[bos], added_tokens_decoder={}, bos_token_id=bos, decode=str)
    infos = []
    monkeypatch.setattr(pretrain_data, "get_logger", lambda: SimpleNamespace(info=infos.append))
    assert detect_separator(TokenShards([tmp_path / "a"]), tokenizer) is None
    assert len(infos) == 1
