import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import prime_rl.trainer.pretrain.data as pretrain_data
from prime_rl.configs.pretrain import PretrainDataConfig
from prime_rl.trainer.data import setup_dataloader
from prime_rl.trainer.pretrain.data import (
    TokenShards,
    detect_separator,
    document_edges,
    permute,
    setup_token_dataset,
    warn_if_no_document_separator,
)
from prime_rl.utils.bin_packing import fill_rows


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
        source=[{"name": name, "path": tmp_path / name, **settings} for name, settings in sources.items()],
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
    units in another order. Without document masking, a row is one sequence."""
    lengths = [*LENGTHS, 4 * 4 * 16 - sum(LENGTHS)]
    write_source(tmp_path / "a", lengths)
    write_shard(tmp_path / "a" / "02.ds", [[], []])
    dataset = setup_token_dataset(data_config(tmp_path, {"a": {}}, seed=3), dp_rank=0, dp_size=1)
    rows = take(dataset, 4 * 4)
    starts = np.cumsum([0, *lengths])[:-1]
    within = {(t, t + 1) for start, n in zip(starts, lengths) for t in range(start, start + n - 1)}
    ends, firsts = {s + n - 1 for s, n in zip(starts, lengths) if n}, {s for s, n in zip(starts, lengths) if n}
    scored, other = Counter(), []
    for row in rows:
        assert len(row["input_ids"]) == sum(row["seq_lens"]) == 16
        assert row["position_ids"].tolist() == [p for n in row["seq_lens"] for p in range(n)]
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
    # Without document masking, a row is one sequence.
    config = data_config(tmp_path, {"a": {}}, document_masking=False)
    row = take(setup_token_dataset(config, dp_rank=0, dp_size=1), 1)[0]
    assert row["seq_lens"] == [16] and row["position_ids"].tolist() == list(range(16))


def test_every_step_holds_the_mixture(tmp_path):
    """Without weights, a and b get their natural shares, by their tokens; with weights, their weight over the sum of
    the weights (0 stops b). After every step, every source's tokens are within seq_len of its share, the rest of a
    unit cut at a step's end included."""
    write_source(tmp_path / "a", LENGTHS * 20)
    write_source(tmp_path / "b", [6, 30, 9, 2, 18, 4] * 30, 10**6)
    for weights in ((None, None), (1.0, 3.0), (1.0, 0.0)):
        sources = {name: {} if weight is None else {"weight": weight} for name, weight in zip("ab", weights)}
        sampler = setup_token_dataset(data_config(tmp_path, sources, batch_size=8), dp_rank=0, dp_size=1).sampler
        for step, cursor in enumerate(plan(sampler, 20)[1]):
            assert np.abs(np.array(cursor["tokens"]) - sampler.weights * (step + 1) * 128).max() <= 16


def test_repetitions_read_a_subset_that_many_times(tmp_path):
    """With 2.5 repetitions over 12 steps, a reads a subset of a fifth of its tokens (weight 0.5): every token of it
    two or three times, no other token, and exactly its share of the run. A larger budget's subset holds the smaller
    one's. The metrics count tokens, unique tokens, repetitions and the mean piece length in a step's rows."""
    write_source(tmp_path / "a", [5, 40, 7, 16, 17, 3, 33, 9, 12, 20, 27, 11] * 10)
    write_shard(tmp_path / "b" / "00.ds", consecutive_documents([8] * 200, 10**6))
    sources = {"a": {"weight": 0.5, "repetitions": 2.5}, "b": {"weight": 0.5}}

    def read(max_steps: int) -> tuple[Counter, object, dict]:
        sampler = setup_token_dataset(
            data_config(tmp_path, sources, batch_size=8), max_steps=max_steps, dp_rank=0, dp_size=1
        ).sampler
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
    """A subset larger than its source fails at startup. A resume fails when the seed, the number of workers, a source's
    subset (max_steps with repetitions) or a source's files differ, naming what changed; max_steps may change without
    repetitions, and a source may move to another folder."""
    write_shard(tmp_path / "a" / "00.ds", consecutive_documents([8] * 100))
    with pytest.raises(ValueError, match="needs 1,600 unique tokens but has 800"):
        setup_token_dataset(
            data_config(tmp_path, {"a": {"repetitions": 1.0}}, batch_size=10), max_steps=10, dp_rank=0, dp_size=1
        )
    state = setup_token_dataset(data_config(tmp_path, {"a": {}}), max_steps=10, dp_rank=0, dp_size=1).state_dict()
    setup_token_dataset(data_config(tmp_path, {"a": {}}), max_steps=20, dp_rank=0, dp_size=1).load_state_dict(state)
    shutil.copytree(tmp_path / "a", tmp_path / "moved" / "a")
    moved = data_config(tmp_path / "moved", {"a": {}})
    setup_token_dataset(moved, max_steps=10, dp_rank=0, dp_size=1).load_state_dict(state)
    with pytest.raises(ValueError, match="seed 0 -> 1; num_workers 1 -> 2"):
        setup_token_dataset(
            data_config(tmp_path, {"a": {}}, seed=1, num_workers=2), max_steps=10, dp_rank=0, dp_size=1
        ).load_state_dict(state)
    state = setup_token_dataset(
        data_config(tmp_path, {"a": {"repetitions": 2.0}}), max_steps=10, dp_rank=0, dp_size=1
    ).state_dict()
    with pytest.raises(ValueError, match="Source a: subset 320 tokens -> 512 tokens"):
        setup_token_dataset(
            data_config(tmp_path, {"a": {"repetitions": 2.0}}), max_steps=16, dp_rank=0, dp_size=1
        ).load_state_dict(state)
    write_shard(tmp_path / "a" / "01.ds", consecutive_documents([8] * 10, 800))
    with pytest.raises(ValueError, match="Source a: its files"):
        setup_token_dataset(
            data_config(tmp_path, {"a": {"repetitions": 2.0}}), max_steps=10, dp_rank=0, dp_size=1
        ).load_state_dict(state)


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
    sampler = setup_token_dataset(config, dp_rank=0, dp_size=1).sampler
    cursor, wrap = sampler.start(), None
    while wrap is None:
        cursor = sampler.plan(cursor)[1]
        wrap = cursor["rows"] // config.batch_size if cursor["units"][1] >= sampler.num_units[1] else None
    uninterrupted, _ = global_rows(config, gpus_before, wrap + 3)
    for stop in (3, wrap):
        before, states = global_rows(config, gpus_before, stop)
        after, _ = global_rows(config, gpus_after, wrap + 3 - stop, states)
        assert before + after == uninterrupted


def read_by_source(steps: list[list]) -> dict[int, list[int]]:
    """The tokens in the rows of ``steps`` by source: a's are below 10**6, b's below 2 * 10**6, then c's."""
    read = defaultdict(list)
    for token in (token for rows in steps for row in rows for token in row):
        read[token // 10**6].append(token)
    return read


def first_in_order(sampler, source: int, num_tokens: int) -> list[int]:
    """The source's first ``num_tokens`` tokens in the order of its first pass."""
    units = sampler.locate(np.full(sampler.num_units[source], source), np.arange(sampler.num_units[source]))
    tokens = [token for _, file, begin, end, _, _ in units for token in sampler.shards.tokens(file)[begin:end]]
    return tokens[:num_tokens]


def test_resume_with_another_mix_reads_on(tmp_path):
    """Resumed with other weights and a larger batch, every source reads on in its order: no unit twice, none skipped.
    From the change on, every step holds the new mixture (the source that is behind does not catch up), and the
    metrics count every source's tokens since the start."""
    write_source(tmp_path / "a", LENGTHS * 10)
    write_source(tmp_path / "b", [6, 30, 9, 2, 18, 4] * 40, 10**6)
    settings = {"micro_batch_size": 1, "num_workers": 2}
    old = data_config(tmp_path, {"a": {"weight": 0.5}, "b": {"weight": 0.5}}, batch_size=8, **settings)
    new = data_config(tmp_path, {"a": {"weight": 0.2}, "b": {"weight": 0.8}}, batch_size=16, **settings)
    before, states = global_rows(old, 2, 4)
    after, states = global_rows(new, 2, 4, states)
    sampler = setup_token_dataset(new, dp_rank=0, dp_size=1).sampler
    read = read_by_source(before + after)
    for source in (0, 1):
        assert sorted(read[source]) == sorted(first_in_order(sampler, source, len(read[source])))
    for steps in range(1, 5):
        since = read_by_source(after[:steps])
        assert all(abs(len(since[s]) - sampler.weights[s] * steps * 16 * 16) <= 16 for s in (0, 1))
    state = next(iter(states[0]["_snapshot"]["_worker_snapshots"].values()))["dataset_state"]["dataset"]
    metrics = sampler.progress(state["cursor"])
    assert [metrics["progress/a/num_tokens"], metrics["progress/b/num_tokens"]] == [len(read[0]), len(read[1])]
    assert state["step"] == 4 * 8 + 4 * 16


def test_resume_adds_and_removes_sources(tmp_path):
    """Resumed without b (the first source, so a's files move) and with c: a reads on in its order, c from its start,
    and b no more, whether the unit cut at the step's end was a's or b's."""
    write_source(tmp_path / "a", LENGTHS * 10)
    write_source(tmp_path / "b", [6, 30, 9, 2, 18, 4] * 40, 10**6)
    write_source(tmp_path / "c", [3, 21, 8, 13] * 40, 2 * 10**6)
    settings = {"micro_batch_size": 1, "num_workers": 2, "batch_size": 8}
    old = data_config(tmp_path, {"b": {"weight": 0.5}, "a": {"weight": 0.5}}, **settings)
    new = data_config(tmp_path, {"a": {"weight": 0.5}, "c": {"weight": 0.5}}, **settings)
    sampler = setup_token_dataset(old, dp_rank=0, dp_size=1).sampler
    cursor, stops = sampler.start(), {}
    while len(stops) < 2:
        cursor = sampler.plan(cursor)[1]
        if cursor["rest"]:
            stops.setdefault(cursor["rest"][0], cursor["rows"] // 8)
    sampler = setup_token_dataset(new, dp_rank=0, dp_size=1).sampler
    for stop in stops.values():
        before, states = global_rows(old, 2, stop)
        after, _ = global_rows(new, 2, 4, states)
        read = read_by_source(before + after)
        assert 1 not in read_by_source(after)
        assert sorted(read[0]) == sorted(first_in_order(sampler, 0, len(read[0])))
        assert sorted(read[2]) == sorted(first_in_order(sampler, 1, len(read[2])))


def test_eod_mask_loss(tmp_path):
    """With the separator, an EOS input has no loss and every other position keeps its loss, and the last word of a
    document still learns to predict EOS."""
    eos = 999
    words = consecutive_documents([2, 6, 1, 11, 3, 4, 9, 5, 2, 7] * 6)
    write_shard(tmp_path / "a" / "00.ds", [[*document, eos] for document in words])
    config = data_config(tmp_path, {"a": {}})
    off = take(setup_token_dataset(config, dp_rank=0, dp_size=1), 40)
    masked_dataset = setup_token_dataset(config, dp_rank=0, dp_size=1)
    masked_dataset.eod_mask_token = eos
    on = take(masked_dataset, 40)
    eos_inputs = word_to_eos = 0
    for plain, masked in zip(off, on):
        assert all(np.array_equal(plain[k], masked[k]) for k in ("input_ids", "target_ids", "position_ids", "seq_lens"))
        for token, target, before, after in zip(
            masked["input_ids"], masked["target_ids"], plain["loss_mask"], masked["loss_mask"]
        ):
            assert after == (before and token != eos)
            eos_inputs += before and token == eos
            if before and target == eos and token != eos:
                word_to_eos += 1
    assert eos_inputs > 0 and word_to_eos > 0


def test_long_documents_are_cached(tmp_path):
    """The table of long documents is cached and read back, and a changed folder gets a new one."""
    write_source(tmp_path / "a", LENGTHS)
    config = data_config(tmp_path, {"a": {}})
    for _ in range(2):
        sampler = setup_token_dataset(config, cache_dir=tmp_path / "cache", dp_rank=0, dp_size=1).sampler
        assert (sampler.long_documents.tolist(), sampler.pieces_before.tolist()) == (
            [1, 5, 7, 11, 13],
            [0, 2, 3, 5, 6, 7],
        )
    write_source(tmp_path / "a", [40, *LENGTHS[1:], 3])
    changed = setup_token_dataset(config, cache_dir=tmp_path / "cache", dp_rank=0, dp_size=1).sampler
    assert changed.long_documents.tolist()[0] == 0


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
