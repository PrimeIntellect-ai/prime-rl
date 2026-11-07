"""[val] per_subset: each subset validated on its own, in stored order, exactly as a run with that subset alone."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from prime_rl.configs.sft import SFTDataConfig, SFTValConfig
from prime_rl.trainer.sft.data import load_sft_dataset, load_val_subsets


def write_compound(root: Path, sizes: dict[str, int]) -> None:
    """Validation-only subsets in the layout of a compound dataset directory."""
    yaml = "---\nconfigs:\n"
    for name, n in sizes.items():
        (root / name).mkdir(parents=True)
        rows = [{"id": f"{name}-{i}", "messages": [{"role": "assistant", "content": f"{name} {i}"}]} for i in range(n)]
        pq.write_table(pa.Table.from_pylist(rows), root / name / "validation-00000-of-00001.parquet")
        yaml += f"- config_name: {name}\n  data_files:\n  - split: validation\n    path: {name}/validation-*.parquet\n"
    (root / "README.md").write_text(yaml + "---\n")


def val_config(root: Path, subsets: list[str], **kwargs) -> SFTDataConfig:
    return SFTDataConfig(name=str(root), subsets=subsets, splits=["validation"] * len(subsets), shuffle=False, **kwargs)


def test_load_val_subsets_keeps_every_row_once_in_stored_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    sizes = {"a": 3, "b": 8, "c": 5}
    write_compound(tmp_path / "data", sizes)
    subsets = load_val_subsets(val_config(tmp_path / "data", list(sizes)))
    assert list(subsets) == list(sizes)
    for name, n in sizes.items():
        alone = load_sft_dataset(val_config(tmp_path / "data", [name]))
        assert subsets[name]["id"][:] == alone["id"][:] == [f"{name}-{i}" for i in range(n)]
        assert subsets[name]["__subset"][:] == [name] * n


def test_interleaving_the_same_subsets_repeats_rows(tmp_path, monkeypatch):
    """Why per_subset exists: the default interleave oversamples the smaller subsets to the largest one's length."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    sizes = {"a": 3, "b": 8, "c": 5}
    write_compound(tmp_path / "data", sizes)
    mixed = load_sft_dataset(val_config(tmp_path / "data", list(sizes)))
    assert len(mixed) == 3 * 8


@pytest.mark.parametrize("subsets", [None, [], ["a", "a"]])
def test_per_subset_needs_distinct_subsets(tmp_path, subsets):
    data = SFTDataConfig(name=str(tmp_path), subsets=subsets)
    with pytest.raises(ValidationError):
        SFTValConfig(per_subset=True, data=data)


def test_per_subset_refuses_probabilities(tmp_path):
    data = SFTDataConfig(name=str(tmp_path), subsets=["a", "b"], probabilities=[0.5, 0.5])
    with pytest.raises(ValidationError):
        SFTValConfig(per_subset=True, data=data)
