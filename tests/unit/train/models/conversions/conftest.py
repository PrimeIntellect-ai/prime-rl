import copy
import torch


def assert_state_dicts_equal(a: dict, b: dict, label: str = "") -> None:
    """Same keys, identical values (and shapes)."""
    ak, bk = set(a), set(b)
    assert ak == bk, f"{label}: key mismatch\n  only in A: {sorted(ak - bk)}\n  only in B: {sorted(bk - ak)}"
    for k in a:
        assert a[k].shape == b[k].shape, f"{label}: shape mismatch for {k}: {a[k].shape} vs {b[k].shape}"
        assert torch.equal(a[k], b[k]), f"{label}: value mismatch for {k}"


def clone(sd: dict) -> dict:
    return {k: v.clone() for k, v in sd.items()}
