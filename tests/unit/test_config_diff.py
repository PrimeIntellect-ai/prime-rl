import pytest
from pydantic import BaseModel, Field

from prime_rl.configs.diff import LIVE, LIVE_INSIDE, diff_config, frozen_raw_changes, merge_resumed_config


class Leaf(BaseModel):
    a: int = 1
    b: int = 2


class OtherLeaf(BaseModel):
    a: int = 1


class Source(BaseModel):
    name: str = Field(frozen=True)
    weight: float = 1.0


class Algo(BaseModel):
    kind: str = "grpo"
    penalty: Leaf | None = Field(None, json_schema_extra=LIVE)


class Root(BaseModel):
    live: Leaf = Leaf()
    frozen: Leaf = Field(Leaf(), frozen=True)
    swappable: Leaf | OtherLeaf = Leaf()
    optional: Leaf | None = None
    source: list[Source] = [Source(name="math"), Source(name="code")]
    algo: Algo = Field(Algo(), frozen=True, json_schema_extra=LIVE_INSIDE)
    default_algo: Algo = Field(Algo(), frozen=True)


@pytest.mark.parametrize(
    ("new", "changes", "frozen"),
    [
        (Root(live=Leaf(b=3)), ["live.b"], []),
        (Root(frozen=Leaf(b=3)), [], ["frozen"]),
        (Root(swappable=OtherLeaf()), [], ["swappable"]),
        (Root(optional=Leaf()), [], ["optional"]),
        (Root(source=[Source(name="math", weight=2.0), Source(name="code")]), ["source[0].weight"], []),
        (Root(source=[Source(name="code"), Source(name="math")]), [], ["source[0].name", "source[1].name"]),
        (Root(source=[Source(name="math")]), [], ["source"]),
        (Root(source=[Source(name="math"), Source(name="code"), Source(name="chat")]), [], ["source"]),
        (Root(algo=Algo(penalty=Leaf())), ["algo.penalty"], []),
        (Root(algo=Algo(kind="echo")), [], ["algo"]),
        (Root(algo=Algo(kind="echo", penalty=Leaf())), [], ["algo"]),
        (Root(default_algo=Algo(penalty=Leaf())), [], ["default_algo"]),
    ],
    ids=[
        "live_leaf",
        "frozen_subtree",
        "class_swap",
        "none_to_config",
        "source_reweighted",
        "source_reordered",
        "source_removed",
        "source_added",
        "live_field_in_frozen_subtree",
        "frozen_field_beside_live_field",
        "live_and_frozen_change_together",
        "live_field_below_plain_frozen_field",
    ],
)
def test_diff_config(new, changes, frozen):
    diff = diff_config(Root(), new)

    assert [change.path for change in diff.changes] == changes
    assert diff.frozen == frozen


def test_live_field_in_frozen_subtree_changes_in_place():
    diff = diff_config(Root(algo=Algo(penalty=Leaf())), Root(algo=Algo(penalty=Leaf(b=3))))

    assert [change.path for change in diff.changes] == ["algo.penalty.b"]
    assert diff.frozen == []


@pytest.mark.parametrize(
    ("field", "algo", "frozen"),
    [
        ("algo", {"kind": "grpo", "penalty": {"a": 1, "b": 3}}, []),
        ("algo", {"kind": "echo", "penalty": None}, ["algo"]),
        ("default_algo", {"kind": "grpo", "penalty": {"a": 1, "b": 3}}, ["default_algo"]),
    ],
    ids=["live_field", "frozen_field", "live_field_below_plain_frozen_field"],
)
def test_frozen_raw_changes_skips_live_fields(field, algo, frozen):
    root = Root()
    raw = root.model_dump() | {field: algo}

    assert frozen_raw_changes(root, root.model_dump(), raw) == frozen


A = {"batch_size": 64, "max_staleness": 8, "source": [{"ratio": 1.0}, {"ratio": 1.0}]}
A_LIVE = {"batch_size": 128, "max_staleness": 8, "source": [{"ratio": 2.0}, {"ratio": 1.0}]}


@pytest.mark.parametrize(
    ("launch", "expected"),
    [
        (A, A_LIVE),
        (A | {"batch_size": 32}, A_LIVE | {"batch_size": 32}),
        (A | {"max_staleness": 4}, A_LIVE | {"max_staleness": 4}),
        (A | {"source": [{"ratio": 1.0}, {"ratio": 3.0}]}, A_LIVE | {"source": [{"ratio": 2.0}, {"ratio": 3.0}]}),
        (A | {"source": [{"ratio": 1.0}]}, A_LIVE | {"source": [{"ratio": 1.0}]}),
    ],
    ids=["unchanged", "launch_wins", "both_kept", "source_per_index", "source_count_changed"],
)
def test_merge_resumed_config(launch, expected):
    assert merge_resumed_config(A, A_LIVE, launch) == expected
