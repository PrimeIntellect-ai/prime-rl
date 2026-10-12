from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from pydantic import ValidationError
from torch import nn

from prime_rl.configs.sft import SFTConfig, SFTObjectiveConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.sft.data import CatDataset, SFTObjectiveDataset, cat_collate
from prime_rl.trainer.sft.objective import ObjectiveInputs, ObjectiveOutputs, SFTObjective, load_objective


class PairObjective:
    """Two sequences per row (the text, then its reverse), read at their last tokens; the row's label travels as its objective_data."""

    def __init__(self, offset: int = 0):
        self.offset = offset

    def make_sample(self, example, renderer):
        ids = [ord(char) + self.offset for char in example["text"]]
        sequences = [ids, ids[::-1]]
        input_ids = [token for sequence in sequences for token in sequence]
        return {
            "input_ids": input_ids,
            "target_ids": [0] * len(input_ids),
            "loss_mask": [i == len(s) - 1 for s in sequences for i in range(len(s))],
            "position_ids": [i for s in sequences for i in range(len(s))],
            "seq_lens": [len(s) for s in sequences],
            "objective_data": {"label": example["label"]},
        }

    def loss(self, inputs: ObjectiveInputs) -> ObjectiveOutputs:
        readout = inputs.hidden_states[inputs.batch["loss_mask"]].sum(-1)
        labels = torch.tensor([data["label"] for data in inputs.batch["objective_data"]], dtype=readout.dtype)
        errors = readout.view(-1, 2).mean(-1) - labels
        return ObjectiveOutputs(
            loss=(errors**2).sum(), count=len(labels), metrics={"abs_error": errors.abs().sum().detach()}
        )


def objective_dataset(texts, seq_len, **kwargs):
    rows = Dataset.from_list([{"text": text, "label": float(i)} for i, text in enumerate(texts)])
    return SFTObjectiveDataset(rows, lambda example: None, PairObjective(), shuffle=False, seq_len=seq_len, **kwargs)


def test_objective_samples_pack_with_their_extras():
    # Samples of 4, 6, 2 and 8 tokens; packs of 10 tokens.
    packs = list(CatDataset(objective_dataset(["ab", "cde", "f", "ghij"], seq_len=10, max_epochs=1), seq_len=10))

    assert [pack["seq_lens"] for pack in packs] == [[4, 6], [2, 8]]
    assert [[data["label"] for data in pack["objective_data"]] for pack in packs] == [[0.0, 1.0], [2.0, 3.0]]
    assert packs[0]["position_ids"] == [0, 1, 0, 1, 0, 1, 2, 0, 1, 2]
    assert sum(packs[0]["loss_mask"]) == 4

    batch = cat_collate([packs[0]])
    assert batch["input_ids"].shape == (1, 10)
    assert batch["objective_data"] == packs[0]["objective_data"]


def test_padded_pack_keeps_one_extra_per_sample():
    (pack,) = CatDataset(objective_dataset(["ab", "c"], seq_len=16, max_epochs=1), seq_len=16)
    assert pack["seq_lens"] == [4, 2 + 10]  # the padding joins the last sample
    assert [data["label"] for data in pack["objective_data"]] == [0.0, 1.0]


def test_overlong_samples_are_skipped():
    samples = list(objective_dataset(["ab", "cdefgh", "i"], seq_len=6, max_epochs=1))
    assert [sample["objective_data"]["label"] for sample in samples] == [0.0, 2.0]


def test_resume_keeps_the_pending_sample_and_its_extras():
    texts = ["ab", "cd", "efg", "h", "ij"]  # 4 + 4 tokens, then "efg" (6) overflows and waits as the pending sample
    reference = list(CatDataset(objective_dataset(texts, seq_len=10, max_epochs=1), seq_len=10))

    packing = CatDataset(objective_dataset(texts, seq_len=10, max_epochs=1), seq_len=10)
    first = next(iter(packing))
    state = packing.state_dict()
    assert state["pending_sample"]["objective_data"] == {"label": 2.0}

    resumed = CatDataset(objective_dataset(texts, seq_len=10, max_epochs=1), seq_len=10)
    resumed.load_state_dict(state)
    assert [first, *resumed] == reference


def test_packs_without_extras_are_unchanged():
    sample = {
        "input_ids": [1, 2],
        "target_ids": [2, 3],
        "loss_mask": [True, True],
        "position_ids": [0, 1],
        "seq_lens": [2],
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }
    pack = next(iter(CatDataset([sample, sample], seq_len=4)))
    assert "objective_data" not in pack
    assert "objective_data" not in cat_collate([pack])


def test_objective_loss_on_a_collated_batch():
    (pack,) = CatDataset(objective_dataset(["ab", "c"], seq_len=8, max_epochs=1), seq_len=8)
    batch = cat_collate([pack])
    hidden_states = torch.ones(1, 8, 3)
    outputs = PairObjective().loss(ObjectiveInputs(model=None, hidden_states=hidden_states, batch=batch))
    assert outputs.count == 2
    assert outputs.metrics["abs_error"].item() == pytest.approx(3 + 2)
    assert outputs.loss.item() == pytest.approx((3 - 0) ** 2 + (3 - 1) ** 2)


def test_load_objective():
    config = SFTObjectiveConfig(import_path=f"{__name__}.PairObjective", kwargs={"offset": 1})
    objective = load_objective(config)
    assert isinstance(objective, SFTObjective) and objective.offset == 1
    with pytest.raises(TypeError, match="must implement"):
        load_objective(SFTObjectiveConfig(import_path="collections.OrderedDict"))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 11, hidden: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.model = self.backbone
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def backbone(self, input_ids, position_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.norm(self.embed(input_ids)))


def test_forward_returns_hidden_states():
    model = TinyCausalLM()
    model.model = nn.Module()  # inject_prime_lm_head expects a backbone module at model.model
    model.model.forward = model.backbone
    inject_prime_lm_head(model)
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[0, 1, 2]])
    seq_lens = torch.tensor([3])

    out = forward(model, input_ids, position_ids, seq_lens=seq_lens, return_hidden_states=True)
    assert out["logits"] is None and out["loss"] is None
    torch.testing.assert_close(out["hidden_states"], model.backbone(input_ids, position_ids).last_hidden_state)

    out = forward(model, input_ids, position_ids, seq_lens=seq_lens)
    assert out["hidden_states"] is None and out["logits"].shape == (1, 3, 11)


def test_objective_config_rejects_unsupported_setups():
    objective = {"import_path": f"{__name__}.PairObjective"}
    assert SFTConfig(objective=objective).objective.import_path.endswith("PairObjective")
    with pytest.raises(ValidationError, match="objective requires data.type='sft'"):
        SFTConfig(objective=objective, data={"type": "fake"})
    with pytest.raises(ValidationError, match="not supported with context parallelism"):
        SFTConfig(objective=objective, model={"cp": 2})
