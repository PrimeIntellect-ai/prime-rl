from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_MODELS, custom_causal_lm_from_config
from prime_rl.trainer.models.inkling2 import Inkling2TextConfig
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopTextConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.rl.warmup import synthetic_micro_batch, warmup_micro_batches, warmup_routed_experts
from prime_rl.trainer.vlm import get_language_model
from tests.unit.train.models.test_scratch_init import tiny_config


@pytest.mark.parametrize("seq_len,multiple", [(131072, 4), (4096, 1), (65536, 256), (100, 32)])
def test_warmup_micro_batches(seq_len: int, multiple: int) -> None:
    full, near_full, long = warmup_micro_batches(seq_len, multiple)
    assert sum(full) == -(-seq_len // multiple) * multiple
    assert sum(near_full) == sum(long) == sum(full) - multiple
    assert max(full) < sum(full) // 8
    assert max(long) == sum(long) // 2
    assert all(doc_len > 0 for doc_len in full + near_full + long)


@pytest.mark.parametrize("score_centering", [False, True])
def test_synthetic_micro_batch(score_centering: bool) -> None:
    model = SimpleNamespace(lm_head=SimpleNamespace(num_logits=1000))
    loss = SimpleNamespace(type="icepop", score_centering=score_centering, score_centering_topk=4)
    config = SimpleNamespace(enable_router_replay=False, loss=loss)
    doc_lens = [5, 3, 8]
    micro_batch = synthetic_micro_batch(doc_lens, model, config, torch.Generator().manual_seed(0))

    assert micro_batch["input_ids"].shape == (1, 16)
    assert int(micro_batch["input_ids"].max()) < 1000
    assert micro_batch["position_ids"][0].tolist() == [*range(5), *range(3), *range(8)]
    assert micro_batch["seq_lens"].tolist() == doc_lens
    assert not micro_batch["loss_mask"].any()
    if not score_centering:
        assert micro_batch["support_offsets"] is None
        return
    offsets = micro_batch["support_offsets"]
    assert offsets.tolist() == list(range(0, 16 * 4 + 1, 4))
    rows = micro_batch["support_ids"].view(16, 4)
    assert all(len(set(row.tolist())) == 4 for row in rows)
    assert int(rows.max()) < 1000
    assert (micro_batch["support_logprobs"] <= 0).all()
    assert not micro_batch["support_replayed"].any()


def tiny_trainer_model(config_cls, **text_config):
    """A tiny model of the family on the meta device, with the trainer's LM head."""
    config = tiny_config(config_cls)
    for name, value in text_config.items():
        setattr(config.get_text_config(), name, value)
    with torch.device("meta"):
        model = custom_causal_lm_from_config(config, attn_implementation="flash_attention_3")
    inject_prime_lm_head(model)
    return model


@pytest.mark.parametrize(
    "config_cls", [config_cls for config_cls, _ in _CUSTOM_CAUSAL_LM_MODELS], ids=lambda cls: cls.model_type
)
def test_lm_head_counts_the_vocabulary_without_padding(config_cls):
    model = tiny_trainer_model(config_cls)
    assert model.lm_head.num_logits == model.config.get_text_config().vocab_size


@pytest.mark.parametrize("config_cls", [Inkling2TextConfig, Inkling2LoopTextConfig], ids=lambda cls: cls.model_type)
def test_synthetic_ids_stay_among_the_logits_of_a_padded_unembedding(config_cls):
    """Inkling 2 checkpoints pad the unembedding to the embedding's rows. Inkling 2 Small's head keeps the padding
    rows and leaves them out of its projection; the Mini's head holds the logit rows alone. Either way the head
    gives `unpadded_vocab_size` logits, and a token or support id past them has no logit."""
    model = tiny_trainer_model(config_cls, unpadded_vocab_size=100)
    assert model.config.vocab_size == model.get_input_embeddings().num_embeddings == 128
    assert model.lm_head.num_logits == 100
    head = model.lm_head.to_empty(device="cpu")
    assert head(torch.zeros(1, 2, head.in_features))["logits"].shape == (1, 2, 100)

    loss = SimpleNamespace(type="icepop", score_centering=True, score_centering_topk=4)
    config = SimpleNamespace(enable_router_replay=False, loss=loss)
    micro_batch = synthetic_micro_batch([200, 56], model, config, torch.Generator().manual_seed(0))
    assert 90 <= int(micro_batch["input_ids"].max()) < 100
    assert 90 <= int(micro_batch["support_ids"].max()) < 100


@pytest.mark.parametrize(
    "config_cls,num_layers,num_layer_visits",
    [(Inkling2TextConfig, 48, 48), (Inkling2LoopTextConfig, 24, 48)],
    ids=lambda value: getattr(value, "model_type", value),
)
def test_warmup_routing_has_one_row_per_layer_visit(config_cls, num_layers: int, num_layer_visits: int):
    """Inkling 2 Small runs each layer once. The Mini runs layers 6 to 17 three times and replays a row per visit."""
    model = tiny_trainer_model(config_cls)
    assert len(get_language_model(model).layers) == num_layers
    routed_experts = warmup_routed_experts(model, 5, torch.Generator().manual_seed(0))
    assert routed_experts.shape == (1, 5, num_layer_visits, 2)
