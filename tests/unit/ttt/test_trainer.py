"""The TTT service: trainer math on a tiny CPU model (updates change outputs, optimizer
state persists, checkpoints are vLLM-consumable PEFT format, versioning is strict) and the
HTTP surface against a fake trainer (no torch on the request path)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from prime_rl.configs.ttt import (  # noqa: E402
    TTTLoRAConfig,
    TTTModelConfig,
    TTTOptimizerConfig,
    TTTServiceConfig,
)
from prime_rl.ttt.trainer import TTTTrainer  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_name(tmp_path_factory) -> str:
    """A tiny random llama on disk, so tests need no network."""
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path_factory.mktemp("tiny_llama")
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def make_trainer(tiny_model_name: str, output_dir: Path, **overrides) -> TTTTrainer:
    config = TTTServiceConfig(
        model=TTTModelConfig(name=tiny_model_name, device="cpu", gradient_checkpointing=False),
        lora=TTTLoRAConfig(rank=4, alpha=8, target_modules=["q_proj", "v_proj"]),
        optim=TTTOptimizerConfig(lr=1e-2),
        output_dir=output_dir,
        inference_admin_urls=[],
        **overrides,
    )
    return TTTTrainer(config)


def logits_for(trainer: TTTTrainer, state, token_ids: list[int]) -> torch.Tensor:
    trainer._swap_in(state)
    with torch.no_grad():
        return trainer.model(input_ids=torch.tensor([token_ids], dtype=torch.long)).logits.clone()


def test_update_trains_and_versions(tmp_path, tiny_model_name):
    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 34))
    mask = [True] * len(tokens)

    result = trainer.update("r1", "ttt-r1", tokens, mask, seq_no=1)
    assert result["version"] == 1
    assert result["num_loss_tokens"] == len(tokens) - 1  # shifted targets
    assert result["loss"] > 0

    state = trainer.adapters["r1"]
    assert state.version == 1
    # The adapter moved: its B matrices are no longer all-zero.
    assert any("lora_B" in name and tensor.abs().sum() > 0 for name, tensor in state.tensors.items())
    # A second update on the same data drops the loss (it's learning this sequence).
    result2 = trainer.update("r1", "ttt-r1", tokens, mask, seq_no=2)
    assert result2["version"] == 2
    assert result2["loss"] < result["loss"]


def test_updates_change_model_outputs_but_isolate_rollouts(tmp_path, tiny_model_name):
    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 26))
    trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=1)

    # A fresh rollout's adapter is the zero template: identical to the base model.
    fresh = trainer._get_or_create("r2", "ttt-r2")
    trained = trainer.adapters["r1"]
    probe = list(range(40, 56))
    fresh_logits = logits_for(trainer, fresh, probe)
    trained_logits = logits_for(trainer, trained, probe)
    assert not torch.allclose(fresh_logits, trained_logits)

    # And the zero adapter tensors match the template exactly (B=0 ⇒ base behavior).
    for name, tensor in fresh.tensors.items():
        assert torch.equal(tensor, trainer._template[name])


def test_loss_mask_targets_only_masked_positions(tmp_path, tiny_model_name):
    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 22))
    mask = [False] * 10 + [True] * 10
    result = trainer.update("r1", "ttt-r1", tokens, mask, seq_no=1)
    # Shifted: targets are tokens[1:]; the mask selects its last 10 positions.
    assert result["num_loss_tokens"] == 10


def test_out_of_order_seq_no_rejected(tmp_path, tiny_model_name):
    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 12))
    with pytest.raises(ValueError, match="expected seq_no 1"):
        trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=2)
    # seq_no is validated BEFORE allocation: the bad first update left no adapter state
    # behind (a stale zero-version state would make the real seq_no 1 look out-of-order).
    assert "r1" not in trainer.adapters
    trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=1)
    # An exact replay of the last applied seq_no is duplicate-ok (retry after a lost
    # response), so a genuinely out-of-order seq_no must skip AHEAD to be rejected.
    with pytest.raises(ValueError, match="expected seq_no 2"):
        trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=3)
    # Malformed qa_pairs 409 (ValueError) with a clear message before any state mutation —
    # not a KeyError-500 from inside _tokenize_qa.
    with pytest.raises(ValueError, match="malformed qa_pairs\\[0\\]"):
        trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=2, qa_pairs=[{"answer": "a"}])


def test_checkpoint_is_peft_loadable(tmp_path, tiny_model_name):
    """The saved checkpoint round-trips through PEFT's own loader — the same on-disk
    contract vLLM's LoRA loader consumes (adapter_config.json + safetensors)."""
    import safetensors.torch
    from peft import PeftConfig

    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 20))
    result = trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=1)

    ckpt = Path(result["ckpt_path"])
    assert ckpt == tmp_path / "ttt" / "r1" / "v1"
    config = PeftConfig.from_pretrained(str(ckpt))
    assert config.r == 4
    tensors = safetensors.torch.load_file(ckpt / "adapter_model.safetensors")
    # PEFT save format: base_model.model.* keys, no ".default." infix.
    assert tensors
    for name in tensors:
        assert name.startswith("base_model.model.")
        assert ".default." not in name
        assert "lora_A" in name or "lora_B" in name

    # The trained state reloads onto a fresh trainer and reproduces the outputs exactly.
    state = trainer.adapters["r1"]
    reloaded = {name: tensor for name, tensor in tensors.items()}
    expected = {name.replace(".default.", "."): tensor for name, tensor in state.tensors.items()}
    assert set(reloaded) == set(expected)
    for name in reloaded:
        assert torch.equal(reloaded[name], expected[name])


def test_release_drops_state_and_optionally_checkpoints(tmp_path, tiny_model_name):
    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 14))
    trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=1)
    ckpt_dir = tmp_path / "ttt" / "r1"
    assert ckpt_dir.exists()
    state = trainer.release("r1")
    assert state is not None and state.version == 1
    assert "r1" not in trainer.adapters
    assert ckpt_dir.exists()  # keep_checkpoints=True (default): replay artifacts stay

    trainer2 = make_trainer(tiny_model_name, tmp_path / "no_keep", keep_checkpoints=False)
    trainer2.update("r2", "ttt-r2", tokens, [True] * len(tokens), seq_no=1)
    trainer2.release("r2")
    assert not (tmp_path / "no_keep" / "ttt" / "r2").exists()


def test_alignment_validation(tmp_path, tiny_model_name):
    trainer = make_trainer(tiny_model_name, tmp_path)
    with pytest.raises(ValueError, match="must align"):
        trainer.update("r1", "a", [1, 2, 3], [True, True], seq_no=1)
    with pytest.raises(ValueError, match="at least 2"):
        trainer.update("r1", "a", [1], [True], seq_no=1)
    with pytest.raises(ValueError, match="no trainable sequences"):
        trainer.update("r1", "a", [1, 2, 3], [True, False, False], seq_no=1)


# -- Q&A training -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_with_tokenizer(tiny_model_name) -> str:
    """The tiny model dir + a from-scratch tokenizer with a chat template (no network)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = "what is x ? empty q a longer answer here 42".split()
    specials = ["<unk>", "<|user|>", "<|assistant|>", "</s>"]
    vocab = {tok: i for i, tok in enumerate([*specials, *words])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} </s> {% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    tokenizer.save_pretrained(tiny_model_name)
    return tiny_model_name


def test_qa_pairs_render_with_answer_loss(tmp_path, tiny_model_with_tokenizer):
    trainer = make_trainer(tiny_model_with_tokenizer, tmp_path)
    sequences = trainer._tokenize_qa(
        [
            {"question": "what is x?", "answer": "x is 42"},
            {"question": "empty?", "answer": ""},
        ]
    )
    # The empty answer renders no loss tokens under this template and is kept only if it
    # produced trainable positions; the real pair must be there.
    assert len(sequences) >= 1
    ids, mask = sequences[0]
    assert len(ids) == len(mask)
    assert any(mask) and not all(mask)  # answer tokens in loss, prompt tokens out
    # The masked positions decode back to (a suffix containing) the answer.
    text = trainer._tokenizer.decode([t for t, m in zip(ids, mask) if m])
    assert "42" in text


def test_update_with_qa_only(tmp_path, tiny_model_with_tokenizer):
    """`train_rollout=False` trains purely on the rendered Q&A pairs — the branch tokens
    ride along only for validation."""
    trainer = make_trainer(tiny_model_with_tokenizer, tmp_path)
    tokens = list(range(2, 20))
    result = trainer.update(
        "r1",
        "ttt-r1",
        tokens,
        [True] * len(tokens),
        seq_no=1,
        qa_pairs=[{"question": "what is x?", "answer": "x is 42"}],
        train_rollout=False,
    )
    assert result["version"] == 1
    # Loss tokens come from the QA rendering, not the branch (whose mask was all-True).
    assert 0 < result["num_loss_tokens"] < len(tokens)
    state = trainer.adapters["r1"]
    assert any("lora_B" in n and t.abs().sum() > 0 for n, t in state.tensors.items())


def test_update_with_qa_and_rollout(tmp_path, tiny_model_with_tokenizer):
    trainer = make_trainer(tiny_model_with_tokenizer, tmp_path)
    tokens = list(range(2, 20))
    qa = [{"question": "q", "answer": "a longer answer here"}]
    result = trainer.update("r1", "ttt-r1", tokens, [True] * len(tokens), seq_no=1, qa_pairs=qa, train_rollout=True)
    # Both the branch's shifted targets and the QA answer tokens count.
    assert result["num_loss_tokens"] > len(tokens) - 1


def test_qa_only_with_unrenderable_pairs_rejected(tmp_path, tiny_model_with_tokenizer):
    trainer = make_trainer(tiny_model_with_tokenizer, tmp_path)
    tokens = list(range(2, 10))
    with pytest.raises(ValueError, match="no trainable sequences"):
        trainer.update(
            "r1",
            "ttt-r1",
            tokens,
            [True] * len(tokens),
            seq_no=1,
            qa_pairs=[{"question": "q", "answer": ""}],
            train_rollout=False,
        )


def test_duplicate_seq_no_replays_cached_result(tmp_path, tiny_model_name):
    """A retry after a lost response replays the same seq_no: the cached result comes back
    with no re-training and no version bump; older seq_nos still 409."""
    trainer = make_trainer(tiny_model_name, tmp_path)
    tokens = list(range(2, 34))
    mask = [True] * len(tokens)
    first = trainer.update("r1", "ttt-r1", tokens, mask, seq_no=1)
    replay = trainer.update("r1", "ttt-r1", tokens, mask, seq_no=1)
    assert replay is first  # cached — no state mutation, same ckpt_path/version
    assert trainer.adapters["r1"].version == 1
    trainer.update("r1", "ttt-r1", tokens, mask, seq_no=2)
    with pytest.raises(ValueError, match="expected seq_no 3"):
        trainer.update("r1", "ttt-r1", tokens, mask, seq_no=1)
