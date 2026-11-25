"""Clef-Flash's strategy renders a record's images as the release and vLLM v0.34.0 do."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.sft.system_one import clef_flash
from prime_rl.trainer.sft.system_one.clef_flash import ClefFlashStrategy
from prime_rl.trainer.sft.system_one.data import render_sample
from prime_rl.trainer.sft.system_one.record import parse_record
from prime_rl.trainer.vlm import setup_processor
from tests.unit.train.models.qwen3_5.clef_flash_release import tiny_config, tokenizer, write_processor
from tests.unit.train.sft.image_urls import EPS, TALL, WIDE, data_url

QUESTIONS = {"q": {"type": "choice", "instructions": "What is it?", "criteria": {"cat": None, "dog": None}}}
COLUMNS = SystemOneColumnsConfig()
IMAGES = [data_url(WIDE), data_url(TALL)]
# An image has a token per merged 2x2 block of 16x16 patches
TOKENS = [3 * 2, 2 * 4]


def row(images=None, state="A photo.") -> dict:
    return {
        "id": "r",
        "state": json.dumps(state),
        "questions": json.dumps(QUESTIONS),
        "label": json.dumps({"q": "dog"}),
        "images": None if images is None else json.dumps(images),
    }


def request(images=None, state="A photo."):
    return parse_record(row(images, state), COLUMNS)[0]


@pytest.fixture(scope="module")
def processor(tmp_path_factory):
    directory = tmp_path_factory.mktemp("release")
    tiny_config().save_pretrained(directory)
    write_processor(directory)
    processor = setup_processor(ModelConfig(name=str(directory)))
    assert processor is not None
    return processor


@pytest.fixture(scope="module")
def strategy(processor) -> ClefFlashStrategy:
    return ClefFlashStrategy(tokenizer(), tiny_config(), processor)


def test_renders_images_before_the_state_as_the_release_does(strategy, processor):
    """The release's joint_schema_model.py runs its processor on the placeholders and a newline, and puts the result
    between the system prompt and the state; so does vLLM."""
    text = strategy.render_request(request(), max_tokens=4096)
    rendered = strategy.render_request(request(IMAGES), max_tokens=4096)
    reference = processor(text=[strategy.IMAGE_PLACEHOLDER * 2 + "\n"], images=[WIDE, TALL], return_tensors="pt")

    media = reference["input_ids"][0].tolist()
    image_token = tiny_config().image_token_id
    assert media.count(image_token) == sum(TOKENS)
    assert rendered["input_ids"] == strategy.prefix + media + text["input_ids"][len(strategy.prefix) :]
    start, end = len(strategy.prefix), len(strategy.prefix) + len(media)
    assert rendered["mm_token_type_ids"][start:end] == reference["mm_token_type_ids"][0].tolist()
    assert not any(rendered["mm_token_type_ids"][:start] + rendered["mm_token_type_ids"][end:])
    assert rendered["mm_kwargs"].keys() == {"pixel_values", "image_grid_thw"}
    for key, value in rendered["mm_kwargs"].items():
        assert torch.equal(value, reference[key]), key
    # The spans move with the text they are in
    assert rendered["read_spans"] == [
        [(s + len(media), e + len(media)) for s, e in span] for span in text["read_spans"]
    ]
    assert "mm_kwargs" not in text and "mm_token_type_ids" not in text


def test_the_state_is_cut_to_fit_with_the_images(strategy, processor, monkeypatch):
    full = strategy.render_request(request(IMAGES), max_tokens=4096)
    monkeypatch.setattr(ClefFlashStrategy, "MAX_LENGTH", len(full["input_ids"]) - 3)
    cut = ClefFlashStrategy(tokenizer(), tiny_config(), processor).render_request(request(IMAGES), max_tokens=4096)
    assert len(cut["input_ids"]) == len(full["input_ids"]) - 3
    assert sum(cut["mm_token_type_ids"]) == sum(TOKENS)
    assert "\nA pho\n\nSCHEMA FIELDS:" in strategy.tokenizer.decode(cut["input_ids"])


def test_skips_a_request_whose_questions_do_not_fit_with_the_images(processor, monkeypatch):
    def render(max_length: int):
        monkeypatch.setattr(ClefFlashStrategy, "MAX_LENGTH", max_length)
        strategy = ClefFlashStrategy(tokenizer(), tiny_config(), processor)
        return strategy.render_request(request(IMAGES, state=""), max_tokens=4096)

    fixed = len(render(16384)["input_ids"])
    assert render(fixed) is not None and render(fixed - 1) is None


def test_renders_what_vllm_renders(strategy):
    """vLLM's prompt after the media, which its renderer expands."""
    strategies = pytest.importorskip("vllm.entrypoints.openai.decisions.strategies")
    from vllm.entrypoints.openai.decisions.question_types import build_question

    context = SimpleNamespace(
        online_renderer=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: strategy.tokenizer)),
        engine_client=SimpleNamespace(model_config=SimpleNamespace(max_model_len=ClefFlashStrategy.MAX_LENGTH)),
    )
    vllm_strategy = strategies.ClefFlashStrategy(context)
    assert strategy.IMAGE_PLACEHOLDER == vllm_strategy.PLACEHOLDERS["image"]
    question = QUESTIONS["q"]
    questions = [build_question("q", question["type"], question["instructions"], question["criteria"], 26)]
    prefix, _ = strategy._render_images(tuple(IMAGES))
    prompt, schema, _ = vllm_strategy.encode(questions, vllm_strategy.render_state("A photo."), prefix)
    rendered = strategy.render_request(request(IMAGES), max_tokens=4096)
    assert rendered["input_ids"] == prompt
    spans = [*schema["question_spans"], *schema["option_spans"]]
    assert rendered["read_spans"] == [[(start + len(prompt), end + len(prompt)) for start, end in spans]]


def test_skips_images_it_cannot_read(strategy, monkeypatch):
    warnings = []
    monkeypatch.setattr(clef_flash, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    assert strategy.render_request(request(["data:image/png;base64,AAAA"]), max_tokens=4096) is None
    # The processor's limit on an image's aspect ratio
    assert strategy.render_request(request([data_url(Image.new("RGB", (4096, 16)))]), max_tokens=4096) is None
    assert strategy.render_request(request([EPS]), max_tokens=4096) is None
    assert len(warnings) == 3
    assert all(warning.startswith("Skipping record 'r': vLLM refuses it: media: ") for warning in warnings)


@pytest.mark.parametrize("images", [None, IMAGES])
def test_skips_a_record_whose_text_holds_the_image_token(strategy, images):
    assert render_sample(strategy, request(images), 4096) is not None
    assert render_sample(strategy, request(images, "A <|image_pad|>."), 4096) is None


def test_a_tokenizer_with_another_image_token_is_refused(processor):
    config = tiny_config()
    config.image_token_id += 1
    with pytest.raises(ValueError, match="is not the model's image token"):
        ClefFlashStrategy(tokenizer(), config, processor)
