"""A System One record's images, through Clef-Flash's strategy into a packed micro-batch."""

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from PIL import Image
from torch import nn

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.data import CatDataset
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.sft.system_one import clef_flash
from prime_rl.trainer.sft.system_one.clef_flash import ClefFlashStrategy
from prime_rl.trainer.sft.system_one.data import SystemOneSource, check_records, check_rendered_request, render_sample
from prime_rl.trainer.sft.system_one.forward import decision_forward
from prime_rl.trainer.sft.system_one.images import load_image
from prime_rl.trainer.sft.system_one.loss import system_one_collate
from prime_rl.trainer.sft.system_one.record import parse_record
from prime_rl.trainer.vlm import setup_processor
from tests.unit.train.models.qwen3_5.clef_flash_release import tiny_config, tokenizer, write_processor

QUESTIONS = {"q": {"type": "choice", "instructions": "What is it?", "criteria": {"cat": None, "dog": None}}}
COLUMNS = SystemOneColumnsConfig()


def data_url(image: Image.Image, image_format: str = "PNG", **kwargs) -> str:
    buffer = BytesIO()
    image.save(buffer, image_format, **kwargs)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def gradient(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    image.putdata([(x * 7 % 256, y * 5 % 256, (x + y) % 256) for y in range(height) for x in range(width)])
    return image


WIDE, TALL = gradient(96, 64), gradient(64, 128)
IMAGES = [data_url(WIDE), data_url(TALL)]
# An image has a token per merged 2x2 block of 16x16 patches
TOKENS = [3 * 2, 2 * 4]


def row(images=None, state="A photo.", record_id="r") -> dict:
    return {
        "id": record_id,
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
    # A format vLLM refuses, whatever the URL says
    eps = data_url(WIDE, "EPS").replace("image/eps", "image/png")
    assert strategy.render_request(request([eps]), max_tokens=4096) is None
    assert all(warning.startswith("Skipping record 'r': vLLM refuses it: media: ") for warning in warnings)
    # The model would put image features at the text's own image token
    assert strategy.render_request(request(IMAGES, "A <|image_pad|>."), max_tokens=4096) is None
    assert strategy.render_request(request(None, "A <|image_pad|>."), max_tokens=4096) is None
    assert len(warnings) == 5 and warnings[3:] == ["Skipping record 'r': its text holds the image token"] * 2


def test_a_tokenizer_with_another_image_token_is_refused(processor):
    config = tiny_config()
    config.image_token_id += 1
    with pytest.raises(ValueError, match="is not the model's image token"):
        ClefFlashStrategy(tokenizer(), config, processor)


def test_a_run_without_images_refuses_records_with_them():
    strategy = ClefFlashStrategy(tokenizer(), tiny_config())
    with pytest.raises(ValueError, match=r"record 'r' has images, which need \[model.vlm\]"):
        render_sample(strategy, request(IMAGES), 4096)
    dataset = Dataset.from_list([row(), row(IMAGES)])
    with pytest.raises(ValueError, match=r"record 'r' has images, which need \[model.vlm\]"):
        check_records(dataset, COLUMNS)
    check_records(dataset, COLUMNS, takes_images=True)
    check_records(Dataset.from_list([row(), row([])]), COLUMNS)


@pytest.mark.parametrize("images", ["data:image/png;base64,AAAA", ["https://example.com/cat.png"], [1], {"a": 1}])
def test_images_are_a_list_of_data_urls(images):
    with pytest.raises(ValueError, match="record 'r': column 'images' must hold a list of base64 data URLs"):
        request(images)


def test_the_images_column_holds_json_text():
    dataset = Dataset.from_list([{**row(), "images": IMAGES}])
    with pytest.raises(ValueError, match="data.columns.images is 'images', a column of type"):
        check_records(dataset, COLUMNS, takes_images=True)


def test_a_rendering_with_broken_image_inputs_is_refused(strategy):
    rendered = strategy.render_request(request(IMAGES), max_tokens=4096)
    without = {key: value for key, value in rendered.items() if not key.startswith("mm_")}
    for broken, asked, error in [
        ({**without, "mm_kwargs": rendered["mm_kwargs"]}, IMAGES, "go together"),
        ({**rendered, "mm_token_type_ids": [0]}, IMAGES, "1 mm_token_type_ids for"),
        (without, IMAGES, "must be given for a request with images"),
        (rendered, None, "must be given for a request with images"),
    ]:
        with pytest.raises(ValueError, match=error):
            check_rendered_request(broken, request(asked), vocab_size=288)


def test_a_row_of_records_with_and_without_images_reaches_the_model(strategy):
    source = SystemOneSource(Dataset.from_list([row()]), strategy, seq_len=4096)
    rows = [row(), row(IMAGES[:1]), row(IMAGES[1:])]
    samples = [source._process(example) for example in rows]
    lengths = [len(sample["input_ids"]) for sample in samples]
    (packed,) = CatDataset(samples, seq_len=sum(lengths) + 8, pad_to_multiple_of=sum(lengths) + 8)
    batch = system_one_collate([packed])

    image_token = tiny_config().image_token_id
    assert batch["mm_token_type_ids"].tolist() == (batch["input_ids"] == image_token).long().tolist()
    assert batch["seq_lens"].tolist() == [*lengths, 8]
    assert batch["mm_kwargs"]["image_grid_thw"].tolist() == [[1, 4, 6], [1, 8, 4]]
    for key, value in batch["mm_kwargs"].items():
        assert torch.equal(value, torch.cat([sample["mm_kwargs"][key] for sample in samples[1:]]))

    class Backbone(nn.Module):
        def forward(self, input_ids, position_ids, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(last_hidden_state=torch.zeros(1, input_ids.shape[1], 4))

    class Strategy(ClefFlashStrategy):
        def option_logits(self, model, hidden_states, input_ids, questions):
            return torch.zeros(sum(len(question.options) for question in questions))

    model = nn.Module()
    model.model = Backbone()
    model.lm_head = nn.Linear(4, 8, bias=False)
    inject_prime_lm_head(model)
    stub = Strategy(tokenizer(), tiny_config())
    decision_forward(model, batch, strategy=stub, brier_weight=0.1, device="cpu")
    for key in ("pixel_values", "image_grid_thw"):
        assert torch.equal(model.model.kwargs[key], batch["mm_kwargs"][key])
    assert torch.equal(model.model.kwargs["mm_token_type_ids"], batch["mm_token_type_ids"])

    (packed,) = CatDataset(samples[:1], seq_len=4096, pad_to_multiple_of=1)
    decision_forward(model, system_one_collate([packed]), strategy=stub, brier_weight=0.1, device="cpu")
    assert "pixel_values" not in model.model.kwargs and "mm_token_type_ids" not in model.model.kwargs


def images_of_every_kind() -> dict[str, str]:
    transparent = WIDE.convert("RGBA")
    transparent.putalpha(Image.linear_gradient("L").resize(WIDE.size))
    rotated = Image.Exif()
    rotated[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to view
    return {
        "rgb": data_url(WIDE),
        "grey": data_url(WIDE.convert("L")),
        "rgba": data_url(transparent),
        "la": data_url(transparent.convert("LA")),
        "palette": data_url(WIDE.convert("P")),
        "palette with transparency": data_url(WIDE.convert("P"), transparency=0),
        "rgb with a transparent colour": data_url(WIDE, transparency=(0, 0, 0)),
        "jpeg": data_url(WIDE, "JPEG"),
        "rotated jpeg": data_url(WIDE, "JPEG", exif=rotated),
    }


@pytest.mark.parametrize("kind", images_of_every_kind())
def test_loads_an_image_as_vllm_loads_it(kind):
    media = pytest.importorskip("vllm.multimodal.media.image")
    url = images_of_every_kind()[kind]
    reference = media.ImageMediaIO().load_base64("image/png", url.partition(",")[2]).media
    image = load_image(url)
    assert image.mode == reference.mode == "RGB" and image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_transparency_is_white_and_orientation_is_applied():
    assert load_image(data_url(Image.new("RGBA", (2, 2), (255, 0, 0, 0)))).getpixel((0, 0)) == (255, 255, 255)
    assert load_image(images_of_every_kind()["rotated jpeg"]).size == (64, 96)
