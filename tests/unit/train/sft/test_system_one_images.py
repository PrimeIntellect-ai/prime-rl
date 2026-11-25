"""A System One record's images, from its row to the model's forward, through a strategy that reads images."""

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from PIL import Image
from torch import nn
from transformers import PretrainedConfig

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.data import CatDataset
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.sft.system_one import data as system_one_data
from prime_rl.trainer.sft.system_one.data import (
    RenderedRequest,
    SystemOneSource,
    SystemOneStrategy,
    check_records,
    check_rendered_request,
    render_sample,
)
from prime_rl.trainer.sft.system_one.forward import decision_forward
from prime_rl.trainer.sft.system_one.images import load_image
from prime_rl.trainer.sft.system_one.loss import system_one_collate
from prime_rl.trainer.sft.system_one.record import parse_record
from tests.unit.train.sft.image_urls import EPS, TALL, WIDE, data_url, images_of_every_kind

QUESTIONS = {"q": {"type": "choice", "instructions": "What is it?", "criteria": {"cat": None, "dog": None}}}
COLUMNS = SystemOneColumnsConfig()
IMAGES = [data_url(WIDE), data_url(TALL)]
IMAGE_TOKEN = 9
CONFIG = PretrainedConfig(vocab_size=32, image_token_id=IMAGE_TOKEN)
PROCESSOR = SimpleNamespace()


def row(images=None, state=(10, 11), record_id="r") -> dict:
    return {
        "id": record_id,
        "state": json.dumps(state),
        "questions": json.dumps(QUESTIONS),
        "label": json.dumps({"q": "dog"}),
        "images": None if images is None else json.dumps(images),
    }


def request(images=None, state=(10, 11)):
    return parse_record(row(images, state), COLUMNS)[0]


class ImageStrategy(SystemOneStrategy):
    """A state of token ids after the images, an image having a token per 32 x 32 pixels. The last token is read."""

    modalities = frozenset({"image"})

    def render_request(self, request, max_tokens):
        grids = [(1, image.height // 16, image.width // 16) for image in map(load_image, request.images)]
        image_ids = [IMAGE_TOKEN] * sum(h * w // 4 for _, h, w in grids)
        input_ids = image_ids + list(request.state) + [1]
        rendered = RenderedRequest(
            input_ids=input_ids,
            seq_lens=[len(input_ids)],
            read_spans=[[(len(input_ids) - 1, len(input_ids))]],
        )
        if grids:
            patches = sum(h * w for _, h, w in grids)
            rendered["mm_kwargs"] = {
                "pixel_values": torch.arange(patches * 3.0).view(patches, 3),
                "image_grid_thw": torch.tensor(grids),
            }
            rendered["mm_token_type_ids"] = [1] * len(image_ids) + [0] * (len(input_ids) - len(image_ids))
        return rendered

    def option_logits(self, model, hidden_states, input_ids, questions):
        model.read_ids = input_ids
        return torch.zeros(sum(len(question.options) for question in questions))


@pytest.fixture
def strategy() -> ImageStrategy:
    return ImageStrategy(None, CONFIG, PROCESSOR)


def test_a_run_without_images_refuses_records_with_them():
    with pytest.raises(ValueError, match=r"record 'r' has images, which need \[model.vlm\]"):
        render_sample(ImageStrategy(None, CONFIG), request(IMAGES), 4096)
    dataset = Dataset.from_list([row(), row(IMAGES)])
    with pytest.raises(ValueError, match=r"record 'r' has images, which need \[model.vlm\]"):
        check_records(dataset, COLUMNS)
    check_records(dataset, COLUMNS, takes_images=True)
    check_records(Dataset.from_list([row(), row([])]), COLUMNS)


@pytest.mark.parametrize(
    "images",
    [
        "data:image/png;base64,AAAA",
        ["https://example.com/cat.png"],
        ["data:video/mp4;base64,AAAA"],
        ["data:image/png,AAAA"],
        [1],
        {"a": 1},
    ],
)
def test_images_are_a_list_of_data_urls(images):
    with pytest.raises(ValueError, match="record 'r': column 'images' must hold a list of base64 data URLs"):
        request(images)


def test_the_images_column_holds_json_text():
    dataset = Dataset.from_list([{**row(), "images": IMAGES}])
    with pytest.raises(ValueError, match="data.columns.images is 'images', a column of type"):
        check_records(dataset, COLUMNS, takes_images=True)


def test_images_are_not_labels():
    with pytest.raises(ValueError, match="needs a label column"):
        check_records(Dataset.from_list([{**row(IMAGES), "label": None}]), COLUMNS, takes_images=True)


def test_warns_of_a_source_without_the_images_column(monkeypatch):
    warnings = []
    monkeypatch.setattr(system_one_data, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    without = {key: value for key, value in row().items() if key != "images"}
    dataset = Dataset.from_list([{**without, "pictures": json.dumps(IMAGES)}] * 2)
    check_records(dataset, COLUMNS, source="records", split="test", takes_images=True)
    assert len(warnings) == 1 and warnings[0].startswith("Source 'records', split 'test' has no 'images' column")
    check_records(dataset, COLUMNS)  # a text-only run
    check_records(dataset, COLUMNS, rank=1, world_size=2, takes_images=True)
    check_records(Dataset.from_list([row()]), COLUMNS, takes_images=True)
    assert len(warnings) == 1


def test_a_renamed_images_column_must_exist():
    renamed = SystemOneColumnsConfig(images="pictures")
    with pytest.raises(ValueError, match="data.columns.images is 'pictures'"):
        check_records(Dataset.from_list([row(IMAGES)]), renamed, takes_images=True)
    moved = {key: value for key, value in row(IMAGES).items() if key != "images"} | {"pictures": json.dumps(IMAGES)}
    assert parse_record(moved, renamed)[0].images == tuple(IMAGES)


def test_a_rendering_with_broken_image_inputs_is_refused(strategy):
    rendered = strategy.render_request(request(IMAGES), max_tokens=4096)
    check_rendered_request(rendered, request(IMAGES), 32, IMAGE_TOKEN)
    without = {key: value for key, value in rendered.items() if not key.startswith("mm_")}
    video = {**rendered, "mm_kwargs": {"pixel_values_videos": rendered["mm_kwargs"]["pixel_values"]}}
    types, grids = rendered["mm_token_type_ids"], rendered["mm_kwargs"]["image_grid_thw"]
    other = {**rendered, "input_ids": [12 if token == IMAGE_TOKEN else token for token in rendered["input_ids"]]}
    for broken, asked, error in [
        ({**without, "mm_kwargs": rendered["mm_kwargs"]}, IMAGES, "go together"),
        ({**rendered, "mm_token_type_ids": [0]}, IMAGES, "1 mm_token_type_ids for"),
        (without, IMAGES, "must be given for a request with images"),
        (rendered, None, "must be given for a request with images"),
        (video, IMAGES, "video inputs are not supported"),
        ({**rendered, "mm_kwargs": {}}, IMAGES, r"mm_kwargs lacks \['image_grid_thw', 'pixel_values'\]"),
        ({**rendered, "mm_kwargs": {"image_grid_thw": grids}}, IMAGES, r"mm_kwargs lacks \['pixel_values'\]"),
        (strategy.render_request(request(IMAGES[:1]), 4096), IMAGES, "1 images rendered of the record's 2"),
        ({**rendered, "mm_token_type_ids": [0] * len(types)}, IMAGES, "must be 1 at the images' tokens"),
        ({**rendered, "mm_token_type_ids": [2 * t for t in types]}, IMAGES, "must be 1 at the images' tokens"),
        ({**rendered, "mm_token_type_ids": [0, *types[:-1]]}, IMAGES, "must be the model's image token 9, got"),
        (other, IMAGES, "must be the model's image token 9, got"),
    ]:
        with pytest.raises(ValueError, match=error):
            check_rendered_request(broken, request(asked), 32, IMAGE_TOKEN)
    # A strategy with a prompt per question shows the images in each
    check_rendered_request(rendered, request(IMAGES[:1]), 32, IMAGE_TOKEN)


@pytest.mark.parametrize("images", [None, IMAGES])
def test_skips_a_record_whose_text_holds_the_image_token(strategy, images, monkeypatch):
    warnings = []
    monkeypatch.setattr(system_one_data, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    assert render_sample(strategy, request(images), 4096) is not None
    assert render_sample(strategy, request(images, state=(10, IMAGE_TOKEN)), 4096) is None
    assert len(warnings) == 1
    assert warnings[0].startswith("Skipping record 'r': its rendering has the model's image token outside its images")
    # A run without images puts no image features anywhere
    assert render_sample(ImageStrategy(None, CONFIG), request(None, state=(10, IMAGE_TOKEN)), 4096) is not None


def test_a_row_of_records_with_and_without_images_reaches_the_model(strategy):
    source = SystemOneSource(Dataset.from_list([row()]), strategy, seq_len=4096)
    rows = [row(), row(IMAGES[:1]), row(IMAGES[1:])]
    samples = [source._process(example) for example in rows]
    lengths = [len(sample["input_ids"]) for sample in samples]
    assert lengths == [3, 9, 11]  # the images have 6 and 8 tokens
    (packed,) = CatDataset(samples, seq_len=sum(lengths) + 8, pad_to_multiple_of=sum(lengths) + 8)
    batch = system_one_collate([packed])

    assert batch["mm_token_type_ids"].tolist() == (batch["input_ids"] == IMAGE_TOKEN).long().tolist()
    assert batch["seq_lens"].tolist() == [*lengths, 8]
    assert batch["mm_kwargs"]["image_grid_thw"].tolist() == [[1, 4, 6], [1, 8, 4]]
    for key, value in batch["mm_kwargs"].items():
        assert torch.equal(value, torch.cat([sample["mm_kwargs"][key] for sample in samples[1:]]))

    class Backbone(nn.Module):
        def forward(self, input_ids, position_ids, **kwargs):
            self.kwargs = {**kwargs, "input_ids": input_ids, "position_ids": position_ids}
            return SimpleNamespace(last_hidden_state=torch.zeros(1, input_ids.shape[1], 4))

    model = nn.Module()
    model.model = Backbone()
    model.lm_head = nn.Linear(4, 8, bias=False)
    inject_prime_lm_head(model)
    out = decision_forward(model, batch, strategy=strategy, brier_weight=0.1, device="cpu")
    # Uniform probabilities over two options, for three labelled questions
    torch.testing.assert_close(out["loss"], 3 * (torch.tensor(2.0).log() + 0.1 * 0.5))
    for key in ("pixel_values", "image_grid_thw"):
        assert torch.equal(model.model.kwargs[key], batch["mm_kwargs"][key])
    for key in ("input_ids", "position_ids", "seq_lens", "mm_token_type_ids"):
        assert torch.equal(model.model.kwargs[key], batch[key])
    assert torch.equal(model.read_ids, batch["input_ids"][0])

    (packed,) = CatDataset(samples[:1], seq_len=4096, pad_to_multiple_of=1)
    decision_forward(model, system_one_collate([packed]), strategy=strategy, brier_weight=0.1, device="cpu")
    assert "pixel_values" not in model.model.kwargs and "mm_token_type_ids" not in model.model.kwargs


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


@pytest.mark.parametrize(
    "url, error",
    [
        ("data:image/png;base64,AAAA", "cannot identify image file"),
        ("data:image/png;base64,!", ""),
        (EPS, "cannot identify image file"),  # not one of the formats, so Ghostscript is not looked for
    ],
)
def test_an_image_that_does_not_load_raises(strategy, url, error):
    with pytest.raises(ValueError, match=f"Failed to load image: .*{error}"):
        load_image(url)
    with pytest.raises(ValueError, match="record 'r': Failed to load image"):
        render_sample(strategy, request([url]), 4096)


def test_any_decoder_error_is_a_value_error(monkeypatch):
    def broken(*args, **kwargs):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(Image, "open", broken)
    with pytest.raises(ValueError, match="Failed to load image: division by zero"):
        load_image(IMAGES[0])
