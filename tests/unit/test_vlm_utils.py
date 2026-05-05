from types import SimpleNamespace

from prime_rl.utils.vlm import (
    get_language_model,
    get_layer_prefix,
    get_vision_encoder,
    is_vlm_architecture,
)


def test_gemma4_registry_points_at_text_layers() -> None:
    language_model = object()
    vision_tower = object()
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="gemma4"),
        model=SimpleNamespace(
            language_model=language_model,
            vision_tower=vision_tower,
        ),
    )

    assert get_language_model(model) is language_model
    assert get_vision_encoder(model) is vision_tower
    assert get_layer_prefix(model.config) == "model.language_model.layers."
    assert is_vlm_architecture(model.config)
