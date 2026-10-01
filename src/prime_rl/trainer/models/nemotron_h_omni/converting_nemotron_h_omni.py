from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename, Rename
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain as language_conversion_chain


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return "language_model.backbone.embeddings.weight" in state_dict or any(
        name.startswith("language_model.backbone.layers.") for name in state_dict
    )


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.startswith("model.language_model.") for name in state_dict)


def conversion_chain(config) -> list[ConvOp]:
    return [
        *language_conversion_chain(
            config.llm_config, hf_prefix="language_model.backbone", prime_prefix="model.language_model"
        ),
        Rename("language_model.lm_head.weight", "lm_head.weight"),
        PrefixRename("vision_model.radio_model.model.blocks.", "model.visual.blocks."),
        Rename("vision_model.radio_model.model.patch_generator.embedder.weight", "model.visual.patch_embed.weight"),
        Drop("vision_model.radio_model.model.patch_generator.video_embedder.weight"),
        Rename("vision_model.radio_model.model.patch_generator.pos_embed", "model.visual.pos_embed"),
        Rename("vision_model.radio_model.model.patch_generator.cls_token.token", "model.visual.prefix_tokens"),
        Rename("vision_model.radio_model.input_conditioner.norm_mean", "model.visual.norm_mean"),
        Rename("vision_model.radio_model.input_conditioner.norm_std", "model.visual.norm_std"),
        Rename("mlp1.0.weight", "model.vision_projector.norm.weight"),
        Rename("mlp1.1.weight", "model.vision_projector.linear1.weight"),
        Rename("mlp1.3.weight", "model.vision_projector.linear2.weight"),
        Drop("sound_encoder.", is_prefix=True),
        Drop("sound_projection.", is_prefix=True),
    ]
