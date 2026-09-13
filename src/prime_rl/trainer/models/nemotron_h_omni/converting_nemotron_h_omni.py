from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename, Rename, SplitConcat
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain as language_conversion_chain


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return "language_model.backbone.embeddings.weight" in state_dict


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return "model.language_model.embed_tokens.weight" in state_dict or any(
        name.startswith("model.language_model.layers.")
        and any(namespace in name for namespace in (".mamba.", ".self_attn.", ".mlp.router."))
        for name in state_dict
    )


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = [
        Drop("backbone.", is_prefix=True),
        Drop("mtp.", is_prefix=True),
        Drop("lm_head.weight"),
        *language_conversion_chain(
            config.llm_config,
            hf_prefix="language_model.",
            prime_prefix="model.language_model",
        ),
        Rename("language_model.lm_head.weight", "lm_head.weight"),
        Drop("vision_model.radio_model.input_conditioner.", is_prefix=True),
        PrefixRename(
            "vision_model.radio_model.model.patch_generator.video_embedder.",
            "model.vision_model.embeddings.video_patch_projection.",
        ),
        PrefixRename(
            "vision_model.radio_model.model.patch_generator.embedder.",
            "model.vision_model.embeddings.patch_projection.",
        ),
        Rename(
            "vision_model.radio_model.model.patch_generator.pos_embed",
            "model.vision_model.embeddings.position_embedding",
        ),
        Rename(
            "vision_model.radio_model.model.patch_generator.cls_token.token",
            "model.vision_model.embeddings.cls_register_token",
        ),
        Rename("vision_model.summary_idxs", "model.vision_model.summary_idxs"),
        PrefixRename(
            "vision_model.radio_model.model.blocks.",
            "model.vision_model.encoder.layer.",
        ),
    ]

    for layer_idx in range(config.vision_config.num_hidden_layers):
        prefix = f"model.vision_model.encoder.layer.{layer_idx}"
        ops.extend(
            [
                Rename(f"{prefix}.attn.proj.weight", f"{prefix}.attention.output.dense.weight"),
                Rename(f"{prefix}.attn.proj.bias", f"{prefix}.attention.output.dense.bias"),
                SplitConcat(
                    combined=f"{prefix}.attn.qkv.weight",
                    parts=[
                        (f"{prefix}.attention.attention.query.weight", None),
                        (f"{prefix}.attention.attention.key.weight", None),
                        (f"{prefix}.attention.attention.value.weight", None),
                    ],
                    dim=0,
                ),
                SplitConcat(
                    combined=f"{prefix}.attn.qkv.bias",
                    parts=[
                        (f"{prefix}.attention.attention.query.bias", None),
                        (f"{prefix}.attention.attention.key.bias", None),
                        (f"{prefix}.attention.attention.value.bias", None),
                    ],
                    dim=0,
                ),
            ]
        )

    ops.extend(
        [
            Rename("mlp1.0.weight", "model.vision_projector.mlp1.norm.weight"),
            Rename("mlp1.1.weight", "model.vision_projector.mlp1.linear1.weight"),
            Rename("mlp1.3.weight", "model.vision_projector.mlp1.linear2.weight"),
            PrefixRename(
                "vision_projector.vision_final_layernorm.",
                "model.vision_projector.vision_final_layernorm.",
            ),
        ]
    )
    return ops


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
