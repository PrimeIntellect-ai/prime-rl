from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.vlm import supports_packed_multimodal_training, validate_multi_modal_pack


def resolve_pack_multimodal(config: TrainerConfig, model) -> bool:
    pack_multimodal = config.pack_multimodal and (
        config.model.vlm is not None or supports_packed_multimodal_training(model)
    )
    if not pack_multimodal:
        return False

    if config.model.cp > 1:
        raise ValueError(
            "Multimodal packing is not supported with context parallelism; MM+CP support is deferred to a follow-up PR."
        )

    validate_multi_modal_pack(model, attn_impl=config.model.attn)
    return True
