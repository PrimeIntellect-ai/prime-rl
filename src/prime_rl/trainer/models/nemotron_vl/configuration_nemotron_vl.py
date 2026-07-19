from transformers.configuration_utils import PretrainedConfig

from prime_rl.trainer.models.nemotron_h import NemotronHConfig


class RadioViTConfig(PretrainedConfig):
    """Configuration for the CRADIO v4-H vision tower (timm ViT-H/16 with RADIO's CPE patch generator).

    Matches the architecture NVIDIA grafted into Nemotron-3-Nano-Omni: `vit_huge_patch16_224`
    created with `model_norm=False` (final norm is Identity), a `ViTPatchGenerator` with
    absolute position embeddings stored at `max_img_size` resolution and interpolated per
    input size, and `num_cls_tokens` prefix tokens (cls + registers) that are dropped from
    the output features.
    """

    model_type = "radio_vit"

    def __init__(
        self,
        hidden_size: int = 1280,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        intermediate_size: int = 5120,
        patch_size: int = 16,
        max_img_size: int = 2048,
        num_cls_tokens: int = 10,
        layer_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.max_img_size = max_img_size
        self.num_cls_tokens = num_cls_tokens
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        super().__init__(**kwargs)


class NemotronVLConfig(PretrainedConfig):
    """Configuration for Nemotron-VL: a frozen CRADIO vision tower grafted onto a frozen
    NemotronH backbone through a trainable InternVL-style `mlp1` projector.

    The image pipeline fields (`force_image_size`, `downsample_ratio`, tiling bounds, image
    token ids) are copied from Nano Omni's config by the checkpoint assembly script and are
    the single source of truth for both the renderer and the model.
    """

    model_type = "nemotron_vl"
    sub_configs = {"text_config": NemotronHConfig, "vision_config": RadioViTConfig}

    def __init__(
        self,
        text_config: NemotronHConfig | dict | None = None,
        vision_config: RadioViTConfig | dict | None = None,
        vit_hidden_size: int = 1280,
        projector_hidden_size: int = 20480,
        downsample_ratio: float = 0.5,
        ps_version: str = "v2",
        force_image_size: int = 512,
        patch_size: int = 16,
        use_thumbnail: bool = True,
        min_num_patches: int = 1024,
        max_num_patches: int = 13312,
        image_tag_type: str = "internvl",
        img_context_token_id: int = 18,
        img_start_token_id: int = 19,
        img_end_token_id: int = 20,
        img_context_token: str = "<image>",
        img_start_token: str = "<img>",
        img_end_token: str = "</img>",
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
        lm_weights_prefixed: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = NemotronHConfig(**text_config)
        elif text_config is None:
            text_config = NemotronHConfig()
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = RadioViTConfig(**vision_config)
        elif vision_config is None:
            vision_config = RadioViTConfig()
        self.vision_config = vision_config

        self.vit_hidden_size = vit_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.downsample_ratio = downsample_ratio
        self.ps_version = ps_version
        self.force_image_size = force_image_size
        self.patch_size = patch_size
        self.use_thumbnail = use_thumbnail
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.image_tag_type = image_tag_type
        self.img_context_token_id = img_context_token_id
        self.img_start_token_id = img_start_token_id
        self.img_end_token_id = img_end_token_id
        self.img_context_token = img_context_token
        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
        self.norm_mean = norm_mean or [0.48145466, 0.4578275, 0.40821073]
        self.norm_std = norm_std or [0.26862954, 0.26130258, 0.27577711]
        # Whether LM weights in the checkpoint carry the `language_model.` prefix
        # (assembly `--mode rewrite`) or keep the original text-checkpoint names
        # (`--mode hardlink`). Conversion handles both; recorded for provenance.
        self.lm_weights_prefixed = lm_weights_prefixed

        super().__init__(**kwargs)
        # Composite configs don't get top-level tie_word_embeddings from PretrainedConfig,
        # but the trainer reads it off the root config; mirror the text config's value.
        self.tie_word_embeddings = self.text_config.tie_word_embeddings

    @property
    def num_image_token(self) -> int:
        """Tokens the LM sees per image tile (256 for 512px tiles, patch 16, 2x2 shuffle)."""
        return int((self.force_image_size // self.patch_size) ** 2 * (self.downsample_ratio**2))


__all__ = ["NemotronVLConfig", "RadioViTConfig"]
