from transformers.configuration_utils import PretrainedConfig

from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig


class RadioConfig(PretrainedConfig):
    model_type = "radio"

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        patch_size: int = 16,
        num_channels: int = 3,
        num_prefix_tokens: int = 10,
        max_resolution: int = 2048,
        args: dict | None = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_prefix_tokens = num_prefix_tokens
        self.max_resolution = max_resolution
        self.args = args or {}
        if self.args.get("model", "vit_huge_patch16_224") != "vit_huge_patch16_224":
            raise ValueError("Nemotron Omni currently supports the RADIO ViT-H/16 checkpoint layout")
        super().__init__(**kwargs)


class NemotronHOmniConfig(PretrainedConfig):
    model_type = "NemotronH_Nano_Omni_Reasoning_V3"
    sub_configs = {"llm_config": NemotronHConfig, "vision_config": RadioConfig}

    def __init__(
        self,
        llm_config: dict | NemotronHConfig | None = None,
        vision_config: dict | RadioConfig | None = None,
        vit_hidden_size: int = 1280,
        projector_hidden_size: int = 20480,
        downsample_ratio: float = 0.5,
        img_context_token_id: int = 18,
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.llm_config = (
            NemotronHConfig(**llm_config) if isinstance(llm_config, dict) else llm_config or NemotronHConfig()
        )
        self.vision_config = (
            RadioConfig(**vision_config) if isinstance(vision_config, dict) else vision_config or RadioConfig()
        )
        self.vit_hidden_size = vit_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.downsample_ratio = downsample_ratio
        self.img_context_token_id = img_context_token_id
        self.norm_mean = norm_mean or [0.48145466, 0.4578275, 0.40821073]
        self.norm_std = norm_std or [0.26862954, 0.26130258, 0.27577711]
        self.use_cache = use_cache
        if tie_word_embeddings or self.llm_config.tie_word_embeddings:
            raise ValueError("Nemotron Omni requires untied input embeddings and LM head")
        if vit_hidden_size != self.vision_config.hidden_size:
            raise ValueError("vit_hidden_size must match the RADIO hidden size")
        super().__init__(tie_word_embeddings=False, **kwargs)

    @property
    def text_config(self) -> NemotronHConfig:
        return self.llm_config

    def get_text_config(self, decoder: bool = False, encoder: bool = False) -> NemotronHConfig:
        return self.llm_config
