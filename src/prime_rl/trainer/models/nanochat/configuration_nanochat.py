from transformers.configuration_utils import PretrainedConfig


class NanochatConfig(PretrainedConfig):
    """nanochat GPT (karpathy/nanochat @ ff46300, ``nanochat/gpt.py``): rotary embeddings, QK norm, RMS norm without
    learnable parameters, relu^2 MLP, per-layer ``resid_lambda``/``x0_lambda`` scalars and value embeddings on
    alternating layers."""

    model_type = "nanochat"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 6,
        max_position_embeddings: int = 2048,
        window_pattern: str = "SSSL",
        logit_softcap: float = 15.0,
        init_seed: int = 42,
        residual: dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.window_pattern = window_pattern
        self.logit_softcap = logit_softcap
        self.init_seed = init_seed
        self.residual = {"type": "plain"} if residual is None else residual
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
