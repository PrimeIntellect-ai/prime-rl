from types import SimpleNamespace

import pytest

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import resolve_cp_style
from prime_rl.trainer.models.base import ALL_CP_STYLES


@pytest.mark.parametrize(
    ("cp", "q_heads", "kv_heads", "styles", "expected"),
    [
        (2, 32, 4, ALL_CP_STYLES, "ulysses"),
        (8, 32, 2, ALL_CP_STYLES, "ulysses"),
        (3, 32, 4, ALL_CP_STYLES, "ring"),
        (4, 32, 6, ALL_CP_STYLES, "ring"),
        (3, 32, 4, frozenset({"ulysses"}), "ulysses"),
        (2, 64, 1, frozenset({"ring"}), "ring"),
    ],
)
def test_resolve_cp_style_auto(cp, q_heads, kv_heads, styles, expected):
    config = ModelConfig(cp=cp)
    hf_config = SimpleNamespace(text_config=SimpleNamespace(num_attention_heads=q_heads, num_key_value_heads=kv_heads))
    resolve_cp_style(config, hf_config, styles)
    assert config.cp_style == expected
