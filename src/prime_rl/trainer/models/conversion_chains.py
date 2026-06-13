"""Per-model declarative conversion chains.

Each ``build_*_chain(num_layers, ...)`` returns the flat list of
:class:`~prime_rl.trainer.models.conversion_ops.ConvOp` that defines a model's
HF<->prime conversion, with concrete (fully templated) keys. The model classes
expose these via ``conversion_ops`` and the base ``convert_to_*`` methods play
them forward/backward. Equivalence to the legacy imperative converters is
checked in tests/unit/train/models/conversions.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import (
    ConvOp,
    FusedGateUp,
    MoEExperts,
    Rename,
)

# Per-layer routed-expert proj order shared by the Llama-style MoE models:
# prime w1=gate, w2=down, w3=up.
_GATE_DOWN_UP = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))


def _routed_experts_op(
    prefix: str,
    *,
    hf_experts: str,
    tt_experts: str,
    proj_order=_GATE_DOWN_UP,
    hf_proj_suffix: str = ".weight",
    fused: bool = False,
) -> MoEExperts:
    """Build the MoEExperts op for one layer.

    ``hf_experts``/``tt_experts`` are the (relative) expert container names,
    e.g. ``mlp.experts`` / ``block_sparse_moe.experts``. ``proj_order`` maps
    prime ``wN`` to the HF per-expert proj name."""
    projs = {
        f"{prefix}.{tt_experts}.{wn}": f"{prefix}.{hf_experts}.{{e}}.{hf_proj}{hf_proj_suffix}"
        for wn, hf_proj in proj_order
    }
    fused_spec = None
    if fused:
        fused_spec = FusedGateUp(
            gate_up=f"{prefix}.{hf_experts}.gate_up_proj",
            down=f"{prefix}.{hf_experts}.down_proj",
            w_gate=f"{prefix}.{tt_experts}.w1",
            w_down=f"{prefix}.{tt_experts}.w2",
            w_up=f"{prefix}.{tt_experts}.w3",
            split_dim=1,
        )
    return MoEExperts(projs=projs, fused=fused_spec)


def build_qwen3_moe_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(_routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
    return ops


# --------------------------------------------------------------------------- #
# Registry: model_type -> a builder that takes the model config and returns the
# full conversion chain. Each model's per-layer builder lives in its own
# package (``models/<name>/conversion_chain.py``); this maps the HF
# ``config.model_type`` to it. ``PreTrainedModelPrimeRL.conversion_ops`` (in
# base.py) dispatches here so the chain is reachable from the model class.
# --------------------------------------------------------------------------- #


def _by_num_layers(builder):
    return lambda config: builder(config.num_hidden_layers)


def _conversion_chain_registry() -> dict[str, "callable"]:
    # Imported lazily inside the function to avoid import cycles at module load.
    from prime_rl.trainer.models.afmoe.conversion_chain import build_afmoe_chain
    from prime_rl.trainer.models.glm4_moe.conversion_chain import build_glm4_moe_chain
    from prime_rl.trainer.models.glm_moe_dsa.conversion_chain import build_glm_moe_dsa_chain
    from prime_rl.trainer.models.gpt_oss.conversion_chain import build_gpt_oss_chain
    from prime_rl.trainer.models.laguna.conversion_chain import build_laguna_chain
    from prime_rl.trainer.models.minimax_m2.conversion_chain import build_minimax_m2_chain
    from prime_rl.trainer.models.nemotron_h.conversion_chain import build_nemotron_h_chain
    from prime_rl.trainer.models.qwen3_5_moe.conversion_chain import build_qwen3_5_moe_chain

    return {
        "qwen3_moe": _by_num_layers(build_qwen3_moe_chain),
        "qwen3_5_moe_text": _by_num_layers(build_qwen3_5_moe_chain),
        "glm4_moe": _by_num_layers(build_glm4_moe_chain),
        "glm_moe_dsa": _by_num_layers(build_glm_moe_dsa_chain),
        "minimax_m2": _by_num_layers(build_minimax_m2_chain),
        "laguna": _by_num_layers(build_laguna_chain),
        "afmoe": _by_num_layers(build_afmoe_chain),
        "nemotron_h": lambda config: build_nemotron_h_chain(config.layers_block_type),
        "gpt_oss": lambda config: build_gpt_oss_chain(),
    }


CONVERSION_CHAIN_MODEL_TYPES = frozenset(
    {
        "qwen3_moe",
        "qwen3_5_moe_text",
        "glm4_moe",
        "glm_moe_dsa",
        "minimax_m2",
        "laguna",
        "afmoe",
        "nemotron_h",
        "gpt_oss",
    }
)


def build_conversion_chain(model_type: str, config) -> list[ConvOp]:
    """Return the declarative HF<->prime conversion chain for ``model_type``."""
    registry = _conversion_chain_registry()
    if model_type not in registry:
        raise NotImplementedError(
            f"No declarative conversion chain registered for model_type {model_type!r}; available: {sorted(registry)}"
        )
    return registry[model_type](config)
