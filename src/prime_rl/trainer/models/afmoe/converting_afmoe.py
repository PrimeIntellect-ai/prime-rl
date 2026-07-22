"""HF<->prime weight conversion for AFMoE, as a declarative op chain.

AFMoE specifics:

* Per-layer MLP prefix is ``model.layers.{i}.mlp``.
* The router shares its name between HF and prime, so it is *not* renamed.
* Shared experts: HF ``shared_experts.{gate,down,up}_proj.weight`` map to prime
  ``shared_expert.{w1,w2,w3}`` (no ``.weight`` suffix on the prime side).
* Routed experts: HF per-expert ``experts.{e}.{gate,down,up}_proj.weight`` stack
  into prime ``experts.{w1,w2,w3}`` along dim 0 (no fused gate_up layout).
* Prime-only runtime buffers ``mlp.tokens_per_expert`` and ``mlp.reorderer.*``
  are dropped on the way back to HF.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import GATE_DOWN_UP, ConvOp, Drop, Rename, Stack


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}.mlp"
        for wn, hf_proj in GATE_DOWN_UP:
            ops.append(Rename(f"{p}.shared_experts.{hf_proj}.weight", f"{p}.shared_expert.{wn}"))
        for wn, hf_proj in GATE_DOWN_UP:
            ops.append(Stack(stacked=f"{p}.experts.{wn}", item=f"{p}.experts.{{e}}.{hf_proj}.weight"))
        ops.append(Drop(f"{p}.tokens_per_expert"))
        ops.append(Drop(f"{p}.reorderer", is_prefix=True))
    return ops
