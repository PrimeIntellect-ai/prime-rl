from torch import Tensor

from prime_rl.trainer.models.base import CPSupport, PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)


class NemotronHPreTrainedModel(PreTrainedModelPrimeRL):
    config: NemotronHConfig

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        return CPSupport(
            frozenset({"ulysses"}),
            "Mamba layers require Ulysses to reconstruct full sequences while sharding Mamba heads",
        )

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith(("mamba.A_log", "mamba.D", "mlp.router.selection_bias"))

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config: NemotronHConfig):
        return conversion_chain(config)

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        import re

        for name in list(state_dict):
            hf_name = re.sub(r"(\.layers\.\d+)\.(?:self_attn|mlp|mamba)\.", r"\1.mixer.", name)
            if hf_name != name:
                state_dict[hf_name] = state_dict.pop(name)
        return state_dict

    def init_buffers_post_meta(self) -> None:
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()
