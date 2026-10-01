import torch
from torch import Tensor, nn

from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.nemotron_h.backbone import NemotronHModel
from prime_rl.trainer.models.nemotron_h.base import NemotronHPreTrainedModel
from prime_rl.trainer.models.nemotron_h_omni.configuration_nemotron_h_omni import NemotronHOmniConfig
from prime_rl.trainer.models.nemotron_h_omni.converting_nemotron_h_omni import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.nemotron_h_omni.vision import RadioVisionModel


class NemotronOmniProjector(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(input_size, eps=1e-5))
        self.linear1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear2 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear2(torch.relu(self.linear1(self.norm(hidden_states))).square())


class NemotronHOmniModel(nn.Module):
    def __init__(self, config: NemotronHOmniConfig) -> None:
        super().__init__()
        self.language_model = NemotronHModel(config.llm_config)
        self.visual = RadioVisionModel(config.vision_config, config.norm_mean, config.norm_std)
        self.vision_projector = NemotronOmniProjector(
            config.vit_hidden_size * int(1 / config.downsample_ratio) ** 2,
            config.projector_hidden_size,
            config.llm_config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: torch.Tensor | None = None,
    ) -> Tensor:
        return self.language_model(
            input_ids=input_ids,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            routed_experts=routed_experts,
        )


class NemotronHOmniForCausalLM(NemotronHPreTrainedModel):
    config_class = NemotronHOmniConfig
    supports_packed_multimodal_training = False

    def __init__(self, config: NemotronHOmniConfig) -> None:
        super().__init__(config)
        self.model = NemotronHOmniModel(config)
        self.lm_head = VanillaOutputLinear(config.llm_config.hidden_size, config.llm_config.vocab_size)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return super().keep_in_fp32_for_weight_transfer(name) or name.endswith(("visual.norm_mean", "visual.norm_std"))

    is_hf_state_dict = staticmethod(is_hf_state_dict)
    is_prime_state_dict = staticmethod(is_prime_state_dict)
    conversion_chain = staticmethod(conversion_chain)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.language_model.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        temperature: torch.Tensor | None = None,
        sampling_mask: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        hidden_states = self.model(
            input_ids, seq_lens=seq_lens, seq_lens_are_pre_shard=seq_lens_are_pre_shard, routed_experts=routed_experts
        )
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        state_dict = super().convert_adapter_to_hf(state_dict)
        return {
            name.replace("model.language_model.", "language_model.backbone.", 1): tensor
            for name, tensor in state_dict.items()
        }
