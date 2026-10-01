import torch

from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.nemotron_h.backbone import NemotronHModel
from prime_rl.trainer.models.nemotron_h.base import NemotronHPreTrainedModel
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig


class NemotronHForCausalLM(NemotronHPreTrainedModel):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__(config)
        self.model = NemotronHModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)

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
        outputs = self.model(
            input_ids=input_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        return self.lm_head(
            outputs,
            labels,
            temperature=temperature,
            sampling_mask=sampling_mask,
        )


__all__ = [
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
]
