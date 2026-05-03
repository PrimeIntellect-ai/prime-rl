from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5RMSNorm,
    create_causal_mask,
)

from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.mtp import (
    detached_lm_head_cross_entropy,
    make_viewless_tensor_with_grad,
    mtp_masks_from_label_mask,
    roll_tensor,
)


def _make_dense_mtp_layer_config(config: Qwen3_5TextConfig) -> Qwen3_5TextConfig:
    mtp_config = copy.deepcopy(config)
    mtp_config.num_hidden_layers = 1
    mtp_config.layer_types = ["full_attention"]
    return mtp_config


class Qwen3_5DenseMTP(nn.Module):
    """Qwen3.5 dense MTP module using Qwen's native HF checkpoint names."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(_make_dense_mtp_layer_config(config), layer_idx=0)
                for _ in range(getattr(config, "mtp_num_hidden_layers", 0))
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        layer_idx: int,
        token_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        token_embeddings = self.pre_fc_norm_embedding(token_embeddings)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([token_embeddings, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        hidden_states = self.layers[layer_idx](
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return self.norm(hidden_states)


def _compute_dense_mtp_loss(
    self: Qwen3_5ForConditionalGeneration,
    hidden_states: torch.Tensor,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    loss_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
) -> PrimeLmOutput:
    if not hasattr(self, "mtp") or len(self.mtp.layers) == 0:
        raise ValueError("MTP was enabled but this Qwen3.5 dense config has no MTP layers.")
    if position_ids is None:
        raise ValueError("Dense Qwen3.5 MTP training is only supported for text-only inputs with position_ids.")
    if position_ids.ndim == 3:
        position_ids = position_ids[0]
    if position_ids.ndim != 2:
        raise ValueError(f"Dense Qwen3.5 MTP expected [batch, seq] position_ids, got {tuple(position_ids.shape)}.")

    language_model = self.model.language_model
    mtp_hidden = make_viewless_tensor_with_grad(hidden_states)
    mtp_input_ids = input_ids
    mtp_labels = labels
    mtp_position_ids = position_ids
    mtp_losses = []
    mtp_masks = list(mtp_masks_from_label_mask(loss_mask, position_ids, len(self.mtp.layers)))

    for layer_idx, mtp_mask in enumerate(mtp_masks):
        mtp_input_ids = roll_tensor(
            mtp_input_ids,
            position_ids=position_ids,
            fill_value=language_model.embed_tokens.padding_idx or 0,
        )
        mtp_labels = roll_tensor(mtp_labels, position_ids=position_ids, fill_value=0)
        mtp_position_ids = roll_tensor(mtp_position_ids, position_ids=position_ids, fill_value=0)
        mtp_embeddings = language_model.embed_tokens(mtp_input_ids).detach()
        position_embeddings = language_model.rotary_emb(mtp_hidden, mtp_position_ids)
        attention_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=mtp_hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=mtp_position_ids,
        )
        mtp_hidden = self.mtp(
            layer_idx,
            mtp_embeddings,
            mtp_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=mtp_position_ids,
        )
        mtp_losses.append(detached_lm_head_cross_entropy(self.lm_head, mtp_hidden, mtp_labels, mtp_mask))

    mtp_loss = torch.stack(mtp_losses).mean() * self._text_config.prime_mtp_loss_scale
    return PrimeLmOutput(
        mtp_loss=mtp_loss,
        mtp_loss_per_depth=tuple(mtp_losses),
        mtp_token_count=torch.stack([mask.sum() for mask in mtp_masks]).sum(),
    )


def patch_qwen3_5_dense_mtp() -> None:
    """Attach Qwen3.5 dense MTP modules before HF checkpoint loading."""

    if getattr(Qwen3_5ForConditionalGeneration.__init__, "_prl_dense_mtp_patched", False):
        return

    original_init = Qwen3_5ForConditionalGeneration.__init__

    def patched_init(self: Qwen3_5ForConditionalGeneration, config):
        original_init(self, config)
        text_config = config.text_config
        self._text_config = text_config
        if (
            getattr(text_config, "prime_mtp_enabled", False)
            and getattr(text_config, "model_type", None) == "qwen3_5_text"
        ):
            self.mtp = Qwen3_5DenseMTP(text_config)
            self._compute_mtp_loss = types.MethodType(_compute_dense_mtp_loss, self)

    patched_init._prl_dense_mtp_patched = True
    Qwen3_5ForConditionalGeneration.__init__ = patched_init
