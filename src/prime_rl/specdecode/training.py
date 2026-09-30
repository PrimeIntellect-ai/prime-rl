from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from prime_rl.configs.specdecode import SpeculatorConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.cp import CPContext, gather_for_cp_wo_grad
from prime_rl.utils.vlm import get_language_model


def draft_loss_mask(
    loss_mask: torch.Tensor, seq_lens: torch.Tensor, block_size: int, predict_next: bool
) -> torch.Tensor:
    """Align prediction masks and exclude anchors whose block crosses a packed document."""
    mask = loss_mask.clone()
    if predict_next:
        mask[..., :-1] = loss_mask[..., 1:]
        mask[..., -1] = False
    ends = seq_lens.cumsum(0)
    positions = torch.arange(mask.shape[-1], device=mask.device)
    document = torch.searchsorted(ends, positions, right=True)
    remaining = ends[document] - positions
    return mask & (remaining > block_size)


@torch.no_grad()
def copy_sharded_weight(destination: torch.Tensor, source: torch.Tensor, rows: torch.Tensor | None = None) -> None:
    source = source.detach()
    if rows is not None:
        if isinstance(source, DTensor):
            source = source.full_tensor()
        source = source[rows.to(source.device)]
    if isinstance(destination, DTensor):
        if isinstance(source, DTensor):
            source = source.redistribute(destination.device_mesh, destination.placements)
        else:
            source = distribute_tensor(source.to(destination.device), destination.device_mesh, destination.placements)
        destination.to_local().copy_(source.to_local())
    else:
        if isinstance(source, DTensor):
            source = source.full_tensor()
        destination.copy_(source)


class SpeculatorTraining:
    """Upstream draft loss and detached policy features on the trainer's FSDP mesh."""

    def __init__(self, model: nn.Module, config: SpeculatorConfig, model_config: ModelConfig, dims: ParallelDims):
        from speculators.config import SpeculatorModelConfig
        from speculators.model import SpeculatorModel
        from speculators.train.config import TrainConfig

        self.model = model
        self.config = config
        self.cp_context = CPContext()
        self.features: dict[int | str, torch.Tensor] = {}
        self.metric_totals: dict[str, torch.Tensor] = {}
        self.capturing = False
        self.handles = []

        draft_config = SpeculatorModelConfig.from_pretrained(config.name, revision=config.revision)
        draft_config.transformer_layer_config._attn_implementation = config.attn
        draft = SpeculatorModel.from_pretrained(
            config.name, config=draft_config, revision=config.revision, verifier=model_config.name
        )
        self.algorithm = draft.config.speculators_config.algorithm
        upstream_defaults = TrainConfig(speculator_type=self.algorithm).flatten()
        unknown = config.training.keys() - upstream_defaults.keys()
        if unknown:
            raise ValueError(f"Unknown speculator training options: {sorted(unknown)}")
        upstream = TrainConfig.from_flat({**upstream_defaults, **config.training})
        self.loss_kwargs, _ = draft.get_trainer_kwargs(**upstream.flatten())
        self.layer_ids = list(draft.target_layer_ids)
        language_model = get_language_model(model)
        layers = language_model.layers
        self.norm = getattr(language_model, "norm", None) or language_model.norm_f
        self.embedding = getattr(language_model, "embed_tokens", None) or language_model.embeddings
        for layer_id in self.layer_ids:
            if not 0 <= layer_id <= len(layers):
                raise ValueError(f"Draft requests hidden state {layer_id}, but the policy has {len(layers)} layers")
            module = layers[layer_id] if layer_id < len(layers) else self.norm
            self.handles.append(module.register_forward_pre_hook(self._capture(layer_id), with_kwargs=True))
        self.handles.append(self.norm.register_forward_pre_hook(self._capture("final"), with_kwargs=True))

        if config.gradient_checkpointing:
            draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        draft.to(device="cuda", dtype=getattr(torch, model_config.optimization_dtype))
        fsdp_options = dict(
            mesh=dims.get_mesh("hsdp"),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=getattr(torch, model_config.reduce_dtype)
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=True) if model_config.fsdp_cpu_offload else OffloadPolicy(),
            reshard_after_forward=model_config.reshard_after_forward,
        )
        for layer in draft.layers:
            fully_shard(layer, **fsdp_options)
        for name in ("embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm"):
            module = getattr(draft, name, None)
            if module is not None:
                fully_shard(module, **fsdp_options)
        fully_shard(draft, **fsdp_options)
        model.add_module("speculator", draft)
        self.draft = draft

    def _capture(self, key):
        def capture(module, args, kwargs):
            if self.capturing:
                hidden = args[0] if args else kwargs["hidden_states"]
                self.features[key] = hidden.detach()

        return capture

    @contextmanager
    def capture(self):
        self.features.clear()
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False

    @torch.no_grad()
    def refresh_verifier_weights(self, *, reset_metrics: bool = True):
        """Frozen target-owned projections follow the live policy, including after resume."""
        if reset_metrics:
            self.metric_totals.clear()
        copy_sharded_weight(self.draft.embed_tokens.weight, self.embedding.weight)
        rows = self.draft.t2d if getattr(self.draft, "use_draft_vocab", False) else None
        copy_sharded_weight(self.draft.verifier_lm_head.weight, self.model.lm_head.weight, rows)
        if not getattr(self.draft, "use_draft_vocab", False):
            copy_sharded_weight(self.draft.lm_head.weight, self.model.lm_head.weight)
        if hasattr(self.draft, "verifier_norm"):
            copy_sharded_weight(self.draft.verifier_norm.weight, self.norm.weight)

    def loss(self, micro_batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = torch.cat([self.features.pop(i) for i in self.layer_ids], dim=-1)
        final = self.features.pop("final")
        if self.cp_context.cp_enabled:
            context = self.cp_context
            hidden = gather_for_cp_wo_grad(hidden, context.cp_world_size, context.cp_group)
            final = gather_for_cp_wo_grad(final, context.cp_world_size, context.cp_group)
        ids = micro_batch["input_ids"].to(hidden.device)
        lengths = micro_batch["seq_lens"].to(hidden.device)
        block_size = getattr(self.draft, "block_size", 1)
        mask = draft_loss_mask(
            micro_batch["loss_mask"].to(hidden.device),
            lengths,
            block_size,
            getattr(self.draft.config, "sample_from_anchor", False),
        )
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=hidden.device), lengths, output_size=ids.numel()
        ).view_as(ids)
        batch = dict(
            hidden_states=hidden,
            verifier_last_hidden_states=final,
            input_ids=ids,
            loss_mask=mask,
            document_ids=document_ids,
            position_ids=micro_batch["position_ids"].to(hidden.device),
        )
        if self.algorithm in ("eagle3", "peagle"):
            batch["hidden_states"] = hidden[:, :-1]
            for key in ("input_ids", "verifier_last_hidden_states", "loss_mask", "position_ids", "document_ids"):
                batch[key] = batch[key][:, 1:]
            batch["loss_mask"] = batch["loss_mask"] & (document_ids[:, 1:] == document_ids[:, :-1])
        elif self.algorithm == "mtp":
            batch["hidden_states"] = batch.pop("verifier_last_hidden_states")
        _, loss, metrics = self.draft(**batch, **self.loss_kwargs)
        for key, value in metrics.items():
            self.metric_totals[key] = self.metric_totals.get(key, 0) + value.detach().float().sum()
        return loss, metrics

    def metrics(self) -> dict[str, float]:
        keys = sorted(self.metric_totals)
        values = torch.stack([self.metric_totals[key] for key in keys])
        dist.all_reduce(values)
        totals = dict(zip(keys, values.tolist()))
        return {
            f"speculator/{key[:-4]}": value / totals[key[:-4] + "_total"]
            for key, value in totals.items()
            if key.endswith("_sum") and totals.get(key[:-4] + "_total", 0) > 0
        }
