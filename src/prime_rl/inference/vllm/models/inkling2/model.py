# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 text model (SM90/SM100).

Decoder layers are tied together by block attention residuals: instead of a
single residual stream, each sublayer input is a learned softmax mix over the
outputs of previous blocks of `n_layers_per_attn_res_block` layers plus the
running partial block. Residual blocks are stored in bf16; mixes are fp32.

Looped checkpoints (`loop_config`, Inkling 2 Mini) run the core layers
`[core_start, core_end)` `num_passes` times: a prelude, then each pass from
the AttnRes state after the prelude, then the coda from the last pass's
state. Each pass of a core layer has its own KV and conv-state caches. In
passes after the first, every core sublayer input (attention q/k/v and MLP)
is blended with a message: a normalized-value AttnRes read of the previous
pass's blocks, merged by a gated low-rank controller.
"""

import os
import re
from collections import Counter
from collections.abc import Callable, Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .attention import Inkling2Attention
from .configs import Inkling2MMConfig, Inkling2ModelConfig
from .conv_state import InklingConvState
from .moe import Inkling2MoE
from .ops.attention import compute_log_scaling_tau
from .ops.attn_res import attn_res
from .ops.loop import blend_rms_norm, pair_rms_norm
from .ops.norm import rms_norm
from .sconv import ATTN_STREAM, MLP_STREAM, Inkling2ShortConv

# TP-sharded short-conv streams save 8x conv-page bytes in the KV cache but add
# two all-gathers per layer; off by default (decode is collective-latency bound).
SHARDED_CONV = os.environ.get("INKLING2_SHARDED_CONV", "0") == "1"

# The sublayer inputs a loop message blends into, in controller order.
LOOP_STREAMS = ("attn", "k", "v", "mlp")


class Inkling2RMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks the weights of per-input norms (the
    block AttnRes mixes apply them in the same kernel)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.bfloat16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs, hidden_size), requires_grad=False)
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by
    `hidden_size**-0.5` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False) for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False)
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class Inkling2LoopControllers(nn.Module):
    """The message controllers of a loop core layer, one per stream
    (`LOOP_STREAMS`), with their weights stacked over the streams.

    A controller blends a loop message into a sublayer input: from the
    rank-`r` projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    The stacked weights are the loaded Parameters themselves, so a weight
    reload writes them in place and there is no derived state to refresh.
    """

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        streams = len(LOOP_STREAMS)
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16 like the checkpoint.
        self.down = nn.Parameter(torch.empty(streams, rank, 2 * hidden_size, dtype=torch.bfloat16), requires_grad=False)
        self.gain_forget = nn.Parameter(
            torch.empty(streams, 2 * hidden_size, rank, dtype=torch.bfloat16), requires_grad=False
        )
        self.down.weight_loader = self._load_down
        self.gain_forget.weight_loader = self._load_gain_forget

    @staticmethod
    def _load_down(param: nn.Parameter, loaded: torch.Tensor, shard_id: int) -> None:
        param.data[shard_id].copy_(loaded)

    def _load_gain_forget(self, param: nn.Parameter, loaded: torch.Tensor, shard_id: int) -> None:
        stream, part = divmod(shard_id, 2)  # part 0: gain, 1: forget
        param.data[stream, part * self.hidden_size : (part + 1) * self.hidden_size].copy_(loaded)

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        pair = pair_rms_norm(dest, message)
        hidden = torch.bmm(pair, self.down[streams].transpose(1, 2))
        # fp32 gates from the bf16 projection, like the trainer.
        gain_forget = torch.bmm(hidden.float(), self.gain_forget[streams].float().transpose(1, 2))
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        # A loop core layer keeps one conv state (and one KV cache) per pass.
        in_loop_core = layer_id in config.loop_core_layers
        num_passes = config.num_loop_passes if in_loop_core else 1

        def conv_state(state_prefix: str) -> InklingConvState:
            return InklingConvState(
                num_kv_heads=num_kv_heads,
                head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                hidden_size=config.hidden_size,
                kernel_size=config.sconv_kernel_size,
                prefix=state_prefix,
                replicate_hidden=not SHARDED_CONV,
            )

        self.conv_state = conv_state(f"{prefix}.conv_state")
        self.pass_conv_states = nn.ModuleList(
            conv_state(f"{prefix}.conv_state.pass{pass_idx}") for pass_idx in range(1, num_passes)
        )
        self.conv_states = (self.conv_state, *self.pass_conv_states)
        self.attn = Inkling2Attention(
            config,
            self.is_local,
            self.conv_state,
            prefix=f"{prefix}.attn",
            pass_conv_states=tuple(self.pass_conv_states),
        )
        eps = config.rms_norm_eps
        # Local layers: attn_norm; full layers: q/k/v_input_norm, stacked.
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(config.hidden_size, eps, num_inputs=self.num_attn_inputs)
        self.mlp_norm = Inkling2RMSNorm(config.hidden_size, eps)
        self.controllers = (
            Inkling2LoopControllers(config.hidden_size, config.loop_config["gate_rank"]) if in_loop_core else None
        )
        # A looped model routes once per layer visit.
        visit_ids = config.layer_visit_ids(layer_id) if config.loop_config is not None else None
        self.mlp = Inkling2MoE(config, quant_config, f"{prefix}.mlp", visit_ids)
        # Hidden-stream conv states are TP-sharded (each rank convolves the
        # hidden slice of its KV heads), so the conv weights hold that slice.
        hidden_shard = (
            config.hidden_size // get_tensor_model_parallel_world_size() if SHARDED_CONV else config.hidden_size
        )
        self.attn_sconv = Inkling2ShortConv(hidden_shard, config.sconv_kernel_size, self.conv_state, ATTN_STREAM)
        self.mlp_sconv = Inkling2ShortConv(hidden_shard, config.sconv_kernel_size, self.conv_state, MLP_STREAM)

    def attention(
        self,
        positions: torch.Tensor,
        inputs: list[torch.Tensor],
        tau: torch.Tensor | None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """This rank's partial attention output on the normalized `inputs`:
        one for q/k/v, or one each (full layers, and local layers after
        their messages are blended in)."""
        if len(inputs) == 1:
            return self.attn(positions, inputs[0], pass_idx=pass_idx)
        return self.attn(positions, inputs[0], inputs[1], inputs[2], None if self.is_local else tau, pass_idx=pass_idx)


class Inkling2Model(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sharded_conv = SHARDED_CONV
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(config.padded_vocab_size, config.hidden_size)
        self.embed_norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, out_dtype=None)
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Layers whose input (the partial block) the DSpark draft reads; layer
        # id num_hidden_layers stands for the final-norm output.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mix(
        self,
        stack: torch.Tensor,
        last: torch.Tensor,
        query: torch.Tensor,
        norm: Inkling2RMSNorm,
        addend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return attn_res(stack, last, addend, query, norm.weight, norm.eps, self.normalize_values)

    def _hidden_conv(
        self,
        conv: Inkling2ShortConv,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        """Short conv of a hidden-stream conv whose state is TP-sharded.

        Each rank convolves its own hidden slice (the slice of its KV heads) and
        the results are all-gathered, so the global hidden vector that the
        AttnRes mixes need is reconstructed without replicating the conv state.
        `owner` is the conv state of the loop pass (default: the layer's).
        """
        if self.tp_size == 1 or not self.sharded_conv:
            return conv(x, positions, addend, residual, out, owner)
        width = conv.shard_width
        shard = x.narrow(1, self.tp_rank * width, width)
        conv_out = conv(shard, positions, owner=owner)
        conv_full = get_tp_group().all_gather(conv_out, dim=1)
        if addend is None:
            return conv_full
        acc = addend.float() if residual is None else residual.float() + addend.float()
        acc = acc + conv_full.float()
        if out is not None:
            out.copy_(acc.to(addend.dtype))
            return out
        return acc.to(addend.dtype)

    def _new_stack(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """The AttnRes state: `stack[:num_blocks]` holds completed blocks and
        `stack[num_blocks]` the running partial block, at first the embeddings."""
        num_tokens, hidden = inputs_embeds.shape
        max_blocks = -(-len(self.layers) // self.block_size)
        stack = inputs_embeds.new_empty((max_blocks + 1, num_tokens, hidden))
        stack[0].copy_(inputs_embeds)
        return stack

    def _log_scaling_tau(self, positions: torch.Tensor) -> torch.Tensor | None:
        if self.config.log_scaling_n_floor is None:
            return None
        return compute_log_scaling_tau(positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha)

    def _run_layer(
        self,
        i: int,
        stack: torch.Tensor,
        num_blocks: int,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        pass_idx: int = 0,
        bank: torch.Tensor | None = None,
    ) -> int:
        """Run decoder layer `i` on the AttnRes state in `stack`, updating it
        in place, and return the number of completed blocks after it.

        `pass_idx` is the loop pass of a core layer. After the first pass,
        `bank` holds the previous pass's blocks, which the layer's messages
        read."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        messages = None
        if bank is not None:
            # All four streams' messages, then the fp32 attention mixes they blend into.
            messages = attn_res(
                bank[:-1], bank[-1], None, self.message_queries[i - self.loop_core.start], None, 0.0, True
            )
            query = self.attn_res.attn_queries[i - 1]
            dests = attn_res(stack[:num_blocks], partial, None, query, None, 0.0, self.normalize_values)
            norm_weight = layer.attn_norm.weight
            if layer.is_local:
                # One mix and one norm, blended with the q, k and v messages.
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            inputs = layer.controllers.blend(slice(0, 3), dests, messages[:3], norm_weight, layer.attn_norm.eps)
        elif num_blocks == 0:
            inputs = [layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)]
        else:
            query = self.attn_res.attn_queries[i - 1]
            inputs = self._mix(stack[:num_blocks], partial, query, layer.attn_norm)
        attn_partial = layer.attention(positions, inputs, tau, pass_idx)
        attn_out = self._hidden_conv(
            layer.attn_sconv,
            tensor_model_parallel_all_reduce(attn_partial),
            positions,
            owner=owner,
        )

        mlp_query = self.attn_res.mlp_queries[i]
        # A message blends into the fp32 mix, before the MLP norm.
        mlp_norm_weight = layer.mlp_norm.weight if messages is None else None
        mix_args = (mlp_query, mlp_norm_weight, layer.mlp_norm.eps, self.normalize_values)
        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        if first:
            mlp_in = attn_res(stack[: num_blocks + 1], attn_out, None, *mix_args)
        else:
            mlp_in = attn_res(stack[:num_blocks], partial, attn_out, *mix_args)
        if messages is not None:
            mlp_in = layer.controllers.blend(
                slice(3, 4), mlp_in, messages[3:], layer.mlp_norm.weight, layer.mlp_norm.eps
            )
        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        self._hidden_conv(
            layer.mlp_sconv,
            layer.mlp(mlp_in[0], pass_idx),
            positions,
            addend=attn_out,
            residual=residual,
            out=stack[num_blocks],
            owner=owner,
        )
        return num_blocks

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        aux_hidden_states = []
        stack = self._new_stack(inputs_embeds)
        num_blocks = 0
        tau = self._log_scaling_tau(positions)
        for i in range(len(self.layers)):
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            num_blocks = self._run_layer(i, stack, num_blocks, positions, tau)

        out = self._mix(
            stack[:num_blocks],
            stack[num_blocks],
            self.attn_res.output_query,
            self.norm,
        )[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopModel(Inkling2Model):
    """Looped Inkling2: the core layers run once per pass, each pass with its
    own caches and, after the first, messages from the previous pass."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config,
        prefix: str,
    ) -> None:
        # `hf_overrides` (a truncated model) are applied after the config's constructor.
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        core = self.loop_core = config.loop_core_layers
        # (layer, pass) visits; the prelude and the coda run as pass 0.
        self.visits = [(i, 0) for i in range(core.start)]
        self.visits += [(i, pass_idx) for pass_idx in range(config.num_loop_passes) for i in core]
        self.visits += [(i, 0) for i in range(core.stop, config.num_hidden_layers)]
        # Message pseudo-queries per core layer, in `LOOP_STREAMS` order and
        # pre-scaled by `hidden_size**-0.5` at load time. The checkpoint holds
        # two tables of two rows per core layer: attn/mlp and k/v.
        self.message_queries = nn.Parameter(
            torch.zeros(len(core), len(LOOP_STREAMS), config.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.message_queries.weight_loader = self._load_message_queries

    def _load_message_queries(self, param: nn.Parameter, loaded: torch.Tensor, shard_id: int) -> None:
        pairs = loaded.view(-1, 2, self.config.hidden_size)
        first, second = ((0, 3), (1, 2))[shard_id]
        param.data[:, first].copy_(pairs[:, 0]).mul_(self.attn_res.scale)
        param.data[:, second].copy_(pairs[:, 1]).mul_(self.attn_res.scale)

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Prelude, `num_passes` core passes, then the coda."""
        if self.aux_hidden_state_layers:
            raise NotImplementedError("DSpark drafts are not supported for looped Inkling2")
        stack = self._new_stack(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        num_blocks = restart_blocks = 0
        bank = None
        for i, pass_idx in self.visits:
            if i == self.loop_core.start:
                if pass_idx == 0:
                    restart_blocks = num_blocks
                else:
                    # Every pass restarts from the state the prelude left. The
                    # first core layer seals the prelude's partial as a block, so
                    # it is still in place at `stack[restart_blocks]`; what the
                    # previous pass added after it is the bank of the messages.
                    bank = stack[restart_blocks + 1 : num_blocks + 1].clone()
                    num_blocks = restart_blocks
            num_blocks = self._run_layer(i, stack, num_blocks, positions, tau, pass_idx, bank if pass_idx else None)
        return self._mix(
            stack[:num_blocks],
            stack[num_blocks],
            self.attn_res.output_query,
            self.norm,
        )[0]


class Inkling2ForCausalLM(nn.Module):
    """Text-only Inkling2 (`Inkling2ForConditionalGeneration` and its looped
    `Inkling2LoopForConditionalGeneration`, minus images)."""

    # Weights mirror the checkpoint names after these substitutions, except for
    # the fused per-layer weights in ``_shards``.
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
        (".mlp.gate.bias", ".mlp.gate.expert_bias"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
        # Looped checkpoints: the message queries and the controllers, stacked
        # over `LOOP_STREAMS`.
        "middle_cycle_cross_pass_queries": ("message_queries", 0),
        "middle_cycle_cross_pass_kv_queries": ("message_queries", 1),
        **{
            f".controllers.{stream}.down_weight": (".controllers.down", index)
            for index, stream in enumerate(LOOP_STREAMS)
        },
        **{
            f".controllers.{stream}.{part}_weight": (".controllers.gain_forget", 2 * index + offset)
            for index, stream in enumerate(LOOP_STREAMS)
            for offset, part in enumerate(("gain", "forget"))
        },
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if vllm_config.model_config.enable_prompt_embeds:
            # The reference applies embed_norm to raw input embeddings, which
            # cannot be told apart from already-normalized ones here.
            raise ValueError("Inkling2 does not support prompt embeds")
        self.config = config
        self.text_config = text_config = config.text_config
        # Per-tensor cache of this rank's expert ids (see `_load_experts`).
        self._local_expert_ids: dict[str, tuple[int, ...]] = {}
        model_cls = Inkling2Model if text_config.loop_config is None else Inkling2LoopModel
        self.model = model_cls(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(text_config.padded_vocab_size, text_config.hidden_size)
        self.logits_processor = LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self.model

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def _load_experts(
        self,
        moe: Inkling2MoE,
        name: str,
        weight: torch.Tensor,
        num_experts: int,
        intermediate_size: int,
    ) -> None:
        """Load a stacked release expert tensor into the (TP/EP) FusedMoE.

        `w13_weight` rows interleave gate and up (row 2i gates output i), so
        each expert's gate/up rows are strided views of the checkpoint tensor;
        the FusedMoE weight loader copies them straight into the kernel layout
        (and does the TP/EP sharding).
        """
        experts = moe.experts.routed_experts
        # Only the rank's own experts need a copy: under expert parallelism the
        # rest of the stacked tensor is not ours. The mapping is static, so cache
        # it across reloads.
        expert_ids = self._local_expert_ids.get(name)
        if expert_ids is None:
            map_fn = experts._map_global_expert_id_to_local_expert_id
            expert_ids = tuple(e for e in range(num_experts) if map_fn(e) != -1)
            self._local_expert_ids[name] = expert_ids
        # The transport either delivers the full stacked tensor or, with sharded
        # expert transfer, only this rank's contiguous expert range.
        local = weight.shape[0] == len(expert_ids)
        assert local or weight.shape[0] == num_experts, (name, weight.shape, expert_ids)
        index = {expert_id: i for i, expert_id in enumerate(expert_ids)} if local else None

        def row(expert_id: int) -> torch.Tensor:
            return weight[index[expert_id]] if index is not None else weight[expert_id]

        if name.endswith("w13_weight"):
            param = experts.w13_weight
            for expert_id in expert_ids:
                loader = param.weight_loader
                loader(
                    param=param,
                    loaded_weight=row(expert_id)[0::2],
                    weight_name=name,
                    shard_id="w1",
                    expert_id=expert_id,
                    return_success=True,
                )
                loader(
                    param=param,
                    loaded_weight=row(expert_id)[1::2],
                    weight_name=name,
                    shard_id="w3",
                    expert_id=expert_id,
                    return_success=True,
                )
        elif name.endswith("w2_weight"):
            param = experts.w2_weight
            for expert_id in expert_ids:
                param.weight_loader(
                    param=param,
                    loaded_weight=row(expert_id),
                    weight_name=name,
                    shard_id="w2",
                    expert_id=expert_id,
                    return_success=True,
                )
        else:
            raise ValueError(f"Unexpected expert tensor: {name}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        num_experts = self.text_config.n_routed_experts
        intermediate_size = self.text_config.intermediate_size
        # Truncated models (e.g. fewer layers via hf_overrides) skip the rest.
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        expert_modules: dict[str, Inkling2MoE] = {}
        for name, weight in weights:
            if name.startswith("model.visual."):
                continue
            orig_name = name
            for old, new in self._renames:
                name = name.replace(old, new)
            if ".mlp.experts." in name:
                # Skip layers the (possibly truncated) model does not have:
                # a 48-layer checkpoint file also holds tensors of layers
                # outside a truncated model.
                m = re.search(r"\.layers\.(\d+)\.", name)
                if m is not None and int(m.group(1)) >= num_layers:
                    continue
                moe = self._find_module(name.split(".mlp.experts.")[0] + ".mlp")
                self._load_experts(moe, name, weight, num_experts, intermediate_size)
                expert_modules[name.split(".mlp.experts.")[0] + ".mlp"] = moe
                loaded.add(name.replace(".mlp.experts.", ".mlp.experts.routed_experts."))
                continue
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name)
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {orig_name}")
            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}")
            loader: Callable[..., None] = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        # When qkvr's rows per TP rank are not a multiple of 8, a padding shard (6) aligns them. No
        # checkpoint tensor fills it, and vLLM rejects a stacked weight with an unfilled shard.
        tp_size = get_tensor_model_parallel_world_size()
        for module in self.modules():
            if isinstance(module, Inkling2Attention) and len(module.split_sizes) > 6:
                weight = module.qkvr.weight
                padding = weight.new_zeros(module.split_sizes[6] * tp_size, weight.shape[1])
                weight.weight_loader(weight, padding, 6)
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        # The router bias is the gate's Parameter, aliased onto the routed
        # experts; it is loaded through `mlp.gate.expert_bias` and re-linked
        # below, so it never arrives under the experts name.
        missing = {name for name in params.keys() - loaded if not name.endswith(".e_score_correction_bias")}
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        for moe in expert_modules.values():
            moe.refresh_router_bias()
        return loaded

    def _find_module(self, dotted_name: str) -> Inkling2MoE:
        module = self
        for part in dotted_name.split("."):
            module = getattr(module, part)
        return module
