# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modifications copyright (c) 2025 Prime Intellect, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import dataclass
from functools import cached_property

import torch.distributed as dist
from torch._utils import _get_available_device_type
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from prime_rl.configs.trainer import ModelConfig
from prime_rl.utils.logger import format_time, get_logger

device_type = _get_available_device_type() or "cuda"

__all__ = ["ParallelDims"]


"""
[ParallelDims Mesh Breakdown]

The trainer's ranks are organized into nested groups, and the mesh dims are the quotients between them.

# Notation

Mesh names double as their sizes. Three operations build one mesh from two dims A and B. Write a and
b for local ranks on A and B (as in DeviceMesh.get_local_rank), so 0 <= a < A and 0 <= b < B:
  - A x B (merge): one dim of size A * B, row-major with B fastest: local rank a * B + b.
  - A_mod_B (split): the row-major split of A into (A_mod_B, B), so A = A_mod_B x B; B must divide A.
    Local rank a on A becomes a % B on B, its position within its group of B ranks, and a // B on
    A_mod_B, which of the A / B such groups it is in.
  - (A, B) (stack): a 2-D mesh that keeps A and B as separate dims.

# Nested groups

The groups are hierarchically nested, cp ⊂ ep ⊂ fsdp ⊂ world (ep only with EP), and each is a
contiguous block of ranks inside the next:
  - cp: ranks that split each sequence into chunks (context parallelism).
  - ep (EP only): ranks that split the routed experts. The expert all-to-all group.
  - fsdp: ranks that together hold one sharded copy of the params, except the routed experts when EP
    is on (see expert_hsdp below).
  - world: every rank.

# Mesh dims

The world mesh is built from the quotients between consecutive groups, slowest-varying first:

  without EP:  dp_replicate, fsdp_mod_cp,            cp, fsdp_mod_vocab
  with EP:     dp_replicate, fsdp_mod_ep, ep_mod_cp, cp, fsdp_mod_vocab

  - cp: which chunk of each sequence the rank holds.
  - ep_mod_cp (EP): which cp group the rank is in, within its EP group.
  - fsdp_mod_cp (no EP): which cp group the rank is in, within its fsdp group.
  - fsdp_mod_ep (EP): which EP group the rank is in, within its fsdp group. Ranks along it hold the same
    experts and FSDP-shard them, so each holds 1 / (ep * fsdp_mod_ep) = 1 / fsdp of the expert params.
  - dp_replicate = world_mod_fsdp: which replica the rank is in. Replicas hold identical params and
    all-reduce gradients.
  - fsdp_mod_vocab: always size 1, see vocab below.

Size-1 dims are dropped, except fsdp_mod_cp and fsdp_mod_ep, which are always built, and fsdp_mod_vocab,
which is built whenever fsdp > 1.

# Other named meshes

  - fsdp = fsdp_mod_cp x cp, or fsdp_mod_ep x ep_mod_cp x cp with EP: the FSDP shard group.
  - ep = ep_mod_cp x cp (EP): the expert all-to-all group.
  - dp = world_mod_cp: ranks with distinct data; sizes the data loader and token counts.
  - world: every rank, flattened into one dim; all-reduces the loss and token counts.
  - vocab = fsdp under a second name: the group over which NGramEmbedding's table is vocab-parallel
    (EmbeddingParallel). Like EP, the sharded weights are never gathered; token ids move instead. Each
    rank holds a slice of the table's rows; the group all-gathers every rank's token ids (padded to equal
    length), each rank looks up the rows it owns (zeros elsewhere), and a reduce-scatter sums the partial
    outputs and returns each rank's own tokens. Since vocab = fsdp, fsdp_mod_vocab = fsdp / vocab = 1.

# 2-D FSDP meshes

Each FSDP mesh is (dp_replicate, shard dim) when dp_replicate > 1, and just the 1-D shard dim otherwise
(the hsdp names are kept in that case).
Params are sharded along the shard dim and replicated along dp_replicate.
  - hsdp = (dp_replicate, fsdp): every other fully_shard (vision encoder, router, blocks, embeddings,
    lm_head and norm, root).
  - expert_hsdp = (dp_replicate, fsdp_mod_ep): the expert params. Each expert slice has one owner per EP
    group, and expert_hsdp spans exactly those owners: ep * expert_hsdp = world, which ParallelDims
    checks, so every token's gradient reaches each slice exactly once.
  - vocab_hsdp = (dp_replicate, fsdp_mod_vocab): the NGramEmbedding.

# Example

With EP: 16 GPUs, dp_replicate=2, cp=2, ep=4, so fsdp=8, fsdp_mod_ep=2, and ep_mod_cp=2.

                             dp_replicate=0                               dp_replicate=1
               +--------------------------------------+  +--------------------------------------+
               |  fsdp_mod_ep=0      fsdp_mod_ep=1    |  |  fsdp_mod_ep=0      fsdp_mod_ep=1    |
               |  +-------------+    +-------------+  |  |  +-------------+    +-------------+  |
               |  | +---------+ |    | +---------+ |  |  |  | +---------+ |    | +---------+ |  |
  ep_mod_cp=0  |  | |  0    1 | |    | |  4    5 | |  |  |  | |  8    9 | |    | | 12   13 | |  |
               |  | +---------+ |    | +---------+ |  |  |  | +---------+ |    | +---------+ |  |
               |  | +---------+ |    | +---------+ |  |  |  | +---------+ |    | +---------+ |  |
  ep_mod_cp=1  |  | |  2    3 | |    | |  6    7 | |  |  |  | | 10   11 | |    | | 14   15 | |  |
               |  | +---------+ |    | +---------+ |  |  |  | +---------+ |    | +---------+ |  |
               |  +-------------+    +-------------+  |  |  +-------------+    +-------------+  |
               +--------------------------------------+  +--------------------------------------+

Boxes nest as cp ⊂ ep ⊂ fsdp: outer boxes are fsdp groups, middle boxes EP groups, inner boxes cp groups
(left rank cp=0, right cp=1). Non-expert params shard across each outer box. Ranks at the same position
in different EP groups own the same expert slice: they shard it along fsdp_mod_ep ({0, 4}) and replicate
it along dp_replicate ({0, 8}), so expert_hsdp for that slice is {0, 4, 8, 12}.
Without EP, erase the middle boxes: fsdp_mod_ep x ep_mod_cp becomes the single dim fsdp_mod_cp.

# Design choices, not constraints

The nesting cp ⊂ ep ⊂ fsdp ⊂ world, and the divisibility it implies, is a reasonable design choice
rather than a requirement of the parallelisms themselves; ParallelDims raises NotImplementedError for
EP layouts outside it. Other valid layouts include:
  - cp outside fsdp: shard params over the data dims only, keeping cp intra-node and FSDP traffic
    inter-node; cp ranks then hold identical params and all-reduce gradients, like replicas.
  - ep spanning replicas: split experts over up to the whole world, with experts getting their own
    data-parallel group of world / ep ranks; the expert all-to-all then crosses the slowest dim.
  - ep not containing cp: ep inside cp, or independent of it. expert_hsdp must still satisfy
    ep * expert_hsdp = world, so its shard part then includes part of cp and is no longer one dim.
  - independent expert replication: correctness only needs ep * expert_hsdp = world, so the expert
    params' replication degree could differ from dp_replicate, trading expert memory for communication.
The current nesting gives expert and non-expert params the same 1 / fsdp share per rank, keeps the
expert all-to-all within one replica, makes the expert FSDP shard group a single dim (fsdp_mod_ep),
and builds every group from one mesh.
"""


@dataclass
class ParallelDims:
    dp_replicate: int
    cp: int
    ep: int
    world_size: int

    _world_mesh: DeviceMesh = None
    _submeshes: dict = None

    def __post_init__(self):
        self._submeshes = {}
        self._validate()

    def _validate(self):
        for name, degree in (("dp_replicate", self.dp_replicate), ("cp", self.cp), ("ep", self.ep)):
            if degree < 1:
                raise ValueError(f"{name} ({degree}) must be >= 1")

        if self.world_size % (self.dp_replicate * self.cp) != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"dp_replicate ({self.dp_replicate}) * cp ({self.cp})"
            )

        if self.ep > 1:
            if self.ep % self.cp != 0:
                raise NotImplementedError(
                    f"ep ({self.ep}) must be a multiple of cp ({self.cp}): EP groups that do not contain whole cp "
                    "groups are a valid layout but not implemented. See [ParallelDims Mesh Breakdown]."
                )
            if self.fsdp % self.ep != 0:
                raise NotImplementedError(
                    f"world_size / dp_replicate ({self.fsdp}) must be a multiple of ep ({self.ep}): EP groups that "
                    "span replicas are a valid layout but not implemented. See [ParallelDims Mesh Breakdown]."
                )

    def build_mesh(self) -> DeviceMesh:
        mesh = self._build_mesh_with_ep() if self.ep > 1 else self._build_mesh_without_ep()
        self._submeshes["vocab"] = self._submeshes["fsdp"]
        if "fsdp_mod_vocab" in mesh.mesh_dim_names:
            self._submeshes["vocab_hsdp"] = self._slice_hsdp(mesh, "fsdp_mod_vocab")
        return mesh

    def _slice_hsdp(self, mesh: DeviceMesh, shard_dim_name: str) -> DeviceMesh:
        if self.dp_replicate_enabled:
            return mesh["dp_replicate", shard_dim_name]
        return mesh[shard_dim_name]

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # See [ParallelDims Mesh Breakdown].
        fsdp_mod_ep = self.fsdp // self.ep
        ep_mod_cp = self.ep // self.cp

        dims = []
        names = []
        for d, name in zip(
            [
                self.dp_replicate,
                fsdp_mod_ep,
                ep_mod_cp,
                self.cp,
                1,
            ],
            ["dp_replicate", "fsdp_mod_ep", "ep_mod_cp", "cp", "fsdp_mod_vocab"],
        ):
            # fsdp_mod_ep is needed even if it's 1, whose FSDP wrapping
            # helps the MoE layers do mixed precision training
            if d > 1 or name in ("fsdp_mod_ep", "fsdp_mod_vocab"):
                dims.append(d)
                names.append(name)

        self.logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        t0 = time.perf_counter()
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        self.logger.debug(f"Built device mesh in {format_time(time.perf_counter() - t0)}")

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        dp_mesh_dim_names = []
        fsdp_mesh_dim_names = []
        world_mesh_dim_names = []
        ep_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            world_mesh_dim_names.append("dp_replicate")
        # fsdp_mod_ep is always needed, even if it's 1
        dp_mesh_dim_names.append("fsdp_mod_ep")
        fsdp_mesh_dim_names.append("fsdp_mod_ep")
        world_mesh_dim_names.append("fsdp_mod_ep")
        if "ep_mod_cp" in names:
            dp_mesh_dim_names.append("ep_mod_cp")
            fsdp_mesh_dim_names.append("ep_mod_cp")
            world_mesh_dim_names.append("ep_mod_cp")
            ep_mesh_dim_names.append("ep_mod_cp")
        if self.cp_enabled:
            fsdp_mesh_dim_names.append("cp")
            world_mesh_dim_names.append("cp")
            ep_mesh_dim_names.append("cp")

        self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        self._submeshes["fsdp"] = mesh[tuple(fsdp_mesh_dim_names)]._flatten(mesh_dim_name="fsdp")
        self._submeshes["world"] = mesh[tuple(world_mesh_dim_names)]._flatten(mesh_dim_name="world")
        self._submeshes["ep"] = mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        if self.dp_replicate_enabled:
            parent = mesh[tuple(["dp_replicate"] + fsdp_mesh_dim_names)]
            hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
            self._submeshes["hsdp"] = DeviceMesh(device_type, hsdp_tensor, mesh_dim_names=("dp_replicate", "fsdp"))
        else:
            self._submeshes["hsdp"] = self._submeshes["fsdp"]

        self._submeshes["expert_hsdp"] = self._slice_hsdp(mesh, "fsdp_mod_ep")
        assert self.ep * self._submeshes["expert_hsdp"].size() == self.world_size

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
        # See [ParallelDims Mesh Breakdown].
        fsdp_mod_cp = self.fsdp // self.cp

        dims = []
        names = []
        for d, name in zip(
            [self.dp_replicate, fsdp_mod_cp, self.cp, 1],
            ["dp_replicate", "fsdp_mod_cp", "cp", "fsdp_mod_vocab"],
        ):
            if d > 1 or name == "fsdp_mod_cp" or (name == "fsdp_mod_vocab" and self.fsdp > 1):
                dims.append(d)
                names.append(name)

        self.logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        t0 = time.perf_counter()
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        self.logger.debug(f"Built device mesh in {format_time(time.perf_counter() - t0)}")

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        dp_mesh_dim_names = []
        fsdp_mesh_dim_names = []
        world_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            world_mesh_dim_names.append("dp_replicate")
        dp_mesh_dim_names.append("fsdp_mod_cp")
        fsdp_mesh_dim_names.append("fsdp_mod_cp")
        world_mesh_dim_names.append("fsdp_mod_cp")
        if self.cp_enabled:
            fsdp_mesh_dim_names.append("cp")
            world_mesh_dim_names.append("cp")

        self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        self._submeshes["fsdp"] = mesh[tuple(fsdp_mesh_dim_names)]._flatten(mesh_dim_name="fsdp")
        self._submeshes["world"] = mesh[tuple(world_mesh_dim_names)]._flatten(mesh_dim_name="world")

        if self.dp_replicate_enabled:
            parent = mesh[tuple(["dp_replicate"] + fsdp_mesh_dim_names)]
            hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
            self._submeshes["hsdp"] = DeviceMesh(device_type, hsdp_tensor, mesh_dim_names=("dp_replicate", "fsdp"))
        else:
            self._submeshes["hsdp"] = self._submeshes["fsdp"]

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        # doing late init so ParallelDims can still be used as a lightweight
        # dataclass without having to initialize the world mesh
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    def get_mesh(self, name: str) -> DeviceMesh:
        mesh = self.world_mesh  # ensure lazy init has run
        if name in self._submeshes:
            return self._submeshes[name]
        return mesh[name]

    @property
    def fsdp(self) -> int:
        return self.world_size // self.dp_replicate

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def fsdp_enabled(self):
        return self.fsdp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @cached_property
    def seq_len_divisor(self):
        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.cp * 2

    @cached_property
    def logger(self):
        return get_logger()


def _is_moe_model(config: ModelConfig) -> bool:
    """Return True if the model has MoE layers, by loading its HuggingFace config."""
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    model_config = getattr(model_config, "text_config", model_config)
    return hasattr(model_config, "num_experts") or hasattr(model_config, "n_routed_experts")


def resolve_ep(config: ModelConfig) -> None:
    """Resolve ``ep="auto"`` in-place to a concrete integer.

    For MoE models, resolves to ``min(fsdp_island_size, 8)`` where
    ``fsdp_island_size = world_size // dp_replicate``. For non-MoE
    models, resolves to 1 (no-op).
    """
    if config.ep != "auto":
        return

    world_size = dist.get_world_size()

    if not _is_moe_model(config):
        config.ep = 1
        get_logger().info("EP auto: model is not MoE, resolving ep=1")
        return

    dp_replicate = config.dp_replicate
    fsdp_island_size = world_size // dp_replicate
    resolved_ep = min(fsdp_island_size, 8)

    config.ep = resolved_ep
    get_logger().info(f"EP auto: world_size={world_size}, dp_replicate={dp_replicate} -> resolved ep={resolved_ep}")


def get_parallel_dims(config: ModelConfig, seq_len: int | None = None) -> ParallelDims:
    assert isinstance(config.ep, int), (
        f"config.ep must be resolved to an int before get_parallel_dims; got {config.ep!r}. "
        "Call resolve_ep(config) first."
    )

    # Initialize parallel dimensions
    parallel_dims = ParallelDims(
        dp_replicate=config.dp_replicate,
        cp=config.cp,
        ep=config.ep,
        world_size=dist.get_world_size(),
    )

    # Validate sequence length against parallel dimensions requirements
    if seq_len is not None and seq_len % parallel_dims.seq_len_divisor != 0:
        raise ValueError(
            f"Sequence length ({seq_len}) must be divisible by "
            f"seq_len_divisor ({parallel_dims.seq_len_divisor}) for the given parallel dimensions. "
            f"This requirement comes from context parallel (CP={config.cp})."
        )

    return parallel_dims
