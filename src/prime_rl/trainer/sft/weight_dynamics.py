"""Instrumentation for the FP8 weight-dynamics experiments (int4/experiments/fp8-qat-weight-dynamics in research-prod)."""

import hashlib
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.utils.logger import get_logger
from prime_rl.utils.weights import resolve_fqn

UPDATE_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _local(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if isinstance(param, DTensor) else param


def _full(param: torch.Tensor) -> torch.Tensor:
    return param.full_tensor() if isinstance(param, DTensor) else param


def _canonical_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {resolve_fqn(model, name): param for name, param in model.named_parameters()}


def _hf_tensors(snapshot: Path, names: set[str]) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in names.intersection(f.keys()):
                tensors[name] = f.get_tensor(name)
    assert set(tensors) == names, f"missing in {snapshot}: {names - set(tensors)}"
    return tensors


def check_start_weights(model: nn.Module, snapshot: Path, num_layers: int) -> None:
    """Assert that sampled q_proj and expert-0 weights equal the bf16 tensors in ``snapshot``, on every rank."""
    logger = get_logger()
    params = _canonical_params(model)
    layers = (0, num_layers - 1)
    pairs = []
    for layer in layers:
        prefix = f"model.layers.{layer}"
        q_rows = None
        if f"{prefix}.self_attn.q_proj.weight" in params:
            q_param = params[f"{prefix}.self_attn.q_proj.weight"]
        else:
            q_param = params[f"{prefix}.self_attn.qkv_proj.weight"]
            q_rows = model.config.num_attention_heads * model.config.head_dim
        pairs.append((f"{prefix}.self_attn.q_proj.weight", q_param, lambda w, r=q_rows: w if r is None else w[:r]))
        if f"{prefix}.mlp.experts.gate_proj" in params:
            gate_param, gate_select = params[f"{prefix}.mlp.experts.gate_proj"], lambda w: w[0]
        else:
            gate_param = params[f"{prefix}.mlp.experts.gate_up_proj"]
            gate_select = lambda w, r=model.config.moe_intermediate_size: w[0, :r]  # noqa: E731
        pairs.append((f"{prefix}.mlp.experts.0.gate_proj.weight", gate_param, gate_select))
        pairs.append(
            (f"{prefix}.mlp.experts.0.down_proj.weight", params[f"{prefix}.mlp.experts.down_proj"], lambda w: w[0])
        )
    expected = _hf_tensors(snapshot, {hf_name for hf_name, _, _ in pairs})
    for hf_name, param, select in pairs:
        loaded = select(_full(param.detach())).to(torch.bfloat16).cpu()
        digest = hashlib.sha256(loaded.view(torch.int16).numpy().tobytes()).hexdigest()[:16]
        matches = torch.equal(loaded, expected[hf_name])
        logger.info(f"Start weight check {hf_name}: sha256={digest} matches_snapshot={matches}")
        assert matches, f"{hf_name} does not match {snapshot}"


class WeightUpdateTracker:
    """Tracks update RMS, update autocorrelation at UPDATE_LAGS, and RMS drift from the start on a fixed
    random element subset of 16 local weight shards (attention and expert matrices at four depths)."""

    def __init__(self, model: nn.Module, num_layers: int, num_elements: int = 65536, seed: int = 0):
        params = _canonical_params(model)
        layers = sorted({0, num_layers // 3, 2 * num_layers // 3, num_layers - 1})
        names = []
        for layer in layers:
            prefix = f"model.layers.{layer}"
            qkv = f"{prefix}.self_attn.qkv_proj.weight"
            names.append(qkv if qkv in params else f"{prefix}.self_attn.q_proj.weight")
            names.append(f"{prefix}.self_attn.o_proj.weight")
            gate_up = f"{prefix}.mlp.experts.gate_up_proj"
            names.append(gate_up if gate_up in params else f"{prefix}.mlp.experts.gate_proj")
            names.append(f"{prefix}.mlp.experts.down_proj")
        generator = torch.Generator().manual_seed(seed)
        self.tags = [re.sub(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.", r"L\1.", name) for name in names]
        self.params = [params[name] for name in names]
        self.indices = []
        for param in self.params:
            numel = _local(param).numel()
            count = min(num_elements, numel)
            self.indices.append(torch.randperm(numel, generator=generator)[:count].to(_local(param).device))
        self.start = [self._sample(i) for i in range(len(self.params))]
        self.previous = [s.clone() for s in self.start]
        self.history = [torch.zeros(UPDATE_LAGS[-1], s.numel(), device=s.device) for s in self.start]
        self.num_updates = 0
        get_logger().info(f"Tracking weight updates for {self.tags}")

    def _layout(self) -> dict:
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        return {"world_size": world_size, "local_numel": [_local(param).numel() for param in self.params]}

    def save(self, path: Path) -> None:
        """Write the start sample, update history and update count, so a resumed run continues every lag."""
        state = {
            "layout": self._layout(),
            "tags": self.tags,
            "start": [start.cpu() for start in self.start],
            "history": [history.cpu() for history in self.history],
            "num_updates": self.num_updates,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Restore what ``save`` wrote; ``previous`` is already the loaded weights' sample from construction."""
        state = torch.load(path, map_location="cpu")
        assert state["layout"] == self._layout(), f"{path} was written with a different layout"
        assert state["tags"] == self.tags, (state["tags"], self.tags)
        self.start = [start.to(current.device) for start, current in zip(state["start"], self.start)]
        self.history = [history.to(current.device) for history, current in zip(state["history"], self.history)]
        assert [h.shape for h in self.history] == [torch.Size([UPDATE_LAGS[-1], s.numel()]) for s in self.start]
        self.num_updates = state["num_updates"]
        get_logger().info(f"Resumed weight-update tracking after {self.num_updates} updates from {path}")

    def _sample(self, i: int) -> torch.Tensor:
        return _local(self.params[i]).detach().reshape(-1)[self.indices[i]].float()

    @torch.no_grad()
    def step(self) -> dict[str, float]:
        keys, values = [], []
        slot = self.num_updates % UPDATE_LAGS[-1]
        for i, tag in enumerate(self.tags):
            current = self._sample(i)
            update = current - self.previous[i]
            self.previous[i] = current
            n = math.sqrt(update.numel())
            keys += [f"weight_dynamics/{tag}/update_rms", f"weight_dynamics/{tag}/delta_rms"]
            values += [update.norm() / n, (current - self.start[i]).norm() / n]
            for lag in UPDATE_LAGS:
                if self.num_updates >= lag:
                    past = self.history[i][(self.num_updates - lag) % UPDATE_LAGS[-1]]
                    keys.append(f"weight_dynamics/{tag}/update_cos_lag{lag}")
                    values.append(torch.nn.functional.cosine_similarity(update, past, dim=0))
            self.history[i][slot] = update
        self.num_updates += 1
        return dict(zip(keys, torch.stack(values).tolist()))
