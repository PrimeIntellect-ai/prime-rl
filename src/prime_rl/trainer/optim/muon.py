from collections.abc import Mapping
from typing import Any

import torch
from dion import Muon as DionMuon
from torch import nn

from prime_rl.trainer.models.fusions import RuntimeFusion


class Muon(DionMuon):
    """Run Muon on logical parameter views while retaining fused storage."""

    def __init__(
        self,
        *args: Any,
        parameter_fusions: Mapping[nn.Parameter, tuple[nn.Module, type[RuntimeFusion]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.parameter_fusions = parameter_fusions

    @torch.no_grad()
    def step(self, closure=None):
        if not self.parameter_fusions:
            return super().step(closure)

        original_parameters = []
        logical_parameters = []

        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue

            original_parameters.append((group, group["params"]))
            parameters = []
            for parameter in group["params"]:
                fusion_info = self.parameter_fusions.get(parameter)
                if fusion_info is None or parameter.grad is None:
                    parameters.append(parameter)
                    continue

                module, fusion = fusion_info
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)

                parameter_views = fusion.logical_parameter_views(module, parameter)
                gradient_views = fusion.logical_parameter_views(module, parameter.grad)
                momentum_views = fusion.logical_parameter_views(module, state["momentum"])
                for name, parameter_view in parameter_views.items():
                    logical_parameter = parameter_view.detach()
                    logical_parameter.grad = gradient_views[name]
                    self.state[logical_parameter] = {"momentum": momentum_views[name]}
                    parameters.append(logical_parameter)
                    logical_parameters.append(logical_parameter)

            group["params"] = parameters

        try:
            return super().step(closure)
        finally:
            for group, parameters in original_parameters:
                group["params"] = parameters
            for parameter in logical_parameters:
                del self.state[parameter]
