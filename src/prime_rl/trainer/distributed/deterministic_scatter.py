# Adapted from torchtitan/ops/scatter_add.py (BSD-3-Clause).
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Compile-compatible scatter addition with scoped PyTorch determinism."""

import torch


@torch.library.custom_op("prime_rl::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # Keep the cast inside the custom op so compilation preserves rounded inputs.
    dtype = torch.float32 if src.dtype in (torch.bfloat16, torch.float16) else src.dtype
    accumulation_out = out.to(dtype)
    accumulation_src = src.to(dtype)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        result = accumulation_out.scatter_add(0, index, accumulation_src)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return result.to(out.dtype)


@deterministic_scatter_add.register_fake
def _fake(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _setup_context(ctx, inputs, output):
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _backward(ctx, grad_output):
    (index,) = ctx.saved_tensors
    return grad_output, None, torch.gather(grad_output, 0, index)


deterministic_scatter_add.register_autograd(_backward, setup_context=_setup_context)
