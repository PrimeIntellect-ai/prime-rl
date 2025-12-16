import math
from typing import Any

import torch
from torch import nn

_LORA_PREFIX = "base_layer."

OFFSETS = None


def set_offsets(offsets: torch.Tensor, reset_reference: bool = False) -> None:
    global OFFSETS
    if OFFSETS is None or reset_reference:
        OFFSETS = offsets
    else:
        OFFSETS.copy_(offsets)


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer.
    Implements the low-rank decomposition: ΔW = B @ A
    where A ∈ R^(rank x in_features), B ∈ R^(out_features x rank)
    Forward pass: y = x @ (W + ΔW).T = x @ W.T + x @ A.T @ B.T * (alpha / rank)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        in_features: int | None = None,
        out_features: int | None = None,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if in_features is None:
            in_features = base_layer.in_features
        if out_features is None:
            out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    def _init_lora_parameters(self, generator: torch.Generator | None = None):
        """Initialize LoRA parameters following standard LoRA initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + lora_output"""
        base_output = self.base_layer(x)
        lora_x = self.lora_dropout(x)
        lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_output + lora_output

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_layer, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.base_layer.__getitem__(key)  # type: ignore[operator]

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this LoRA module is executed.
        For ``LoRALinear``, it will strip LoRA module prefix,
        so that this module can be loaded into non-LoRALinear modules.
        It would still be able to be loaded into LoRALinear modules as this class,
        adds the prefix back before loading the state_dict.
        """
        old_prefix = f"{prefix}{_LORA_PREFIX}"
        new_prefix = prefix
        for key in list(state_dict.keys()):
            if not key.startswith(old_prefix):
                continue
            new_key = new_prefix + key[len(old_prefix) :]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_load_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.
        For ``LoRALinear``, it will add back the module
        prefix so that non-LoRALinear modules can be loaded into
        LoRALinear modules properly.
        """
        old_prefix = prefix
        new_prefix = f"{prefix}{_LORA_PREFIX}"
        for key in list(state_dict.keys()):
            if not key.startswith(old_prefix) or key.endswith("lora_A") or key.endswith("lora_B"):
                continue
            new_key = new_prefix + key[len(old_prefix) :]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]


def _run_lora_grouped_mm(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, offsets: torch.LongTensor):
    _a_out = torch._grouped_mm(x, lora_A.transpose(-2, -1), offsets)
    lora_out = torch._grouped_mm(_a_out, lora_B.transpose(-2, -1), offsets)
    return lora_out


def _run_lora_for_loop(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, offsets: torch.LongTensor):
    lora_out_splits = []
    for i in range(offsets.shape[0]):
        if i == 0:
            _a_out = torch.matmul(x[0 : offsets[i]], lora_A[i].transpose(-2, -1))
            lora_out = torch.matmul(_a_out, lora_B[i].transpose(-2, -1))
        else:
            _a_out = torch.matmul(x[offsets[i - 1] : offsets[i]], lora_A[i].transpose(-2, -1))
            lora_out = torch.matmul(_a_out, lora_B[i].transpose(-2, -1))
        lora_out_splits.append(lora_out)
    return torch.cat(lora_out_splits, dim=0)


class MultiLoRALinear(nn.Module):
    """
    Linear + multi-LoRA with grouped GEMM.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        n_adapters: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_grouped_mm: bool = True,
    ):
        super().__init__()
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")

        # Set use_grouped_mm to False if CUDA compute capability < 9.0
        if torch.cuda.is_available():
            cc_major, _ = torch.cuda.get_device_capability()
            if cc_major != 9:
                use_grouped_mm = False
        else:
            use_grouped_mm = False
        if rank % 8 != 0 or base_layer.in_features % 8 != 0 or base_layer.out_features % 8 != 0:
            use_grouped_mm = False

        self.base_layer = base_layer
        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = alpha / rank
        self.use_grouped_mm = use_grouped_mm
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # LoRA weights: one low-rank pair per adapter
        # [n_adapters, in, r]
        self.lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        rank,
                        self.in_features,
                        device=self.base_layer.weight.device,
                        dtype=self.base_layer.weight.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        # [n_adapters, r, out]
        self.lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.out_features,
                        rank,
                        device=self.base_layer.weight.device,
                        dtype=self.base_layer.weight.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.reset_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self, index: int | None = None):
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            nn.init.kaiming_uniform_(self.lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[index])

    def forward(self, x: torch.Tensor):
        """
        x: [..., in_features]
        offsets: [n_adapters]
        """
        global OFFSETS
        offsets = OFFSETS
        ori_shape = x.shape
        new_shape = ori_shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        assert offsets[-1] == x.shape[0]

        base_out = self.base_layer(x)
        lora_x = self.lora_dropout(x)

        combined_lora_A = torch.stack([i for i in self.lora_A], dim=0)
        combined_lora_B = torch.stack([i for i in self.lora_B], dim=0)
        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(lora_x, combined_lora_A, combined_lora_B, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, combined_lora_A, combined_lora_B, offsets)
        return (base_out + self.scaling * lora_out).view(new_shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, n_adapters={self.n_adapters}, alpha={self.alpha}, dropout={self.lora_dropout})"
