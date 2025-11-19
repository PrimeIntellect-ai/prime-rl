import torch
import torch.nn as nn


def _run_lora_grouped_mm(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, offsets: torch.LongTensor):
    _a_out = torch._grouped_mm(x, lora_A, offsets)
    lora_out = torch._grouped_mm(_a_out, lora_B, offsets)
    return lora_out


def _run_lora_for_loop(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, offsets: torch.LongTensor):
    lora_out_splits = []
    for i in range(offsets.shape[0]):
        if i == 0:
            _a_out = torch.matmul(x[0 : offsets[i]], lora_A[i])
            lora_out = torch.matmul(_a_out, lora_B[i])
        else:
            _a_out = torch.matmul(x[offsets[i - 1] : offsets[i]], lora_A[i])
            lora_out = torch.matmul(_a_out, lora_B[i])
        lora_out_splits.append(lora_out)
    return torch.cat(lora_out_splits, dim=0)


class MultiLoRALinear(nn.Module):
    """
    Linear + multi-LoRA with grouped GEMM (batched matmul).

    - Base weight: standard nn.Linear
    - LoRA weights:
        lora_A: [n_adapters, in_features, r]
        lora_B: [n_adapters, r, out_features]

    Forward:
        y = x @ W^T + b + lora(x; adapter_ids)

    adapter_ids:
        Long tensor, one adapter index per *element* in the flattened batch.
        Example shapes:
          - for x: [B, in], adapter_ids: [B]
          - for x: [B, T, in], adapter_ids: [B, T]
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        n_adapters: int,
        alpha: float = 16.0,
        use_grouped_mm: bool = True,
    ):
        super().__init__()
        if r <= 0 or n_adapters <= 0:
            raise ValueError("r and n_adapters must be > 0")

        # Set use_grouped_mm to False if CUDA compute capability < 9.0
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cc_major, _ = torch.cuda.get_device_capability(device)
            if cc_major < 9:
                use_grouped_mm = False
        else:
            use_grouped_mm = False

        self.base = base_linear
        self.r = r
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.scaling = alpha / r
        self.use_grouped_mm = use_grouped_mm

        # LoRA weights: one low-rank pair per adapter
        # [n_adapters, in, r]
        self.lora_A = nn.Parameter(
            torch.empty(n_adapters, self.in_features, r, device=self.base.weight.device, dtype=self.base.weight.dtype)
        )
        # [n_adapters, r, out]
        self.lora_B = nn.Parameter(
            torch.empty(n_adapters, r, self.out_features, device=self.base.weight.device, dtype=self.base.weight.dtype)
        )

        self.reset_parameters()

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def reset_parameters(self, index: int | None = None, init_std: float = 1e-4, init_base: bool = False):
        if init_base:
            self.base.reset_parameters()

        if index is None:
            nn.init.normal_(self.lora_A, mean=0.0, std=init_std)
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.normal_(self.lora_A[index], mean=0.0, std=init_std)
            nn.init.zeros_(self.lora_B[index])

    def forward(self, x: torch.Tensor, offsets: torch.LongTensor):
        """
        x: [..., in_features]
        offsets: [n_adapters]
        """
        assert x.dim() == 2
        assert offsets[-1] == x.shape[0]
        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(x, self.lora_A, self.lora_B, offsets)
        else:
            lora_out = _run_lora_for_loop(x, self.lora_A, self.lora_B, offsets)
        return self.base(x) + lora_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(base={self.base}, r={self.r}, n_adapters={self.n_adapters}, alpha={self.alpha})"
        )
