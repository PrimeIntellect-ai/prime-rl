"""Honor DeepSeek V4's FP32 router setting in the serving gate."""

import torch
import torch.nn.functional as F


def patch_deepseek_v4_router():
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4.nvidia import model

    if getattr(model.GateLinear, "_prime_rl_fp32_router", False):
        return

    original_gate = model.GateLinear

    class DeepseekV4GateLinear(original_gate):
        _prime_rl_fp32_router = True

        def __init__(self, *args, **kwargs):
            config = get_current_vllm_config()
            self.fp32_compute = getattr(config.model_config.hf_config, "moe_router_dtype", None) == "float32"
            if self.fp32_compute:
                if config.kernel_config.moe_backend in model.MEGA_MOE_BACKENDS:
                    raise ValueError("DeepSeek V4 FP32 routing requires a backend that calls GateLinear")
                kwargs["params_dtype"] = torch.float32
                kwargs["out_dtype"] = torch.float32
            super().__init__(*args, **kwargs)

        def forward(self, x):
            if not self.fp32_compute:
                return super().forward(x)
            with torch.autocast(device_type=x.device.type, enabled=False):
                output = F.linear(x.float(), self.weight, self.bias)
            return self._return(output)

    model.GateLinear = DeepseekV4GateLinear
