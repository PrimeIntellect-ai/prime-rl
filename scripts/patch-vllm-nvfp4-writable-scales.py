"""Apply the approved writable-scale fix to the pinned vLLM wheel in an image.

The experiment intentionally builds from the exact upstream nightly wheel.  The
fix is tiny but not yet present in that immutable wheel, so fail closed unless
the source matches the reviewed change exactly.
"""

from pathlib import Path

import vllm

path = Path(vllm.__file__).parent / "model_executor/layers/quantization/utils/flashinfer_fp4_moe.py"
source = path.read_text()
replacements = {
    "a13_scale.max().to(torch.float32).expand(num_experts)": "a13_scale.max().to(torch.float32).repeat(num_experts)",
    "a2_scale.max().to(torch.float32).expand(num_experts)": "a2_scale.max().to(torch.float32).repeat(num_experts)",
}

for old, new in replacements.items():
    old_count = source.count(old)
    new_count = source.count(new)
    if (old_count, new_count) == (2, 0):
        source = source.replace(old, new)
    elif (old_count, new_count) != (0, 2):
        raise RuntimeError(f"Unexpected vLLM source in {path}: {old_count=} {new_count=}")

path.write_text(source)
print(f"Applied writable NVFP4 MoE scale fix to {path}")
