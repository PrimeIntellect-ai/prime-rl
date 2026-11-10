"""Offline check of the export's key mapping on the real Nemotron 3.5 Super checkpoint, without weights.

Every model key of a real training checkpoint (its DCP ``.metadata``: the same names and shapes ``model.state_dict()``
yields during training), converted one key at a time with prime-rl's NemotronH chain as ``collect_hf_tensors``
does, must give exactly the text-only base's fine-tuned keys (everything but ``mtp.*``) with the base's shapes, and
the base's tensors must split into fine-tuned + kept-as-is exactly as ``hf_export`` splits them. Also checks that no
fine-tuned tensor travels in a narrower dtype than the base stores it in.

Inputs (skipped when unset):
  PRL_TEST_DCP_METADATA   a checkpoint's ``.metadata`` (e.g. step_104 of the swe-marathon run)
  PRL_TEST_TEXT_CONFIG    the base's config.json (/volume/models/nemotron-3.5-super-text)
  PRL_TEST_TEXT_INDEX     the base's model.safetensors.index.json
  PRL_TEST_HEADERS_DIR    safetensors headers as JSON (one file per shard) of the text base, or of the VL release
                          (keys prefixed with ``language_model.``, which is stripped)
"""

import json
import os
import pickle
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import torch

PATHS = {
    name: os.environ.get(name)
    for name in ("PRL_TEST_DCP_METADATA", "PRL_TEST_TEXT_CONFIG", "PRL_TEST_TEXT_INDEX", "PRL_TEST_HEADERS_DIR")
}
pytestmark = pytest.mark.skipif(not all(PATHS.values()), reason=f"set {', '.join(PATHS)}")

SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64}


# NemotronHPreTrainedModel.keep_in_fp32_for_weight_transfer, for environments without the GPU stack.
FP32_SUFFIXES = ("mamba.A_log", "mamba.D", "mlp.router.selection_bias")


def load_nemotron_h():
    """The conversion chain and fp32 rule; without the GPU stack, skip the package __init__ files (they import it)."""
    try:
        from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHPreTrainedModel

        return NemotronHPreTrainedModel.keep_in_fp32_for_weight_transfer
    except ImportError:
        src = Path(__file__).resolve().parents[4] / "src" / "prime_rl" / "trainer" / "models"
        for name, path in [
            ("prime_rl.trainer.models", src),
            ("prime_rl.trainer.models.nemotron_h", src / "nemotron_h"),
        ]:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
        source = (src / "nemotron_h" / "modeling_nemotron_h.py").read_text()
        assert f"return name.endswith({FP32_SUFFIXES!r})".replace("'", '"') in source, "fp32 rule changed"
        return lambda name: name.endswith(FP32_SUFFIXES)


def test_dcp_keys_map_onto_the_text_base():
    keep_in_fp32 = load_nemotron_h()
    from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
    from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain
    from prime_rl.trainer.sft.hf_export import is_fine_tuned

    config = NemotronHConfig(**json.loads(Path(PATHS["PRL_TEST_TEXT_CONFIG"]).read_text()))
    chain = conversion_chain(config)

    metadata = pickle.loads(Path(PATHS["PRL_TEST_DCP_METADATA"]).read_bytes())
    model_keys = {
        key[len("app.model.") :]: tuple(md.size)
        for key, md in metadata.state_dict_metadata.items()
        if key.startswith("app.model.")
    }
    converted: dict[str, tuple] = {}
    wire: dict[str, torch.dtype] = {}
    for key, shape in model_keys.items():
        dtype = torch.float32 if keep_in_fp32(key) else torch.bfloat16
        for hf_key, tensor in apply_prime_to_hf({key: torch.empty(shape, device="meta", dtype=dtype)}, chain).items():
            assert hf_key not in converted, f"duplicate {hf_key}"
            converted[hf_key] = tuple(tensor.shape)
            wire[hf_key] = tensor.dtype

    weight_map = json.loads(Path(PATHS["PRL_TEST_TEXT_INDEX"]).read_text())["weight_map"]
    base: dict[str, tuple] = {}
    dtypes: dict[str, str] = {}
    for path in sorted(Path(PATHS["PRL_TEST_HEADERS_DIR"]).glob("*.json")):
        for key, info in json.loads(path.read_text()).items():
            if key == "__metadata__":
                continue
            key = key.removeprefix("language_model.")
            if key in weight_map:
                base[key], dtypes[key] = tuple(info["shape"]), info["dtype"]
    assert set(base) == set(weight_map), "headers do not cover the base index"

    fine_tuned = {k: s for k, s in base.items() if is_fine_tuned(k, ("mtp.",))}
    kept = sorted(set(base) - set(fine_tuned))
    print(
        f"DCP model keys {len(model_keys)}, base {len(base)} = {len(fine_tuned)} fine-tuned + {len(kept)} kept (mtp.*)"
    )
    print("base dtypes of fine-tuned keys", Counter(dtypes[k] for k in fine_tuned))
    print("wire -> base dtype pairs", Counter((str(wire[k]), dtypes[k]) for k in fine_tuned))

    missing = sorted(set(fine_tuned) - set(converted))
    extra = sorted(set(converted) - set(fine_tuned))
    bad_shape = sorted(k for k in set(fine_tuned) & set(converted) if fine_tuned[k] != converted[k])
    assert not missing, missing[:5]
    assert not extra, extra[:5]
    assert not bad_shape, [(k, fine_tuned[k], converted[k]) for k in bad_shape[:5]]
    assert all(k.startswith("mtp.") for k in kept)
    # A tensor sent in bf16 into an F32 base slot would lose the fp32 master weights' precision.
    narrowed = [k for k in fine_tuned if SAFETENSORS_DTYPES[dtypes[k]].itemsize > wire[k].itemsize]
    assert not narrowed, narrowed[:5]
