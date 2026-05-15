# This script must be run with "https://github.com/nreHieW/transformers" which uses the unfused official implementation
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from transformers import ZayaForCausalLM

from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers, preprocess_layer_checkpoint
from prime_rl.trainer.weights import get_max_layer_num, load_state_dict, save_state_dict
from prime_rl.utils.vlm import get_layer_prefix
from test_vllm_hf_kl import load_prime_model


def assert_state_dict_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise AssertionError(f"key mismatch: missing={missing[:20]} extra={extra[:20]}")

    for key in sorted(expected_keys):
        actual_tensor = actual[key].detach().cpu()
        expected_tensor = expected[key].detach().cpu()
        if actual_tensor.dtype != expected_tensor.dtype:
            actual_tensor = actual_tensor.float()
            expected_tensor = expected_tensor.float()
        if not torch.allclose(actual_tensor, expected_tensor, atol=1e-6):
            max_diff = (actual_tensor - expected_tensor).abs().max().item()
            raise AssertionError(f"tensor mismatch at {key}: max_diff={max_diff}")


def load_safetensors_state_dict(repo_id: str) -> dict[str, torch.Tensor]:
    snapshot = Path(snapshot_download(repo_id=repo_id, repo_type="model"))
    state_dict: dict[str, torch.Tensor] = {}
    for path in snapshot.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def filesystem_broadcast_state_dict(prime_model) -> dict[str, torch.Tensor]:
    state_dict = dict(prime_model.state_dict())
    return prime_model.convert_to_vllm(state_dict)


def filesystem_saved_state_dict(prime_model) -> dict[str, torch.Tensor]:
    with TemporaryDirectory() as tmpdir:
        save_state_dict(filesystem_broadcast_state_dict(prime_model), Path(tmpdir), "safetensors", save_sharded=True)
        return load_state_dict(Path(tmpdir))


def resolve_dtensors(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    for key, value in list(state_dict.items()):
        if isinstance(value, DTensor):
            state_dict[key] = value.to(dtype).full_tensor()
    return state_dict


def nccl_broadcast_state_dict(prime_model, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    state_dict = prime_model.state_dict()
    layer_prefix = get_layer_prefix(prime_model.config)
    num_layers = get_max_layer_num(state_dict, layer_prefix)

    converted: dict[str, torch.Tensor] = {}
    for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
        chunk = resolve_dtensors(dict(layer_state_dict), dtype)
        chunk = preprocess_layer_checkpoint(prime_model, chunk, layer_idx)
        overlap = set(converted).intersection(chunk)
        if overlap:
            raise AssertionError(f"NCCL chunks produced duplicate keys: {sorted(overlap)[:20]}")
        converted.update(chunk)
    return converted


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Loads the original vLLM-format Zyphra checkpoint into the HF reference class.
    vllm_model = ZayaForCausalLM.from_pretrained("Zyphra/ZAYA1-8B", torch_dtype=dtype).to(device).eval()
    prime_model = load_prime_model("JJJYmmm/ZAYA1-8B-HF", dtype, device).eval()

    vllm_model_state_dict = {key: value.detach().cpu() for key, value in vllm_model.state_dict().items()}
    vllm_model_state_dict.pop("lm_head.weight", None)

    vllm_checkpoint_state_dict = load_safetensors_state_dict("Zyphra/ZAYA1-8B")
    vllm_checkpoint_state_dict.pop("lm_head.weight", None)

    fs_state_dict = filesystem_broadcast_state_dict(prime_model)
    assert_state_dict_equal(fs_state_dict, vllm_checkpoint_state_dict)
    assert_state_dict_equal(filesystem_saved_state_dict(prime_model), vllm_checkpoint_state_dict)
    print("filesystem broadcast conversion matches Zyphra checkpoint")

    nccl_state_dict = nccl_broadcast_state_dict(prime_model)
    assert_state_dict_equal(nccl_state_dict, vllm_checkpoint_state_dict)
    print("NCCL broadcast conversion matches Zyphra checkpoint")

    assert_state_dict_equal(fs_state_dict, nccl_state_dict)
    print("filesystem and NCCL broadcast conversions match")

    assert_state_dict_equal(fs_state_dict, vllm_model_state_dict)
    print("broadcast conversion matches loaded vLLM-format model state_dict")

    print("All tests passed")
