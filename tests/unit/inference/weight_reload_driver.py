"""Subprocess driver for the NCCL weight-reload GPU regression test.

Each mode runs one process so every vLLM engine gets a clean CUDA/distributed
lifetime. The test module (``test_nccl_weight_reload.py``) invokes this file
with ``sys.executable`` and asserts on the emitted JSON.

Modes:
  prepare    write the tiny GLM-MoE-DSA checkpoints (serialized blockwise-FP8
             and plain bf16) used by the other modes.
  update     start a TP engine from the FP8 checkpoint, receive TWO successive
             quantized NCCL weight updates sent from a "trainer-side" prime
             model over the production wire protocol, and dump deterministic
             greedy generations taken before / after each update. The wire
             rounds are also saved as reference FP8 checkpoints.
  reference  fresh-load an engine from a checkpoint dir and dump the same
             greedy generations.
  reject     start a TP1 engine that quantizes a bf16 checkpoint on the fly
             (``quantization="fp8"``, no serialized fp8 checkpoint) and assert
             that ``init_broadcaster`` with ``quantize_in_weight_transfer=true``
             is rejected loudly, before any collective.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT / "src", REPO_ROOT / "packages" / "prime-rl-configs" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TOKENIZER_SOURCE = "samsja/mini-glm-moe"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]

# A two-layer GLM-MoE-DSA (MLA + sparse indexer + dense layer 0 + MoE layer 1)
# whose sharded dimensions all divide 8-way tensor parallelism.
TINY_CONFIG = dict(
    vocab_size=151552,
    hidden_size=512,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=2.5,
    kv_lora_rank=256,
    q_lora_rank=256,
    qk_rope_head_dim=64,
    v_head_dim=128,
    qk_nope_head_dim=128,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    norm_topk_prob=True,
    hidden_act="silu",
    max_position_embeddings=512,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    tie_word_embeddings=False,
    rope_interleave=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    rope_theta=10000.0,
    attention_bias=False,
    attention_dropout=0.0,
    index_n_heads=8,
    index_head_dim=128,
    indexer_rope_interleave=True,
    pad_token_id=0,
    index_topk=16,
    use_index_cache=False,
    index_topk_freq=1,
    index_topk_pattern=None,
    indexer_types=["full", "full"],
    index_skip_topk_offset=0,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    n_group=1,
    topk_group=1,
    moe_router_dtype="float32",
)

# Fixed prompts (raw token ids — no tokenizer semantics involved).
PROMPTS = [[(i * 7 + j) % 1000 + 5 for j in range(64)] for i in range(6)]


def _hf_config_dict() -> dict:
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "eos_token_id": 151329,
        "transformers_version": "5.6.2",
    }
    config.update(TINY_CONFIG)
    config["head_dim"] = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    config["qk_head_dim"] = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000.0}
    return config


def _build_hf_state(seed: int):
    """Random bf16 weights in the HF checkpoint naming of the tiny model."""
    import torch

    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(dtype)

    c = TINY_CONFIG
    hidden = c["hidden_size"]
    heads = c["num_attention_heads"]
    q_lora = c["q_lora_rank"]
    kv_lora = c["kv_lora_rank"]
    nope = c["qk_nope_head_dim"]
    rope = c["qk_rope_head_dim"]
    v_dim = c["v_head_dim"]
    n_experts = c["n_routed_experts"]
    state = {
        "model.embed_tokens.weight": randn(c["vocab_size"], hidden),
        "model.norm.weight": randn(hidden),
        "lm_head.weight": randn(c["vocab_size"], hidden),
    }
    for layer in range(c["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        a = f"{p}.self_attn"
        state[f"{p}.input_layernorm.weight"] = randn(hidden)
        state[f"{p}.post_attention_layernorm.weight"] = randn(hidden)
        state[f"{a}.q_a_proj.weight"] = randn(q_lora, hidden)
        state[f"{a}.kv_a_proj_with_mqa.weight"] = randn(kv_lora + rope, hidden)
        state[f"{a}.q_a_layernorm.weight"] = randn(q_lora)
        state[f"{a}.kv_a_layernorm.weight"] = randn(kv_lora + rope)
        state[f"{a}.q_b_proj.weight"] = randn(heads * (nope + rope), q_lora)
        state[f"{a}.kv_b_proj.weight"] = randn(heads * (nope + v_dim), kv_lora)
        state[f"{a}.o_proj.weight"] = randn(hidden, heads * v_dim)
        state[f"{a}.indexer.wq_b.weight"] = randn(c["index_n_heads"] * c["index_head_dim"], q_lora)
        state[f"{a}.indexer.wk.weight"] = randn(c["index_head_dim"], hidden)
        state[f"{a}.indexer.weights_proj.weight"] = randn(c["index_n_heads"], hidden)
        state[f"{a}.indexer.k_norm.weight"] = randn(c["index_head_dim"])
        state[f"{a}.indexer.k_norm.bias"] = randn(c["index_head_dim"])
        if layer < c["first_k_dense_replace"]:
            state[f"{p}.mlp.gate_proj.weight"] = randn(c["intermediate_size"], hidden)
            state[f"{p}.mlp.up_proj.weight"] = randn(c["intermediate_size"], hidden)
            state[f"{p}.mlp.down_proj.weight"] = randn(hidden, c["intermediate_size"])
        else:
            state[f"{p}.mlp.gate.weight"] = randn(n_experts, hidden)
            state[f"{p}.mlp.gate.e_score_correction_bias"] = randn(n_experts, dtype=torch.float32)
            for e in range(n_experts):
                state[f"{p}.mlp.experts.{e}.gate_proj.weight"] = randn(c["moe_intermediate_size"], hidden)
                state[f"{p}.mlp.experts.{e}.up_proj.weight"] = randn(c["moe_intermediate_size"], hidden)
                state[f"{p}.mlp.experts.{e}.down_proj.weight"] = randn(hidden, c["moe_intermediate_size"])
            state[f"{p}.mlp.shared_experts.gate_proj.weight"] = randn(c["moe_intermediate_size"], hidden)
            state[f"{p}.mlp.shared_experts.up_proj.weight"] = randn(c["moe_intermediate_size"], hidden)
            state[f"{p}.mlp.shared_experts.down_proj.weight"] = randn(hidden, c["moe_intermediate_size"])
    return state


def _quantized_hf_state(seed: int) -> dict:
    """The seed's HF state quantized to the serialized blockwise-FP8 checkpoint layout."""
    from prime_rl.trainer.models.fp8 import quantize_to_fp8_checkpoint
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import _keep_unquantized

    state = _build_hf_state(seed)
    quantized = quantize_to_fp8_checkpoint(state, keep_unquantized=_keep_unquantized)
    # Embeddings and the LM head stay bf16 in real checkpoints (the wire quantizer
    # only ever sees per-layer dicts, so it has no say about them).
    for name in ("model.embed_tokens.weight_scale_inv", "lm_head.weight_scale_inv"):
        quantized.pop(name, None)
    quantized["model.embed_tokens.weight"] = state["model.embed_tokens.weight"]
    quantized["lm_head.weight"] = state["lm_head.weight"]
    return quantized


def _write_checkpoint_dir(path: Path, state: dict, quantized: bool) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    config = _hf_config_dict()
    if quantized:
        config["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [],
        }
    (path / "config.json").write_text(json.dumps(config, indent=2))
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(path / "model.safetensors"))


def _fetch_tokenizer_files(destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    for filename in TOKENIZER_FILES:
        source = hf_hub_download(TOKENIZER_SOURCE, filename)
        shutil.copyfile(source, destination / filename)


def mode_prepare(args: argparse.Namespace) -> None:
    fp8_dir = Path(args.fp8_dir)
    bf16_dir = Path(args.bf16_dir)
    _write_checkpoint_dir(fp8_dir, _quantized_hf_state(seed=args.seed), quantized=True)
    _write_checkpoint_dir(bf16_dir, _build_hf_state(seed=args.seed), quantized=False)
    # The tiny config needs a real tokenizer for the engines; reuse the CI model's.
    _fetch_tokenizer_files(fp8_dir)
    for filename in TOKENIZER_FILES:
        shutil.copyfile(fp8_dir / filename, bf16_dir / filename)


def _copy_engine_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "config.json", destination / "config.json")
    for filename in TOKENIZER_FILES:
        shutil.copyfile(source / filename, destination / filename)


def _make_engine(model_dir: str, tp: int, expert_parallel: bool, **overrides):
    from vllm import LLM

    import prime_rl.trainer.models  # noqa: F401 - registers glm_moe_dsa with AutoConfig

    return LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        enable_expert_parallel=expert_parallel,
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        disable_log_stats=True,
        worker_extension_cls="prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
        **overrides,
    )


def _generate(llm) -> list[list[int]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt) for prompt in PROMPTS],
        SamplingParams(temperature=0.0, max_tokens=24, detokenize=False),
    )
    return [list(output.outputs[0].token_ids) for output in outputs]


def _build_prime_model(seed: int):
    """A trainer-side tiny GLM-MoE-DSA model on cuda:0, deterministically filled."""
    import torch

    from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    config = GlmMoeDsaConfig(**TINY_CONFIG)
    model = GlmMoeDsaForCausalLM(config).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        fill = torch.randn(param.shape, generator=generator).to(param.dtype)
        param.data.copy_(fill)
        del fill
    # The router bias travels in fp32 (the engine asserts fp32 for it).
    for name, param in model.named_parameters():
        if name.endswith("mlp.router.selection_bias"):
            param.data = param.data.to(torch.float32)
    return model.cuda()


def _wire_layers(model) -> list[dict]:
    """The per-layer wire stream a quantized NCCL broadcast sends for ``model``."""
    import torch

    from prime_rl.trainer.conversion_utils import get_max_layer_num
    from prime_rl.transports.weights.nccl import (
        filter_state_dict_by_layers,
        preprocess_layer_quantized,
        resolve_dtensors,
    )
    from prime_rl.utils.vlm import get_layer_prefix

    state_dict = model.state_dict()
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    layer_prefix = get_layer_prefix(model.config)
    num_layers = get_max_layer_num(state_dict, layer_prefix)
    layers = []
    for layer_id, layer_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
        layer_dict = resolve_dtensors(layer_dict, keep_in_fp32, torch.bfloat16)
        layers.append(preprocess_layer_quantized(model, layer_dict, layer_id))
    return layers


def _save_reference_checkpoint(path: Path, wire_layers: list[dict], template_dir: Path) -> None:
    from safetensors.torch import save_file

    _copy_engine_files(template_dir, path)
    tensors = {}
    for layer in wire_layers:
        for name, tensor in layer.items():
            if "inv_freq" in name:
                continue
            tensors[name] = tensor.contiguous()
    save_file(tensors, str(path / "model.safetensors"))


def _send_round(communicator, wire_layers: list[dict]) -> None:
    from prime_rl.transports.wire import broadcast_integer, broadcast_state_dict

    broadcast_integer(len(wire_layers), communicator)
    for layer in wire_layers:
        broadcast_state_dict(layer, communicator)


def _broadcast_and_update(llm, communicator, wire_layers: list[dict], timeout_s: float = 900.0) -> None:
    """Mirror the production handshake: the update RPC drives the workers into
    the NCCL receive; the trainer-side sender broadcasts concurrently."""
    result: dict = {}

    def rpc() -> None:
        try:
            llm.collective_rpc("update_weights_from_path", args=("reload-test",))
            result["ok"] = True
        except BaseException as error:  # noqa: BLE001 - surfaced to the test
            result["error"] = repr(error)

    rpc_thread = threading.Thread(target=rpc)
    rpc_thread.start()
    time.sleep(2.0)
    _send_round(communicator, wire_layers)
    rpc_thread.join(timeout=timeout_s)
    if rpc_thread.is_alive():
        raise RuntimeError("update_weights_from_path did not return — receiver likely wedged in a NCCL read")
    if result.get("error"):
        raise RuntimeError(f"update_weights_from_path failed: {result['error']}")
    if not result.get("ok"):
        raise RuntimeError(f"update RPC returned without success: {result}")


def mode_update(args: argparse.Namespace) -> None:
    import torch
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    import prime_rl.trainer.models  # noqa: F401 - AutoConfig registration
    from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

    # Trainer-side wire rounds, and the matching on-disk reference checkpoints.
    template_dir = Path(args.model_dir)
    wire1 = _wire_layers(_build_prime_model(seed=args.seed1))
    _save_reference_checkpoint(Path(args.ref1_dir), wire1, template_dir)
    wire2 = _wire_layers(_build_prime_model(seed=args.seed2))
    _save_reference_checkpoint(Path(args.ref2_dir), wire2, template_dir)

    llm = _make_engine(args.model_dir, tp=args.tp, expert_parallel=args.ep == "on")
    baseline = _generate(llm)

    disable_nccl_p2p_if_unavailable()
    pg = StatelessProcessGroup.create(
        host="127.0.0.1", port=args.port, rank=0, world_size=args.tp + 1, store_timeout=args.timeout
    )
    communicator = PyNcclCommunicator(pg, device=torch.device("cuda:0"))
    llm.collective_rpc(
        "init_broadcaster",
        args=("127.0.0.1", args.port, 0, args.tp, args.timeout, True, "reload-test"),
    )

    _broadcast_and_update(llm, communicator, wire1)
    after_first = _generate(llm)
    _broadcast_and_update(llm, communicator, wire2)
    after_second = _generate(llm)

    Path(args.out).write_text(
        json.dumps({"baseline": baseline, "after_first": after_first, "after_second": after_second})
    )


def mode_reference(args: argparse.Namespace) -> None:
    llm = _make_engine(args.model_dir, tp=args.tp, expert_parallel=args.ep == "on")
    tokens = _generate(llm)
    Path(args.out).write_text(json.dumps({"tokens": tokens}))


def mode_reject(args: argparse.Namespace) -> None:
    """An online-fp8 engine (bf16 source, quantization="fp8") must refuse the
    quantized wire format at init, before any collective starts."""
    llm = _make_engine(args.model_dir, tp=1, expert_parallel=False, quantization="fp8")
    rejected, message = False, ""
    try:
        llm.collective_rpc("init_broadcaster", args=("127.0.0.1", args.port, 0, 1, args.timeout, True, "reload-test"))
    except BaseException as error:  # noqa: BLE001
        message = repr(error)
        chain = []
        candidate = error
        while candidate is not None:
            chain.append(str(candidate))
            candidate = candidate.__cause__ or candidate.__context__
        rejected = any("quantize_in_weight_transfer requires" in entry for entry in chain)
    Path(args.out).write_text(json.dumps({"rejected": rejected, "message": message}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--fp8-dir", required=True)
    prepare.add_argument("--bf16-dir", required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.set_defaults(func=mode_prepare)

    update = subparsers.add_parser("update")
    update.add_argument("--model-dir", required=True)
    update.add_argument("--ref1-dir", required=True)
    update.add_argument("--ref2-dir", required=True)
    update.add_argument("--out", required=True)
    update.add_argument("--tp", type=int, required=True)
    update.add_argument("--ep", choices=["on", "off"], required=True)
    update.add_argument("--port", type=int, required=True)
    update.add_argument("--seed1", type=int, default=101)
    update.add_argument("--seed2", type=int, default=202)
    update.add_argument("--timeout", type=int, default=900)
    update.set_defaults(func=mode_update)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--model-dir", required=True)
    reference.add_argument("--out", required=True)
    reference.add_argument("--tp", type=int, required=True)
    reference.add_argument("--ep", choices=["on", "off"], required=True)
    reference.set_defaults(func=mode_reference)

    reject = subparsers.add_parser("reject")
    reject.add_argument("--model-dir", required=True)
    reject.add_argument("--out", required=True)
    reject.add_argument("--port", type=int, required=True)
    reject.add_argument("--timeout", type=int, default=900)
    reject.set_defaults(func=mode_reject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
