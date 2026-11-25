"""Generate ``inkling2_loop_golden.safetensors``, the reference logits of ``test_inkling2_loop.py``.

The reference is transformers' ``Inkling2LoopForCausalLM`` as of PR 51050 (commit 6173a2f, version 5.24.0.dev0),
which the repo's environment does not have. Run from the repo root with an interpreter that has it:

    <python with transformers PR 51050> tests/unit/train/models/fixtures/make_inkling2_loop_golden.py [output]

The test imports the recipe at the top of this file (the config and the checkpoint tensors, seeded by name), so
the fixture has to be regenerated whenever the recipe changes. The reference loads the recipe's checkpoint from
the release layout through its own conversion and runs in FP64, so the stored logits carry no rounding noise of
their own. The two documents are chosen token by token so that no routing decision is near a tie, and the
statistics printed at the end show how much each path of the model takes part.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
import types
import zlib
from collections import defaultdict
from pathlib import Path

import torch

# Release config keys of a tiny model: blocks of two layers, a sliding then a full one, with layers 2..5 looped three
# times. Sliding and full layers differ in head count and head dim so that a mix-up changes shapes. The clip
# thresholds and the softcaps are low enough to bind at the recipe's weight scales, and the norm epsilon is far from
# the fixed 1e-6 of the depth mixes and the clips, so that using one for the other shows.
TEXT_CONFIG = dict(
    vocab_size=96,
    unpadded_vocab_size=80,
    hidden_size=32,
    num_hidden_layers=8,
    n_layers_per_attn_res_block=2,
    loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=4, is_varloop=False),
    local_layer_ids=[0, 2, 4, 6],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    swa_num_attention_heads=6,
    swa_num_key_value_heads=3,
    swa_head_dim=4,
    sliding_window_size=8,
    rel_extent=12,
    d_rel=4,
    sconv_kernel_size=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    intermediate_size=16,
    route_scale=2.0,
    log_scaling_n_floor=4,
    log_scaling_alpha=0.1,
    qk_l2_clip_threshold=1.0,
    rel_l2_clip_threshold=1.0,
    mlp_gate_softcap=0.5,
    mlp_up_softcap=0.75,
    final_logit_softcapping=4.0,
    rms_norm_eps=1e-3,
    eos_token_id=1,
)
# Both documents are longer than the sliding window, the relative-bias extent and the log-scaling floor.
SEQ_LENS = (23, 40)
# Projections are noise of `scale / sqrt(fan_in)`, with `scale` 1 unless listed here by checkpoint-name suffix. The
# experts and the loop's message gates compound from one layer visit to the next, as do the depth-mix queries: with
# all of them at a scale of 1, the 16 visits amplify a rounding error thousands of times. The other three are sized
# for the clips to bind on a part of the vectors.
PROJECTION_SCALES = {
    "w13_weight": 0.5,
    "w2_weight": 0.5,
    "down_weight": 0.5,
    "gain_weight": 0.5,
    "forget_weight": 0.5,
    "wq_du.weight": 0.5,
    "wr_du.weight": 0.6,
    "weight_proj": 0.5,
}
# Smallest gap between the last selected and the first rejected expert that a routing decision may have. The BF16
# expert compute moves a selection score by up to 2e-3, an FP32 forward by 1e-6.
MIN_SELECTION_GAP = 1.5e-2
INPUT_SEED = 1
GOLDEN = Path(__file__).with_name("inkling2_loop_golden.safetensors")


def checkpoint_tensor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """The FP32 checkpoint tensor ``name`` (release layout), seeded by its name. The rows of the unembedding past
    the logits are noise like the others: the release does not zero them either."""
    # Uniform noise of unit variance: a seeded `torch.rand` gives the same bits under every CPU instruction set,
    # which `torch.randn` does not.
    uniform = torch.rand(shape, generator=torch.Generator().manual_seed(zlib.crc32(name.encode())))
    noise = (2 * uniform - 1) * 3**0.5
    if name.endswith(("norm.weight", "l2_clip.scale")):
        return 1.0 + 0.3 * noise
    if ".attn_res." in name or "middle_cycle_cross_pass" in name:
        return 0.5 * noise
    if name.endswith("gate.bias"):
        return 0.1 * noise
    if name.endswith(("sconv.weight", "static_w")):
        return 0.3 * noise
    scale = next((scale for suffix, scale in PROJECTION_SCALES.items() if name.endswith(suffix)), 1.0)
    return scale * noise / shape[-1] ** 0.5


def checkpoint_digest(checkpoint: dict[str, torch.Tensor]) -> torch.Tensor:
    """SHA-256 of a checkpoint's names and values, as 32 bytes. The fixture holds the digest of the checkpoint that
    the reference loaded, so the test can tell a recipe that no longer gives those bits from a model that differs."""
    digest = hashlib.sha256()
    for name in sorted(checkpoint):
        digest.update(name.encode())
        digest.update(checkpoint[name].contiguous().numpy().tobytes())
    return torch.frombuffer(bytearray(digest.digest()), dtype=torch.uint8)


def fp64_reference() -> types.ModuleType:
    """The reference's modeling module with its FP32 islands (``.float()``, ``torch.float32``) widened to FP64."""
    name = "transformers.models.inkling2_loop.modeling_inkling2_loop"
    path = importlib.util.find_spec(name).origin
    source = Path(path).read_text().replace(".float()", ".double()").replace("torch.float32", "torch.float64")
    module = types.ModuleType(f"{name}_fp64")
    module.__package__, module.__file__ = name.rpartition(".")[0], path
    sys.modules[module.__name__] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def load_reference(reference: types.ModuleType, dtype: torch.dtype) -> tuple[torch.nn.Module, torch.Tensor]:
    """The reference model on the recipe's checkpoint, which it reads in the release layout, and that checkpoint's
    digest."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = dict(TEXT_CONFIG)
    sliding = config.pop("local_layer_ids")
    layer_types = ["hybrid_sliding" if index in sliding else "hybrid" for index in range(config["num_hidden_layers"])]
    config = reference.Inkling2LoopTextConfig(**config, layer_types=layer_types)
    with tempfile.TemporaryDirectory() as directory:
        # The release layout's names and shapes, from the reference's own reverse conversion.
        reference.Inkling2LoopForCausalLM(config).save_pretrained(directory, save_original_format=True)
        weights = Path(directory) / "model.safetensors"
        with safe_open(weights, framework="pt") as file:
            shapes = {name: tuple(file.get_slice(name).get_shape()) for name in file.keys()}
        checkpoint = {name: checkpoint_tensor(name, shape) for name, shape in shapes.items()}
        save_file(checkpoint, weights, metadata={"format": "pt"})
        model = reference.Inkling2LoopForCausalLM.from_pretrained(
            directory, dtype=dtype, attn_implementation="eager", experts_implementation="eager"
        )
    return model.eval(), checkpoint_digest(checkpoint)


def record(model: torch.nn.Module, reference: types.ModuleType) -> dict[str, list[torch.Tensor]]:
    """Hook the reference's modules. Every forward appends to the returned lists, one list per statistic: the
    selection gap of every routing decision, and how much each path of the model takes part."""
    records = defaultdict(list)
    layer_types = model.config.layer_types
    top_k = model.config.num_experts_per_tok
    kernel_size = model.config.conv_kernel_size
    clipped = {"q_proj": "query", "k_proj": "key", "relative_position_bias": "relative-state"}

    def router(module, args, output):
        scores = module.score_fn(output[0])
        ordered = (scores + module.e_score_correction_bias).sort(dim=-1, descending=True).values
        records["selection gap"].append(ordered[:, top_k - 1] - ordered[:, top_k])
        unbiased, picked = scores.topk(top_k).indices.sort().values, output[2].sort().values
        records["routing decisions that the selection bias changes"].append((unbiased != picked).any(-1))
        experts = (picked * module.num_experts ** torch.arange(top_k)).sum(-1)
        records["routing decisions off the most common experts of their layer visit"].append(
            experts != experts.mode().values
        )

    def clip(label):
        def hook(module, args, output):
            values = args[0] * module.scale if hasattr(module, "scale") else args[0]
            records[label].append(torch.linalg.vector_norm(values, dim=-1) > module.threshold)

        return hook

    def softcap(label):
        # tanh shrinks an input of 0.4 softcaps by 5%.
        return lambda module, args, output: records[label].append(args[0].abs() > 0.4 * module.softcap)

    def sink(label):
        return lambda module, args, output: records[label].append(1 - output[1].sum(-1))

    def message_gate(module, args, output):
        change = torch.linalg.vector_norm(output - args[0], dim=-1) / torch.linalg.vector_norm(args[0], dim=-1)
        records["message gates: change of the sub-layer input, relative to it"].append(change)

    def convolution(label, num_taps):
        def hook(module, args, output):
            taps = output * (num_taps + 1) ** 0.5 - args[0]
            ratio = torch.linalg.vector_norm(taps, dim=-1) / torch.linalg.vector_norm(args[0], dim=-1)
            records[label].append(ratio)

        return hook

    for name, module in model.named_modules():
        parts = name.split(".")
        kind = None
        if len(parts) > 2 and parts[1] == "layers":
            kind = "sliding" if layer_types[int(parts[2])] == "hybrid_sliding" else "full"
        if isinstance(module, reference.Inkling2LoopTopKRouter):
            module.register_forward_hook(router)
        elif isinstance(module, (reference.Inkling2LoopL2Clip, reference.Inkling2LoopScaledL2Clip)):
            stream = next(label for part, label in clipped.items() if part in parts)
            module.register_forward_hook(clip(f"{kind} layers: {stream} vectors over the clip threshold"))
        elif isinstance(module, reference.Inkling2LoopSoftcap):
            module.register_forward_hook(softcap(f"{parts[-1]}: inputs that it shrinks by over 5%"))
        elif isinstance(module, (reference.Inkling2LoopSlidingAttention, reference.Inkling2LoopFullAttention)):
            module.register_forward_hook(sink(f"{kind} layers: the sink's share of the attention"))
        elif isinstance(module, reference.Inkling2LoopMessageGate):
            module.register_forward_hook(message_gate)
        elif isinstance(module, reference.Inkling2LoopShortConvolution):
            module.register_forward_hook(convolution("static convolutions: taps relative to the input", kernel_size))
        elif isinstance(module, reference.Inkling2LoopDynamicShortConvolution):
            label = "dynamic convolutions: taps relative to the input"
            module.register_forward_hook(convolution(label, 2 * kernel_size))
    return records


def logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids[None], use_cache=False).logits[0]


def choose_documents(model: torch.nn.Module, records: dict[str, list[torch.Tensor]]) -> list[torch.Tensor]:
    """Seeded documents whose every routing decision clears ``MIN_SELECTION_GAP``. Each token is the first of a
    shuffled vocabulary that keeps its own decisions clear: the model is causal, so later tokens do not move them."""
    generator = torch.Generator().manual_seed(INPUT_SEED)
    documents = []
    for length in SEQ_LENS:
        document = torch.empty(0, dtype=torch.long)
        while len(document) < length:
            # Token ids 0 and 1 are left out, as padding and end of sequence would be.
            for token in 2 + torch.randperm(TEXT_CONFIG["unpadded_vocab_size"] - 2, generator=generator):
                records.clear()
                logits(model, torch.cat([document, token[None]]))
                if min(gaps[-1] for gaps in records["selection gap"]) >= MIN_SELECTION_GAP:
                    break
            else:
                raise RuntimeError(f"No token keeps the routing of position {len(document)} clear of a tie")
            document = torch.cat([document, token[None]])
        documents.append(document)
    return documents


def main() -> None:
    import transformers
    from safetensors.torch import save_file
    from transformers.models.inkling2_loop import modeling_inkling2_loop

    torch.set_num_threads(1)
    reference = fp64_reference()
    model, digest = load_reference(reference, torch.float64)
    records = record(model, reference)
    documents = choose_documents(model, records)
    records.clear()
    golden = [logits(model, document) for document in documents]
    assert all(reference_logits.dtype == torch.float64 for reference_logits in golden)

    gaps = torch.cat(records.pop("selection gap"))
    assert gaps.min() >= MIN_SELECTION_GAP, f"a routing decision has a selection gap of {gaps.min():.1e}"
    print(f"{len(gaps)} routing decisions: smallest selection gap {gaps.min():.3f}, median {gaps.median():.3f}")
    for label, values in sorted(records.items()):
        values = torch.cat([value.flatten() for value in values])
        if values.dtype == torch.bool:
            share = values.double().mean()
            assert 0.05 <= share <= 0.95, f"{label}: {share:.0%}, so one of the two cases is hardly exercised"
            print(f"{label}: {share:.0%} of {len(values)}")
        else:
            print(f"{label}: median {values.median():.2f}, from {values.min():.2f} to {values.max():.2f}")
    positions = torch.cat([torch.arange(length) for length in SEQ_LENS])
    for label, key in (
        ("tokens with earlier tokens outside the sliding window", "sliding_window_size"),
        ("tokens with earlier tokens beyond the relative-bias extent", "rel_extent"),
        ("tokens past the log-scaling floor", "log_scaling_n_floor"),
    ):
        print(f"{label}: {(positions >= TEXT_CONFIG[key]).sum()} of {len(positions)}")

    # The unwidened reference shows how much noise an FP32 forward of this model carries.
    fp32_model, _ = load_reference(modeling_inkling2_loop, torch.float32)
    fp32_noise = max((logits(fp32_model, d) - g).abs().max().item() for d, g in zip(documents, golden))
    largest = max(reference_logits.abs().max().item() for reference_logits in golden)
    print(f"largest |logit| {largest:.2f}; the reference's own FP32 forward is within {fp32_noise:.1e} of its FP64 one")

    install = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json") or "{}")
    commit = install.get("vcs_info", {}).get("commit_id", "unknown commit")
    # One metadata entry: safetensors writes several in an arbitrary order, and the file would not be reproducible.
    source = f"transformers {transformers.__version__} ({commit}), Inkling2LoopForCausalLM in FP64, rounded to FP32"
    tensors = {"checkpoint_sha256": digest}
    tensors |= {f"input_ids_{index}": document for index, document in enumerate(documents)}
    tensors |= {f"logits_{index}": reference_logits.float() for index, reference_logits in enumerate(golden)}
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    save_file(tensors, output, metadata={"source": source})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
