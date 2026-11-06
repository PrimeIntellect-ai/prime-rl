"""vLLM references for Qwen3.8 video inputs, and the trainer side of the logprob comparison.

    python video_parity.py processor --model MODEL --video clip.mp4 --out processor.pt
    python video_parity.py vllm --model DIR --video clip.mp4 --out vllm.pt --tensor-parallel-size 4
    torchrun --nproc-per-node 8 video_parity.py trainer --model DIR --video clip.mp4 --vllm vllm.pt --out trainer.pt

The vLLM phases run without prime-rl's trainer models imported, which register their own config class for the
model type. ``processor`` runs vLLM's Qwen3-VL multimodal processor and MRoPE on the video. ``vllm`` decodes the
video with vLLM's media connector and scores the prompt with the engine settings of the Qwen3.8 RL example,
returning the routed experts. ``trainer`` builds the sample the way the orchestrator and trainer do (video ref
from the expanded prompt, ``materialize_images``, ``setup_model`` + ``forward`` with router replay), on every rank
of an FSDP mesh.
"""

import argparse
import base64
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "This is a short gameplay clip. What is the player doing, and how does the view move?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "The player walks forward across the terrain while the camera slowly turns to the right.<|im_end|>\n"
)


def video_url(path: str) -> str:
    return "data:video/mp4;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def run_processor(args: argparse.Namespace) -> None:
    from vllm.config import ModelConfig
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.media import MediaConnector
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    model_config = ModelConfig(model=args.model, revision=args.revision)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    prompt_ids = processor.info.get_tokenizer().encode(PROMPT)
    video = MediaConnector().fetch_video(video_url(args.video), video_processor="Qwen3VLVideoProcessor").media
    mm_items = processor.info.parse_mm_data({"video": [video]})
    output = processor.apply(ProcessorInputs(prompt=prompt_ids, mm_data_items=mm_items), TimingContext())
    (placeholder,) = output["mm_placeholders"]["video"]
    features = [
        MultiModalFeatureSpec(
            data=output["mm_kwargs"]["video"][0], modality="video", identifier="video", mm_position=placeholder
        )
    ]
    positions, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        output["prompt_token_ids"], features, model_config.hf_config
    )
    kwargs = output["mm_kwargs"].get_data()
    torch.save(
        {
            "prompt_ids": output["prompt_token_ids"],
            "placeholder": (placeholder.offset, placeholder.length),
            "pixel_values_videos": kwargs["pixel_values_videos"],
            "video_grid_thw": kwargs["video_grid_thw"],
            "timestamps": kwargs["timestamps"],
            "positions": positions,
        },
        args.out,
    )


def run_vllm(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.multimodal.media import MediaConnector

    prompt_ids = AutoTokenizer.from_pretrained(args.model).encode(PROMPT)
    video = MediaConnector().fetch_video(video_url(args.video), video_processor="Qwen3VLVideoProcessor").media
    has_ngram_embeddings = bool(AutoConfig.from_pretrained(args.model).text_config.ple_layer_ids)
    # The engine settings of examples/extra/vlm/rl-qwen3.8.toml that change numerics or memory.
    llm = LLM(
        model=args.model,
        seed=0,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.tensor_parallel_size > 1,
        mm_encoder_tp_mode="data",
        mamba_cache_dtype="float32",
        engram_config={"cpu_offload": True} if has_ngram_embeddings else None,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,  # required for returning routed experts; nothing to reuse in one request
        limit_mm_per_prompt={"image": 0, "video": 1},
        mm_processor_cache_gb=0,
        # The trainer replays the experts vLLM routed to (trainer.enable_router_replay)
        enable_return_routed_experts=True,
        # prime-rl's inference defaults for RL (enable_fp32_router_logits / enable_fp32_lm_head)
        hf_overrides={"moe_router_dtype": "float32", "head_dtype": "float32"},
    )
    (output,) = llm.generate(
        {"prompt_token_ids": prompt_ids, "multi_modal_data": {"video": [video]}},
        SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False),
    )
    ids = list(output.prompt_token_ids)
    logprobs = [float("nan")] + [entry[token].logprob for entry, token in zip(output.prompt_logprobs[1:], ids[1:])]
    routed_experts = torch.from_numpy(output.outputs[0].routed_experts[: len(ids)].astype(np.int32))
    torch.save({"prompt_ids": ids, "logprobs": logprobs, "routed_experts": routed_experts}, args.out)


def run_trainer(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    from prime_rl.configs.shared import VLMConfig
    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.orchestrator.trajectories import _build_mm_refs
    from prime_rl.trainer.model import forward, setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.rl.loss import shift_tensor_left
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.trainer.vlm import setup_processor
    from prime_rl.trainer.world import get_world

    reference = torch.load(args.vllm)
    ids = np.asarray(reference["prompt_ids"])
    config = ModelConfig(name=args.model, vlm=VLMConfig())
    processor = setup_processor(config)
    token_types = np.zeros_like(ids)
    token_types[ids == processor.video_token_id] = 2
    refs = _build_mm_refs([], [video_url(args.video)], ids, token_types)

    setup_torch_distributed(timeout=timedelta(hours=1))
    torch.cuda.set_device(get_world().local_rank)
    resolve_ep(config)
    model = setup_model(config, get_parallel_dims(config))
    model.eval()
    mm_kwargs = {key: value.cuda() for key, value in model.materialize_images(refs, processor).items()}
    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        out = forward(
            model,
            input_ids,
            torch.arange(len(ids), device="cuda").unsqueeze(0),
            labels=shift_tensor_left(input_ids),
            temperature=torch.ones(input_ids.shape, device="cuda"),
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=torch.tensor([token_types], device="cuda"),
            seq_lens=torch.tensor([len(ids)], device="cuda"),
            routed_experts=reference["routed_experts"].unsqueeze(0).cuda(),
        )
    if get_world().is_master:
        logprobs = [float("nan")] + out["logprobs"][0, :-1].tolist()
        torch.save({"logprobs": logprobs, "video": (refs.videos[0].offset, refs.videos[0].length)}, args.out)
    dist.barrier()
    dist.destroy_process_group()


def compare(vllm_out: dict, trainer_out: dict, video_token_id: int) -> dict[str, dict[str, float]]:
    """Absolute logprob difference per predicted token, over the video pads, the rest of the video placeholder
    (timestamps and markers) and the text."""
    ids = np.asarray(vllm_out["prompt_ids"])
    diff = np.abs(np.asarray(vllm_out["logprobs"]) - np.asarray(trainer_out["logprobs"]))[1:]
    offset, length = trainer_out["video"]
    in_video = np.zeros(len(ids), dtype=bool)
    in_video[offset : offset + length] = True
    groups = {
        "video pads": ids == video_token_id,
        "video timestamps and markers": in_video & (ids != video_token_id),
        "text": ~in_video,
    }
    return {
        name: {"tokens": int(mask[1:].sum()), "max": float(diff[mask[1:]].max()), "mean": float(diff[mask[1:]].mean())}
        for name, mask in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["processor", "vllm", "trainer"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vllm", help="the vLLM phase's output (trainer phase)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    {"processor": run_processor, "vllm": run_vllm, "trainer": run_trainer}[args.phase](args)


if __name__ == "__main__":
    main()
