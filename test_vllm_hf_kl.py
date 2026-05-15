# This script should be run with "https://github.com/nreHieW/transformers" which uses the unfused official implementation to compare the official vs the unfused vs vllm
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from prime_rl.trainer.models.zaya import ZayaConfig
from prime_rl.trainer.models.zaya import ZayaForCausalLM as PrimeZayaForCausalLM
from prime_rl.trainer.weights import load_state_dict

PROMPTS_PER_ITERATION = 50
NUM_ITERATIONS = 50

SAMPLING = dict(max_tokens=32, temperature=1.0, top_p=1.0, logprobs=1)
VLLM_KWARGS = dict(
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=2048,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
)


@dataclass
class SequenceLogprobs:
    prompt: str | None
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    vllm_logprobs: list[float]


RunBatch = tuple[list[SequenceLogprobs], list[list[dict[str, Any]]]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare PrimeRL-style mismatch KL between HuggingFace and vLLM ZAYA.")
    p.add_argument("--prime-rl-model")
    p.add_argument("--hf-model")
    p.add_argument("--vllm-model", default="Zyphra/ZAYA1-8B")
    return p.parse_args()


def reverse_text_chat_messages(num_prompts: int, seed: int) -> list[list[dict[str, Any]]]:
    from verifiers.utils.env_utils import load_environment

    env = load_environment("reverse-text")
    rng = random.Random(seed)
    n = min(num_prompts, len(env.dataset))
    out = []
    for i in rng.sample(range(len(env.dataset)), k=n):
        row = env.dataset[i]
        msgs = [dict(m) for m in row["prompt"]]
        msgs.append({"role": "assistant", "content": str(row["answer"])})
        out.append(msgs)
    return out


def chat_template_prompt_token_ids(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    """Same path as vLLM ``chat()``: chat template + generation prompt."""
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(encoded, dict) or (hasattr(encoded, "keys") and "input_ids" in encoded):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(t) for t in encoded]


def extract_vllm_sequences(outputs) -> list[SequenceLogprobs]:
    out: list[SequenceLogprobs] = []
    for ro in outputs:
        c = ro.outputs[0]
        tids = list(c.token_ids)
        lps = c.logprobs or []
        if len(lps) != len(tids):
            raise ValueError(f"vLLM returned {len(lps)} logprob entries for {len(tids)} tokens")
        selected: list[float] = []
        for tid, cand in zip(tids, lps, strict=True):
            if tid not in cand:
                raise KeyError(f"Sampled token {tid} missing from vLLM logprobs")
            selected.append(float(cand[tid].logprob))
        out.append(
            SequenceLogprobs(
                prompt=getattr(ro, "prompt", None),
                prompt_token_ids=list(ro.prompt_token_ids),
                completion_token_ids=tids,
                vllm_logprobs=selected,
            )
        )
    return out


def model_completion_logprobs(model, device: torch.device, sequences: list[SequenceLogprobs]):
    input_ids = [s.prompt_token_ids + s.completion_token_ids for s in sequences]
    max_len = max(len(ids) for ids in input_ids)
    pad = int(model.config.pad_token_id if model.config.pad_token_id is not None else 0)
    batch = torch.full((len(input_ids), max_len), pad, dtype=torch.long, device=device)
    attn = torch.zeros_like(batch)
    loss_mask = torch.zeros_like(batch, dtype=torch.bool)
    for i, (ids, s) in enumerate(zip(input_ids, sequences, strict=True)):
        L, p = len(ids), len(s.prompt_token_ids)
        batch[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, :L] = 1
        loss_mask[i, p:L] = True
    with torch.no_grad():
        o = model(input_ids=batch, attention_mask=attn, use_cache=False)
        logits = (o["logits"] if isinstance(o, dict) else o.logits).float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, batch[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp, loss_mask[:, 1:]


def vllm_logprobs_grid(model_lp: torch.Tensor, sequences: list[SequenceLogprobs], device: torch.device) -> torch.Tensor:
    g = torch.zeros_like(model_lp)
    for row, s in enumerate(sequences):
        a = len(s.prompt_token_ids) - 1
        g[row, a : a + len(s.vllm_logprobs)] = torch.tensor(s.vllm_logprobs, dtype=model_lp.dtype, device=device)
    return g


def mismatch_kl_masked(model_lp: torch.Tensor, vllm_lp: torch.Tensor, loss_mask: torch.Tensor):
    d = model_lp - vllm_lp
    k = d.exp() - d - 1
    return k[loss_mask], d[loss_mask]


def batch_mismatch_kl_cpu(
    model, device: torch.device, sequences: list[SequenceLogprobs]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_lp, mask = model_completion_logprobs(model, device, sequences)
    vllm_lp = vllm_logprobs_grid(model_lp, sequences, device)
    mk, md = mismatch_kl_masked(model_lp, vllm_lp, mask)
    return mk.detach().cpu(), md.detach().cpu(), mask


def load_prime_model(model_name: str, dtype: torch.dtype, device: torch.device):
    snap = Path(snapshot_download(repo_id=model_name, repo_type="model"))
    cfg = ZayaConfig.from_pretrained(snap)
    m = PrimeZayaForCausalLM._from_config(cfg)
    sd = load_state_dict(snap)
    PrimeZayaForCausalLM.convert_to_prime(sd)
    bad = m.load_state_dict(sd, strict=False)
    if bad.missing_keys or bad.unexpected_keys:
        print(
            "WARNING: PrimeRL load_state_dict non-strict "
            f"missing={len(bad.missing_keys)} unexpected={len(bad.unexpected_keys)}"
        )
    return m.to(device=device, dtype=dtype).eval()


def warn_tokenizer_mismatches(
    vllm_tok, hf_tok, sequences: list[SequenceLogprobs], messages: list[list[dict[str, Any]]]
):
    for seq, msg in zip(sequences, messages, strict=True):
        rt = chat_template_prompt_token_ids(vllm_tok, msg)
        if rt != seq.prompt_token_ids:
            print("WARNING: vLLM tokenizer apply_chat_template != RequestOutput.prompt_token_ids")
            print("  template:", rt, "\n  vLLM:    ", seq.prompt_token_ids)
        hf_ids = chat_template_prompt_token_ids(hf_tok, msg)
        if hf_ids == seq.prompt_token_ids:
            continue
        print(
            "NOTE: HF tokenizer chat-template ids differ from vLLM; HF/PrimeRL still use vLLM prompt_token_ids. "
            "Align --vllm-model with --hf-model for identical tokenization."
        )
        if seq.prompt:
            p = seq.prompt
            print("  prompt (decoded):", (p[:200] + "...") if len(p) > 200 else p)
        print("  HF (apply_chat_template):", hf_ids, "\n  vLLM:                      ", seq.prompt_token_ids)


def collect_vllm_runs(llm: LLM) -> list[RunBatch]:
    runs: list[RunBatch] = []
    for it in range(NUM_ITERATIONS):
        msgs = reverse_text_chat_messages(PROMPTS_PER_ITERATION, seed=it)
        sp = SamplingParams(**SAMPLING, seed=it)
        seqs = extract_vllm_sequences(llm.chat(msgs, sampling_params=sp))
        runs.append((seqs, msgs))
        print(f"vLLM iteration {it + 1}/{NUM_ITERATIONS}: {len(seqs)} sequences")
    return runs


def print_aggregate_kl(name: str, mismatch_kl: torch.Tensor, logprob_delta: torch.Tensor) -> None:
    n = mismatch_kl.numel()
    print(f"\n{name} vs vLLM (aggregate over {n} tokens)")
    print(f"tokens compared: {n}")
    print(f"mismatch_kl/mean: {mismatch_kl.mean().item():.8f}")
    print(f"mismatch_kl/max : {mismatch_kl.max().item():.8f}")
    print(f"logprob_delta mean/max_abs: {logprob_delta.mean().item():.8f} / {logprob_delta.abs().max().item():.8f}")


def cuda_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.prime_rl_model is None:
        args.prime_rl_model = args.hf_model
        print(f"WARNING: --prime-rl-model not set, using --hf-model: {args.prime_rl_model}")

    n_prompts = PROMPTS_PER_ITERATION * NUM_ITERATIONS
    print(f"Running {NUM_ITERATIONS} × {PROMPTS_PER_ITERATION} prompts = {n_prompts} total")

    print(f"Loading vLLM model: {args.vllm_model}")
    llm = LLM(model=args.vllm_model, **VLLM_KWARGS)
    runs = collect_vllm_runs(llm)

    dtype = torch.bfloat16
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(f"Loading HF model: {args.hf_model} on {device}")
    hf_tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    hf = (
        AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    )
    vllm_tok = llm.get_tokenizer()

    hf_kl, hf_d, hf_masks = [], [], []
    for it, (seqs, msgs) in enumerate(runs):
        if it == 0:
            warn_tokenizer_mismatches(vllm_tok, hf_tok, seqs, msgs)
        mk, md, lm = batch_mismatch_kl_cpu(hf, device, seqs)
        hf_kl.append(mk)
        hf_d.append(md)
        hf_masks.append(lm)

    del hf
    cuda_gc()

    print(f"Loading PrimeRL model from HF checkpoint: {args.prime_rl_model}")
    prime = load_prime_model(args.prime_rl_model, dtype, device)
    prime_kl, prime_d = [], []
    for it, (seqs, _) in enumerate(runs):
        mk, md, lm = batch_mismatch_kl_cpu(prime, device, seqs)
        if not torch.equal(hf_masks[it], lm):
            raise ValueError(f"HF and PrimeRL loss masks differ for iteration {it}")
        prime_kl.append(mk)
        prime_d.append(md)

    del prime
    cuda_gc()

    hf_kl_all, hf_d_all = torch.cat(hf_kl), torch.cat(hf_d)
    prime_kl_all, prime_d_all = torch.cat(prime_kl), torch.cat(prime_d)

    print("\nPrimeRL mismatch KL: exp(dlogp) - dlogp - 1   where dlogp = model_logp - vllm_logp")
    print_aggregate_kl(f"HF ({args.hf_model})", hf_kl_all, hf_d_all)
    print_aggregate_kl(f"PrimeRL ({args.prime_rl_model})", prime_kl_all, prime_d_all)


if __name__ == "__main__":
    main()
