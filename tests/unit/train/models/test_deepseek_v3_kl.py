import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import pytest
from transformers import AutoTokenizer


def load_math_dataset(
    split: str = "test",
    max_samples: int = 512,
    dataset_name: str = "EleutherAI/hendrycks_math",
):
    """
    Loads the Hendrycks MATH dataset with optional subsampling.

    Returns:
        list[dict]: Each dict has keys: "problem", "solution", "type", "level".
    """
    dataset = load_dataset(dataset_name, "algebra", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return [
        {
            "problem": row["problem"],
            "solution": row["solution"],
            "type": row["type"],
            "level": row["level"],
        }
        for row in dataset
    ]


def format_math_prompt(problem: str) -> str:
    """
    Formats the MATH problem into a prompt ready for model generation.
    Example: "Problem: <text>\nAnswer:"
    """
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"{problem}"},
    ]


def get_devices():
    device1 = "cuda:0"  # prime model
    device2 = "cuda:1"
    return device1, device2


@pytest.mark.skip(reason="up to 20 min with 2 gpus A100 (80gb vram)")
def test_kl_divergence_on_math(
    batch_size=64,
    max_steps=20,  # number of batches (not generation steps)
    atol_kl=0.015,
):
    """
    Evaluate KL divergence between HF and Prime DeepSeek-V3 on real MATH prompts.

    Args:
        atol_kl: Max acceptable avg KL per token. Must be < 0.015 across all batches.
    """

    # lazy import
    from tests.unit.train.models.test_deepseek_v3 import get_model_pairs

    math_dataset = load_math_dataset(split="train", max_samples=batch_size * max_steps)

    device_prime, device_hf = get_devices()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

    # Pre-tokenize prompts
    prompts = [format_math_prompt(d["problem"]) for d in math_dataset]

    # offload to cpu
    with torch.device("cpu"):
        tokenized = tokenizer.apply_chat_template(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        input_ids = tokenized.input_ids.to("cpu")
        attention_mask = tokenized.attention_mask.to("cpu")

    hf_model, prime_model = get_model_pairs(
        device_prime=device_prime, device_hf=device_hf, vocab_size=len(tokenizer)
    )

    # Ensure models are in eval mode
    hf_model.eval()
    prime_model.eval()

    # Batched eval loop
    all_kls = []
    total_tokens = 0
    batch_idx = 0

    with torch.no_grad():
        for i in tqdm(
            range(0, len(input_ids), batch_size),
            desc="KL eval",
            total=len(input_ids) // batch_size,
        ):
            if batch_idx >= max_steps:
                break  # Avoid going too far

            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]

            if batch_input_ids.size(0) == 0:
                continue

            # Run forward on full prompt (pre-fill only — no generation)
            hf_out = hf_model(
                input_ids=batch_input_ids.to(device_hf),
                attention_mask=batch_attention_mask.to(device_hf),
            )
            prime_out = prime_model(batch_input_ids.to(device_prime))

            # Compute log-probs on CPU to reduce memory consumption
            logprobs_hf = torch.log_softmax(hf_out.logits, dim=-1).to("cpu")
            logprobs_prime = torch.log_softmax(prime_out["logits"], dim=-1).to("cpu")

            # clean cache
            del hf_out, prime_out

            # Mask to avoid padding
            mask = batch_attention_mask.bool().to("cpu")
            num_valid = mask.sum()

            # KL[P||Q] = Σ p log(p/q)
            # where p = softmax(logprobs_prime), q = softmax(logprobs_hf)
            kl_per_token = torch.exp(logprobs_prime) * (logprobs_prime - logprobs_hf)
            kl_batch = (kl_per_token * mask.unsqueeze(-1)).sum() / (num_valid + 1e-8)
            all_kls.append(kl_batch.item())
            total_tokens += num_valid.item()

            batch_idx += 1

    mean_kl = sum(all_kls) / len(all_kls) if all_kls else 0.0
    print(
        f"\n[KL Mismatch Test] Avg KL per token = {mean_kl:.10f} over {batch_idx} batches"
    )
    print(f"Total tokens evaluated: {total_tokens}")
    print(f"KL per batch: {all_kls}")

    assert mean_kl < atol_kl, (
        f"KL mismatch too high: {mean_kl:.10f} ≥ {atol_kl}. "
        "Model outputs diverge significantly from HF baseline."
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # add project root (prime-rl/tests/..)
    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))
    test_kl_divergence_on_math()
