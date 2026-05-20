import sys

import torch
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config

from prime_rl.trainer.models.olmo3 import Olmo3ForCausalLM
from prime_rl.utils.utils import default_dtype


def _olmo3_7b_debug_config(attn_impl: str, vocab_size: int = 100278, pad_token_id: int = 100277) -> Olmo3Config:
    config = Olmo3Config(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=4096,
        intermediate_size=11008,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        max_position_embeddings=65536,
        num_attention_heads=32,
        num_hidden_layers=4,
        num_key_value_heads=32,
        pad_token_id=pad_token_id,
        rms_norm_eps=1e-6,
        rope_parameters={
            "attention_factor": 1.2079441541679836,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "rope_type": "yarn",
        },
        sliding_window=4096,
        tie_word_embeddings=False,
        vocab_size=vocab_size,
    )
    config._attn_implementation = attn_impl
    return config


def main() -> int:
    if len(sys.argv) not in (2, 3, 4):
        raise ValueError("usage: olmo3_flash_probe.py <attn_impl> [vocab_size] [pad_token_id]")
    attn_impl = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 100278
    pad_token_id = int(sys.argv[3]) if len(sys.argv) == 4 else min(100277, vocab_size - 1)

    torch.manual_seed(0)
    config = _olmo3_7b_debug_config(attn_impl, vocab_size, pad_token_id)
    print("stage=construct", file=sys.stderr, flush=True)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Olmo3ForCausalLM._from_config(config)
    model.eval()

    sequences = [
        torch.randint(0, model.config.vocab_size, (1, 4101), device="cuda"),
        torch.randint(0, model.config.vocab_size, (1, 4099), device="cuda"),
    ]
    packed_input_ids = torch.cat(sequences, dim=1)
    packed_position_ids = torch.cat([torch.arange(seq.shape[1], device="cuda") for seq in sequences]).unsqueeze(0)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        print("stage=packed_forward", file=sys.stderr, flush=True)
        packed_output = model.model(input_ids=packed_input_ids, position_ids=packed_position_ids).last_hidden_state
        print("stage=independent_forward", file=sys.stderr, flush=True)
        independent_output = torch.cat(
            [
                model.model(
                    input_ids=seq,
                    position_ids=torch.arange(seq.shape[1], device="cuda").unsqueeze(0),
                ).last_hidden_state
                for seq in sequences
            ],
            dim=1,
        )

    hidden_diff = packed_output - independent_output
    if not torch.allclose(hidden_diff, torch.zeros_like(hidden_diff), atol=5e-2, rtol=5e-2):
        print(f"Max hidden-state diff: {hidden_diff.abs().max()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
