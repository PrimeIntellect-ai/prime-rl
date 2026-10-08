"""LOCAL MOCK, not for commit: fake RL micro batches carrying real JPEG images for Qwen3.5 VLMs.

Configured by PRL_MOCK_MM_IMAGE_PX (square side, multiple of 32) and PRL_MOCK_MM_IMAGES_PER_SAMPLE.
"""

import base64
import io
import os

import numpy as np
import torch
from PIL import Image

from prime_rl.transports.batch import MMImageRef, MMRefs

VISION_START, IMAGE_PAD, VISION_END = 248053, 248056, 248054
PROMPT_TOKENS, COMPLETION_TOKENS, POOL_SIZE = 64, 256, 16


def enabled() -> bool:
    return "PRL_MOCK_MM_IMAGE_PX" in os.environ


def _photo_like_data_url(side: int, rng: np.random.Generator) -> str:
    y, x = np.mgrid[0:side, 0:side]
    phase = rng.uniform(0, 2 * np.pi, 3)
    base = np.stack([127 + 100 * np.sin(x / (17 + 7 * c) + y / (23 + 5 * c) + phase[c]) for c in range(3)], -1)
    pixels = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class MockMMMicroBatches:
    def __init__(self, seq_len: int):
        self.side = int(os.environ["PRL_MOCK_MM_IMAGE_PX"])
        if self.side % 32:
            raise ValueError("PRL_MOCK_MM_IMAGE_PX must be a multiple of 32")
        self.images_per_sample = int(os.environ.get("PRL_MOCK_MM_IMAGES_PER_SAMPLE", "2"))
        self.image_tokens = (self.side // 32) ** 2
        self.sample_len = PROMPT_TOKENS + self.images_per_sample * (self.image_tokens + 2) + COMPLETION_TOKENS
        if self.sample_len > seq_len:
            raise ValueError(f"One mock sample ({self.sample_len} tokens) exceeds seq_len {seq_len}")
        self.seq_len = seq_len
        rng = np.random.default_rng(0)
        self.urls = [_photo_like_data_url(self.side, rng) for _ in range(POOL_SIZE)]

    def micro_batch(self, generator: torch.Generator) -> dict:
        input_ids: list[int] = []
        mm_token_type_ids: list[int] = []
        loss_mask: list[bool] = []
        position_ids: list[int] = []
        sequence_lengths: list[int] = []
        refs: list[MMImageRef] = []
        while len(input_ids) + self.sample_len <= self.seq_len:
            sample: list[int] = torch.randint(0, 120000, (PROMPT_TOKENS,), generator=generator).tolist()
            types = [0] * PROMPT_TOKENS
            for _ in range(self.images_per_sample):
                url = self.urls[int(torch.randint(0, POOL_SIZE, (1,), generator=generator))]
                refs.append(MMImageRef(url=url, offset=len(input_ids) + len(sample) + 1, length=self.image_tokens))
                sample += [VISION_START] + [IMAGE_PAD] * self.image_tokens + [VISION_END]
                types += [0] + [1] * self.image_tokens + [0]
            sample += torch.randint(0, 120000, (COMPLETION_TOKENS,), generator=generator).tolist()
            types += [0] * COMPLETION_TOKENS
            loss_mask += [False] * (len(sample) - COMPLETION_TOKENS) + [True] * COMPLETION_TOKENS
            input_ids += sample
            mm_token_type_ids += types
            position_ids += list(range(len(sample)))
            sequence_lengths.append(len(sample))
        num_tokens = len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            "position_ids": torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
            "advantages": torch.randn(num_tokens, generator=generator).unsqueeze(0),
            "inference_logprobs": -torch.rand(num_tokens, generator=generator).unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(num_tokens).unsqueeze(0),
            "env_names": ["mock-mm"] * num_tokens,
            "sequence_lengths": sequence_lengths,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": torch.tensor([num_tokens], dtype=torch.int32),
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "mm_refs": MMRefs(images=refs),
            "mm_token_type_ids": torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0),
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
        }
