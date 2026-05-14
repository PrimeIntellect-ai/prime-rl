"""CLIPScorer: per-GPU fixed CLIP models for cosine similarity scoring.

Loads one CLIP model per GPU at startup, never migrates. The
score_on_gpu method is thread-safe (read-only inference). GPU selection
and concurrency control are handled by SemaphoreRouter externally.

compute_clip_cosine_similarity is migrated from rubric.py to keep
the Score Service self-contained.
"""

from __future__ import annotations

from pathlib import Path

import open_clip
import torch
from PIL import Image


def compute_clip_cosine_similarity(
    image_a: str | Path,
    image_b: str | Path,
    *,
    model,
    preprocess,
    device: str,
) -> float:
    """CLIP cosine similarity between two images."""
    img_a = Image.open(image_a).convert("RGB")
    img_b = Image.open(image_b).convert("RGB")
    a = preprocess(img_a).unsqueeze(0).to(device)
    b = preprocess(img_b).unsqueeze(0).to(device)
    with torch.no_grad():
        emb_a = model.encode_image(a)
        emb_b = model.encode_image(b)
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
        cos = (emb_a * emb_b).sum(dim=-1)
    return float(cos.item())


class CLIPScorer:
    """Per-GPU CLIP model pool, loaded at startup, never migrated."""

    def __init__(
        self,
        gpu_pool: list[int],
        model_name: str,
        pretrained: str,
    ):
        self._models: dict[int, tuple] = {}
        for gpu in gpu_pool:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=f"cuda:{gpu}",
                force_quick_gelu=(pretrained == "openai"),
            )
            model.eval()
            self._models[gpu] = (model, preprocess)
        self._model_name = model_name

    def score_on_gpu(self, image_a: str, image_b: str, gpu_id: int) -> float:
        """Compute CLIP cosine similarity on the specified GPU."""
        model, preprocess = self._models[gpu_id]
        return compute_clip_cosine_similarity(
            Path(image_a),
            Path(image_b),
            model=model,
            preprocess=preprocess,
            device=f"cuda:{gpu_id}",
        )
