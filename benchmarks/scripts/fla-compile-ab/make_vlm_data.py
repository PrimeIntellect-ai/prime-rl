"""Write a synthetic image SFT set: random-pixel images at a few resolutions plus text of varied length."""

import argparse
import base64
import io
import json
import os
import random

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--num-samples", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

RESOLUTIONS = [(224, 224), (320, 240), (448, 448), (512, 384), (256, 512)]
WORDS = "the a red blue green square circle image shows pixel noise pattern color bright dark left right top bottom middle corner random texture grain value".split()

rng = random.Random(args.seed)
np_rng = np.random.default_rng(args.seed)


def data_url(width, height):
    pixels = np_rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def words(lo, hi):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))


os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "train.jsonl"), "w") as handle:
    for index in range(args.num_samples):
        num_images = 1 + (index % 3 == 0)
        content = []
        for _ in range(num_images):
            width, height = rng.choice(RESOLUTIONS)
            content.append({"type": "image_url", "image_url": {"url": data_url(width, height)}})
        content.append({"type": "text", "text": "Describe the image. " + words(5, 60)})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": words(10, 600)}]},
        ]
        handle.write(json.dumps({"messages": messages}) + "\n")
print(f"wrote {args.num_samples} samples to {args.out_dir}/train.jsonl")
