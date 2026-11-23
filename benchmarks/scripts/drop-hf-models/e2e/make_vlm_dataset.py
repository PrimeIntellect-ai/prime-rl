"""Write a deterministic synthetic image-captioning SFT dataset (messages column, JSONL) for the VLM cell."""

import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (40, 80, 220), "yellow": (230, 210, 40)}
SHAPES = ("square", "circle", "triangle")
SIDES = (128, 160, 224, 256)


def draw(shape: str, color: tuple[int, int, int], background: tuple[int, int, int], width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), background)
    canvas = ImageDraw.Draw(image)
    box = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    if shape == "square":
        canvas.rectangle(box, fill=color)
    elif shape == "circle":
        canvas.ellipse(box, fill=color)
    else:
        canvas.polygon([(width // 2, height // 4), (width // 4, 3 * height // 4), (3 * width // 4, 3 * height // 4)], fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=1024)
    args = parser.parse_args()

    rng = random.Random(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.jsonl").open("w") as file:
        for _ in range(args.num_samples):
            shape = rng.choice(SHAPES)
            color, background = rng.sample(sorted(COLORS), 2)
            width, height = rng.choice(SIDES), rng.choice(SIDES)
            image_url = draw(shape, COLORS[color], COLORS[background], width, height)
            answer = f"A {color} {shape} on a {background} background, in a {width} by {height} image."
            row = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "Describe the shape, its color, the background and the image size."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            }
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
