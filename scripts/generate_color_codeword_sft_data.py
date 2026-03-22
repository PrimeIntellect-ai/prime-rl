"""Generate synthetic SFT data for the color-codeword environment.

Creates a HF dataset with multimodal prompt/completion pairs that teach the
model the color-to-letter mapping:
  Red→A, Green→B, Blue→C, Yellow→D, Purple→E, Cyan→F, Orange→G, White→H, Black→I
"""

import base64
import io
import json
import random
from pathlib import Path

from PIL import Image

COLOR_TO_LETTER = {
    "red": "A",
    "green": "B",
    "blue": "C",
    "yellow": "D",
    "purple": "E",
    "cyan": "F",
    "orange": "G",
    "white": "H",
    "black": "I",
}

COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

SYSTEM_PROMPT = (
    "You are looking at colored squares. Each color maps to a letter:\n"
    "Red=A, Green=B, Blue=C, Yellow=D, Purple=E, Cyan=F, Orange=G, White=H, Black=I\n"
    "After seeing the squares, output ONLY the corresponding letters with NO spaces."
)


def make_color_image(color_name: str, size: int = 100) -> str:
    """Create a solid-color image and return as base64 data URL."""
    rgb = COLOR_RGB[color_name]
    img = Image.new("RGB", (size, size), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def make_single_turn_example(colors: list[str]) -> dict:
    """Create a single-turn example: show N colors, expect the codeword."""
    image_items = [{"type": "image_url", "image_url": {"url": make_color_image(c)}} for c in colors]
    n = len(colors)
    text = f"Here are {n} square{'s' if n > 1 else ''}."
    codeword = "".join(COLOR_TO_LETTER[c] for c in colors)

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_items + [{"type": "text", "text": text}]},
    ]
    completion = [{"role": "assistant", "content": codeword}]
    return {"prompt": prompt, "completion": completion}


def make_multi_turn_example(colors_per_turn: list[list[str]]) -> dict:
    """Create a multi-turn example with accumulated codeword."""
    prompt = [{"role": "system", "content": SYSTEM_PROMPT}]
    accumulated = ""

    for turn_idx, colors in enumerate(colors_per_turn):
        image_items = [{"type": "image_url", "image_url": {"url": make_color_image(c)}} for c in colors]
        n = len(colors)

        if turn_idx == 0:
            text = f"Here are {n} square{'s' if n > 1 else ''}."
        elif turn_idx == len(colors_per_turn) - 1:
            total = sum(len(t) for t in colors_per_turn)
            text = f"Here are {n} more square{'s' if n > 1 else ''}. Combine your previous answer with these new letters to output all {total} letters."
        else:
            text = f"Here are {n} more square{'s' if n > 1 else ''}."

        prompt.append({"role": "user", "content": image_items + [{"type": "text", "text": text}]})
        accumulated += "".join(COLOR_TO_LETTER[c] for c in colors)

        if turn_idx < len(colors_per_turn) - 1:
            prompt.append({"role": "assistant", "content": accumulated})

    completion = [{"role": "assistant", "content": accumulated}]
    return {"prompt": prompt, "completion": completion}


def generate_dataset(num_examples: int = 500, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    color_names = list(COLOR_TO_LETTER.keys())
    examples = []

    # Single-turn: 1-3 colors
    for _ in range(num_examples // 2):
        n_colors = rng.randint(1, 3)
        colors = [rng.choice(color_names) for _ in range(n_colors)]
        examples.append(make_single_turn_example(colors))

    # Multi-turn: 2-3 turns, 1 color per turn
    for _ in range(num_examples - num_examples // 2):
        n_turns = rng.randint(2, 3)
        colors_per_turn = [[rng.choice(color_names)] for _ in range(n_turns)]
        examples.append(make_multi_turn_example(colors_per_turn))

    rng.shuffle(examples)
    return examples


def main():
    output_dir = Path("data/color_codeword_sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_dataset(num_examples=500, seed=42)

    # Save as JSONL with prompt/completion serialized as JSON strings
    # (PyArrow can't handle mixed string/list content types in nested dicts)
    output_path = output_dir / "train.jsonl"
    with open(output_path, "w") as f:
        for ex in examples:
            row = {
                "prompt": json.dumps(ex["prompt"]),
                "completion": json.dumps(ex["completion"]),
            }
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(examples)} examples to {output_path}")

    # Verify a sample
    sample = examples[0]
    print(f"\nSample prompt roles: {[m['role'] for m in sample['prompt']]}")
    print(f"Sample completion: {sample['completion'][0]['content']}")


if __name__ == "__main__":
    main()
