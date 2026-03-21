"""Evaluate a VLM on the color-codeword task.

Generates random color images and checks if the model outputs the correct codeword.
Usage:
    uv run python scripts/eval_color_codeword.py --model Qwen/Qwen3-VL-4B-Instruct [--sft-weights path/to/weights]
"""

import argparse
import base64
import io
import random

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

COLOR_TO_LETTER = {
    "red": "A", "green": "B", "blue": "C", "yellow": "D",
    "purple": "E", "cyan": "F", "orange": "G", "white": "H", "black": "I",
}
COLOR_RGB = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "purple": (128, 0, 128), "cyan": (0, 255, 255),
    "orange": (255, 165, 0), "white": (255, 255, 255), "black": (0, 0, 0),
}

SYSTEM_PROMPT = (
    "You are looking at colored squares. Each color maps to a letter:\n"
    "Red=A, Green=B, Blue=C, Yellow=D, Purple=E, Cyan=F, Orange=G, White=H, Black=I\n"
    "After seeing the squares, output ONLY the corresponding letters with NO spaces."
)


def make_color_image(color_name: str) -> Image.Image:
    return Image.new("RGB", (100, 100), COLOR_RGB[color_name])


def evaluate(model, processor, num_examples=50, seed=123):
    rng = random.Random(seed)
    color_names = list(COLOR_TO_LETTER.keys())
    correct = 0

    for i in range(num_examples):
        n_colors = rng.randint(1, 3)
        colors = [rng.choice(color_names) for _ in range(n_colors)]
        expected = "".join(COLOR_TO_LETTER[c] for c in colors)
        images = [make_color_image(c) for c in colors]

        image_items = [{"type": "image"} for _ in colors]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": image_items + [{"type": "text", "text": f"Here are {n_colors} squares."}]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract letters A-I from response
        extracted = "".join(c for c in response.upper() if c in "ABCDEFGHI")

        is_correct = extracted == expected
        if is_correct:
            correct += 1
        if i < 5:
            print(f"  Colors: {colors} | Expected: {expected} | Got: {response!r} | Extracted: {extracted} | {'OK' if is_correct else 'WRONG'}")

    accuracy = correct / num_examples
    print(f"\nAccuracy: {correct}/{num_examples} = {accuracy:.1%}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--sft-weights", default=None, help="Path to SFT weight checkpoint")
    parser.add_argument("--num-examples", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading processor from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model_path = args.sft_weights if args.sft_weights else args.model
    print(f"Loading model from {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()

    print(f"\nEvaluating {model_path} on {args.num_examples} color-codeword examples:")
    evaluate(model, processor, num_examples=args.num_examples)


if __name__ == "__main__":
    main()
