import base64, io, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor
from prime_rl.trainer.multimodal import _load_image

torch.set_num_threads(1)
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B").image_processor
rng = np.random.default_rng(0)

def photo_like(w, h):
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 / w), (y * 255 / h), ((x + y) * 127 / (w + h))], -1)
    return Image.fromarray(np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8))

def data_url(img, fmt):
    buf = io.BytesIO(); img.save(buf, fmt, quality=90) if fmt == "JPEG" else img.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

cases = [("100x100 solid PNG (color-codeword)", data_url(Image.new("RGB", (100, 100), (255, 0, 0)), "PNG"))]
for w, h in [(275, 275), (512, 512), (1024, 1024), (1448, 1448)]:
    cases.append((f"{w}x{h} photo-like JPEG", data_url(photo_like(w, h), "JPEG")))
for name, url in cases:
    for _ in range(2):
        proc(images=[_load_image(url)], return_tensors="pt")
    n = 20; t0 = time.perf_counter()
    for _ in range(n):
        out = proc(images=[_load_image(url)], return_tensors="pt")
    ms = (time.perf_counter() - t0) / n * 1e3
    tokens = int(out["image_grid_thw"].prod()) // (proc.merge_size ** 2)
    print(f"{name:36s} {ms:8.1f} ms/image  {tokens:6d} image tokens  {len(url)/1e3:8.0f} kB base64")
