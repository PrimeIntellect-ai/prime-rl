# Ported from vLLM v0.34.0 (Apache-2.0, Copyright contributors to the vLLM project), with vLLM's names:
# - vllm/multimodal/image.py: ALLOWED_IMAGE_FORMATS, open_image, normalize_image, rgba_to_rgb, _has_transparency and
#   convert_image_mode, to RGB on white.
# - vllm/multimodal/media/image.py: ImageMediaIO's load_base64 and load_bytes, with its defaults, as load_image.
# Left out, as they bound a server's work: the limits on pixels and decoded bytes.
"""A record's images as vLLM loads a request's."""

import base64
import contextlib
from io import BytesIO

from PIL import Image, ImageOps

# The still-image formats Pillow decodes in-process. Others may run an external program: EPS runs Ghostscript.
ALLOWED_IMAGE_FORMATS = (
    "AVIF BLP BMP CUR DCX DDS DIB FITS FLI FTEX GBR GIF ICNS ICO IM IMT IPTC JPEG JPEG2000 MCIDAS MSP "
    "PCD PCX PIXAR PNG PPM PSD QOI SGI SPIDER SUN TGA TIFF WEBP XBM XPM XVTHUMB"
).split()


def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize EXIF orientation so the pixel data matches visual display."""
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    return image


def _has_transparency(image: Image.Image) -> bool:
    """Detect whether an image carries transparency data (RGBA, LA, PA, or tRNS chunk in P/L/RGB PNGs)."""
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in getattr(image, "info", {})


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """The image in RGB, with transparency on a white background."""
    if _has_transparency(image):
        image = image.convert("RGBA")
        converted = Image.new("RGB", image.size, (255, 255, 255))
        converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return converted
    return image.convert("RGB")


def load_image(url: str) -> Image.Image:
    """The RGB image of a base64 data URL. Raises ValueError if it does not decode."""
    try:
        data = BytesIO(base64.b64decode(url.partition(",")[2], validate=True))
        image = Image.open(data, formats=ALLOWED_IMAGE_FORMATS)
        image = normalize_image(image)
        image.load()
        return convert_to_rgb(image)
    # Pillow raises SyntaxError for broken PNG chunks and RuntimeError for undecodable data
    except (ValueError, OSError, SyntaxError, RuntimeError, Image.DecompressionBombError) as error:
        raise ValueError(f"Failed to load image: {error}") from error
