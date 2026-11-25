"""Images as the base64 data URLs of a System One record."""

import base64
from io import BytesIO

from PIL import Image


def data_url(image: Image.Image, image_format: str = "PNG", **kwargs) -> str:
    buffer = BytesIO()
    image.save(buffer, image_format, **kwargs)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def gradient(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    image.putdata([(x * 7 % 256, y * 5 % 256, (x + y) % 256) for y in range(height) for x in range(width)])
    return image


WIDE, TALL = gradient(96, 64), gradient(64, 128)
EPS = data_url(WIDE, "EPS").replace("image/eps", "image/png")  # a format vLLM refuses, whatever the URL says


def images_of_every_kind() -> dict[str, str]:
    transparent = WIDE.convert("RGBA")
    transparent.putalpha(Image.linear_gradient("L").resize(WIDE.size))
    rotated = Image.Exif()
    rotated[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to view
    return {
        "rgb": data_url(WIDE),
        "grey": data_url(WIDE.convert("L")),
        "rgba": data_url(transparent),
        "la": data_url(transparent.convert("LA")),
        "palette": data_url(WIDE.convert("P")),
        "palette with transparency": data_url(WIDE.convert("P"), transparency=0),
        "rgb with a transparent colour": data_url(WIDE, transparency=(0, 0, 0)),
        "jpeg": data_url(WIDE, "JPEG"),
        "rotated jpeg": data_url(WIDE, "JPEG", exif=rotated),
    }
