from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .glm_ocr_extract import render_pdf_pages
from .schema import (
    ExtractedBlock,
    ExtractedDocument,
    PageClassification,
    SourceManifestItem,
    problem_artifact_id,
)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OCR_PROMPT = """Extract the full readable text from this physics PDF page.

Return clean Markdown. Preserve equation structure in LaTeX where possible.
Do not summarize, solve, omit context, or add commentary."""
GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def extract_with_gemini(
    item: SourceManifestItem,
    work_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    prompt: str | None = DEFAULT_OCR_PROMPT,
    max_output_tokens: int = 8192,
    dpi: int = 180,
    max_pages: int | None = None,
) -> ExtractedDocument:
    artifact_dir = work_dir / problem_artifact_id(item)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    page_images = render_pdf_pages(Path(item.local_path), artifact_dir, dpi=dpi, max_pages=max_pages)
    texts = [
        generate_gemini_content(
            model_name,
            [
                {"text": prompt or DEFAULT_OCR_PROMPT},
                _image_part(image_path),
            ],
            max_output_tokens=max_output_tokens,
        ).strip()
        for image_path in page_images
    ]

    blocks = []
    markdown_parts = []
    for page_number, text in enumerate(texts, start=1):
        page_path = artifact_dir / f"page_{page_number:04d}.md"
        page_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        markdown_parts.append(f"\n\n<!-- page:{page_number} -->\n\n{text.rstrip()}")
        blocks.append(
            ExtractedBlock(
                page=page_number,
                block_id=f"p{page_number:04d}_b0001",
                type="paragraph",
                text=text,
                ocr_engine=f"gemini:{model_name}",
                confidence=None,
            )
        )

    markdown = "\n".join(markdown_parts).strip() + "\n"
    classifications = [
        PageClassification(
            page_number=page_number,
            extraction_mode="ocr",
            text_coverage=1.0 if text.strip() else 0.0,
            formula_density=_formula_density(text),
            ocr_needed=False,
        )
        for page_number, text in enumerate(texts, start=1)
    ]
    return ExtractedDocument(
        source_id=item.source_id,
        sha256=item.sha256,
        local_path=item.local_path,
        page_classifications=classifications,
        blocks=blocks,
        canonical_markdown=markdown,
        extractor=f"gemini:{model_name}",
    )


def generate_gemini_content(
    model_name: str,
    parts: list[dict[str, Any]],
    *,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY must be set to use Gemini extraction or judging.")

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    request = urllib.request.Request(
        GENERATE_CONTENT_URL.format(model=model_name),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    payload: dict[str, Any] | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"Gemini API request failed with HTTP {exc.code}: {detail}") from exc
            time.sleep(2**attempt)
        except urllib.error.URLError as exc:
            if attempt == 3:
                raise RuntimeError(f"Gemini API request failed: {exc}") from exc
            time.sleep(2**attempt)

    if payload is None:
        raise RuntimeError("Gemini API request failed without a response payload.")
    text = _response_text(payload)
    if not text.strip():
        raise RuntimeError(f"Gemini API returned no text: {payload}")
    return text


def _image_part(path: Path) -> dict[str, Any]:
    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def _response_text(payload: dict[str, Any]) -> str:
    parts = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _formula_density(text: str) -> float:
    if not text.strip():
        return 0.0
    formula_chars = sum(1 for char in text if char in "=+-*/^_{}\\")
    return formula_chars / max(1, len(text))
