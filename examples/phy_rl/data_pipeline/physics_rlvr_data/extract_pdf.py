from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .schema import ExtractedBlock, ExtractedDocument, PageClassification, SourceManifestItem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_embedded_pdf_text(item: SourceManifestItem) -> ExtractedDocument:
    path = Path(item.local_path)
    pages = _extract_with_pymupdf(path)
    extractor = "pymupdf"
    if pages is None:
        pages = _extract_with_pypdf(path)
        extractor = "pypdf"
    if pages is None:
        raise RuntimeError(
            "Embedded extraction requires PyMuPDF or pypdf. "
            "Use --extractor glm-ocr for scanned or encoded PDFs."
        )

    blocks: list[ExtractedBlock] = []
    classifications: list[PageClassification] = []
    markdown_parts: list[str] = []
    for page_number, text in enumerate(pages, start=1):
        coverage = _text_coverage(text)
        formula_density = _formula_density(text)
        ocr_needed = coverage < 0.55 or _looks_corrupt(text)
        mode = "ocr" if ocr_needed else "embedded_text"
        classifications.append(
            PageClassification(
                page_number=page_number,
                extraction_mode=mode,
                text_coverage=coverage,
                formula_density=formula_density,
                ocr_needed=ocr_needed,
            )
        )
        block_id = f"p{page_number:04d}_b0001"
        blocks.append(
            ExtractedBlock(
                page=page_number,
                block_id=block_id,
                type="paragraph",
                text=text.strip(),
                ocr_engine=extractor,
                confidence=None,
            )
        )
        markdown_parts.append(f"\n\n<!-- page:{page_number} -->\n\n{text.strip()}")

    return ExtractedDocument(
        source_id=item.source_id,
        sha256=item.sha256,
        local_path=item.local_path,
        page_classifications=classifications,
        blocks=blocks,
        canonical_markdown="\n".join(markdown_parts).strip() + "\n",
        extractor=extractor,
    )


def _extract_with_pymupdf(path: Path) -> list[str] | None:
    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        return None
    document = fitz.open(path)
    return [page.get_text("text") for page in document]


def _extract_with_pypdf(path: Path) -> list[str] | None:
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError:
        return None
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def _text_coverage(text: str) -> float:
    if not text:
        return 0.0
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0
    useful = [char for char in non_space if char.isprintable() and char != "\ufffd"]
    return len(useful) / len(non_space)


def _formula_density(text: str) -> float:
    if not text.strip():
        return 0.0
    formula_chars = sum(1 for char in text if char in "=+-*/^_{}\\")
    return formula_chars / max(1, len(text))


def _looks_corrupt(text: str) -> bool:
    if "\ufffd" in text:
        return True
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(text) > 200 and len(words) < 5:
        return True
    return False
