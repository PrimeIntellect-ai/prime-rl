from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import (
    ExtractedBlock,
    ExtractedDocument,
    PageClassification,
    SourceManifestItem,
    problem_artifact_id,
)

DEFAULT_MODEL = "zai-org/GLM-OCR"
DEFAULT_PROMPT = "Text Recognition:"


def extract_with_glm_ocr(
    item: SourceManifestItem,
    work_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 8192,
    dpi: int = 180,
    max_pages: int | None = None,
) -> ExtractedDocument:
    artifact_dir = work_dir / problem_artifact_id(item)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    page_images = render_pdf_pages(Path(item.local_path), artifact_dir, dpi=dpi, max_pages=max_pages)
    texts = run_glm_ocr_on_images(
        page_images,
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

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
                ocr_engine=f"glm-ocr:{model_name}",
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
        extractor=f"glm-ocr:{model_name}",
    )


def render_pdf_pages(pdf_path: Path, out_dir: Path, *, dpi: int, max_pages: int | None = None) -> list[Path]:
    try:
        import fitz  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("GLM-OCR extraction requires PyMuPDF to render PDF pages.") from exc

    image_paths = []
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    document = fitz.open(pdf_path)
    for page_index, page in enumerate(document, start=1):
        if max_pages is not None and page_index > max_pages:
            break
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = out_dir / f"page_{page_index:04d}.png"
        pixmap.save(image_path)
        image_paths.append(image_path)
    return image_paths


def run_glm_ocr_on_images(
    image_paths: list[Path],
    *,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("GLM-OCR extraction requires torch and transformers.") from exc

    processor, model = _load_glm_ocr_runtime(model_name)
    device = _select_device(torch)
    model = model.to(device)

    outputs = []
    for image_path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )
        outputs.append(output_text.strip())
    return outputs


def _load_glm_ocr_processor(model_name: str) -> Any:
    try:
        from transformers import AutoProcessor  # noqa: PLC0415

        return AutoProcessor.from_pretrained(model_name)
    except ValueError:
        from transformers import AutoTokenizer  # noqa: PLC0415
        from transformers.models.glm46v import (  # noqa: PLC0415
            Glm46VImageProcessor,
            Glm46VProcessor,
            Glm46VVideoProcessor,
        )

        return Glm46VProcessor(
            image_processor=Glm46VImageProcessor.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            video_processor=Glm46VVideoProcessor.from_pretrained(model_name),
        )


@lru_cache(maxsize=2)
def _load_glm_ocr_runtime(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText  # noqa: PLC0415

    processor = _load_glm_ocr_processor(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype="auto",
    )
    return processor, model


def _select_device(torch: Any) -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _formula_density(text: str) -> float:
    if not text.strip():
        return 0.0
    formula_chars = sum(1 for char in text if char in "=+-*/^_{}\\")
    return formula_chars / max(1, len(text))
