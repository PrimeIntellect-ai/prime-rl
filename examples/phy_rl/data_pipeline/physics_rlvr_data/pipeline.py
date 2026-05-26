from __future__ import annotations

import csv
import hashlib
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from .dedup import deduplicate
from .extract_pdf import extract_embedded_pdf_text, sha256_file
from .filters import admissibility_rejection
from .glm_ocr_extract import extract_with_glm_ocr
from .io import ensure_parent, read_json, read_jsonl, write_json, write_jsonl
from .policy import choose_split, validate_training_policy
from .schema import (
    Answer,
    ExtractedDocument,
    FinalItem,
    Provenance,
    RejectedItem,
    SourceManifestItem,
    answer_from_dict,
    final_item_from_dict,
    problem_artifact_id,
    to_dict,
)

DEFAULT_DATA_DIR = Path("examples/phy_rl/data_pipeline/data")
Extractor = Literal["glm-ocr", "embedded", "auto"]


class PdfLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href and href.lower().endswith(".pdf"):
            self.links.append(href)


def init_layout(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    for relative in [
        "raw_pdfs",
        "extracted",
        "candidates",
        "verified",
        "rejected",
        "final",
        "audits",
        "manual_corrections",
    ]:
        (data_dir / relative).mkdir(parents=True, exist_ok=True)


def manifest_from_local_pdfs(pdf_root: Path, out_path: Path, source_id: str = "local_pdf") -> int:
    rows = []
    for path in sorted(pdf_root.rglob("*.pdf")):
        metadata = infer_pdf_metadata(path)
        split = choose_split(
            source=source_id,
            competition=metadata["competition"],
            year=metadata["year"],
            requested_split=None,
        )
        item = SourceManifestItem(
            source_id=source_id,
            competition=metadata["competition"],
            year=metadata["year"],
            paper_type=metadata["paper_type"],
            url=None,
            local_path=str(path),
            sha256=sha256_file(path),
            problem_number=metadata["problem_number"],
            split=split,
        )
        rows.append(to_dict(item))
    return write_jsonl(out_path, rows)


def crawl_pdf_manifest(base_url: str, out_path: Path, source_id: str) -> int:
    with urllib.request.urlopen(base_url, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = PdfLinkParser()
    parser.feed(html)

    rows = []
    for href in sorted(set(parser.links)):
        url = urllib.parse.urljoin(base_url, href)
        metadata = infer_pdf_metadata(Path(urllib.parse.urlparse(url).path))
        split = choose_split(
            source=source_id,
            competition=metadata["competition"],
            year=metadata["year"],
            requested_split=None,
        )
        rows.append(
            to_dict(
                SourceManifestItem(
                    source_id=source_id,
                    competition=metadata["competition"],
                    year=metadata["year"],
                    paper_type=metadata["paper_type"],
                    url=url,
                    local_path="",
                    sha256="",
                    problem_number=metadata["problem_number"],
                    split=split,
                )
            )
        )
    return write_jsonl(out_path, rows)


def download_manifest(manifest_path: Path, raw_dir: Path, out_path: Path) -> int:
    rows = []
    for raw in read_jsonl(manifest_path):
        url = raw.get("url")
        if not url:
            rows.append(raw)
            continue
        target = raw_dir / _download_name(raw, url)
        ensure_parent(target)
        urllib.request.urlretrieve(url, target)
        raw["local_path"] = str(target)
        raw["sha256"] = sha256_file(target)
        rows.append(raw)
    return write_jsonl(out_path, rows)


def extract_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    extractor: Extractor = "glm-ocr",
    vlm_work_dir: Path | None = None,
    vlm_model: str = "zai-org/GLM-OCR",
    vlm_prompt: str = "Text Recognition:",
    vlm_max_new_tokens: int = 8192,
    vlm_dpi: int = 180,
    vlm_max_pages: int | None = None,
) -> int:
    count = 0
    if vlm_work_dir is None:
        vlm_work_dir = out_dir / "_vlm_raw"
    for raw in read_jsonl(manifest_path):
        item = SourceManifestItem(**raw)
        if not item.local_path:
            continue
        extracted = _extract_document(
            item,
            extractor=extractor,
            vlm_work_dir=vlm_work_dir,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt,
            vlm_max_new_tokens=vlm_max_new_tokens,
            vlm_dpi=vlm_dpi,
            vlm_max_pages=vlm_max_pages,
        )
        artifact_id = problem_artifact_id(item)
        write_json(out_dir / f"{artifact_id}.json", to_dict(extracted))
        markdown_path = out_dir / f"{artifact_id}.md"
        ensure_parent(markdown_path)
        markdown_path.write_text(extracted.canonical_markdown, encoding="utf-8")
        count += 1
    return count


def _extract_document(
    item: SourceManifestItem,
    *,
    extractor: Extractor,
    vlm_work_dir: Path,
    vlm_model: str,
    vlm_prompt: str,
    vlm_max_new_tokens: int,
    vlm_dpi: int,
    vlm_max_pages: int | None,
) -> ExtractedDocument:
    if extractor == "embedded":
        return extract_embedded_pdf_text(item)
    if extractor == "auto":
        embedded = extract_embedded_pdf_text(item)
        if not any(page.ocr_needed for page in embedded.page_classifications):
            return embedded
    elif extractor != "glm-ocr":
        raise ValueError(f"unknown extractor: {extractor}")
    return extract_with_glm_ocr(
        item,
        vlm_work_dir,
        model_name=vlm_model,
        prompt=vlm_prompt,
        max_new_tokens=vlm_max_new_tokens,
        dpi=vlm_dpi,
        max_pages=vlm_max_pages,
    )


def build_candidates(manifest_path: Path, extracted_dir: Path, out_path: Path) -> int:
    documents: dict[tuple[str, int | None, str | None, str], dict[str, Any]] = {}
    manifests: dict[tuple[str, int | None, str | None, str], SourceManifestItem] = {}
    for raw in read_jsonl(manifest_path):
        item = SourceManifestItem(**raw)
        artifact_id = problem_artifact_id(item)
        extracted_path = extracted_dir / f"{artifact_id}.json"
        if not extracted_path.exists():
            continue
        key = (item.competition, item.year, item.problem_number, item.paper_type)
        documents[key] = read_json(extracted_path)
        manifests[key] = item

    candidates: list[dict[str, Any]] = []
    for key, problem_doc in documents.items():
        competition, year, problem_number, paper_type = key
        if paper_type != "problem":
            continue
        solution_key = (competition, year, problem_number, "solution")
        marking_key = (competition, year, problem_number, "marking_scheme")
        solution_item = manifests.get(solution_key) or manifests.get(marking_key)
        solution_doc = documents.get(solution_key) or documents.get(marking_key)
        problem_item = manifests[key]
        if solution_doc is None or solution_item is None:
            rejected = RejectedItem(
                problem_id=_problem_id(problem_item),
                source=problem_item.source_id,
                rejection_reason="missing_solution",
                detail="No matching solution or marking scheme in manifest",
                item=to_dict(problem_item),
            )
            candidates.append(to_dict(rejected))
            continue

        split = choose_split(
            source=problem_item.source_id,
            competition=competition,
            year=year,
            requested_split=problem_item.split,
        )
        final_item = FinalItem(
            problem_id=_problem_id(problem_item),
            source=problem_item.source_id,
            competition=competition,
            year=year,
            problem_number=problem_number,
            subproblem_id=None,
            problem_text=problem_doc["canonical_markdown"],
            shared_context="",
            question=problem_doc["canonical_markdown"],
            official_solution=solution_doc["canonical_markdown"],
            answers=[],
            requires_diagram=False,
            language=problem_item.language,
            split=split,
            provenance=Provenance(
                pdf_url=problem_item.url,
                page_range=None,
                ocr_engine=problem_doc["extractor"],
                ocr_confidence=None,
                source_hash=problem_item.sha256,
            ),
        )
        candidates.append(to_dict(final_item))
    return write_jsonl(out_path, candidates)


def import_structured_jsonl(input_path: Path, out_path: Path, source_id: str, split: str | None) -> int:
    rows = []
    for index, raw in enumerate(read_jsonl(input_path)):
        if "problem_id" in raw and "provenance" in raw:
            rows.append(raw)
            continue
        question = str(raw.get("question") or raw.get("problem") or raw.get("problem_text") or "")
        official_solution = str(raw.get("official_solution") or raw.get("solution") or raw.get("answer") or "")
        answers = [_answer_from_structured(answer) for answer in raw.get("structured_answers", [])]
        source = str(raw.get("source") or source_id)
        competition = str(raw.get("competition") or source)
        year = raw.get("year")
        chosen_split = choose_split(
            source=source,
            competition=competition,
            year=year,
            requested_split=split,
        )
        digest = hashlib.sha256(f"{source}\n{question}\n{official_solution}".encode("utf-8")).hexdigest()
        item = FinalItem(
            problem_id=str(raw.get("problem_id") or f"{source_id}_{index:08d}_{digest[:12]}"),
            source=source,
            competition=competition,
            year=year,
            problem_number=raw.get("problem_number"),
            subproblem_id=raw.get("subproblem_id"),
            problem_text=question,
            shared_context=str(raw.get("shared_context") or ""),
            question=question,
            official_solution=official_solution,
            answers=answers,
            requires_diagram=bool(raw.get("requires_diagram", False)),
            language=str(raw.get("language") or "en"),
            split=chosen_split,
            provenance=Provenance(
                pdf_url=raw.get("pdf_url"),
                page_range=raw.get("page_range"),
                ocr_engine=raw.get("ocr_engine") or "structured_dataset",
                ocr_confidence=raw.get("ocr_confidence"),
                source_hash=str(raw.get("source_hash") or digest),
            ),
        )
        rows.append(to_dict(item))
    return write_jsonl(out_path, rows)


def filter_candidates(input_path: Path, verified_path: Path, rejected_path: Path) -> tuple[int, int]:
    verified = []
    rejected = []
    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            rejected.append(raw)
            continue
        item = final_item_from_dict(raw)
        try:
            validate_training_policy(
                source=item.source,
                competition=item.competition,
                year=item.year,
                split=item.split,
            )
        except ValueError as exc:
            rejected.append(to_dict(_reject(item, "test_leakage", str(exc))))
            continue
        rejection = admissibility_rejection(item)
        if rejection is not None:
            reason, detail = rejection
            rejected.append(to_dict(_reject(item, reason, detail)))
            continue
        verified.append(to_dict(item))
    return write_jsonl(verified_path, verified), write_jsonl(rejected_path, rejected)


def apply_answer_key(input_path: Path, answer_key_path: Path, out_path: Path) -> int:
    answer_key = list(read_jsonl(answer_key_path))
    rows = []
    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            rows.append(raw)
            continue
        item = final_item_from_dict(raw)
        match = _find_answer_key_match(item, answer_key)
        if match is not None:
            item = FinalItem(
                problem_id=item.problem_id,
                source=item.source,
                competition=item.competition,
                year=item.year,
                problem_number=item.problem_number,
                subproblem_id=match.get("subproblem_id", item.subproblem_id),
                problem_text=item.problem_text,
                shared_context=match.get("shared_context", item.shared_context),
                question=match.get("question", item.question),
                official_solution=item.official_solution,
                answers=[answer_from_dict(answer) for answer in match["answers"]],
                requires_diagram=bool(match.get("requires_diagram", item.requires_diagram)),
                language=item.language,
                split=match.get("split", item.split),
                provenance=item.provenance,
            )
        rows.append(to_dict(item))
    return write_jsonl(out_path, rows)


def dedup_file(input_path: Path, out_path: Path, report_path: Path) -> int:
    items = [final_item_from_dict(raw) for raw in read_jsonl(input_path)]
    kept = deduplicate(items, report_path)
    return write_jsonl(out_path, [to_dict(item) for item in kept])


def export_final(input_path: Path, out_dir: Path) -> dict[str, int]:
    splits = {"train": [], "dev": [], "frozen_test": []}
    for raw in read_jsonl(input_path):
        item = final_item_from_dict(raw)
        splits[item.split].append(to_dict(item))

    counts = {}
    for split, rows in splits.items():
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", rows)
    write_json(
        out_dir / "metadata.json",
        {
            "dataset_version": "physics_rlvr_v1",
            "counts": counts,
            "contract": "English text-only physics problems with deterministic verifiers.",
        },
    )
    return counts


def write_validation_report(input_path: Path, report_path: Path) -> int:
    from .schema import validate_final_item

    ensure_parent(report_path)
    count = 0
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["problem_id", "error"])
        writer.writeheader()
        for raw in read_jsonl(input_path):
            item = final_item_from_dict(raw)
            for error in validate_final_item(item):
                writer.writerow({"problem_id": item.problem_id, "error": error})
                count += 1
    return count


def infer_pdf_metadata(path: Path) -> dict[str, Any]:
    text = " ".join([*path.parts, path.stem]).replace("_", " ")
    year_match = re.search(r"(19|20)\d{2}", text)
    competition_match = re.search(
        r"(?<![A-Za-z0-9])(IPhO|APhO|EuPhO|NBPhO|RMPh|WoPhO|HiPhO|PanPhO|PanMechanics|F=MA)(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    problem_match = re.search(
        r"(?:problem|prob|solution|sol|p|s)[_\-\s]?(\d+[a-z]?)",
        text,
        flags=re.IGNORECASE,
    )
    lowered = text.lower()
    if "mark" in lowered or "scheme" in lowered:
        paper_type = "marking_scheme"
    elif "sol" in lowered or "answer" in lowered:
        paper_type = "solution"
    elif "problem" in lowered or "question" in lowered:
        paper_type = "problem"
    else:
        paper_type = "unknown"
    return {
        "competition": competition_match.group(1) if competition_match else "unknown",
        "year": int(year_match.group(0)) if year_match else None,
        "problem_number": problem_match.group(1) if problem_match else None,
        "paper_type": paper_type,
    }


def _download_name(raw: dict[str, Any], url: str) -> Path:
    parsed_name = Path(urllib.parse.urlparse(url).path).name or "paper.pdf"
    competition = raw.get("competition") or "unknown"
    year = raw.get("year") or "unknown"
    paper_type = raw.get("paper_type") or "unknown"
    return Path(str(competition)) / str(year) / str(paper_type) / parsed_name


def _problem_id(item: SourceManifestItem) -> str:
    return problem_artifact_id(item).replace("__problem__", "__")


def _reject(item: FinalItem, reason: str, detail: str) -> RejectedItem:
    return RejectedItem(
        problem_id=item.problem_id,
        source=item.source,
        rejection_reason=reason,
        detail=detail,
        item=to_dict(item),
    )


def _answer_from_structured(raw: dict[str, Any]) -> Answer:
    answer_type = str(raw.get("answer_type") or "string")
    if answer_type == "numerical":
        answer_type = "numeric"
    verifier = str(raw.get("verifier") or _default_verifier(answer_type))
    return Answer(
        value=str(raw.get("value") or raw.get("final_answer") or ""),
        unit=raw.get("unit"),
        answer_type=answer_type,
        tolerance=raw.get("tolerance"),
        verifier=verifier,
        equivalent_forms=list(raw.get("equivalent_forms", [])),
        subproblem_id=raw.get("subproblem_id"),
    )


def _default_verifier(answer_type: str) -> str:
    if answer_type == "numeric":
        return "numeric"
    if answer_type in {"symbolic", "expression"}:
        return "sympy"
    if answer_type == "multiple_choice":
        return "mcq"
    if answer_type in {"set", "multi_select"}:
        return "set"
    return "string"


def _find_answer_key_match(item: FinalItem, answer_key: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in answer_key:
        if "answers" not in row:
            raise ValueError("answer key row must contain answers")
        if row.get("problem_id") == item.problem_id:
            return row
        if (
            row.get("source") == item.source
            and str(row.get("year")) == str(item.year)
            and str(row.get("problem_number")) == str(item.problem_number)
        ):
            return row
    return None
