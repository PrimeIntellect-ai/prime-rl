from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from .answer_extraction import extract_answer_draft, extract_answer_key
from .dedup import deduplicate
from .extract_pdf import extract_embedded_pdf_text, sha256_file
from .filters import admissibility_rejection
from .gemini_extract import DEFAULT_MODEL as DEFAULT_GEMINI_MODEL
from .gemini_extract import extract_with_gemini
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
from .subproblem_curation import build_rlvr_subproblems

DEFAULT_DATA_DIR = Path("examples/phy_rl/data_pipeline/data")
Extractor = Literal["gemini", "glm-ocr", "embedded", "auto"]
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


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
    request = urllib.request.Request(base_url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
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
        _download_url(url, target)
        raw["local_path"] = str(target)
        raw["sha256"] = sha256_file(target)
        rows.append(raw)
    return write_jsonl(out_path, rows)


def _download_url(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def extract_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    extractor: Extractor = "gemini",
    vlm_work_dir: Path | None = None,
    vlm_model: str = DEFAULT_GEMINI_MODEL,
    vlm_prompt: str = "",
    vlm_max_new_tokens: int = 8192,
    vlm_dpi: int = 180,
    vlm_max_pages: int | None = None,
    jobs: int = 1,
    vlm_devices: list[str] | None = None,
) -> int:
    if vlm_work_dir is None:
        vlm_work_dir = out_dir / "_vlm_raw"
    rows = list(read_jsonl(manifest_path))
    if jobs > 1:
        return _extract_manifest_parallel(
            rows,
            out_dir,
            extractor=extractor,
            vlm_work_dir=vlm_work_dir,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt,
            vlm_max_new_tokens=vlm_max_new_tokens,
            vlm_dpi=vlm_dpi,
            vlm_max_pages=vlm_max_pages,
            jobs=jobs,
            vlm_devices=vlm_devices,
        )
    return _extract_manifest_rows(
        rows,
        out_dir,
        extractor=extractor,
        vlm_work_dir=vlm_work_dir,
        vlm_model=vlm_model,
        vlm_prompt=vlm_prompt,
        vlm_max_new_tokens=vlm_max_new_tokens,
        vlm_dpi=vlm_dpi,
        vlm_max_pages=vlm_max_pages,
        cuda_device=None,
    )


def _extract_manifest_rows(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    extractor: Extractor,
    vlm_work_dir: Path,
    vlm_model: str,
    vlm_prompt: str,
    vlm_max_new_tokens: int,
    vlm_dpi: int,
    vlm_max_pages: int | None,
    cuda_device: str | None,
) -> int:
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    count = 0
    for raw in rows:
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


def _extract_manifest_parallel(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    extractor: Extractor,
    vlm_work_dir: Path,
    vlm_model: str,
    vlm_prompt: str,
    vlm_max_new_tokens: int,
    vlm_dpi: int,
    vlm_max_pages: int | None,
    jobs: int,
    vlm_devices: list[str] | None,
) -> int:
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if vlm_devices is None:
        vlm_devices = [str(index) for index in range(jobs)]
    if not vlm_devices:
        raise ValueError("vlm_devices must contain at least one device when jobs > 1")

    worker_count = min(jobs, len(vlm_devices))
    shards = [rows[index::worker_count] for index in range(worker_count)]
    total = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _extract_manifest_rows,
                shard,
                out_dir,
                extractor=extractor,
                vlm_work_dir=vlm_work_dir,
                vlm_model=vlm_model,
                vlm_prompt=vlm_prompt,
                vlm_max_new_tokens=vlm_max_new_tokens,
                vlm_dpi=vlm_dpi,
                vlm_max_pages=vlm_max_pages,
                cuda_device=vlm_devices[index],
            )
            for index, shard in enumerate(shards)
        ]
        for future in concurrent.futures.as_completed(futures):
            total += future.result()
    return total


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
        extractor = "gemini"
    if extractor == "gemini":
        return extract_with_gemini(
            item,
            vlm_work_dir,
            model_name=vlm_model,
            prompt=vlm_prompt or None,
            max_output_tokens=vlm_max_new_tokens,
            dpi=vlm_dpi,
            max_pages=vlm_max_pages,
        )
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
        problem_text = _curate_document_section(problem_doc["canonical_markdown"], problem_number)
        solution_text = _curate_document_section(solution_doc["canonical_markdown"], problem_number)
        final_item = FinalItem(
            problem_id=_problem_id(problem_item),
            source=problem_item.source_id,
            competition=competition,
            year=year,
            problem_number=problem_number,
            subproblem_id=None,
            problem_text=problem_text,
            shared_context="",
            question=problem_text,
            official_solution=solution_text,
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


def extract_answer_key_file(input_path: Path, out_path: Path, review_path: Path, min_score: int = 4) -> tuple[int, int]:
    return extract_answer_key(input_path, out_path, review_path, min_score=min_score)


def extract_answer_draft_file(input_path: Path, out_path: Path, audit_path: Path) -> tuple[int, int]:
    return extract_answer_draft(input_path, out_path, audit_path)


def build_rlvr_subproblems_file(
    input_path: Path,
    verified_path: Path,
    review_path: Path,
    rejected_path: Path,
    min_score: int = 5,
    judge_model: str | None = None,
) -> tuple[int, int, int]:
    return build_rlvr_subproblems(
        input_path,
        verified_path,
        review_path,
        rejected_path,
        min_score=min_score,
        judge_model=judge_model,
    )


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
        r"(?:problem|prob|question|solution|sol|[qps])[_\-\s]?(\d+[a-z]?)",
        text,
        flags=re.IGNORECASE,
    )
    lowered = text.lower()
    if "mark" in lowered or "scheme" in lowered:
        paper_type = "marking_scheme"
    elif re.search(r"(?:^|[_\-\s])s\d+[a-z]?(?:$|[_\-\s])", lowered) or "sol" in lowered or "answer" in lowered:
        paper_type = "solution"
    elif re.search(r"(?:^|[_\-\s])q\d+[a-z]?(?:$|[_\-\s])", lowered) or "problem" in lowered or "question" in lowered:
        paper_type = "problem"
    else:
        paper_type = "unknown"
    return {
        "competition": competition_match.group(1) if competition_match else "unknown",
        "year": int(year_match.group(0)) if year_match else None,
        "problem_number": problem_match.group(1) if problem_match else None,
        "paper_type": paper_type,
    }


def _curate_document_section(markdown: str, problem_number: str | None) -> str:
    if not problem_number:
        return markdown
    start_match = _find_problem_section_start(markdown, problem_number)
    if start_match is None:
        return markdown
    next_match = _find_next_problem_section_start(markdown, start_match.end())
    end = next_match.start() if next_match else len(markdown)
    return markdown[start_match.start() : end].strip() + "\n"


def _find_problem_section_start(markdown: str, problem_number: str) -> re.Match[str] | None:
    escaped = re.escape(problem_number)
    patterns = [
        rf"(?mi)^\s*(?:\*\*)?(?:problem|question)\s+{escaped}\b",
        rf"(?mi)^\s*(?:\*\*)?(?:solution\s+of\s+)?(?:problem|question|task)\s+{escaped}\b",
    ]
    matches = [match for pattern in patterns if (match := re.search(pattern, markdown))]
    return min(matches, key=lambda match: match.start()) if matches else None


def _find_next_problem_section_start(markdown: str, start: int) -> re.Match[str] | None:
    pattern = r"(?mi)^\s*(?:\*\*)?(?:problem|question|solution\s+of\s+(?:problem|question|task))\s+\d+[a-z]?\b"
    return re.compile(pattern).search(markdown, pos=start)


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
