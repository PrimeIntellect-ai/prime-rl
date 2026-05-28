from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_DATA_DIR,
    apply_answer_key,
    build_rlvr_subproblems_file,
    build_candidates,
    crawl_pdf_manifest,
    dedup_file,
    download_manifest,
    export_final,
    extract_answer_draft_file,
    extract_answer_key_file,
    extract_manifest,
    filter_candidates,
    import_structured_jsonl,
    init_layout,
    manifest_from_local_pdfs,
    write_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics RLVR data curation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.set_defaults(run=_run_init)

    p = sub.add_parser("manifest-local")
    p.add_argument("--pdf-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-id", default="local_pdf")
    p.set_defaults(run=lambda a: _count("manifest rows", manifest_from_local_pdfs(a.pdf_root, a.out, a.source_id)))

    p = sub.add_parser("crawl")
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-id", default="ipho_olimpicos")
    p.set_defaults(run=lambda a: _count("crawled PDF rows", crawl_pdf_manifest(a.base_url, a.out, a.source_id)))

    p = sub.add_parser("download")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=lambda a: _count("downloaded manifest rows", download_manifest(a.manifest, a.raw_dir, a.out)))

    p = sub.add_parser("extract")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--extractor", choices=["gemini", "glm-ocr", "embedded", "auto"], default="gemini")
    p.add_argument("--vlm-work-dir", type=Path)
    p.add_argument("--vlm-model", default="gemini-3.1-flash-lite")
    p.add_argument("--vlm-prompt", default="")
    p.add_argument("--vlm-max-new-tokens", type=int, default=8192)
    p.add_argument("--vlm-dpi", type=int, default=180)
    p.add_argument("--vlm-max-pages", type=int)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--vlm-devices", help="Comma-separated CUDA device ids for parallel GLM-OCR workers")
    p.set_defaults(run=_run_extract)

    p = sub.add_parser("build-candidates")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--extracted-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=lambda a: _count("candidates", build_candidates(a.manifest, a.extracted_dir, a.out)))

    p = sub.add_parser("import-structured")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-id", default="structured_physics")
    p.add_argument("--split", choices=["train", "dev", "frozen_test"])
    p.set_defaults(run=lambda a: _count("structured candidates", import_structured_jsonl(a.input, a.out, a.source_id, a.split)))

    p = sub.add_parser("filter-candidates")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--verified", type=Path, required=True)
    p.add_argument("--rejected", type=Path, required=True)
    p.set_defaults(run=_run_filter)

    p = sub.add_parser("apply-answer-key")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--answer-key", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=lambda a: _count("answer-keyed candidates", apply_answer_key(a.input, a.answer_key, a.out)))

    p = sub.add_parser("extract-answer-key")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--min-score", type=int, default=6)
    p.set_defaults(run=_run_extract_answer_key)

    p = sub.add_parser("extract-answer-draft")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.set_defaults(run=_run_extract_answer_draft)

    p = sub.add_parser("build-rlvr-subproblems")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--verified", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--rejected", type=Path, required=True)
    p.add_argument("--min-score", type=int, default=3)
    p.add_argument("--judge-model", default=None)
    p.set_defaults(run=_run_build_rlvr_subproblems)

    p = sub.add_parser("dedup")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.set_defaults(run=lambda a: _count("deduplicated items", dedup_file(a.input, a.out, a.report)))

    p = sub.add_parser("export-final")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=lambda a: print(f"exported {export_final(a.input, a.out_dir)}"))

    p = sub.add_parser("validate-final")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.set_defaults(run=lambda a: _count("validation errors", write_validation_report(a.input, a.report)))

    args = parser.parse_args()
    args.run(args)


def _run_extract(args: argparse.Namespace) -> None:
    count = extract_manifest(
        args.manifest,
        args.out_dir,
        extractor=args.extractor,
        vlm_work_dir=args.vlm_work_dir,
        vlm_model=args.vlm_model,
        vlm_prompt=args.vlm_prompt,
        vlm_max_new_tokens=args.vlm_max_new_tokens,
        vlm_dpi=args.vlm_dpi,
        vlm_max_pages=args.vlm_max_pages,
        jobs=args.jobs,
        vlm_devices=_split_csv(args.vlm_devices),
    )
    _count("documents", count)


def _run_init(args: argparse.Namespace) -> None:
    init_layout(args.data_dir)
    print(f"initialized {args.data_dir}")


def _run_filter(args: argparse.Namespace) -> None:
    verified, rejected = filter_candidates(args.input, args.verified, args.rejected)
    print(f"verified {verified}; rejected {rejected}")


def _run_extract_answer_key(args: argparse.Namespace) -> None:
    accepted, review = extract_answer_key_file(args.input, args.out, args.review, args.min_score)
    print(f"wrote {accepted} answer-key rows; wrote {review} review rows")


def _run_extract_answer_draft(args: argparse.Namespace) -> None:
    rows, audit = extract_answer_draft_file(args.input, args.out, args.audit)
    print(f"wrote {rows} draft answer-key rows; wrote {audit} audit rows")


def _run_build_rlvr_subproblems(args: argparse.Namespace) -> None:
    verified, review, rejected = build_rlvr_subproblems_file(
        args.input,
        args.verified,
        args.review,
        args.rejected,
        args.min_score,
        judge_model=args.judge_model,
    )
    print(f"verified {verified}; review {review}; rejected {rejected}")


def _count(name: str, count: int) -> None:
    print(f"wrote {count} {name}")


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]
