"""Fetch media files for the vision-graft SFT blend.

Reads datasets/nemotron_vl_sft/media_manifest.json (written by
prepare_vlm_sft_data.py) and downloads only the referenced files,
one source kind at a time. All commands are resumable: existing
target files are skipped.

Run from the repo root, e.g.:
    uv run python scripts/fetch_vlm_sft_media.py openimages --workers 32
    uv run python scripts/fetch_vlm_sft_media.py cc3m
    uv run python scripts/fetch_vlm_sft_media.py gqa
    uv run python scripts/fetch_vlm_sft_media.py chartqa
    uv run python scripts/fetch_vlm_sft_media.py docvqa
    uv run --with pymupdf python scripts/fetch_vlm_sft_media.py ccpdf --workers 16

Sources:
    openimages  per-image GET from the official OpenImages S3 bucket
    cc3m        streamed extraction from pixparse/cc3m-wds tar shards
    gqa         extraction from the Stanford images.zip (downloaded separately)
    chartqa     copy from a sparse checkout of the ChartQA GitHub repo
    docvqa      streamed from HuggingFaceM4/DocumentVQA parquet (embedded images)
    ccpdf       PDFs range-read from Digital Corpora zips, pages rendered via pymupdf
"""

import argparse
import io
import json
import shutil
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent.parent
MANIFEST = REPO_ROOT / "datasets" / "nemotron_vl_sft" / "media_manifest.json"  # overridable via --manifest
MEDIA_SRC = REPO_ROOT / "datasets" / "media_src"

OPENIMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/train/{name}"
CC3M_SHARD_URL = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/{shard}"
CCPDF_ZIP_URL = (
    "https://digitalcorpora.s3.amazonaws.com/corpora/files/"
    "CC-MAIN-2021-31-PDF-UNTRUNCATED/zipfiles/{prefix}000-{prefix}999/{zipname}.zip"
)


_manifest_path = MANIFEST


def load_kind(kind: str) -> dict[Path, dict]:
    """target path (absolute) -> manifest entry, for one source kind, minus existing files."""
    with open(_manifest_path) as f:
        manifest = json.load(f)
    todo, done = {}, 0
    for target, entry in manifest.items():
        if target.split("/")[3] != kind:
            continue
        abs_target = REPO_ROOT / target
        if abs_target.exists():
            done += 1
        else:
            todo[abs_target] = entry
    print(f"[{kind}] {len(todo)} to fetch, {done} already present")
    return todo


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def http_fetch_many(url_by_target: dict[Path, str], workers: int) -> None:
    session_local = {}

    def fetch(target: Path, url: str) -> str | None:
        import threading

        sess = session_local.setdefault(threading.get_ident(), requests.Session())
        resp = sess.get(url, timeout=60)
        if resp.status_code == 404:
            return f"404 {url}"
        resp.raise_for_status()
        atomic_write(target, resp.content)
        return None

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, t, u): t for t, u in url_by_target.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            err = fut.result()
            if err:
                errors.append(err)
            if i % 2000 == 0:
                print(f"  {i}/{len(futures)} ({len(errors)} errors)", flush=True)
    print(f"  done: {len(url_by_target) - len(errors)} fetched, {len(errors)} errors")
    for e in errors[:20]:
        print(f"  ERROR {e}")


def cmd_openimages(args) -> None:
    todo = load_kind("openimages")
    urls = {t: OPENIMAGES_URL.format(name=t.name) for t in todo}
    http_fetch_many(urls, args.workers)


def cmd_chartqa(args) -> None:
    dataset_root = MEDIA_SRC / "chartqa_repo" / "ChartQA Dataset"
    todo = load_kind("chartqa")
    missing = 0
    for target, entry in todo.items():
        # raw_path keeps its split subdir (train/png/x, val/png/x, test/png/x);
        # fall back to the train dir for legacy flat targets.
        candidates = [dataset_root / entry["raw_path"], dataset_root / "train" / "png" / target.name]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            missing += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    print(f"  done: {len(todo) - missing} copied, {missing} missing from repo checkout")


def cmd_gqa(args) -> None:
    zip_path = MEDIA_SRC / "gqa_images.zip"
    todo = load_kind("gqa")
    missing = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for target in todo:
            member = f"images/{target.name}"
            if member not in names:
                missing += 1
                continue
            atomic_write(target, zf.read(member))
    print(f"  done: {len(todo) - missing} extracted, {missing} missing from zip")


def cmd_cc3m(args) -> None:
    todo = load_kind("cc3m")
    # wds key "000000320" <-> raw ref image_000000320.jpg (target keeps the raw name)
    target_by_key = {t.name.removeprefix("image_").removesuffix(".jpg"): t for t in todo}
    if not target_by_key:
        return
    from huggingface_hub import HfApi

    shards = sorted(
        f
        for f in HfApi().list_repo_files("pixparse/cc3m-wds", repo_type="dataset")
        if f.endswith(".tar") and "train" in f
    )
    remaining = dict(target_by_key)

    def scan_shard(shard: str) -> int:
        if not remaining:
            return 0
        found = 0
        with requests.get(CC3M_SHARD_URL.format(shard=shard), stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with tarfile.open(fileobj=resp.raw, mode="r|") as tf:
                for member in tf:
                    if not member.name.endswith(".jpg"):
                        continue
                    key = member.name.removesuffix(".jpg")
                    target = remaining.get(key)
                    if target is not None:
                        atomic_write(target, tf.extractfile(member).read())
                        remaining.pop(key, None)
                        found += 1
        return found

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_shard, s): s for s in shards}
        done_shards = 0
        for fut in as_completed(futures):
            fut.result()
            done_shards += 1
            if done_shards % 25 == 0:
                print(f"  shards {done_shards}/{len(shards)}, {len(remaining)} images remaining", flush=True)
    print(f"  done: {len(target_by_key) - len(remaining)} extracted, {len(remaining)} not found")


def cmd_docvqa(args) -> None:
    todo = load_kind("docvqa")
    if not todo:
        return
    wanted = {t.name: t for t in todo}
    from datasets import load_dataset

    found = 0
    for split in ("train", "validation"):
        if not wanted:
            break
        ds = load_dataset("HuggingFaceM4/DocumentVQA", split=split, streaming=True)
        for row in ds:
            name = f"{row['ucsf_document_id']}_{row['ucsf_document_page_no']}.png"
            target = wanted.pop(name, None)
            if target is None:
                continue
            buf = io.BytesIO()
            row["image"].save(buf, format="PNG")
            atomic_write(target, buf.getvalue())
            found += 1
            if found % 200 == 0:
                print(f"  {found} found, {len(wanted)} remaining", flush=True)
    print(f"  done: {found} extracted, {len(wanted)} not found")


def _render_ccpdf_page(task: tuple[str, int, str]) -> str | None:
    """Render one PDF page to PNG. Module-level so ProcessPoolExecutor can pickle it."""
    import pymupdf
    from PIL import Image

    pdf_file, page_no, target = task
    target = Path(target)
    try:
        doc = pymupdf.Document(pdf_file)
        page = doc.load_page(page_no)
        zoom = min(1024 / page.rect.width, 1280 / page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        atomic_write(target, buf.getvalue())
        return None
    except Exception as e:  # noqa: BLE001 — per-file fault tolerance, reported by caller
        return f"{pdf_file} p{page_no}: {e}"


def cmd_ccpdf(args) -> None:
    sys.path.insert(0, str(MEDIA_SRC / "nvd2" / "scripts"))
    from ccpdf_download import KeepAliveZipReader

    todo = load_kind("ccpdf")
    pdf_dir = MEDIA_SRC / "ccpdf_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # target .../media/ccpdf/<batch>/<dirs...>/<pdfid>_pdf/<page>.png; raw_path <dirs...>/<pdfid>_pdf/<page>.png
    render_tasks = []  # (pdf_id, page, target)
    for target, entry in todo.items():
        pdf_id = target.parent.name.removesuffix("_pdf")
        page = int(target.stem)
        render_tasks.append((pdf_id, page, target))
    pdf_ids = sorted({pdf_id for pdf_id, _, _ in render_tasks})
    to_download = [p for p in pdf_ids if not (pdf_dir / f"{p}.pdf").exists()]
    print(f"  {len(pdf_ids)} unique PDFs ({len(to_download)} to download), {len(render_tasks)} pages to render")

    zip_reader = KeepAliveZipReader()

    def download(pdf_id: str) -> str | None:
        url = CCPDF_ZIP_URL.format(prefix=pdf_id[0], zipname=pdf_id[:4])
        try:
            zip_reader.download_file(url, f"{pdf_id}.pdf", pdf_dir / f"{pdf_id}.pdf")
            return None
        except Exception as e:  # noqa: BLE001 — per-file fault tolerance, reported below
            return f"{pdf_id}: {e}"

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, p) for p in to_download]
        for i, fut in enumerate(as_completed(futures), 1):
            err = fut.result()
            if err:
                errors.append(err)
            if i % 500 == 0:
                print(f"  pdfs {i}/{len(to_download)} ({len(errors)} errors)", flush=True)
    print(f"  pdf download done ({len(errors)} errors)")
    for e in errors[:10]:
        print(f"  ERROR {e}")

    pickle_tasks = []
    render_errors = []
    for pdf_id, page_no, target in render_tasks:
        pdf_file = pdf_dir / f"{pdf_id}.pdf"
        if pdf_file.exists():
            pickle_tasks.append((str(pdf_file), page_no, str(target)))
        else:
            render_errors.append(f"{pdf_id}: pdf missing")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, err in enumerate(pool.map(_render_ccpdf_page, pickle_tasks, chunksize=16), 1):
            if err:
                render_errors.append(err)
            if i % 5000 == 0:
                print(f"  pages {i}/{len(render_tasks)} ({len(render_errors)} errors)", flush=True)
    print(f"  done: {len(render_tasks) - len(render_errors)} pages rendered, {len(render_errors)} errors")
    for e in render_errors[:10]:
        print(f"  ERROR {e}")


def _extract_by_suffix(archive_names, todo: dict, read_member) -> None:
    """Match archive members to targets by path suffix and write them out.

    todo: abs target -> manifest entry (raw_path is the suffix to look for).
    """
    by_suffix = {}
    for abs_target, entry in todo.items():
        by_suffix[entry["raw_path"]] = abs_target
    written = 0
    for name in archive_names:
        for suffix, abs_target in list(by_suffix.items()):
            if name == suffix or name.endswith("/" + suffix):
                atomic_write(abs_target, read_member(name))
                by_suffix.pop(suffix)
                written += 1
                if written % 5000 == 0:
                    print(f"  {written} extracted", flush=True)
    print(f"  extracted {written}; unmatched {len(by_suffix)}")
    if by_suffix:
        print("  e.g. missing:", list(by_suffix)[:5])


def cmd_flickr30k(args) -> None:
    """nlphuji/flickr30k ships a single flickr30k-images.zip."""
    from huggingface_hub import hf_hub_download

    todo = load_kind("flickr30k")
    if not todo:
        return
    zip_path = hf_hub_download("nlphuji/flickr30k", "flickr30k-images.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        _extract_by_suffix(zf.namelist(), todo, lambda n: zf.read(n))


def cmd_coco(args) -> None:
    """COCO 2017 images (a-okvqa references them by numeric id)."""
    todo = load_kind("coco")
    if not todo:
        return
    for split in ("train2017", "val2017"):
        if not todo:
            break
        zip_path = MEDIA_SRC / f"{split}.zip"
        if not zip_path.exists():
            print(f"  downloading {split}.zip ...", flush=True)
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(f"http://images.cocodataset.org/zips/{split}.zip", stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = zip_path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=1 << 22)
                tmp.rename(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            still = {}
            for abs_target, entry in todo.items():
                member = f"{split}/{entry['raw_path']}"
                if member in names:
                    atomic_write(abs_target, zf.read(member))
                else:
                    still[abs_target] = entry
            print(f"  {split}: {len(todo) - len(still)} extracted, {len(still)} remaining")
            todo = still
    if todo:
        print(f"  !! {len(todo)} coco files not found in train2017/val2017")


def cmd_mulberry(args) -> None:
    """HuanjinYao/Mulberry-SFT ships one mulberry_images.tar with the nested source tree."""
    from huggingface_hub import hf_hub_download

    todo = load_kind("mulberry")
    if not todo:
        return
    tar_path = hf_hub_download("HuanjinYao/Mulberry-SFT", "mulberry_images.tar", repo_type="dataset")
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        _extract_by_suffix(names, todo, lambda n: tf.extractfile(n).read())


def _cmd_nvidia_shards(kind: str, subset: str) -> None:
    """Media hosted on the Nemotron-Image-Training-v3 repo as webdataset tar shards."""
    from huggingface_hub import HfApi, hf_hub_download

    todo = load_kind(kind)
    if not todo:
        return
    files = HfApi().list_repo_files("nvidia/Nemotron-Image-Training-v3", repo_type="dataset")
    shards = [f for f in files if f.startswith(f"{subset}/media/shard_") and f.endswith(".tar")]
    for shard in shards:
        if not todo:
            break
        tar_path = hf_hub_download("nvidia/Nemotron-Image-Training-v3", shard, repo_type="dataset")
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
            before = len(todo)
            _extract_by_suffix(names, todo, lambda n: tf.extractfile(n).read())
            done_targets = [t for t in todo if t.exists()]
            for t in done_targets:
                todo.pop(t, None)
            print(f"  {shard}: {before - len(todo)} matched, {len(todo)} remaining")


def cmd_clevr(args) -> None:
    _cmd_nvidia_shards("clevr", "clevr_1")


def cmd_plotqa(args) -> None:
    _cmd_nvidia_shards("plotqa", "plotqa_1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=[
            "openimages",
            "cc3m",
            "gqa",
            "chartqa",
            "docvqa",
            "ccpdf",
            "flickr30k",
            "coco",
            "mulberry",
            "clevr",
            "plotqa",
        ],
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    global _manifest_path
    _manifest_path = args.manifest
    globals()[f"cmd_{args.kind}"](args)


if __name__ == "__main__":
    main()
