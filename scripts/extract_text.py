"""Discover local documents, register provenance and extract text without altering raw files.

Usage: python scripts/extract_text.py [--ocr] [--extensions txt html xml]
All paths are relative to --project-root. Use --in data/raw_txt --out
data/interim/imported_text to register manually supplied text instead of raw sources.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from importlib import metadata as package_metadata
from pathlib import Path

if __package__:
    from . import corpus_common as core
else:
    import corpus_common as core

SUPPORTED = {".txt", ".pdf", ".docx", ".html", ".htm", ".xml", ".epub"}


class NeedsOCR(ValueError):
    pass


def html_text(markup: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(markup, "html.parser")
    for node in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        node.decompose()
    for node in soup.find_all(["p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "br"]):
        node.insert_before("\n\n" if node.name != "br" else "\n")
        if node.name != "br":
            node.insert_after("\n\n")
    return core.normalize(soup.get_text())


def xml_text(data: bytes) -> str:
    from defusedxml import ElementTree
    root = ElementTree.fromstring(data)
    # Prefer an explicit content body to avoid importing bibliographic headers.
    bodies = [node for node in root.iter() if node.tag.split("}")[-1].lower() == "body"]
    body = bodies[0] if bodies else root
    def content(node):
        chunks = [node.text or ""]
        for child in node:
            if child.tag.split("}")[-1].lower() in {"p", "para", "paragraph", "div", "section", "br"}:
                chunks.extend(["\n\n", content(child), "\n\n"])
            else:
                chunks.append(content(child))
            chunks.append(child.tail or "")
        return "".join(chunks)
    return core.normalize(content(body))


def pdf_text(path: Path, args) -> tuple[str, dict]:
    import fitz
    pages, methods, suspect = [], [], []
    engine_version = ""
    with fitz.open(path) as document:
        if document.needs_pass:
            raise ValueError("Encrypted PDF requires a password; no text extracted")
        for number, page in enumerate(document, 1):
            text = page.get_text("text", sort=True)
            stats = core.script_stats(text)
            # A non-Unicode font can yield lots of Latin junk: length alone is insufficient.
            needs_ocr = (stats["letters"] < args.pdf_min_letters
                         or stats["devanagari_ratio"] < args.pdf_min_devanagari
                         or text.count("\ufffd") > 0)
            if needs_ocr:
                suspect.append(number)
                if args.ocr:
                    import pytesseract
                    from PIL import Image
                    if args.tesseract:
                        pytesseract.pytesseract.tesseract_cmd = args.tesseract
                    engine_version = str(pytesseract.get_tesseract_version())
                    pixmap = page.get_pixmap(dpi=args.dpi, alpha=False, colorspace=fitz.csRGB)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    config = f'--psm {args.psm}'
                    if args.tessdata:
                        config += f' --tessdata-dir "{args.tessdata}"'
                    text = pytesseract.image_to_string(image, lang=args.ocr_lang,
                                                      config=config, timeout=args.ocr_timeout)
                    methods.append("tesseract")
                else:
                    methods.append("needs_ocr")
            else:
                methods.append("pdf_text_layer")
            pages.append(core.normalize(text))
    if suspect and not args.ocr:
        raise NeedsOCR(f"Pages needing OCR/script review: {suspect}. Rerun with --ocr; no partial document exported.")
    return "\n\f\n".join(pages), {
        "method": "pdf", "pages": len(pages), "page_methods": methods,
        "ocr_pages": suspect, "ocr_engine": "tesseract" if suspect else "",
        "ocr_version": engine_version, "ocr_languages": args.ocr_lang if suspect else "",
        "dpi": args.dpi if suspect else None, "psm": args.psm if suspect else None,
    }


def extract_file(path: Path, args) -> tuple[str, dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_text(path, args)
    if suffix == ".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        document = Document(path)
        blocks = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                blocks.append(block.text)
            elif isinstance(block, Table):
                blocks.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
        return core.normalize("\n\n".join(blocks)), {"method": "docx"}
    if suffix == ".epub":
        from ebooklib import epub
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        chapters = []
        for identity, linear in book.spine:
            if linear == "no":
                continue
            item = book.get_item_with_id(identity)
            if item is not None and "nav" not in getattr(item, "properties", []):
                decoded, _ = core.decode_bytes(item.get_content())
                chapters.append(html_text(decoded))
        return "\n\n".join(chapters), {"method": "epub_spine"}
    if suffix == ".xml":
        return xml_text(path.read_bytes()), {"method": "xml"}
    text, encoding = core.decode_bytes(path.read_bytes())
    if suffix in {".html", ".htm"}:
        text = html_text(text)
    return text, {"method": suffix.lstrip("."), "input_encoding": encoding}


def allocate_id(path: Path, category: str, used: set[str], reserved: set[str] | None = None) -> str:
    match = re.match(r"(KOK_[A-Z]+_\d+)(?:_|\b)", path.stem)
    if match:
        identity = match[1]
        if identity in used:
            raise ValueError(f"ID {identity} already belongs to another file; register unique IDs in sources.csv")
        return identity
    prefix = f"KOK_{core.CATEGORIES[category]}_"
    numbers = [int(value[len(prefix):]) for value in used | (reserved or set())
               if value.startswith(prefix) and value[len(prefix):].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1:04d}"


def main():
    parser = core.parser(__doc__)
    parser.add_argument("--in", dest="input", default="data/raw")
    parser.add_argument("--out", default="data/raw_txt")
    parser.add_argument("--extensions", nargs="+", default=sorted(SUPPORTED))
    parser.add_argument("--force", action="store_true", help="Re-extract owned outputs; never overwrite unregistered text")
    parser.add_argument("--ocr", action="store_true", help="Enable Tesseract only on suspect PDF pages")
    parser.add_argument("--ocr-lang", default="mar+hin")
    parser.add_argument("--tesseract", help="Path to Tesseract executable")
    parser.add_argument("--tessdata", help="Directory containing traineddata files")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--psm", type=int, default=3)
    parser.add_argument("--ocr-timeout", type=int, default=180)
    parser.add_argument("--pdf-min-letters", type=int, default=20)
    parser.add_argument("--pdf-min-devanagari", type=float, default=.5)
    args = parser.parse_args()
    root = core.setup(args)
    source_dir, output_dir = core.resolve(root, args.input), core.resolve(root, args.out)
    core.protect_raw(root, output_dir, source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {source_dir}")
    if args.dpi < 72 or not 0 <= args.pdf_min_devanagari <= 1:
        parser.error("DPI must be >=72 and PDF Devanagari ratio must be in [0,1]")
    extensions = {"." + suffix.lower().lstrip(".") for suffix in args.extensions}
    if extensions - SUPPORTED:
        parser.error(f"Unsupported extensions: {sorted(extensions - SUPPORTED)}")
    rows = core.load_sources(root)
    by_file = {core.resolve(root, row["source_file"]): row for row in rows if row["source_file"]}
    used = {row["document_id"] for row in rows if row["document_id"]}
    # Reserve filename IDs before assigning any generated numbers: directory
    # sorting may encounter an unnumbered file before KOK_LIT_0001_story.txt.
    reserved = {match[1] for folder in source_dir.iterdir() if folder.is_dir() and folder.name in core.ALIASES
                for candidate in folder.rglob("*") if candidate.is_file() and candidate.suffix.lower() in extensions
                if (match := re.match(r"(KOK_[A-Z]+_\d+)(?:_|\b)", candidate.stem))}
    explicit_owners = {}
    for folder in source_dir.iterdir():
        if not folder.is_dir() or folder.name not in core.ALIASES:
            continue
        for candidate in folder.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in extensions:
                continue
            match = re.match(r"(KOK_[A-Z]+_\d+)(?:_|\b)", candidate.stem)
            if not match or candidate.resolve() in by_file:
                continue
            identity = match[1]
            owner = next((row for row in rows if row["document_id"] == identity), None)
            if identity in explicit_owners or (owner and owner["source_file"]):
                raise ValueError(f"ID {identity} belongs to multiple sources; fix sources.csv/filenames before extraction")
            explicit_owners[identity] = candidate
    previous_path = core.stage_path(root, "extracted")
    previous = {row["document_id"]: row for row in core.read_jsonl(previous_path)} if previous_path.exists() else {}
    exclusions = core.manual_exclusions(root)
    documents, dependencies, failures = [], {"@manual_exclusions": core.manual_exclusions_hash(root)}, 0
    settings = {"script": "extract_text", "ocr": args.ocr, "lang": args.ocr_lang,
                "dpi": args.dpi, "psm": args.psm, "tesseract": args.tesseract,
                "tessdata": args.tessdata, "min_letters": args.pdf_min_letters,
                "min_devanagari": args.pdf_min_devanagari}
    versions = {}
    for package in ("regex", "PyMuPDF", "python-docx", "beautifulsoup4", "defusedxml", "EbookLib", "pytesseract", "Pillow"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    settings.update(python_version=sys.version.split()[0], library_versions=versions)
    extractor_hash = core.text_hash(json.dumps(settings, sort_keys=True)
                                    + core.sha256_file(Path(__file__))
                                    + core.sha256_file(Path(core.__file__)))
    discovered = set()
    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in core.ALIASES:
            logging.info("Outside primary categories, skipping: %s", folder.name)
            continue
        category = core.ALIASES[folder.name]
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                logging.warning("Skipping source link outside category: %s", path)
                continue
            source_file = core.relative(root, path)
            discovered.add(path.resolve())
            source = by_file.get(path.resolve())
            if source is None:
                # Match a metadata row with an explicit filename ID before allocating a new one.
                candidates = [r for r in rows if r["document_id"] and not r["source_file"]
                              and re.match(re.escape(r["document_id"]) + r"(?:_|\b)", path.stem)]
                source = candidates[0] if candidates else {key: "" for key in core.SOURCE_FIELDS}
                if not candidates:
                    rows.append(source)
                source["source_file"] = source_file
            if not source.get("document_id"):
                source["document_id"] = allocate_id(path, category, used, reserved)
                used.add(source["document_id"])
            identity = source["document_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
                raise ValueError(f"Unsafe document ID: {identity}")
            source.update(category=category, source_format=path.suffix.lower().lstrip("."))
            source["review_status"] = source.get("review_status") or "pending"
            source["language_status"] = source.get("language_status") or "unknown"
            source["license_status"] = source.get("license_status") or "unknown"
            digest = core.sha256_file(path)
            if source.get("sha256") and source["sha256"] != digest:
                source["review_status"] = "pending"
                logging.warning("Source changed; resetting review to pending: %s", identity)
            source["sha256"] = digest
            dependencies[source_file] = digest
            destination = output_dir / category / f"{identity}.txt"
            record = {**source, "extracted_file": core.relative(root, destination)}
            manual = exclusions.get(identity) or exclusions.get(source_file)
            old = previous.get(identity, {})
            if manual:
                record.update(processing_status="excluded", exclusion_reason=manual["reason"])
            else:
                try:
                    cached = (not args.force and old.get("sha256") == digest
                              and old.get("extractor_hash") == extractor_hash
                              and old.get("processing_status") == "extracted"
                              and old.get("extracted_file") == record["extracted_file"]
                              and destination.is_file()
                              and core.sha256_file(destination) == old.get("extracted_sha256"))
                    if cached:
                        record.update({key: old[key] for key in ("extraction", "extracted_sha256", "extractor_hash")})
                        logging.info("Cached: %s", identity)
                    else:
                        if destination.exists() and (old.get("extracted_file") != record["extracted_file"]
                                or core.sha256_file(destination) != old.get("extracted_sha256")):
                            raise ValueError(f"Existing extracted text is unregistered or edited; preserve it and choose a new --out: {destination}")
                        text, provenance = extract_file(path, args)
                        if not text.strip():
                            raise ValueError("Extraction produced no text")
                        core.atomic_text(destination, core.normalize(text).strip() + "\n")
                        record.update(extraction=provenance, extracted_sha256=core.sha256_file(destination),
                                      extractor_hash=extractor_hash)
                        logging.info("Extracted: %s (%s)", identity, path.name)
                    record["processing_status"] = "extracted"
                    dependencies[record["extracted_file"]] = record["extracted_sha256"]
                except Exception as exc:
                    failures += 1
                    record.update(processing_status="needs_ocr" if isinstance(exc, NeedsOCR) else "error", error=str(exc))
                    logging.error("%s: %s", source_file, exc)
            source["processing_status"] = record["processing_status"]
            source["extracted_file"] = record["extracted_file"]
            documents.append(record)
    for source in rows:
        path = core.resolve(root, source["source_file"]) if source["source_file"] else None
        if path and path.is_relative_to(source_dir) and path not in discovered and not path.exists():
            source["processing_status"] = "missing_source"
            documents.append({**source, "error": "Registered source is missing; old extraction is not used"})
            logging.warning("Missing source: %s", source["source_file"])
    core.write_csv(root / "metadata/sources.csv", rows, core.SOURCE_FIELDS)
    core.save_stage(root, "extracted", documents, dependencies, {**settings, "extensions": sorted(extensions)})
    logging.info("%d documents registered for this run; %d extraction failures", len(documents), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    core.cli(main)
