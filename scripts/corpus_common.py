"""Shared corpus schemas, Unicode policies and reproducible stage storage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import tempfile
import unicodedata
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "01_literature": "LIT", "02_news_media": "NEWS",
    "03_education_science": "EDU", "04_culture_history": "CULT",
    "05_public_information": "GOV", "06_web_articles": "WEB",
    "07_oral_transcripts": "ORAL",
}
ALIASES = {**{key: key for key in CATEGORIES},
           "02_news_and_media": "02_news_media", "06_webarticles": "06_web_articles"}
SOURCE_FIELDS = [
    "document_id", "category", "subcategory", "title", "author_publisher",
    "source_url", "source_format", "script", "language_status", "license_status",
    "collected_by", "collection_date", "sha256", "processing_status",
    "review_status", "notes", "source_file", "extracted_file",
]
EXCLUSION_FIELDS = ["file", "reason", "kept_instead", "document_id", "origin"]
WORD_RE = regex.compile(
    r"\p{L}[\p{L}\p{M}\u200c\u200d]*(?:['’\-]\p{L}[\p{L}\p{M}\u200c\u200d]*)*"
    r"|\p{N}+(?:[.,]\p{N}+)*"
)
DEVA_LETTER = regex.compile(r"[\p{Script=Devanagari}&&\p{L}]", regex.V1)
LATIN_LETTER = regex.compile(r"[\p{Script=Latin}&&\p{L}]", regex.V1)


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--project-root", type=Path, default=ROOT,
                        help="Project containing data/, metadata/, results/ (default: repository)")
    result.add_argument("--verbose", action="store_true")
    return result


def setup(args) -> Path:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    return args.project_root.resolve()


def cli(main):
    try:
        raise SystemExit(main() or 0)
    except (ValueError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def protect_raw(root: Path, destination: Path, input_dir: Path | None = None):
    """Reject writes inside raw data, including through symlinks."""
    target = destination.resolve()
    for protected in (root / "data/raw", input_dir):
        if protected is not None and target.is_relative_to(protected.resolve()):
            raise ValueError(f"Refusing to write inside source directory: {target}")


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n"))


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Strict decoding; no silent replacement or Latin-1 fallback."""
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encoding = "utf-32"
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif data.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        # BOM-less UTF-16 includes NULs in ASCII or repeated 09 bytes in Devanagari.
        encoding = "utf-8"
        if len(data) >= 4 and len(data) % 2 == 0:
            even, odd = data[::2], data[1::2]
            high_even = sum(b in (0, 9) for b in even) / len(even)
            high_odd = sum(b in (0, 9) for b in odd) / len(odd)
            if high_odd > .6 and high_odd > high_even + .3:
                encoding = "utf-16-le"
            elif high_even > .6 and high_even > high_odd + .3:
                encoding = "utf-16-be"
    text = data.decode(encoding, errors="strict")
    if "\x00" in text:
        raise ValueError("NUL characters remain: unsupported encoding or binary file")
    return normalize(text), encoding


def tokens(text: str, numbers: str = "keep") -> list[str]:
    words = WORD_RE.findall(normalize(text))
    if numbers == "normalize":
        return ["<NUM>" if word[0].isnumeric() else word for word in words]
    if numbers == "exclude":
        return [word for word in words if not word[0].isnumeric()]
    return words


def script_stats(text: str) -> dict:
    count = len(regex.findall(r"\p{L}", text))
    deva = len(DEVA_LETTER.findall(text))
    latin = len(LATIN_LETTER.findall(text))
    return {"letters": count, "devanagari_ratio": deva / count if count else 0.0,
            "latin_ratio": latin / count if count else 0.0,
            "other_script_ratio": (count - deva - latin) / count if count else 0.0}


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows, fields: list[str], delimiter=","):
    rows = list(rows)
    fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def load_sources(root: Path) -> list[dict]:
    rows = read_csv(root / "metadata/sources.csv")
    ids, paths = set(), set()
    for row in rows:
        row["document_id"] = row.get("document_id") or row.get("doc_id", "")
        row["source_file"] = (row.get("source_file") or row.get("file", "")).replace("\\", "/")
        row["category"] = ALIASES.get(row.get("category", ""), row.get("category", ""))
        identity = row["document_id"]
        path = row["source_file"]
        if identity and identity in ids:
            raise ValueError(f"Duplicate document_id in sources.csv: {identity}")
        if path and path.casefold() in paths:
            raise ValueError(f"Duplicate source_file in sources.csv: {path}")
        if identity:
            ids.add(identity)
        if path:
            paths.add(path.casefold())
    return rows


def stage_path(root: Path, name: str) -> Path:
    return root / "data/processed/stages" / f"{name}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl_text(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def load_stage(root: Path, name: str) -> tuple[list[dict], dict]:
    path = stage_path(root, name)
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() or not meta_path.exists():
        raise ValueError(f"Missing {name} stage. Run the preceding script first: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dependencies = {**meta["dependencies"], relative(root, path): meta["output_sha256"]}
    for file, expected in dependencies.items():
        if file == "@manual_exclusions":
            if manual_exclusions_hash(root) != expected:
                raise ValueError(f"Stale {name} stage: manual exclusions changed. Rerun extraction and subsequent stages.")
            continue
        actual = resolve(root, file)
        if not actual.is_file() or sha256_file(actual) != expected:
            raise ValueError(f"Stale {name} stage: {file} changed. Rerun from the affected stage.")
    return read_jsonl(path), dependencies


def save_stage(root: Path, name: str, rows, dependencies: dict, options: dict):
    path = stage_path(root, name)
    dependencies = dict(dependencies)
    for code in (Path(__file__), Path(__file__).with_name(f"{options['script']}.py")):
        dependencies[relative(root, code)] = sha256_file(code)
    atomic_text(path, jsonl_text(rows))
    atomic_text(path.with_suffix(".meta.json"), json.dumps({
        "schema_version": 1, "stage": name, "options": options,
        "dependencies": dependencies, "output_sha256": sha256_file(path),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def refresh_exclusions(root: Path, origin: str, rows):
    path = root / "metadata/excluded_documents.csv"
    preserved = [r for r in read_csv(path) if r.get("origin", "") != origin]
    write_csv(path, [*preserved, *rows], EXCLUSION_FIELDS)


def manual_exclusions(root: Path) -> dict[str, dict]:
    result = {}
    for row in read_csv(root / "metadata/excluded_documents.csv"):
        if row.get("origin", "") not in ("", "manual"):
            continue
        for key in (row.get("document_id"), row.get("file", "").replace("\\", "/")):
            if key:
                result[key] = row
    return result


def manual_exclusions_hash(root: Path) -> str:
    # Generated CSV columns may be added by later stages. Ignore empty columns.
    rows = {key: {k: v for k, v in row.items() if v} for key, row in manual_exclusions(root).items()}
    return text_hash(json.dumps(rows, sort_keys=True, ensure_ascii=False))


def exclusion(row: dict, reason: str, origin: str, kept="") -> dict:
    return {"file": row["source_file"], "document_id": row["document_id"],
            "reason": reason, "kept_instead": kept, "origin": origin}
