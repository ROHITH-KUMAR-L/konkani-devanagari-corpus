"""Collect every source into the iteration-0 document tree (raw text, UTF-8, LF).

Iteration 0 is the corpus *before* cleaning: only the encoding is fixed
(UTF-8, BOM removed, CRLF -> LF). Raw files under data/raw are never modified.

Sources gathered:
  data/raw/01_literature/*.txt, data/raw/04_culture_history/*.txt   book OCR text
  data/interim/03_education_science/*.txt                            textbook OCR (fetch_epaper.py ocr)
  data/interim/02_news_and_media/epaper/<TITLE>/<date>_<id>.txt       e-paper OCR (fetch_epaper.py)
  data/interim/08_ldcil_gold_sample/*.txt                             LDC-IL gold sample (extract_ldcil.py)
  data/raw/06_web_articles/wikipedia/*.xml.bz2                        Konkani Wikipedia dump

Files listed in metadata/excluded_documents.csv are skipped. A manifest of all
documents is written to metadata/documents.csv.

Usage:
    python scripts/extract_text.py --out data/interim/iter0
"""
import argparse
import bz2
import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from corpus_utils import devanagari_ratio, tokenize, write_doc

ROOT = Path(__file__).resolve().parent.parent

BOOK_DIRS = [("data/raw/01_literature", "01_literature", "lit"),
             ("data/raw/04_culture_history", "04_culture_history", "cult")]
OCR_DIRS = [("data/interim/03_education_science", "03_education_science", "edu")]
EPAPER_DIR = "data/interim/02_news_and_media/epaper"
LDCIL_DIR = "data/interim/08_ldcil_gold_sample"
WIKI_DIR = "data/raw/06_web_articles/wikipedia"
WIKI_SHARDS = 20          # wiki articles are grouped into shards so splits stay document-level
WIKI_MIN_DEVA = 0.7       # skip Romi (Roman-script) and other non-Devanagari articles


def slug(name):
    s = re.sub(r"[^0-9A-Za-z]+", "_", name.strip()).strip("_").lower()
    return s or "doc"


def fix_encoding(text):
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def load_excluded():
    path = ROOT / "metadata/excluded_documents.csv"
    with open(path, encoding="utf-8") as f:
        return {(ROOT / r["file"]).resolve() for r in csv.DictReader(f)}


# --------------------------------------------------------------- wikipedia

WIKI_DROP_SECTIONS = re.compile(r"^(संदर्भ|हेय पळयात|भायले दुवे|बाह्य दुवे|References|External links)\s*$")


def wiki_articles(dump):
    """Yield (title, plain_text) for main-namespace, non-redirect articles."""
    import mwparserfromhell  # only needed for the Wikipedia source

    with bz2.open(dump, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if not el.tag.endswith("}page"):
                continue
            ns = el.find("{*}ns")
            title = el.find("{*}title").text or ""
            text_el = el.find("{*}revision/{*}text")
            raw = text_el.text if text_el is not None else None
            redirect = el.find("{*}redirect") is not None
            ns = ns.text if ns is not None else None
            el.clear()
            if ns != "0" or not raw or redirect:
                continue
            if raw.lstrip().upper().startswith(("#REDIRECT", "#पुनर्निर्देशन")):
                continue
            code = mwparserfromhell.parse(raw)
            for node in code.filter_templates(recursive=False) + code.filter_tags(
                    matches=lambda t: str(t.tag) in ("ref", "table", "gallery", "math")):
                try:
                    code.remove(node)
                except ValueError:
                    pass
            lines, keep = [], True
            for line in code.strip_code(normalize=True, collapse=True).split("\n"):
                line = line.strip()
                head = re.fullmatch(r"=+\s*(.*?)\s*=+", line)
                if head:
                    keep = not WIKI_DROP_SECTIONS.match(head.group(1))
                    continue
                if keep and line and not line.startswith(("[[", "Category:", "वर्ग:")):
                    lines.append(line)
            body = "\n".join(lines)
            if body and devanagari_ratio(body)[0] >= WIKI_MIN_DEVA:
                yield title, body


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/interim/iter0")
    ap.add_argument("--manifest", default="metadata/documents.csv")
    args = ap.parse_args()
    out = ROOT / args.out
    excluded = load_excluded()
    rows = []

    def add(cat, doc_id, text, source_type, source_path, title):
        text = unicodedata.normalize("NFC", fix_encoding(text)) if source_type == "wikipedia" else fix_encoding(text)
        write_doc(out / cat / f"{doc_id}.txt", text)
        rows.append({"doc_id": doc_id, "category": cat, "source_type": source_type,
                     "source_path": source_path, "title": title, "tokens": len(tokenize(text))})

    for rel, cat, prefix in BOOK_DIRS:
        for p in sorted((ROOT / rel).glob("*.txt")):
            if p.resolve() in excluded:
                continue
            add(cat, f"{prefix}_{slug(p.stem)}", p.read_text(encoding="utf-8"), "book_ocr",
                p.relative_to(ROOT).as_posix(), p.stem.strip())

    for rel, cat, prefix in OCR_DIRS:
        for p in sorted((ROOT / rel).glob("*.txt")):
            add(cat, f"{prefix}_{slug(p.stem)}", p.read_text(encoding="utf-8"), "pdf_ocr",
                p.relative_to(ROOT).as_posix(), p.stem)

    for title_dir in sorted((ROOT / EPAPER_DIR).iterdir()):
        if not title_dir.is_dir():
            continue
        for p in sorted(title_dir.glob("*.txt")):          # issue files; page files live in subfolders
            add("02_news_media", f"news_{slug(title_dir.name)}_{slug(p.stem)}",
                p.read_text(encoding="utf-8"), "epaper_ocr", p.relative_to(ROOT).as_posix(),
                f"{title_dir.name} {p.stem.split('_')[0]}")

    for p in sorted((ROOT / LDCIL_DIR).glob("*.txt")):
        xml = ROOT / "data/raw/08_ldcil_gold_sample" / f"{p.stem}.xml"
        if xml.resolve() in excluded:
            continue
        add("01_literature", f"ldcil_{p.stem.lower()}", p.read_text(encoding="utf-8"),
            "ldcil_gold", p.relative_to(ROOT).as_posix(), p.stem)

    for dump in sorted((ROOT / WIKI_DIR).glob("*.xml.bz2")):
        shards = [[] for _ in range(WIKI_SHARDS)]
        for i, (title, body) in enumerate(wiki_articles(dump)):
            shards[i % WIKI_SHARDS].append(body)
        for i, arts in enumerate(shards):
            add("06_web_articles", f"wiki_gom_{i:02d}", "\n\n".join(arts) + "\n", "wikipedia",
                dump.relative_to(ROOT).as_posix(), f"Konkani Wikipedia shard {i:02d} ({len(arts)} articles)")

    manifest = ROOT / args.manifest
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    by_cat = {}
    for r in rows:
        n, t = by_cat.get(r["category"], (0, 0))
        by_cat[r["category"]] = (n + 1, t + r["tokens"])
    for cat, (n, t) in sorted(by_cat.items()):
        print(f"{cat:24s} {n:5d} docs {t:>10,d} tokens")
    print(f"{'total':24s} {len(rows):5d} docs {sum(r['tokens'] for r in rows):>10,d} tokens -> {out}")


if __name__ == "__main__":
    main()
