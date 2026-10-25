"""Extract plain text and metadata from LDC-IL Konkani XML sample files.

The LDC-IL "Gold Standard Konkani Raw Text Corpus" files are hand-typed and
proofread, so they need no OCR cleaning. Each <p> in <body> is written as one
line (one paragraph per line) and the <Header> fields go into a metadata CSV.

Usage:
    python scripts/extract_ldcil.py \
        --in data/raw/08_ldcil_gold_sample \
        --out data/interim/08_ldcil_gold_sample \
        --meta metadata/ldcil_sample_metadata.csv
"""
import argparse
import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

META_FIELDS = [
    "index", "category", "subcategory", "text", "title", "author", "editor",
    "translator", "publisher", "pubPlace", "pubDate", "words", "letters",
    "samplingDesc",
]


def clean_paragraph(p):
    p = unicodedata.normalize("NFC", p)
    return re.sub(r"\s+", " ", p).strip()


def parse_file(path):
    root = ET.parse(path).getroot()
    meta = {}
    for field in META_FIELDS:
        node = root.find(f".//{field}")
        meta[field] = (node.text or "").strip() if node is not None else ""
    paragraphs = []
    for p in root.iter("p"):
        text = clean_paragraph("".join(p.itertext()))
        if text:
            paragraphs.append(text)
    return meta, paragraphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/08_ldcil_gold_sample")
    ap.add_argument("--out", default="data/interim/08_ldcil_gold_sample")
    ap.add_argument("--meta", default="metadata/ldcil_sample_metadata.csv")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for xml_path in sorted(Path(args.inp).glob("*.xml")):
        meta, paragraphs = parse_file(xml_path)
        out_path = out_dir / f"{xml_path.stem}.txt"
        # newline="\n" keeps LF endings on Windows too
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(paragraphs) + "\n")
        meta["file"] = xml_path.name
        meta["paragraphs"] = len(paragraphs)
        meta["tokens_counted"] = sum(len(p.split()) for p in paragraphs)
        rows.append(meta)
        print(f"{xml_path.name}: {len(paragraphs)} paragraphs, "
              f"{meta['tokens_counted']} tokens -> {out_path}")

    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file"] + META_FIELDS + ["paragraphs", "tokens_counted"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Total: {sum(r['tokens_counted'] for r in rows)} tokens; metadata -> {args.meta}")


if __name__ == "__main__":
    main()
