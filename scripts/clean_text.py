"""Cleaning iteration 1: normalise text, remove junk lines, rebuild paragraphs.

Input/output are document trees. Output has one paragraph per line.

Steps (PRD FR-N1..N5):
  1. Unicode NFC; unescape HTML entities; drop BOM, zero-width space, soft hyphen,
     control characters; ASCII '|' and doubled dandas -> '।' / '॥'.
  2. Junk lines removed: fewer than 2 Devanagari letters (page numbers, OCR
     specks), Devanagari share of letters < 0.7 (English blurbs, other scripts),
     publisher/contact boilerplate (©, ISBN, phone, e-mail, URL, price ...),
     U+FFFD (broken extraction), and short running headers that repeat
     >= 3 times in a document as stand-alone lines.
  3. Inside kept lines: Latin-script words and letters of other scripts are
     removed; the space OCR inserts before punctuation is removed.
  4. Paragraphs rebuilt: for wrapped sources (books, newspapers, textbooks)
     lines are joined until a blank line; a word hyphenated across a line
     break is rejoined; a paragraph cut by a page break (blank gap that held
     removed junk, no sentence-final punctuation) is joined across the gap.
     Wikipedia and LDC-IL files already have one paragraph per line.
  5. Paragraphs with fewer than 3 word tokens are dropped (headings, titles).

Writes one row per document to metadata/clean_log.csv.

Usage:
    python scripts/clean_text.py --in data/interim/iter0 --out data/interim/iter1
"""
import argparse
import csv
import html
import re
import unicodedata
from collections import Counter

from corpus_utils import DEVA_LETTER, devanagari_ratio, iter_docs, read_doc, tokenize, write_doc

LINE_SOURCES = ("wiki_", "ldcil_")      # already one paragraph per line
MIN_DEVA_LETTERS = 2
MIN_DEVA_RATIO = 0.7
MIN_PARA_TOKENS = 3
HEADER_MAX_TOKENS = 5
HEADER_MIN_REPEATS = 3

BOILERPLATE = re.compile(
    r"©|\(c\)|ISBN|ISSN|Printed|Publish|Copyright|All rights|https?://|www\.|@\w|\.com\b|"
    r"\b(?:Ph|Phone|Tel|Fax|E-?mail)\b|फोन\s*:|मोल\s*:|किंमत\s*:|छापपी\s*:|छापणावळ|मुद्रक|"
    r"जुळणी\s*:|मुखचित्र\s*:|प्रकाशन\s+क्र",
    re.IGNORECASE)
REMOVE_CHARS = dict.fromkeys(map(ord, "​­﻿⁠"), None)
LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]+")
OTHER_LETTER = re.compile(rf"[^\W\d_{DEVA_LETTER}‌‍]")
SPACE_BEFORE = re.compile(r"\s+([।॥,.?!:;)\]}”’])")
SPACE_AFTER = re.compile(r"([(\[{“‘])\s+")
SENT_END = re.compile(r"[।॥.?!”\"'’]\s*$")


def normalise(text):
    text = unicodedata.normalize("NFC", html.unescape(text)).translate(REMOVE_CHARS)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = re.sub(r"\s*\|\s*\|\s*|\s*।\s*।\s*", " ॥ ", text)
    text = re.sub(r"(?<=\S)\s*\|", " ।", text)
    return text


def is_junk(line):
    if "�" in line or BOILERPLATE.search(line):
        return True
    ratio, deva = devanagari_ratio(line)
    return deva < MIN_DEVA_LETTERS or ratio < MIN_DEVA_RATIO


def fix_line(line):
    line = LATIN_WORD.sub(" ", line)
    line = OTHER_LETTER.sub("", line)
    line = SPACE_BEFORE.sub(r"\1", line)
    line = SPACE_AFTER.sub(r"\1", line)
    line = re.sub(r"([.,।])(?:\s*\1)+", r"\1", line)     # ". ." -> "."
    line = re.sub(r"\s*-\s*$", "-", line)               # keep a trailing hyphen marker
    return re.sub(r"[ \t]+", " ", line).strip()


def running_headers(lines):
    """Short stand-alone lines (blank before and after) that repeat in the document."""
    counts = Counter()
    for i, l in enumerate(lines):
        s = l.strip()
        alone = (i == 0 or not lines[i - 1].strip()) and (i + 1 == len(lines) or not lines[i + 1].strip())
        if s and alone and len(s.split()) <= HEADER_MAX_TOKENS:
            counts[s] += 1
    return {s for s, c in counts.items() if c >= HEADER_MIN_REPEATS}


def join_wrapped(parts):
    """parts: list of ('line', text) and ('gap', had_junk). Returns paragraphs."""
    paras, cur, pending_gap = [], "", None
    for kind, val in parts:
        if kind == "gap":
            pending_gap = (pending_gap or False) or val
            continue
        if cur and pending_gap is not None:
            # blank gap: new paragraph, unless a page break cut a sentence in half
            if not (pending_gap and not SENT_END.search(cur)):
                paras.append(cur)
                cur = ""
        pending_gap = None
        if cur.endswith("-") and re.match(rf"[{DEVA_LETTER}]", val):
            cur = cur[:-1] + val
        else:
            cur = f"{cur} {val}" if cur else val
    if cur:
        paras.append(cur)
    return paras


def clean_doc(text, doc_id):
    lines = normalise(text).split("\n")
    headers = running_headers(lines)
    stats = Counter(lines_in=sum(1 for l in lines if l.strip()))
    parts = []
    for l in lines:
        s = l.strip()
        if not s:
            parts.append(("gap", False))
            continue
        if s in headers or is_junk(s):
            stats["junk_lines"] += 1
            parts.append(("gap", True))
            continue
        s = fix_line(s)
        if s:
            parts.append(("line", s))
    if doc_id.startswith(LINE_SOURCES):
        paras = [v for k, v in parts if k == "line"]
    else:
        paras = join_wrapped(parts)
    out = []
    for p in paras:
        p = re.sub(r"(?<=\S)-$", "", p).strip()
        if len(tokenize(p)) >= MIN_PARA_TOKENS:
            out.append(p)
        else:
            stats["short_paragraphs"] += 1
    stats["paragraphs_out"] = len(out)
    return "\n".join(out) + ("\n" if out else ""), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="metadata/clean_log.csv")
    args = ap.parse_args()
    total, rows = Counter(), []
    for cat, doc_id, path in iter_docs(args.inp):
        text, st = clean_doc(read_doc(path), doc_id)
        write_doc(f"{args.out}/{cat}/{doc_id}.txt", text)
        total.update(st)
        rows.append({"doc_id": doc_id, "category": cat, "lines_in": st["lines_in"], "junk_lines": st["junk_lines"],
                     "short_paragraphs": st["short_paragraphs"], "paragraphs_out": st["paragraphs_out"]})
    with open(args.log, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"lines in {total['lines_in']:,}; junk lines removed {total['junk_lines']:,}; "
          f"short paragraphs dropped {total['short_paragraphs']:,}; paragraphs out {total['paragraphs_out']:,}")


if __name__ == "__main__":
    main()
