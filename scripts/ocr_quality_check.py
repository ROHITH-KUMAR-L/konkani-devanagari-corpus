"""Estimate OCR quality by aligning hand-typed gold text with an OCR'd copy.

LDC-IL sample KO00002 (ghaNAghAya niyatIcE, Ashok Kamat, 2005) is the same
book as our OCR file 01_literature/GHANAGHAINIYATICHE.txt. Each gold paragraph
is fuzzy-aligned to its best-matching span in the OCR text, then we measure:
  - CER: character error rate (Levenshtein distance / gold length)
  - WER: word error rate (same, over whitespace tokens)
  - CER after removing the space OCR inserts before punctuation
  - the most frequent character-level confusions (guides clean_text.py rules)

Usage:
    python scripts/ocr_quality_check.py \
        --gold data/interim/08_ldcil_gold_sample/KO00002.txt \
        --ocr "data/raw/01_literature/GHANAGHAINIYATICHE.txt" \
        --out results/ocr_quality
"""
import argparse
import csv
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

MIN_GOLD_TOKENS = 15   # short paragraphs align ambiguously
MIN_ALIGN_SCORE = 80   # below this the paragraph is probably missing from the OCR


def normalise(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def fix_punct_spacing(text):
    return re.sub(r" ([.,?!;:।॥])", r"\1", text)


def char_name(ch):
    if ch == "":
        return "<none>"
    if ch == " ":
        return "<space>"
    return f"{ch} ({unicodedata.name(ch, hex(ord(ch)))})"


def confusions(gold, ocr):
    ops = Counter()
    for op in Levenshtein.editops(gold, ocr):
        src = gold[op.src_pos] if op.tag in ("replace", "delete") else ""
        dst = ocr[op.dest_pos] if op.tag in ("replace", "insert") else ""
        ops[(op.tag, src, dst)] += 1
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/interim/08_ldcil_gold_sample/KO00002.txt")
    ap.add_argument("--ocr", default="data/raw/01_literature/GHANAGHAINIYATICHE.txt")
    ap.add_argument("--out", default="results/ocr_quality")
    args = ap.parse_args()

    ocr = normalise(open(args.ocr, encoding="utf-8").read())
    gold_paras = [normalise(l) for l in open(args.gold, encoding="utf-8")]
    gold_paras = [p for p in gold_paras if len(p.split()) >= MIN_GOLD_TOKENS]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ops = [], Counter()
    gold_chars = err_chars = err_chars_punct = 0
    gold_words = err_words = err_words_punct = 0
    for i, gold in enumerate(gold_paras, 1):
        align = fuzz.partial_ratio_alignment(gold, ocr)
        if align.score < MIN_ALIGN_SCORE:
            rows.append({"para": i, "score": round(align.score, 1), "aligned": False})
            continue
        seg = ocr[align.dest_start:align.dest_end]
        cer_d = Levenshtein.distance(gold, seg)
        punct_d = Levenshtein.distance(gold, fix_punct_spacing(seg))
        wer_d = Levenshtein.distance(gold.split(), seg.split())
        wer_punct_d = Levenshtein.distance(gold.split(), fix_punct_spacing(seg).split())
        gold_chars += len(gold)
        err_chars += cer_d
        err_chars_punct += punct_d
        gold_words += len(gold.split())
        err_words += wer_d
        err_words_punct += wer_punct_d
        ops += confusions(gold, fix_punct_spacing(seg))
        rows.append({
            "para": i, "score": round(align.score, 1), "aligned": True,
            "gold_chars": len(gold), "cer": round(cer_d / len(gold), 4),
            "cer_punct_fixed": round(punct_d / len(gold), 4),
            "wer": round(wer_d / len(gold.split()), 4),
            "wer_punct_fixed": round(wer_punct_d / len(gold.split()), 4),
            "gold_start": gold[:60], "ocr_start": seg[:60],
        })

    fields = ["para", "score", "aligned", "gold_chars", "cer", "cer_punct_fixed",
              "wer", "wer_punct_fixed", "gold_start", "ocr_start"]
    with open(out / "per_paragraph.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(out / "confusions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "gold_char", "ocr_char", "count"])
        for (tag, src, dst), n in ops.most_common():
            writer.writerow([tag, char_name(src), char_name(dst), n])

    aligned = [r for r in rows if r["aligned"]]
    lines = [
        f"Gold:  {args.gold}",
        f"OCR:   {args.ocr}",
        f"Gold paragraphs (>= {MIN_GOLD_TOKENS} tokens): {len(gold_paras)}; "
        f"aligned (score >= {MIN_ALIGN_SCORE}): {len(aligned)}",
        f"Characters compared: {gold_chars}; words compared: {gold_words}",
        f"CER (raw OCR):                  {err_chars / gold_chars:.2%}",
        f"CER (punct spacing fixed):      {err_chars_punct / gold_chars:.2%}",
        f"WER (raw OCR):                  {err_words / gold_words:.2%}",
        f"WER (punct spacing fixed):      {err_words_punct / gold_words:.2%}",
        f"Median paragraph CER:           {statistics.median(r['cer'] for r in aligned):.2%}",
        "",
        "Top 20 character confusions (after punct spacing fix):",
    ]
    lines += [f"  {n:5d}  {tag:7s}  {char_name(s)}  ->  {char_name(d)}"
              for (tag, s, d), n in ops.most_common(20)]
    summary = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    print(summary)


if __name__ == "__main__":
    main()
