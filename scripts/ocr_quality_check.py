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

E-paper mode (--epaper): newspaper OCR quality for the Bhaangarbhuin e-paper.
Article regions listed in data/eval/ocr_gold/epaper/manifest.csv are cropped
from the page PDFs rendered exactly as scripts/fetch_epaper.py does (PyMuPDF,
150 dpi, greyscale), OCR'd with the pipeline's Tesseract settings (mar+hin,
--psm 3) and compared with a reference transcription <region>.txt. The
reference is AI-transcribed (Claude vision), not human-typed; see
data/eval/ocr_gold/epaper/README.md. No fuzzy alignment is needed: the
reference and the OCR cover the same crop.
    python scripts/ocr_quality_check.py --epaper             # reuse crops/OCR if present
    python scripts/ocr_quality_check.py --epaper --render --rerun-ocr
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


# ---------------------------------------------------------------- e-paper mode

EPAPER_DIR = "data/eval/ocr_gold/epaper"
PIPELINE_LANG = "mar+hin"   # same defaults as scripts/fetch_epaper.py
PIPELINE_PSM = 3
MIN_ORDER_CHARS = 12        # very short lines ("आसा.") align anywhere; skip them for line order
REGION_GAP = 30             # white rows between stacked column parts (px at 150 dpi)


def char_class(ch):
    """Coarse error type for a single character (used for the e-paper summary)."""
    if ch == "":
        return "<none>"
    if ch.isspace():
        return "space"
    o = ord(ch)
    if ch in "\u0901\u0902":
        return "anusvara/chandrabindu"
    if o == 0x093C:
        return "nukta"
    if o == 0x094D:
        return "virama"
    if 0x093E <= o <= 0x094C or o in (0x0962, 0x0963):
        return "vowel sign (matra)"
    if 0x0904 <= o <= 0x0914:
        return "independent vowel"
    if 0x0915 <= o <= 0x0939 or 0x0958 <= o <= 0x095F:
        return "consonant"
    if ch.isdigit():
        return "digit"
    if unicodedata.category(ch).startswith("P") or ch in "।॥":
        return "punctuation"
    if 0x0900 <= o <= 0x097F:
        return "other Devanagari"
    return "non-Devanagari"


def read_manifest(gold_dir):
    regions = {}
    with open(Path(gold_dir) / "manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            regions.setdefault(row["region_id"], []).append(row)
    for parts in regions.values():
        parts.sort(key=lambda r: int(r["part"]))
    return regions


def render_region(parts):
    """Render each part like fetch_epaper.page_text (PyMuPDF, greyscale) and stack them."""
    import fitz
    from PIL import Image
    crops = []
    for p in parts:
        with fitz.open(p["pdf"]) as doc:
            pix = doc[0].get_pixmap(dpi=int(p["dpi"]), colorspace=fitz.csGRAY)
        page = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        x, y, w, h = (int(p[k]) for k in ("x", "y", "w", "h"))
        crops.append(page.crop((x, y, x + w, y + h)))
    width = max(c.width for c in crops)
    height = sum(c.height for c in crops) + REGION_GAP * (len(crops) - 1)
    img = Image.new("L", (width, height), 255)
    top = 0
    for c in crops:
        img.paste(c, (0, top))
        top += c.height + REGION_GAP
    return img


def uncertain_spans(gold, words):
    """Character spans of every whitespace token of `gold` that is an uncertain word."""
    spans, pos = [], 0
    for tok in gold.split(" "):
        if tok.strip("\"'.,;:?!()") in words:
            spans.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return spans


def errors_outside(gold, ocr, spans):
    """Edit distance and gold length after dropping edits that touch uncertain spans."""
    inside = lambda i: any(a <= i < b for a, b in spans)
    kept = sum(1 for op in Levenshtein.editops(gold, ocr) if not inside(op.src_pos))
    excluded_chars = sum(b - a for a, b in spans)
    return kept, len(gold) - excluded_chars


def align_lines(lines, text):
    """Best-matching span of `text` for each reference line (order-free).

    The fuzzy window is snapped to word boundaries (whichever nearby start/end
    gives the smallest edit distance). Every line is kept, however low its
    score, so lines that the OCR lost or broke apart still count as errors.
    Returns (line, span_text, score, start, end) tuples."""
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == " "]
    ends = [i for i, c in enumerate(text) if c == " "] + [len(text)]
    out = []
    for line in lines:
        al = fuzz.partial_ratio_alignment(line, text)
        cs = [x for x in starts if abs(x - al.dest_start) <= 15] or [al.dest_start]
        ce = [x for x in ends if abs(x - al.dest_end) <= 15] or [al.dest_end]
        best = min(((Levenshtein.distance(line, text[a:b]), a, b)
                    for a in cs for b in ce if b > a), default=(0, al.dest_start, al.dest_end))
        _, s, e = best
        out.append((line, text[s:e], al.score, s, e))
    return out


def order_kept(aligned):
    """Share of consecutive reference lines that are also adjacent in the OCR text."""
    pairs = [(a, b) for a, b in zip(aligned, aligned[1:])
             if len(a[0]) >= MIN_ORDER_CHARS and len(b[0]) >= MIN_ORDER_CHARS]
    ok = sum(1 for a, b in pairs if 0 <= b[3] - a[4] <= 2)
    return ok, len(pairs)


def score(gold, ocr, unc_words=()):
    """Edit counts for one gold/OCR pair (OCR punct spacing fixed)."""
    ocr = fix_punct_spacing(ocr)
    spans = uncertain_spans(gold, unc_words)
    d_sure, n_sure = errors_outside(gold, ocr, spans)
    return Counter(chars=len(gold), words=len(gold.split()),
                   d=Levenshtein.distance(gold, ocr),
                   w=Levenshtein.distance(gold.split(), ocr.split()),
                   d_sure=d_sure, n_sure=n_sure, unc=len(spans))


def line_scores(lines, text, unc_words):
    aligned = align_lines(lines, text)
    tot = Counter()
    for line, seg, *_ in aligned:
        tot += score(line, seg, unc_words)
    tot["low"] = sum(1 for a in aligned if a[2] < MIN_ALIGN_SCORE)
    return tot, order_kept(aligned)


def epaper_main(args):
    import pytesseract
    gold_dir = Path(args.gold_dir)
    img_dir = gold_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    regions = read_manifest(gold_dir)
    uncertain = {}
    unc_path = gold_dir / "uncertain.tsv"
    if unc_path.exists():
        with open(unc_path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                uncertain.setdefault(row["region"], set()).add(unicodedata.normalize("NFC", row["word"]))

    rows, ops, classes = [], Counter(), Counter()
    tot_raw, tot, tot_line, tot_corp = Counter(), Counter(), Counter(), Counter()
    for rid, parts in regions.items():
        p = parts[0]
        img_path = img_dir / f"{rid}.png"
        if args.render or not img_path.exists():
            render_region(parts).save(img_path)
        ocr_path = gold_dir / f"{rid}.tesseract.txt"
        if args.rerun_ocr or not ocr_path.exists():
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(img_path), lang=args.lang,
                                               config=f"--psm {args.psm}")
            ocr_path.write_text(text, encoding="utf-8", newline="\n")
        gold_text = (gold_dir / f"{rid}.txt").read_text(encoding="utf-8")
        gold = normalise(gold_text)
        lines = [normalise(l) for l in gold_text.splitlines() if l.strip()]
        raw = normalise(ocr_path.read_text(encoding="utf-8"))
        unc = uncertain.get(rid, set())

        # (a) whole region, reading order matters (the requested CER/WER)
        r_raw = Counter(d=Levenshtein.distance(gold, raw),
                        w=Levenshtein.distance(gold.split(), raw.split()))
        r = score(gold, raw, unc)
        tot_raw += r_raw
        tot += r
        reg_ops = confusions(gold, fix_punct_spacing(raw))
        ops += reg_ops
        for (tag, src, dst), n in reg_ops.items():
            classes[(tag, char_class(src), char_class(dst))] += n
        # (b) same crop OCR, each printed line aligned on its own (layout-independent)
        rl, crop_order = line_scores(lines, raw, unc)
        tot_line += rl
        # (c) the text actually stored in the corpus (full-page OCR), line-aligned
        page_txt = Path(args.corpus_dir) / f"{p['issue_date']}_{p['issue_id']}" / f"page_{int(p['page']):02d}.txt"
        rc, corp_order = line_scores(lines, normalise(page_txt.read_text(encoding="utf-8")), unc)
        tot_corp += rc
        tot_corp.update(pairs_ok=corp_order[0], pairs=corp_order[1])
        tot_line.update(pairs_ok=crop_order[0], pairs=crop_order[1])

        rows.append({
            "region": rid, "issue_date": p["issue_date"], "issue_id": p["issue_id"],
            "page": p["page"], "parts": len(parts), "gold_lines": len(lines),
            "gold_chars": r["chars"], "gold_words": r["words"],
            "cer_raw": round(r_raw["d"] / r["chars"], 4), "cer": round(r["d"] / r["chars"], 4),
            "wer_raw": round(r_raw["w"] / r["words"], 4), "wer": round(r["w"] / r["words"], 4),
            "uncertain_tokens": r["unc"], "cer_excl_uncertain": round(r["d_sure"] / r["n_sure"], 4),
            "line_cer": round(rl["d"] / rl["chars"], 4), "line_wer": round(rl["w"] / rl["words"], 4),
            "line_order_kept": round(crop_order[0] / crop_order[1], 3),
            "corpus_line_cer": round(rc["d"] / rc["chars"], 4),
            "corpus_line_wer": round(rc["w"] / rc["words"], 4),
            "corpus_line_order_kept": round(corp_order[0] / corp_order[1], 3),
            "corpus_lines_score_lt80": rc["low"],
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "epaper_per_region.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "epaper_confusions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "gold_char", "ocr_char", "count"])
        for (tag, src, dst), n in ops.most_common():
            writer.writerow([tag, char_name(src), char_name(dst), n])

    pct = lambda a, b: f"{a / b:.2%}"
    n_err = sum(ops.values())
    lines = [
        "E-paper OCR quality (Bhaangarbhuin, FR-E6)",
        "Reference: AI-transcribed reference (Claude vision), not human-typed; "
        "team spot-check recommended.",
        f"Regions: {gold_dir / 'manifest.csv'} ({len(rows)} article regions from "
        f"{len({r['issue_id'] for r in rows})} issues: "
        f"{', '.join(sorted({r['issue_date'] for r in rows}))})",
        f"Characters compared: {tot['chars']}; words compared: {tot['words']}; "
        f"printed lines: {sum(r['gold_lines'] for r in rows)}",
        "",
        f"(a) Tesseract {args.lang} --psm {args.psm} on each region crop (rendered "
        f"{p['dpi']} dpi greyscale like fetch_epaper.py), whole region, reading order counts:",
        f"    CER (raw OCR):                {pct(tot_raw['d'], tot['chars'])}",
        f"    CER (punct spacing fixed):    {pct(tot['d'], tot['chars'])}",
        f"    WER (raw OCR):                {pct(tot_raw['w'], tot['words'])}",
        f"    WER (punct spacing fixed):    {pct(tot['w'], tot['words'])}",
        f"    CER excl. uncertain words:    {pct(tot['d_sure'], tot['n_sure'])}  "
        f"({tot['chars'] - tot['n_sure']} reference chars in {tot['unc']} uncertain tokens dropped)",
        f"    Median region CER:            {statistics.median(r['cer'] for r in rows):.2%}",
        f"(b) Same crop OCR, every printed line aligned on its own (layout-independent):",
        f"    CER {pct(tot_line['d'], tot_line['chars'])}   WER {pct(tot_line['w'], tot_line['words'])}   "
        f"CER excl. uncertain {pct(tot_line['d_sure'], tot_line['n_sure'])}   "
        f"line order kept {pct(tot_line['pairs_ok'], tot_line['pairs'])}",
        f"(c) Text actually stored in the corpus ({args.corpus_dir}/<issue>/page_NN.txt, "
        "full-page OCR), every printed line aligned on its own:",
        f"    CER {pct(tot_corp['d'], tot_corp['chars'])}   WER {pct(tot_corp['w'], tot_corp['words'])}   "
        f"CER excl. uncertain {pct(tot_corp['d_sure'], tot_corp['n_sure'])}   "
        f"line order kept {pct(tot_corp['pairs_ok'], tot_corp['pairs'])}   "
        f"lines with alignment score < {MIN_ALIGN_SCORE}: {tot_corp['low']}",
        "    'line order kept' = share of consecutive printed lines that are also adjacent in the OCR "
        "text; a low value means lines of neighbouring columns are interleaved.",
        "All rates micro-averaged; 'punct spacing fixed' removes the space OCR puts before "
        "punctuation, as in summary.txt.",
        "",
        "Per region:            words  (a) CER    WER   excl.unc | (b) CER    WER  | (c) CER    WER   order",
    ]
    lines += [f"  {r['region']}  {r['issue_date']} p{int(r['page']):02d}  {r['gold_words']:4d}  "
              f"{r['cer']:7.2%} {r['wer']:6.2%} {r['cer_excl_uncertain']:7.2%} | "
              f"{r['line_cer']:6.2%} {r['line_wer']:6.2%} | {r['corpus_line_cer']:6.2%} "
              f"{r['corpus_line_wer']:6.2%} {r['corpus_line_order_kept']:6.1%}"
              for r in rows]
    lines += ["", f"Error types in (a) (character class gold -> OCR; {n_err} edits):"]
    lines += [f"  {n:5d}  {n / n_err:6.1%}  {tag:7s}  {s} -> {d}"
              for (tag, s, d), n in classes.most_common(12)]
    lines += ["", "Top 20 character confusions in (a) (after punct spacing fix; full list in "
              "epaper_confusions.csv):"]
    lines += [f"  {n:5d}  {tag:7s}  {char_name(s)}  ->  {char_name(d)}"
              for (tag, s, d), n in ops.most_common(20)]
    summary = "\n".join(lines) + "\n"
    (out / "epaper_summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/interim/08_ldcil_gold_sample/KO00002.txt")
    ap.add_argument("--ocr", default="data/raw/01_literature/GHANAGHAINIYATICHE.txt")
    ap.add_argument("--out", default="results/ocr_quality")
    ap.add_argument("--epaper", action="store_true",
                    help="e-paper mode: score Tesseract on the regions in --gold-dir/manifest.csv")
    ap.add_argument("--gold-dir", default=EPAPER_DIR)
    ap.add_argument("--corpus-dir", default="data/interim/02_news_and_media/epaper/BHAANGARBHUIN",
                    help="(--epaper) per-page OCR text stored by fetch_epaper.py")
    ap.add_argument("--render", action="store_true", help="(--epaper) re-crop region images from the PDFs")
    ap.add_argument("--rerun-ocr", action="store_true", help="(--epaper) re-run Tesseract on the crops")
    ap.add_argument("--lang", default=PIPELINE_LANG)
    ap.add_argument("--psm", type=int, default=PIPELINE_PSM)
    args = ap.parse_args()
    if args.epaper:
        epaper_main(args)
        return

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
