# E-paper OCR reference (FR-E6)

Reference text for measuring the newspaper OCR quality (CER/WER) of the
Bhaangarbhuin e-paper, the same way the book OCR was measured against the
LDC-IL gold (`results/ocr_quality/summary.txt`).

**Caveat:** this is an AI-transcribed reference (Claude vision), not
human-typed. We recommend that the team spot-check it. The transcriber did not
see the Tesseract output while transcribing. The text is written exactly as
printed, line by line, in Unicode NFC. The newspaper's own spelling is kept,
including misprints.

## What is here

| file | content |
|---|---|
| `manifest.csv` | one row per crop part: page PDF, page, dpi (150), pixel box `x,y,w,h` at that dpi |
| `<region>.txt` | reference transcription, one printed line per line (A1, A2, B1, B2, C1, C2) |
| `<region>.tesseract.txt` | Tesseract output for the same crop with the pipeline settings |
| `uncertain.tsv` | words the transcriber was not fully sure of (region, word, reason) |
| `images/` | the crops (git-ignored, because the newspaper images are copyrighted; regenerate them, see below) |

The six regions are ordinary running article text from page 3 of three issues:
2025-04-15 (A1, A2), 2025-11-18 (B1, B2) and 2026-06-16 (C1, C2). Together
they hold 902 words, 6,070 characters and 203 printed lines. Bylines,
headlines, captions and rule lines are left out. A region with two parts is
two adjacent columns of one article. The parts are stacked top to bottom,
with a 30 px white gap between them.

## How the OCR side is produced

`scripts/fetch_epaper.py` OCRs every page the same way: PyMuPDF renders the
page at 150 dpi in greyscale, then Tesseract runs with `mar+hin` and
`--psm 3`. The `--epaper` mode of `scripts/ocr_quality_check.py` repeats
those steps on each crop. It also scores the text that is actually stored in
the corpus (`data/interim/02_news_and_media/epaper/BHAANGARBHUIN/<issue>/page_03.txt`,
which is full-page OCR). Each printed line is aligned to that text on its own.
The script reports three measures:

- **(a) crop, whole region:** reading order counts (the requested CER/WER).
- **(b) crop, line-aligned:** recognition accuracy without layout effects.
- **(c) stored corpus text, line-aligned:** the character quality the corpus
  really has, plus "line order kept", which measures how often lines of
  neighbouring columns get interleaved.

The normalisation (NFC, whitespace collapse, removing the space before
punctuation), the metric code and the confusion table are reused from the
book evaluation. Results are in `results/ocr_quality/epaper_summary.txt`,
`epaper_per_region.csv` and `epaper_confusions.csv`.

## Regenerate

```bash
# needs the page PDFs from scripts/fetch_epaper.py in data/raw/02_news_and_media/epaper/
python scripts/ocr_quality_check.py --epaper --render --rerun-ocr
# scoring only (reuses the crops and the *.tesseract.txt files):
python scripts/ocr_quality_check.py --epaper
```

Tesseract 5.5.4 was used, with the Fedora `mar`/`hin` traineddata in
`/usr/share/tesseract/tessdata`. A different traineddata version will change
the `*.tesseract.txt` files a little.
