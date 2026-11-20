# Collection guidelines

How to add a new source so that it flows through the pipeline and stays traceable.

1. **Konkani in Devanagari only.** Romi, Kannada or Malayalam-script Konkani is out of scope. Marathi or Hindi text is removed automatically, but avoid collecting it.
2. **Put the file in `data/raw/<category>/`** and never edit it there. Categories: `01_literature`, `02_news_and_media`, `03_education_science`, `04_culture_history`, `05_public_information`, `06_web_articles`, `07_oral_transcripts`.
3. **Convert it to text.**
   - PDF with a real Unicode text layer: `fetch_epaper.py ocr` keeps the layer automatically.
   - Scanned or legacy-font PDF: `python scripts/fetch_epaper.py ocr --pdf-dir data/raw/<cat> --out data/interim/<cat> --dpi 300 --force-ocr`.
   - Check a page of the output by eye. Doubled vowel signs (`पुुस्तक`) mean the text layer is broken; use `--force-ocr`.
4. **Register it.**
   - Add one row to `metadata/sources.csv`: title, author/editor, publisher, year, URL, licence, who collected it, and the date. Leave a field blank rather than guess.
   - If the source is a new folder or format, add it to the source list in `scripts/extract_text.py`.
5. **Duplicates.** If a file duplicates another, add it to `metadata/excluded_documents.csv` with the reason and the file kept instead.
6. **Licence and robots.** Respect `robots.txt` and site terms. Use the site's own download links with a delay (`fetch_epaper.py` waits 1 s between requests). Record the licence. Copyrighted text is for academic use only and must not be pushed to the public repository.
7. **Re-run** `scripts/run_phase1.sh`. Check `results/document_audit.csv`: a new source that keeps less than about 60% of its tokens needs a look (wrong language, OCR garbage, or duplicates).
