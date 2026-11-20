# Data cleaning log

Each cleaning iteration is run by one script, with the analysis repeated after it (PRD FR-N10).
Every number below comes from `scripts/run_phase1.sh`. The per-iteration outputs are in `results/iter0`, `results/iter1`, `results/iter2` and `results/` (final).

> **Snapshot: 20 Nov 2026.** The full Bhaangarbhuin e-paper fetch was still running when these numbers were made: 43 of about 527 newspaper issues were in. Re-run `scripts/run_phase1.sh` after the fetch finishes. The method stays the same; the counts grow.

## Iteration 0: raw corpus (`extract_text.py`)

Iteration 0 is every source after encoding fixes only (UTF-8, BOM removed, CRLF → LF). Raw files in `data/raw` are never changed.

| Source | Docs | Raw tokens |
|---|---|---|
| Literature: 9 OCR'd books + 4 LDC-IL gold files | 13 | 243,015 |
| News: Bhaangarbhuin e-paper issues (OCR) | 43 | 727,354 |
| Education: NCERT Konkani textbook (OCR at 300 dpi) | 1 | 6,257 |
| Culture/history: 6 OCR'd books | 6 | 204,627 |
| Web: Konkani Wikipedia, Devanagari articles in 20 shards | 20 | 1,434,598 |
| **Total** | **83** | **2,615,851** |

Excluded before iteration 0 (see `metadata/excluded_documents.csv`):
- `MAHAROOKH.txt`, a duplicate of *Konkani Katha Sangraha*.
- LDC-IL `KO00002`, the same book as `GHANAGHAINIYATICHE`. It is kept only as the OCR gold reference.
- The broken textbook extractions `extracted1–3.txt`. The textbook PDF was re-OCR'd with `--force-ocr`, because its text layer has doubled vowel signs (`पाठ्यपुुस्तक`).

### Observations on iteration 0 (brief step 6)

1. **Template text in Wikipedia.** Ranks 6–11 of the frequency list are `गांवांत`, `उपलब्ध`, `सगळ्यांत`, `लागीं`, `अंतराचेर` and `किलोमिटर`. A bot created hundreds of village stubs from census tables ("गांवांत ट्रॅक्टर उपलब्ध ना."). Of 75,916 Wikipedia lines, only 38,888 are unique. If this is not removed, these few templates dominate the frequency list and the coverage curve.
2. **Numbers are the most frequent "word".** `<NUM>` makes up 4.8% of tokens (124,803). It comes from dates, scores, census figures, page numbers, verse numbers and OCR misreadings of `।`/`॥` as `1` or `॥`.
3. **Latin and other scripts make up 1.23% of tokens.** The top Latin tokens are Wikipedia image markup (`thumb`, `px`, `right`), English front matter in books (`of`, `the`, `Published`), and capitalised Goan place names from newspaper advertisements (`MARGAO`, `PANAJI`). There are also stray Gurmukhi, Bengali and Kannada letters and IPA symbols (`ː`) that OCR or Wikipedia introduced.
4. **Marathi and Hindi contamination.** Counting marker words found:
   - `HISTORY.txt` is almost entirely **Marathi** (कोंकणाख्यान with a Marathi commentary): 2 Konkani markers against 676 Marathi ones.
   - *Some Aspects of Konkani Culture Heritage* is a Konkani–**Hindi** parallel grammar: 3,176 Hindi markers.
   - *Nataka* contains Marathi dialogue.
5. **OCR noise.** Newspaper mastheads and ad pages produce Devanagari-looking garbage (`भिबेभाइभ0५॥णा`, `€090श.प€ट०भा`). Books have spaces before punctuation (`आसा .`), and words are split at line ends (`कर-\nतल्यो`). The OCR quality check (`results/ocr_quality/`) shows the most frequent character error is a **dropped anusvara** (118 edits), followed by dropped viramas and dropped `ा`.
6. **Running headers and page numbers.** Every book page carries a header (`कोकणी कथासंग्रह`) and a page number. Page breaks cut paragraphs mid-sentence.
7. **Spelling variation is real Konkani variation, not noise.** Examples: `पूण` 6,194 vs `पुण` 847; `म्हूण` 2,736 vs `म्हणून` 2,072; `सगळ्यांत` 2,261 vs `सगल्यांत` 221; `हांव` 3,193 vs `हाव` 80. The *Konkani Shuddhlekhanache Nem* rules give a standard form, but we did **not** normalise spelling: it would erase dialect information and needs a linguist's review.
8. **Long tail.** 62.6% of types are hapax legomena. A random sample of hapaxes mixes genuine inflected forms (`पागोट्याचें`, `दुबायच्या`: postpositions attached to nouns) with OCR errors (`विधार्थ्र्यांक`, missing spaces such as `दुसऱ्याचेरआशिल्ल्या`).

## Iteration 1: normalisation and junk removal (`clean_text.py`)

The rules, each motivated by an observation above:
- **(obs. 3, 5)** Unicode NFC, HTML entities decoded, zero-width spaces and soft hyphens removed, ASCII `|` → `।`, doubled danda → `॥`.
- **(obs. 3, 6)** A line is junk if it has fewer than 2 Devanagari letters, its Devanagari share of letters is below 0.7, or it matches publisher/contact boilerplate (©, ISBN, phone, e-mail, URL, price). Short running headers that repeat at least 3 times as stand-alone lines are also removed.
- **(obs. 3)** Latin words and letters of other scripts are deleted from kept lines.
- **(obs. 5)** The space before punctuation is removed, which fixes most of the 43% → 18% WER gap in the OCR check.
- **(obs. 5, 6)** Paragraphs are rebuilt:
  - wrapped lines are joined;
  - words hyphenated across a line break are rejoined;
  - a paragraph cut by a page break (a gap that held removed junk and no sentence-final punctuation) is joined across the break, e.g. `…अखेरेक मन घट` + `करून हांव…`.
- Paragraphs of fewer than 3 tokens (titles, headings) are dropped.

Result: 12,869 junk lines and 17,395 short paragraphs removed. Latin and other-script tokens went from 1.23% to 0.00%, and numbers from 4.77% to 4.33%.

## Iteration 2: Konkani-only, well-formed text (`validate_konkani.py`)

- **(obs. 4) Language ID by marker words.** Each paragraph's Konkani markers (`आसा, आनी, हांव, म्हजो, जाल्यार, न्हय, कितें…`) are counted against Marathi (`आहे, आणि, मी, माझा, नाही, झाले…`) and Hindi (`है, और, में, नहीं, था…`) markers. A paragraph is dropped if it has at least 2 Marathi/Hindi markers and they outnumber the Konkani ones. A whole document is dropped if it has fewer than 1 Konkani marker per 10 Marathi/Hindi markers.
- **(obs. 5) Orthography (FR-N8).** A paragraph is dropped if more than 15% of its tokens are impossible in Devanagari (start with a vowel sign, virama, nukta or anusvara; two vowel signs in a row; doubled virama), or if it mixes letters and digits inside a word (OCR garbage).
- **(obs. 2)** A paragraph is dropped if more than 40% of its tokens are numbers (tables, census rows, dates).

| Dropped | Paragraphs |
|---|---|
| Marathi | 177 |
| Hindi | 575 (mostly the Hindi half of *Some Aspects…*, plus ad-page garbage that happens to hit Hindi markers) |
| Orthography / OCR garbage | 561 |
| Numeric | 1,430 |
| Whole document: `cult_history` (Marathi) | 1 |

Per-document counts are in `metadata/quality_checks.csv`.

## Iteration 3: deduplication (`deduplicate.py`)

- **(obs. 1) Exact paragraph duplicates.** Paragraphs are compared on their letters only (punctuation, digits and spaces ignored), across the whole corpus; the first occurrence is kept. This removed **32,420 paragraphs (31.3%)**: almost all Wikipedia village templates, plus repeated newspaper boilerplate (mastheads, weather boxes, standing columns).
- **Near-duplicate documents.** MinHash over word 5-grams, Jaccard ≥ 0.8. None were found; the known book duplicate had already been excluded.

Per-document counts are in `metadata/dedup_log.csv`.

## Before/after statistics

| Iteration | Tokens | Types | TTR | Hapax (% types) | Devanagari % | Number % | Latin+other % | Types for 50% / 90% coverage | Zipf slope | Heaps β |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 raw | 2,615,851 | 282,711 | 0.1081 | 176,864 (62.6%) | 94.00 | 4.77 | 1.23 | 1,006 / 64,926 | -1.023 | 0.759 |
| 1 cleaned | 2,530,271 | 270,313 | 0.1068 | 168,602 (62.4%) | 95.67 | 4.33 | 0.00 | 975 / 61,466 | -1.026 | 0.759 |
| 2 Konkani only | 2,444,245 | 259,983 | 0.1064 | 161,723 (62.2%) | 96.45 | 3.55 | 0.00 | 959 / 59,002 | -1.031 | 0.758 |
| 3 deduplicated (final) | 2,140,209 | 259,913 | 0.1214 | 162,435 (62.5%) | 97.12 | 2.88 | 0.00 | 1,393 / 71,686 | -0.988 | 0.762 |

| Category | 0 raw | 1 cleaned | 2 Konkani only | 3 final | kept |
|---|---|---|---|---|---|
| 01_literature | 243,015 | 236,500 | 235,042 | 234,999 | 96.7% |
| 02_news_media | 727,354 | 709,811 | 672,914 | 665,943 | 91.6% |
| 03_education_science | 6,257 | 6,020 | 5,177 | 5,177 | 82.7% |
| 04_culture_history | 204,627 | 195,421 | 155,544 | 154,942 | 75.7% |
| 06_web_articles | 1,434,598 | 1,382,519 | 1,375,568 | 1,079,148 | 75.2% |

Per-document survival through every iteration is in `results/document_audit.csv`.

### What the iterations changed

- **Deduplication moved coverage the most.** Before dedup, 959 types covered 50% of tokens; after, 1,393 are needed. Repeated template sentences had been inflating the counts of a few words. TTR rose from 0.106 to 0.121 for the same reason: 304k duplicate tokens were removed but only 70 types disappeared.
- **The Zipf slope moved from -1.03 to -0.99 after dedup**, closer to the classic −1. The template words had created a "bump" in the mid ranks.
- **Cleaning barely changed the hapax share (62.6% → 62.5%).** Most hapaxes are real inflected forms, not noise. This is the agglutination effect that LDC-IL also reports (see the report, §5).
- **Numbers fell from 4.8% to 2.9% of tokens** but stay the single most frequent type. They are kept, as `<NUM>`, so dates and amounts remain in sentences for the language models.

## Known remaining noise

- A 100-paragraph random audit of the final corpus is in `data/eval/language_audit_sample.tsv` (first pass by Claude; the team should verify and extend it to 200):
  - about 1% possibly non-Konkani (a Portuguese transliteration);
  - about 3% OCR garbage or partial garbage;
  - about 4% newspaper fragments where OCR merged columns.
- **Anusvara loss from OCR is not corrected** (FR-N9 is optional). It inflates the type count with pairs like `आसा`/`आसां`.
- **Newspaper OCR sometimes interleaves two columns**, which produces fragments ("एकसारकी दिस ता. नंदकिशोर आनी").
- **Words joined because OCR missed a space** (`दुसऱ्याचेरआशिल्ल्या`) remain.
