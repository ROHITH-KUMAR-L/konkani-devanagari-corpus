# Data cleaning log

Each cleaning iteration is run by one script, with the analysis repeated after it (PRD FR-N10).
Every number below comes from `scripts/run_phase1.sh` and is filled in by `scripts/build_report.py`. The per-iteration outputs are in `results/iter0`, `results/iter1`, `results/iter2` and `results/` (final).

> **Run of {{date}}:** {{news_issues}} Bhaangarbhuin issues ({{news_first}} to {{news_last}}).

## Iteration 0: raw corpus (`extract_text.py`)

Iteration 0 is every source after encoding fixes only (UTF-8, BOM removed, CRLF → LF). Raw files in `data/raw` are never changed.

{{table_sources_iter0}}

Excluded before iteration 0 (see `metadata/excluded_documents.csv`):
- `MAHAROOKH.txt`, a duplicate of *Konkani Katha Sangraha*.
- LDC-IL `KO00002`, the same book as `GHANAGHAINIYATICHE`. It is kept only as the OCR gold reference.
- The broken textbook extractions `extracted1–3.txt`. The textbook PDF was re-OCR'd with `--force-ocr`, because its text layer has doubled vowel signs (`पाठ्यपुुस्तक`).

### Observations on iteration 0 (brief step 6)

1. **Template text in Wikipedia.** These words rank near the top of the raw frequency list: {{template_ranks}}. A bot created hundreds of village stubs from census tables ("गांवांत ट्रॅक्टर उपलब्ध ना."). Of {{wiki_lines}} Wikipedia lines, only {{wiki_unique}} are unique. If this is not removed, these few templates dominate the frequency list and the coverage curve.
2. **Numbers are the most frequent "word".** `<NUM>` makes up {{raw_num_pct}}% of tokens ({{raw_num}}). It comes from dates, scores, census figures, page numbers, verse numbers and OCR misreadings of `।`/`॥` as `1` or `॥`.
3. **Latin and other scripts make up {{raw_latin_pct}}% of tokens.** The top Latin tokens are Wikipedia image markup (`thumb`, `px`, `right`), English front matter in books (`of`, `the`, `Published`), and capitalised Goan place names from newspaper advertisements (`MARGAO`, `PANAJI`). There are also stray Gurmukhi, Bengali and Kannada letters and IPA symbols (`ː`) that OCR or Wikipedia introduced.
4. **Marathi and Hindi contamination.** Counting marker words found:
   - `HISTORY.txt` is almost entirely **Marathi** (कोंकणाख्यान with a Marathi commentary): {{history_kok}} Konkani markers against {{history_mar}} Marathi ones.
   - *Some Aspects of Konkani Culture Heritage* is a Konkani–**Hindi** parallel grammar: {{aspects_hin}} Hindi markers.
   - *Nataka* contains Marathi dialogue ({{nataka_mar}} Marathi markers).
5. **OCR noise.** Newspaper mastheads and ad pages produce Devanagari-looking garbage (`भिबेभाइभ0५॥णा`, `€090श.प€ट०भा`). Books have spaces before punctuation (`आसा .`), and words are split at line ends (`कर-\nतल्यो`). The OCR quality check (`results/ocr_quality/`) shows the most frequent character error is a **dropped anusvara**, followed by dropped viramas and dropped `ा`.
6. **Running headers and page numbers.** Every book page carries a header (`कोकणी कथासंग्रह`) and a page number. Page breaks cut paragraphs mid-sentence.
7. **Spelling variation is real Konkani variation, not noise.** Final-corpus counts: {{variants}}. The *Konkani Shuddhlekhanache Nem* rules give a standard form, but we did **not** normalise spelling: it would erase dialect information and needs a linguist's review.
8. **Long tail.** {{raw_hapax_pct}}% of types are hapax legomena. A random sample of hapaxes mixes genuine inflected forms (`पागोट्याचें`, `दुबायच्या`: postpositions attached to nouns) with OCR errors (`विधार्थ्र्यांक`, missing spaces such as `दुसऱ्याचेरआशिल्ल्या`).

## Iteration 1: normalisation and junk removal (`clean_text.py`)

The rules, each motivated by an observation above:
- **(obs. 3, 5)** Unicode NFC, HTML entities decoded, zero-width spaces and soft hyphens removed, ASCII `|` → `।`, doubled danda → `॥`.
- **(obs. 3, 6)** A line is junk if it has fewer than 2 Devanagari letters, its Devanagari share of letters is below 0.7, or it matches publisher/contact boilerplate (©, ISBN, phone, e-mail, URL, price). Short running headers that repeat at least 3 times as stand-alone lines are also removed.
- **(obs. 3)** Latin words and letters of other scripts are deleted from kept lines.
- **(obs. 5)** The space before punctuation is removed, which fixes most of the {{ocr_wer_raw}}% → {{ocr_wer}}% WER gap in the OCR check.
- **(obs. 5, 6)** Paragraphs are rebuilt:
  - wrapped lines are joined;
  - words hyphenated across a line break are rejoined;
  - a paragraph cut by a page break (a gap that held removed junk and no sentence-final punctuation) is joined across the break, e.g. `…अखेरेक मन घट` + `करून हांव…`.
- Paragraphs of fewer than 3 tokens (titles, headings) are dropped.

Result: {{junk_lines}} junk lines and {{short_paras}} short paragraphs removed (per document: `metadata/clean_log.csv`). Latin and other-script tokens went from {{raw_latin_pct}}% to {{it1_latin_pct}}%, and numbers from {{raw_num_pct}}% to {{it1_num_pct}}%.

## Iteration 2: Konkani-only, well-formed text (`validate_konkani.py`)

- **(obs. 4) Language ID by marker words.** Each paragraph's Konkani markers (`आसा, आनी, हांव, म्हजो, जाल्यार, न्हय, कितें…`) are counted against Marathi (`आहे, आणि, मी, माझा, नाही, झाले…`) and Hindi (`है, और, में, नहीं, था…`) markers. A paragraph is dropped if it has at least 2 Marathi/Hindi markers and they outnumber the Konkani ones. A whole document is dropped if it has fewer than 1 Konkani marker per 10 Marathi/Hindi markers.
- **(obs. 5) Orthography (FR-N8).** A paragraph is dropped if more than 15% of its tokens are impossible in Devanagari (start with a vowel sign, virama, nukta or anusvara; two vowel signs in a row; doubled virama), or if it mixes letters and digits inside a word (OCR garbage).
- **(obs. 2)** A paragraph is dropped if more than 40% of its tokens are numbers (tables, census rows, dates).

| Dropped | Paragraphs |
|---|---|
| Marathi | {{drop_marathi}} |
| Hindi | {{drop_hindi}} (mostly the Hindi half of *Some Aspects…*, plus ad-page garbage that happens to hit Hindi markers) |
| Orthography / OCR garbage | {{drop_orthography}} |
| Numeric | {{drop_numeric}} |
| Whole documents | {{docs_dropped}}: {{docs_dropped_list}} |

Per-document counts are in `metadata/quality_checks.csv`.

## Iteration 3: deduplication (`deduplicate.py`)

- **(obs. 1) Exact paragraph duplicates.** Paragraphs are compared on their letters only (punctuation, digits and spaces ignored), across the whole corpus; the first occurrence is kept. This removed **{{dedup_paras}} paragraphs ({{dedup_pct}}%)**: almost all Wikipedia village templates, plus repeated newspaper boilerplate (mastheads, weather boxes, standing columns).
- **Near-duplicate documents.** MinHash over word 5-grams, Jaccard ≥ 0.8: {{near_dup_docs}} found (the known book duplicate had already been excluded).

Per-document counts are in `metadata/dedup_log.csv`.

## Before/after statistics

{{table_iterations}}

{{table_category_iterations}}

Per-document survival through every iteration is in `results/document_audit.csv`.

### What the iterations changed

- **Deduplication moved coverage the most.** Before dedup, {{cov50_before}} types covered 50% of tokens; after, {{cov50_after}} are needed. Repeated template sentences had been inflating the counts of a few words. TTR rose from {{ttr_before}} to {{ttr_after}} for the same reason: {{dedup_tok}} duplicate tokens were removed but only {{dedup_types_lost}} types disappeared.
- **The Zipf slope moved from {{zipf_before}} to {{zipf_after}} after dedup**, towards the classic −1. The template words had created a "bump" in the mid ranks.
- **Cleaning barely changed the hapax share ({{raw_hapax_pct}}% → {{fin_hapax_pct}}%).** Most hapaxes are real inflected forms, not noise. This is the agglutination effect that LDC-IL also reports (see the report, §8).
- **Numbers fell from {{raw_num_pct}}% of tokens to {{fin_num_pct}}%** but stay the single most frequent type. They are kept, as `<NUM>`, so dates and amounts remain in sentences for the language models.

## Known remaining noise

- A {{audit_n}}-paragraph random audit of the final corpus is in `data/eval/language_audit_sample.tsv` (first pass by Claude; the team should verify it):
  - {{audit_nonkok}} non-Konkani ({{audit_nonkok_pct}}%): a Hindi line from the parallel grammar and a Portuguese transliteration;
  - {{audit_garbage}} pure OCR garbage ({{audit_garbage_pct}}%) and {{audit_partial}} partly garbled ({{audit_partial_pct}}%);
  - {{audit_fragment}} newspaper fragments where OCR merged columns ({{audit_fragment_pct}}%);
  - {{audit_dialect}} Konkani in other dialect spellings (Mangalorean, Cochin, Bible/Romi-style transliteration), which are kept.
- **Anusvara loss from OCR is not corrected** (FR-N9, optional). Automatic restoring is unsafe because both forms are often real words (`आसा` "is" vs `आसां` "I am").
- **Newspaper OCR sometimes interleaves two columns**, which produces fragments ("एकसारकी दिस ता. नंदकिशोर आनी").
- **Words joined because OCR missed a space** (`दुसऱ्याचेरआशिल्ल्या`) remain.
