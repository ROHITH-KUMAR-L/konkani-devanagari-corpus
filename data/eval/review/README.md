# Second-pass review of the evaluation gold sets

**What this is:** an independent second annotation pass by Claude (AI) on 2026-11-22. It is **not** human verification. The first pass was also done by Claude, so both passes may share the same blind spots.

## Method

- **Segmentation** (`segmentation_gold.txt`, 40 dev paragraphs; `segmentation_gold_heldout.txt`, 154 held-out paragraphs). For each paragraph, every candidate boundary (each token ending in `.` `?` `!` `।`, plus every existing `||` mark) was listed with its context. Each one was decided again from scratch with the Konkani rules: no split after abbreviations, initials, decimals, list numbers or datelines, and a closing quote goes with the sentence before it. These decisions were then compared with the first pass. Only the `||` marks were changed; the paragraph text is byte-identical.
- **Language audit** (`language_audit_sample.tsv`, 200 paragraphs). Each paragraph was re-labelled with the existing label set: `konkani`, `konkani_fragment`, `konkani_other_dialect_spelling`, `ocr_garbage`, `ocr_garbage_partial`, `non_konkani_hindi` and `non_konkani_unclear`. The new labels were then compared with the first pass. Only the label column was changed.

Every disagreement is listed in `segmentation_review.tsv` and `language_audit_review.tsv`.

## Agreement and changes

| File | Agreement (first pass vs independent pass) | Changes |
|---|---|---|
| segmentation_gold.txt (dev) | 99 of 101 boundaries in the union (98.0%) | 1 boundary added (line 7); 1 borderline case kept as is (line 6) |
| segmentation_gold_heldout.txt | 310 of 311 boundaries (99.7%) | 1 boundary removed (line 75) |
| language_audit_sample.tsv | 195 of 200 labels (97.5%) | 5 relabels: 4 to `konkani_other_dialect_spelling` and 1 to `ocr_garbage_partial` |

None of the labels that decide the language changed. The audit still has 2 non-Konkani paragraphs out of 200 (**1.0%**). The other counts are now:

- OCR garbage: 4 (2.0%)
- Partly garbled: 8 (4.0%, was 3.5%)
- Fragments: 21 (10.5%)
- Other dialect spelling: 11 (was 7)

## Segmentation scores after the review

The scores below come from `scripts/segment_sentences.py --eval`. The rules themselves are unchanged.

- Dev: P 0.9899, R 0.9800, F1 0.9849 (before: 0.9899 / 0.9899 / 0.9899)
- Held-out: P 0.9841, R 0.9968, F1 0.9904 (before: 0.9873 / 0.9968 / 0.9920)

## Still to do

A human Konkani reader should spot-check both review TSVs. They should also check a random sample of the boundaries and labels that both passes agreed on, because agreement between two AI passes is not independent evidence. The `konkani` versus `konkani_fragment` line for short truncated news lines is fuzzy in the first pass. It was left unchanged because neither choice is clearly wrong.
