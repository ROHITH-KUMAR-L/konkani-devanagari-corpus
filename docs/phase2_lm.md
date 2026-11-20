# Phase 2 (part A): tokenisers and n-gram language models

This document covers requirements FR-P2-1 (tokenisers), FR-P2-2 (n-gram language models) and the perplexity half of FR-P2-7 (learning curve) from [PRD.md](../PRD.md) section 6.

> **Data snapshot.** All numbers below come from the splits of 21 Nov 2026, before the final corpus rebuild: train 291,106 sentences / 2,763,685 word tokens, dev 39,489 sentences / 383,610 tokens, test 32,775 sentences / 326,175 tokens (document-level 80/10/10 split). The final corpus is about 3-4 times larger. Re-run `bash scripts/run_phase2_lm.sh` on the final splits and update the tables from `results/phase2/*.csv`. The method and most observations should still hold, but the exact values will change.

## 1. How to run

```bash
bash scripts/run_phase2_lm.sh          # uses .venv-phase2/bin/python; override with PYTHON=...
```

| Step | Script | Outputs |
|---|---|---|
| Tokenisers | `scripts/phase2/tokenizers.py` | `results/phase2/tokenizer_eval.csv`, `results/phase2/tokenizer_normalisation.txt`, models in `data/models/` (git-ignored) |
| Sanity checks | `scripts/phase2/ngram_lm.py --selftest` | console only; fails loudly if a distribution does not sum to 1 |
| N-gram LMs | `scripts/phase2/ngram_lm.py` | `results/phase2/ngram_perplexity.csv`, `subword_perplexity.csv`, `learning_curve.csv`, `plots/ngram_perplexity.png`, `plots/learning_curve.png` |

Runtime on the snapshot: about 107 s end to end (tokenisers 70 s, self-test 4 s, LMs 33 s). Peak memory was 1.36 GB, reached during SentencePiece Unigram training. Seeds are fixed (`--seed 13`). All scripts accept `--help`.

## 2. Method

### 2.1 Text given to every model

Each line of `data/splits/*.txt` is one sentence. It is tokenised with `corpus_utils.tokenize`, the same function Phase 1 used for all statistics: punctuation is removed and every number becomes `<NUM>`. Sentences with no word token (a punctuation-only line) are skipped. The word models, the SentencePiece tokenisers and the subword LM all see exactly this word stream, so their numbers can be compared directly.

### 2.2 Tokenisers (FR-P2-1)

* **Word baseline.** The vocabulary is either every train word type (273,462 types) or only the types seen at least twice (105,807). The second one is the n-gram LM vocabulary.
* **SentencePiece BPE and Unigram** with 8k, 16k and 32k pieces. Training text is the train word stream, one sentence per line. A deterministic sample of at most 500k sentences is used to bound trainer memory; the whole snapshot fits under that cap. Settings: `character_coverage=1.0` (the corpus is cleaned Devanagari, so every character is kept), `user_defined_symbols=<NUM>`, `num_threads=4`.
* **Normalisation: `identity`, not the default `nmt_nfkc`.** The script measures what each rule would change in the train types (`results/phase2/tokenizer_normalisation.txt`):
  * `nmt_nfkc` alters 253 train types (429 tokens). It deletes the ZERO WIDTH NON-JOINER (U+200C), which Konkani uses after a virama to force a visible halant, e.g. `बदलप्‌` becomes `बदलप्`. Sometimes it even turns the ZWNJ into a word boundary: `उक्त्‌या` becomes `उक्त्▁या`.
  * Plain `nfkc` alters 0 types. Our text is already NFC, and Devanagari has no compatibility characters in it.
  * So `identity` keeps the NFC text byte for byte. It loses nothing compared with `nfkc`, and it avoids the ZWNJ damage of `nmt_nfkc`.

Metrics (computed on dev and test; the tables show test):

* **fertility**: subword pieces per word token.
* **split share**: share of word tokens cut into more than one piece.
* **unknown-piece rate**: share of pieces that are `<unk>`.
* **OOV token rate** and **OOV type rate**: for the word baseline, the share of test tokens (or types) missing from the vocabulary.

SentencePiece never merges across a space, so each word type is encoded once and weighted by its frequency. The script checks on 500 test sentences that this gives exactly the same pieces as encoding whole sentences.

### 2.3 N-gram language models (FR-P2-2)

* **Vocabulary.** Train words with count ≥ 2, plus `<UNK>`, `<s>`, `</s>` (V = 105,810). Every other word maps to `<UNK>`. On test, 8.77% of predicted tokens are `<UNK>` (dev: 12.54%).
* **Sentences.** Each sentence is wrapped as `<s> w1 … wn </s>`, and the model predicts w1 … wn and `</s>`. There is a single `<s>`, so at the start of a sentence the history is just shorter, as in SRILM and KenLM. Perplexity = 2^(−mean log2 P) over all predicted tokens, `</s>` included.
* **Storage.** Implemented from scratch with numpy; no LM library is used. A k-gram (h, w) is stored as the int64 key `id(h)·V + w`, where id(h) is the index of h among the sorted (k−1)-grams. Counting is `np.unique` over these keys, and lookup is `np.searchsorted`. This keeps 4-gram training on the snapshot within a few hundred MB.
* **Smoothing methods:**

| Name | Definition |
|---|---|
| `laplace` | add-1 on the highest available order: P = (c(h,w)+1)/(c(h)+V′), with V′ = V − 1 (everything except `<s>`) |
| `addk` | add-k with k chosen per n on dev from {1, 0.5, …, 1e-5} (16 values) |
| `katz` | Katz backoff with Good-Turing discounts for counts ≤ 5. Backoff weights α(h) make every distribution sum to 1. Contexts whose continuations are all frequent (so no GT discount applies) keep a floor of 0.5/c(h) of mass for unseen words. Unigram = MLE. |
| `kn` | interpolated Kneser-Ney with one discount per order, D = n1/(n1+2n2) |
| `mkn` | interpolated modified Kneser-Ney (Chen & Goodman) with discounts D1, D2, D3+ per order from count-of-counts |

  Both KN variants use continuation counts N1+(• h w) for the lower orders. N-grams that start with `<s>` have no left context, so they keep their raw counts. The recursion ends in a uniform distribution over V′.
* **Self-test (`--selftest`).** Runs on a toy corpus and on the first 20k train sentences, for all five methods with 4-grams. For 5 random contexts × every history length (0 to 3 words), it computes P(· | h) over the whole vocabulary and checks that it sums to 1 within 1e-6; the worst error was 4.6e-8. It also checks that the vectorised scorer gives the same log-probability as this full-distribution path at 20 random positions.
* **Subword LM.** The same `mkn` model over SentencePiece Unigram-16k pieces of the same word stream, for n = 1 … 5 (n is counted in pieces). Its perplexity is normalised **per word**:

  PPL_word = 2^(−Σ log2 P(pieces) / (number of word tokens + number of `</s>`))

  **Why per-word normalisation matters.** Perplexity is the inverse geometric mean probability *per predicted unit*. A piece-level model predicts 1.37 units per word on test, and each piece is easier to guess than a whole word. Its per-piece perplexity (308 at n = 5) therefore looks better than any word model, but it measures a different, easier task. The total log-probability of the test text is the quantity both models share. Dividing it by the same denominator (words + `</s>`) gives numbers on one scale, i.e. per-piece PPL^(pieces/words). With this normalisation, the subword and word models below are compared on the same events.

### 2.4 Learning curve (FR-P2-7)

The `mkn` trigram is trained on the first 10, 25, 50 and 100% of train sentences, and test perplexity is reported in two ways:

* **(a) own vocabulary:** count ≥ 2 within the subset. This is how you would train if you only had that much data.
* **(b) fixed vocabulary:** the full-train vocabulary, so every row predicts the same events and the `<UNK>` rate stays at 8.77%.

Only (b) is a fair comparison across sizes.

## 3. Results

### 3.1 Tokenisers (test split)

| Tokeniser | Vocab | Fertility | Words split into > 1 piece | Unk-piece rate | OOV tokens | OOV types |
|---|---:|---:|---:|---:|---:|---:|
| word, all train types | 273,462 | 1.000 | 0% | – | 7.33% | 33.32% |
| word, train count ≥ 2 | 105,807 | 1.000 | 0% | – | 9.65% | 43.11% |
| BPE 8k | 8,000 | 1.578 | 37.8% | 0 | 0 | 0 |
| BPE 16k | 16,000 | 1.412 | 28.9% | 0 | 0 | 0 |
| BPE 32k | 32,000 | **1.292** | **21.7%** | 0 | 0 | 0 |
| Unigram 8k | 8,000 | 1.563 | 35.7% | 0 | 0 | 0 |
| Unigram 16k | 16,000 | 1.408 | 28.2% | 0 | 0 | 0 |
| Unigram 32k | 32,000 | 1.304 | 22.7% | 0 | 0 | 0 |

Dev has higher fertility (BPE 32k 1.349; Unigram 16k 1.473) and higher word OOV (11.14% of tokens with all train types), because dev documents are further from train.

### 3.2 Word n-gram perplexity (dev / test)

| n | Laplace | add-k (k on dev) | Katz + GT | KN | modified KN |
|---|---:|---:|---:|---:|---:|
| 1 | 2,030 / 2,425 | 2,009 / 2,398 (k=1e-5) | 2,009 / 2,398 | 2,009 / 2,398 | 2,009 / 2,398 |
| 2 | 8,737 / 10,015 | 1,517 / 1,536 (k=0.002) | 653 / 668 | 631 / 646 | 626 / 642 |
| 3 | 35,196 / 38,690 | 9,777 / 9,664 (k=5e-4) | 617 / 610 | 561 / 553 | 554 / 548 |
| 4 | 50,288 / 53,175 | 24,706 / 23,762 (k=2e-4) | 633 / 626 | 552 / 540 | **545 / 536** |

* Modified-KN discounts for the highest order: D1/D2/D3+ = 0.768/1.224/1.464 (n=2), 0.889/1.347/1.455 (n=3), 0.943/1.478/1.497 (n=4).
* At n = 1, all methods except Laplace reduce to the maximum-likelihood unigram. The vocabulary has no count-1 words, so count-of-counts give no discount estimate.

Plot: `results/phase2/plots/ngram_perplexity.png`.

### 3.3 Subword LM vs word LM (test, per-word perplexity)

| Model | All tokens | Tokens whose word is in the LM vocabulary (91.2%) | OOV tokens only |
|---|---:|---:|---:|
| word mKN 4-gram (OOV predicted as `<UNK>`) | 535.6 | 752.8 | (only `<UNK>`) |
| subword mKN, n=3 pieces | 2,778 | 771.5 | 1.7e9 |
| subword mKN, n=4 pieces | 2,596 | 722.8 | 1.6e9 |
| subword mKN, n=5 pieces | **2,568** | **715.1** | 1.5e9 (≈ 30.5 bits per OOV word) |

Per-piece perplexity of the subword model at n = 5 is 307.6, with 1.37 pieces per word incl. `</s>`. It is not comparable with the word models (see section 2.3).

### 3.4 Learning curve (mKN trigram, test)

| Train share | Sentences | Tokens | (a) own vocab V | (a) `<UNK>` rate | (a) PPL | (b) fixed vocab PPL (`<UNK>` 8.77%) |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 29,111 | 203,554 | 15,769 | 34.1% | 237.7 | 4,690.6 |
| 25% | 72,776 | 602,239 | 35,246 | 16.2% | 458.2 | 1,127.2 |
| 50% | 145,553 | 1,295,906 | 55,970 | 12.6% | 517.8 | 829.6 |
| 100% | 291,106 | 2,763,685 | 105,810 | 8.8% | 547.9 | 547.9 |

Plot: `results/phase2/plots/learning_curve.png`.

## 4. Observations and inferences

1. **Konkani words are many and long-tailed, so a word vocabulary leaks.** A third of test word *types* (33.3%) and 7.3% of test tokens never occur in 2.76M train tokens. With the LM threshold (count ≥ 2), 9.7% of tokens are OOV. Konkani is agglutinative: case markers, postpositions and clitics attach to the stem (e.g. `भुरग्यांक`, `आपल्या`, `गांवच्या`), so one lemma shows up as many surface forms. Each form is rare even when the stem is common. *Inference:* a closed word vocabulary will always throw away a noticeable share of the text in this language. Subword units are needed for any model that has to produce or score every word (embeddings with FastText, neural LMs, NER).

2. **SentencePiece removes OOV completely at a cost of 1.3-1.6 pieces per word.** With `character_coverage=1.0` the unknown-piece rate is 0 on both dev and test. Fertility drops as the vocabulary grows: about 1.57 (8k), 1.41 (16k) and 1.30 (32k). At 32k, about 78% of word tokens stay whole and only the rarer, longer forms are split. *Inference:* 16k-32k is the useful range. 32k gives the shortest sequences, while 16k gives better-trained piece embeddings when data is small.

3. **BPE and Unigram are almost tied.** Unigram has slightly lower fertility at 8k (1.563 vs 1.578) and 16k (1.408 vs 1.412), and BPE at 32k (1.292 vs 1.304). Every gap is under 1%. *Inference:* fertility alone does not choose between them. Unigram is the better default for the LM because it gives a probabilistic segmentation, which matters for subword regularisation in Phase 3. **Best tokeniser on these metrics: BPE-32k** (lowest fertility and split share); Unigram-32k is within 1%.

4. **Normalisation is not a neutral default.** SentencePiece's default `nmt_nfkc` would delete the ZWNJ in 253 train types and sometimes split a word in two. Plain NFKC changes nothing on our NFC text. *Inference:* for Indic scripts, normalisation must be checked on the data, not taken from a default. `identity` is the safe choice for this corpus.

5. **Laplace gets worse as n grows; KN gets better.** Laplace test perplexity rises from 2,425 (n=1) to 53,175 (n=4), while modified KN falls from 2,398 to 536. Add-1 gives every one of the V′ = 105,809 possible next words a pseudo-count of 1, so a context seen c times keeps only c/(c+105,809) of its mass for what was actually observed. Higher-order contexts are seen rarely (most 3- and 4-gram contexts only a few times), so almost all mass goes to unseen words. *Inference:* add-one is unusable at this vocabulary size. This confirms the PRD Phase 2 acceptance criterion: the KN trigram (548) is 70× lower than the Laplace trigram (38,690).

6. **Tuning k helps but cannot fix additive smoothing.** The best k falls with n (0.002 for bigrams, 2e-4 for 4-grams). Add-k still gets worse beyond the bigram (1,536 → 9,664 → 23,762). It has no backoff: a word unseen after a known context gets the same tiny probability whether it is common or rare in general. *Inference:* information from lower orders (backoff or interpolation) is what makes higher orders usable.

7. **Kneser-Ney beats Katz/Good-Turing, and modified KN beats plain KN.** At every n ≥ 2 the order is mKN < KN < Katz (e.g. trigram 548 < 553 < 610). Katz also gets worse from n=3 to n=4 (610 → 626), while KN still improves (548 → 536). KN's lower orders use continuation counts: how many different words a word follows, not how often it occurs. A word frequent in only one fixed phrase therefore gets little backoff mass, which is the right behaviour for unseen contexts. Interpolation also mixes lower orders in for every word, not only unseen ones. The three discounts of modified KN (e.g. 0.94/1.48/1.50 at n=4) fit count-1, count-2 and count-3+ n-grams separately, which gains a further 1-2%. *Inference:* use modified KN as the n-gram baseline for the Phase 3 neural LMs.

8. **Most of the gain from context comes from the bigram.** Modified KN goes 2,398 → 642 (bigram, −73%) → 548 (trigram, −15%) → 536 (4-gram, −2%). With 2.76M training tokens, most 4-gram histories in test were never seen, so the model backs off. *Inference:* with this amount of data, n = 3 is the sensible operating point. Longer contexts need more data or a neural model that shares statistics across similar histories.

9. **The word model's 536 is optimistic, because `<UNK>` is a cheap prediction.** On test, 8.8% of tokens are `<UNK>`, and predicting "some rare word" is far easier than predicting the word itself. Restricted to tokens whose word is in the vocabulary, the word 4-gram scores 752.8. On the same tokens the subword 5-gram scores 715.1, i.e. it is **better** while having a 16k vocabulary instead of 106k. It shares statistics across inflected forms of one stem. On all tokens the subword model scores 2,568, because it pays the real cost of spelling every OOV word, about 30.5 bits each. *Inference:* when comparing models with different vocabularies, compare per-word perplexity over the same events. A word-LM perplexity with `<UNK>` is not comparable with an open-vocabulary model.

10. **More data still helps a lot; the curve has not flattened.** With the vocabulary fixed (b), each doubling of training data cuts trigram perplexity strongly: 1,127 → 830 (−26%, 2.2× data) → 548 (−34%, 2.1× data). There is no sign of saturation at 2.76M tokens. *Inference:* the final corpus (3-4× larger) should lower n-gram perplexity clearly again, and more text is still the most effective improvement for these models.

11. **Comparing perplexity across vocabularies can mislead.** Under (a), perplexity *rises* with more data (238 → 458 → 518 → 548). The small subsets have tiny vocabularies (15,769 words at 10%), so 34% of test tokens collapse into one `<UNK>` and the task becomes easier. This is the same effect as observation 9, seen from the data side. *Inference:* only a fixed evaluation vocabulary (or per-word open-vocabulary perplexity) shows the true benefit of more data.

## 5. Limitations and notes

* The learning-curve subsets are the *first* X% of train sentences. Train is ordered by document, so the subsets are not domain-balanced. The 10% subset has 7.4% of the tokens, so its sentences are shorter than average. A shuffled-document curve would be a cleaner FR-P2-7 measurement.
* Good-Turing is used inside Katz backoff with K = 5. Simple Good-Turing (Gale & Sampson's log-linear smoothing of n_r) is not implemented. The 0.5/c(h) floor for contexts without discountable counts is our own choice; without it, such contexts would give an unseen word probability 0.
* KenLM (mentioned in the PRD for 4/5-grams) is not used. Our own implementation reaches 4-grams on words and 5-grams on pieces within the memory budget. It is checked by the `--selftest` normalisation test, not against KenLM output.
* The subword LM is trained only for Unigram-16k. Other tokenisers can be passed with `--spm data/models/spm_<algo>_<size>.model`.
* Dev and test differ: dev has more OOV words (12.5% vs 8.8% `<UNK>`) and higher fertility, yet a lower unigram perplexity (2,009 vs 2,398), because `<UNK>` is one frequent, cheap event. From n = 2 on, dev and test perplexities of the Katz and KN models are within about 3% of each other. These are properties of the document-level split, not of the models.
