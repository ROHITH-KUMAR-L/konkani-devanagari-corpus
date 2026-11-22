# Phase 2 (part A): tokenisers and n-gram language models

This document covers requirements FR-P2-1 (tokenisers), FR-P2-2 (n-gram language models) and the perplexity half of FR-P2-7 (learning curve) from [PRD.md](../PRD.md) section 6.

> **Data.** All numbers come from the final corpus splits (`corpus-v1.0`, 22 Nov 2026): train 893,710 sentences / 8,461,056 word tokens, dev 106,823 sentences / 1,008,663 tokens, test 109,907 sentences / 1,036,051 tokens (document-level 80/10/10 split). An earlier draft of this document used a 2.76M-token snapshot; the method is unchanged.

## 1. How to run

```bash
bash scripts/run_phase2_lm.sh          # uses .venv-phase2/bin/python; override with PYTHON=...
```

| Step | Script | Outputs |
|---|---|---|
| Tokenisers | `scripts/phase2/tokenizers.py` | `results/phase2/tokenizer_eval.csv`, `results/phase2/tokenizer_normalisation.txt`, models in `data/models/` (git-ignored) |
| Sanity checks | `scripts/phase2/ngram_lm.py --selftest` | console only; fails loudly if a distribution does not sum to 1 |
| N-gram LMs | `scripts/phase2/ngram_lm.py` | `results/phase2/ngram_perplexity.csv`, `subword_perplexity.csv`, `learning_curve.csv`, `plots/ngram_perplexity.png`, `plots/learning_curve.png` |

Runtime on the final splits: about 2.5 min end to end. Seeds are fixed (`--seed 13`). All scripts accept `--help`.

## 2. Method

### 2.1 Text given to every model

Each line of `data/splits/*.txt` is one sentence. It is tokenised with `corpus_utils.tokenize`, the same function Phase 1 used for all statistics: punctuation is removed and every number becomes `<NUM>`. Sentences with no word token (a punctuation-only line) are skipped. The word models, the SentencePiece tokenisers and the subword LM all see exactly this word stream, so their numbers can be compared directly.

### 2.2 Tokenisers (FR-P2-1)

* **Word baseline.** The vocabulary is either every train word type (472,014 types) or only the types seen at least twice (192,360). The second one is the n-gram LM vocabulary.
* **SentencePiece BPE and Unigram** with 8k, 16k and 32k pieces. Training text is the train word stream, one sentence per line. A deterministic sample of at most 500k sentences (seed 13) is used to bound trainer memory. Settings: `character_coverage=1.0` (the corpus is cleaned Devanagari, so every character is kept), `user_defined_symbols=<NUM>`, `num_threads=4`.
* **Normalisation: `identity`, not the default `nmt_nfkc`.** The script measures what each rule would change in the train types (`results/phase2/tokenizer_normalisation.txt`):
  * `nmt_nfkc` alters 702 train types (1,689 tokens). It deletes the ZERO WIDTH NON-JOINER (U+200C), which Konkani uses after a virama to force a visible halant, e.g. `बदलप्‌` becomes `बदलप्`. Sometimes it even turns the ZWNJ into a word boundary: `उक्त्‌या` becomes `उक्त्▁या`.
  * Plain `nfkc` alters 0 types. Our text is already NFC, and Devanagari has no compatibility characters in it.
  * So `identity` keeps the NFC text byte for byte. It loses nothing compared with `nfkc`, and it avoids the ZWNJ damage of `nmt_nfkc`.

Metrics (computed on dev and test; the tables show test):

* **fertility**: subword pieces per word token.
* **split share**: share of word tokens cut into more than one piece.
* **unknown-piece rate**: share of pieces that are `<unk>`.
* **OOV token rate** and **OOV type rate**: for the word baseline, the share of test tokens (or types) missing from the vocabulary.

SentencePiece never merges across a space, so each word type is encoded once and weighted by its frequency. The script checks on 500 test sentences that this gives exactly the same pieces as encoding whole sentences.

### 2.3 N-gram language models (FR-P2-2)

* **Vocabulary.** Train words with count ≥ 2, plus `<UNK>`, `<s>`, `</s>` (V = 192,363). Every other word maps to `<UNK>`. On test, 4.96% of predicted tokens are `<UNK>` (dev: 4.72%).
* **Sentences.** Each sentence is wrapped as `<s> w1 … wn </s>`, and the model predicts w1 … wn and `</s>`. There is a single `<s>`, so at the start of a sentence the history is just shorter, as in SRILM and KenLM. Perplexity = 2^(−mean log2 P) over all predicted tokens, `</s>` included.
* **Storage.** Implemented from scratch with numpy; no LM library is used. A k-gram (h, w) is stored as the int64 key `id(h)·V + w`, where id(h) is the index of h among the sorted (k−1)-grams. Counting is `np.unique` over these keys, and lookup is `np.searchsorted`. This keeps 4-gram training on 8.5M tokens within a few GB.
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

  **Why per-word normalisation matters.** Perplexity is the inverse geometric mean probability *per predicted unit*. A piece-level model predicts 1.32 units per word on test, and each piece is easier to guess than a whole word. Its per-piece perplexity (201 at n = 5) therefore looks better than any word model, but it measures a different, easier task. The total log-probability of the test text is the quantity both models share. Dividing it by the same denominator (words + `</s>`) gives numbers on one scale, i.e. per-piece PPL^(pieces/words). With this normalisation, the subword and word models below are compared on the same events.

### 2.4 Learning curve (FR-P2-7)

The `mkn` trigram is trained on the first 10, 25, 50 and 100% of train sentences, and test perplexity is reported in two ways:

* **(a) own vocabulary:** count ≥ 2 within the subset. This is how you would train if you only had that much data.
* **(b) fixed vocabulary:** the full-train vocabulary, so every row predicts the same events and the `<UNK>` rate stays at 4.96%.

Only (b) is a fair comparison across sizes.

## 3. Results

### 3.1 Tokenisers (test split)

| Tokeniser | Vocab | Fertility | Words split into > 1 piece | Unk-piece rate | OOV tokens | OOV types |
|---|---:|---:|---:|---:|---:|---:|
| word, all train types | 472,014 | 1.000 | 0% | – | 4.12% | 30.31% |
| word, train count ≥ 2 | 192,360 | 1.000 | 0% | – | 5.49% | 39.63% |
| BPE 8k | 8,000 | 1.536 | 36.0% | 0 | 0 | 0 |
| BPE 16k | 16,000 | 1.368 | 26.6% | 0 | 0 | 0 |
| BPE 32k | 32,000 | **1.252** | **19.4%** | 0 | 0 | 0 |
| Unigram 8k | 8,000 | 1.513 | 33.3% | 0 | 0 | 0 |
| Unigram 16k | 16,000 | 1.359 | 25.4% | 0 | 0 | 0 |
| Unigram 32k | 32,000 | 1.255 | 19.5% | 0 | 0 | 0 |

Dev is very close to test (BPE 32k fertility 1.245; word OOV 3.91% of tokens with all train types).

### 3.2 Word n-gram perplexity (dev / test)

| n | Laplace | add-k (k on dev) | Katz + GT | KN | modified KN |
|---|---:|---:|---:|---:|---:|
| 1 | 3,426 / 3,352 | 3,406 / 3,333 (k=1e-5) | 3,406 / 3,333 | 3,406 / 3,333 | 3,406 / 3,333 |
| 2 | 11,909 / 11,543 | 1,426 / 1,371 (k=0.001) | 684 / 661 | 659 / 637 | 656 / 635 |
| 3 | 57,091 / 55,103 | 9,680 / 8,780 (k=2e-4) | 593 / 558 | 538 / 507 | 533 / 503 |
| 4 | 85,176 / 82,670 | 29,593 / 26,295 (k=1e-4) | 603 / 565 | 521 / 488 | **516 / 484** |

* Modified-KN discounts for the highest order: D1/D2/D3+ = 0.758/1.191/1.428 (n=2), 0.874/1.298/1.409 (n=3), 0.931/1.406/1.428 (n=4).
* At n = 1, all methods except Laplace reduce to the maximum-likelihood unigram. The vocabulary has no count-1 words, so count-of-counts give no discount estimate.

Plot: `results/phase2/plots/ngram_perplexity.png`.

### 3.3 Subword LM vs word LM (test, per-word perplexity)

| Model | All tokens | Tokens whose word is in the LM vocabulary (94.5%) | OOV tokens only |
|---|---:|---:|---:|
| word mKN 4-gram (OOV predicted as `<UNK>`) | 484.2 | 563.6 | (only `<UNK>`) |
| subword mKN, n=3 pieces | 1,262 | 586.4 | 3.0e9 |
| subword mKN, n=4 pieces | 1,144 | 533.6 | 2.5e9 |
| subword mKN, n=5 pieces | **1,121** | **523.0** | 2.5e9 (≈ 31.2 bits per OOV word) |

Per-piece perplexity of the subword model at n = 5 is 200.7, with 1.32 pieces per word incl. `</s>`. It is not comparable with the word models (see section 2.3).

### 3.4 Learning curve (mKN trigram, test)

| Train share | Sentences | Tokens | (a) own vocab V | (a) `<UNK>` rate | (a) PPL | (b) fixed vocab PPL (`<UNK>` 4.96%) |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 89,371 | 768,691 | 41,410 | 13.0% | 511.5 | 1,400.1 |
| 25% | 223,428 | 2,030,557 | 74,452 | 9.0% | 555.8 | 931.6 |
| 50% | 446,855 | 4,124,580 | 113,700 | 6.9% | 532.6 | 679.0 |
| 100% | 893,710 | 8,461,056 | 192,363 | 5.0% | 502.6 | 502.6 |

Plot: `results/phase2/plots/learning_curve.png`.

## 4. Observations and inferences

1. **Konkani words are many and long-tailed, so a word vocabulary leaks.** Even with 8.46M training tokens, 30.3% of test word *types* and 4.1% of test tokens never occur in train. With the LM threshold (count ≥ 2), 5.5% of tokens are OOV. Konkani attaches case markers, postpositions and clitics to the stem (e.g. `भुरग्यांक`, `आपल्या`, `गांवच्या`), so one lemma shows up as many rare surface forms. Tripling the data (2.76M → 8.46M tokens) only cut the OOV token rate from 7.3% to 4.1%. *Inference:* a closed word vocabulary will always throw away part of the text in this language. Subword units are needed for any model that has to produce or score every word (FastText embeddings, neural LMs, NER).

2. **SentencePiece removes OOV completely at a cost of 1.25-1.54 pieces per word.** With `character_coverage=1.0` the unknown-piece rate is 0 on dev and test. Fertility drops as the vocabulary grows: about 1.54 (8k), 1.36 (16k) and 1.25 (32k). At 32k, about 81% of word tokens stay whole and only rarer, longer forms are split. *Inference:* 16k-32k is the useful range: 32k gives the shortest sequences, 16k better-trained piece embeddings.

3. **BPE and Unigram are almost tied.** Unigram has slightly lower fertility at 8k (1.513 vs 1.536) and 16k (1.359 vs 1.368); at 32k they are equal within 0.3% (1.252 vs 1.255). *Inference:* fertility alone does not choose between them. Unigram is the better default for the LM because its segmentation is probabilistic (subword regularisation). **Best tokeniser on these metrics: BPE-32k**, with Unigram-32k within 0.3%.

4. **Normalisation is not a neutral default.** SentencePiece's default `nmt_nfkc` would delete the ZWNJ in 702 train types and sometimes split a word in two. Plain NFKC changes nothing on our NFC text. *Inference:* for Indic scripts, check normalisation on the data; `identity` is the safe choice here.

5. **Laplace gets worse as n grows; KN gets better.** Laplace test perplexity rises from 3,352 (n=1) to 82,670 (n=4), while modified KN falls from 3,333 to 484. Add-1 gives each of the V′ = 192,362 possible next words a pseudo-count of 1, so a context seen c times keeps only c/(c+192,362) of its mass for what was observed. *Inference:* add-one is unusable at this vocabulary size; the KN trigram (503) is 110× lower than the Laplace trigram (55,103).

6. **Tuning k helps but cannot fix additive smoothing.** The best k falls with n (0.001 for bigrams, 1e-4 for 4-grams), yet add-k still gets worse beyond the bigram (1,371 → 8,780 → 26,295). It has no backoff: a word unseen after a known context gets the same tiny probability whether it is common or rare. *Inference:* information from lower orders (backoff or interpolation) is what makes higher orders usable.

7. **Kneser-Ney beats Katz/Good-Turing, and modified KN beats plain KN.** At every n ≥ 2 the order is mKN < KN < Katz (trigram 503 < 507 < 558). Katz gets worse from n=3 to n=4 (558 → 565) while KN still improves (503 → 484). KN's lower orders use continuation counts (how many different words a word follows), so a word frequent in one fixed phrase gets little backoff mass. Modified KN's three discounts (0.93/1.41/1.43 at n=4) gain a further 1%. *Inference:* modified KN is the n-gram baseline for the Phase 3 neural LMs.

8. **Most of the gain from context comes from the bigram, but larger data makes the 4-gram pay off.** Modified KN goes 3,333 → 635 (bigram, −81%) → 503 (trigram, −21%) → 484 (4-gram, −4%). On the 2.76M-token draft the 4-gram gained only 2%; with 3× more data more 4-gram histories are seen in training. *Inference:* longer contexts need more data, or a neural model that shares statistics across similar histories.

9. **The word model's 484 is optimistic, because `<UNK>` is a cheap prediction.** On test, 5.0% of tokens are `<UNK>`, and predicting "some rare word" is far easier than predicting the word itself. Restricted to tokens whose word is in the vocabulary, the word 4-gram scores 563.6, while the subword 5-gram scores **523.0** on the same tokens: better, with a 16k vocabulary instead of 192k, because it shares statistics across inflected forms. On all tokens the subword model scores 1,121, because it pays the real cost of spelling every OOV word (about 31 bits each). *Inference:* compare models with different vocabularies only on the same events.

10. **More data still helps; the curve has not flattened.** With the vocabulary fixed (b), each doubling of training data cuts trigram perplexity strongly: 1,400 → 932 (−33%) → 679 (−27%) → 503 (−26%). There is no sign of saturation at 8.46M tokens. *Inference:* more text is still the most effective improvement for these models, which justifies collecting all 573 newspaper issues.

11. **Comparing perplexity across vocabularies can mislead.** Under (a), perplexity barely moves with more data (512 → 556 → 533 → 503): small subsets have small vocabularies (41,410 words at 10%), so 13% of test tokens collapse into one cheap `<UNK>`. *Inference:* only a fixed evaluation vocabulary (or open-vocabulary per-word perplexity) shows the true benefit of more data.

## 5. Limitations and notes

* The learning-curve subsets are the *first* X% of train sentences. Train is ordered by document, so the subsets are not domain-balanced. The 10% subset has 9.1% of the tokens. A shuffled-document curve would be a cleaner FR-P2-7 measurement.
* Good-Turing is used inside Katz backoff with K = 5. Simple Good-Turing (Gale & Sampson's log-linear smoothing of n_r) is not implemented. The 0.5/c(h) floor for contexts without discountable counts is our own choice; without it, such contexts would give an unseen word probability 0.
* KenLM (mentioned in the PRD for 4/5-grams) is not used. Our own implementation reaches 4-grams on words and 5-grams on pieces within the memory budget. It is checked by the `--selftest` normalisation test, not against KenLM output.
* The subword LM is trained only for Unigram-16k. Other tokenisers can be passed with `--spm data/models/spm_<algo>_<size>.model`.
* Dev and test are close on the final splits (`<UNK>` 4.7% vs 5.0%; perplexities within about 7%), unlike the small draft snapshot.
