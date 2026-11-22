# Phase 2 (part B): word embeddings

This document covers FR-P2-3 (word embeddings) and the embedding half of FR-P2-7 (learning curve) from [PRD.md](../PRD.md). Tokenisers and n-gram LMs are in [phase2_lm.md](phase2_lm.md).

> **Data.** Final corpus splits (`corpus-v1.0`, 22 Nov 2026): 8,461,056 training word tokens. Evaluation sets were drafted by Claude (AI) and **still need team verification**: `data/eval/word_similarity.tsv` (213 rated pairs) and `data/eval/analogies.tsv` (118 analogies in 13 categories).

## 1. How to run

```bash
bash scripts/run_phase2_embeddings.sh      # about 25 min on the team laptop (CPU)
```

| Step | Script | Outputs |
|---|---|---|
| Train | `scripts/phase2/train_embeddings.py` | `data/models/{cbow,sg,ft,ft25,glove}.kv` (git-ignored); learning-curve models in `data/models/lc/` |
| Evaluate | `scripts/phase2/eval_embeddings.py` | `results/phase2/embedding_eval.csv`, `similarity_pairs.csv`, `analogy_by_category.csv`, `oov_robustness.csv`, `embedding_learning_curve.csv`, `eval_set_coverage.csv`, `nearest_neighbours.md`, `plots/tsne_*.png`, `plots/embedding_learning_curve.png` |

Training time on the full train split: CBOW 43 s, skip-gram 82 s, FastText 145 s, FastText 2–5 149 s, GloVe 729 s.

## 2. Method

* **Input.** Train sentences tokenised with `corpus_utils.tokenize` (punctuation removed, numbers → `<NUM>`), cached once and streamed by gensim (`corpus_file=`).
* **Models** (all 100 dimensions, window 5, minimum count 3, 10 epochs, seed 42):

| Name | Model |
|---|---|
| `cbow` | word2vec CBOW, negative sampling (5), subsampling 1e-4 |
| `sg` | word2vec skip-gram, same settings |
| `ft` | FastText skip-gram, character n-grams of 3–6 code points, 500k hash buckets |
| `ft25` | FastText skip-gram, character n-grams of 2–5 code points (one akshara is often 2–3 code points) |
| `glove` | GloVe, own numpy implementation: top 30,000 words, window 10 with 1/distance weighting, f(x) = (x/x_max)^0.75, AdaGrad, 15 epochs |

* **Word similarity.** Spearman ρ between cosine similarity and the 213 rated pairs, with a bootstrap 95% interval; also split into low- and high-frequency pairs, and mean cosine per relation type.
* **Analogies.** 3CosAdd (answer = nearest word to b − a + c, question words excluded), top-1 accuracy.
* **OOV robustness.** 803 frequent words are damaged the way OCR damages them (anusvara added or dropped, vowel sign dropped, virama dropped, i/u length swapped). We measure the cosine between the damaged and the original word, and whether the original is recovered among the damaged word's nearest neighbours.
* **Learning curve.** `sg` and `ft` retrained on nested 10/25/50% subsets of the training sentences.

## 3. Results

### 3.1 Similarity and analogies

| Model | Vocabulary | Spearman ρ (95% CI) | Analogies correct (of 118) |
|---|---:|---|---:|
| CBOW | 130,861 | 0.350 (0.23–0.47) | 38 (32.2%) |
| skip-gram | 130,861 | 0.439 (0.32–0.56) | 45 (38.1%) |
| **FastText 3–6** | 130,861 | **0.466 (0.36–0.58)** | 44 (37.3%) |
| **FastText 2–5** | 130,861 | 0.447 (0.34–0.56) | **50 (42.4%)** |
| GloVe | 30,000 | 0.236 (0.10–0.38) | 25 (21.2%) |

### 3.2 Analogies by category (correct / total)

| Category | CBOW | SG | FT 3–6 | FT 2–5 | GloVe |
|---|---:|---:|---:|---:|---:|
| gender (nouns) | 5/10 | 6/10 | 4/10 | 3/10 | 2/10 |
| gender (verbs, adjectives) | 5/10 | 6/10 | 6/10 | 8/10 | 4/10 |
| singular → plural | 3/10 | 3/10 | 2/10 | 2/10 | 2/10 |
| locative `-ांत` | 2/10 | 2/10 | 3/10 | 4/10 | 1/10 |
| dative `-ाक` | 3/10 | 1/10 | 6/10 | 6/10 | 0/10 |
| present singular → plural | 9/10 | 9/10 | 10/10 | 9/10 | 7/10 |
| verbal noun → purposive | 6/10 | 8/10 | 9/10 | 10/10 | 5/10 |
| capital : state / country | 2/10 | 2/10 | 1/10 | 1/10 | 0/10 |
| Goa town : taluka | 2/10 | 6/10 | 2/10 | 2/10 | 0/10 |
| cardinal → ordinal | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 |
| number / weekday / month successor | 1/18 | 1/18 | 1/18 | 5/18 | 4/18 |

### 3.3 Robustness to OCR-style damage (803 damaged words)

| Model | Damaged words with a vector | Mean cosine to the original | Original in top 10 neighbours |
|---|---:|---:|---:|
| CBOW | 422 | 0.41 | 10.7% |
| skip-gram | 422 | 0.54 | 12.7% |
| FastText 3–6 | **803** | 0.66 | 33.0% |
| FastText 2–5 | **803** | **0.73** | **37.9%** |
| GloVe | 212 | 0.31 | 3.5% |

### 3.4 Learning curve (test of similarity / analogies)

| Training tokens | SG ρ | SG analogies | FT ρ | FT analogies |
|---:|---:|---:|---:|---:|
| 0.84M (10%) | 0.289 | 11.9% | 0.387 | 39.0% |
| 2.11M (25%) | 0.370 | 20.3% | 0.478 | 38.1% |
| 4.23M (50%) | 0.392 | 30.5% | 0.473 | 38.1% |
| 8.46M (100%) | 0.439 | 38.1% | 0.466 | 37.3% |

Nearest neighbours for 20 probe words: `results/phase2/nearest_neighbours.md`. t-SNE plots: `results/phase2/plots/tsne_*.png`.

## 4. Observations and inferences

1. **FastText is the best Konkani embedding.** It has the highest similarity correlation (0.466) and, with 2–5-character pieces, the most analogies (42%). Character n-grams let all inflected forms of a stem share information, and Konkani glues case endings onto words. *Inference:* use FastText vectors for downstream models (the NER BiLSTM-CRF in Phase 3).

2. **Morphology is learned well, world knowledge badly.** Grammatical analogies are often solved: present singular → plural 9–10/10, verbal noun → purposive up to 10/10, dative `-ाक` 6/10 with FastText. Factual ones fail: capital : state 0–2/10, cardinal → ordinal 0–1/10. *Inference:* 8.5M words of mostly news are enough to learn how Konkani words are built, not enough facts about the world; those relations need much more varied text.

3. **FastText survives OCR damage; word-level models do not.** Only 422 of 803 damaged words exist in the word2vec vocabulary at all, while FastText builds a vector for every one. FastText 2–5 keeps damaged words close to the original (mean cosine 0.73) and finds the original among the 10 nearest neighbours 38% of the time, against 11–13% for word2vec. *Inference:* subword embeddings are the right choice for an OCR-built corpus, and could drive an OCR post-corrector.

4. **Skip-gram beats CBOW; GloVe is weakest.** Skip-gram is better on rare words (low-frequency ρ 0.44 vs 0.30), as expected. Our GloVe keeps only the top 30,000 words and is trained with a simple implementation, so it lags (0.24). *Inference:* for a small, morphologically rich corpus, predictive models with negative sampling are the better baseline.

5. **Skip-gram still improves with data; FastText levels off early.** Skip-gram ρ rises steadily (0.29 → 0.37 → 0.39 → 0.44) and analogies triple (12% → 38%). FastText reaches about 0.47 already at 25% of the data, because character n-grams give it good vectors for rare words early. *Inference:* the word-level model is still data-limited; FastText is the better choice when data is scarce.

6. **The evaluation sets are small.** With 213 pairs, the 95% intervals overlap for SG, FT and FT 2–5 (about ±0.11). Only the gaps to CBOW and GloVe are clearly real. *Inference:* the similarity and analogy sets should be verified by the team and, if possible, enlarged before final claims.

## 5. Limitations

* The similarity ratings and analogies were drafted by Claude (AI), not by native speakers, and must be checked before the numbers are quoted as final.
* No hyper-parameters were tuned (one setting for all models).
* Pretrained multilingual embeddings (fastText `gom` vectors, IndicBERT) were not compared yet.
