# Phase 3 (part A): neural language models

This document covers requirement FR-P3-1 (neural LMs: feed-forward, LSTM, small Transformer, compared by perplexity with the Phase 2 n-gram LMs) from [PRD.md](../PRD.md) section 7. It reuses the units and the evaluation of [phase2_lm.md](phase2_lm.md), so the numbers are directly comparable.

> **Data snapshot.** All numbers below come from the splits in `data/splits/` as of 21 Nov 2026 (08:54): train 501,782 sentences / 4,774,124 word tokens, dev 69,390 / 640,177, test 66,240 / 643,969. These are **larger than the splits used in `docs/phase2_lm.md`** (2.76M train tokens), so the Phase 2 tables there are not the baseline here. The n-gram baselines were re-run on the current splits with the unchanged Phase 2 script (`results/phase3/ngram_baseline/`). The splits will be regenerated with more news data. Re-run `bash scripts/run_phase3_lm.sh` and update the tables from `results/phase3/*.csv`.

## 1. How to run

```bash
bash scripts/run_phase3_lm.sh                     # .venv-phase2/bin/python; override with PYTHON=...
SPLITS=/abs/path/to/splits bash scripts/run_phase3_lm.sh
bash scripts/run_phase3_lm.sh --models lstm transformer --budget lstm=25 transformer=25 \
    --eval-every 300 --out results/phase3/extended --models-dir data/models/extended   # the longer run of §3.2
```

The script installs the PyTorch CPU wheel into `.venv-phase2` if it is missing (`pip`, or `uv pip` for uv-made venvs without pip). It then runs three steps:

| Step | Script | Outputs |
|---|---|---|
| SentencePiece Unigram-16k (only if missing) | `scripts/phase3/neural_lm.py --prepare-only` | `data/models/spm_unigram_16000.model` (git-ignored); same settings as `scripts/phase2/tokenizers.py` |
| n-gram baselines on the same splits | `scripts/phase2/ngram_lm.py --skip curve` | `results/phase3/ngram_baseline/{ngram,subword}_perplexity.csv`, `plots/ngram_perplexity.png` |
| Neural LMs | `scripts/phase3/neural_lm.py` | `results/phase3/neural_perplexity.csv`, `training_log.csv`, `comparison.csv`, `samples.md`, `plots/loss_curves.png`, `plots/comparison.png`; checkpoints `data/models/phase3_*.pt` (git-ignored) |

**Runtime (default run).** 23 min end to end on the team laptop (AMD Ryzen 7 8845HS, 8 cores, no GPU; `torch.cuda.is_available()` is False): SentencePiece 35 s, n-gram baselines 54 s (peak 1.3 GB), neural LMs 21.6 min (peak 3.0 GB). The neural step is time-boxed per model: FFNN 2.5 min, LSTM 7 min, Transformer 7 min, char LSTM 3.5 min of training, plus about 0.3 min of test scoring each. Other jobs were running on the machine at the same time (load average 10-14), so a free machine trains further within the same budget. The extended run (§3.2) took 51 min more (peak 2.9 GB). Seed: 2026 (`--seed`). All options: `--help`.

## 2. Method

### 2.1 Units and events (same as Phase 2)

* Text: every line of `data/splits/*.txt`, tokenised with `corpus_utils.tokenize` (punctuation removed, numbers become `<NUM>`) and loaded with `scripts/phase2/common.py`. Lines without a word token are skipped.
* Subword models use the Phase 2 **SentencePiece Unigram-16k** model (identity normalisation, `<NUM>` as one piece). The char model uses the 85 symbols of the train text (characters, the space, and `<NUM>` as one private-use symbol) plus `<unk>`, `<s>`, `</s>`: 88 units. No unknown character occurs in dev or test. Words are separated by one space, and `<NUM>` counts as one character.
* **Every sentence is modelled on its own**, exactly like the n-gram LMs: input `<s> u1 … um`, targets `u1 … um </s>`. No context crosses a sentence boundary, so the neural models do not get extra information the n-gram models lack.
* Metrics on the full test split:
  * **per-word perplexity** = 2^(−Σ log2 P / (word tokens + sentences)), the normalisation of the Phase 2 subword LM. This is the number to compare across units (pieces, characters, n-gram or neural).
  * **per-unit perplexity** (per piece, or per character for the char model). Only comparable between models with the same units.
  * **bits per character (bpc)** = −Σ log2 P / characters, where characters = word characters + one space between words + one end-of-sentence per sentence (4,189,690 on test). For the n-gram rows it is derived from their per-word perplexity with the same denominators.
  * **in-vocabulary** per-word perplexity: only tokens whose word has train count ≥ 2 (plus `</s>`), the events of the Phase 2 word LM, where OOV tokens become `<UNK>`. This lets the open-vocabulary models be compared with the word 4-gram on the same events. Each word token owns its pieces (and, in the char model, its leading space).

### 2.2 Models

| Model | Architecture | Parameters |
|---|---|---:|
| `ffnn` (Bengio et al. 2003) | embeddings (128) of the previous 3 pieces (like a 4-gram), concatenated → tanh 512 → 256 → softmax | 4.1M |
| `lstm` | embedding 256 → 1-layer LSTM 512, dropout 0.2 → softmax | 9.2M |
| `transformer` | 4 pre-LN decoder layers, d = 256, 4 heads, FF 1024 (GELU), learned positions, context 128 pieces, dropout 0.1 → softmax | 9.0M |
| `char_lstm` | the LSTM above over characters, plain softmax over 88 | 1.6M |

* **Output layer.** The subword models end in an **adaptive softmax** (Grave et al. 2017; `nn.AdaptiveLogSoftmaxWithLoss`): the 2,000 most frequent pieces (by train frequency) form the head, pieces 2,000-6,000 and 6,000-16,000 two tail clusters with 2× and 4× smaller projections. It is an exact, normalised distribution, so the perplexities are exact. On this CPU a full 16k softmax took about 0.9 s per 4,096-token batch, more than the rest of the network; the adaptive softmax makes it several times cheaper.
* **bfloat16.** The network body runs under bfloat16 autocast (2-4× faster on CPUs with AVX512-BF16); the output layer always runs in float32. `--no-bf16` turns it off.

### 2.3 Training and early stopping

* AdamW (weight decay 0.01), learning rate 2e-3 (Transformer 1e-3), linear warm-up over 200 steps, gradient clipping at 1.0.
* Sentences are bucketed by length into batches of about 4,096 target units. Sentences longer than 128 pieces (256 characters) are cut into chunks for training only. At test time the LSTMs read whole sentences, and the Transformer scores longer sentences with a sliding window (stride 64) so every piece has at least 64 pieces of context.
* **Early stopping on dev.** Every 200 steps (300 in the extended run) the model is scored on a fixed random sample of 3,000 dev sentences (seed 2026). If dev per-word perplexity did not improve by at least 0.1%, the learning rate is halved; after 3 such evaluations in a row training stops. Training also stops at the model's time budget or after 5 epochs. The best dev checkpoint is the one tested.
* All train sentences are used (`--max-train-sentences 0`); the time budget is what limits training.

## 3. Results (test split)

### 3.1 Default run (about 23 min in total)

| Model | Unit | Training | Epochs | Per-word PPL | Per-unit PPL | bpc | In-vocab per-word PPL |
|---|---|---|---:|---:|---:|---:|---:|
| mKN 3-gram | Unigram-16k pieces | counts, 54 s for all n | – | 2,895 | 324.1 | 1.949 | 778.5 |
| mKN 4-gram | Unigram-16k pieces | | – | 2,691 | 307.4 | 1.932 | 727.1 |
| **mKN 5-gram** | Unigram-16k pieces | | – | **2,657** | **304.5** | **1.928** | **717.9** |
| mKN 4-gram | words (count ≥ 2, OOV → `<UNK>`) | | – | (553.9, not comparable) | – | – | 765.7 |
| FFNN | Unigram-16k pieces | 2.5 min | 0.56 | 10,352 | 816.6 | 2.261 | 2,760 |
| LSTM | Unigram-16k pieces | 7 min | 0.86 | 4,953 | 478.4 | 2.081 | 1,391 |
| Transformer | Unigram-16k pieces | 7 min | 0.49 | 7,286 | 633.0 | 2.175 | 2,036 |
| char LSTM | characters | 3.5 min | 0.23 | 18,352 | 5.28 per char | 2.401 | 5,691 |

All four neural models stopped at their time budget, not by early stopping: dev perplexity was still falling at every evaluation (`results/phase3/plots/loss_curves.png`).

### 3.2 Extended run (25 min per model)

| Model | Training | Epochs | Steps | Per-word PPL | Per-piece PPL | bpc | In-vocab per-word PPL |
|---|---|---:|---:|---:|---:|---:|---:|
| mKN 4-gram (pieces) | – | – | – | 2,691 | 307.4 | 1.932 | 727.1 |
| mKN 5-gram (pieces) | – | – | – | **2,657** | **304.5** | **1.928** | **717.9** |
| LSTM | 25 min | 3.13 | 5,375 | 2,739 | 311.3 | 1.936 | 775.8 |
| Transformer | 25 min | 1.77 | 3,037 | 2,868 | 321.9 | 1.947 | 811.4 |

From `results/phase3/extended/` (same code; `--models lstm transformer --budget lstm=25 transformer=25 --eval-every 300`).

Samples (temperature 0.8, top-k 40): `results/phase3/samples.md` and `results/phase3/extended/samples.md`.

## 4. Observations and inferences

1. **With the default 30-minute budget, every neural model is worse than modified Kneser-Ney.** The best one, the LSTM, reaches 4,953 per word (478 per piece) against 2,657 (305) for the mKN 5-gram. None of them finished one pass over the 7.0M training pieces, and their dev perplexity was still falling at every evaluation. The mKN model is built from exact counts of all the data in under a minute. *Inference:* on a laptop CPU the comparison in the default run is between a fully trained n-gram model and neural models that have seen half an epoch. It shows the cost of neural training, not the limit of the architecture.

2. **With 25 minutes of training, the LSTM reaches the n-gram level.** After 3.1 epochs it scores 2,739 per word: better than the mKN trigram (2,895), 1.8% behind the 4-gram (2,691) and 3.1% behind the 5-gram (2,657). In bits per character the gap is 0.008 (1.936 vs 1.928). The Transformer (1.8 epochs) scores 2,868, also better than the trigram. *Inference:* the neural models catch up with the best count-based model within one hour of CPU time. With more training (neither run reached early stopping or a learning-rate halving), they should pass it, as they do on English benchmarks.

3. **Neural models are better on unseen words and worse on common ones.** On test tokens whose word is not in the n-gram vocabulary, the extended LSTM spends 30.6 bits per word, the mKN 5-gram 31.3 bits (per-word perplexity 1.6e9 vs 2.6e9). On in-vocabulary tokens the n-gram is still better (718 vs 776). The LSTM's piece embeddings are shared across every word that contains a piece, so it spells rare and new inflected forms better. The n-gram model has exact counts for frequent phrases, which a 9M-parameter model after 3 epochs has not memorised. *Inference:* for an agglutinative language with 7-9% OOV tokens, the open-vocabulary neural model already does the hard part better. Its remaining gap is on frequent, memorisable events, which is the part that more training fixes.

4. **The Transformer learns more per token, the LSTM more per second.** At the same number of training pieces (12.2-12.4M), dev per-word perplexity is 2,422 for the Transformer and 2,868 for the LSTM. But one Transformer step took about 0.49 s against 0.28 s for the LSTM on this CPU, so in a fixed time the LSTM sees 1.8× more data and ends ahead (2,739 vs 2,868 on test). The same happens in the default run (Transformer 0.49 epochs, LSTM 0.86 epochs). *Inference:* with a GPU or a longer budget, the Transformer is the better choice; under a strict CPU time box the LSTM is.

5. **The feed-forward model is the weakest neural model, even per token.** With a 3-piece history (the context of a 4-gram), the FFNN reaches 10,352 per word after 0.56 epochs; the LSTM had 8,401 on dev at a similar number of pieces (2.4M) and the FFNN 13,448. A fixed window cannot use the rest of the sentence. Konkani case markers and verb agreement often depend on words more than three pieces back, and one word is already 1.38 pieces on average. *Inference:* the FFNN reproduces the n-gram's Markov limit with a smoother estimate. Recurrence or attention over the whole sentence is what the later models add.

6. **The character LSTM is far behind per word, but close in bits per character.** It scores 2.40 bpc (18,352 per word) after 0.23 epochs, against 2.08 for the subword LSTM after 7 min. A character model has to predict 4.3 times more units than the piece model (4.19M vs 0.98M on test) through a recurrence, so in the same time it sees much less text. Its samples, however, are mostly well-formed Konkani words (`results/phase3/samples.md`). *Inference:* character-level modelling is viable for spelling, but at this compute it is not competitive as a language model; subword units are the better trade-off.

7. **Per-piece numbers alone mislead in the same way as in Phase 2.** The char LSTM's 5.28 per character looks far better than any per-piece number, and the word 4-gram's 554 far better than the 2,657 of the subword 5-gram. Neither is comparable: they predict different events. Only per-word perplexity (or bpc) over the same text, and in-vocabulary perplexity over the same tokens, put all models on one scale.

8. **The samples show the news domain and the numbers.** Most samples are short news phrases with dates, places and `<NUM>` tokens (`<NUM> मे दिसा`, `पणजी`, `जिल्हो पंचायत`, cricket scores). The corpus is mostly news, and `<NUM>` is one of the most frequent tokens. The extended LSTM produces locally fluent clauses (`पणजी गोंयांत हो अपघात बुधवारा <NUM> मे दिसा मध्यान रातीं …`), the 7-min FFNN mostly repeats `<NUM>`. *Inference:* the coming news-heavy splits will lower perplexity on news, but generation quality for other domains needs balanced data (`data/splits/balanced/`).

## 5. Limitations and notes

* **Numbers are from the current splits** (21 Nov 2026, 4.77M train tokens). The splits are about to be regenerated with more news text; all tables must be refreshed from the CSVs after re-running.
* **Compute, not data, limits the neural models here.** No model reached early stopping, and no hyper-parameters were tuned (one setting per model, chosen for CPU speed). Learning rate, dropout, depth and the adaptive-softmax cut-offs are all untuned, so the neural numbers are lower bounds on what these architectures can do on this corpus.
* The timings were measured while other jobs shared the CPU, so the default budgets give somewhat different amounts of training on another machine. That changes the numbers (the budget is in minutes, not steps); use `--budget` or `--max-epochs` for exact reproduction of a given amount of training.
* bfloat16 autocast changes the network's arithmetic, but the output distribution is computed in float32 and is normalised, so the reported perplexities are exact for the model that was trained.
* Dev early stopping uses a 3,000-sentence dev sample to keep evaluation cheap; the full dev split is scored only with `--final-splits dev test`.
* The FFNN pads the history with `<s>` at the start of a sentence; the n-gram LMs use a shorter history instead. Both see only the current sentence.
