# Phase 3 (part A): neural language models

This document covers requirement FR-P3-1 (neural LMs: feed-forward, LSTM, small Transformer, compared by perplexity with the Phase 2 n-gram LMs) from [PRD.md](../PRD.md) section 7. It reuses the units and the evaluation of [phase2_lm.md](phase2_lm.md), so the numbers are directly comparable.

> **Data.** All numbers come from the final corpus splits (`corpus-v1.0`, 22 Nov 2026): train 893,710 sentences / 8,461,056 word tokens, dev 106,823 / 1,008,663, test 109,907 / 1,036,051, the same splits as [phase2_lm.md](phase2_lm.md). The n-gram baselines were re-run on these splits (`results/phase3/ngram_baseline/`). An earlier draft used a 4.77M-token snapshot; the method is unchanged.

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

Neural LM checkpoints are written atomically with a format version, model name,
and vocabulary size. The loader accepts older plain state-dict checkpoints and
rejects versioned checkpoints with a mismatched model or vocabulary. Training
fails explicitly for empty train/dev splits and invalid positive settings.

**Runtime.** Default run 22 min end to end on the team laptop (AMD Ryzen 7 8845HS, 8 cores, no GPU). The neural step is time-boxed per model: FFNN 2.5 min, LSTM 7 min, Transformer 7 min, char LSTM 3.5 min of training, plus test scoring. The extended run (§3.2) took about 52 min more. Other jobs (embedding training) shared the CPU during the default run, so a free machine trains further within the same budget. Seed: 2026 (`--seed`). All options: `--help`.

## 2. Method

### 2.1 Units and events (same as Phase 2)

* Text: every line of `data/splits/*.txt`, tokenised with `corpus_utils.tokenize` (punctuation removed, numbers become `<NUM>`) and loaded with `scripts/phase2/common.py`. Lines without a word token are skipped.
* Subword models use the Phase 2 **SentencePiece Unigram-16k** model (identity normalisation, `<NUM>` as one piece). The char model uses the 85 symbols of the train text (characters, the space, and `<NUM>` as one private-use symbol) plus `<unk>`, `<s>`, `</s>`: 88 units. No unknown character occurs in dev or test. Words are separated by one space, and `<NUM>` counts as one character.
* **Every sentence is modelled on its own**, exactly like the n-gram LMs: input `<s> u1 … um`, targets `u1 … um </s>`. No context crosses a sentence boundary, so the neural models do not get extra information the n-gram models lack.
* Metrics on the full test split:
  * **per-word perplexity** = 2^(−Σ log2 P / (word tokens + sentences)), the normalisation of the Phase 2 subword LM. This is the number to compare across units (pieces, characters, n-gram or neural).
  * **per-unit perplexity** (per piece, or per character for the char model). Only comparable between models with the same units.
  * **bits per character (bpc)** = −Σ log2 P / characters, where characters = word characters + one space between words + one end-of-sentence per sentence (6,734,582 on test). For the n-gram rows it is derived from their per-word perplexity with the same denominators.
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

### 3.1 Default run (about 22 min in total)

| Model | Unit | Training | Epochs | Per-word PPL | Per-unit PPL | bpc | In-vocab per-word PPL |
|---|---|---|---:|---:|---:|---:|---:|
| mKN 3-gram | Unigram-16k pieces | counts, about 1 min for all n | – | 1,262 | 219.4 | 1.753 | 586.4 |
| mKN 4-gram | Unigram-16k pieces | | – | 1,144 | 203.8 | 1.729 | 533.6 |
| **mKN 5-gram** | Unigram-16k pieces | | – | **1,121** | **200.7** | **1.724** | **523.0** |
| mKN 4-gram | words (count ≥ 2, OOV → `<UNK>`) | | – | (484.2, not comparable) | – | – | 563.6 |
| FFNN | Unigram-16k pieces | 2.5 min | 0.37 | 5,255 | 644.2 | 2.103 | 2,392 |
| LSTM | Unigram-16k pieces | 7 min | 0.57 | 2,620 | 380.9 | 1.932 | 1,212 |
| Transformer | Unigram-16k pieces | 7 min | 0.33 | 3,405 | 464.2 | 1.997 | 1,576 |
| char LSTM | characters | 3.5 min | 0.16 | 10,223 | 4.81 per char | 2.267 | 5,026 |

All four neural models stopped at their time budget, not by early stopping: dev perplexity was still falling at every evaluation (`results/phase3/plots/loss_curves.png`).

### 3.2 Extended run (25 min per model)

| Model | Training | Epochs | Steps | Per-word PPL | Per-piece PPL | bpc | In-vocab per-word PPL |
|---|---|---:|---:|---:|---:|---:|---:|
| mKN 4-gram (pieces) | – | – | – | 1,144 | 203.8 | 1.729 | 533.6 |
| mKN 5-gram (pieces) | – | – | – | **1,121** | **200.7** | **1.724** | **523.0** |
| LSTM | 25 min | 2.18 | 6,588 | 1,389 | 235.9 | 1.776 | 650.8 |
| Transformer | 25 min | 1.03 | 3,115 | 1,587 | 260.9 | 1.809 | 737.7 |

From `results/phase3/extended/` (same code; `--models lstm transformer --budget lstm=25 transformer=25 --eval-every 300`). Neither model reached early stopping or a learning-rate halving; dev perplexity was still falling.

Samples (temperature 0.8, top-k 40): `results/phase3/samples.md` and `results/phase3/extended/samples.md`.

## 4. Observations and inferences

1. **With the default budget, every neural model is far worse than modified Kneser-Ney.** The best, the LSTM, reaches 2,620 per word (381 per piece) against 1,121 (201) for the mKN 5-gram, after seeing just over half of the 12.3M training pieces once. The mKN model is built from exact counts of all the data in about a minute. *Inference:* on a laptop CPU, the default run compares a fully trained n-gram model with neural models that have not finished one epoch; it shows the cost of neural training, not the limit of the architectures.

2. **With 25 minutes, the LSTM halves its perplexity but stays 24% behind the 5-gram.** After 2.2 epochs it scores 1,389 per word (1.776 bpc) against 1,121 (1.724 bpc) for the mKN 5-gram; the Transformer (1.0 epoch) scores 1,587. On the earlier 4.77M-token draft the 25-minute LSTM was within 3% of the 5-gram. *Inference:* more data helped the n-gram model immediately (2,657 → 1,121) because counting uses all of it at once, while a neural model needs proportionally more training time to use 1.8× more text. The neural models are now compute-bound: their dev curves were still falling, so longer training or a GPU is required to catch up.

3. **The n-gram model now wins on both frequent and unseen words.** On in-vocabulary tokens the 5-gram scores 523 against the LSTM's 651. On tokens whose word is outside the n-gram vocabulary, the LSTM spends 31.4 bits per word and the 5-gram 31.2 (on the draft snapshot the LSTM was ahead here, 30.6 vs 31.3). *Inference:* the LSTM's shared piece embeddings were its advantage on rare inflected forms; with 8.5M tokens the n-gram has seen enough piece sequences to close that gap, and the under-trained LSTM has not yet caught up.

4. **The Transformer learns more per token, the LSTM more per second.** After the same 12.3M training pieces, dev per-word perplexity is 1,577 for the Transformer and 1,859 for the LSTM. But one Transformer step took about 0.48 s against 0.23 s for the LSTM on this CPU, so in 25 minutes the LSTM saw 2.1× more data and ends ahead on test (1,389 vs 1,587). *Inference:* with a GPU or a longer budget the Transformer is the better choice; under a strict CPU time box the LSTM is.

5. **The feed-forward model is the weakest neural model.** With a 3-piece history (the context of a 4-gram) the FFNN reaches 5,255 per word after 0.37 epochs. A fixed window cannot use the rest of the sentence, while Konkani case markers and verb agreement often depend on words more than three pieces back (one word is 1.32 pieces on average). *Inference:* the FFNN reproduces the n-gram's Markov limit with a smoother estimate; recurrence or attention over the whole sentence is what the later models add.

6. **The character LSTM is far behind per word, but closer in bits per character.** It scores 2.27 bpc (10,223 per word) after 0.16 epochs, against 1.93 for the subword LSTM after 7 minutes. A character model predicts 4.4× more units than the piece model (6.73M vs 1.52M on test), so in the same time it sees much less text. Its samples are still mostly well-formed Konkani words. *Inference:* character-level modelling suits spelling tasks, but at this compute subword units are the better trade-off for language modelling.

7. **Per-unit numbers alone mislead in the same way as in Phase 2.** The char LSTM's 4.81 per character and the word 4-gram's 484 both look far better than the subword 5-gram's 1,121, but they predict different events. Only per-word perplexity (or bpc) over the same text, and in-vocabulary perplexity over the same tokens, put all models on one scale.

8. **The samples show the news domain and the numbers.** Most samples are news phrases with dates, places, officials and `<NUM>` tokens, e.g. the extended LSTM's `हे विशीं उलयतना मुख्यमंत्री डॉ प्रमोद सावंत हांणी सांगलें` ("speaking about this, Chief Minister Dr Pramod Sawant said"). The corpus is 86% news and `<NUM>` is one of the most frequent tokens. *Inference:* generation in other domains needs the balanced splits (`data/splits/balanced/`).

## 5. Limitations and notes

* **Compute, not data, limits the neural models here.** No model reached early stopping, and no hyper-parameters were tuned (one setting per model, chosen for CPU speed). Learning rate, dropout, depth and the adaptive-softmax cut-offs are all untuned, so the neural numbers are lower bounds on what these architectures can do on this corpus.
* The timings were measured while other jobs shared the CPU, so the default budgets give somewhat different amounts of training on another machine. That changes the numbers (the budget is in minutes, not steps); use `--budget` or `--max-epochs` for exact reproduction of a given amount of training.
* bfloat16 autocast changes the network's arithmetic, but the output distribution is computed in float32 and is normalised, so the reported perplexities are exact for the model that was trained.
* Dev early stopping uses a 3,000-sentence dev sample to keep evaluation cheap; the full dev split is scored only with `--final-splits dev test`.
* The FFNN pads the history with `<s>` at the start of a sentence; the n-gram LMs use a shorter history instead. Both see only the current sentence.
