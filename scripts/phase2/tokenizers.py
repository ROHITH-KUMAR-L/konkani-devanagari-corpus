"""Train and evaluate tokenisers for the Konkani corpus (Phase 2, FR-P2-1).

Tokenisers compared
  * word baseline: corpus_utils.tokenize (punctuation removed, numbers -> <NUM>);
    vocabulary = all train word types (and, for reference, types seen >= 2 times,
    which is the n-gram LM vocabulary)
  * SentencePiece BPE and Unigram with vocabulary sizes 8k, 16k, 32k

SentencePiece is trained on the same tokenised word stream as the baseline (words
of a sentence joined by single spaces, <NUM> kept as one user-defined symbol), so
fertility = subword pieces per baseline word token, measured on identical text.
Normalisation is "identity": Devanagari NFC text goes in unchanged. The script
measures how many word tokens the default nmt_nfkc rule (and plain nfkc) would
alter, to justify that choice.

Metrics per tokenizer and evaluation split (dev, test)
  fertility           pieces per word token
  split_share         share of word tokens cut into more than one piece
  unk_piece_rate      share of pieces that are <unk> (SentencePiece only)
  oov_token_rate      share of word tokens not in the word vocabulary (word rows)
  oov_type_rate       share of word types not in the word vocabulary (word rows)

Outputs
  data/models/spm_{bpe,unigram}_{8000,16000,32000}.{model,vocab}   (git-ignored)
  results/phase2/tokenizer_eval.csv
  results/phase2/tokenizer_normalisation.txt

Usage
  python scripts/phase2/tokenizers.py [--splits data/splits] [--models data/models]
      [--out results/phase2] [--sizes 8000 16000 32000] [--threads 4]
      [--max-sentences 500000] [--seed 13]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

from common import ROOT, Corpus

NUM = "<NUM>"


def train_spm(text_path, prefix, model_type, size, threads):
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(prefix),
        model_type=model_type,
        vocab_size=size,
        character_coverage=1.0,              # corpus is cleaned Devanagari: keep every character
        normalization_rule_name="identity",   # do not touch NFC Devanagari (see normalisation check)
        user_defined_symbols=[NUM],
        input_sentence_size=0,                # use the whole (already sampled) text file
        num_threads=threads,
        minloglevel=2,
    )


def normalisation_check(types, counts):
    """How many word tokens/types each SentencePiece normaliser would change."""
    lines = []
    ident = spm.SentencePieceNormalizer(rule_name="identity")
    for rule in ("nmt_nfkc", "nfkc"):
        norm = spm.SentencePieceNormalizer(rule_name=rule)
        changed_types, changed_tokens, examples = 0, 0, []
        for t, c in zip(types, counts):
            if c == 0:
                continue
            if norm.normalize(t) != ident.normalize(t):
                changed_types += 1
                changed_tokens += int(c)
                if len(examples) < 5:
                    examples.append(f"{t!r} -> {norm.normalize(t).lstrip(chr(0x2581))!r}")
        n_types, n_tok = int((counts > 0).sum()), int(counts.sum())
        lines.append(f"{rule}: alters {changed_types} of {n_types} train types "
                     f"({100 * changed_types / n_types:.3f}%), "
                     f"{changed_tokens} of {n_tok} tokens ({100 * changed_tokens / n_tok:.3f}%)")
        lines += [f"    e.g. {e}" for e in examples]
    return lines


def word_rows(corpus, split, vocab_mask, label):
    pids, _ = corpus.data[split]
    cnt = corpus.counts(split)
    seen = cnt > 0
    oov_tok = cnt[~vocab_mask & seen].sum() / cnt.sum()
    oov_type = (~vocab_mask & seen).sum() / seen.sum()
    return dict(tokenizer=label, algo="word", vocab_size=int(vocab_mask.sum()), split=split,
                word_tokens=int(cnt.sum()), pieces=int(cnt.sum()), fertility=1.0,
                split_share=0.0, unk_piece_rate="", oov_token_rate=round(float(oov_tok), 5),
                oov_type_rate=round(float(oov_type), 5))


def spm_rows(corpus, sp, label, algo, size):
    """Encode each word type once (pieces never cross spaces), weight by frequency."""
    rows = []
    unk = sp.unk_id()
    for split in ("dev", "test"):
        cnt = corpus.counts(split)
        idx = np.flatnonzero(cnt)
        enc = sp.encode([corpus.types[i] for i in idx], out_type=int)
        n_pieces = np.array([len(e) for e in enc])
        n_unk = np.array([e.count(unk) for e in enc])
        c = cnt[idx]
        rows.append(dict(tokenizer=label, algo=algo, vocab_size=size, split=split,
                         word_tokens=int(c.sum()), pieces=int((c * n_pieces).sum()),
                         fertility=round(float((c * n_pieces).sum() / c.sum()), 4),
                         split_share=round(float(c[n_pieces > 1].sum() / c.sum()), 4),
                         unk_piece_rate=round(float((c * n_unk).sum() / (c * n_pieces).sum()), 7),
                         oov_token_rate="", oov_type_rate=""))
    return rows


def check_type_cache(corpus, sp, n=500):
    """Assert that encoding whole sentences == concatenating per-word encodings."""
    for k, words in enumerate(corpus.sentences("test")):
        if k >= n:
            break
        whole = sp.encode(" ".join(words), out_type=int)
        parts = [p for w in words for p in sp.encode(w, out_type=int)]
        assert whole == parts, f"per-word encoding differs on sentence {k}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(ROOT / "data/splits"))
    ap.add_argument("--models", default=str(ROOT / "data/models"))
    ap.add_argument("--out", default=str(ROOT / "results/phase2"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 16000, 32000])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-sentences", type=int, default=500_000,
                    help="train SentencePiece on at most this many train sentences, "
                         "sampled with --seed (bounds trainer memory)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    models, out = Path(args.models), Path(args.out)
    models.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    corpus = Corpus(args.splits)
    print(f"loaded splits in {time.time() - t0:.0f}s; {len(corpus.types)} word types overall")

    # SentencePiece input: the tokenised train sentences, one per line
    text = models / "spm_train_text.txt"
    with open(text, "w", encoding="utf-8", newline="\n") as f:
        n_sent = len(corpus.data["train"][1])
        keep = np.ones(n_sent, bool)
        if n_sent > args.max_sentences:
            keep[:] = False
            keep[np.random.default_rng(args.seed).choice(n_sent, args.max_sentences, replace=False)] = True
        for k, words in zip(keep, corpus.sentences("train")):
            if k:
                f.write(" ".join(words) + "\n")
    print(f"SentencePiece training text: {int(keep.sum())} of {n_sent} train sentences")

    train_cnt = corpus.counts("train")
    norm_lines = normalisation_check(corpus.types, train_cnt)
    (out / "tokenizer_normalisation.txt").write_text(
        "SentencePiece normalisation check on train word types (identity is used):\n"
        + "\n".join(norm_lines) + "\n", encoding="utf-8")
    print("\n".join(norm_lines))

    rows = []
    vocab_all = train_cnt > 0
    vocab_min2 = train_cnt >= 2
    for split in ("dev", "test"):
        rows.append(word_rows(corpus, split, vocab_all, "word (all train types)"))
        rows.append(word_rows(corpus, split, vocab_min2, "word (train count>=2)"))

    for algo in ("bpe", "unigram"):
        for size in args.sizes:
            prefix = models / f"spm_{algo}_{size}"
            t1 = time.time()
            train_spm(text, prefix, algo, size, args.threads)
            sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
            check_type_cache(corpus, sp)
            rows += spm_rows(corpus, sp, f"spm-{algo}-{size // 1000}k", algo, size)
            print(f"  {algo} {size}: trained + evaluated in {time.time() - t1:.0f}s")
    text.unlink()

    fields = list(rows[0])
    with open(out / "tokenizer_eval.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["split"] == "test":
            print(f"{r['tokenizer']:24s} fert={r['fertility']:<7} split={r['split_share']:<7} "
                  f"unk={r['unk_piece_rate']!s:<9} oov_tok={r['oov_token_rate']!s:<8} oov_type={r['oov_type_rate']}")
    print(f"done in {time.time() - t0:.0f}s -> {out / 'tokenizer_eval.csv'}")


if __name__ == "__main__":
    main()
