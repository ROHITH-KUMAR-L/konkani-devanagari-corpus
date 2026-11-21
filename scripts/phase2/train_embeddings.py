"""Train Konkani word embeddings (FR-P2-3): word2vec CBOW / skip-gram, FastText, GloVe.

Input is the sentence-per-line training split. Every line is tokenised with
scripts/corpus_utils.tokenize (punctuation removed, numbers -> <NUM>) and cached
once as a space-separated file, which gensim then reads with `corpus_file=`
(streams from disk, so memory stays flat as the corpus grows).

Models (all dim=100, window=5, min_count=3, negative sampling, seed 42):
  cbow      word2vec CBOW
  sg        word2vec skip-gram
  ft        FastText skip-gram, character n-grams 3-6 code points
  ft25      FastText skip-gram, character n-grams 2-5 code points (ablation)
  glove     GloVe, a small numpy implementation (AdaGrad) over the top-N words

Each model is saved as gensim KeyedVectors in <out>/<name>.kv (FastText keeps
its n-gram table, so it can build vectors for unseen words).

--fractions trains the learning-curve models (FR-P2-7) on nested random
subsets of the training sentences (a 10% subset is contained in the 25% one,
and so on) and saves them as <out>/lc/<model>_<pct>.kv. 100% is the main model.

Usage:
  python scripts/phase2/train_embeddings.py --train data/splits/train.txt --out data/models
  python scripts/phase2/train_embeddings.py --models sg ft --fractions 0.1 0.25 0.5
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_utils import tokenize  # noqa: E402

ALL_MODELS = ["cbow", "sg", "ft", "ft25", "glove"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data

def tokenise_file(src, dst, fraction=1.0, seed=42):
    """Write one space-separated tokenised sentence per line; return (sentences, tokens).

    With fraction < 1, keep a line when its seeded uniform draw is below the
    fraction, so smaller subsets are nested inside larger ones.
    """
    rng = np.random.default_rng(seed)
    n_sent = n_tok = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            u = rng.random()            # drawn for every line so subsets stay nested
            if u >= fraction:
                continue
            toks = tokenize(line)
            if toks:
                fout.write(" ".join(toks) + "\n")
                n_sent += 1
                n_tok += len(toks)
    return n_sent, n_tok


# ------------------------------------------------------------------ gensim models

def train_gensim(name, corpus_file, args):
    from gensim.models import FastText, Word2Vec

    common = dict(corpus_file=str(corpus_file), vector_size=args.dim, window=args.window,
                  min_count=args.min_count, epochs=args.epochs, negative=5, sample=1e-4,
                  workers=args.workers, seed=args.seed)
    if name == "cbow":
        model = Word2Vec(sg=0, **common)
    elif name == "sg":
        model = Word2Vec(sg=1, **common)
    elif name == "ft":
        model = FastText(sg=1, min_n=3, max_n=6, bucket=args.bucket, **common)
    elif name == "ft25":
        model = FastText(sg=1, min_n=2, max_n=5, bucket=args.bucket, **common)
    else:
        raise ValueError(name)
    return model.wv


# ------------------------------------------------------------------ GloVe (numpy)

def glove_vocab(corpus_file, max_vocab, min_count):
    from collections import Counter
    c = Counter()
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            c.update(line.split())
    words = [w for w, n in c.most_common(max_vocab) if n >= min_count]
    return words


def glove_cooccurrence(corpus_file, w2i, window, chunk=50000):
    """Symmetric co-occurrence counts weighted 1/distance (as in the GloVe paper).

    Out-of-vocabulary tokens keep their position (they count towards distance)
    but contribute no pairs.
    """
    V = len(w2i)
    M = sparse.csr_matrix((V, V), dtype=np.float32)

    def flush(ids, sid):
        nonlocal M
        ids = np.asarray(ids, dtype=np.int32)
        sid = np.asarray(sid, dtype=np.int32)
        rows, cols, vals = [], [], []
        for d in range(1, window + 1):
            a, b = ids[:-d], ids[d:]
            m = (sid[:-d] == sid[d:]) & (a >= 0) & (b >= 0)
            rows += [a[m], b[m]]
            cols += [b[m], a[m]]
            vals += [np.full(int(m.sum()), 1.0 / d, dtype=np.float32)] * 2
        r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        M = M + sparse.coo_matrix((v, (r, c)), shape=(V, V)).tocsr()

    ids, sid, n = [], [], 0
    with open(corpus_file, encoding="utf-8") as f:
        for k, line in enumerate(f):
            toks = line.split()
            ids += [w2i.get(t, -1) for t in toks]
            sid += [k] * len(toks)
            n += 1
            if n % chunk == 0:
                flush(ids, sid)
                ids, sid = [], []
    if ids:
        flush(ids, sid)
    return M.tocoo()


def _scatter(idx):
    """Unique indices and a (n_unique x batch) 0/1 matrix that sums rows sharing an index."""
    u, inv = np.unique(idx, return_inverse=True)
    S = sparse.csr_matrix((np.ones(len(idx), np.float32), (inv, np.arange(len(idx)))),
                          shape=(len(u), len(idx)))
    return u, S


def train_glove(corpus_file, args):
    from gensim.models import KeyedVectors

    words = glove_vocab(corpus_file, args.glove_vocab, args.min_count)
    w2i = {w: i for i, w in enumerate(words)}
    log(f"  glove: vocab {len(words)}; counting co-occurrences (window {args.glove_window})")
    X = glove_cooccurrence(corpus_file, w2i, args.glove_window)
    I, J, x = X.row.astype(np.int32), X.col.astype(np.int32), X.data.astype(np.float32)
    del X
    log(f"  glove: {len(x):,} non-zero co-occurrence cells")
    logx = np.log(x)
    fx = np.minimum(1.0, (x / args.glove_xmax) ** 0.75).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    V, d = len(words), args.dim
    W = ((rng.random((V, d)) - 0.5) / d).astype(np.float32)
    C = ((rng.random((V, d)) - 0.5) / d).astype(np.float32)
    bw = np.zeros(V, np.float32)
    bc = np.zeros(V, np.float32)
    HW, HC = np.ones((V, d), np.float32), np.ones((V, d), np.float32)   # AdaGrad accumulators
    hbw, hbc = np.ones(V, np.float32), np.ones(V, np.float32)
    lr, B = args.glove_lr, 100_000
    for ep in range(args.glove_epochs):
        perm = rng.permutation(len(x))
        loss = 0.0
        for s in range(0, len(x), B):
            p = perm[s:s + B]
            i, j = I[p], J[p]
            wi, cj = W[i], C[j]
            diff = np.einsum("ij,ij->i", wi, cj) + bw[i] + bc[j] - logx[p]
            fd = fx[p] * diff
            loss += 0.5 * float(np.dot(fd, diff))
            gw = fd[:, None] * cj
            gc = fd[:, None] * wi
            # AdaGrad step; repeated indices in a batch have their updates summed
            ui, Si = _scatter(i)
            uj, Sj = _scatter(j)
            W[ui] += Si @ (-lr * gw / np.sqrt(HW[i]))
            C[uj] += Sj @ (-lr * gc / np.sqrt(HC[j]))
            HW[ui] += Si @ (gw * gw)
            HC[uj] += Sj @ (gc * gc)
            bw[ui] += Si @ (-lr * fd / np.sqrt(hbw[i]))
            bc[uj] += Sj @ (-lr * fd / np.sqrt(hbc[j]))
            hbw[ui] += Si @ (fd * fd)
            hbc[uj] += Sj @ (fd * fd)
        log(f"  glove epoch {ep + 1}/{args.glove_epochs}: mean weighted loss {loss / len(x):.4f}")
    kv = KeyedVectors(vector_size=d)
    kv.add_vectors(words, W + C)        # sum of word and context vectors, as in the paper
    return kv


# ------------------------------------------------------------------ main

def train_one(name, corpus_file, out_path, args):
    t0 = time.time()
    kv = train_glove(corpus_file, args) if name == "glove" else train_gensim(name, corpus_file, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kv.save(str(out_path))
    log(f"  {name}: {len(kv.index_to_key):,} words, {time.time() - t0:.0f}s -> {out_path}")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/splits/train.txt")
    ap.add_argument("--out", default="data/models")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--fractions", nargs="*", type=float, default=[],
                    help="learning-curve subset sizes (< 1); trains --models on each")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bucket", type=int, default=500_000, help="FastText n-gram hash buckets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--glove-vocab", type=int, default=30_000)
    ap.add_argument("--glove-window", type=int, default=10)
    ap.add_argument("--glove-epochs", type=int, default=15)
    ap.add_argument("--glove-xmax", type=float, default=100.0)
    ap.add_argument("--glove-lr", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.out)
    timings = []
    if not args.fractions:
        tok = out / "train.tok.txt"
        n_sent, n_tok = tokenise_file(args.train, tok)
        log(f"tokenised {args.train}: {n_sent:,} sentences, {n_tok:,} tokens -> {tok}")
        for name in args.models:
            log(f"training {name}")
            timings.append((name, 1.0, train_one(name, tok, out / f"{name}.kv", args)))
    else:
        for frac in sorted(args.fractions):
            pct = int(round(frac * 100))
            tok = out / "lc" / f"train_{pct}.tok.txt"
            n_sent, n_tok = tokenise_file(args.train, tok, fraction=frac, seed=args.seed)
            log(f"subset {pct}%: {n_sent:,} sentences, {n_tok:,} tokens")
            with open(out / "lc" / f"train_{pct}.size", "w") as f:
                f.write(f"{n_sent}\t{n_tok}\n")
            for name in args.models:
                log(f"training {name} on {pct}%")
                timings.append((name, frac, train_one(name, tok, out / "lc" / f"{name}_{pct}.kv", args)))
    with open(out / ("timings_lc.tsv" if args.fractions else "timings.tsv"), "w") as f:
        f.write("model\tfraction\tseconds\n")
        for name, frac, s in timings:
            f.write(f"{name}\t{frac}\t{s:.1f}\n")


if __name__ == "__main__":
    main()
