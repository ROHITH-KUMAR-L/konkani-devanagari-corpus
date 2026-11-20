"""Word- and subword-level n-gram language models (Phase 2, FR-P2-2 and FR-P2-7).

Data
  Sentences of data/splits/{train,dev,test}.txt, tokenised with
  corpus_utils.tokenize. Each sentence is wrapped as  <s> w1 ... wn </s>
  (one <s>; at the start of a sentence the history is simply shorter, as in
  SRILM/KenLM). Word vocabulary = train words with count >= 2 plus <UNK>, <s>,
  </s>; every other word becomes <UNK>. The model predicts w1..wn and </s>.

Smoothing methods (all implemented here with numpy, no LM library)
  laplace  add-1 on the highest available order
  addk     add-k, k chosen on dev from --k-grid separately for every n
  katz     Katz backoff with Good-Turing discounts for counts <= 5
  kn       interpolated Kneser-Ney, one discount per order
  mkn      interpolated modified Kneser-Ney (Chen & Goodman), 3 discounts per order
Kneser-Ney uses continuation counts N1+(. h w) for lower orders (raw counts for
n-grams that start with <s>) and ends in a uniform distribution over the
vocabulary, so every in-vocabulary word gets a non-zero probability.

N-grams are stored compactly: the k-gram (h, w) gets the integer key
id(h) * V + w, where id(h) is the index of h among the sorted unique
(k-1)-grams. All counting and scoring is vectorised over these int64 keys.

Subword LM
  The same mkn model over SentencePiece Unigram-16k pieces of the same word
  stream (no <UNK> needed). Perplexity is normalised per WORD:
  2 ** (-total log2 prob / (word tokens + sentences)), so it is directly
  comparable with the word models; per-piece perplexity is also reported.

Learning curve
  mkn trigram trained on the first 10/25/50/100% of train sentences, test
  perplexity with (a) the vocabulary of that subset and (b) the full-train
  vocabulary held fixed.

Outputs (results/phase2/)
  ngram_perplexity.csv, subword_perplexity.csv, learning_curve.csv,
  plots/ngram_perplexity.png, plots/learning_curve.png

Usage
  python scripts/phase2/ngram_lm.py [--splits data/splits] [--out results/phase2]
      [--max-n 4] [--spm data/models/spm_unigram_16000.model] [--seed 13]
  python scripts/phase2/ngram_lm.py --selftest     # distributions sum to 1, scorer consistency
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from common import ROOT, Corpus, prefix

UNK, BOS, EOS = 0, 1, 2          # same ids as SentencePiece's default <unk>, <s>, </s>
SPECIALS = ["<UNK>", "<s>", "</s>"]
K_GRID = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]


# ------------------------------------------------------------------ data

def build_vocab(train_cnt, min_count=2):
    """Map provisional ids -> model ids (0..2 specials, then words by frequency)."""
    keep = np.flatnonzero(train_cnt >= min_count)
    keep = keep[np.argsort(-train_cnt[keep], kind="stable")]
    table = np.full(len(train_cnt), UNK, dtype=np.int32)
    table[keep] = np.arange(3, 3 + len(keep), dtype=np.int32)
    return table, 3 + len(keep)


class Stream:
    """Sentences with boundaries: flat token array plus position bookkeeping."""

    def __init__(self, tok, lens):
        lens = np.asarray(lens, dtype=np.int64)
        L = lens + 2
        total = int(L.sum())
        starts = np.zeros(len(L), np.int64)
        starts[1:] = np.cumsum(L)[:-1]
        s = np.empty(total, np.int32)
        inner = np.ones(total, bool)
        inner[starts] = False
        inner[starts + L - 1] = False
        s[starts], s[starts + L - 1], s[inner] = BOS, EOS, tok
        sid = np.repeat(np.arange(len(L)), L)
        ar = np.arange(total)
        self.tok = s
        self.pos = (ar - starts[sid]).astype(np.int32)         # index inside its sentence (<s> = 0)
        self.rem = (starts[sid] + L[sid] - ar).astype(np.int32)  # tokens left incl. this one
        self.n_sent = len(L)


# ------------------------------------------------------------------ counts

class Counts:
    """Unique k-grams for k = 1..N as sorted int64 keys with counts."""

    def __init__(self, stream, V, N):
        self.V, self.N = V, N
        self.keys = [None, None]
        self.cnt = [None, np.bincount(stream.tok, minlength=V).astype(np.int64)]
        self.suf = [None, None]   # suf[k][g]: (k-1)-gram id of the k-gram g without its first word
        tok, rem = stream.tok, stream.rem
        prev = tok.astype(np.int32)
        for k in range(2, N + 1):
            valid = np.flatnonzero(rem >= k)
            key = prev[valid].astype(np.int64) * V + tok[valid + k - 1]
            uk, first, inv, c = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
            del key
            self.keys.append(uk)
            self.cnt.append(c.astype(np.int32))
            self.suf.append(prev[valid[first] + 1].astype(np.int32))
            cur = np.full(len(tok), -1, np.int32)
            cur[valid] = inv
            del inv, first, valid
            prev = cur

    def n_unique(self, k):
        return self.V if k == 1 else len(self.keys[k])

    def lookup(self, k, ctx, w):
        """Id of the k-gram (ctx, w) given ctx = its (k-1)-gram id; -1 if unseen."""
        if k == 1:
            return np.asarray(w, np.int64)
        ctx = np.asarray(ctx, np.int64)
        keys = self.keys[k]
        key = np.where(ctx >= 0, ctx * self.V + w, -1)
        idx = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
        return np.where((ctx >= 0) & (keys[idx] == key), idx, -1)

    def position_ids(self, stream, N=None):
        """ids[k][p] = id of the k-gram starting at position p (-1 if unseen/invalid)."""
        N = N or self.N
        ids = [None, stream.tok.astype(np.int64)]
        for k in range(2, N + 1):
            cur = np.full(len(stream.tok), -1, np.int64)
            valid = np.flatnonzero(stream.rem >= k)
            cur[valid] = self.lookup(k, ids[k - 1][valid], stream.tok[valid + k - 1])
            ids.append(cur)
        return ids


# ------------------------------------------------------------------ models
# Each model gives a unigram vector p1 (with p1[<s>] = 0) and, for order k >= 2,
# combine(k, ctx, g, p_lower) -> p, where ctx is the (k-1)-gram id of the
# history (-1 if unseen), g the k-gram id (-1 if unseen), p_lower the
# probability from orders < k.

def kn_discounts(a, modified):
    """Chen & Goodman discounts from count-of-counts of (adjusted) counts a."""
    a = np.rint(a[a > 0]).astype(np.int64)
    n = np.bincount(np.minimum(a, 5), minlength=6)
    if n[1] == 0 or n[2] == 0:     # e.g. unigrams over a count>=2 vocabulary: no estimate, no discount
        return np.zeros(4)
    n1, n2, n3, n4 = (max(int(x), 1) for x in n[1:5])
    Y = n1 / (n1 + 2 * n2)
    if not modified:
        return np.array([0.0, Y, Y, Y])
    D = np.array([0.0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3])
    return np.clip(D, 0.0, [0, 1, 2, 3])


class KneserNey:
    def __init__(self, C, modified=True):
        self.name = "mkn" if modified else "kn"
        N, V = C.N, C.V
        self.D, self.num, self.denom, self.gamma = {}, {}, {}, {}
        for k in range(1, N + 1):
            raw = C.cnt[k]
            if k == N:
                a = raw.astype(np.float64)
            else:   # continuation count; n-grams starting with <s> have none -> raw count
                cont = np.bincount(C.suf[k + 1], minlength=len(raw))
                a = np.where(cont > 0, cont, raw).astype(np.float64)
            if k == 1:
                a[BOS] = 0.0
            D = kn_discounts(a, modified)
            d = np.where(a > 0, D[np.minimum(a, 3).astype(np.int64)], 0.0)
            num = np.maximum(a - d, 0.0)
            self.D[k] = D
            if k == 1:
                total = a.sum()
                self.p1 = num / total + (d.sum() / total) / (V - 1)
                self.p1[BOS] = 0.0
            else:
                pre = C.keys[k] // V
                m = C.n_unique(k - 1)
                den = np.bincount(pre, weights=a, minlength=m)
                gs = np.bincount(pre, weights=d, minlength=m)
                self.num[k] = num.astype(np.float32)
                self.denom[k] = den.astype(np.float32)
                self.gamma[k] = np.divide(gs, den, out=np.zeros_like(gs), where=den > 0).astype(np.float32)

    def combine(self, k, ctx, g, pl):
        den = np.where(ctx >= 0, self.denom[k][np.maximum(ctx, 0)], 0.0)
        ok = den > 0
        c = np.maximum(ctx, 0)
        num = np.where(g >= 0, self.num[k][np.maximum(g, 0)], 0.0)
        return np.where(ok, num / np.where(ok, den, 1.0) + self.gamma[k][c] * pl, pl)


class AddK:
    """Add-k on the highest available order; unseen history -> uniform."""

    def __init__(self, C, k=1.0):
        self.k, self.name = k, ("laplace" if k == 1.0 else "addk")
        V = C.V
        self.Vp = V - 1   # predictable symbols: everything but <s>
        c1 = C.cnt[1].astype(np.float64).copy()
        c1[BOS] = 0
        self.p1 = (c1 + k) / (c1.sum() + k * self.Vp)
        self.p1[BOS] = 0.0
        self.cnt, self.denom = {}, {}
        for n in range(2, C.N + 1):
            self.cnt[n] = C.cnt[n]
            self.denom[n] = np.bincount(C.keys[n] // V, weights=C.cnt[n], minlength=C.n_unique(n - 1))

    def combine(self, n, ctx, g, pl):
        den = np.where(ctx >= 0, self.denom[n][np.maximum(ctx, 0)], 0.0)
        c = np.where(g >= 0, self.cnt[n][np.maximum(g, 0)], 0)
        return (c + self.k) / (den + self.k * self.Vp)


class Katz:
    """Katz backoff with Good-Turing discounted counts (counts <= K discounted)."""

    def __init__(self, C, K=5):
        self.name = "katz"
        V = C.V
        c1 = C.cnt[1].astype(np.float64).copy()
        c1[BOS] = 0
        self.p1 = c1 / c1.sum()
        if (self.p1[np.arange(V) != BOS] == 0).any():   # only with a fixed external vocabulary
            self.p1 = 0.999 * self.p1 + 0.001 / (V - 1)
            self.p1[BOS] = 0.0
        self.pstar, self.alpha, self.denom, self.K = {}, {}, {}, {}
        seen_prev = self.p1          # probability of each seen (k-1)-gram under the model
        for k in range(2, C.N + 1):
            c = C.cnt[k]
            dr, Kk = self.gt_discounts(c, K)
            self.K[k] = Kk
            d = np.where(c <= Kk, dr[np.minimum(c, Kk)], 1.0)
            pre = C.keys[k] // V
            m = C.n_unique(k - 1)
            den = np.bincount(pre, weights=c, minlength=m)
            ps = d * c / den[pre]
            seen = np.bincount(pre, weights=ps, minlength=m)
            # contexts whose continuations are all frequent (no GT discount) keep
            # a floor of 0.5/c(h) for unseen words, so no test word gets p = 0
            left = np.maximum(1.0 - seen, 0.5 / np.maximum(den, 1.0))
            scale = np.divide(1.0 - left, seen, out=np.ones_like(seen), where=seen > 0)
            ps *= scale[pre]
            low = np.bincount(pre, weights=seen_prev[C.suf[k]], minlength=m)
            self.alpha[k] = np.where(den > 0, left / np.maximum(1.0 - low, 1e-12), 1.0)
            self.pstar[k], self.denom[k] = ps, den
            seen_prev = ps

    @staticmethod
    def gt_discounts(c, K):
        n = np.bincount(np.minimum(c, K + 2), minlength=K + 3).astype(np.float64)
        if n[1] == 0:
            return np.ones(1), 0
        for Kk in range(K, 0, -1):
            R = (Kk + 1) * n[Kk + 1] / n[1]
            r = np.arange(1, Kk + 1)
            rstar = (r + 1) * n[r + 1] / np.maximum(n[r], 1)
            dr = (rstar / r - R) / (1 - R)
            if np.all((dr > 0) & (dr <= 1)):
                return np.concatenate([[1.0], dr]), Kk
        return np.ones(1), 0

    def combine(self, k, ctx, g, pl):
        den = np.where(ctx >= 0, self.denom[k][np.maximum(ctx, 0)], 0.0)
        a = self.alpha[k][np.maximum(ctx, 0)]
        ps = self.pstar[k][np.maximum(g, 0)]
        return np.where(den > 0, np.where(g >= 0, ps, a * pl), pl)


# ------------------------------------------------------------------ scoring

def log2probs(model, C, stream, N):
    """log2 P(token) for every predicted token (all but <s>) of a stream."""
    ids = C.position_ids(stream, N)
    pred = np.flatnonzero(stream.tok != BOS)
    p = model.p1[stream.tok[pred]]
    for k in range(2, N + 1):
        sel = np.flatnonzero(stream.pos[pred] >= k - 1)
        start = pred[sel] - (k - 1)
        p[sel] = model.combine(k, ids[k - 1][start], ids[k][start], p[sel])
    return np.log2(p)


def perplexity(model, C, stream, N, n_norm=None):
    lp = log2probs(model, C, stream, N)
    if not np.all(np.isfinite(lp)):
        return float("inf")
    return 2.0 ** (-lp.sum() / (n_norm or len(lp)))


def prob_all(model, C, hist, N):
    """Full distribution P(. | hist) over the vocabulary, computed word by word."""
    V = C.V
    w = np.arange(V, dtype=np.int64)
    p = model.p1.copy()
    ctx = None
    for k in range(2, min(N, len(hist) + 1) + 1):
        h = hist[-(k - 1):]
        ctx = h[0]
        for j, t in enumerate(h[1:], start=2):
            ctx = int(C.lookup(j, np.array([ctx]), np.array([t]))[0])
        g = C.lookup(k, np.full(V, ctx), w)
        p = model.combine(k, np.full(V, ctx, np.int64), g, p)
    p[BOS] = 0.0
    return p


# ------------------------------------------------------------------ experiments

def make_models(C, k_values=()):
    """Yield the models one at a time, so only one set of model arrays is in memory."""
    yield AddK(C, 1.0)
    for k in k_values:
        yield AddK(C, k)
    yield Katz(C)
    yield KneserNey(C, False)
    yield KneserNey(C, True)


def run_word(corpus, args, out):
    table, V = build_vocab(corpus.counts("train"), 2)
    st = {s: Stream(table[corpus.data[s][0]], corpus.data[s][1]) for s in ("train", "dev", "test")}
    rows = []
    for split in ("dev", "test"):
        unk = (st[split].tok == UNK).sum() / (st[split].tok != BOS).sum()
        print(f"{split}: {(st[split].tok != BOS).sum()} predicted tokens, <UNK> rate {unk:.4f}")
    print(f"vocabulary V = {V} (incl. <UNK>, <s>, </s>)")
    invocab = st["test"].tok[st["test"].tok != BOS] != UNK   # predicted test tokens that are not <UNK>
    for n in range(1, args.max_n + 1):
        t = time.time()
        C = Counts(st["train"], V, n)
        best = None
        for k in K_GRID[1:]:      # tune add-k on dev
            pp = perplexity(AddK(C, k), C, st["dev"], n)
            if best is None or pp < best[1]:
                best = (k, pp)
        for m in make_models(C, [best[0]]):
            lp = log2probs(m, C, st["test"], n)
            r = dict(n=n, smoothing=m.name, param="", V=V,
                     dev_ppl=round(perplexity(m, C, st["dev"], n), 2),
                     test_ppl=round(2.0 ** (-lp.mean()), 2),
                     test_ppl_invocab=round(2.0 ** (-lp[invocab].mean()), 2))
            if m.name in ("laplace", "addk"):
                r["param"] = f"k={m.k:g}"
            elif m.name in ("kn", "mkn"):
                r["param"] = "D=" + "/".join(f"{x:.3f}" for x in m.D[n][1:(4 if m.name == 'mkn' else 2)])
            elif m.name == "katz":
                r["param"] = f"GT K={m.K.get(n, '-')}"
            rows.append(r)
            print(f"  n={n} {m.name:8s} {r['param']:24s} dev={r['dev_ppl']:>10} test={r['test_ppl']:>10}")
            del m, lp
        print(f"  n={n}: {sum(C.n_unique(k) for k in range(2, n + 1))} unique 2..n-grams, {time.time() - t:.0f}s")
        del C
    write_csv(out / "ngram_perplexity.csv", rows)
    return rows, st


def run_subword(corpus, args, out):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(args.spm))
    assert (sp.unk_id(), sp.bos_id(), sp.eos_id()) == (UNK, BOS, EOS)
    V = sp.vocab_size()
    # pieces of every word type, as a CSR table indexed by provisional id
    enc = sp.encode(corpus.types, out_type=int)
    plen = np.array([len(e) for e in enc], np.int64)
    off = np.zeros(len(enc) + 1, np.int64)
    off[1:] = np.cumsum(plen)
    flat = np.fromiter((p for e in enc for p in e), np.int32, count=int(off[-1]))
    del enc

    def pieces(split):
        pids, lens = corpus.data[split]
        L = plen[pids]
        tot = int(L.sum())
        base = np.repeat(off[pids] - (np.cumsum(L) - L), L)
        tok = flat[base + np.arange(tot)]
        sid = np.repeat(np.arange(len(lens)), lens)
        return Stream(tok, np.bincount(sid, weights=L, minlength=len(lens)).astype(np.int64))

    st = {s: pieces(s) for s in ("train", "dev", "test")}
    # group test pieces into words: each word owns plen[pid] pieces, each </s> one piece
    pids, lens = corpus.data["test"]
    ws = Stream(plen[pids].astype(np.int32), lens)
    unit_len = np.where(ws.rem == 1, 1, ws.tok)[ws.pos > 0]
    group = np.repeat(np.arange(len(unit_len)), unit_len)
    wtab, _ = build_vocab(corpus.counts("train"), 2)
    wst = Stream(wtab[pids], lens)
    invocab = wst.tok[wst.pos > 0] != UNK   # same positions as the word models' in-vocabulary tokens
    rows = []
    for n in range(1, args.max_n_subword + 1):
        C = Counts(st["train"], V, n)
        m = KneserNey(C, True)
        r = dict(n=n, smoothing="mkn", unit=f"spm-unigram-{V // 1000}k")
        for split in ("dev", "test"):
            lp = log2probs(m, C, st[split], n)
            words = len(corpus.data[split][0]) + len(corpus.data[split][1])   # words + </s>
            r[f"{split}_ppl_per_word"] = round(2 ** (-lp.sum() / words), 2)
            r[f"{split}_ppl_per_piece"] = round(2 ** (-lp.sum() / len(lp)), 2)
            r[f"{split}_pieces_per_word"] = round(len(lp) / words, 4)
        word_lp = np.bincount(group, weights=lp, minlength=len(unit_len))
        r["test_ppl_per_word_invocab"] = round(2 ** (-word_lp[invocab].mean()), 2)
        r["test_ppl_per_word_oov_only"] = round(2 ** (-word_lp[~invocab].mean()), 2)
        rows.append(r)
        print(f"  subword n={n}: test per-word ppl={r['test_ppl_per_word']} per-piece={r['test_ppl_per_piece']}")
        del C
    write_csv(out / "subword_perplexity.csv", rows)
    return rows


def run_learning_curve(corpus, args, out):
    full_cnt = corpus.counts("train")
    full_table, full_V = build_vocab(full_cnt, 2)
    test_pids, test_lens = corpus.data["test"]
    rows = []
    for frac in args.fractions:
        pids, lens = prefix(corpus.data["train"], frac)
        sub_cnt = np.bincount(pids, minlength=len(corpus.types))
        r = dict(fraction=frac, train_sentences=len(lens), train_tokens=int(lens.sum()))
        for label, (table, V) in (("own_vocab", build_vocab(sub_cnt, 2)), ("fixed_vocab", (full_table, full_V))):
            tr, te = Stream(table[pids], lens), Stream(table[test_pids], test_lens)
            C = Counts(tr, V, 3)
            r[f"V_{label}"] = V
            r[f"unk_rate_{label}"] = round(float((te.tok == UNK).sum() / (te.tok != BOS).sum()), 4)
            r[f"test_ppl_{label}"] = round(perplexity(KneserNey(C, True), C, te, 3), 2)
            del C
        rows.append(r)
        print(f"  {frac:>5.0%}: {r['train_tokens']} tokens  ppl own={r['test_ppl_own_vocab']} "
              f"(unk {r['unk_rate_own_vocab']})  fixed={r['test_ppl_fixed_vocab']} (unk {r['unk_rate_fixed_vocab']})")
    write_csv(out / "learning_curve.csv", rows)
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ plots

COLORS = {"laplace": "#eb6834", "addk": "#eda100", "katz": "#e87ba4", "kn": "#1baf7a", "mkn": "#2a78d6"}
LABELS = {"laplace": "Laplace (add-1)", "addk": "add-k (k tuned on dev)", "katz": "Katz + Good-Turing",
          "kn": "interpolated KN", "mkn": "modified KN"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_perplexity(rows, sub_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    for name in ("laplace", "addk", "katz", "kn", "mkn"):
        pts = [(r["n"], r["test_ppl"]) for r in rows if r["smoothing"] == name]
        ax.plot(*zip(*pts), marker="o", markersize=5, linewidth=2, color=COLORS[name], label=LABELS[name])
    if sub_rows:
        pts = [(r["n"], r["test_ppl_per_word"]) for r in sub_rows]
        ax.plot(*zip(*pts), marker="s", markersize=5, linewidth=2, linestyle="--", color="#555555",
                label="modified KN, Unigram-16k pieces (per-word ppl)")
    ax.set_yscale("log")
    ax.set_xticks(range(1, max(r["n"] for r in rows + (sub_rows or [])) + 1))
    ax.set_xlabel("n-gram order n (for subwords: order in pieces)")
    ax.set_ylabel("test perplexity (log scale)")
    ax.set_title("Test perplexity by smoothing method and order", loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_learning_curve(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    x = [r["train_tokens"] for r in rows]
    ax.plot(x, [r["test_ppl_own_vocab"] for r in rows], marker="o", linewidth=2, color="#2a78d6",
            label="vocabulary of the subset (count >= 2)")
    ax.plot(x, [r["test_ppl_fixed_vocab"] for r in rows], marker="o", linewidth=2, color="#eb6834",
            label="full-train vocabulary held fixed")
    for r in rows:
        ax.annotate(f"{r['fraction']:.0%}", (r["train_tokens"], r["test_ppl_fixed_vocab"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (log scale)")
    ax.set_ylabel("test perplexity")
    ax.set_title("Learning curve: modified-KN trigram", loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ self-test

def selftest(args):
    rng = np.random.default_rng(args.seed)
    print("self-test 1: toy corpus")
    toy_V = 8
    sents = [rng.integers(3, toy_V, size=rng.integers(1, 7)) for _ in range(200)]
    sents += [np.array([3, 4, 5])] * 20      # frequent pattern -> high counts for Katz
    tok = np.concatenate(sents).astype(np.int32)
    lens = np.array([len(s) for s in sents])
    cases = [("toy", Stream(tok, lens), toy_V, 4)]
    print("self-test 2: first 20k train sentences")
    corpus = Corpus(args.splits, splits=("train",))
    pids, lens = prefix(corpus.data["train"], 1.0)
    m = int(min(20000, len(lens)))
    pids, lens = pids[: int(lens[:m].sum())], lens[:m]
    table, V = build_vocab(np.bincount(pids, minlength=len(corpus.types)), 2)
    cases.append(("train-20k", Stream(table[pids], lens), V, 4))
    worst = 0.0
    for label, st, V, N in cases:
        C = Counts(st, V, N)
        cand = np.flatnonzero(st.pos >= N - 1)
        for model in make_models(C, [0.01]):
            for p0 in rng.choice(cand, size=5, replace=False):
                hist = list(st.tok[p0 - (N - 1): p0])
                for cut in range(len(hist) + 1):       # also shorter histories, incl. empty
                    h = hist[cut:]
                    dist = prob_all(model, C, h, N)
                    err = abs(dist.sum() - 1.0)
                    worst = max(worst, err)
                    assert err < 1e-6, f"{label} {model.name} hist={h}: sum={dist.sum()!r}"
            # scorer consistency: vectorised log-probs == prob_all on the same positions
            lp = log2probs(model, C, st, N)
            pred = np.flatnonzero(st.tok != BOS)
            for i in rng.choice(len(pred), size=20, replace=False):
                p = pred[i]
                h = list(st.tok[max(p - st.pos[p], p - (N - 1)): p])
                ref = math.log2(prob_all(model, C, h, N)[st.tok[p]])
                assert abs(ref - lp[i]) < 1e-9, f"{label} {model.name}: scorer mismatch {ref} vs {lp[i]}"
            print(f"  {label:10s} {model.name:8s} sums to 1 over V\\{{<s>}} for 5 random contexts "
                  f"x all history lengths; scorer matches")
    print(f"self-test passed (max |sum - 1| = {worst:.2e})")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(ROOT / "data/splits"))
    ap.add_argument("--out", default=str(ROOT / "results/phase2"))
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--max-n-subword", type=int, default=5)
    ap.add_argument("--spm", default=str(ROOT / "data/models/spm_unigram_16000.model"))
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.0])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--selftest", action="store_true", help="run the sanity checks only")
    ap.add_argument("--skip", nargs="*", default=[], choices=["word", "subword", "curve"])
    args = ap.parse_args()
    np.random.seed(args.seed)
    if args.selftest:
        selftest(args)
        return
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    corpus = Corpus(args.splits)
    print(f"loaded splits in {time.time() - t0:.0f}s")
    rows = sub_rows = None
    if "word" not in args.skip:
        print("== word n-gram LMs")
        rows, _ = run_word(corpus, args, out)
    if "subword" not in args.skip:
        print("== subword n-gram LM (modified KN over SentencePiece pieces)")
        sub_rows = run_subword(corpus, args, out)
    if rows:
        plot_perplexity(rows, sub_rows, out / "plots/ngram_perplexity.png")
    if "curve" not in args.skip:
        print("== learning curve (modified-KN trigram)")
        lc = run_learning_curve(corpus, args, out)
        plot_learning_curve(lc, out / "plots/learning_curve.png")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
