"""Intrinsic evaluation of the Konkani embeddings (FR-P2-4, FR-P2-5, FR-P2-7).

For every model in <models>/*.kv (written by train_embeddings.py):
  * word similarity: Spearman rho between the draft gold scores
    (data/eval/word_similarity.tsv) and cosine similarity, with a 95% bootstrap
    interval. Coverage = pairs where both words have a vector. FastText is
    scored twice: in-vocabulary pairs only, and all pairs (n-gram vectors for
    unseen words).
  * analogies (data/eval/analogies.tsv): 3CosAdd, top-1, question words
    excluded, search over the model's whole vocabulary; accuracy over covered
    questions and over all questions (uncovered = wrong), overall and per category.
  * nearest neighbours of probe words -> results/phase2/nearest_neighbours.md
  * t-SNE map of frequent words -> results/phase2/plots/tsne_<model>.png
  * OOV robustness: spelling/OCR-style damage to frequent words (drop anusvara,
    drop a vowel sign, long/short i/u swap, drop nukta-free virama...) and the
    cosine between damaged and correct form, word2vec vs FastText.
  * learning curve: models in <models>/lc/ (10/25/50%) plus the main model (100%).

Outputs (in --out, default results/phase2):
  embedding_eval.csv             model, metric, value, coverage, n
  analogy_by_category.csv        model, category, n, covered, correct, acc_covered, acc_all
  similarity_pairs.csv           gold and per-model cosine for every pair
  eval_set_coverage.csv          train frequency of every evaluation word
  oov_robustness.csv             per damage type and model
  embedding_learning_curve.csv   model, fraction, tokens, vocab, spearman, analogy acc
  nearest_neighbours.md, plots/*.png

Usage:
  python scripts/phase2/eval_embeddings.py --models data/models --out results/phase2
"""
import argparse
import csv
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

MODEL_ORDER = ["cbow", "sg", "ft", "ft25", "glove"]
MODEL_LABEL = {"cbow": "word2vec CBOW", "sg": "word2vec skip-gram", "ft": "FastText (n-grams 3-6)",
               "ft25": "FastText (n-grams 2-5)", "glove": "GloVe (numpy, top 30k)"}
PROBES = ["आवय", "भाव", "गोंय", "पणजी", "कोंकणी", "सोमार", "जानेवारी", "तांबडो", "हात", "घर",
          "घरांत", "शाळा", "गेलो", "करता", "उदक", "पावस", "शिक्षक", "आमदार", "कविता", "दोन", "मोग"]

ANUSVARA, CANDRABINDU, NUKTA, VIRAMA = "ं", "ँ", "़", "्"
MATRAS = set("ािीुूृेैोौॅॉ")
SWAPS = {"ि": "ी", "ी": "ि", "ु": "ू", "ू": "ु"}


# ------------------------------------------------------------------ loading

def load_models(model_dir):
    from gensim.models import KeyedVectors
    models = {}
    for name in MODEL_ORDER:
        p = Path(model_dir) / f"{name}.kv"
        if p.exists():
            models[name] = KeyedVectors.load(str(p))
    return models


def is_fasttext(kv):
    return hasattr(kv, "vectors_ngrams")


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def train_counts(tok_file):
    c = Counter()
    with open(tok_file, encoding="utf-8") as f:
        for line in f:
            c.update(line.split())
    return c


def unit(m):
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1
    return m / n


def cos(kv, a, b):
    va, vb = kv[a], kv[b]
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


# ------------------------------------------------------------------ similarity

def similarity(kv, pairs, oov_ok=False, n_boot=1000, seed=42):
    """Spearman over pairs with vectors. oov_ok: use FastText n-gram vectors for unseen words."""
    gold, pred = [], []
    for p in pairs:
        a, b = p["word1"], p["word2"]
        if oov_ok or (a in kv.key_to_index and b in kv.key_to_index):
            gold.append(float(p["score"]))
            pred.append(cos(kv, a, b))
    if len(gold) < 3:
        return float("nan"), (float("nan"), float("nan")), len(gold)
    rho = spearmanr(gold, pred).statistic
    if n_boot == 0:
        return float(rho), (float("nan"), float("nan")), len(gold)
    rng = np.random.default_rng(seed)
    g, q = np.array(gold), np.array(pred)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(g), len(g))
        boots.append(spearmanr(g[idx], q[idx]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), (float(lo), float(hi)), len(gold)


# ------------------------------------------------------------------ analogies

class Index:
    """Unit-normalised vocabulary matrix for fast cosine search."""

    def __init__(self, kv):
        self.kv = kv
        self.words = kv.index_to_key
        self.M = unit(kv.vectors.astype(np.float32))
        self.w2i = kv.key_to_index

    def vec(self, w):
        if w in self.w2i:
            return self.M[self.w2i[w]]
        return unit(self.kv[w][None, :])[0]          # FastText OOV vector

    def nearest(self, v, k=10, exclude=()):
        sims = self.M @ (v / (np.linalg.norm(v) + 1e-12))
        for w in exclude:
            if w in self.w2i:
                sims[self.w2i[w]] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.words[i], float(sims[i])) for i in top]


def analogies(index, questions):
    per_cat = defaultdict(lambda: [0, 0, 0])          # n, covered, correct
    for q in questions:
        a, b, c, d = q["a"], q["b"], q["c"], q["d"]
        rec = per_cat[q["category"]]
        rec[0] += 1
        if not all(w in index.w2i for w in (a, b, c, d)):
            continue
        rec[1] += 1
        target = index.M[index.w2i[b]] - index.M[index.w2i[a]] + index.M[index.w2i[c]]
        pred = index.nearest(target, k=1, exclude=(a, b, c))[0][0]
        rec[2] += int(pred == d)
    return per_cat


def analogy_summary(per_cat):
    n = sum(r[0] for r in per_cat.values())
    cov = sum(r[1] for r in per_cat.values())
    cor = sum(r[2] for r in per_cat.values())
    return n, cov, cor


# ------------------------------------------------------------------ OOV robustness

def damage_variants(word):
    """Yield (damage_type, variant) for a word; one variant per type (first position)."""
    out = {}
    if ANUSVARA in word:
        out["drop_anusvara"] = word.replace(ANUSVARA, "", 1)
    elif word[-1] in MATRAS:
        out["add_anusvara"] = word + ANUSVARA
    for i, ch in enumerate(word):
        if ch in SWAPS and "swap_i_u_length" not in out:
            out["swap_i_u_length"] = word[:i] + SWAPS[ch] + word[i + 1:]
    for i in range(len(word) - 1, 0, -1):            # last internal vowel sign
        if word[i] in MATRAS and i < len(word) - 1:
            out["drop_matra"] = word[:i] + word[i + 1:]
            break
    if VIRAMA in word:
        out["drop_virama"] = word.replace(VIRAMA, "", 1)
    return out


def oov_robustness(models, counts, n_words=400, seed=42):
    """Damage frequent content words; compare cosine(damaged, correct) across models."""
    rng = random.Random(seed)
    freq = [w for w, n in counts.most_common(5000)[100:] if w != "<NUM>" and len(w) >= 3]
    words = freq[:n_words]
    items = []                                       # (type, correct, variant, variant_seen_in_train)
    for w in words:
        for t, v in damage_variants(w).items():
            if v != w:
                items.append((t, w, v, counts.get(v, 0) > 0))
    rows = []
    rand_pairs = [(rng.choice(words), rng.choice(words)) for _ in range(500)]
    indexes = {name: Index(kv) for name, kv in models.items()}
    for name, kv in models.items():
        idx = indexes[name]
        base = np.mean([cos(kv, a, b) for a, b in rand_pairs if a != b and a in idx.w2i and b in idx.w2i])
        for t in sorted({i[0] for i in items}) + ["ALL"]:
            sel = [i for i in items if (t == "ALL" or i[0] == t) and i[1] in idx.w2i]
            if not sel:
                continue
            have, cs, rec1, rec10 = 0, [], 0, 0
            cs_oov, cs_inv = [], []
            for _, w, v, _seen in sel:
                in_vocab = v in idx.w2i
                if in_vocab or is_fasttext(kv):
                    have += 1
                    c_ = cos(kv, w, v)
                    cs.append(c_)
                    (cs_inv if in_vocab else cs_oov).append(c_)
                    nn = [x for x, _ in idx.nearest(idx.vec(v), k=11, exclude=(v,))][:10]
                    rec1 += int(nn[0] == w)
                    rec10 += int(w in nn)
            rows.append(dict(model=name, damage=t, n=len(sel),
                             variant_in_vocab=sum(1 for i in sel if i[2] in idx.w2i),
                             has_vector=have,
                             mean_cos=round(float(np.mean(cs)), 4) if cs else "",
                             mean_cos_invocab_variants=round(float(np.mean(cs_inv)), 4) if cs_inv else "",
                             mean_cos_oov_variants=round(float(np.mean(cs_oov)), 4) if cs_oov else "",
                             recover_top1=round(rec1 / len(sel), 4),
                             recover_top10=round(rec10 / len(sel), 4),
                             random_pair_cos=round(float(base), 4)))
    examples = items[:0]
    seen_types = set()
    for it in items:
        if it[0] not in seen_types and not it[3]:
            seen_types.add(it[0])
            examples.append(it)
    return rows, examples


# ------------------------------------------------------------------ plots

def devanagari_font():
    """Path of an installed Devanagari font via fc-list, or None."""
    try:
        out = subprocess.run(["fc-list", ":lang=hi", "file", "family"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    paths = [l.split(":")[0].strip() for l in out.splitlines() if l.strip()]
    for pref in ("NotoSansDevanagari", "Lohit", "DroidSansDevanagari", "NotoSerifDevanagari"):
        for p in paths:
            if pref.lower() in Path(p).name.lower().replace("-", ""):
                return p
    return paths[0] if paths else None


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    path = devanagari_font()
    if not path:
        return None
    fm.fontManager.addfont(path)
    family = fm.FontProperties(fname=path).get_name()
    matplotlib.rcParams["font.family"] = [family, "DejaVu Sans"]
    return family


def tsne_plot(name, kv, words, out_png, labels=True, seed=42):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    X = unit(np.stack([kv[w] for w in words]))
    Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, metric="cosine").fit_transform(X)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(Y[:, 0], Y[:, 1], s=6, c="#4c72b0")
    if labels:
        for (x, y), w in zip(Y, words):
            ax.annotate(w, (x, y), fontsize=8, alpha=0.85, xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"t-SNE of {len(words)} frequent words: {MODEL_LABEL.get(name, name)}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def learning_curve_plot(rows, out_png):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in sorted({r["model"] for r in rows}):
        rs = sorted([r for r in rows if r["model"] == name], key=lambda r: r["tokens"])
        x = [r["tokens"] / 1e6 for r in rs]
        axes[0].plot(x, [r["spearman_invocab"] for r in rs], "o-", label=MODEL_LABEL.get(name, name))
        axes[1].plot(x, [r["analogy_acc_all"] for r in rs], "o-", label=MODEL_LABEL.get(name, name))
    axes[0].set_ylabel("Spearman rho (in-vocab pairs)")
    axes[1].set_ylabel("analogy accuracy (all questions)")
    for ax in axes:
        ax.set_xlabel("training tokens (millions)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Embedding learning curve (10/25/50/100% of train)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="data/models")
    ap.add_argument("--sim", default="data/eval/word_similarity.tsv")
    ap.add_argument("--analogies", default="data/eval/analogies.tsv")
    ap.add_argument("--out", default="results/phase2")
    ap.add_argument("--tsne-words", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models)
    models = load_models(mdir)
    if not models:
        sys.exit(f"no models in {mdir}")
    counts = train_counts(mdir / "train.tok.txt")
    pairs = read_tsv(args.sim)
    questions = read_tsv(args.analogies)
    print(f"models: {', '.join(models)}; {len(pairs)} pairs, {len(questions)} analogies")

    # coverage of the evaluation words in the training data
    eval_words = Counter()
    for p in pairs:
        eval_words.update([p["word1"], p["word2"]])
    for q in questions:
        eval_words.update([q["a"], q["b"], q["c"], q["d"]])
    with open(out / "eval_set_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "train_freq", "uses_in_eval"] + list(models))
        for word in sorted(eval_words, key=lambda x: -counts.get(x, 0)):
            w.writerow([word, counts.get(word, 0), eval_words[word]]
                       + [int(word in kv.key_to_index) for kv in models.values()])

    rows = []                                          # embedding_eval.csv
    cat_rows = []
    sim_table = {(p["word1"], p["word2"]): {"gold": p["score"], "note": p["note"]} for p in pairs}
    indexes = {}
    for name, kv in models.items():
        idx = indexes[name] = Index(kv)
        rho, ci, n = similarity(kv, pairs, seed=args.seed)
        rows.append(dict(model=name, metric="similarity_spearman_invocab", value=round(rho, 4),
                         coverage=round(n / len(pairs), 4), n=n, ci95=f"{ci[0]:.3f}..{ci[1]:.3f}"))
        if is_fasttext(kv):
            rho2, ci2, n2 = similarity(kv, pairs, oov_ok=True, seed=args.seed)
            rows.append(dict(model=name, metric="similarity_spearman_allpairs", value=round(rho2, 4),
                             coverage=round(n2 / len(pairs), 4), n=n2, ci95=f"{ci2[0]:.3f}..{ci2[1]:.3f}"))
        # frequency bands: split covered pairs at the median of min(freq1, freq2)
        cov_pairs = [p for p in pairs if p["word1"] in kv.key_to_index and p["word2"] in kv.key_to_index]
        mf = [min(counts.get(p["word1"], 0), counts.get(p["word2"], 0)) for p in cov_pairs]
        med = float(np.median(mf)) if mf else 0
        for band, sel in (("low", [p for p, f in zip(cov_pairs, mf) if f <= med]),
                          ("high", [p for p, f in zip(cov_pairs, mf) if f > med])):
            r, _, nb = similarity(kv, sel, n_boot=0)
            rows.append(dict(model=name, metric=f"similarity_spearman_{band}freq(minfreq{'<=' if band == 'low' else '>'}{med:.0f})",
                             value=round(r, 4), coverage="", n=nb, ci95=""))
        # mean cosine by pair family (inflection/case vs unrelated)
        fam = defaultdict(list)
        for p in cov_pairs:
            note = p["note"]
            key = ("inflection_or_case" if note.startswith(("inflection", "case form")) else
                   "synonym" if "synonym" in note else "antonym" if note.startswith("antonym") else
                   "unrelated" if note == "unrelated" else None)
            if key:
                fam[key].append(cos(kv, p["word1"], p["word2"]))
        for key, vals in sorted(fam.items()):
            rows.append(dict(model=name, metric=f"mean_cos_{key}", value=round(float(np.mean(vals)), 4),
                             coverage="", n=len(vals), ci95=""))
        for p in pairs:
            a, b = p["word1"], p["word2"]
            ok = (a in kv.key_to_index and b in kv.key_to_index) or is_fasttext(kv)
            sim_table[(a, b)][name] = round(cos(kv, a, b), 4) if ok else ""

        per_cat = analogies(idx, questions)
        nq, ncov, ncor = analogy_summary(per_cat)
        rows.append(dict(model=name, metric="analogy_acc_covered", value=round(ncor / max(ncov, 1), 4),
                         coverage=round(ncov / nq, 4), n=ncov, ci95=""))
        rows.append(dict(model=name, metric="analogy_acc_all", value=round(ncor / nq, 4),
                         coverage=round(ncov / nq, 4), n=nq, ci95=""))
        for cat, (n_, c_, k_) in per_cat.items():
            cat_rows.append(dict(model=name, category=cat, n=n_, covered=c_, correct=k_,
                                 acc_covered=round(k_ / c_, 4) if c_ else "", acc_all=round(k_ / n_, 4)))
        rows.append(dict(model=name, metric="vocab_size", value=len(kv.index_to_key), coverage="", n="", ci95=""))
        print(f"  {name:6s} rho={rho:.3f} (cov {n}/{len(pairs)})  analogy {ncor}/{nq} (covered {ncov})")

    def write_csv(path, recs, fields=None):
        fields = fields or list(recs[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(recs)

    write_csv(out / "embedding_eval.csv", rows, ["model", "metric", "value", "coverage", "n", "ci95"])
    write_csv(out / "analogy_by_category.csv", cat_rows)
    write_csv(out / "similarity_pairs.csv",
              [dict(word1=a, word2=b, **v) for (a, b), v in sim_table.items()],
              ["word1", "word2", "gold", "note"] + list(models))

    # nearest neighbours
    with open(out / "nearest_neighbours.md", "w", encoding="utf-8") as f:
        f.write("# Nearest neighbours of probe words\n\n"
                "Top 10 neighbours by cosine similarity (generated by `scripts/phase2/eval_embeddings.py`).\n"
                "`(OOV)` marks a probe the model has no trained vector for.\n\n")
        for probe in PROBES:
            f.write(f"## {probe} (train freq {counts.get(probe, 0)})\n\n")
            f.write("| model | neighbours |\n|---|---|\n")
            for name, idx in indexes.items():
                if probe not in idx.w2i and not is_fasttext(idx.kv):
                    f.write(f"| {name} | (OOV) |\n")
                    continue
                nn = idx.nearest(idx.vec(probe), k=11, exclude=(probe,))[:10]
                f.write(f"| {name} | " + ", ".join(f"{w} ({s:.2f})" for w, s in nn) + " |\n")
            f.write("\n")
        f.write("## OOV probes (FastText only)\n\n"
                "Damaged spellings that never occur in the training data.\n\n")

    # OOV robustness
    oov_rows, examples = oov_robustness(models, counts, seed=args.seed)
    write_csv(out / "oov_robustness.csv", oov_rows)
    with open(out / "nearest_neighbours.md", "a", encoding="utf-8") as f:
        f.write("| damage | correct | damaged (OOV) | model | neighbours of damaged form |\n|---|---|---|---|---|\n")
        for t, w, v, _ in examples:
            for name, idx in indexes.items():
                if is_fasttext(idx.kv):
                    nn = idx.nearest(idx.vec(v), k=5)
                    f.write(f"| {t} | {w} | {v} | {name} | " + ", ".join(f"{x} ({s:.2f})" for x, s in nn) + " |\n")

    # t-SNE
    family = setup_matplotlib()
    if family is None:
        print("  no Devanagari font found: t-SNE plots without labels")
    words = [w for w, _ in counts.most_common(args.tsne_words + 1) if w != "<NUM>"][:args.tsne_words]
    for name, kv in models.items():
        ws = [w for w in words if w in kv.key_to_index]
        tsne_plot(name, kv, ws, out / "plots" / f"tsne_{name}.png", labels=family is not None, seed=args.seed)
    print(f"  t-SNE plots written (font: {family})")

    # learning curve
    lc_rows = []
    lc_dir = mdir / "lc"
    if lc_dir.is_dir():
        from gensim.models import KeyedVectors
        full_tokens = sum(counts.values())
        for name in MODEL_ORDER:
            files = sorted(lc_dir.glob(f"{name}_*.kv"))
            if not files:
                continue
            entries = []
            for p in files:
                pct = int(p.stem.split("_")[1])
                size = (lc_dir / f"train_{pct}.size").read_text().split()
                entries.append((pct / 100, int(size[1]), KeyedVectors.load(str(p))))
            if name in models:
                entries.append((1.0, full_tokens, models[name]))
            for frac, ntok, kv in entries:
                rho, _, n = similarity(kv, pairs, n_boot=0)
                nq, ncov, ncor = analogy_summary(analogies(Index(kv), questions))
                rec = dict(model=name, fraction=frac, tokens=ntok, vocab=len(kv.index_to_key),
                           spearman_invocab=round(rho, 4), sim_coverage=round(n / len(pairs), 4),
                           analogy_acc_covered=round(ncor / max(ncov, 1), 4),
                           analogy_acc_all=round(ncor / nq, 4), analogy_coverage=round(ncov / nq, 4))
                if is_fasttext(kv):
                    rec["spearman_allpairs"] = round(similarity(kv, pairs, oov_ok=True, n_boot=0)[0], 4)
                lc_rows.append(rec)
                del kv
        if lc_rows:
            fields = list(lc_rows[0].keys()) + (["spearman_allpairs"] if "spearman_allpairs" not in lc_rows[0] else [])
            write_csv(out / "embedding_learning_curve.csv",
                      [{k: r.get(k, "") for k in fields} for r in lc_rows], fields)
            learning_curve_plot(lc_rows, out / "plots" / "embedding_learning_curve.png")
            print(f"  learning curve: {len(lc_rows)} points")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
