"""NER baselines on the silver data: rule/gazetteer tagger and a CRF.

Systems (all evaluated on data/eval/ner/silver_test.conll, entity-level P/R/F1
with seqeval, conlleval-style matching):

  gaz_seed        seed gazetteer (data/eval/ner/gazetteer_seed.tsv) + DATE and
                  honorific rules; uses no annotated data at all
  gaz_seed_dev    gaz_seed + every entity string seen in silver_dev
  crf_dev         CRF (sklearn-crfsuite, L-BFGS, c1/c2 picked by 3-fold CV on
                  dev) trained on silver_dev only
  crf_dev_weak    CRF trained on silver_dev + pool sentences 301..N that were
                  auto-labelled by gaz_seed_dev (weak supervision)

It also runs 5-fold cross-validation of the crf_dev setup over all 300 silver
sentences, because a 150-sentence test set gives very wide error bars.

Outputs: results/phase3/ner_scores.csv, results/phase3/ner_cv.csv,
         results/phase3/ner_test_predictions.tsv,
         results/phase3/ner_token_confusion_matrix.csv (token-level diagnostic)
"""
import argparse
import csv
import random
import re
import sys
import warnings
from pathlib import Path

import sklearn_crfsuite
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_common import (LABELS, gazetteer_from_sents, gazetteer_tag, load_gazetteer,  # noqa: E402
                        read_conll, strip_suffix)

warnings.filterwarnings("ignore")
SEED = 2026

MONTHS = {
    "जानेवारी", "जानेवर", "फेब्रुवारी", "फेब्रेवारी", "फेब्रुवरी", "मार्च", "एप्रील", "एप्रिल",
    "जून", "जुलय", "जुलै", "ऑगस्ट", "आगोस्त", "सप्टेंबर", "सप्टेंबर", "सेंतेंब्र", "सेंतेंब्रा",
    "ऑक्टोबर", "ऑक्टोबर", "नोव्हेंबर", "नोवेंबर", "डिसेंबर",
}
WEEKDAYS = {"सोमार", "मंगळार", "बुधवार", "बुधवारा", "बिरेस्तार", "शुक्रार", "शुक्रा", "शेनवार", "आयतार"}
HONORIFICS = {"डॉ", "श्री", "प्रा", "सौ", "श्रीमती", "स्वामी", "ॲड", "ऍड", "कुमारी", "पंडित", "श्रीम", "कु", "शिरीमान"}
PER_TRIGGERS = {"हांणी", "हांचे", "हांच्या", "हांकां", "हांची", "हांचो", "हांचें", "हांणीं"}
STOP = PER_TRIGGERS | {"हे", "हें", "हो", "ही", "नांवाच्या", "आनी", "ह्या", "तांणी", "ताणें", "म्हणून", "म्हूण", "आसा", "जालें"}
NUM_RE = re.compile(r"^[0-9०-९]+([.,:/][0-9०-९]+)*$")
YEAR_RE = re.compile(r"^(1[0-9]|20)[0-9]{2}$|^(१[०-९]|२०)[०-९]{2}$")
LATIN_RE = re.compile(r"^[A-Za-z]+$")


def is_month(t):
    return t in MONTHS or strip_suffix(t) in MONTHS


def is_weekday(t):
    return t in WEEKDAYS or strip_suffix(t) in WEEKDAYS


def rule_tag(tokens, gaz):
    tags = gazetteer_tag(tokens, gaz)
    n = len(tokens)

    def free(a, b):
        return all(tags[k] == "O" for k in range(a, b))

    def put(a, b, typ):
        tags[a] = "B-" + typ
        for k in range(a + 1, b):
            tags[k] = "I-" + typ

    # DATE: [day] month [year], weekday, bare 4-digit year
    for i, t in enumerate(tokens):
        if is_month(t) and tags[i] == "O":
            a, b = i, i + 1
            if i > 0 and NUM_RE.match(tokens[i - 1]) and tags[i - 1] == "O":
                a = i - 1
            if i + 1 < n and YEAR_RE.match(tokens[i + 1]) and tags[i + 1] == "O":
                b = i + 2
            if a < i or b > i + 1:          # a month name alone is too ambiguous (जून)
                put(a, b, "DATE")
    for i, t in enumerate(tokens):
        if tags[i] == "O" and (is_weekday(t) or YEAR_RE.match(t)):
            put(i, i + 1, "DATE")
    # PER: honorific + up to 3 tokens, or up to 2 tokens before हांणी/हांचे...
    for i, t in enumerate(tokens):
        if t in HONORIFICS:
            j = i + 1
            while j < n and j - i <= 3 and tokens[j] not in STOP and tokens[j] not in HONORIFICS \
                    and tags[j] == "O" and not NUM_RE.match(tokens[j]):
                j += 1
            if j > i + 1:
                put(i + 1, j, "PER")
        if t in PER_TRIGGERS and i > 0:
            a = i
            while a > 0 and i - a < 2 and tags[a - 1] == "O" and tokens[a - 1] not in STOP \
                    and tokens[a - 1] not in HONORIFICS and not NUM_RE.match(tokens[a - 1]):
                a -= 1
            if a < i:
                put(a, i, "PER")
    return tags


# ------------------------------------------------------------------ CRF

def word_feats(tok, prefix, gtag):
    stem = strip_suffix(tok)
    return {
        f"{prefix}w": tok,
        f"{prefix}stem": stem,
        f"{prefix}suf": tok[len(stem):],
        f"{prefix}gaz": gtag[2:] if gtag != "O" else "O",
    }


def sent2features(tokens, seed_gaz):
    gtags = gazetteer_tag(tokens, seed_gaz)
    feats = []
    for i, t in enumerate(tokens):
        f = {
            "bias": 1.0,
            "w": t,
            "stem": strip_suffix(t),
            "suf1": t[-1:], "suf2": t[-2:], "suf3": t[-3:], "suf4": t[-4:],
            "pre1": t[:1], "pre2": t[:2], "pre3": t[:3],
            "len": min(len(t), 10),
            "num": bool(NUM_RE.match(t)),
            "year": bool(YEAR_RE.match(t)),
            "latin": bool(LATIN_RE.match(t)),
            "devanagari": any("\u0900" <= ch <= "\u097f" for ch in t),
            "nukta": "़" in t,
            "virama": "्" in t,
            "month": is_month(t),
            "weekday": is_weekday(t),
            "gaz": gtags[i],
            "prev_honorific": i > 0 and tokens[i - 1] in HONORIFICS,
            "next_per_trigger": i + 1 < len(tokens) and tokens[i + 1] in PER_TRIGGERS,
            "next2_per_trigger": i + 2 < len(tokens) and tokens[i + 2] in PER_TRIGGERS,
        }
        if i == 0:
            f["BOS"] = True
        else:
            f.update(word_feats(tokens[i - 1], "-1:", gtags[i - 1]))
        if i == len(tokens) - 1:
            f["EOS"] = True
        else:
            f.update(word_feats(tokens[i + 1], "+1:", gtags[i + 1]))
        if i > 1:
            f.update(word_feats(tokens[i - 2], "-2:", gtags[i - 2]))
        if i + 2 < len(tokens):
            f.update(word_feats(tokens[i + 2], "+2:", gtags[i + 2]))
        feats.append(f)
    return feats


def train_crf(train, seed_gaz, c1, c2, iters=200):
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=iters,
                               all_possible_transitions=True)
    crf.fit([sent2features(s.tokens, seed_gaz) for s in train], [s.tags for s in train])
    return crf


def predict(crf, sents, seed_gaz):
    return crf.predict([sent2features(s.tokens, seed_gaz) for s in sents])


def micro_f1(gold, pred):
    rep = classification_report(gold, pred, output_dict=True, zero_division=0)
    return rep["micro avg"]["f1-score"]


def kfold(items, k, seed):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    for f in range(k):
        te = set(idx[f::k])
        yield [items[i] for i in idx if i not in te], [items[i] for i in sorted(te)]


GRID = [(0.05, 0.01), (0.1, 0.05), (0.1, 0.1), (0.3, 0.05), (0.5, 0.1), (1.0, 0.01)]


def pick_hparams(train, seed_gaz):
    best = None
    for c1, c2 in GRID:
        scores = []
        for tr, va in kfold(train, 3, SEED):
            crf = train_crf(tr, seed_gaz, c1, c2)
            scores.append(micro_f1([s.tags for s in va], predict(crf, va, seed_gaz)))
        m = sum(scores) / len(scores)
        if best is None or m > best[0]:
            best = (m, c1, c2)
    return best


def report_rows(system, train_desc, gold, pred):
    rep = classification_report(gold, pred, output_dict=True, zero_division=0)
    rows = []
    for k in LABELS + ["micro avg", "macro avg"]:
        if k not in rep:
            rows.append([system, train_desc, k, 0, 0, 0, 0])
            continue
        r = rep[k]
        rows.append([system, train_desc, k, round(r["precision"], 4), round(r["recall"], 4),
                     round(r["f1-score"], 4), int(r["support"])])
    return rows


def write_token_confusion(path, gold, pred):
    """Write a token-level BIO confusion matrix; entity-level scores remain primary."""
    labels = ["O"] + [f"{p}-{label}" for label in LABELS for p in ("B", "I")]
    gold_flat = [tag for sent in gold for tag in sent]
    pred_flat = [tag for sent in pred for tag in sent]
    matrix = confusion_matrix(gold_flat, pred_flat, labels=labels)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold\\pred"] + labels)
        for label, row in zip(labels, matrix):
            w.writerow([label] + row.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner-dir", default="data/eval/ner")
    ap.add_argument("--outdir", default="results/phase3")
    ap.add_argument("--weak-n", type=int, default=1500, help="pool sentences to auto-label")
    a = ap.parse_args()

    d = Path(a.ner_dir)
    dev = read_conll(d / "silver_dev.conll")
    test = read_conll(d / "silver_test.conll")
    allsil = read_conll(d / "silver_all.conll")
    pool = read_conll(d / "pool.conll")
    silver_ids = {s.meta["sent_id"] for s in allsil}
    seed_gaz = load_gazetteer(d / "gazetteer_seed.tsv")
    dev_gaz = dict(seed_gaz)
    dev_gaz.update(gazetteer_from_sents(dev))
    gold = [s.tags for s in test]

    rows, preds = [], {}

    preds["gaz_seed"] = [rule_tag(s.tokens, seed_gaz) for s in test]
    rows += report_rows("gaz_seed", "none (seed lists + rules)", gold, preds["gaz_seed"])

    preds["gaz_seed_dev"] = [rule_tag(s.tokens, dev_gaz) for s in test]
    rows += report_rows("gaz_seed_dev", "dev lexicon", gold, preds["gaz_seed_dev"])

    cv, c1, c2 = pick_hparams(dev, seed_gaz)
    print(f"CRF hparams c1={c1} c2={c2} (3-fold dev F1 {cv:.3f})")
    crf = train_crf(dev, seed_gaz, c1, c2)
    preds["crf_dev"] = predict(crf, test, seed_gaz)
    rows += report_rows("crf_dev", f"silver_dev ({len(dev)} sents) c1={c1} c2={c2}", gold, preds["crf_dev"])

    from ner_common import Sentence
    weak = [Sentence(s.meta, s.tokens, rule_tag(s.tokens, dev_gaz))
            for s in pool if s.meta["sent_id"] not in silver_ids][: a.weak_n]
    crf_w = train_crf(dev + weak, seed_gaz, c1, c2)
    preds["crf_dev_weak"] = predict(crf_w, test, seed_gaz)
    rows += report_rows("crf_dev_weak", f"silver_dev + {len(weak)} gazetteer-labelled pool sents",
                        gold, preds["crf_dev_weak"])

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ner_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "training_data", "entity", "precision", "recall", "f1", "support"])
        w.writerows(rows)

    # 5-fold CV of crf_dev and gaz_seed over all 300 silver sentences
    cv_rows = []
    for fold, (tr, te) in enumerate(kfold(allsil, 5, SEED), 1):
        g = [s.tags for s in te]
        crf_f = train_crf(tr, seed_gaz, c1, c2)
        lex = dict(seed_gaz)
        lex.update(gazetteer_from_sents(tr))
        cv_rows.append([fold, len(tr), len(te),
                        round(micro_f1(g, [rule_tag(s.tokens, seed_gaz) for s in te]), 4),
                        round(micro_f1(g, [rule_tag(s.tokens, lex) for s in te]), 4),
                        round(micro_f1(g, predict(crf_f, te, seed_gaz)), 4)])
    with open(out / "ner_cv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "train_sents", "test_sents", "gaz_seed_f1", "gaz_seed_trainlex_f1", "crf_f1"])
        w.writerows(cv_rows)
        means = [round(float(sum(r[i] for r in cv_rows)) / len(cv_rows), 4) for i in (3, 4, 5)]
        w.writerow(["mean", "", ""] + means)

    with open(out / "ner_test_predictions.tsv", "w", encoding="utf-8") as f:
        sysnames = list(preds)
        f.write("sent_id\ttoken\tgold\t" + "\t".join(sysnames) + "\n")
        for k, s in enumerate(test):
            for i, t in enumerate(s.tokens):
                f.write(f"{s.meta['sent_id']}\t{t}\t{s.tags[i]}\t" +
                        "\t".join(preds[n][k][i] for n in sysnames) + "\n")

    write_token_confusion(out / "ner_token_confusion_matrix.csv", gold, preds["crf_dev"])

    for r in rows:
        if r[2] == "micro avg":
            print(f"{r[0]:14s} P={r[3]:.3f} R={r[4]:.3f} F1={r[5]:.3f}")
    print("5-fold CV means (gaz_seed, gaz_seed+trainlex, crf):", means)


if __name__ == "__main__":
    main()
