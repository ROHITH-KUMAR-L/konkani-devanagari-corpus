"""Sentence segmentation and assembly of the final processed corpus.

Reads the final cleaned document tree (one paragraph per line) and writes:
    data/processed/by_category/<cat>/corpus_clean.txt   paragraphs, one per line
    data/processed/by_category/<cat>/sentences.tsv      doc_id, sent_id, n_tokens, sentence
    data/processed/combined/konkani_devanagari_clean.txt
    data/processed/combined/sentences.txt               one sentence per line
    data/processed/combined/sentences.tsv
    data/processed/combined/documents.jsonl             {"doc_id", "category", "text"}

Rules (PRD FR-S1..S4):
  - Boundaries after । ॥ ? ! and '.', plus any closing quotes/brackets that follow.
  - A verse number "॥ १ ॥" is one boundary.
  - A newspaper dateline ("पणजी:", "'मडगांव:") starts a new sentence; a list
    number ("... ना 13. समासांत") starts a new sentence before the number.
  - '.' is NOT a boundary in: decimals (3.5), ellipses (...), known abbreviations
    (डॉ. श्री. क्र. किमी. ...), and initials: a one-akshara word before '.'
    (के.के., आर्.पी.आय., पी. नायक) unless it is a common one-akshara word
    that often ends a sentence (ना, हो, जा, ...).

Usage:
    python scripts/segment_sentences.py --in data/interim/iter3 --out data/processed
    python scripts/segment_sentences.py --eval data/eval/segmentation_gold.txt
"""
import argparse
import json
import re
from pathlib import Path

from corpus_utils import CATEGORIES, aksharas, iter_docs, read_doc, tokenize, write_doc

ABBREVIATIONS = set("""डॉ डा श्री श्रीम सौ कुं कु प्रा प्रो क्र किमी कि मी सेंमी सेमी मिमी चौ इ स ता नं रु
वि अॅड ॲड ज मा सं पृ पा आ ले उ दि जि ऑ स्व कै मिं""".split()) | set(
    # English letter names used as initials (के.आर., जी.एन., पी.ओ.)
    """ए बी सी डी ई इ एफ जी एच आय जे के एल एम एन ओ पी क्यू आर एस टी यू व्ही डब्ल्यू एक्स वाय झेड""".split())
FINAL_MONO = set("ना हो जा घे दी ये यो खा वा गे गो रे तो ती ते तें ही हे को".split())
TERMINATOR = re.compile(r"॥\s*[०-९0-9]+\s*॥|[।॥?!]+|\.{2,}|\.")
CLOSERS = re.compile(r"[”’\"')\]]+")


def is_boundary(text, m, start=0):
    tok = m.group()
    if tok != ".":
        return not tok.startswith("..")             # ellipsis is not a boundary
    before, after = text[:m.start()], text[m.end():]
    if re.search(r"[0-9०-९]$", before) and re.match(r"[0-9०-९]", after):
        return False                                 # decimal
    if re.fullmatch(r"\s*[(\[]?[0-9०-९]{1,3}", text[start:m.start()]):
        return False                                 # list number "12." at sentence start
    w = re.search(r"([^\s.()\"“”'‘’]+)$", before)
    if not w:
        return True
    word = w.group(1)
    if word in ABBREVIATIONS:
        return False
    if len(aksharas(word)) == 1 and word not in FINAL_MONO and not re.fullmatch(r"[0-9०-९]+", word):
        return False                                 # initial
    return True


DATELINE_PLACES = """पणजी मडगांव म्हापशें म्हापसा पेडणें फोंडें दिल्ली सत्तरी दिवचल डिचोली माशेल सांखळी वाळपय बाळपय
मुरगांव वास्को मुंबय केरी-सत्तरी सांगें दिवची काणकोण केपें कुडचडें पर्वरी हणजूण कळंगुट शिवोली बेळगांव कारवार
बार्देस तिसवाडी साश्टी कुंकळ्ळी नावेली मंड्या बंगळूर""".split()
# a newspaper dateline ("'पणजी: ...") starts a new sentence even when the headline before it has no full stop
DATELINE = re.compile(r"(?<!\S)['‘’]?(?:" + "|".join(map(re.escape, DATELINE_PLACES)) + r")\s?[:ः]\s")
# "... केलें ना 13. समासांत ..." : a list number that starts the next item
LIST_NUMBER = re.compile(r"(?<=\s)[(\[]?[0-9०-९]{1,3}$")


def split_sentences(para):
    cuts, start = set(), 0
    for m in TERMINATOR.finditer(para):
        if m.start() < start:
            continue
        if m.group() == "." and re.match(r"\s*[^\s0-9०-९]", para[m.end():]):
            n = LIST_NUMBER.search(para, start, m.start())
            if n and tokenize(para[start:n.start()]):
                cuts.add(n.start())                  # boundary before the number, not after it
                start = n.start()
                continue
        if not is_boundary(para, m, start):
            continue
        end = m.end()
        c = CLOSERS.match(para, end)
        if c:
            end = c.end()
        cuts.add(end)
        start = end
    for d in DATELINE.finditer(para):
        if tokenize(para[:d.start()]):
            cuts.add(d.start())
    out, prev = [], 0
    for c in sorted(cuts) + [len(para)]:
        piece = para[prev:c].strip()
        if piece:
            out.append(piece)
        prev = c
    merged = []
    for s in out:                                    # punctuation-only pieces join the previous sentence
        if not tokenize(s):
            if merged:
                merged[-1] += " " + s
        else:
            merged.append(s)
    return merged


def build(inp, out):
    out = Path(out)
    comb = out / "combined"
    files = {}
    for c in CATEGORIES:
        d = out / "by_category" / c
        d.mkdir(parents=True, exist_ok=True)
        files[c] = (open(d / "corpus_clean.txt", "w", encoding="utf-8", newline="\n"),
                    open(d / "sentences.tsv", "w", encoding="utf-8", newline="\n"))
    comb.mkdir(parents=True, exist_ok=True)
    f_clean = open(comb / "konkani_devanagari_clean.txt", "w", encoding="utf-8", newline="\n")
    f_sent = open(comb / "sentences.txt", "w", encoding="utf-8", newline="\n")
    f_tsv = open(comb / "sentences.tsv", "w", encoding="utf-8", newline="\n")
    f_docs = open(comb / "documents.jsonl", "w", encoding="utf-8", newline="\n")
    header = "doc_id\tsent_id\tn_tokens\tsentence\n"
    f_tsv.write(header)
    for _, tsv in files.values():
        tsv.write(header)
    n_sent = 0
    for cat, doc_id, path in iter_docs(inp):
        text = read_doc(path).strip()
        if not text:
            continue
        f_docs.write(json.dumps({"doc_id": doc_id, "category": cat, "text": text}, ensure_ascii=False) + "\n")
        clean, tsv = files[cat]
        clean.write(text + "\n")
        f_clean.write(text + "\n")
        sid = 0
        for para in text.split("\n"):
            for s in split_sentences(para):
                sid += 1
                row = f"{doc_id}\t{sid}\t{len(tokenize(s))}\t{s}\n"
                tsv.write(row)
                f_tsv.write(row)
                f_sent.write(s + "\n")
        n_sent += sid
    for fh in [f_clean, f_sent, f_tsv, f_docs] + [h for pair in files.values() for h in pair]:
        fh.close()
    print(f"{n_sent:,} sentences -> {comb}")


def evaluate(gold_path):
    """Gold file: one paragraph per line, sentence boundaries marked with ' || '."""
    tp = fp = fn = 0
    errors = []
    for line in Path(gold_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        gold_sents = [s.strip() for s in line.split("||")]
        para = " ".join(gold_sents)

        def offsets(sents):
            # boundary = number of word tokens before the cut, so the side a
            # quote mark or bracket lands on does not count as an error
            pos, cuts = 0, set()
            for s in sents[:-1]:
                pos += len(tokenize(s))
                cuts.add(pos)
            return cuts
        g, p = offsets(gold_sents), offsets(split_sentences(para))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if g != p:
            errors.append(" || ".join(split_sentences(para)))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"boundaries_gold": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/interim/iter3")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--eval", help="gold file to evaluate against instead of building the corpus")
    ap.add_argument("--eval-out", default="results/segmentation_eval.txt")
    args = ap.parse_args()
    if args.eval:
        res, errors = evaluate(args.eval)
        lines = [f"Sentence segmentation evaluation on {args.eval}",
                 f"gold boundaries {res['boundaries_gold']}, TP {res['tp']}, FP {res['fp']}, FN {res['fn']}",
                 f"precision {res['precision']:.4f}  recall {res['recall']:.4f}  F1 {res['f1']:.4f}",
                 "", f"paragraphs with errors ({len(errors)}), system output:"] + errors
        Path(args.eval_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines[:3]))
    else:
        build(args.inp, args.out)


if __name__ == "__main__":
    main()
