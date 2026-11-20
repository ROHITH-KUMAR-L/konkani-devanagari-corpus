"""Cleaning iteration 2: keep only Konkani paragraphs with well-formed Devanagari.

Input/output are document trees with one paragraph per line (output of clean_text.py).

Per paragraph (PRD FR-N6, FR-N8):
  - Language ID with marker words. A paragraph is labelled Marathi or Hindi and
    dropped when it has >= 2 Marathi/Hindi markers and more of them than
    Konkani markers. Paragraphs with no markers are kept.
  - Orthography: a token is invalid if it starts with a vowel sign, virama,
    nukta or anusvara, has two vowel signs in a row, or a doubled virama/nukta.
    A paragraph with > 15% invalid tokens, or a letter-digit-letter mix
    (OCR garbage), is dropped.
  - Numeric: a paragraph whose tokens are > 40% numbers (tables, census rows) is dropped.
Per document: if the whole document has fewer than 1 Konkani marker per 10
Marathi/Hindi markers it is dropped as non-Konkani.

Writes one row per document to metadata/quality_checks.csv.

Usage:
    python scripts/validate_konkani.py --in data/interim/iter1 --out data/interim/iter2
"""
import argparse
import csv
import re
from collections import Counter

from corpus_utils import NUM, iter_docs, read_doc, tokenize, write_doc

KONKANI = set("""आसा आसात आसलो आसली आसलें आशिल्लो आशिल्ली आशिल्लें आनी हांव हांवें म्हजो म्हजी म्हजें
म्हाका तुका ताका तिका तांकां जाल्यार जाल्ल्या जालो जाली जालें न्हय कितें खंय आमी तुमी पूण पुण उपरांत
खातीर थावन कडेन हांगा थंय केन्ना अशें तशें ताचो ताची ताचें तिचो करतात म्हळें म्हणटात आयलो वचून येवन
हांणी तांणी जावन घेवन दिवन आसून आशिल्ल्या""".split())
MARATHI = set("""आहे आहेत आहोत आणि मी माझा माझी माझे माझ्या नाही नाहीत होते होता होती झाले झाला झाली
मध्ये आम्ही तुम्ही आपण साठी त्यांनी त्याला तिला मला तुला केले पाहिजे येथे तेथे""".split())
HINDI = set("है हैं और में नहीं लिए यह वह था थे थी भी गया किया रहा हुआ हुए को से".split())

MATRA = "ऺऻा-ौॎॏॕ-ॗॢॣ"
BAD_START = re.compile(rf"^[{MATRA}ऀ-ः़्]")
BAD_INSIDE = re.compile(rf"[{MATRA}][{MATRA}]|््|़़")
LETTER_DIGIT_MIX = re.compile(r"[क-ह][0-9०-९]+[क-हा-ौ]")

MAX_INVALID = 0.15
MAX_NUMERIC = 0.40
MIN_DOC_KOK_RATIO = 0.1


def markers(tokens):
    return (sum(t in KONKANI for t in tokens), sum(t in MARATHI for t in tokens),
            sum(t in HINDI for t in tokens))


def invalid_token(t):
    return bool(BAD_START.search(t) or BAD_INSIDE.search(t))


def judge(para):
    """Return (reason or None, n_tokens, n_invalid, (k, m, h))."""
    toks = tokenize(para)
    words = [t for t in toks if t != NUM]
    k, m, h = markers(words)
    bad = sum(invalid_token(t) for t in words)
    if m + h >= 2 and m + h > k:
        return ("marathi" if m >= h else "hindi"), len(toks), bad, (k, m, h)
    if words and bad / len(words) > MAX_INVALID or LETTER_DIGIT_MIX.search(para):
        return "orthography", len(toks), bad, (k, m, h)
    if toks and (len(toks) - len(words)) / len(toks) > MAX_NUMERIC:
        return "numeric", len(toks), bad, (k, m, h)
    return None, len(toks), bad, (k, m, h)


FIELDS = ["doc_id", "category", "paragraphs_in", "tokens_in", "kok_markers", "mar_markers", "hin_markers",
          "dropped_marathi", "dropped_hindi", "dropped_orthography", "dropped_numeric", "doc_dropped",
          "invalid_token_rate", "paragraphs_out", "tokens_out"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="metadata/quality_checks.csv")
    args = ap.parse_args()
    rows, total = [], Counter()
    for cat, doc_id, path in iter_docs(args.inp):
        paras = [p for p in read_doc(path).split("\n") if p.strip()]
        st = Counter()
        kept = []
        for p in paras:
            reason, n, bad, (k, m, h) = judge(p)
            st.update(tokens_in=n, invalid=bad, kok=k, mar=m, hin=h)
            if reason:
                st[f"dropped_{reason}"] += 1
            else:
                kept.append(p)
                st["tokens_out"] += n
        doc_dropped = st["kok"] < MIN_DOC_KOK_RATIO * (st["mar"] + st["hin"])
        if doc_dropped:
            kept, st["tokens_out"] = [], 0
        write_doc(f"{args.out}/{cat}/{doc_id}.txt", "\n".join(kept) + ("\n" if kept else ""))
        row = {"doc_id": doc_id, "category": cat, "paragraphs_in": len(paras), "tokens_in": st["tokens_in"],
               "kok_markers": st["kok"], "mar_markers": st["mar"], "hin_markers": st["hin"],
               "dropped_marathi": st["dropped_marathi"], "dropped_hindi": st["dropped_hindi"],
               "dropped_orthography": st["dropped_orthography"], "dropped_numeric": st["dropped_numeric"],
               "doc_dropped": int(doc_dropped),
               "invalid_token_rate": f"{st['invalid'] / st['tokens_in']:.4f}" if st["tokens_in"] else "0",
               "paragraphs_out": len(kept), "tokens_out": st["tokens_out"]}
        rows.append(row)
        total.update({k: v for k, v in row.items() if isinstance(v, int)})
    with open(args.report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("; ".join(f"{k} {total[k]:,}" for k in FIELDS[2:] if k in total))
    for r in rows:
        if r["doc_dropped"]:
            print(f"document dropped as non-Konkani: {r['doc_id']} "
                  f"(markers kok {r['kok_markers']}, mar {r['mar_markers']}, hin {r['hin_markers']})")


if __name__ == "__main__":
    main()
