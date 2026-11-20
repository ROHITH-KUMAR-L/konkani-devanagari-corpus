"""Fill the Phase 1 report and cleaning log with the numbers of the latest pipeline run.

docs/templates/{report,data_cleaning_log}.md hold the text with {{placeholders}};
this script computes every placeholder from results/, metadata/ and data/eval/
and writes docs/report.md and docs/data_cleaning_log.md. Edit the templates,
not the generated files. An unknown placeholder is an error.

Usage (after scripts/run_phase1.sh):
    python scripts/build_report.py
"""
import csv
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

from corpus_utils import iter_docs, read_doc

ROOT = Path(__file__).resolve().parent.parent
CAT_NAMES = {"01_literature": "Literature", "02_news_media": "News", "03_education_science": "Education",
             "04_culture_history": "Culture/history", "05_public_information": "Public information",
             "06_web_articles": "Web (Wikipedia)", "07_oral_transcripts": "Oral transcripts"}
GLOSS = {"आनी": "and", "ह्या": "this, oblique", "आसा": "is", "एक": "one", "ना": "not", "ताका": "to him",
         "पूण": "but", "तो": "he/that", "ते": "they/that", "हे": "this/these", "खातीर": "for", "अशें": "thus",
         "आसात": "are", "जाता": "becomes", "हो": "this"}
VARIANTS = [("पूण", "पुण"), ("म्हूण", "म्हणून"), ("सगळ्यांत", "सगल्यांत"), ("हांव", "हाव")]
TEMPLATE_WORDS = ["गांवांत", "उपलब्ध", "सगळ्यांत", "लागीं", "अंतराचेर", "किलोमिटर"]


def n(x):
    return f"{x:,}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def rows(p, delim=","):
    with open(ROOT / p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim, quoting=csv.QUOTE_NONE if delim == "\t" else csv.QUOTE_MINIMAL))


def freq_table(p):
    with open(ROOT / p, encoding="utf-8") as f:
        return {r["type"]: (int(r["rank"]), int(r["freq"])) for r in csv.DictReader(f)}


def seg_eval(p):
    t = (ROOT / p).read_text(encoding="utf-8")
    b = re.search(r"gold boundaries (\d+), TP (\d+), FP (\d+), FN (\d+)", t)
    s = re.search(r"precision ([\d.]+)\s+recall ([\d.]+)\s+F1 ([\d.]+)", t)
    return {"boundaries": int(b.group(1)), "p": float(s.group(1)), "r": float(s.group(2)), "f1": float(s.group(3))}


def gold_paragraphs(p):
    return sum(1 for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#"))


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def values():
    V = {"date": dt.date.today().strftime("%d %b %Y")}
    S = [load("results/iter0/stats.json"), load("results/iter1/stats.json"),
         load("results/iter2/stats.json"), load("results/stats.json")]
    O = [s["overall"] for s in S]
    F, R = O[3], O[0]
    docs = rows("metadata/documents.csv")
    audit = rows("results/document_audit.csv")
    qc = {r["doc_id"]: r for r in rows("metadata/quality_checks.csv")}
    dedup = rows("metadata/dedup_log.csv")
    clean = rows("metadata/clean_log.csv")
    lens = {r["scope"]: r for r in rows("results/sentence_length_stats.csv")}
    split = load("results/split_summary.json")

    # ---- sources
    news = sorted(d["doc_id"] for d in docs if d["category"] == "02_news_media")
    dates = sorted(re.search(r"(\d{4})_(\d{2})_(\d{2})", d).groups() for d in news)
    V["news_issues"] = n(len(news))
    V["news_first"] = "-".join(dates[0]) if dates else "n/a"
    V["news_last"] = "-".join(dates[-1]) if dates else "n/a"
    st = Counter((d["category"], d["source_type"]) for d in docs)
    V["lit_books"] = st[("01_literature", "book_ocr")]
    V["ldcil_files"] = st[("01_literature", "ldcil_gold")]
    V["cult_books"] = st[("04_culture_history", "book_ocr")]
    V["wiki_articles"] = n(sum(int(re.search(r"\((\d+) articles\)", d["title"]).group(1))
                               for d in docs if d["source_type"] == "wikipedia"))
    desc = {"01_literature": f"Literature: {V['lit_books']} OCR'd books + {V['ldcil_files']} LDC-IL gold files",
            "02_news_media": "News: Bhaangarbhuin e-paper issues (OCR)",
            "03_education_science": "Education: NCERT Konkani textbook (OCR at 300 dpi)",
            "04_culture_history": f"Culture/history: {V['cult_books']} OCR'd books",
            "06_web_articles": f"Web: Konkani Wikipedia, {V['wiki_articles']} Devanagari articles in 20 shards"}
    c0 = S[0]["categories"]
    V["table_sources_iter0"] = md_table(["Source", "Docs", "Raw tokens"],
                                        [[desc.get(c, c), s["documents"], n(s["tokens"])] for c, s in sorted(c0.items())]
                                        + [["**Total**", f"**{S[0]['documents']}**", f"**{n(R['tokens'])}**"]])
    present = {c for c, s in S[3]["categories"].items() if s["tokens"] > 0}
    V["n_domains"] = len(present)
    V["empty_categories"] = ", ".join(f"{c[:2]} ({CAT_NAMES[c].lower()})" for c in CAT_NAMES if c not in present)

    # ---- summary
    V["fin_tokens"], V["fin_types"] = n(F["tokens"]), n(F["types"])
    V["fin_sentences"] = n(int(lens["overall"]["sentences"]))
    V["fin_docs"] = n(sum(1 for r in audit if int(r["tokens_iter3"]) > 0))
    V["raw_docs"] = n(len(audit))
    V["fin_nondeva_pct"] = pct(F["token_share"]["latin"] + F["token_share"]["other"], 2)
    V["fin_num_pct"] = pct(F["token_share"]["num"], 2)
    tot = sum(v["tokens"] for v in split["splits"].values())
    for s in ("train", "dev", "test"):
        V[f"{s}_pct"] = pct(split["splits"][s]["tokens"] / tot)
        V[f"{s}_docs"] = split["splits"][s]["documents"]

    # ---- OCR
    ocr = (ROOT / "results/ocr_quality/summary.txt").read_text(encoding="utf-8")
    grab = lambda k: re.search(re.escape(k) + r":\s+([\d.]+)%", ocr).group(1)
    V["ocr_cer_raw"], V["ocr_cer"] = grab("CER (raw OCR)"), grab("CER (punct spacing fixed)")
    V["ocr_wer_raw"], V["ocr_wer"] = grab("WER (raw OCR)"), grab("WER (punct spacing fixed)")

    # ---- segmentation
    dev, held = seg_eval("results/segmentation_eval.txt"), seg_eval("results/segmentation_eval_heldout.txt")
    V["seg_dev_paras"] = gold_paragraphs("data/eval/segmentation_gold.txt")
    V["seg_held_paras"] = gold_paragraphs("data/eval/segmentation_gold_heldout.txt")
    for k, e in (("dev", dev), ("held", held)):
        V[f"seg_{k}_bounds"] = e["boundaries"]
        for m in ("p", "r", "f1"):
            V[f"seg_{k}_{m}"] = f"{e[m]:.3f}"

    # ---- language audit
    lab = Counter(r["label"] for r in rows("data/eval/language_audit_sample.tsv", "\t"))
    na = sum(lab.values())
    V["audit_n"] = na
    V["audit_nonkok"] = sum(v for k, v in lab.items() if k.startswith("non_konkani"))
    V["audit_nonkok_pct"] = pct(V["audit_nonkok"] / na)
    V["audit_garbage_pct"] = pct(lab["ocr_garbage"] / na)
    V["audit_partial_pct"] = pct(lab["ocr_garbage_partial"] / na)
    V["audit_fragment_pct"] = pct(lab["konkani_fragment"] / na)
    V["audit_garbage"], V["audit_partial"] = lab["ocr_garbage"], lab["ocr_garbage_partial"]
    V["audit_fragment"], V["audit_dialect"] = lab["konkani_fragment"], lab["konkani_other_dialect_spelling"]

    # ---- iterations
    names = ["0 raw", "1 cleaned", "2 Konkani only", "3 deduplicated (final)"]
    for i, o in enumerate(O):
        V[f"tok{i}"] = n(o["tokens"])
    V["table_iterations"] = md_table(
        ["Iteration", "Tokens", "Types", "TTR", "Hapax (% types)", "Devanagari %", "Number %", "Latin+other %",
         "Types for 50% / 90% coverage", "Zipf slope", "Heaps β"],
        [[nm, n(o["tokens"]), n(o["types"]), f"{o['ttr']:.4f}", f"{n(o['hapax'])} ({pct(o['hapax_ratio_types'])}%)",
          pct(o["token_share"]["deva"], 2), pct(o["token_share"]["num"], 2),
          pct(o["token_share"]["latin"] + o["token_share"]["other"], 2),
          f"{n(o['coverage']['50'])} / {n(o['coverage']['90'])}", f"{s['zipf']['slope']:.3f}",
          f"{s['heaps']['beta']:.3f}"] for nm, o, s in zip(names, O, S)])
    cats = sorted(S[0]["categories"])
    body = []
    for c in cats:
        t = [s["categories"].get(c, {}).get("tokens", 0) for s in S]
        body.append([c] + [n(x) for x in t] + [f"{pct(t[3] / t[0])}%" if t[0] else "–"])
    V["table_category_iterations"] = md_table(["Category", "0 raw", "1 cleaned", "2 Konkani only", "3 final", "kept"], body)

    dpar = sum(int(r["exact_duplicate_paragraphs"]) for r in dedup)
    pin = sum(int(r["paragraphs_in"]) for r in dedup)
    V["dedup_paras"], V["dedup_pct"] = n(dpar), pct(dpar / pin)
    V["near_dup_docs"] = sum(1 for r in dedup if r["near_duplicate_of"])
    V["junk_lines"] = n(sum(int(r["junk_lines"]) for r in clean))
    V["short_paras"] = n(sum(int(r["short_paragraphs"]) for r in clean))
    for k in ("marathi", "hindi", "orthography", "numeric"):
        V[f"drop_{k}"] = n(sum(int(r[f"dropped_{k}"]) for r in qc.values()))
    dropped_docs = [r for r in qc.values() if r["doc_dropped"] == "1"]
    V["docs_dropped"] = len(dropped_docs)
    V["docs_dropped_list"] = ", ".join(f"`{r['doc_id']}` ({r['kok_markers']} Konkani vs {r['mar_markers']} Marathi / "
                                       f"{r['hin_markers']} Hindi markers)" for r in dropped_docs) or "none"
    hist = qc.get("cult_history", {})
    V["history_kok"], V["history_mar"] = hist.get("kok_markers", "?"), hist.get("mar_markers", "?")
    V["aspects_hin"] = qc.get("cult_some_aspects_of_konkani_culture_heritage", {}).get("hin_markers", "?")
    V["nataka_mar"] = qc.get("lit_nataka", {}).get("mar_markers", "?")

    # ---- iteration-0 observations
    f0, ff = freq_table("results/iter0/word_frequencies.csv"), freq_table("results/word_frequencies.csv")
    V["template_ranks"] = ", ".join(f"`{w}` (rank {f0[w][0]})" for w in TEMPLATE_WORDS if w in f0)
    wiki_lines = [l for c, d, p in iter_docs(ROOT / "data/interim/iter0") if c == "06_web_articles"
                  for l in read_doc(p).split("\n") if l.strip()]
    V["wiki_lines"], V["wiki_unique"] = n(len(wiki_lines)), n(len(set(wiki_lines)))
    V["raw_num_pct"], V["raw_num"] = pct(R["token_share"]["num"], 2), n(f0.get("<NUM>", (0, 0))[1])
    V["raw_latin_pct"] = pct(R["token_share"]["latin"] + R["token_share"]["other"], 2)
    V["raw_hapax_pct"], V["fin_hapax_pct"] = pct(R["hapax_ratio_types"]), pct(F["hapax_ratio_types"])
    V["fin_hapax"], V["fin_hapax_tok_pct"] = n(F["hapax"]), pct(F["hapax_ratio_tokens"])
    V["variants"] = "; ".join(f"`{a}` {n(ff.get(a, (0, 0))[1])} vs `{b}` {n(ff.get(b, (0, 0))[1])}" for a, b in VARIANTS)
    for i in (1,):
        V[f"it{i}_num_pct"] = pct(O[i]["token_share"]["num"], 2)
        V[f"it{i}_latin_pct"] = pct(O[i]["token_share"]["latin"] + O[i]["token_share"]["other"], 2)

    # ---- type-token / coverage
    V["table_typetoken"] = md_table(["", "Raw (iter 0)", "Final"], [
        ["Tokens N", n(R["tokens"]), n(F["tokens"])], ["Types V", n(R["types"]), n(F["types"])],
        ["TTR", f"{R['ttr']:.3f}", f"{F['ttr']:.3f}"],
        ["Hapax legomena", f"{n(R['hapax'])} ({pct(R['hapax_ratio_types'])}% of types)",
         f"{n(F['hapax'])} ({pct(F['hapax_ratio_types'])}% of types, {pct(F['hapax_ratio_tokens'])}% of tokens)"],
        ["Zipf slope (ranks 10–10,000)", f"{S[0]['zipf']['slope']:.3f}", f"{S[3]['zipf']['slope']:.3f}"],
        ["Heaps' law V = K·N^β", f"K {S[0]['heaps']['K']:.2f}, β {S[0]['heaps']['beta']:.3f}",
         f"K {S[3]['heaps']['K']:.2f}, β {S[3]['heaps']['beta']:.3f}"]])
    fc = {c: s for c, s in S[3]["categories"].items() if s["tokens"] > 0}
    V["table_categories"] = md_table(["Category", "Docs", "Tokens", "Share", "Types", "TTR", "Hapax % of types"], [
        [CAT_NAMES[c], s["documents"], n(s["tokens"]), f"{pct(s['share_of_corpus'])}%", n(s["types"]),
         f"{s['ttr']:.3f}", pct(s["hapax_ratio_types"])] for c, s in sorted(fc.items())])
    top = sorted(ff.items(), key=lambda x: x[1][0])[:11]
    V["top_words"] = ", ".join(f"`{w}`" + (f" ({GLOSS[w]})" if w in GLOSS else "") for w, _ in top)
    lv = ["10", "25", "50", "75", "90", "95", "99"]
    V["table_coverage"] = md_table(["Token coverage"] + [f"{l}%" for l in lv], [
        ["Types needed (final)"] + [n(F["coverage"][l]) for l in lv],
        ["% of vocabulary"] + [f"{100 * F['coverage'][l] / F['types']:.3g}" for l in lv],
        ["Types needed (raw)"] + [n(R["coverage"][l]) for l in lv]])

    # ---- lengths
    body = [["Overall"] + [n(int(lens["overall"]["sentences"]))] + [lens["overall"][k] for k in ("mean",)]
            + [f"{float(lens['overall']['p50']):.0f}", lens["overall"]["std"], f"{float(lens['overall']['p95']):.0f}"]]
    for c in sorted(fc):
        r = lens.get(c)
        if r:
            body.append([CAT_NAMES[c], n(int(r["sentences"])), r["mean"], f"{float(r['p50']):.0f}", r["std"],
                         f"{float(r['p95']):.0f}"])
    V["table_sentence_length"] = md_table(["Sentence length (tokens)", "Sentences", "Mean", "Median", "Std", "p95"], body)
    wl = rows("results/word_length_distribution.csv")

    def wstats(col):
        cnt = {int(r["length"]): int(r[col]) for r in wl}
        tot_ = sum(cnt.values())
        mean = sum(k * v for k, v in cnt.items()) / tot_
        run, med = 0, 0
        for k in sorted(cnt):
            run += cnt[k]
            if run >= tot_ / 2:
                med = k
                break
        return mean, med
    W = {c: wstats(c) for c in ("chars_token_weighted", "chars_type_weighted", "aksharas_token_weighted",
                                "aksharas_type_weighted")}
    V["table_word_length"] = md_table(["Word length", "Token-weighted mean (median)", "Type-weighted mean (median)"], [
        ["Characters (code points)", "%.2f (%d)" % W["chars_token_weighted"], "%.2f (%d)" % W["chars_type_weighted"]],
        ["Aksharas", "%.2f (%d)" % W["aksharas_token_weighted"], "%.2f (%d)" % W["aksharas_type_weighted"]]])
    V["chars_per_akshara"] = f"{W['chars_token_weighted'][0] / W['aksharas_token_weighted'][0]:.2f}"
    V["aks_tok"], V["aks_type"] = f"{W['aksharas_token_weighted'][0]:.1f}", f"{W['aksharas_type_weighted'][0]:.1f}"

    # ---- observations
    V["heaps_beta"] = f"{S[3]['heaps']['beta']:.2f}"
    V["cov90"], V["cov90_pct"] = n(F["coverage"]["90"]), f"{100 * F['coverage']['90'] / F['types']:.1f}"
    V["dedup_tok_pct"] = pct(1 - O[3]["tokens"] / O[2]["tokens"])
    V["dedup_tok"] = n(O[2]["tokens"] - O[3]["tokens"])
    V["dedup_types_lost"] = n(O[2]["types"] - O[3]["types"])
    V["cov50_before"], V["cov50_after"] = n(O[2]["coverage"]["50"]), n(F["coverage"]["50"])
    V["cov50_change"] = pct(F["coverage"]["50"] / O[2]["coverage"]["50"] - 1, 0)
    V["ttr_before"], V["ttr_after"] = f"{O[2]['ttr']:.3f}", f"{F['ttr']:.3f}"
    V["zipf_before"], V["zipf_after"] = f"{S[2]['zipf']['slope']:.2f}", f"{S[3]['zipf']['slope']:.2f}"
    med = {c: float(lens[c]["p50"]) for c in fc if c in lens}
    lo, hi = min(med, key=med.get), max(med, key=med.get)
    V["short_cat"], V["short_med"] = CAT_NAMES[lo], f"{med[lo]:.0f}"
    V["long_cat"], V["long_med"] = CAT_NAMES[hi], f"{med[hi]:.0f}"
    big = {c: s for c, s in fc.items() if s["tokens"] >= 100_000}
    lt = min(big, key=lambda c: big[c]["ttr"])
    V["lowttr_cat"], V["lowttr"] = CAT_NAMES[lt], f"{big[lt]['ttr']:.3f}"
    hh = sorted(big, key=lambda c: -big[c]["hapax_ratio_types"])[:2]
    V["highhapax_cats"] = " and ".join(CAT_NAMES[c] for c in hh)
    largest = max(fc, key=lambda c: fc[c]["share_of_corpus"])
    V["largest_cat"], V["largest_share"] = CAT_NAMES[largest], pct(fc[largest]["share_of_corpus"])
    b = split["balanced"]
    V["max_share"] = pct(split["max_share"], 0)
    V["bal_docs"], V["bal_tokens"] = b["documents"], n(b["tokens"])
    V["bal_shares"] = ", ".join(f"{CAT_NAMES[c].lower()} {pct(t / b['tokens'])}%" for c, t in b["category_tokens"].items())
    V["balance_status"] = (f"{V['largest_cat']} is {V['largest_share']}% of the corpus, above the {V['max_share']}% cap"
                           if fc[largest]["share_of_corpus"] > split["max_share"] else
                           f"No category is above the {V['max_share']}% cap; the largest is {V['largest_cat']} "
                           f"({V['largest_share']}%)")
    return V


def render(template, V):
    def sub(m):
        k = m.group(1)
        if k not in V:
            raise KeyError(f"unknown placeholder {{{{{k}}}}}")
        return str(V[k])
    return re.sub(r"\{\{(\w+)\}\}", sub, template)


def main():
    V = values()
    for name in ("report.md", "data_cleaning_log.md"):
        t = (ROOT / "docs/templates" / name).read_text(encoding="utf-8")
        head = f"<!-- Generated by scripts/build_report.py from docs/templates/{name}; edit the template. -->\n"
        (ROOT / "docs" / name).write_text(head + render(t, V), encoding="utf-8", newline="\n")
        print(f"docs/{name} written")


if __name__ == "__main__":
    main()
