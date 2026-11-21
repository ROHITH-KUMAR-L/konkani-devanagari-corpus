"""Shared helpers for the Phase 3 NER scripts: CoNLL I/O, suffix stripping, gazetteer."""
from collections import namedtuple
from pathlib import Path

LABELS = ["PER", "LOC", "ORG", "MISC", "DATE"]

Sentence = namedtuple("Sentence", "meta tokens tags")

# Case markers / postpositions that Konkani writes attached to a noun
# (गोंयांत, भारताचो, मुंबयच्या, पुलिसांक ...). Longest first so stripping is greedy.
# Used only to *match* gazetteer entries and as a CRF feature; tokens are never split.
CASE_SUFFIXES = sorted({
    "ांतल्यान", "ांतल्या", "ांतलो", "ांतली", "ांतलें", "ांतले", "ांत", "ात",
    "ेंतल्यान", "ेंतल्या", "ेंतलो", "ेंतली", "ेंतले", "ेंत",
    "तल्यान", "तल्या", "तलो", "तली", "तलें", "तले",
    "ाच्या", "ाचो", "ाची", "ाचें", "ाचे", "ाचेर", "ाक", "ान", "ार",
    "च्या", "चो", "ची", "चें", "चे", "चेर", "क", "न",
    "ेच्या", "ेचो", "ेची", "ेचें", "ेचे", "ेक", "ेन", "ेर",
    "ांच्या", "ांचो", "ांची", "ांचें", "ांचे", "ांक", "ांनी", "ांचेर", "ां", "नी",
    "कडेन", "ाकडेन", "खातीर", "भर", "य", "ूय", "ूच", "च",
}, key=len, reverse=True)


def strip_suffix(tok, min_stem=2):
    """Remove one attached case suffix (greedy, longest first); keep >= min_stem chars."""
    for s in CASE_SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= min_stem:
            return tok[: -len(s)]
    return tok


def read_conll(path):
    sents, meta, toks, tags = [], {}, [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            k, _, v = line[2:].partition(" = ")
            meta[k] = v
        elif not line.strip():
            if toks:
                sents.append(Sentence(meta, toks, tags))
            meta, toks, tags = {}, [], []
        else:
            t, tag = line.split("\t")
            toks.append(t)
            tags.append(tag)
    if toks:
        sents.append(Sentence(meta, toks, tags))
    return sents


def write_conll(path, sents, header=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            for h in header:
                f.write(f"# {h}\n")
            f.write("\n")
        for s in sents:
            for k, v in s.meta.items():
                f.write(f"# {k} = {v}\n")
            for t, g in zip(s.tokens, s.tags):
                f.write(f"{t}\t{g}\n")
            f.write("\n")


def spans(tags):
    """BIO tags -> list of (type, start, end_exclusive)."""
    out, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                out.append((cur[0], cur[1], i))
                cur = None
            if t != "O":
                cur = (t[2:], i)
    return out


# ------------------------------------------------------------------ gazetteer

def load_gazetteer(path):
    """TSV: TYPE<TAB>entry (space-separated tokens). Lines starting with # ignored."""
    gaz = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        typ, entry = line.split("\t")
        gaz[tuple(entry.split())] = typ
    return gaz


def gazetteer_from_sents(sents):
    """Entity strings seen in annotated sentences (surface + suffix-stripped last token)."""
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)
    for s in sents:
        for typ, a, b in spans(s.tags):
            toks = tuple(s.tokens[a:b])
            votes[toks][typ] += 1
            votes[toks[:-1] + (strip_suffix(toks[-1]),)][typ] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def gazetteer_tag(tokens, gaz, max_len=6):
    """Greedy longest-match tagging; last token may carry a case suffix."""
    tags = ["O"] * len(tokens)
    stems = [strip_suffix(t) for t in tokens]
    i = 0
    while i < len(tokens):
        hit = None
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            surf = tuple(tokens[i:i + n])
            stem = surf[:-1] + (stems[i + n - 1],)
            typ = gaz.get(surf) or gaz.get(stem)
            if typ:
                hit = (typ, n)
                break
        if hit:
            typ, n = hit
            tags[i] = "B-" + typ
            for j in range(i + 1, i + n):
                tags[j] = "I-" + typ
            i += n
        else:
            i += 1
    return tags
