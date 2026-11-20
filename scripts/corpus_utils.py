"""Shared helpers for the Phase 1 pipeline: document trees, tokenisation, aksharas.

Every pipeline stage reads and writes a *document tree*:

    <root>/<category>/<doc_id>.txt     one paragraph per line, UTF-8, LF

so each stage is `--in <tree> --out <tree>` and the analysis script can run on
any iteration (iter0 = raw, iter1 = cleaned, ...).
"""
import re
import unicodedata
from pathlib import Path

CATEGORIES = [
    "01_literature", "02_news_media", "03_education_science", "04_culture_history",
    "05_public_information", "06_web_articles", "07_oral_transcripts",
]

# --------------------------------------------------------------- characters

DEVA_LETTER = r"ऀ-ॣॱ-ॿ"           # letters, signs, nukta, virama (no danda, no digits)
DEVA_DIGITS = "०१२३४५६७८९"
ZW = "‌‍"                                     # ZWNJ, ZWJ (meaningful inside Devanagari words)

TOKEN_RE = re.compile(
    rf"(?P<num>[0-9{DEVA_DIGITS}]+(?:[.,:/][0-9{DEVA_DIGITS}]+)*)"
    rf"|(?P<deva>[{DEVA_LETTER}][{DEVA_LETTER}{ZW}]*)"
    rf"|(?P<latin>[A-Za-z]+)"
    rf"|(?P<other>[^\W\d_]+)"
)
NUM = "<NUM>"


def tokenize(text, keep_kind=False):
    """Word tokens with punctuation removed and every number mapped to <NUM>.

    Tokens of other scripts (Latin, Tamil, ...) are kept so the raw analysis
    shows the noise; cleaning removes them later.
    """
    out = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        tok = NUM if kind == "num" else m.group()
        out.append((tok, kind) if keep_kind else tok)
    return out


def is_deva_letter(ch):
    o = ord(ch)
    return (0x0900 <= o <= 0x0963 or 0x0971 <= o <= 0x097F) and unicodedata.category(ch)[0] in "LM"


def devanagari_ratio(text):
    """Share of alphabetic characters (letters + combining marks) that are Devanagari."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0, 0
    deva = sum(is_deva_letter(c) for c in letters)
    return deva / len(letters), deva


# --------------------------------------------------------------- aksharas

_C = "क-हक़-य़ॸ-ॿ"          # consonants
_V = "ऄ-औॠॡॲ-ॷ"          # independent vowels
_MATRA = "ऺऻा-ौॎॏॕ-ॗॢॣ"
_MOD = "ऀ-ः"                                  # candrabindu, anusvara, visarga
_NUKTA, _VIRAMA = "़", "्"
AKSHARA_RE = re.compile(
    rf"(?:[{_C}]{_NUKTA}?{_VIRAMA}[{ZW}]?)*[{_C}]{_NUKTA}?(?:{_VIRAMA}[{ZW}]?|[{_MATRA}]*)[{_MOD}]*"
    rf"|[{_V}][{_MATRA}]*[{_MOD}]*"
    rf"|."
)


def aksharas(word):
    """Split a Devanagari word into aksharas (orthographic syllables), e.g. क्षेत्र -> क्षे|त्र."""
    return AKSHARA_RE.findall(word)


# --------------------------------------------------------------- document trees

def iter_docs(root):
    """Yield (category, doc_id, path) for every document in a tree, sorted."""
    root = Path(root)
    for cat in CATEGORIES:
        d = root / cat
        if d.is_dir():
            for p in sorted(d.glob("*.txt")):
                yield cat, p.stem, p


def read_doc(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_doc(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def paragraphs(text):
    return [p for p in text.split("\n") if p.strip()]
