"""Shared helpers for the Phase 2 scripts (tokenisers and n-gram LMs).

Not a command-line script. It loads the sentence-per-line splits, tokenises
every sentence with corpus_utils.tokenize (punctuation removed, numbers ->
<NUM>) and stores each split as two small integer arrays:

    pids  int32  one "provisional id" per word token, all sentences concatenated
    lens  int32  number of word tokens in each sentence

Provisional ids index one shared list of word types (train types first, then
types seen only in dev/test), so no split ever holds a Python list of strings.
Sentences with no word token (pure punctuation) are skipped.
"""
import sys
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from corpus_utils import tokenize  # noqa: E402

SPLITS = ("train", "dev", "test")


class Corpus:
    """Tokenised splits sharing one type list (see module docstring)."""

    def __init__(self, split_dir, splits=SPLITS):
        self.types = []          # provisional id -> word string
        self._index = {}         # word string -> provisional id
        self.data = {}           # split -> (pids, lens)
        for name in splits:
            self.data[name] = self._load(Path(split_dir) / f"{name}.txt")
        self._index = None       # free the dict; types list is enough from here on

    def _load(self, path):
        pids, lens = array("i"), array("i")
        index, types = self._index, self.types
        with open(path, encoding="utf-8") as f:
            for line in f:
                toks = tokenize(line)
                if not toks:
                    continue
                for t in toks:
                    i = index.get(t)
                    if i is None:
                        i = index[t] = len(types)
                        types.append(t)
                    pids.append(i)
                lens.append(len(toks))
        return np.frombuffer(pids, dtype=np.int32).copy(), np.frombuffer(lens, dtype=np.int32).copy()

    def counts(self, split):
        """Token count of every provisional id in one split (length = len(types))."""
        return np.bincount(self.data[split][0], minlength=len(self.types))

    def sentences(self, split):
        """Yield each sentence of a split as a list of word strings."""
        pids, lens = self.data[split]
        types, pos = self.types, 0
        for n in lens:
            yield [types[i] for i in pids[pos:pos + n]]
            pos += n


def prefix(split_data, frac):
    """First `frac` of the sentences of one split (used by the learning curve)."""
    pids, lens = split_data
    m = int(round(len(lens) * frac))
    return pids[: int(lens[:m].sum())], lens[:m]
