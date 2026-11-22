"""Konkani NLP demo: one local web page that runs the project's own code and models.

Tabs (each backed by the module named):
  Language check     validate_konkani.judge          Konkani / Marathi / Hindi marker counts, orthography
  Sentences          clean_text + segment_sentences  sentence splitting, tokens, aksharas
  Tokenisers         SentencePiece models            word vs BPE-32k vs Unigram-16k pieces, fertility
  Autocomplete       word trigram counts (train)     next-word suggestions, stupid backoff
  Similar words      FastText (ft25)                 nearest neighbours and analogies
  Generate           Phase 3 LSTM                    continue a prompt with the neural LM
  Names (NER)        ner_baselines.rule_tag          gazetteer + rule tagger (PER, LOC, ORG, MISC, DATE)
  Corpus search      data/processed sentences        word frequency, rank and example sentences
  Translate buttons  IndicTrans2 (AI4Bharat)         Konkani -> English per sentence; an OUTSIDE pretrained
                                                     model (we have no parallel data), not project work

Everything is loaded lazily on first use and cached in data/models/demo_*.
Needs the Phase 2/3 environment (.venv-phase2) and the trained models in data/models.
Translation also needs `transformers<4.50` and `IndicTransToolkit`, and Hugging Face access to the
gated model ai4bharat/indictrans2-indic-en-dist-200M (accept its terms, then `.venv-phase2/bin/hf auth login`).

Usage:
    .venv-phase2/bin/python scripts/demo/app.py            # then open http://127.0.0.1:8050
    .venv-phase2/bin/python scripts/demo/app.py --port 8060
"""
import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# scripts/phase2/tokenizers.py would shadow Hugging Face's `tokenizers` package (needed by transformers
# for translation), so import the real package before the project folders go on sys.path.
try:
    import tokenizers as _hf_tokenizers  # noqa: F401
except ImportError:
    pass
for sub in ("scripts", "scripts/phase2", "scripts/phase3"):
    sys.path.insert(0, str(ROOT / sub))

from corpus_utils import NUM, aksharas, devanagari_ratio, tokenize  # noqa: E402

MODELS = ROOT / "data/models"
HTML = Path(__file__).with_name("index.html")
_lock = threading.Lock()
_cache = {}


def cached(name, build):
    """Build each heavy resource once, under a lock, and report how long it took."""
    with _lock:
        if name not in _cache:
            t = time.time()
            _cache[name] = build()
            print(f"loaded {name} in {time.time() - t:.1f}s", flush=True)
        return _cache[name]


# ------------------------------------------------------------------ language check

def api_langid(text):
    from validate_konkani import judge
    out = []
    for para in [p for p in text.split("\n") if p.strip()]:
        reason, n, bad, (k, m, h) = judge(para)
        ratio, _ = devanagari_ratio(para)
        label = {"marathi": "Marathi", "hindi": "Hindi", "orthography": "OCR garbage",
                 "numeric": "Number table"}.get(reason, "Konkani" if k else "Konkani (no markers)")
        out.append(dict(text=para, label=label, kept=reason is None, konkani=k, marathi=m, hindi=h,
                        tokens=n, invalid=bad, deva=round(ratio * 100, 1)))
    return {"paragraphs": out}


# ------------------------------------------------------------------ sentences

def api_segment(text):
    from clean_text import fix_line, normalise
    from segment_sentences import split_sentences
    sents = []
    for para in normalise(text).split("\n"):
        para = fix_line(para)
        for s in split_sentences(para) if para else []:
            words = [w for w in tokenize(s) if w != NUM]
            sents.append(dict(text=s, tokens=len(tokenize(s)),
                              aksharas=[{"w": w, "a": aksharas(w)} for w in words[:12]]))
    return {"sentences": sents}


# ------------------------------------------------------------------ tokenisers

def _spm(name):
    import sentencepiece as spm
    return cached(name, lambda: spm.SentencePieceProcessor(model_file=str(MODELS / f"{name}.model")))


def api_tokenize(text):
    words = tokenize(text)
    line = " ".join(words)
    rows = [dict(name="Whole words", pieces=words, fertility=1.0)]
    for name, label in (("spm_bpe_32000", "BPE 32k"), ("spm_unigram_16000", "Unigram 16k"),
                        ("spm_bpe_8000", "BPE 8k")):
        pieces = _spm(name).encode(line, out_type=str)
        rows.append(dict(name=label, pieces=pieces, fertility=round(len(pieces) / max(1, len(words)), 2)))
    return {"words": len(words), "rows": rows}


# ------------------------------------------------------------------ autocomplete (word trigram, stupid backoff)

class Trigram:
    """Sorted int64 n-gram keys with counts; a context's continuations are one searchsorted range."""

    def __init__(self):
        path = MODELS / "demo_ngram.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self.vocab = list(z["vocab"])
            self.uni, self.k2, self.c2, self.k3, self.c3 = z["uni"], z["k2"], z["c2"], z["k3"], z["c3"]
        else:
            self._build(path)
        self.V = len(self.vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.top_uni = np.argsort(-self.uni)[:2000]

    def _build(self, path):
        index, vocab, ids = {"<s>": 0}, ["<s>"], []
        with open(ROOT / "data/splits/train.txt", encoding="utf-8") as f:
            for line in f:
                toks = tokenize(line)
                if not toks:
                    continue
                ids.append(0)
                for t in toks:
                    i = index.get(t)
                    if i is None:
                        i = index[t] = len(vocab)
                        vocab.append(t)
                    ids.append(i)
        a = np.array(ids, np.int64)
        V = len(vocab)
        self.vocab = vocab
        self.uni = np.bincount(a, minlength=V)
        b2 = a[:-1] * V + a[1:]
        b2 = b2[a[1:] != 0]                               # never predict <s>
        self.k2, self.c2 = np.unique(b2, return_counts=True)
        t3 = (a[:-2] * V + a[1:-1]) * V + a[2:]
        t3 = t3[(a[2:] != 0) & (a[1:-1] != 0)]           # no n-gram across a sentence start
        self.k3, self.c3 = np.unique(t3, return_counts=True)
        np.savez(path, vocab=np.array(vocab), uni=self.uni, k2=self.k2, c2=self.c2, k3=self.k3, c3=self.c3)

    def _range(self, keys, counts, base):
        lo, hi = np.searchsorted(keys, [base, base + self.V])
        return keys[lo:hi] - base, counts[lo:hi]

    def suggest(self, history, prefix="", k=10):
        h = [self.index.get(w, -1) for w in (["<s>", "<s>"] + history)[-2:]]
        scores = {}

        def add(ids, cnts, weight):
            if len(cnts) == 0:
                return
            total = cnts.sum()
            for i, c in zip(ids.tolist(), cnts.tolist()):
                w = self.vocab[i]
                if (not prefix or w.startswith(prefix)) and w != "<s>":
                    scores[w] = max(scores.get(w, 0), weight * c / total)

        sources = []
        if h[0] >= 0 and h[1] >= 0:
            add(*self._range(self.k3, self.c3, (h[0] * self.V + h[1]) * self.V), 1.0)
            sources.append("trigram")
        if len(scores) < k and h[1] >= 0:
            add(*self._range(self.k2, self.c2, h[1] * self.V), 0.4)
            sources.append("bigram")
        if len(scores) < k:
            add(self.top_uni, self.uni[self.top_uni], 0.16)
            sources.append("unigram")
        best = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [dict(word=w, score=round(s, 4)) for w, s in best], sources


def api_complete(text):
    model = cached("trigram", Trigram)
    words = tokenize(text)
    prefix = ""
    if text and not text[-1].isspace() and words:
        prefix = words.pop()                            # still typing this word
    sugg, sources = model.suggest(words, prefix)
    return {"suggestions": sugg, "prefix": prefix, "backoff": sources}


# ------------------------------------------------------------------ embeddings

def _kv():
    from gensim.models import KeyedVectors
    name = "ft25" if (MODELS / "ft25.kv").exists() else "ft"
    return cached("fasttext", lambda: KeyedVectors.load(str(MODELS / f"{name}.kv")))


def api_similar(word, a="", b="", c=""):
    kv = _kv()
    out = {}
    if word:
        word = word.strip()
        known = word in kv.key_to_index
        out["neighbours"] = [dict(word=w, sim=round(s, 3)) for w, s in kv.most_similar(word, topn=12)]
        out["known"] = known
    if a and b and c:
        res = kv.most_similar(positive=[b.strip(), c.strip()], negative=[a.strip()], topn=5)
        out["analogy"] = [dict(word=w, sim=round(s, 3)) for w, s in res]
    return out


# ------------------------------------------------------------------ LSTM generation

class Generator:
    def __init__(self):
        import torch
        import sentencepiece as spm
        import neural_lm as nl
        self.torch, self.nl = torch, nl
        self.sp = spm.SentencePieceProcessor(model_file=str(MODELS / "spm_unigram_16000.model"))
        V = self.sp.vocab_size()
        counts_path = MODELS / "demo_piece_counts.npy"
        if counts_path.exists():
            counts = np.load(counts_path)
        else:                                       # same unit counts the trainer used for the adaptive softmax
            from common import Corpus
            corpus = Corpus(ROOT / "data/splits", splits=("train",))
            flat, off, length = nl.spm_table(corpus, self.sp)
            d = nl.build_split(corpus, "train", flat, off, length, 0, corpus.counts("train") >= 2)
            counts = np.bincount(d.seq, minlength=V)
            np.save(counts_path, counts)
        args = SimpleNamespace(ff_context=3, max_len=128)
        self.model = nl.make_model("lstm", V, counts, args)
        ckpt = MODELS / "extended/phase3_lstm.pt"
        if not ckpt.exists():
            ckpt = MODELS / "phase3_lstm.pt"
        if not ckpt.exists():
            raise FileNotFoundError("no LSTM checkpoint yet: run `bash scripts/run_phase3_lm.sh` first "
                                    "(it saves data/models/phase3_lstm.pt)")
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.ckpt = str(ckpt.relative_to(ROOT))

    def generate(self, prompt, n=3, temperature=0.8, top_k=40, max_units=40):
        torch, nl = self.torch, self.nl
        prompt_ids = self.sp.encode(" ".join(tokenize(prompt)), out_type=int) if prompt.strip() else []
        gen = torch.Generator().manual_seed(int(time.time()))
        outs = []
        with torch.no_grad():
            for _ in range(n):
                seq = [nl.BOS] + prompt_ids
                for _ in range(max_units):
                    h = nl.forward(self.model, torch.tensor([seq]))[0, -1:]
                    logits = self.model.out.full_logprob(h)[0] / temperature
                    logits[nl.UNK] = -float("inf")
                    v, i = torch.topk(logits, top_k)
                    nxt = int(i[torch.multinomial(torch.softmax(v, -1), 1, generator=gen)])
                    if nxt == nl.EOS:
                        break
                    seq.append(nxt)
                outs.append(self.sp.decode(seq[1 + len(prompt_ids):]))
        return outs


def api_generate(prompt, temperature=0.8):
    g = cached("lstm", Generator)
    t = max(0.3, min(1.5, float(temperature)))
    return {"prompt": " ".join(tokenize(prompt)), "continuations": g.generate(prompt, temperature=t),
            "checkpoint": g.ckpt}


# ------------------------------------------------------------------ NER

def ner_tokens(line):
    """Same tokens as the NER data: corpus words with punctuation dropped and numbers kept as written."""
    from corpus_utils import TOKEN_RE
    return [m.group() for m in TOKEN_RE.finditer(line)]


def _ner():
    from ner_baselines import rule_tag
    from ner_common import gazetteer_from_sents, load_gazetteer, read_conll
    gaz = load_gazetteer(ROOT / "data/eval/ner/gazetteer_seed.tsv")
    dev = ROOT / "data/eval/ner/silver_dev.conll"
    if dev.exists():
        gaz = {**gazetteer_from_sents(read_conll(dev)), **gaz}
    return rule_tag, gaz


def api_ner(text):
    rule_tag, gaz = cached("ner", _ner)
    out = []
    for line in [l for l in text.split("\n") if l.strip()]:
        toks = ner_tokens(line)
        tags = rule_tag(toks, gaz)
        out.append([dict(t=t, tag=g) for t, g in zip(toks, tags)])
    return {"sentences": out, "gazetteer_entries": len(gaz)}


# ------------------------------------------------------------------ corpus search

def _corpus():
    freq = {}
    with open(ROOT / "results/word_frequencies.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            rank, typ, n, rel, cum = line.rstrip("\n").split(",")[:5]
            freq[typ] = (int(rank), int(n), float(cum))
    sents = []
    with open(ROOT / "data/processed/combined/sentences.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            doc, _, _, s = line.rstrip("\n").split("\t", 3)
            sents.append((doc, s))
    return freq, sents


def api_search(word, limit=25):
    freq, sents = cached("corpus", _corpus)
    word = word.strip()
    pat = re.compile(r"(?<![ऀ-ॿ‌‍])" + re.escape(word) + r"(?![ऀ-ॿ‌‍])")
    hits, total = [], 0
    for doc, s in sents:
        if word in s and pat.search(s):
            total += 1
            if len(hits) < limit:
                hits.append(dict(doc=doc, text=s))
    rank, n, cum = freq.get(word, (None, 0, None))
    return {"word": word, "rank": rank, "count": n, "sentences_total": total, "hits": hits,
            "corpus_sentences": len(sents)}


# ------------------------------------------------------------------ translation (outside model)

MT_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"


class Translator:
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor
        self.torch = torch
        try:
            self.tok = AutoTokenizer.from_pretrained(MT_MODEL, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, trust_remote_code=True).eval()
        except OSError as e:
            if "gated" in str(e) or "401" in str(e):
                raise FileNotFoundError(
                    f"translation model {MT_MODEL} is gated: accept its terms on huggingface.co, "
                    "then run `.venv-phase2/bin/hf auth login` and restart the demo") from None
            raise
        self.ip = IndicProcessor(inference=True)

    def translate(self, sents):
        batch = self.ip.preprocess_batch(sents, src_lang="gom_Deva", tgt_lang="eng_Latn")
        inp = self.tok(batch, truncation=True, padding="longest", return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inp, num_beams=4, max_length=256, use_cache=True)
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(dec, lang="eng_Latn")


def api_translate(sentences):
    sents = [s.strip() for s in sentences if s.strip()][:40]      # keep one request to a few seconds
    if not sents:
        return {"translations": []}
    mt = cached("translator", Translator)
    out = []
    for i in range(0, len(sents), 8):
        out += mt.translate(sents[i:i + 8])
    return {"translations": out, "model": MT_MODEL}


# ------------------------------------------------------------------ server

API = {"langid": api_langid, "segment": api_segment, "tokenize": api_tokenize, "complete": api_complete,
       "similar": api_similar, "generate": api_generate, "ner": api_ner, "search": api_search,
       "translate": api_translate}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        name = self.path.removeprefix("/api/")
        fn = API.get(name)
        if not fn:
            return self._send(404, json.dumps({"error": f"unknown endpoint {name}"}), "application/json")
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            res = fn(**args)
            self._send(200, json.dumps(res, ensure_ascii=False), "application/json; charset=utf-8")
        except FileNotFoundError as e:
            self._send(200, json.dumps({"error": str(e)}, ensure_ascii=False), "application/json; charset=utf-8")
        except KeyError as e:
            self._send(200, json.dumps({"error": f"not in the model's vocabulary: {e}"}, ensure_ascii=False),
                       "application/json; charset=utf-8")
        except Exception as e:  # report to the page instead of dropping the connection
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False),
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--warm", action="store_true", help="load every model at start instead of on first use")
    args = ap.parse_args()
    if args.warm:
        api_tokenize("घर"), api_ner("गोंय"), _kv(), cached("trigram", Trigram), cached("corpus", _corpus)
        try:
            cached("lstm", Generator)
        except FileNotFoundError as e:
            print(f"Generate tab unavailable: {e}", flush=True)
        try:
            cached("translator", Translator)
        except (FileNotFoundError, ImportError) as e:
            print(f"Translation unavailable: {e}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Konkani NLP demo on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
