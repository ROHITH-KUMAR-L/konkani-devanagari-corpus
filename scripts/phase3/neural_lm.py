"""Neural language models for the Konkani corpus (Phase 3, FR-P3-1).

Models (PyTorch, CPU-friendly; a GPU is used only if torch.cuda.is_available())
  ffnn         Bengio-style feed-forward LM: embeddings of the previous
               --ff-context units, concatenated -> tanh layer (512) -> 256
  lstm         LSTM LM: embedding 256 -> 1-layer LSTM 512, dropout 0.2
  transformer  small decoder-only Transformer: 4 pre-LN layers, d=256, 4 heads,
               FF 1024, learned positions, context --max-len pieces
  char_lstm    the LSTM above over characters instead of subword pieces
Subword models end in an adaptive softmax over the 16k pieces (exact and
normalised, several times cheaper on CPU than a full softmax); the char model
uses a plain softmax. The network body runs under bfloat16 autocast unless
--no-bf16; the output layer is always float32.

Units and evaluation are those of Phase 2 (scripts/phase2), so numbers are
directly comparable with the n-gram LMs:
  * text = data/splits/{train,dev,test}.txt tokenised with corpus_utils.tokenize
    (punctuation removed, numbers -> <NUM>), loaded with phase2/common.Corpus
  * subword models use the Phase 2 SentencePiece Unigram-16k model
    (data/models/spm_unigram_16000.model; trained here with the Phase 2
    settings if it is missing)
  * every sentence is modelled on its own: input <s> u1..um, targets u1..um </s>
    (no context crosses a sentence boundary, as in the n-gram LMs)
  * per-word perplexity = 2 ** (-sum log2 P / (word tokens + sentences)), the
    same normalisation as the Phase 2 subword LM; per-unit perplexity and
    bits per character (bpc) are reported too. Characters = the characters of
    the words, one space between words and one end-of-sentence symbol per
    sentence; <NUM> counts as one character.
  * "in-vocabulary" rows score only tokens whose word has train count >= 2 (plus
    </s>), exactly the events of the Phase 2 word LMs.

Training: AdamW, linear warm-up, sentences bucketed by length into batches of
about --batch-tokens units, long sentences cut into --max-len chunks.
Early stopping: dev per-word perplexity on a fixed random sample of
--dev-eval-sentences dev sentences is measured every --eval-every steps; the
learning rate is halved when it does not improve and training stops after
--patience evaluations without improvement, after --max-epochs, or when the
model's time budget (--budget name=minutes) runs out. The best checkpoint is
then scored on the full test split (--final-splits dev test adds dev) and used
for sampling.

Outputs (results/phase3/)
  neural_perplexity.csv   final metrics per model (best dev-sample ppl, test)
  training_log.csv        train loss and dev perplexity at every evaluation
  comparison.csv          neural models vs the n-gram baselines (if
                          --baseline-dir holds the Phase 2 CSVs for these splits)
  samples.md              sampled sentences per model
  plots/loss_curves.png, plots/comparison.png
  data/models/phase3_<model>.pt   best checkpoints (git-ignored)

Usage
  python scripts/phase3/neural_lm.py [--splits data/splits] [--out results/phase3]
      [--models ffnn lstm transformer char_lstm]
      [--budget ffnn=2.5 lstm=7 transformer=7 char_lstm=3.5]
      [--max-train-sentences 0] [--max-epochs 5] [--seed 2026]
      [--baseline-dir results/phase3/ngram_baseline]
  python scripts/phase3/neural_lm.py --prepare-only   # only make sure the SPM model exists
"""
import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2"))
from common import ROOT, Corpus  # noqa: E402

UNK, BOS, EOS = 0, 1, 2          # SentencePiece default ids, also used for the char vocab
NUM = "<NUM>"
NUM_CHAR = ""              # <NUM> as one private-use character in the char model
LOG2E = 1 / math.log(2)
AMP = {"enabled": False}       # bfloat16 autocast for the network body (set by --bf16)


# ------------------------------------------------------------------ data

class SplitData:
    """One split as a flat unit stream, sentence by sentence.

    seq     int64 targets of all sentences concatenated (units of each sentence then EOS)
    soff    start of each sentence in seq;  slen  its number of targets (units + EOS)
    group   for every target, the index of the "word unit" it belongs to
            (each word token is one word unit, each EOS another)
    invocab per word unit: word has train count >= 2, or it is EOS
    """

    def __init__(self, seq, slen, group, invocab, n_words, n_chars):
        self.seq, self.slen, self.group, self.invocab = seq, slen, group, invocab
        self.soff = np.zeros(len(slen), np.int64)
        self.soff[1:] = np.cumsum(slen)[:-1]
        self.n_words, self.n_chars = n_words, n_chars
        self.n_word_units = n_words + len(slen)


def build_split(corpus, split, flat, off, length, first_skip, invocab_type):
    """Gather per-token unit sequences from a CSR table (flat, off, length) per type.

    first_skip: number of leading units to drop for the first word of a sentence
    (1 for the char model, whose table stores a leading space per word).
    The last entry of the table must be the single unit [EOS].
    """
    pids, lens = corpus.data[split]
    nsent, nwords = len(lens), len(pids)
    sid = np.repeat(np.arange(nsent), lens)
    first = np.zeros(nwords, bool)
    first[np.cumsum(lens) - lens] = True
    w_off = off[pids] + first * first_skip
    w_len = length[pids] - first * first_skip
    # word units: words of a sentence, then its EOS
    total = nwords + nsent
    wu_word = np.arange(nwords) + sid
    wu_eos = np.cumsum(lens) + np.arange(nsent)
    u_off = np.empty(total, np.int64)
    u_len = np.empty(total, np.int64)
    u_off[wu_word], u_len[wu_word] = w_off, w_len
    u_off[wu_eos], u_len[wu_eos] = len(flat) - 1, 1
    starts = np.cumsum(u_len) - u_len
    idx = np.repeat(u_off - starts, u_len) + np.arange(int(u_len.sum()))
    seq = flat[idx].astype(np.int64)
    group = np.repeat(np.arange(total), u_len)
    invocab = np.ones(total, bool)
    invocab[wu_word] = invocab_type[pids]
    slen = np.bincount(np.repeat(np.arange(nsent), lens + 1), weights=u_len, minlength=nsent).astype(np.int64)
    return SplitData(seq, slen, group, invocab, nwords, None)


def char_table(corpus, vocab=None):
    """CSR table of ' ' + characters for every word type (+ final [EOS] entry)."""
    strings = [" " + (NUM_CHAR if t == NUM else t) for t in corpus.types]
    if vocab is None:
        train_types = np.flatnonzero(corpus.counts("train"))
        chars = sorted(set("".join(strings[i] for i in train_types)))
        vocab = ["<unk>", "<s>", "</s>"] + chars
    index = {c: i for i, c in enumerate(vocab)}
    length = np.array([len(s) for s in strings], np.int64)
    off = np.zeros(len(strings), np.int64)
    off[1:] = np.cumsum(length)[:-1]
    flat = np.fromiter((index.get(c, UNK) for s in strings for c in s), np.int64, count=int(length.sum()))
    return np.append(flat, EOS), off, length, vocab


def spm_table(corpus, sp):
    enc = sp.encode(corpus.types, out_type=int)
    length = np.array([len(e) for e in enc], np.int64)
    off = np.zeros(len(enc), np.int64)
    off[1:] = np.cumsum(length)[:-1]
    flat = np.fromiter((p for e in enc for p in e), np.int64, count=int(length.sum()))
    return np.append(flat, EOS), off, length


def ensure_spm(corpus, path, seed=13, max_sentences=500_000):
    """Train the Phase 2 Unigram-16k model with the Phase 2 settings if it is missing."""
    path = Path(path)
    if path.exists():
        return
    from tokenizers import train_spm   # scripts/phase2/tokenizers.py (on sys.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.parent / "spm_train_text_phase3.txt"
    n_sent = len(corpus.data["train"][1])
    keep = np.ones(n_sent, bool)
    if n_sent > max_sentences:
        keep[:] = False
        keep[np.random.default_rng(seed).choice(n_sent, max_sentences, replace=False)] = True
    with open(text, "w", encoding="utf-8", newline="\n") as f:
        for k, words in zip(keep, corpus.sentences("train")):
            if k:
                f.write(" ".join(words) + "\n")
    stem = str(path)[: -len(".model")]
    algo, size = Path(stem).name.split("_")[1:3]
    print(f"training SentencePiece {algo} {size} on {int(keep.sum())} sentences -> {path}")
    train_spm(text, stem, algo, int(size), 4)
    text.unlink()


# ------------------------------------------------------------------ models

class Output(nn.Module):
    """Output distribution over the V units.

    Large vocabularies (subword pieces) use an adaptive softmax (Grave et al.
    2017): frequent units in a head, rarer ones in two tail clusters with smaller
    projections. It is an exact, normalised distribution, so perplexities are
    exact; it only makes the CPU cost of the 16k-way softmax several times
    smaller. Units are ranked by train frequency for the cluster assignment.
    Small vocabularies (characters) use a plain softmax.
    """

    def __init__(self, d, V, counts, cutoffs=(2000, 6000)):
        super().__init__()
        self.V = V
        self.adaptive = V > 2 * cutoffs[-1]
        if self.adaptive:
            order = np.argsort(-counts, kind="stable")
            rank = np.empty(V, np.int64)
            rank[order] = np.arange(V)
            self.register_buffer("rank", torch.from_numpy(rank))
            self.asm = nn.AdaptiveLogSoftmaxWithLoss(d, V, list(cutoffs), div_value=2.0, head_bias=True)
        else:
            self.lin = nn.Linear(d, V)

    def target_logprob(self, h, y):
        """Natural-log probability of targets y (N) given hidden states h (N x d)."""
        with torch.autocast(h.device.type, enabled=False):    # output layer always in float32
            return self._target_logprob(h.float(), y)

    def _target_logprob(self, h, y):
        if self.adaptive:
            return self.asm(h, self.rank[y]).output
        return F.log_softmax(self.lin(h), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)

    def full_logprob(self, h):
        """N x V log-probabilities in the original unit-id order."""
        h = h.float()
        if self.adaptive:
            return self.asm.log_prob(h)[:, self.rank]
        return F.log_softmax(self.lin(h), -1)


class FFNNLM(nn.Module):
    def __init__(self, V, counts, d=128, ctx=3, hidden=512, dropout=0.1):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(V + 1, d, padding_idx=V)
        self.hid = nn.Linear(ctx * d, hidden)
        self.proj = nn.Linear(hidden, 256)
        self.drop = nn.Dropout(dropout)
        self.out = Output(256, V, counts)

    def forward(self, x):
        xp = F.pad(x, (self.ctx - 1, 0), value=BOS)       # shorter history at sentence start = <s> padding
        win = xp.unfold(1, self.ctx, 1)                     # B x T x ctx
        e = self.drop(self.emb(win).flatten(2))
        return self.proj(self.drop(torch.tanh(self.hid(e))))


class LSTMLM(nn.Module):
    def __init__(self, V, counts, d=256, hidden=512, layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d, padding_idx=V)
        self.rnn = nn.LSTM(d, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = Output(hidden, V, counts)

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        return self.drop(h)


class TransformerLM(nn.Module):
    def __init__(self, V, counts, d=256, heads=4, layers=4, ff=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(V + 1, d, padding_idx=V)
        self.pos = nn.Embedding(max_len, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out = Output(d, V, counts)

    def forward(self, x):
        T = x.size(1)
        assert T <= self.max_len, T
        h = self.drop(self.emb(x) + self.pos(torch.arange(T, device=x.device)))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.norm(self.enc(h, mask=mask, is_causal=True))


def make_model(name, V, counts, args):
    if name == "ffnn":
        return FFNNLM(V, counts, ctx=args.ff_context)
    if name in ("lstm", "char_lstm"):
        return LSTMLM(V, counts)
    if name == "transformer":
        return TransformerLM(V, counts, max_len=args.max_len)
    raise ValueError(name)


def forward(model, x):
    """Hidden states; the body runs under bfloat16 autocast if enabled, the output layer never does."""
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=AMP["enabled"]):
        return model(x)


def target_logprob(model, x, y):
    """Natural-log probabilities of the valid (y >= 0) targets, flattened."""
    h = forward(model, x)
    valid = y >= 0
    return model.out.target_logprob(h[valid], y[valid])


# ------------------------------------------------------------------ batching

def chunks(data, sent_idx, max_len):
    """(sentence, start, length) for every <= max_len chunk of the given sentences."""
    out = []
    for i in sent_idx:
        n = int(data.slen[i])
        for a in range(0, n, max_len):
            out.append((i, a, min(max_len, n - a)))
    return np.array(out, np.int64).reshape(-1, 3)


def make_batch(data, rows, V, device):
    T = int(rows[:, 2].max())
    x = np.full((len(rows), T), V, np.int64)       # V = padding id
    y = np.full((len(rows), T), -100, np.int64)
    for r, (i, a, n) in enumerate(rows):
        s = data.soff[i] + a
        y[r, :n] = data.seq[s: s + n]
        if a == 0:
            x[r, 0] = BOS
            x[r, 1:n] = data.seq[s: s + n - 1]
        else:
            x[r, :n] = data.seq[s - 1: s + n - 1]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def bucket_batches(ch, batch_tokens, rng):
    """Group chunks of similar length into batches of about batch_tokens units; shuffled."""
    order = np.argsort(ch[:, 2] + rng.random(len(ch)), kind="stable")
    batches, cur, cur_max = [], [], 0
    for k in order:
        n = int(ch[k, 2])
        if cur and max(cur_max, n) * (len(cur) + 1) > batch_tokens:
            batches.append(np.array(cur))
            cur, cur_max = [], 0
        cur.append(k)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(np.array(cur))
    rng.shuffle(batches)
    return batches


# ------------------------------------------------------------------ scoring

@torch.no_grad()
def score(model, data, sent_idx, V, device, eval_tokens=16384):
    """log2 P of every target of the given sentences, in canonical (seq) order.

    Returns (positions into data.seq, log2 probs). Sentences longer than a
    Transformer's max_len are scored with a sliding window (stride max_len/2).
    """
    model.eval()
    max_len = getattr(model, "max_len", None)
    sent_idx = np.asarray(sent_idx)
    pos = np.concatenate([np.arange(data.soff[i], data.soff[i] + data.slen[i]) for i in sent_idx])
    lp = np.empty(len(data.seq), np.float64)
    short = sent_idx if max_len is None else sent_idx[data.slen[sent_idx] <= max_len]
    long_ = np.array([], np.int64) if max_len is None else sent_idx[data.slen[sent_idx] > max_len]
    rows = np.stack([short, np.zeros_like(short), data.slen[short]], 1) if len(short) else np.zeros((0, 3), np.int64)
    rows = rows[np.argsort(rows[:, 2], kind="stable")]
    k = 0
    while k < len(rows):          # rows are sorted by length, so the last row of a batch is the longest
        j = k + 1
        while j < len(rows) and rows[j, 2] * (j - k + 1) <= eval_tokens:
            j += 1
        b = rows[k:j]
        k = j
        x, y = make_batch(data, b, V, device)
        g = target_logprob(model, x, y).cpu().numpy() * LOG2E    # row-major over valid targets
        # each row's valid targets are its first n positions, so g is the rows' targets concatenated
        dst = np.concatenate([np.arange(data.soff[i], data.soff[i] + n) for i, _, n in b])
        lp[dst] = g
    stride = (max_len or 2) // 2
    for i in long_:
        n = int(data.slen[i])
        s0 = int(data.soff[i])
        full = np.concatenate([[BOS], data.seq[s0: s0 + n - 1]])
        for a in range(0, n, stride):
            b_end = min(n, a + stride)
            w0 = max(0, b_end - max_len)
            x = torch.from_numpy(full[w0:b_end]).unsqueeze(0).to(device)
            h = forward(model, x)[0, a - w0: b_end - w0]
            tgt = torch.from_numpy(data.seq[s0 + a: s0 + b_end]).to(device)
            lp[s0 + a: s0 + b_end] = model.out.target_logprob(h, tgt).cpu().numpy() * LOG2E
    return pos, lp[pos]


def metrics(data, pos, lp, sent_idx=None):
    """Per-unit, per-word, in-vocabulary and OOV-only perplexity and bpc."""
    tot = lp.sum()
    if sent_idx is None:
        n_wu, n_chars = data.n_word_units, data.n_chars
    else:
        n_wu = len(np.unique(data.group[pos]))
        n_chars = None
    out = dict(units=len(lp), word_units=n_wu,
               ppl_per_unit=2 ** (-tot / len(lp)), ppl_per_word=2 ** (-tot / n_wu))
    if n_chars:
        out["bpc"] = -tot / n_chars
        wlp = np.bincount(data.group[pos], weights=lp, minlength=data.n_word_units)
        out["ppl_per_word_invocab"] = 2 ** (-wlp[data.invocab].mean())
        out["ppl_per_word_oov_only"] = 2 ** (-wlp[~data.invocab].mean())
    return out


# ------------------------------------------------------------------ training

def train_model(name, model, data, V, args, device, budget_s, log_rows):
    rng = np.random.default_rng(args.seed)
    tr, dv = data["train"], data["dev"]
    n_tr = len(tr.slen)
    tr_idx = np.arange(n_tr)
    if args.max_train_sentences and n_tr > args.max_train_sentences:
        tr_idx = np.sort(rng.choice(n_tr, args.max_train_sentences, replace=False))
    max_len = args.max_len if name != "char_lstm" else args.char_max_len
    ch = chunks(tr, tr_idx, max_len)
    dev_idx = np.sort(np.random.default_rng(args.seed).choice(
        len(dv.slen), min(args.dev_eval_sentences, len(dv.slen)), replace=False))
    lr = args.lr_transformer if name == "transformer" else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"[{name}] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(tr_idx)} train sentences, {int(ch[:, 2].sum())} train targets, budget {budget_s / 60:.1f} min")
    t0 = time.time()
    step, seen, best, best_state, bad = 0, 0, float("inf"), None, 0
    run_loss, run_n = 0.0, 0
    stop = False
    for epoch in range(1, args.max_epochs + 1):
        for b in bucket_batches(ch, args.batch_tokens, rng):
            model.train()
            x, y = make_batch(tr, ch[b], V, device)
            warm = min(1.0, (step + 1) / args.warmup)
            for gparam in opt.param_groups:
                gparam["lr"] = lr * warm
            loss = -target_logprob(model, x, y).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int((y >= 0).sum())
            step += 1
            seen += n
            run_loss += loss.item() * n
            run_n += n
            out_of_time = time.time() - t0 > budget_s
            if step % args.eval_every == 0 or out_of_time:
                pos, lp = score(model, dv, dev_idx, V, device)
                m = metrics(dv, pos, lp, dev_idx)
                dev_ppl = m["ppl_per_word"]
                improved = dev_ppl < best * (1 - 1e-3)
                if dev_ppl < best:
                    best, best_state = dev_ppl, copy.deepcopy(model.state_dict())
                if improved:
                    bad = 0
                else:
                    bad += 1
                    lr /= 2
                row = dict(model=name, step=step, epoch=epoch, train_targets_seen=seen,
                           minutes=round((time.time() - t0) / 60, 2),
                           train_loss_nats_per_unit=round(run_loss / run_n, 4),
                           dev_ppl_per_unit=round(m["ppl_per_unit"], 2),
                           dev_ppl_per_word=round(dev_ppl, 2), lr=lr)
                log_rows.append(row)
                print(f"[{name}] step {step} ep {epoch} {row['minutes']}min train {row['train_loss_nats_per_unit']:.3f} "
                      f"dev/unit {row['dev_ppl_per_unit']} dev/word {row['dev_ppl_per_word']}"
                      f"{'' if improved else ' (no improvement, lr halved)'}")
                run_loss, run_n = 0.0, 0
                if bad >= args.patience:
                    print(f"[{name}] early stop: {bad} evaluations without improvement")
                    stop = True
                elif out_of_time:
                    print(f"[{name}] time budget reached")
                    stop = True
            if stop:
                break
        if stop:
            break
    model.load_state_dict(best_state)
    return dict(steps=step, epochs=round(seen / int(ch[:, 2].sum()), 2),
                train_targets_seen=seen, train_minutes=round((time.time() - t0) / 60, 2),
                best_dev_sample_ppl_per_word=round(best, 2),
                stop_reason="early stop" if bad >= args.patience else ("time budget" if stop else "max epochs"))


@torch.no_grad()
def sample(model, V, n, max_units, temperature, top_k, gen, device):
    model.eval()
    max_len = getattr(model, "max_len", None)
    out = []
    for _ in range(n):
        seq = [BOS]
        for _ in range(max_units):
            ctx = seq[-max_len:] if max_len else seq
            h = forward(model, torch.tensor([ctx], device=device))[0, -1:]
            logits = model.out.full_logprob(h)[0] / temperature
            logits[UNK] = -float("inf")
            v, i = torch.topk(logits, top_k)
            nxt = int(i[torch.multinomial(F.softmax(v, -1), 1, generator=gen)])
            if nxt == EOS:
                break
            seq.append(nxt)
        out.append(seq[1:])
    return out


# ------------------------------------------------------------------ output

def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baseline_rows(base_dir, test):
    """mKN rows from Phase 2 CSVs (run on the same splits), with bpc added."""
    rows = []
    base_dir = Path(base_dir)
    if (base_dir / "subword_perplexity.csv").exists():
        for r in read_csv(base_dir / "subword_perplexity.csv"):
            if r["smoothing"] != "mkn" or int(r["n"]) < 3:
                continue
            ppl = float(r["test_ppl_per_word"])
            rows.append(dict(model=f"mKN {r['n']}-gram", unit=r["unit"],
                             test_ppl_per_word=ppl, test_ppl_per_unit=float(r["test_ppl_per_piece"]),
                             test_bpc=round(math.log2(ppl) * test.n_word_units / test.n_chars, 4),
                             test_ppl_per_word_invocab=float(r["test_ppl_per_word_invocab"]),
                             test_ppl_per_word_oov_only=float(r["test_ppl_per_word_oov_only"])))
    if (base_dir / "ngram_perplexity.csv").exists():
        for r in read_csv(base_dir / "ngram_perplexity.csv"):
            if r["smoothing"] == "mkn" and int(r["n"]) in (3, 4):
                rows.append(dict(model=f"mKN {r['n']}-gram", unit="word (count>=2, OOV->UNK)",
                                 test_ppl_per_word=float(r["test_ppl"]), test_ppl_per_unit=float(r["test_ppl"]),
                                 test_bpc="", test_ppl_per_word_invocab=float(r["test_ppl_invocab"]),
                                 test_ppl_per_word_oov_only=""))
    return rows


def plot_curves(log_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in dict.fromkeys(r["model"] for r in log_rows):
        rs = [r for r in log_rows if r["model"] == name]
        x = [r["train_targets_seen"] / 1e6 for r in rs]
        axes[0].plot(x, [r["train_loss_nats_per_unit"] for r in rs], marker=".", label=name)
        axes[1].plot(x, [r["dev_ppl_per_word"] for r in rs], marker=".", label=name)
    axes[0].set_ylabel("train loss (nats per unit, mean since last eval)")
    axes[1].set_ylabel("dev per-word perplexity (dev sample, log scale)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("training targets seen (millions of units)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Training loss (units differ: pieces vs characters)")
    axes[1].set_title("Dev perplexity per word (comparable across units)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r.get("test_bpc") not in ("", None)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"{r['model']}\n{r['unit']}" for r in rows]
    colors = ["#888888" if r["model"].startswith("mKN") else "#3060c0" for r in rows]
    axes[0].bar(labels, [r["test_ppl_per_word"] for r in rows], color=colors)
    axes[0].set_ylabel("test perplexity per word (all tokens)")
    axes[1].bar(labels, [r["test_ppl_per_word_invocab"] for r in rows], color=colors)
    axes[1].set_ylabel("test perplexity per word (in-vocabulary tokens)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Neural LMs (blue) vs modified-KN n-gram LMs (grey), test split")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=str(ROOT / "data/splits"))
    ap.add_argument("--out", default=str(ROOT / "results/phase3"))
    ap.add_argument("--models-dir", default=str(ROOT / "data/models"))
    ap.add_argument("--spm", default=str(ROOT / "data/models/spm_unigram_16000.model"))
    ap.add_argument("--models", nargs="+", default=["ffnn", "lstm", "transformer", "char_lstm"],
                    choices=["ffnn", "lstm", "transformer", "char_lstm"])
    ap.add_argument("--budget", nargs="*", default=["ffnn=2.5", "lstm=7", "transformer=7", "char_lstm=3.5"],
                    help="training time budget per model, name=minutes")
    ap.add_argument("--max-train-sentences", type=int, default=0, help="0 = all train sentences")
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=128, help="chunk length (pieces) for training; Transformer context")
    ap.add_argument("--char-max-len", type=int, default=256)
    ap.add_argument("--ff-context", type=int, default=3, help="FFNN history length (3 = like a 4-gram)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-transformer", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--dev-eval-sentences", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--final-eval-sentences", type=int, default=0, help="0 = full split")
    ap.add_argument("--final-splits", nargs="+", default=["test"], choices=["dev", "test"],
                    help="splits scored in full with the best checkpoint (dev adds ~1 min per model)")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-bf16", action="store_true",
                    help="disable bfloat16 autocast of the network body (on by default; 2-4x faster on CPUs with AVX512-BF16/AMX)")
    ap.add_argument("--baseline-dir", default=str(ROOT / "results/phase3/ngram_baseline"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prepare-only", action="store_true", help="only train the SPM model if missing")
    args = ap.parse_args()
    budget = {k: float(v) * 60 for k, v in (b.split("=") for b in args.budget)}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    AMP["enabled"] = not args.no_bf16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    print(f"torch {torch.__version__}, device {device}, {torch.get_num_threads()} threads")

    corpus = Corpus(args.splits)
    ensure_spm(corpus, args.spm)
    if args.prepare_only:
        return
    print(f"loaded splits in {time.time() - t_all:.0f}s")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(args.spm))
    assert (sp.unk_id(), sp.bos_id(), sp.eos_id()) == (UNK, BOS, EOS)
    invocab_type = corpus.counts("train") >= 2
    sw_flat, sw_off, sw_len = spm_table(corpus, sp)
    ch_flat, ch_off, ch_len, ch_vocab = char_table(corpus)
    data = {"spm": {}, "char": {}}
    for split in ("train", "dev", "test"):
        c = build_split(corpus, split, ch_flat, ch_off, ch_len, 1, invocab_type)
        s = build_split(corpus, split, sw_flat, sw_off, sw_len, 0, invocab_type)
        s.n_chars = c.n_chars = len(c.seq)
        data["char"][split], data["spm"][split] = c, s
        print(f"  {split}: {len(s.slen)} sentences, {s.n_words} words, {len(s.seq)} pieces (+</s>), "
              f"{len(c.seq)} chars (+</s>)")
    unk_chars = {sp_: int((data['char'][sp_].seq == UNK).sum()) for sp_ in ("dev", "test")}
    print(f"char vocab {len(ch_vocab)}; unknown chars dev/test: {unk_chars}")
    del corpus

    unit_name = {"spm": f"spm-unigram-{sp.vocab_size() // 1000}k", "char": f"char ({len(ch_vocab)})"}
    results, log_rows, samples_md = [], [], []
    gen = torch.Generator().manual_seed(args.seed)
    for name in args.models:
        kind = "char" if name == "char_lstm" else "spm"
        V = len(ch_vocab) if kind == "char" else sp.vocab_size()
        torch.manual_seed(args.seed)
        counts = np.bincount(data[kind]["train"].seq, minlength=V)
        model = make_model(name, V, counts, args).to(device)
        info = train_model(name, model, data[kind], V, args, device, budget.get(name, 5 * 60), log_rows)
        torch.save(model.state_dict(), models_dir / f"phase3_{name}.pt")
        row = dict(model=name, unit=unit_name[kind], params=sum(p.numel() for p in model.parameters()), **info)
        t = time.time()
        for split in args.final_splits:
            d = data[kind][split]
            idx = np.arange(len(d.slen))
            if args.final_eval_sentences:
                idx = idx[: args.final_eval_sentences]
            pos, lp = score(model, d, idx, V, device)
            m = metrics(d, pos, lp, None if not args.final_eval_sentences else idx)
            for k, v in m.items():
                row[f"{split}_{k}"] = round(float(v), 4) if isinstance(v, float) else v
        row["eval_minutes"] = round((time.time() - t) / 60, 2)
        results.append(row)
        print(f"[{name}] test per-word ppl {row['test_ppl_per_word']:.1f}, per-unit {row['test_ppl_per_unit']:.2f}, "
              f"bpc {row.get('test_bpc', float('nan')):.3f}, in-vocab {row.get('test_ppl_per_word_invocab', float('nan')):.1f} "
              f"(eval {row['eval_minutes']} min)")
        samples_md.append(f"## {name} ({unit_name[kind]})\n")
        for s in sample(model, V, args.samples, 80 if kind == "spm" else 300, 0.8, 40, gen, device):
            text = sp.decode(s) if kind == "spm" else "".join(ch_vocab[i] for i in s).replace(NUM_CHAR, NUM)
            samples_md.append(f"* {text.strip()}")
        samples_md.append("")
        write_csv(out / "neural_perplexity.csv", results)
        write_csv(out / "training_log.csv", log_rows)

    (out / "samples.md").write_text(
        "# Samples from the Phase 3 neural LMs\n\nTemperature 0.8, top-k 40, `<unk>` disabled, "
        f"seed {args.seed}. Generated from `<s>` until `</s>`.\n\n" + "\n".join(samples_md) + "\n", encoding="utf-8")
    plot_curves(log_rows, out / "plots/loss_curves.png")

    test = data["spm"]["test"]
    comp = baseline_rows(args.baseline_dir, test)
    for r in results:
        comp.append(dict(model=r["model"], unit=r["unit"], test_ppl_per_word=r["test_ppl_per_word"],
                         test_ppl_per_unit=r["test_ppl_per_unit"], test_bpc=r.get("test_bpc", ""),
                         test_ppl_per_word_invocab=r.get("test_ppl_per_word_invocab", ""),
                         test_ppl_per_word_oov_only=r.get("test_ppl_per_word_oov_only", "")))
    for r in comp:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 4 if k == "test_bpc" else 2)
    write_csv(out / "comparison.csv", comp)
    plot_comparison(comp, out / "plots/comparison.png")
    print(f"test sizes: {test.n_words} words, {len(test.slen)} sentences, {test.n_chars} chars")
    print(f"done in {(time.time() - t_all) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
