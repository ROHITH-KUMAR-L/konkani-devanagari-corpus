"""Experimental word-and-suffix BiLSTM-CRF for the Phase 3 Konkani NER data.

The supplied silver sets are AI-annotated and are NOT human-verified gold.
Scores produced from them are explicitly marked with their input annotation
status and must not be reported as gold evaluation.
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_common import LABELS, Sentence, read_conll, write_conll  # noqa: E402

PAD, UNK = "<PAD>", "<UNK>"
TAG_NAMES = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I")]


def build_vocabs(sentences, min_freq=1):
    word_counts = Counter(token for sent in sentences for token in sent.tokens)
    words = [PAD, UNK] + sorted(token for token, count in word_counts.items() if count >= min_freq)
    suffixes = [PAD, UNK] + sorted({
        token[-3:] for sent in sentences for token in sent.tokens if token
    })
    return ({word: i for i, word in enumerate(words)},
            {suffix: i for i, suffix in enumerate(suffixes)})


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, suffix_size, tag_count, embedding_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for a bidirectional LSTM")
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(suffix_size, 32, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embedding_dim + 32, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.emission = nn.Linear(hidden_dim, tag_count)
        self.transitions = nn.Parameter(torch.zeros(tag_count, tag_count))
        self.start_transitions = nn.Parameter(torch.zeros(tag_count))
        self.end_transitions = nn.Parameter(torch.zeros(tag_count))
        allowed = torch.ones(tag_count, tag_count, dtype=torch.bool)
        start_allowed = torch.ones(tag_count, dtype=torch.bool)
        for to_id, to_tag in enumerate(TAG_NAMES):
            if to_tag.startswith("I-"):
                entity = to_tag[2:]
                start_allowed[to_id] = False
                for from_id, from_tag in enumerate(TAG_NAMES):
                    if from_tag not in (f"B-{entity}", f"I-{entity}"):
                        allowed[from_id, to_id] = False
        self.register_buffer("allowed_transitions", allowed)
        self.register_buffer("allowed_starts", start_allowed)

    def forward(self, words, suffixes, lengths):
        features = torch.cat(
            (self.word_embedding(words), self.suffix_embedding(suffixes)), dim=-1
        )
        packed = pack_padded_sequence(
            self.dropout(features), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=words.size(1))
        return self.emission(self.dropout(encoded))

    def _scores(self):
        transitions = self.transitions.masked_fill(~self.allowed_transitions, -10000.0)
        starts = self.start_transitions.masked_fill(~self.allowed_starts, -10000.0)
        return transitions, starts

    def neg_log_likelihood(self, emissions, tags, mask):
        transitions, starts = self._scores()
        batch, steps, _ = emissions.shape
        alpha = starts.unsqueeze(0) + emissions[:, 0]
        for step in range(1, steps):
            candidate = alpha.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, step].unsqueeze(1)
            next_alpha = torch.logsumexp(candidate, dim=1)
            alpha = torch.where(mask[:, step].unsqueeze(1), next_alpha, alpha)
        partition = torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

        lengths = mask.long().sum(1)
        gold = starts[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for step in range(1, steps):
            current = tags[:, step].clamp_min(0)
            previous = tags[:, step - 1].clamp_min(0)
            score = transitions[previous, current] + emissions[:, step].gather(
                1, current.unsqueeze(1)
            ).squeeze(1)
            gold = gold + score * mask[:, step]
        last = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        gold = gold + self.end_transitions[last]
        return (partition - gold).mean()

    @torch.no_grad()
    def decode(self, emissions, lengths):
        transitions, starts = self._scores()
        outputs = []
        for row, length in zip(emissions, lengths.tolist()):
            score = starts + row[0]
            history = []
            for step in range(1, length):
                candidates = score.unsqueeze(1) + transitions
                score, back = candidates.max(dim=0)
                score = score + row[step]
                history.append(back)
            last = int((score + self.end_transitions).argmax())
            path = [last]
            for back in reversed(history):
                last = int(back[last])
                path.append(last)
            outputs.append(list(reversed(path)))
        return outputs


def encode_batch(sentences, word_vocab, suffix_vocab, tag_vocab, device):
    lengths = torch.tensor([len(sent.tokens) for sent in sentences], dtype=torch.long)
    width = int(lengths.max())
    words = torch.zeros((len(sentences), width), dtype=torch.long)
    suffixes = torch.zeros_like(words)
    tags = torch.full((len(sentences), width), -100, dtype=torch.long)
    for row, sent in enumerate(sentences):
        for col, token in enumerate(sent.tokens):
            words[row, col] = word_vocab.get(token, word_vocab[UNK])
            suffixes[row, col] = suffix_vocab.get(token[-3:], suffix_vocab[UNK])
            tags[row, col] = tag_vocab[sent.tags[col]]
    mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
    return words.to(device), suffixes.to(device), tags.to(device), mask.to(device), lengths


def predict(model, sentences, word_vocab, suffix_vocab, tag_vocab, device, batch_size=32):
    model.eval()
    id_to_tag = {index: tag for tag, index in tag_vocab.items()}
    predictions = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            words, suffixes, _, _, lengths = encode_batch(
                batch, word_vocab, suffix_vocab, tag_vocab, device
            )
            emissions = model(words, suffixes, lengths)
            predictions.extend([[id_to_tag[tag] for tag in path] for path in model.decode(emissions, lengths)])
    return predictions


def score(gold, predicted):
    report = classification_report(gold, predicted, output_dict=True, zero_division=0)
    rows = []
    for label in LABELS + ["micro avg", "macro avg"]:
        values = report.get(label, {})
        rows.append({
            "entity": label,
            "precision": round(values.get("precision", 0), 4),
            "recall": round(values.get("recall", 0), 4),
            "f1": round(values.get("f1-score", 0), 4),
            "support": int(values.get("support", 0)),
        })
    return rows


def write_confusion(path, gold, predicted):
    labels = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I")]
    gold_flat = [tag for sent in gold for tag in sent]
    pred_flat = [tag for sent in predicted for tag in sent]
    matrix = confusion_matrix(gold_flat, pred_flat, labels=labels)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())


def save_model(path, model, word_vocab, suffix_vocab, tag_vocab, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save({
            "format_version": 1,
            "state_dict": model.state_dict(),
            "word_vocab": word_vocab,
            "suffix_vocab": suffix_vocab,
            "tag_vocab": tag_vocab,
            "config": config,
        }, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported BiLSTM-CRF checkpoint format")
    word_vocab = checkpoint["word_vocab"]
    suffix_vocab = checkpoint["suffix_vocab"]
    tag_vocab = checkpoint["tag_vocab"]
    config = checkpoint["config"]
    if tag_vocab != {tag: index for index, tag in enumerate(TAG_NAMES)}:
        raise ValueError("checkpoint tag vocabulary is incompatible with the current BIO label set")
    model = BiLSTMCRF(
        len(word_vocab), len(suffix_vocab), len(tag_vocab),
        embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, word_vocab, suffix_vocab, tag_vocab


def evaluate_and_write(split_name, sentences, predictions, outdir):
    gold = [sent.tags for sent in sentences]
    rows = score(gold, predictions)
    annotation = next(
        (sent.meta.get("annotation") for sent in sentences if sent.meta.get("annotation")),
        "unspecified",
    )
    score_path = Path(outdir) / "ner_bilstm_crf_scores.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "annotation_status", "entity", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        writer.writerows({"split": split_name, "annotation_status": annotation, **row} for row in rows)
    write_confusion(Path(outdir) / "ner_bilstm_crf_token_confusion.csv", gold, predictions)
    output = [
        Sentence(dict(sent.meta, model="experimental-bilstm-crf"), sent.tokens, tags)
        for sent, tags in zip(sentences, predictions)
    ]
    write_conll(Path(outdir) / "ner_bilstm_crf_predictions.conll", output, [
        "Experimental BiLSTM-CRF predictions; token-level confusion matrix is diagnostic only.",
        f"Evaluation annotation status: {annotation}. Silver annotations are NOT human-verified gold.",
    ])
    micro = next(row for row in rows if row["entity"] == "micro avg")
    print(f"{split_name} ({annotation}; NOT necessarily gold): "
          f"P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = read_conll(args.train)
    if any(sent.meta.get("annotation", "").startswith("silver") for sent in source):
        print("WARNING: training annotations are AI-generated silver labels, not human-verified gold.")
    if len(source) < 3:
        raise ValueError("at least three training sentences are needed for an internal validation split")
    indices = list(range(len(source)))
    random.Random(args.seed).shuffle(indices)
    validation_n = min(len(indices) - 1, max(1, round(len(indices) * args.validation_fraction)))
    validation = [source[i] for i in indices[:validation_n]]
    training = [source[i] for i in indices[validation_n:]]
    if not training:
        raise ValueError("validation_fraction leaves no training sentences")
    test = read_conll(args.test)
    if not test:
        raise ValueError("test file contains no sentences")

    word_vocab, suffix_vocab = build_vocabs(training, args.min_freq)
    tag_vocab = {tag: i for i, tag in enumerate(TAG_NAMES)}
    config = {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim}
    model = BiLSTMCRF(
        len(word_vocab), len(suffix_vocab), len(tag_vocab),
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
    ).to(device)
    if args.fasttext:
        from gensim.models import KeyedVectors
        vectors = KeyedVectors.load(str(args.fasttext))
        if vectors.vector_size != args.embedding_dim:
            raise ValueError(
                f"FastText has {vectors.vector_size} dimensions; --embedding-dim is {args.embedding_dim}"
            )
        with torch.no_grad():
            for token, index in word_vocab.items():
                if token in vectors.key_to_index:
                    model.word_embedding.weight[index].copy_(
                        torch.as_tensor(vectors.get_vector(token), device=device)
                    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = training[:]
        random.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start:start + args.batch_size]
            words, suffixes, tags, mask, lengths = encode_batch(
                batch, word_vocab, suffix_vocab, tag_vocab, device
            )
            loss = model.neg_log_likelihood(model(words, suffixes, lengths), tags, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        dev_pred = predict(model, validation, word_vocab, suffix_vocab, tag_vocab, device)
        dev_rows = score([sent.tags for sent in validation], dev_pred)
        dev_f1 = next(row["f1"] for row in dev_rows if row["entity"] == "micro avg")
        print(f"epoch {epoch}: internal validation entity F1={dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)
    save_model(args.checkpoint, model, word_vocab, suffix_vocab, tag_vocab, config)
    test_pred = predict(model, test, word_vocab, suffix_vocab, tag_vocab, device)
    evaluate_and_write("test", test, test_pred, args.outdir)


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word_vocab, suffix_vocab, tag_vocab = load_model(args.checkpoint, device)
    sentences = read_conll(args.input)
    if not sentences:
        raise ValueError("input file contains no sentences")
    predictions = predict(model, sentences, word_vocab, suffix_vocab, tag_vocab, device)
    output = [
        Sentence(dict(sent.meta, model="experimental-bilstm-crf"), sent.tokens, tags)
        for sent, tags in zip(sentences, predictions)
    ]
    write_conll(args.out, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train on silver_dev; keep silver_test untouched")
    fit.add_argument("--train", default="data/eval/ner/silver_dev.conll")
    fit.add_argument("--test", default="data/eval/ner/silver_test.conll")
    fit.add_argument("--checkpoint", default="data/models/phase3_ner_bilstm_crf.pt")
    fit.add_argument("--outdir", default="results/phase3")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=5)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--min-freq", type=int, default=1)
    fit.add_argument("--validation-fraction", type=float, default=0.15)
    fit.add_argument("--embedding-dim", type=int, default=128)
    fit.add_argument("--hidden-dim", type=int, default=128)
    fit.add_argument("--learning-rate", type=float, default=0.001)
    fit.add_argument("--fasttext", help="optional local FastText KeyedVectors file")
    fit.add_argument("--seed", type=int, default=2026)
    fit.set_defaults(run=train)

    prediction = commands.add_parser("predict", help="run a saved model on CoNLL tokens")
    prediction.add_argument("--checkpoint", default="data/models/phase3_ner_bilstm_crf.pt")
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--out", required=True)
    prediction.set_defaults(run=infer)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.min_freq < 1:
            parser.error("--epochs, --patience, --batch-size, and --min-freq must be positive")
        if not 0 < args.validation_fraction < 1:
            parser.error("--validation-fraction must be between 0 and 1")
        if args.embedding_dim < 1 or args.hidden_dim < 2 or args.hidden_dim % 2:
            parser.error("--embedding-dim must be positive and --hidden-dim must be positive and even")
        if args.learning_rate <= 0:
            parser.error("--learning-rate must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
