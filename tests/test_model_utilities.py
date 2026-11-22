import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "phase2"))
sys.path.insert(0, str(ROOT / "scripts" / "phase3"))
sys.path.insert(0, str(ROOT / "scripts" / "demo"))

import neural_lm  # noqa: E402
from ner_bilstm_crf import (  # noqa: E402
    BiLSTMCRF, build_vocabs, load_model, predict, save_model,
)
from ner_common import Sentence  # noqa: E402


class NeuralLmCheckpointTests(unittest.TestCase):
    def test_versioned_checkpoint_round_trip_and_model_guard(self):
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lm.pt"
            neural_lm.save_checkpoint(source, path, "test-model", 17)
            neural_lm.load_checkpoint(target, path, model_name="test-model", vocab_size=17)
            self.assertTrue(torch.equal(source.weight, target.weight))
            with self.assertRaisesRegex(ValueError, "not wrong-model"):
                neural_lm.load_checkpoint(target, path, model_name="wrong-model", vocab_size=17)


class BiLstmCrfTests(unittest.TestCase):
    def test_trainable_loss_decode_and_checkpoint_reload(self):
        sentences = [
            Sentence({}, ["प्रमोद", "सावंत"], ["B-PER", "I-PER"]),
            Sentence({}, ["गोंयांत"], ["B-LOC"]),
            Sentence({}, ["आज"], ["O"]),
        ]
        words, suffixes = build_vocabs(sentences)
        tags = {tag: index for index, tag in enumerate(
            ["O"] + [f"{p}-{label}" for label in ("PER", "LOC", "ORG", "MISC", "DATE")
                     for p in ("B", "I")]
        )}
        model = BiLSTMCRF(len(words), len(suffixes), len(tags), embedding_dim=8, hidden_dim=8)
        from ner_bilstm_crf import encode_batch

        batch = sentences[:2]
        word_ids, suffix_ids, tag_ids, mask, lengths = encode_batch(
            batch, words, suffixes, tags, torch.device("cpu")
        )
        loss = model.neg_log_likelihood(model(word_ids, suffix_ids, lengths), tag_ids, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.eval()
        before = predict(model, sentences, words, suffixes, tags, torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ner.pt"
            config = {"embedding_dim": 8, "hidden_dim": 8}
            save_model(path, model, words, suffixes, tags, config)
            restored, restored_words, restored_suffixes, restored_tags = load_model(
                path, torch.device("cpu")
            )
            after = predict(
                restored, sentences, restored_words, restored_suffixes, restored_tags,
                torch.device("cpu"),
            )
        self.assertEqual(before, after)
        self.assertTrue(all(not tag.startswith("I-") for tag in after[0][:1]))


if __name__ == "__main__":
    unittest.main()
