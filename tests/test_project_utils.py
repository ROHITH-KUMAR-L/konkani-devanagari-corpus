import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase3"))

from corpus_utils import NUM, aksharas, devanagari_ratio, tokenize  # noqa: E402
from ner_common import Sentence, gazetteer_tag, read_conll, spans, write_conll  # noqa: E402
from segment_sentences import split_sentences  # noqa: E402


class CorpusUtilityTests(unittest.TestCase):
    def test_tokenize_maps_devanagari_numbers(self):
        self.assertEqual(tokenize("घरांत १२ आनी 3.5"), ["घरांत", NUM, "आनी", NUM])

    def test_devanagari_ratio_and_aksharas(self):
        self.assertEqual(devanagari_ratio("कोंकणी")[0], 1.0)
        self.assertGreater(len(aksharas("कोंकणी")), 0)

    def test_sentence_segmentation_preserves_two_sentences(self):
        result = split_sentences("पयलें वाक्य. दुसरें वाक्य!")
        self.assertEqual(len(result), 2)


class NerUtilityTests(unittest.TestCase):
    def test_bio_spans(self):
        self.assertEqual(
            spans(["B-PER", "I-PER", "O", "B-LOC"]),
            [("PER", 0, 2), ("LOC", 3, 4)],
        )

    def test_gazetteer_matches_attached_case_suffix(self):
        self.assertEqual(gazetteer_tag(["गोंयांत"], {("गोंय",): "LOC"}), ["B-LOC"])

    def test_conll_round_trip_and_bad_tag(self):
        sentence = Sentence({"sent_id": "ner-test"}, ["गोंयांत"], ["B-LOC"])
        with self.subTest("round trip"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "sample.conll"
                write_conll(path, [sentence])
                loaded = read_conll(path)
                self.assertEqual(loaded[0].tokens, sentence.tokens)
                self.assertEqual(loaded[0].tags, sentence.tags)
        with self.subTest("invalid tag"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.conll"
                path.write_text("गोंयांत\tX-LOC\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid NER tag"):
                    read_conll(path)

    def test_existing_silver_split_is_readable_without_calling_it_gold(self):
        path = ROOT / "data" / "eval" / "ner" / "silver_test.conll"
        sentences = read_conll(path)
        self.assertTrue(sentences)
        self.assertTrue(all(
            sent.meta.get("annotation") == "silver-ai-first-pass" for sent in sentences
        ))


if __name__ == "__main__":
    unittest.main()
