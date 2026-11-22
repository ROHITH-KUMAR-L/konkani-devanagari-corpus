import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts" / "demo"))

import app  # noqa: E402


class DemoInputTests(unittest.TestCase):
    def test_language_check_and_segmentation_return_structured_results(self):
        language = app.api_langid("आनी आसा")
        self.assertEqual(len(language["paragraphs"]), 1)
        self.assertIn("label", language["paragraphs"][0])

        segmented = app.api_segment("पयलें वाक्य. दुसरें वाक्य!")
        self.assertEqual(len(segmented["sentences"]), 2)

    def test_blank_similarity_and_search_inputs_are_rejected_without_loading_models(self):
        with self.assertRaisesRegex(ValueError, "all three analogy terms"):
            app.api_similar(" ")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            app.api_search(" ")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            app.api_langid(None)

    def test_ner_preprocessing_keeps_date_digits_and_drops_punctuation(self):
        self.assertEqual(app.ner_tokens("गोंयांत 15 ऑगस्ट."), ["गोंयांत", "15", "ऑगस्ट"])

    def test_generator_reports_missing_checkpoint_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(app, "MODELS", Path(tmp)):
                with self.assertRaisesRegex(FileNotFoundError, "no LSTM checkpoint found"):
                    app.Generator()

    def test_corpus_search_reports_a_missing_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(app, "ROOT", Path(tmp)), patch.dict(app._cache, {}, clear=True):
                with self.assertRaises(FileNotFoundError):
                    app.api_search("गोंय")


if __name__ == "__main__":
    unittest.main()
