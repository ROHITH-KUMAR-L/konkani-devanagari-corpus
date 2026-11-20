"""Small, local research-pipeline checks. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import corpus_common as core
from scripts.analyze_corpus import coverage_rows, frequency_rows, length_stats, word_lengths
from scripts.clean_text import clean_document
from scripts.deduplicate import deduplicate_documents, find_near_duplicates
from scripts.extract_text import NeedsOCR, html_text, pdf_text, xml_text
from scripts.segment_sentences import segment_sentences, segmentation_flags
from scripts.validate_konkani import inspect_text

REPOSITORY = Path(__file__).resolve().parents[1]
SAMPLE = "हांव गोंयांत रावतां. म्हाका कोंकणी आवडटा! आनी तूं खंय वता?"


class UnicodeTests(unittest.TestCase):
    def test_unicode_decoding_and_preservation(self):
        text = "क़ क्\u200dष क्\u200cष गोंयांत १२३\r\nआनी"
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            decoded, detected = core.decode_bytes(text.encode(encoding))
            self.assertEqual(decoded, core.normalize(text), (encoding, detected))
            self.assertTrue(unicodedata.is_normalized("NFC", decoded))
        with self.assertRaises(UnicodeError):
            core.decode_bytes(b"\xffbad")
        _, cleaned, _ = clean_document(text)
        self.assertIn("\u200d", cleaned)
        self.assertIn("\u200c", cleaned)
        self.assertIn("\u093c", cleaned)
        self.assertIn("१२३", cleaned)

    def test_token_policy_and_script_denominator(self):
        self.assertEqual(core.tokens("गोंयांत, कोंकणी! १२.५ 3.5"), ["गोंयांत", "कोंकणी", "१२.५", "3.5"])
        self.assertEqual(core.tokens("आसा १२.५", "normalize"), ["आसा", "<NUM>"])
        self.assertEqual(core.tokens("आसा १२.५", "exclude"), ["आसा"])
        self.assertEqual(core.script_stats("आसा 123!?।॥")["devanagari_ratio"], 1)
        self.assertEqual(core.script_stats("१२३")["devanagari_ratio"], 0)
        self.assertEqual(inspect_text("यह है और नहीं है")["konkani_markers"], 0)
        self.assertIn("possible_hindi_or_marathi", inspect_text("यह है और नहीं है")["flags"])

    def test_cleaning_is_conservative_and_audited(self):
        minimal, text, changes = clean_document("<pre>कितें ?\n१२\nOpenAI नाव\nक्\u200dष</pre>")
        self.assertNotIn("<pre>", minimal)
        self.assertIn("कितें?", text)
        self.assertIn("१२", text)
        self.assertIn("OpenAI", text)
        self.assertEqual(changes, [])
        _, text, changes = clean_document("१२\nआनी मोग\nEnglish boilerplate", drop_page_numbers=True, min_line_deva=.7)
        self.assertEqual(text, "आनी मोग")
        self.assertEqual(len(changes), 2)
        self.assertEqual(clean_document(text)[1], text)

    def test_repeated_headers_only_at_page_edges(self):
        text = "शीर्षक\nआनी\n\fशीर्षक\nमोग\n\fशीर्षक\nगोंय\n"
        _, cleaned, changes = clean_document(text, repeated_headers=True)
        self.assertNotIn("शीर्षक", cleaned)
        self.assertEqual(len(changes), 3)


class SentenceTests(unittest.TestCase):
    def test_boundaries_and_quotes(self):
        self.assertEqual(segment_sentences('तो म्हणटा, “हांव येतां!” आनी तूं? बरें। दुसरें॥'),
                         ['तो म्हणटा, “हांव येतां!”', 'आनी तूं?', 'बरें।', 'दुसरें॥'])

    def test_abbreviations_decimals_and_initialisms(self):
        text = "डॉ. के.के. कामत 3.5 वाजता आयले. आर्.पी.आय. आनी पी.ओ. हांगा आसात."
        self.assertEqual(len(segment_sentences(text)), 2)
        self.assertIn("3.5", segment_sentences(text)[0])
        self.assertEqual(segment_sentences("Dr. A. B. Kamat येता. आनी हांव."),
                         ["Dr. A. B. Kamat येता.", "आनी हांव."])

    def test_urls_ellipsis_and_verse_numbers(self):
        self.assertEqual(len(segment_sentences("हांव... आतां येतां. पळय https://example.org/a?q=1. आनी वच.")), 3)
        self.assertEqual(segment_sentences("पहिलो श्लोक॥ १ ॥ दुसरो श्लोक॥"),
                         ["पहिलो श्लोक॥ १ ॥", "दुसरो श्लोक॥"])

    def test_paragraphs_and_token_conservation(self):
        text = "हांव\nगोंयांत रावतां।\n\nआनी तूं येता? उरिल्लो मजकूर"
        sentences = segment_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(Counter(core.tokens(text)), Counter(core.tokens(" ".join(sentences))))

    def test_missing_spaces_are_not_mistaken_for_initialisms(self):
        self.assertEqual(segment_sentences("आसा.हांव येतां.तूं येता?"),
                         ["आसा.", "हांव येतां.", "तूं येता?"])

    def test_missing_transcript_punctuation_is_flagged_not_invented(self):
        transcript = " ".join(["आनी"] * 101)
        self.assertEqual(segment_sentences(transcript), [transcript])
        self.assertEqual(segmentation_flags(transcript), ["over_100_tokens", "no_terminal_punctuation"])
        self.assertEqual(segmentation_flags('“हांव येतां!”'), [])


class DedupAndAnalysisTests(unittest.TestCase):
    def test_accepted_copy_wins_and_rejected_cannot_suppress(self):
        def doc(identity, review, included, text=SAMPLE):
            return {"document_id": identity, "effective_review_status": review, "included": included,
                    "source_file": identity + ".txt", "sha256": "same", "text": text}
        documents = [doc("A", "rejected", False), doc("B", "pending", True), doc("C", "accepted", True)]
        result, relationships, excluded = deduplicate_documents(documents)
        self.assertEqual([d["document_id"] for d in result if d["included"]], ["C"])
        self.assertEqual(relationships[0]["duplicate_of"], "C")
        self.assertEqual(len(excluded), 1)

    def test_normalized_duplicate_and_paragraph_audit(self):
        docs = [{"document_id": str(i), "source_file": str(i), "effective_review_status": "accepted",
                 "included": True, "sha256": str(i), "text": text}
                for i, text in enumerate(["आनी मोग", "आनी  मोग", "आनी मोग\n\nगोंयांत रावतां"])]
        _, relationships, _ = deduplicate_documents(docs, paragraph_dedup=True)
        self.assertEqual(relationships[0]["kind"], "normalized_text")
        self.assertEqual(docs[2]["text"], "गोंयांत रावतां")
        self.assertEqual(relationships[1]["kept_paragraph_id"], 1)

    def test_near_duplicates_are_review_only(self):
        text = " ".join(f"शब्द{i}" for i in range(100))
        documents = [{"document_id": "A", "text": text}, {"document_id": "B", "text": text + " नवे शब्द"}]
        matches = find_near_duplicates(documents)
        self.assertEqual(matches[0]["action"], "manual_review")
        self.assertGreater(matches[0]["similarity"], .8)

    def test_known_statistics_and_graphemes(self):
        counts = Counter({"आनी": 5, "हांव": 3, "मोग": 2})
        self.assertEqual([r["types_required"] for r in coverage_rows(counts, "combined")], [1, 1, 1, 2, 3, 3, 3])
        self.assertEqual(frequency_rows(counts, "combined")[-1]["cumulative_coverage"], 1)
        stats = length_stats([1, 3, 5])
        self.assertEqual(stats["mean"], 3)
        self.assertAlmostEqual(stats["std"], (8 / 3) ** .5)
        lengths = word_lengths(Counter({"कि": 2}), "combined")
        self.assertEqual(next(r for r in lengths if r["unit"] == "graphemes")["length"], 1)
        self.assertEqual(next(r for r in lengths if r["unit"] == "codepoints")["length"], 2)
        self.assertEqual(coverage_rows(Counter(), "empty")[0]["types_required"], 0)


class ExtractionTests(unittest.TestCase):
    def test_markup_preserves_inline_boundaries(self):
        self.assertIn("म्हाका कोंकणी आवडटा", html_text("<nav>menu</nav><p>म्हाका <b>कोंकणी</b> आवडटा</p>"))
        self.assertNotIn("menu", html_text("<nav>menu</nav><p>आनी</p>"))
        text = xml_text("<root><Header>exclude</Header><body><p>म्हाका <b>कोंकणी</b> आवडटा</p><p>आनी मोग</p></body></root>".encode())
        self.assertIn("म्हाका कोंकणी आवडटा", text)
        self.assertNotIn("exclude", text)
        from defusedxml.common import DefusedXmlException
        with self.assertRaises(DefusedXmlException):
            xml_text(b'<!DOCTYPE x [<!ENTITY e "unsafe">]><x>&e;</x>')

    def test_pdf_no_partial_export_and_page_specific_ocr(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pdf"
            with fitz.open() as pdf:
                page = pdf.new_page(width=100, height=100)
                page.insert_text((10, 20), "Hello world")
                pdf.new_page(width=100, height=100)
                pdf.save(path)
            args = argparse.Namespace(pdf_min_letters=5, pdf_min_devanagari=0, ocr=False,
                                      tesseract=None, tessdata=None, dpi=72, psm=3, ocr_lang="mar+hin", ocr_timeout=30)
            with self.assertRaises(NeedsOCR):
                pdf_text(path, args)
            args.ocr = True
            with patch("pytesseract.image_to_string", return_value=SAMPLE) as ocr, patch("pytesseract.get_tesseract_version", return_value="test"):
                text, provenance = pdf_text(path, args)
                self.assertEqual(ocr.call_count, 1)
                self.assertEqual(provenance["page_methods"], ["pdf_text_layer", "tesseract"])
                self.assertIn(SAMPLE, text)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_script(self, name, *args, success=True):
        result = subprocess.run([sys.executable, str(REPOSITORY / "scripts" / f"{name}.py"),
                                 "--project-root", str(self.root), *args], capture_output=True, text=True, encoding="utf-8")
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def source(self, category, name, text=SAMPLE, encoding="utf-8"):
        path = self.root / "data/raw" / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode(encoding))
        return path

    def fixture(self):
        from docx import Document
        from ebooklib import epub
        self.source("01_literature", "KOK_LIT_0001_story.txt")
        self.source("01_literature", "copy.txt")
        self.source("02_news_and_media", "news.txt", "आनी खबर आसा। गोंयांत पावस येता!", "utf-16")
        path = self.source("03_education_science", "lesson.docx", "")
        docx = Document()
        docx.add_paragraph("विज्ञानाची म्हायती हांगा आसा।")
        docx.add_table(rows=1, cols=1).cell(0, 0).text = "शिक्षण आनी अभ्यास"
        docx.save(path)
        self.source("04_culture_history", "culture.html", "<nav>MENU</nav><p>गोंयची संस्कृती आनी इतिहास आसा।</p>")
        self.source("05_public_information", "notice.xml", "<root><body><p>सार्वजनिक सुट्टी आसा। आनी खबर पळयात.</p></body></root>")
        path = self.source("06_webarticles", "article.epub", "")
        book = epub.EpubBook()
        book.set_identifier("test-book")
        book.set_title("Fixture")
        book.set_language("kok")
        chapter = epub.EpubHtml(title="Chapter", file_name="chapter.xhtml", lang="kok")
        chapter.content = "<p>कोंकणी भास आनी लिपी हांचो अभ्यास करात.</p>"
        book.add_item(chapter)
        book.spine = [chapter]
        book.toc = [chapter]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        epub.write_epub(str(path), book)
        self.source("07_oral_transcripts", "other_language.txt", "यह हिन्दी है और यह कोंकणी नहीं है।")
        self.source("reference_only", "dictionary.txt", "आसा")
        self.source("08_ldcil_gold_sample", "reference.txt", "आनी")

    def test_end_to_end_formats_review_dedup_and_reproducibility(self):
        self.fixture()
        hashes = {path: core.sha256_file(path) for path in (self.root / "data/raw").rglob("*") if path.is_file()}
        self.run_script("extract_text")
        registry = core.load_sources(self.root)
        self.assertEqual(len(registry), 8)
        self.assertEqual({r["category"] for r in registry}, set(core.CATEGORIES))
        raw_txt_hashes = {p: core.sha256_file(p) for p in (self.root / "data/raw_txt").rglob("*.txt")}
        rerun = self.run_script("extract_text")
        self.assertIn("Cached:", rerun.stderr)
        self.assertEqual(raw_txt_hashes, {p: core.sha256_file(p) for p in raw_txt_hashes})
        self.run_script("clean_text")
        self.run_script("validate_konkani")
        self.assertFalse(any(r["included"] for r in core.read_jsonl(core.stage_path(self.root, "validated"))))
        self.run_script("validate_konkani", "--include-pending")
        pending = core.read_jsonl(core.stage_path(self.root, "validated"))
        self.assertTrue(all(r["provisional"] for r in pending if r["included"]))
        for row in registry:
            row.update(review_status="accepted", language_status="konkani", license_status="unknown")
            if row["category"] == "07_oral_transcripts":
                row.update(review_status="rejected", language_status="hindi")
        core.write_csv(self.root / "metadata/sources.csv", registry, core.SOURCE_FIELDS)
        self.run_script("validate_konkani")
        self.run_script("deduplicate")
        self.run_script("segment_sentences")
        final = core.read_jsonl(self.root / "data/processed/combined/documents.jsonl")
        self.assertEqual(len(final), 6)
        self.assertFalse(any(d["provisional"] for d in final))
        self.assertFalse(any(d["category"] == "07_oral_transcripts" for d in final))
        self.run_script("analyze_corpus")
        stats = core.read_csv(self.root / "results/category_comparison.csv")
        self.assertEqual(len(stats), 8)
        self.assertEqual(int(stats[0]["tokens"]), sum(int(s["tokens"]) for s in stats[1:]))
        self.assertEqual(len(list((self.root / "results/plots").rglob("*.png"))), 49)
        self.assertTrue(all(core.sha256_file(path) == digest for path, digest in hashes.items()))
        outputs = {path: core.sha256_file(path) for path in (self.root / "data/processed/combined").iterdir()}
        self.run_script("clean_text")
        self.run_script("validate_konkani")
        self.run_script("deduplicate")
        self.run_script("segment_sentences")
        self.assertEqual(outputs, {path: core.sha256_file(path) for path in outputs})
        self.run_script("analyze_corpus", "--stage", "extracted", "--no-plots")
        self.assertTrue((self.root / "results/iterations/extracted/word_frequencies.csv").exists())

    def test_new_files_changed_sources_and_missing_inputs(self):
        first = self.source("01_literature", "story.txt")
        self.run_script("extract_text")
        first_id = core.load_sources(self.root)[0]["document_id"]
        self.source("01_literature", "alphabetically_first.txt", SAMPLE + " नवी खबर.")
        self.run_script("extract_text")
        registry = core.load_sources(self.root)
        self.assertEqual(next(r for r in registry if r["source_file"].endswith("story.txt"))["document_id"], first_id)
        self.run_script("clean_text")
        first.write_text(SAMPLE + " बदल", encoding="utf-8")
        self.assertIn("Stale", self.run_script("validate_konkani", success=False).stderr)
        self.run_script("extract_text")
        first.unlink()
        self.run_script("extract_text")
        self.assertTrue(any(d["processing_status"] == "missing_source" for d in core.read_jsonl(core.stage_path(self.root, "extracted"))))

    def test_manual_exclusions_preserved_and_changes_invalidate(self):
        path = self.source("01_literature", "story.txt")
        self.run_script("extract_text")
        self.run_script("clean_text")
        core.write_csv(self.root / "metadata/excluded_documents.csv",
                       [{"file": core.relative(self.root, path), "reason": "manual test", "kept_instead": ""}],
                       ["file", "reason", "kept_instead"])
        self.assertIn("manual exclusions", self.run_script("validate_konkani", success=False).stderr)
        self.run_script("extract_text")
        self.run_script("clean_text")
        self.run_script("validate_konkani")
        self.run_script("deduplicate")
        self.run_script("segment_sentences")
        self.run_script("analyze_corpus", "--no-plots")
        exclusions = core.read_csv(self.root / "metadata/excluded_documents.csv")
        self.assertTrue(any(r["reason"] == "manual test" and not r["origin"] for r in exclusions))
        self.assertEqual(core.read_jsonl(self.root / "data/processed/combined/documents.jsonl"), [])

    def test_existing_extracted_text_and_raw_destination_protected(self):
        path = self.source("01_literature", "KOK_LIT_0001_story.txt")
        destination = self.root / "data/raw_txt/01_literature/KOK_LIT_0001.txt"
        core.atomic_text(destination, "User supplied text")
        self.assertIn("unregistered", self.run_script("extract_text", success=False).stderr)
        self.assertEqual(destination.read_text(), "User supplied text")
        self.run_script("extract_text", "--out", "data/raw", success=False)
        self.assertEqual(path.read_text(encoding="utf-8"), SAMPLE)

    def test_duplicate_filename_ids_fail_before_writing_outputs(self):
        self.source("01_literature", "KOK_LIT_0001_first.txt")
        self.source("01_literature", "KOK_LIT_0001_second.txt", SAMPLE + " आनी मोग.")
        result = self.run_script("extract_text", success=False)
        self.assertIn("multiple sources", result.stderr)
        self.assertFalse((self.root / "data/raw_txt").exists())


if __name__ == "__main__":
    unittest.main()
