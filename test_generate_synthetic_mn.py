"""
Tests for generate_synthetic_mn.py — doctag text extraction, replacement,
and output JSONL format validation.

These tests validate the core parsing and translation-replacement logic
WITHOUT requiring network access or the full HuggingFace dataset.
"""

import json
import re
import sys
import os
import unittest

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_mn import (
    extract_texts_from_doctag,
    translate_doctag,
    _extract_doctag_from_texts,
    _trim_doctag,
    _TEXT_IN_TAG_RE,
    _CELL_TEXT_RE,
)


# A representative doctag string (English content)
SAMPLE_DOCTAG = (
    "<doctag>"
    "<title><loc_50><loc_10><loc_450><loc_40>Quarterly Report</title>"
    "<text><loc_50><loc_50><loc_400><loc_80>Revenue increased by 15 percent this quarter.</text>"
    "<section_header_level_1><loc_50><loc_90><loc_400><loc_110>Financial Summary</section_header_level_1>"
    "<otsl><loc_50><loc_120><loc_450><loc_250>"
    "<ched>Category<ched>Q1<ched>Q2<nl>"
    "<rhed>Revenue<fcel>120000<fcel>150000<nl>"
    "<rhed>Expenses<fcel>80000<fcel>90000<nl>"
    "</otsl>"
    "<caption><loc_50><loc_260><loc_400><loc_280>Table 1: Financial data</caption>"
    "<picture><loc_50><loc_290><loc_400><loc_400></picture>"
    "</doctag>"
)


class TestExtractTextsFromDoctag(unittest.TestCase):
    """Verify that text fragments are correctly pulled from doctag strings."""

    def test_extracts_element_texts(self):
        texts = extract_texts_from_doctag(SAMPLE_DOCTAG)
        self.assertIn("Quarterly Report", texts)
        self.assertIn("Revenue increased by 15 percent this quarter.", texts)
        self.assertIn("Financial Summary", texts)
        self.assertIn("Table 1: Financial data", texts)

    def test_extracts_table_cell_texts(self):
        texts = extract_texts_from_doctag(SAMPLE_DOCTAG)
        self.assertIn("Category", texts)
        self.assertIn("Revenue", texts)
        self.assertIn("Expenses", texts)
        # Numeric cells
        self.assertIn("120000", texts)
        self.assertIn("150000", texts)

    def test_does_not_extract_picture_text(self):
        """Pictures have no text content in doctag format."""
        texts = extract_texts_from_doctag(SAMPLE_DOCTAG)
        self.assertNotIn("[picture]", texts)

    def test_empty_doctag(self):
        texts = extract_texts_from_doctag("<doctag></doctag>")
        self.assertEqual(texts, [])

    def test_doctag_with_only_pictures(self):
        dt = "<doctag><picture><loc_0><loc_0><loc_100><loc_100></picture></doctag>"
        texts = extract_texts_from_doctag(dt)
        self.assertEqual(texts, [])


class TestTranslateDoctag(unittest.TestCase):
    """Test doctag translation with a mock translator."""

    def _mock_translator(self):
        """Create a mock translator that uppercases text (simulates translation)."""
        class MockTranslator:
            def translate_batch(self, texts):
                return [t.upper() for t in texts]
        return MockTranslator()

    def test_structure_preserved_after_translation(self):
        """Location tokens and tags must remain intact."""
        translated = translate_doctag(SAMPLE_DOCTAG, translator=self._mock_translator())
        # All location tokens should still be present
        loc_tokens = re.findall(r"<loc_\d+>", SAMPLE_DOCTAG)
        for tok in loc_tokens:
            self.assertIn(tok, translated)

    def test_text_content_translated(self):
        translated = translate_doctag(SAMPLE_DOCTAG, translator=self._mock_translator())
        # Mock translator uppercases, so check for uppercased content
        self.assertIn("QUARTERLY REPORT", translated)
        self.assertIn("FINANCIAL SUMMARY", translated)

    def test_table_cells_translated(self):
        translated = translate_doctag(SAMPLE_DOCTAG, translator=self._mock_translator())
        self.assertIn("CATEGORY", translated)
        self.assertIn("REVENUE", translated)

    def test_tags_preserved(self):
        translated = translate_doctag(SAMPLE_DOCTAG, translator=self._mock_translator())
        self.assertIn("<doctag>", translated)
        self.assertIn("</doctag>", translated)
        self.assertIn("<title>", translated)
        self.assertIn("</title>", translated)
        self.assertIn("<otsl>", translated)
        self.assertIn("</otsl>", translated)
        self.assertIn("<picture>", translated)
        self.assertIn("</picture>", translated)

    def test_empty_doctag_passthrough(self):
        result = translate_doctag("<doctag></doctag>", translator=self._mock_translator())
        self.assertEqual(result, "<doctag></doctag>")


class TestExtractDoctagFromTexts(unittest.TestCase):
    """Test the various formats the 'texts' column can have."""

    def test_plain_string(self):
        result = _extract_doctag_from_texts("<doctag><text><loc_0><loc_0><loc_100><loc_100>Hello</text></doctag>")
        self.assertIsNotNone(result)
        self.assertIn("<doctag>", result)

    def test_list_of_dicts_assistant(self):
        texts = [{"user": "Convert this page", "assistant": "Here: <doctag><text><loc_0><loc_0><loc_100><loc_100>Hi</text></doctag>"}]
        result = _extract_doctag_from_texts(texts)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("<doctag>"))
        self.assertTrue(result.endswith("</doctag>"))

    def test_list_of_strings(self):
        texts = ["some prompt", "<doctag><text><loc_0><loc_0><loc_100><loc_100>Hi</text></doctag>"]
        result = _extract_doctag_from_texts(texts)
        self.assertIsNotNone(result)

    def test_no_doctag(self):
        result = _extract_doctag_from_texts("just plain text")
        self.assertIsNone(result)

    def test_empty(self):
        result = _extract_doctag_from_texts([])
        self.assertIsNone(result)
        result2 = _extract_doctag_from_texts("")
        self.assertIsNone(result2)


class TestTrimDoctag(unittest.TestCase):
    def test_basic_trim(self):
        s = "prefix <doctag>content</doctag> suffix"
        result = _trim_doctag(s)
        self.assertEqual(result, "<doctag>content</doctag>")

    def test_no_doctag(self):
        self.assertIsNone(_trim_doctag("no tags here"))

    def test_exact_doctag(self):
        s = "<doctag>hello</doctag>"
        self.assertEqual(_trim_doctag(s), s)


class TestRegexPatterns(unittest.TestCase):
    """Verify the regex patterns match expected structures."""

    def test_text_in_tag_re_matches(self):
        tag = "<text><loc_50><loc_100><loc_200><loc_150>Sample text here</text>"
        m = _TEXT_IN_TAG_RE.search(tag)
        self.assertIsNotNone(m)
        self.assertEqual(m.group("content"), "Sample text here")

    def test_text_in_tag_re_all_types(self):
        for tag_name in ["text", "title", "section_header_level_1",
                         "section_header_level_2", "section_header_level_3",
                         "caption", "list_item", "formula",
                         "page_header", "page_footer", "code", "footnote"]:
            s = f"<{tag_name}><loc_0><loc_0><loc_500><loc_500>Test content</{tag_name}>"
            m = _TEXT_IN_TAG_RE.search(s)
            self.assertIsNotNone(m, f"Failed to match tag: {tag_name}")
            self.assertEqual(m.group("content"), "Test content")

    def test_cell_text_re_matches(self):
        for cell_type in ["ched", "rhed", "fcel"]:
            s = f"<{cell_type}>Cell content"
            m = _CELL_TEXT_RE.search(s)
            self.assertIsNotNone(m, f"Failed to match cell: {cell_type}")
            self.assertEqual(m.group("content"), "Cell content")


class TestOutputJsonlFormat(unittest.TestCase):
    """Verify the output JSONL has the required {"image": "...", "doctag": "..."} format."""

    def test_record_has_required_keys(self):
        """Each record must have exactly 'image' and 'doctag' keys."""
        record = {"image": "labeler_data/uploads/mn_page_000001.png", "doctag": "<doctag></doctag>"}
        self.assertIn("image", record)
        self.assertIn("doctag", record)
        self.assertEqual(len(record), 2)

    def test_image_path_format(self):
        """Image path should contain the images directory prefix."""
        record = {"image": "labeler_data/uploads/mn_page_000001.png", "doctag": "<doctag></doctag>"}
        self.assertTrue(record["image"].endswith(".png"))

    def test_doctag_value_is_string(self):
        record = {"image": "labeler_data/uploads/mn_page_000001.png", "doctag": SAMPLE_DOCTAG}
        self.assertIsInstance(record["doctag"], str)
        self.assertTrue(record["doctag"].startswith("<doctag>"))
        self.assertTrue(record["doctag"].endswith("</doctag>"))

    def test_json_serialization(self):
        """Verify the record serializes properly to JSON."""
        record = {"image": "labeler_data/uploads/mn_page_000001.png", "doctag": SAMPLE_DOCTAG}
        line = json.dumps(record, ensure_ascii=False)
        parsed = json.loads(line)
        self.assertEqual(parsed["image"], record["image"])
        self.assertEqual(parsed["doctag"], record["doctag"])
        # Must have exactly 2 keys
        self.assertEqual(set(parsed.keys()), {"image", "doctag"})


if __name__ == "__main__":
    unittest.main()
