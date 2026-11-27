"""Literal segment-level silver lists, without sentence propagation or filters."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from corpus_readers import Granularity, ReferenceKind, discover_corpora, read_celex_segment

from fixtures import ReaderFixtureCase


class CelexReaderTests(ReaderFixtureCase):
    def segment(self):
        return discover_corpora(self.root).celex_segments[0]

    def test_pairing_does_not_depend_on_docsutf8_or_gold_folder_names(self):
        text, key = self.celex()
        segment = self.segment()
        self.assertEqual((segment.text_path, segment.key_path), (text, key))
        self.assertEqual(segment.document_id, "CELEX:32013R0549")
        self.assertEqual(segment.identifier, "CELEX:32013R0549_seg_0")

    def test_full_segment_and_all_stored_annotations_are_preserved(self):
        text, key = self.celex()
        result = read_celex_segment(self.segment())
        self.assertEqual(len(result.units), 1)
        example = result.examples[0]
        self.assertEqual(example.text, text.read_bytes().decode("utf-8"))
        self.assertEqual(example.terms, ("CIF value", "CIF", "Requirements", "European Central Bank", "CIF", "  edge  "))
        self.assertEqual(example.domain, "finance")
        self.assertEqual(result.units[0].raw_annotation_lines, example.terms)
        self.assertEqual({item.path for item in result.sources}, {text, key})

    def test_nonliteral_forms_are_diagnostics_and_are_not_corrected(self):
        self.celex()
        result = read_celex_segment(self.segment())
        missing = [issue for issue in result.diagnostics if issue.code == "nonliteral_key_entry"]
        self.assertEqual({issue.line for issue in missing}, {1, 3})
        self.assertTrue(all(issue.severity == "warning" for issue in missing))
        self.assertIn("CIF value", result.examples[0].terms)
        self.assertNotIn("CIF-value", result.examples[0].terms)
        self.assertIn("Requirements", result.examples[0].terms)
        self.assertEqual(result.examples[0].terms.count("CIF"), 2)
        self.assertIn("duplicate_key_entry", [issue.code for issue in result.diagnostics])

    def test_reference_quality_and_granularity_are_explicit_and_not_human_sentence_gold(self):
        self.celex()
        trace = read_celex_segment(self.segment()).units[0].trace
        self.assertEqual(trace.reference_kind, ReferenceKind.SILVER)
        self.assertEqual(trace.granularity, Granularity.SEGMENT)
        self.assertIsNone(trace.text_line)
        self.assertEqual(trace.segment_id, "0")

    def test_segment_identifier_keeps_source_padding_and_document_grouping(self):
        self.celex(source_id="32013R0549_seg_001")
        self.celex(source_id="32013R0549_seg_2")
        results = [read_celex_segment(segment) for segment in discover_corpora(self.root).celex_segments]
        self.assertEqual({result.examples[0].document_id for result in results}, {"CELEX:32013R0549"})
        self.assertEqual(results[0].examples[0].identifier, "CELEX:32013R0549_seg_001")
        self.assertEqual(results[0].units[0].trace.segment_id, "001")

    def test_multiple_sentences_never_receive_copied_segment_annotations(self):
        self.celex(text="First sentence.\r\nSecond sentence.\nThird sentence.", key="sentence")
        result = read_celex_segment(self.segment())
        self.assertEqual(len(result.examples), 1)
        self.assertEqual(result.examples[0].terms, ("sentence",))
        self.assertIn("Third sentence.", result.examples[0].text)

    def test_missing_key_is_not_regenerated_or_treated_as_a_negative_reference(self):
        _, key = self.celex()
        key.unlink()  # Synthetic file only.
        catalog = discover_corpora(self.root)
        result = read_celex_segment(catalog.celex_segments[0])
        self.assertEqual(result.examples, ())
        self.assertIn("missing_celex_pair", [issue.code for issue in result.diagnostics])
        self.assertFalse(key.exists())

    def test_ambiguous_copies_are_not_paired_arbitrarily(self):
        self.celex(prefix="copy-a")
        self.celex(prefix="copy-b")
        catalog = discover_corpora(self.root)
        self.assertEqual(catalog.celex_segments, ())
        self.assertIn("duplicate_celex_segment", [issue.code for issue in catalog.diagnostics])

    def test_wrong_filename_binding_is_rejected_before_reading(self):
        self.celex()
        segment = replace(self.segment(), segment_id="999")
        with patch.object(Path, "open", side_effect=AssertionError("No source reads")):
            result = read_celex_segment(segment)
        self.assertEqual(result.examples, ())
        self.assertEqual(result.diagnostics[0].code, "wrong_celex_identity")

    def test_empty_existing_key_is_explicit_and_not_the_same_as_a_missing_key(self):
        self.celex(key="")
        result = read_celex_segment(self.segment())
        self.assertEqual(result.examples[0].terms, ())
        self.assertIn("empty_key_reference", [issue.code for issue in result.diagnostics])

    def test_blank_annotation_entries_are_reported_without_silently_dropping_them(self):
        self.celex(key="valid\n\nother")
        result = read_celex_segment(self.segment())
        self.assertEqual(result.examples, ())
        self.assertEqual(result.units[0].raw_annotation_lines, ("valid", "", "other"))
        self.assertIn("empty_key_entry", [issue.code for issue in result.diagnostics])

    def test_unicode_separators_tabs_and_punctuation_inside_terms_are_not_split_or_cleaned(self):
        value = 'quoted "term"\t€\u2028separator'
        self.celex(text="Synthetic " + value + " end.", key=value)
        result = read_celex_segment(self.segment())
        self.assertEqual(result.examples[0].terms, (value,))

    def test_encoding_failure_is_not_repaired_with_a_different_codec(self):
        self.celex(key=b"\xff")
        result = read_celex_segment(self.segment())
        self.assertEqual(result.examples, ())
        self.assertIn("invalid_utf8", [issue.code for issue in result.diagnostics])
