"""Native IOB conversion and diagnostic/quarantine tests on synthetic data."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from corpus_readers import (
    ActerDocument, Granularity, ReferenceKind, discover_corpora, read_acter_document,
)
from syntactic_retrieval.models import AnnotatedExample

from fixtures import ReaderFixtureCase


class ActerReaderTests(ReaderFixtureCase):
    def document(self):
        return discover_corpora(self.root).acter_documents[0]

    def test_discovery_uses_the_native_variant_and_relocated_english_paths(self):
        annotation, tokenized, original = self.acter()
        catalog = discover_corpora(self.root)
        self.assertEqual(len(catalog.acter_documents), 1)
        document = catalog.acter_documents[0]
        self.assertEqual(document.document_id, "ACTER:corp_en_01")
        self.assertEqual(document.domain, "corruption")
        self.assertEqual((document.annotation_path, document.tokenized_path, document.original_path), (annotation, tokenized, original))

    def test_exact_text_terms_case_spacing_duplicates_and_negative_units_are_preserved(self):
        self.acter()
        result = read_acter_document(self.document())
        self.assertEqual(len(result.examples), 2)
        example = result.examples[0]
        self.assertIsInstance(example, AnnotatedExample)
        self.assertEqual(example.identifier, "ACTER:corp_en_01:unit:000001")
        self.assertEqual(example.text, "Central  bank rates and rates .  ")
        self.assertEqual(example.terms, ("Central  bank", "rates", "rates"))
        self.assertEqual(result.examples[1].terms, ())
        self.assertEqual(result.units[0].tokens, ("Central", "bank", "rates", "and", "rates", "."))
        self.assertEqual(result.units[0].iob_labels, ("B", "I", "B", "O", "B", "O"))

    def test_trace_identifies_document_unit_files_lines_and_reference_granularity(self):
        annotation, tokenized, original = self.acter()
        result = read_acter_document(self.document())
        trace = result.units[0].trace
        self.assertEqual(trace.document_id, "ACTER:corp_en_01")
        self.assertEqual(trace.unit_number, 1)
        self.assertEqual(trace.text_line, 1)
        self.assertEqual((trace.annotation_start_line, trace.annotation_end_line), (1, 6))
        self.assertEqual(trace.granularity, Granularity.IOB_UNIT)
        self.assertEqual(trace.reference_kind, ReferenceKind.HUMAN)
        self.assertEqual((trace.annotation_path, trace.text_path, trace.original_text_path), (annotation, tokenized, original))
        self.assertEqual({item.path for item in result.sources}, {annotation, tokenized})

    def test_no_terms_are_inferred_from_a_global_list_or_the_original_raw_text(self):
        _, _, original = self.acter(annotation="long\tB\nterm\tI\n.\tO\n", tokenized="long term .\n")
        self.write(original.parent.parent / "annotations" / "unique_annotation_lists" / "corp_en_terms.tsv", "long\tSpecific_Term\nterm\tCommon_Term\n")
        allowed = {self.document().annotation_path, self.document().tokenized_path}
        actual_open = Path.open

        def only_native_inputs(path, *args, **kwargs):
            if path not in allowed:
                raise AssertionError("Global lists/raw original must not be used to assign unit terms")
            return actual_open(path, *args, **kwargs)

        with patch.object(Path, "open", only_native_inputs):
            result = read_acter_document(self.document())
        self.assertEqual(result.examples[0].terms, ("long term",))

    def test_io_with_named_entities_and_nonenglish_files_are_not_chosen(self):
        self.acter()
        self.acter(source_id="corp_en_02", scheme="io_annotations")
        self.acter(source_id="corp_en_03", named="with_named_entities")
        self.acter(source_id="corp_fr_01")
        catalog = discover_corpora(self.root)
        self.assertEqual([item.source_id for item in catalog.acter_documents], ["corp_en_01"])

    def test_explicitly_supplied_wrong_variant_or_domain_is_rejected_before_content_reads(self):
        annotation, tokenized, original = self.acter(scheme="io_annotations")
        document = ActerDocument("corp_en_01", "corruption", annotation, tokenized, original)
        with patch.object(Path, "open", side_effect=AssertionError("No content reads")):
            self.assertEqual(read_acter_document(document).diagnostics[0].code, "wrong_acter_variant")
        self.acter()
        document = replace(self.document(), domain="finance")
        with patch.object(Path, "open", side_effect=AssertionError("No content reads")):
            self.assertEqual(read_acter_document(document).diagnostics[0].code, "wrong_acter_domain")

    def broken_document(self):
        annotation = "long\tB\nterm\tI\n\ncontinuation\tI\nstops\tO\n\nSafe\tB\nunit\tI\n"
        self.acter(annotation=annotation, tokenized="long term \ncontinuation stops \nSafe unit\n")
        return self.document()

    def test_interrupted_iob_span_quarantines_both_units_without_relabeling_or_merging(self):
        result = read_acter_document(self.broken_document())
        self.assertIsNone(result.units[0].example)
        self.assertIsNone(result.units[1].example)
        self.assertEqual(result.units[1].iob_labels, ("I", "O"))
        self.assertEqual(result.units[1].raw_annotation_lines, ("continuation\tI", "stops\tO"))
        cuts = [issue for issue in result.diagnostics if issue.code == "iob_cross_unit_continuation"]
        self.assertEqual({issue.line for issue in cuts}, {4})
        self.assertEqual(len(cuts), 2)
        self.assertEqual(result.examples[0].terms, ("Safe unit",))
        self.assertNotIn("long term continuation", result.examples[0].terms)

    def test_unit_selection_keeps_source_numbers_and_validates_external_boundary_context(self):
        document = self.broken_document()
        previous = read_acter_document(document, unit_numbers={1})
        self.assertEqual(len(previous.units), 1)
        self.assertEqual(previous.examples, ())
        self.assertIn("iob_cross_unit_continuation", [issue.code for issue in previous.diagnostics])
        valid = read_acter_document(document, unit_numbers={3})
        self.assertEqual(valid.examples[0].identifier, "ACTER:corp_en_01:unit:000003")

    def test_orphan_I_after_O_does_not_turn_into_B(self):
        self.acter(annotation="plain\tO\nbroken\tI\n", tokenized="plain broken\n")
        result = read_acter_document(self.document())
        self.assertEqual(result.examples, ())
        self.assertEqual(result.units[0].iob_labels, ("O", "I"))
        self.assertIn("orphan_iob_I", [issue.code for issue in result.diagnostics])

    def test_malformed_columns_unknown_labels_and_alignment_mismatches_are_explicit(self):
        for annotation, tokenized, code in (
            ("term\tB\textra\n", "term\n", "malformed_iob_row"),
            ("term\tB-Specific_Term\n", "term\n", "unknown_iob_label"),
            ("term\tB\n", "Term\n", "token_text_mismatch"),
        ):
            with self.subTest(code=code):
                self.acter(annotation=annotation, tokenized=tokenized)
                result = read_acter_document(self.document())
                self.assertEqual(result.examples, ())
                self.assertIn(code, [issue.code for issue in result.diagnostics])

    def test_unit_count_mismatch_does_not_create_annotations_for_extra_text(self):
        self.acter(annotation="term\tB\n", tokenized="term\nextra source line\n")
        result = read_acter_document(self.document())
        self.assertEqual(result.examples, ())
        self.assertIn("unit_count_mismatch", [issue.code for issue in result.diagnostics])

    def test_missing_counterparts_and_duplicate_document_copies_are_diagnosed(self):
        _, _, original = self.acter()
        original.unlink()  # Only a synthetic temporary fixture.
        catalog = discover_corpora(self.root)
        self.assertIn("acter_text_counterpart", [issue.code for issue in catalog.diagnostics])
        self.assertEqual(read_acter_document(catalog.acter_documents[0]).examples, ())
        self.acter(prefix="second-copy")
        catalog = discover_corpora(self.root)
        self.assertEqual(catalog.acter_documents, ())
        self.assertIn("duplicate_acter_document", [issue.code for issue in catalog.diagnostics])

    def test_invalid_utf8_and_bom_are_not_silently_replaced_or_removed(self):
        for data, code in ((b"\xff\tB\n", "invalid_utf8"), (b"\xef\xbb\xbfterm\tB\n", "utf8_bom")):
            with self.subTest(code=code):
                self.acter(annotation=data, tokenized="term\n")
                result = read_acter_document(self.document())
                self.assertEqual(result.examples, ())
                self.assertIn(code, [issue.code for issue in result.diagnostics])

    def test_unavailable_requested_units_are_not_fabricated(self):
        self.acter()
        result = read_acter_document(self.document(), unit_numbers={99})
        self.assertEqual(result.examples, ())
        self.assertIn("missing_acter_unit", [issue.code for issue in result.diagnostics])

    def test_wrong_counterpart_identity_is_not_used_even_if_supplied_explicitly(self):
        self.acter()
        document = replace(self.document(), tokenized_path=self.root / "corp_en_99.txt")
        with patch.object(Path, "open", side_effect=AssertionError("No content reads")):
            result = read_acter_document(document)
        self.assertEqual(result.examples, ())
        self.assertEqual(result.diagnostics[0].code, "wrong_acter_identity")

    def test_empty_iob_source_does_not_become_an_unannotated_negative_example(self):
        self.acter(annotation="", tokenized="")
        result = read_acter_document(self.document())
        self.assertEqual(result.examples, ())
        self.assertIn("empty_acter_annotation", [issue.code for issue in result.diagnostics])
