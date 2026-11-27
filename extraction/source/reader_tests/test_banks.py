"""Caller-authorized banks, stable IDs and reserved-document separation."""

from pathlib import Path
from unittest.mock import patch

from corpus_readers import (
    BankAuthorization, BankAuthorizationError, ReferenceKind, build_example_bank,
    discover_corpora, read_acter_document, read_authorized_bank, read_celex_segment,
)
from extractor import TermExtractor

from fixtures import ReaderFixtureCase


class AuthorizedBankTests(ReaderFixtureCase):
    def set_sources(self):
        self.acter(source_id="corp_en_01")
        self.acter(source_id="corp_en_02")
        self.celex(source_id="32013R0549_seg_0")
        self.celex(source_id="32013R0549_seg_1")
        return discover_corpora(self.root)

    def test_bank_can_supply_either_corpus_and_keeps_original_annotated_example_types(self):
        catalog = self.set_sources()
        for documents, reference in (({"ACTER:corp_en_01"}, ReferenceKind.HUMAN), ({"CELEX:32013R0549"}, ReferenceKind.SILVER)):
            with self.subTest(documents=documents):
                bank = read_authorized_bank(catalog, BankAuthorization(documents))
                self.assertEqual(len(bank.examples), 2)
                self.assertEqual(bank.reference_kinds, frozenset({reference}))
                engine = TermExtractor(examples=bank.examples)  # Construction only.
                self.assertEqual(engine._examples, bank.examples)
                self.assertIsNone(engine._retriever)
                self.assertEqual(engine._adapters, {})

    def test_authorization_happens_before_unauthorized_or_evaluation_content_reads(self):
        catalog = self.set_sources()
        authorized = catalog.acter_documents[0]
        allowed = {authorized.annotation_path, authorized.tokenized_path}
        actual_open = Path.open

        def authorized_only(path, *args, **kwargs):
            if path not in allowed:
                raise AssertionError("No unauthorized content read")
            if args != ("rb",):
                raise AssertionError("Only read-only binary opens are allowed")
            return actual_open(path, *args, **kwargs)

        policy = BankAuthorization({authorized.document_id}, evaluation_document_ids={"ACTER:corp_en_02", "CELEX:32013R0549"})
        with patch.object(Path, "open", authorized_only):
            bank = read_authorized_bank(catalog, policy)
        self.assertEqual({item.document_id for item in bank.examples}, {authorized.document_id})

    def test_empty_authorization_and_explicit_empty_example_set_read_no_contents(self):
        catalog = self.set_sources()
        for policy in (BankAuthorization(set()), BankAuthorization({"ACTER:corp_en_01"}, authorized_example_ids=set())):
            with self.subTest(policy=policy), patch.object(Path, "open", side_effect=AssertionError("No content reads")):
                bank = read_authorized_bank(catalog, policy)
            self.assertEqual(bank.examples, ())
            self.assertEqual(bank.sources, ())

    def test_explicit_example_selection_preserves_unique_source_unit_ids(self):
        catalog = self.set_sources()
        ids = {"ACTER:corp_en_01:unit:000002", "CELEX:32013R0549_seg_1"}
        policy = BankAuthorization({"ACTER:corp_en_01", "CELEX:32013R0549"}, authorized_example_ids=ids)
        bank = read_authorized_bank(catalog, policy)
        self.assertEqual({item.identifier for item in bank.examples}, ids)
        self.assertEqual(len(bank.examples), len({item.identifier for item in bank.examples}))
        self.assertEqual(bank.reference_kinds, frozenset({ReferenceKind.HUMAN, ReferenceKind.SILVER}))
        self.assertEqual({record.trace.unit_id for record in bank.records}, ids)

    def test_conflicting_document_or_example_permissions_are_rejected_explicitly(self):
        with self.assertRaises(BankAuthorizationError):
            BankAuthorization({"ACTER:corp_en_01"}, evaluation_document_ids={"ACTER:corp_en_01"})
        with self.assertRaises(BankAuthorizationError):
            BankAuthorization({"ACTER:corp_en_01"}, evaluation_example_ids={"ACTER:corp_en_01:unit:000002"})
        with self.assertRaises(BankAuthorizationError):
            BankAuthorization({"CELEX:32013R0549"}, evaluation_example_ids={"CELEX:32013R0549_seg_1"})
        with self.assertRaises(BankAuthorizationError):
            BankAuthorization(set(), authorized_example_ids={"CELEX:32013R0549_seg_1"})

    def test_in_memory_selection_cannot_mix_reserved_data_with_demonstrations(self):
        catalog = self.set_sources()
        readings = [read_acter_document(document) for document in catalog.acter_documents]
        policy = BankAuthorization({"ACTER:corp_en_01"}, evaluation_document_ids={"ACTER:corp_en_02"})
        bank = build_example_bank(readings, policy)
        self.assertEqual({item.document_id for item in bank.examples}, {"ACTER:corp_en_01"})
        self.assertTrue(all(record.trace.document_id != "ACTER:corp_en_02" for record in bank.records))

    def test_duplicate_readings_are_not_silently_deduplicated(self):
        self.acter()
        reading = read_acter_document(discover_corpora(self.root).acter_documents[0])
        with self.assertRaises(ValueError):
            build_example_bank([reading, reading], BankAuthorization({"ACTER:corp_en_01"}))

    def test_problematic_iob_units_are_withheld_with_explicit_diagnostics(self):
        self.acter(annotation="cut\tB\n\ncontinued\tI\n\nSafe\tB\n", tokenized="cut\ncontinued\nSafe\n")
        bank = read_authorized_bank(discover_corpora(self.root), BankAuthorization({"ACTER:corp_en_01"}))
        self.assertEqual([item.identifier for item in bank.examples], ["ACTER:corp_en_01:unit:000003"])
        self.assertEqual(sum(issue.code == "unit_withheld" for issue in bank.diagnostics), 2)
        self.assertIn("iob_cross_unit_continuation", [issue.code for issue in bank.diagnostics])

    def test_financial_nonliteral_entries_remain_available_with_silver_metadata(self):
        self.celex()
        bank = read_authorized_bank(discover_corpora(self.root), BankAuthorization({"CELEX:32013R0549"}))
        self.assertIn("CIF value", bank.examples[0].terms)
        self.assertIn("European Central Bank", bank.examples[0].terms)
        self.assertIn("nonliteral_key_entry", [issue.code for issue in bank.diagnostics])
        self.assertEqual(bank.records[0].trace.reference_kind, ReferenceKind.SILVER)

    def test_unknown_authorized_documents_or_examples_do_not_trigger_a_fallback(self):
        catalog = self.set_sources()
        bank = read_authorized_bank(catalog, BankAuthorization({"ACTER:corp_en_999"}))
        self.assertEqual(bank.examples, ())
        self.assertIn("authorized_document_not_found", [issue.code for issue in bank.diagnostics])
        bank = read_authorized_bank(catalog, BankAuthorization(
            {"ACTER:corp_en_01"}, authorized_example_ids={"ACTER:corp_en_01:unit:000099"}
        ))
        self.assertEqual(bank.examples, ())
        self.assertIn("requested_example_not_found", [issue.code for issue in bank.diagnostics])
