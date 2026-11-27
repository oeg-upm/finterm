"""Functional integration contracts, all models/parsers replaced by doubles."""

from contextlib import contextmanager
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "evaluation_tests"))
from fixtures import Fixture, Factory, FakeRetriever

from extractor import FinancialExtractor, FinancialExtractionError, PunktSegmenter, SegmentationError
from extractor.financial_prompts import FinancialPromptBuilder, FINANCIAL_ENTITIES, PROMPT_VERSION, build_financial_dister
from extractor.retrieval import SentenceWorkerParser, load_financial_bank, make_financial_retriever
from extractor.segmentation import sentence_slices
from experimental_evaluation.syntax import ForestConfig, tree_data
from prompts import PromptBuilder, default_prompt_config, build_dister_prompt
from prompts.dister import DiSTERPrompt
from syntactic_retrieval import ParsedText, SyntaxTree, RetrievedExample


TEXT = "The European Central Bank assessed credit risk. Alice visited Paris."
FIRST, SECOND = TEXT.split(". ")
FIRST += "."


class Segments:
    info = {"policy": "synthetic-test-spans"}
    def segment(self, text):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S[^.]*\.(?:$|(?=\s))", text)]
        return sentence_slices(text, spans if spans else ([(0, len(text))] if text.strip() else []))


def engine_fixture(model, responses=None, **kwargs):
    fixture = Fixture()
    factory = Factory()
    adapter = factory(fixture.profiles[model])
    runtime = factory.runtimes[0]
    generate = adapter.generate
    def execute(prompt):
        text = prompt.text if isinstance(prompt, DiSTERPrompt) else prompt.messages[1].content.split("### Text\n", 1)[1].rsplit("\n\n### Output", 1)[0]
        value = (responses or {}).get(text, ["European Central Bank", "credit risk"] if text == FIRST else ["credit risk"])
        if isinstance(value, Exception):
            raise value
        runtime.tokenizer.response = value if isinstance(value, str) else json.dumps(value if model == "dister" else {"terms": value})
        return generate(prompt)
    adapter.generate = execute
    return FinancialExtractor(model, adapter=adapter, segmenter=Segments(), **kwargs), fixture, runtime


class FinancialIntegrationTests(unittest.TestCase):
    def test_three_models_default_to_sentence_extraction_and_exact_union(self):
        for model in ("qwen3", "gemma2", "dister"):
            with self.subTest(model=model):
                engine, fixture, runtime = engine_fixture(model)
                try:
                    result = engine.extract_detailed(TEXT)
                    self.assertTrue(result.complete)
                    self.assertEqual(list(result.terms), ["European Central Bank", "credit risk"])
                    self.assertEqual([s["text"] for s in result.sentences], [FIRST, SECOND])
                    self.assertEqual(len(runtime.model.calls), 2)
                    self.assertEqual(result.prompt_version, PROMPT_VERSION)
                    self.assertEqual(result.profile["generation"]["max_new_tokens"], 128 if model == "dister" else 512)
                finally:
                    fixture.close()

    def test_simple_api_returns_one_list(self):
        engine, fixture, _ = engine_fixture("qwen3")
        self.addCleanup(fixture.close)
        self.assertEqual(engine.extract(TEXT), ["European Central Bank", "credit risk"])

    def test_only_exact_duplicates_removed_not_case_punctuation_or_whitespace(self):
        engine, fixture, _ = engine_fixture("qwen3", {FIRST: ["CDS", "CDS", "cds", " CDS"], SECOND: ["M&A", "CDS"]})
        self.addCleanup(fixture.close)
        self.assertEqual(engine.extract(TEXT), ["CDS", "cds", " CDS", "M&A"])

    def test_invalid_output_is_not_silently_partial_success(self):
        engine, fixture, _ = engine_fixture("gemma2", {SECOND: "not JSON"})
        self.addCleanup(fixture.close)
        with self.assertRaises(FinancialExtractionError) as failure:
            engine.extract(TEXT)
        result = failure.exception.result
        self.assertFalse(result.complete)
        self.assertEqual(result.sentences[1]["error"]["phase"], "format")
        self.assertEqual(result.sentences[1]["generation"]["response_text"], "not JSON")
        self.assertEqual(list(result.terms), ["European Central Bank", "credit risk"])

    def test_generation_failure_has_traceback_and_keeps_other_sentences(self):
        engine, fixture, _ = engine_fixture("qwen3", {FIRST: RuntimeError("synthetic failure")})
        self.addCleanup(fixture.close)
        result = engine.extract_detailed(TEXT)
        self.assertFalse(result.complete)
        self.assertIn("synthetic failure", result.sentences[0]["error"]["traceback"])
        self.assertEqual(result.terms, ("credit risk",))

    def test_common_recovery_preserves_raw_gemma_and_native_dister(self):
        for model, wrapped in (("gemma2", '```json\n{"terms":["credit risk"]}\n```'),
                               ("dister", '["credit risk"]\"]')):
            with self.subTest(model=model):
                engine, fixture, _ = engine_fixture(model, {FIRST: wrapped, SECOND: []})
                try:
                    result = engine.extract_detailed(TEXT)
                    self.assertTrue(result.complete)
                    first = result.sentences[0]
                    self.assertFalse(first["formats"]["strict"]["accepted"])
                    self.assertTrue(first["formats"]["recovered"]["accepted"])
                    self.assertEqual(first["generation"]["response_text"], wrapped)
                    self.assertIn("fixture_eos", first["generation"]["raw_response"])
                finally:
                    fixture.close()

    def test_nonliteral_output_is_flagged_not_silently_rewritten(self):
        engine, fixture, _ = engine_fixture("qwen3", {FIRST: ["bid-ask prices"], SECOND: []})
        self.addCleanup(fixture.close)
        result = engine.extract_detailed(TEXT)
        self.assertEqual(result.terms, ("bid-ask prices",))
        self.assertEqual(result.sentences[0]["nonliteral_in_sentence"], ["bid-ask prices"])

    def test_empty_input_never_loads_punkt_or_model(self):
        engine, fixture, runtime = engine_fixture("dister")
        self.addCleanup(fixture.close)
        engine.segmenter = PunktSegmenter("missing-directory")
        self.assertEqual(engine.extract(" \r\n"), [])
        self.assertEqual(runtime.model.calls, [])
        self.assertFalse(engine.adapter.is_loaded)

    def test_missing_content_in_segmenter_is_an_error_before_model(self):
        engine, fixture, runtime = engine_fixture("qwen3")
        self.addCleanup(fixture.close)
        engine.segmenter = Mock(segment=Mock(return_value=sentence_slices(FIRST, [(0, len(FIRST))])))
        with self.assertRaises(SegmentationError):
            engine.extract(TEXT)
        self.assertEqual(runtime.model.calls, [])

    def test_unicode_crlf_slices_are_literal_original(self):
        text = "Équity changes.\r\nDebt falls."
        slices = sentence_slices(text, [(0, text.index(".") + 1), (text.index("Debt"), len(text))])
        self.assertEqual([s.text for s in slices], ["Équity changes.", "Debt falls."])
        self.assertEqual(slices[1].start, text.index("Debt"))

    def test_fastkassim_is_optional_and_only_three_examples_per_sentence(self):
        fixture = Fixture(); self.addCleanup(fixture.close)
        retriever = FakeRetriever(fixture.bank.examples)
        engine, local, runtime = engine_fixture("qwen3", retriever=retriever)
        self.addCleanup(local.close)
        result = engine.extract_detailed(TEXT, query_document_id="user-doc")
        self.assertTrue(result.complete)
        self.assertEqual(result.condition, "few_shot")
        self.assertEqual(len(retriever.calls), 2)
        self.assertTrue(all(len(s["retrieved_examples"]) == 3 for s in result.sentences))
        self.assertTrue(all(call[1]["query_document_id"] == "user-doc" for call in retriever.calls))

    def test_insufficient_retrieval_does_not_fallback_to_basic(self):
        fixture = Fixture(); self.addCleanup(fixture.close)
        retriever = FakeRetriever(fixture.bank.examples[:2])
        engine, local, runtime = engine_fixture("gemma2", retriever=retriever)
        self.addCleanup(local.close)
        result = engine.extract_detailed(TEXT)
        self.assertFalse(result.complete)
        self.assertEqual(runtime.model.calls, [])

    def test_dister_rejects_extra_demonstrations(self):
        with self.assertRaises(ValueError):
            FinancialExtractor("dister", retriever=Mock())

    def test_empty_explicit_document_id_rejected(self):
        engine, fixture, _ = engine_fixture("qwen3"); self.addCleanup(fixture.close)
        with self.assertRaises(ValueError):
            engine.extract(TEXT, query_document_id="")

    def test_from_local_validates_architecture_without_loading_weights(self):
        fixture = Fixture(); self.addCleanup(fixture.close)
        engine = FinancialExtractor.from_local("qwen3", fixture.profiles["qwen3"].load.source,
                                               punkt_data=fixture.resources.nltk_data)
        self.assertFalse(engine.adapter.is_loaded)
        self.assertIsNotNone(engine.model_files)
        with self.assertRaisesRegex(ValueError, "architecture"):
            FinancialExtractor.from_local("gemma2", fixture.profiles["qwen3"].load.source)


class FinancialPromptTests(unittest.TestCase):
    def test_all_new_generalist_conditions_have_contextual_financial_entity_scope(self):
        builder = FinancialPromptBuilder()
        fixture = Fixture(); self.addCleanup(fixture.close)
        selected = tuple(RetrievedExample(e, 0.5) for e in fixture.bank.examples[:3])
        prompts = [builder.build_basic(FIRST), builder.build_patterns(FIRST),
                   builder.build_few_shot(FIRST, selected), builder.build_hybrid(FIRST, ["European Central Bank"])]
        for prompt in prompts:
            self.assertIn(FINANCIAL_ENTITIES, prompt.messages[1].content)
            self.assertNotIn("Do NOT extract: named entities", prompt.messages[1].content)
            self.assertIn("unrelated companies", prompt.messages[1].content)
            self.assertIn("complete name", prompt.messages[1].content)
        self.assertIn("or a relevant named financial entity", prompts[-1].messages[1].content)

    def test_historical_builder_still_excludes_entities_and_does_not_change(self):
        historical = PromptBuilder(default_prompt_config("finance")).build_basic(FIRST)
        self.assertIn("Do NOT extract: named entities", historical.messages[1].content)
        self.assertNotIn(FINANCIAL_ENTITIES, historical.messages[1].content)

    def test_dister_native_schema_preserved_but_task_scope_versioned(self):
        old = build_dister_prompt("id", FIRST, "Finance").to_dict()
        new = build_financial_dister("id", FIRST).to_dict()
        self.assertEqual(set(old), set(new))
        self.assertEqual([x["from"] for x in old["conversations"]], [x["from"] for x in new["conversations"]])
        self.assertEqual(new["conversations"][0], old["conversations"][0])
        self.assertEqual(new["conversations"][-1], {"from": "gpt", "value": ""})
        self.assertIn("relevant named financial entities", new["conversations"][2]["value"])
        self.assertNotIn('{"terms"', json.dumps(new))
        self.assertNotEqual(new["conversations"][2]["value"], old["conversations"][2]["value"])

    def test_dister_adapter_keeps_native_tokenization(self):
        engine, fixture, runtime = engine_fixture("dister"); self.addCleanup(fixture.close)
        result = engine.extract_detailed(TEXT)
        self.assertTrue(result.complete)
        self.assertEqual(runtime.tokenizer.template_calls, [])
        self.assertTrue(all(call[1]["add_special_tokens"] for call in runtime.tokenizer.tokenization_calls))
        self.assertTrue(all(s["generation"]["json_response"] is None for s in result.sentences))


class PunktAndBankTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture(); self.addCleanup(self.fixture.close)

    def test_punkt_uses_original_parameters_and_restores_global_search_path(self):
        package, data = ModuleType("nltk"), ModuleType("nltk.data")
        tokenize, punkt = ModuleType("nltk.tokenize"), ModuleType("nltk.tokenize.punkt")
        package.__path__ = []; tokenize.__path__ = []
        package.data, package.tokenize, tokenize.punkt = data, tokenize, punkt
        data.path = ["before"]
        root = Path(self.fixture.resources.nltk_data) / "tokenizers/punkt_tab/english"
        data.find = Mock(return_value=root)
        punkt.PunktTokenizer = Mock()
        punkt.PunktTokenizer.return_value.span_tokenize.return_value = [(0, len(FIRST)), (len(FIRST) + 1, len(TEXT))]
        modules = {"nltk": package, "nltk.data": data, "nltk.tokenize": tokenize, "nltk.tokenize.punkt": punkt}
        with patch.dict(sys.modules, modules), patch("experimental_evaluation.resources.private_path", side_effect=lambda p: Path(p).resolve()):
            segmenter = PunktSegmenter(self.fixture.resources.nltk_data)
            self.assertEqual([s.text for s in segmenter.segment(TEXT)], [FIRST, SECOND])
        self.assertEqual(data.path, ["before"])
        punkt.PunktTokenizer.assert_called_once_with("english")
        self.assertEqual(len(segmenter.info["sha256"]), 4)

    def test_missing_punkt_is_not_replaced_by_rules_or_silent_truncation(self):
        with patch.dict(sys.modules, {"nltk": None}):
            with self.assertRaises(SegmentationError):
                PunktSegmenter().segment(TEXT)

    def test_bank_never_accepts_reserved_sources_or_key_files(self):
        segmenter = Segments()
        with self.assertRaises(ValueError):
            load_financial_bank(self.fixture.bank_path, authorized_documents=["CELEX:32013R0575"], segmenter=segmenter)
        with self.assertRaises(ValueError):
            load_financial_bank(self.fixture.root / "annotations.key", authorized_documents=["doc"], segmenter=segmenter)

    def test_bank_preserves_terms_and_provisional_review_status(self):
        examples, status = load_financial_bank(self.fixture.bank_path, authorized_documents=["CELEX:32021B0417"], segmenter=Segments())
        self.assertEqual(examples, self.fixture.bank.examples)
        self.assertEqual(status, "provisional_or_review_not_declared")

    def test_retrieval_factory_uses_original_filter_and_syntax_kernel_configuration(self):
        with patch("extractor.retrieval.resource_inventory") as preflight:
            retriever = make_financial_retriever(self.fixture.bank.examples, authorized_documents=["CELEX:32021B0417"],
                                                resources=self.fixture.resources)
        preflight.assert_called_once_with(self.fixture.resources)
        self.assertEqual(retriever.config.top_k, 3)
        self.assertTrue(retriever.config.exclude_same_text and retriever.config.exclude_query_document)
        self.assertIsInstance(retriever.parser, SentenceWorkerParser)

    def test_sentence_parser_sends_whole_unit_without_resegmentation(self):
        tree = SyntaxTree("ROOT", (SyntaxTree("NN"),))
        parser = SentenceWorkerParser(self.fixture.resources, ForestConfig())
        with patch("extractor.retrieval.worker_call", return_value={"tree": tree_data(tree)}) as worker:
            self.assertEqual(parser.parse(FIRST).trees, (tree,))
        self.assertEqual(worker.call_args.args[0]["text"], FIRST)
        self.assertEqual(worker.call_args.args[0]["operation"], "parse")


if __name__ == "__main__":
    unittest.main()
