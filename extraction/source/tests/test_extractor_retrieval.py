"""Existing retriever selection/exclusions, with synthetic parser/score doubles."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from extractor import InvalidExtractionRequest, MissingRetrievalResourcesError, TermExtractor
from prompts import InsufficientExamplesError, PromptBuilder
from syntactic_retrieval import SyntacticRetriever

from extractor_doubles import (
    QUERY, ExistingRetrieverFixture, adapter_fixture, example_bank_fixture,
    prompt_configuration, settings_fixture,
)


class FewShotIntegrationTests(unittest.TestCase):
    def resources(self, key="qwen3", **retrieval_options):
        bank = example_bank_fixture()
        authorized = {example.document_id for example in bank if example.identifier != "unauthorized"}
        settings = settings_fixture(authorized, **retrieval_options)
        fixture = ExistingRetrieverFixture(bank)
        adapter, runtime, loader = adapter_fixture(key)
        engine = TermExtractor(
            examples=bank, retrieval_settings=settings, parser=fixture.parser,
            adapters={key: adapter}, evaluation_example_ids={"heldout-id"},
            evaluation_document_ids={"heldout-doc"},
        )
        return engine, fixture, bank, runtime, loader

    def extract(self, engine, **kwargs):
        return engine.extract(
            QUERY, prompt_configuration(), condition="few_shot",
            query_id="query-id", query_document_id="query-doc", **kwargs,
        )

    def test_exactly_three_examples_are_selected_by_the_existing_retriever(self):
        for key in ("qwen3", "gemma2"):
            with self.subTest(model=key):
                engine, fixture, bank, runtime, _ = self.resources(key, top_k=10)
                with patch("extractor.pipeline.build_retriever", wraps=fixture) as factory, patch(
                    "syntactic_retrieval.fastkassim.FastKassimSimilarity.score",
                    side_effect=AssertionError("No actual LTK execution"),
                ), patch(
                    "syntactic_retrieval.parsers.StanfordPCFGParser.parse",
                    side_effect=AssertionError("No actual parser"),
                ):
                    result = self.extract(engine, model=key)
                self.assertEqual([item.example.identifier for item in result.retrieved_examples], ["b", "d", "c"])
                self.assertEqual([item.score for item in result.retrieved_examples], [0.9, 0.9, 0.8])
                self.assertEqual(len(runtime.model.calls), 1)
                self.assertIsInstance(fixture.created[0], SyntacticRetriever)
                self.assertEqual(fixture.created[0].config.top_k, 10)
                call = fixture.created[0].retrieve.call_args
                self.assertEqual(call.kwargs["top_k"], 3)
                self.assertEqual(call.kwargs["query_id"], "query-id")
                self.assertEqual(call.kwargs["query_document_id"], "query-doc")
                factory.assert_called_once_with(bank, engine._retrieval_settings, parser=fixture.parser)

    def test_evaluation_documents_ids_and_self_text_are_filtered_before_analysis(self):
        engine, fixture, bank, _, _ = self.resources()
        with patch("extractor.pipeline.build_retriever", wraps=fixture):
            result = self.extract(engine)
        forbidden = {"unauthorized", "query-id", "same-text", "heldout-doc-example", "heldout-id"}
        self.assertFalse(forbidden.intersection(item.example.identifier for item in result.retrieved_examples))
        allowed_texts = {example.text for example in bank if example.identifier in {"a", "b", "c", "d"}}
        self.assertEqual(set(fixture.parser.calls), {QUERY} | allowed_texts)
        self.assertEqual(len(fixture.similarity.calls), 4)
        call = fixture.created[0].retrieve.call_args
        self.assertEqual(call.kwargs["excluded_example_ids"], frozenset({"heldout-id"}))
        self.assertEqual(call.kwargs["excluded_document_ids"], frozenset({"heldout-doc"}))
        self.assertTrue(all(not hasattr(parsed, "terms") for pair in fixture.similarity.calls for parsed in pair))

    def test_frozen_prompt_and_original_annotations_are_preserved(self):
        engine, fixture, bank, _, _ = self.resources()
        before = tuple((example.text, example.terms) for example in bank)
        with patch("extractor.pipeline.build_retriever", wraps=fixture):
            result = self.extract(engine)
        expected = PromptBuilder(prompt_configuration()).build_few_shot(QUERY, result.retrieved_examples)
        self.assertEqual(result.prompt.to_messages(), expected.to_messages())
        self.assertEqual(before, tuple((example.text, example.terms) for example in bank))
        self.assertTrue(result.prompt.metadata.complete_few_shot)
        self.assertIs(result.retrieved_examples[0].example, bank[1])
        self.assertNotIn('"score"', result.prompt.messages[1].content)
        self.assertEqual(result.to_dict()["prompt_metadata"]["retrieved_examples"][0]["score"], 0.9)

    def test_query_exclusions_are_unioned_with_persistent_evaluation_exclusions(self):
        engine, fixture, _, runtime, loader = self.resources()
        with patch("extractor.pipeline.build_retriever", wraps=fixture):
            with self.assertRaises(InsufficientExamplesError) as caught:
                self.extract(engine, excluded_example_ids={"b"}, excluded_document_ids={"doc-d"})
        self.assertEqual((caught.exception.required, caught.exception.available), (3, 2))
        call = fixture.created[0].retrieve.call_args
        self.assertEqual(call.kwargs["excluded_example_ids"], frozenset({"heldout-id", "b"}))
        self.assertEqual(call.kwargs["excluded_document_ids"], frozenset({"heldout-doc", "doc-d"}))
        self.assertEqual(runtime.model.calls, [])
        loader.assert_not_called()

    def test_empty_authorization_never_analyzes_examples_or_executes_the_adapter(self):
        bank = example_bank_fixture()
        fixture = ExistingRetrieverFixture(bank)
        adapter, _, loader = adapter_fixture()
        engine = TermExtractor(
            examples=bank, retrieval_settings=settings_fixture(set()), parser=fixture.parser,
            adapters={"qwen3": adapter},
        )
        with patch("extractor.pipeline.build_retriever", wraps=fixture):
            with self.assertRaises(InsufficientExamplesError) as caught:
                self.extract(engine)
        self.assertEqual((caught.exception.required, caught.exception.available), (3, 0))
        self.assertEqual(fixture.parser.calls, [])
        self.assertEqual(fixture.similarity.calls, [])
        loader.assert_not_called()

    def test_fewer_than_three_authorized_examples_fail_before_generation(self):
        engine, fixture, _, _, loader = self.resources()
        with patch("extractor.pipeline.build_retriever", wraps=fixture):
            with self.assertRaises(InsufficientExamplesError):
                self.extract(engine, excluded_document_ids={"doc-a", "doc-b", "doc-c"})
        loader.assert_not_called()

    def test_few_shot_requires_a_document_id_and_the_existing_three_example_policy(self):
        engine, fixture, _, _, loader = self.resources()
        with patch("extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")):
            with self.assertRaises(InvalidExtractionRequest):
                engine.extract(QUERY, prompt_configuration(), condition="few_shot")
            with self.assertRaises(InvalidExtractionRequest):
                engine.extract(QUERY, replace(prompt_configuration(), few_shot_examples=2),
                               condition="few_shot", query_document_id="query-doc")
        self.assertEqual(fixture.created, [])
        loader.assert_not_called()

    def test_missing_retrieval_configuration_has_an_explicit_error(self):
        with self.assertRaises(MissingRetrievalResourcesError):
            self.extract(TermExtractor(examples=example_bank_fixture()))

    def test_weakened_existing_exclusion_flags_are_rejected_without_rewriting_settings(self):
        for option in ("exclude_same_text", "exclude_query_document"):
            engine, fixture, _, _, loader = self.resources(**{option: False})
            with self.subTest(option=option), patch("extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")):
                with self.assertRaises(InvalidExtractionRequest):
                    self.extract(engine)
            self.assertFalse(getattr(engine._retrieval_settings.retrieval, option))
            self.assertEqual(fixture.created, [])
            loader.assert_not_called()

    def test_retriever_is_created_only_for_few_shot_and_reused_by_that_engine(self):
        engine, fixture, _, _, _ = self.resources()
        with patch("extractor.pipeline.build_retriever", wraps=fixture) as factory:
            engine.extract(QUERY, prompt_configuration(), condition="basic")
            factory.assert_not_called()
            self.extract(engine)
            self.extract(engine)
            self.assertEqual(factory.call_count, 1)
        self.assertEqual(fixture.created[0].retrieve.call_count, 2)

    def test_parser_and_similarity_failures_propagate_without_loading_the_llm(self):
        engine, fixture, _, _, loader = self.resources()
        failure = RuntimeError("Synthetic parsing failure")
        with patch("extractor.pipeline.build_retriever", wraps=fixture), patch.object(
            fixture.parser, "parse", side_effect=failure
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.extract(engine)
        self.assertIs(caught.exception, failure)
        loader.assert_not_called()
        engine, fixture, _, _, loader = self.resources()
        failure = RuntimeError("Synthetic similarity failure")
        with patch("extractor.pipeline.build_retriever", wraps=fixture), patch.object(
            fixture.similarity, "score", side_effect=failure
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.extract(engine)
        self.assertIs(caught.exception, failure)
        loader.assert_not_called()

    def test_external_banks_are_interchangeable_without_generating_or_modifying_annotations(self):
        bank = example_bank_fixture()
        second = tuple(replace(example, terms=("another supplied annotation",)) for example in bank)
        outputs = []
        for supplied in (bank, second):
            fixture = ExistingRetrieverFixture(supplied)
            adapter, _, _ = adapter_fixture()
            authorized = {f"doc-{key}" for key in ("a", "b", "c", "d")}
            engine = TermExtractor(examples=supplied, retrieval_settings=settings_fixture(authorized),
                                   parser=fixture.parser, adapters={"qwen3": adapter})
            with patch("extractor.pipeline.build_retriever", wraps=fixture):
                outputs.append(self.extract(engine))
        self.assertEqual(outputs[0].retrieved_examples[0].example.terms, bank[1].terms)
        self.assertEqual(outputs[1].retrieved_examples[0].example.terms, second[1].terms)
        self.assertNotEqual(bank[1].terms, second[1].terms)


if __name__ == "__main__":
    unittest.main()
