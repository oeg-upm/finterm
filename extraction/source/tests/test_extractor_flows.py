"""Single-text flows through the approved prompts/adapters with execution doubles."""

from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

from extractor import (
    ExtractionCondition, ExtractorConfig, InterpretationStatus, InvalidExtractionRequest,
    TermExtractor, UnsupportedModelConditionError,
)
from llm_adapters import (
    AdapterLoadError, AdapterProfile, GenerationOptions, LoadConfig, MODEL_IDS,
)
from prompts import PromptBuilder, build_dister_prompt

from extractor_doubles import QUERY, adapter_fixture, prompt_configuration


class DirectFlowTests(unittest.TestCase):
    def setUp(self):
        self.config = prompt_configuration()

    def test_basic_coordinates_existing_prompt_adapter_and_structured_result(self):
        adapter, runtime, loader = adapter_fixture()
        engine = TermExtractor(adapters={"qwen3": adapter})
        with patch("extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")):
            result = engine.extract(QUERY, self.config)
        self.assertEqual(result.prompt.to_messages(), PromptBuilder(self.config).build_basic(QUERY).to_messages())
        self.assertEqual(result.terms, ("fixture-A", "fixture-A"))
        self.assertIs(result.json_response, result.generation.json_response)
        self.assertEqual(result.interpretation, InterpretationStatus.JSON_TERMS)
        self.assertEqual(result.raw_response, runtime.tokenizer.response + "<fixture_eos>")
        self.assertEqual(result.retrieved_examples, ())
        self.assertEqual(len(runtime.model.calls), 1)
        loader.assert_called_once()

    def test_patterns_preserve_the_existing_prompt_literally(self):
        adapter, runtime, _ = adapter_fixture("gemma2")
        engine = TermExtractor(adapters={"gemma2": adapter})
        with patch("extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")):
            result = engine.extract(QUERY, self.config, model="gemma2", condition="patterns")
        expected = PromptBuilder(self.config).build_patterns(QUERY)
        self.assertEqual(result.prompt.to_messages(), expected.to_messages())
        self.assertEqual(runtime.tokenizer.template_calls[0][0][0]["content"],
                         expected.messages[0].content + "\n\n" + expected.messages[1].content)

    def test_hybrid_receives_external_candidates_without_detecting_or_rewriting_them(self):
        candidates = ['candidate "one"', "candidate\ntwo", "candidate \"one\"", "duplicate", "duplicate"]
        adapter, _, _ = adapter_fixture()
        engine = TermExtractor(adapters={"qwen3": adapter})
        with patch("extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")):
            result = engine.extract(QUERY, self.config, condition="hybrid", candidates=candidates)
        expected = PromptBuilder(self.config).build_hybrid(QUERY, candidates)
        self.assertEqual(result.prompt.to_messages(), expected.to_messages())
        self.assertEqual(result.request.candidates, tuple(candidates))
        self.assertEqual(result.retrieved_examples, ())

    def test_empty_candidates_still_reach_the_adapter_instead_of_fabricating_an_empty_result(self):
        adapter, runtime, _ = adapter_fixture()
        result = TermExtractor(adapters={"qwen3": adapter}).extract(
            QUERY, self.config, condition="hybrid", candidates=[]
        )
        self.assertIn("### Candidates\n[]", result.prompt.messages[1].content)
        self.assertEqual(len(runtime.model.calls), 1)
        self.assertEqual(result.terms, ("fixture-A", "fixture-A"))

    def test_model_keys_and_official_ids_select_the_correct_existing_adapter(self):
        adapters = {key: adapter_fixture(key)[0] for key in MODEL_IDS}
        engine = TermExtractor(adapters=adapters)
        for key in ("qwen3", "gemma2"):
            for selector in (key, MODEL_IDS[key]):
                with self.subTest(model=selector):
                    result = engine.extract(QUERY, self.config, model=selector, condition="basic")
                    self.assertEqual(result.generation.model_id, MODEL_IDS[key])
                    self.assertEqual(result.request.model, key)

    def test_native_dister_is_separate_and_has_no_fastkassim_demonstrations(self):
        response = " Native protocol fixture\n preserve whitespace  "
        adapter, runtime, _ = adapter_fixture("dister", response)
        engine = TermExtractor(adapters={"dister": adapter})
        with patch("extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")), patch.object(
            PromptBuilder, "build_basic", side_effect=AssertionError("No generalist prompt")
        ):
            result = engine.extract(QUERY, self.config, model="dister", condition="native_zero_shot", query_id='id "one"')
        self.assertEqual(result.prompt.to_json(), build_dister_prompt('id "one"', QUERY, self.config.domain.name).to_json())
        self.assertEqual(result.raw_response, response + "<fixture_eos>")
        self.assertEqual(result.generation.response_text, response)
        self.assertIsNone(result.terms)
        self.assertIsNone(result.json_response)
        self.assertEqual(result.interpretation, InterpretationStatus.NATIVE_PENDING)
        self.assertEqual(result.retrieved_examples, ())
        self.assertEqual(runtime.tokenizer.template_calls, [])
        self.assertEqual(result.to_dict()["interpretation"], "native_terms_pending_validation")

    def test_domain_is_arbitrary_and_not_selected_from_a_dataset(self):
        adapter, _, _ = adapter_fixture()
        result = TermExtractor(adapters={"qwen3": adapter}).extract(QUERY, self.config)
        self.assertEqual(result.request.domain.key, "materials_fixture")
        self.assertIn("materials science", result.prompt.messages[0].content)
        self.assertIn(self.config.domain.description, result.prompt.messages[1].content)

    def test_result_serializes_existing_generation_and_format_errors(self):
        adapter, _, _ = adapter_fixture(response="Protocol fixture: invalid JSON")
        result = TermExtractor(adapters={"qwen3": adapter}).extract(QUERY, self.config)
        self.assertEqual(result.interpretation, InterpretationStatus.FORMAT_ERROR)
        self.assertIsNone(result.terms)
        self.assertEqual(result.format_errors, result.json_response.errors)
        data = result.to_dict()
        self.assertFalse(data["generation"]["json_response"]["json_valid"])
        self.assertEqual(data["terms"], None)
        self.assertEqual(json.loads(json.dumps(data))["condition"], "basic")

    def test_valid_empty_terms_remain_distinct_from_a_format_error(self):
        adapter, _, _ = adapter_fixture(response='{"terms": []}')
        result = TermExtractor(adapters={"qwen3": adapter}).extract(QUERY, self.config)
        self.assertEqual(result.terms, ())
        self.assertEqual(result.to_dict()["terms"], [])
        self.assertEqual(result.interpretation, InterpretationStatus.JSON_TERMS)


class RoutingAndFailureTests(unittest.TestCase):
    def setUp(self):
        self.config = prompt_configuration()

    def test_default_adapter_factory_is_lazy_and_reused(self):
        adapter, _, loader = adapter_fixture()
        with patch("extractor.pipeline.Qwen3Adapter", return_value=adapter) as factory:
            engine = TermExtractor()
            factory.assert_not_called()
            engine.extract(QUERY, self.config)
            engine.extract(QUERY, self.config)
            factory.assert_called_once_with()
            loader.assert_called_once()

    def test_existing_adapter_profiles_are_used_without_copying_loading_logic(self):
        adapter, _, _ = adapter_fixture("gemma2")
        profile = AdapterProfile("gemma2", MODEL_IDS["gemma2"], LoadConfig(), GenerationOptions())
        with patch("extractor.pipeline.adapter_from_profile", return_value=adapter) as factory:
            engine = TermExtractor(adapter_profiles={"gemma2": profile})
            factory.assert_not_called()
            engine.extract(QUERY, self.config, model="gemma2")
            factory.assert_called_once_with(profile)

    def test_unsupported_model_condition_pairs_fail_without_any_execution(self):
        invalid = [("dister", condition) for condition in ("basic", "patterns", "few_shot", "hybrid")]
        invalid += [("qwen3", "native_zero_shot"), ("gemma2", "native_zero_shot")]
        engine = TermExtractor()
        with patch.object(engine, "_adapter", side_effect=AssertionError("No adapter")), patch(
            "extractor.pipeline.build_retriever", side_effect=AssertionError("No retrieval")
        ):
            for model, condition in invalid:
                with self.subTest(model=model, condition=condition), self.assertRaises(UnsupportedModelConditionError):
                    engine.extract(QUERY, self.config, model=model, condition=condition)

    def test_unknown_model_or_condition_is_not_substituted(self):
        engine = TermExtractor()
        for options in ({"model": "unknown"}, {"condition": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                engine.extract(QUERY, self.config, **options)

    def test_dister_does_not_silently_map_basic_to_native(self):
        with self.assertRaises(UnsupportedModelConditionError):
            TermExtractor().extract(QUERY, self.config, model="dister", query_id="id")

    def test_native_requires_the_callers_identifier(self):
        with self.assertRaises(InvalidExtractionRequest):
            TermExtractor().extract(QUERY, self.config, model="dister", condition="native_zero_shot")

    def test_hybrid_requires_candidates_and_other_conditions_reject_them(self):
        engine = TermExtractor()
        with self.assertRaises(InvalidExtractionRequest):
            engine.extract(QUERY, self.config, condition="hybrid")
        with self.assertRaises(InvalidExtractionRequest):
            engine.extract(QUERY, self.config, condition="basic", candidates=["candidate"])
        with self.assertRaises(TypeError):
            engine.extract(QUERY, self.config, condition="hybrid", candidates="not a list")

    def test_reference_terms_are_not_an_input_to_extraction(self):
        with self.assertRaises(TypeError):
            TermExtractor().extract(QUERY, self.config, reference_terms=["hidden answer"])

    def test_invalid_text_and_prompt_configuration_are_rejected(self):
        for text, config in ((" ", self.config), (QUERY, "not a PromptConfig")):
            with self.subTest(text=text), self.assertRaises((TypeError, ValueError)):
                TermExtractor().extract(text, config)

    def test_nondeterministic_adapter_or_profile_is_rejected_without_mutation(self):
        adapter, _, _ = adapter_fixture()
        options = GenerationOptions(do_sample=True)
        adapter.generation_options = options
        with self.assertRaises(ValueError):
            TermExtractor(adapters={"qwen3": adapter})
        self.assertIs(adapter.generation_options, options)
        profile = AdapterProfile("qwen3", MODEL_IDS["qwen3"], LoadConfig(), options)
        with self.assertRaises(ValueError):
            TermExtractor(adapter_profiles={"qwen3": profile})

    def test_adapter_model_mismatch_and_ambiguous_configuration_are_rejected(self):
        adapter, _, _ = adapter_fixture("gemma2")
        with self.assertRaises(ValueError):
            TermExtractor(adapters={"qwen3": adapter})
        adapter, _, _ = adapter_fixture()
        profile = AdapterProfile("qwen3", MODEL_IDS["qwen3"], LoadConfig(), GenerationOptions())
        with self.assertRaises(ValueError):
            TermExtractor(adapters={"qwen3": adapter}, adapter_profiles={"qwen3": profile})

    def test_load_error_propagates_without_returning_terms(self):
        adapter, _, loader = adapter_fixture()
        failure = AdapterLoadError("Synthetic loading failure")
        loader.side_effect = failure
        engine = TermExtractor(adapters={"qwen3": adapter})
        with self.assertRaises(AdapterLoadError) as caught:
            engine.extract(QUERY, self.config)
        self.assertIs(caught.exception, failure)

    def test_generation_error_propagates_without_a_fallback(self):
        adapter, runtime, _ = adapter_fixture()
        failure = RuntimeError("Synthetic model protocol failure")
        runtime.model.generate = Mock(side_effect=failure)
        with self.assertRaises(RuntimeError) as caught:
            TermExtractor(adapters={"qwen3": adapter}).extract(QUERY, self.config)
        self.assertIs(caught.exception, failure)

    def test_unexpected_adapter_result_is_not_repaired(self):
        adapter, _, _ = adapter_fixture()
        adapter.generate = Mock(return_value={"terms": ["not a GenerationResult"]})
        with self.assertRaises(TypeError):
            TermExtractor(adapters={"qwen3": adapter}).extract(QUERY, self.config)


if __name__ == "__main__":
    unittest.main()
