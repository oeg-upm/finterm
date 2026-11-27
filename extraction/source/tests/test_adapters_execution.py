"""HF call contracts verified with protocol doubles, never actual LLMs."""

from copy import deepcopy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from adapter_doubles import TensorDouble, TokenizerDouble, runtime_double
from llm_adapters import (
    AdapterExecutionError, AdapterLoadError, DiSTERLlamaAdapter, Gemma2Adapter,
    GenerationOptions, LoadConfig, PreparedInput, Qwen3Adapter,
)
from llm_adapters.runtime import load_huggingface_runtime
from prompts import (
    AnnotationPolicy, DomainSpec, PromptBuilder, PromptConfig,
    build_dister_prompt, default_prompt_config,
)
from syntactic_retrieval.models import AnnotatedExample, RetrievedExample


def generalist_prompt():
    return PromptBuilder(default_prompt_config("finance")).build_basic('Synthetic "input" {x}.\nSecond line.')


class InputPreparationTests(unittest.TestCase):
    def test_qwen_uses_frozen_messages_and_the_official_non_thinking_switch(self):
        prompt = generalist_prompt()
        original = prompt.to_messages()
        loader = Mock(side_effect=AssertionError("No loading during preparation"))
        adapter = Qwen3Adapter(runtime_loader=loader)
        prepared = adapter.prepare_input(prompt)
        tokenizer = TokenizerDouble()
        prepared.tokenize(tokenizer)
        self.assertEqual(tokenizer.template_calls, [(original, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })])
        self.assertFalse(tokenizer.tokenization_calls[0][1]["add_special_tokens"])
        self.assertFalse(tokenizer.tokenization_calls[0][1]["truncation"])
        self.assertEqual(prompt.to_messages(), original)
        self.assertFalse(adapter.is_loaded)
        loader.assert_not_called()

    def test_gemma_merges_system_and_user_without_rewriting_either(self):
        prompt = generalist_prompt()
        original = prompt.to_messages()
        tokenizer = TokenizerDouble()
        prepared = Gemma2Adapter().prepare_input(prompt)
        prepared.tokenize(tokenizer)
        messages, options = tokenizer.template_calls[0]
        self.assertEqual(messages, [{"role": "user", "content": original[0]["content"] + "\n\n" + original[1]["content"]}])
        self.assertEqual(options, {"tokenize": False, "add_generation_prompt": True})
        self.assertFalse(tokenizer.tokenization_calls[0][1]["add_special_tokens"])
        self.assertEqual(prompt.to_messages(), original)

    def test_dister_preserves_the_native_json_and_never_applies_a_chat_template(self):
        native = build_dister_prompt('id "quoted"', 'Synthetic {text}\n€ "q"', "custom-domain")
        tokenizer = TokenizerDouble()
        prepared = DiSTERLlamaAdapter().prepare_input(native)
        rendered, _ = prepared.tokenize(tokenizer)
        self.assertEqual(rendered, native.to_json())
        self.assertEqual(tokenizer.template_calls, [])
        self.assertTrue(tokenizer.tokenization_calls[0][1]["add_special_tokens"])

    def test_all_generalist_conditions_are_compatible_with_qwen_and_gemma(self):
        builder = PromptBuilder(default_prompt_config("heart_failure"))
        examples = [RetrievedExample(AnnotatedExample(str(i), f"Synthetic {i}", (), "demo", "test"), 0.5) for i in range(3)]
        prompts = [builder.build_basic("Synthetic."), builder.build_patterns("Synthetic."),
                   builder.build_few_shot("Synthetic.", examples), builder.build_hybrid("Synthetic.", ["candidate"])]
        for adapter in (Qwen3Adapter(), Gemma2Adapter()):
            for prompt in prompts:
                with self.subTest(adapter=adapter.key, condition=prompt.metadata.condition):
                    prepared = adapter.prepare_input(prompt)
                    self.assertIn(prompt.messages[1].content, prepared.messages[-1].content)
                    self.assertNotIn("score", prepared.messages[-1].content)
                    self.assertFalse(adapter.is_loaded)

    def test_any_configurable_domain_is_accepted_without_a_corpus(self):
        domain = DomainSpec("arbitrary", "materials science", "An arbitrary specialised English domain.")
        prompt = PromptBuilder(PromptConfig(domain, AnnotationPolicy())).build_basic("A synthetic technical input.")
        for adapter in (Qwen3Adapter(), Gemma2Adapter()):
            self.assertIn("materials science", adapter.prepare_input(prompt).messages[0].content)

    def test_wrong_prompt_types_fail_before_loading(self):
        native = build_dister_prompt("id", "Synthetic.", "test")
        for adapter, prompt in [(Qwen3Adapter(), native), (Gemma2Adapter(), native),
                                (DiSTERLlamaAdapter(), generalist_prompt())]:
            with self.subTest(adapter=adapter.key), patch.object(adapter, "load", side_effect=AssertionError("No load")):
                with self.assertRaises(TypeError):
                    adapter.generate(prompt)

    def test_invalid_prepared_formats_are_rejected(self):
        for kwargs in ({"kind": "unknown"}, {"kind": "dister"}, {"kind": "qwen3"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PreparedInput(**kwargs)


class GenerationContractTests(unittest.TestCase):
    def adapter(self, cls=Qwen3Adapter, response=' {"terms": ["fixture-A", "fixture-A"]}\n', **kwargs):
        runtime = runtime_double(response)
        loader = Mock(return_value=runtime)
        return cls(runtime_loader=loader, **kwargs), runtime, loader

    def test_constructor_and_pure_preparation_are_lazy(self):
        adapter, _, loader = self.adapter()
        adapter.prepare_input(generalist_prompt())
        loader.assert_not_called()
        self.assertFalse(adapter.is_loaded)

    def test_explicit_loading_occurs_once_and_uses_the_exact_model(self):
        adapter, runtime, loader = self.adapter()
        self.assertIs(adapter.load(), runtime)
        self.assertIs(adapter.load(), runtime)
        self.assertTrue(adapter.is_loaded)
        loader.assert_called_once_with("Qwen/Qwen3-14B", adapter.load_config)

    def test_defaults_are_deterministic_and_do_not_mutate_repo_generation_settings(self):
        adapter, runtime, _ = self.adapter()
        before = deepcopy(runtime.model.generation_config)
        result = adapter.generate(generalist_prompt())
        used = runtime.model.calls[0]["generation_config"]
        self.assertFalse(used.do_sample)
        self.assertEqual(used.num_beams, 1)
        self.assertEqual(used.num_return_sequences, 1)
        self.assertEqual(used.temperature, 1.0)
        self.assertEqual(used.top_p, 1.0)
        self.assertIsNone(used.min_p)
        self.assertEqual(used.eos_token_id, [20, 21])
        self.assertEqual(used.pad_token_id, 20)
        self.assertFalse(used.return_dict_in_generate)
        self.assertEqual(runtime.model.generation_config, before)
        self.assertEqual(runtime.torch.inference_entries, 1)
        self.assertTrue(result.json_response.schema_valid)

    def test_only_new_token_ids_are_decoded_and_raw_response_is_preserved(self):
        adapter, runtime, _ = self.adapter()
        result = adapter.generate(generalist_prompt())
        self.assertEqual(result.generated_token_ids, (31, 32, 21))
        self.assertEqual(result.input_token_count, 3)
        self.assertEqual(result.raw_response, runtime.tokenizer.response + "<fixture_eos>")
        self.assertEqual(result.response_text, runtime.tokenizer.response)
        self.assertEqual(result.terms, ("fixture-A", "fixture-A"))
        self.assertEqual([ids for ids, _ in runtime.tokenizer.decode_calls], [[31, 32, 21], [31, 32, 21]])
        self.assertTrue(all(not options["clean_up_tokenization_spaces"] for _, options in runtime.tokenizer.decode_calls))
        self.assertEqual(runtime.tokenizer.last_encoding.devices, ["cuda:0"])

    def test_all_three_adapters_have_a_functional_execution_path_with_doubles(self):
        for cls in (Qwen3Adapter, Gemma2Adapter, DiSTERLlamaAdapter):
            adapter, runtime, loader = self.adapter(cls)
            prompt = build_dister_prompt("id", "Synthetic.", "test") if cls is DiSTERLlamaAdapter else generalist_prompt()
            result = adapter.generate(prompt)
            self.assertEqual(result.model_id, adapter.model_id)
            self.assertEqual(len(runtime.model.calls), 1)
            loader.assert_called_once()

    def test_dister_never_interprets_or_normalizes_its_native_output(self):
        response = ' Native fixture: {not JSON}\n  preserve spaces '
        adapter, runtime, _ = self.adapter(DiSTERLlamaAdapter, response=response)
        result = adapter.generate(build_dister_prompt("id", "Synthetic.", "test"))
        self.assertEqual(result.response_text, response)
        self.assertEqual(result.raw_response, response + "<fixture_eos>")
        self.assertIsNone(result.json_response)
        self.assertIsNone(result.terms)
        self.assertEqual(runtime.model.calls[0]["generation_config"].max_new_tokens, 128)

    def test_dister_does_not_assume_json_even_when_the_output_looks_like_it(self):
        adapter, _, _ = self.adapter(DiSTERLlamaAdapter, response='{"terms": ["fixture"]}')
        result = adapter.generate(build_dister_prompt("id", "Synthetic.", "test"))
        self.assertIsNone(result.json_response)
        self.assertIsNone(result.terms)

    def test_invalid_generalist_json_is_reported_without_synthesizing_terms(self):
        response = '```json\n{"terms": ["fixture"]}\n```'
        adapter, _, _ = self.adapter(response=response)
        result = adapter.generate(generalist_prompt())
        self.assertEqual(result.response_text, response)
        self.assertFalse(result.json_response.json_valid)
        self.assertIsNone(result.terms)

    def test_context_overflow_fails_without_truncation_or_generation(self):
        adapter, runtime, _ = self.adapter(load=LoadConfig(context_limit=4))
        with self.assertRaises(AdapterExecutionError):
            adapter.generate(generalist_prompt())
        self.assertEqual(runtime.model.calls, [])
        self.assertFalse(runtime.tokenizer.tokenization_calls[0][1]["truncation"])

    def test_accelerate_embedding_execution_device_is_used_when_present(self):
        adapter, runtime, _ = self.adapter()
        runtime.model.embeddings.weight.device = "meta"
        runtime.model.embeddings._hf_hook = SimpleNamespace(execution_device="cuda:1")
        adapter.generate(generalist_prompt())
        self.assertEqual(runtime.tokenizer.last_encoding.devices, ["cuda:1"])

    def test_meta_device_without_execution_hook_fails_instead_of_cpu_fallback(self):
        adapter, runtime, _ = self.adapter()
        runtime.model.embeddings.weight.device = "meta"
        with self.assertRaises(AdapterExecutionError):
            adapter.generate(generalist_prompt())
        self.assertEqual(runtime.model.calls, [])

    def test_encoder_decoder_and_unexpected_output_are_rejected(self):
        adapter, runtime, _ = self.adapter()
        runtime.model.config.is_encoder_decoder = True
        with self.assertRaises(AdapterExecutionError):
            adapter.generate(generalist_prompt())
        adapter, runtime, _ = self.adapter()
        runtime.model.output = TensorDouble([[11]])
        with self.assertRaises(AdapterExecutionError):
            adapter.generate(generalist_prompt())

    def test_explicit_sampling_options_are_supported_without_affecting_defaults(self):
        options = GenerationOptions(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        adapter, runtime, _ = self.adapter(generation=options)
        adapter.generate(generalist_prompt())
        used = runtime.model.calls[0]["generation_config"]
        self.assertTrue(used.do_sample)
        self.assertEqual((used.temperature, used.top_p, used.top_k), (0.7, 0.8, 20))

    def test_loader_failure_does_not_create_a_loaded_or_fake_model(self):
        loader = Mock(side_effect=AdapterLoadError("Synthetic loader failure"))
        adapter = Qwen3Adapter(runtime_loader=loader)
        with self.assertRaises(AdapterLoadError):
            adapter.generate(generalist_prompt())
        self.assertFalse(adapter.is_loaded)
        loader.assert_called_once()

    def test_result_serialization_distinguishes_raw_text_json_and_terms(self):
        adapter, _, _ = self.adapter()
        data = adapter.generate(generalist_prompt()).to_dict()
        self.assertIn("<fixture_eos>", data["raw_response"])
        self.assertNotIn("<fixture_eos>", data["response_text"])
        self.assertTrue(data["json_response"]["json_valid"])
        self.assertEqual(data["json_response"]["terms"], ["fixture-A", "fixture-A"])


class HuggingFaceLoadingBoundaryTests(unittest.TestCase):
    def fake_modules(self):
        runtime = runtime_double()
        torch = ModuleType("torch")
        torch.bfloat16 = "BFLOAT16_TEST_DOUBLE"
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = runtime.tokenizer
        transformers.AutoModelForCausalLM = Mock()
        transformers.AutoModelForCausalLM.from_pretrained.return_value = runtime.model
        return runtime, torch, transformers

    def test_real_loader_code_calls_hf_factories_only_with_mocked_modules(self):
        runtime, torch, transformers = self.fake_modules()
        config = LoadConfig(source="local-model-fixture", revision="revision-fixture")
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            loaded = load_huggingface_runtime("Qwen/Qwen3-14B", config)
        kwargs = {
            "revision": "revision-fixture", "local_files_only": True,
            "cache_dir": None, "trust_remote_code": False,
        }
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with("local-model-fixture", **kwargs)
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "local-model-fixture", **kwargs, device_map="cuda", torch_dtype="BFLOAT16_TEST_DOUBLE",
            attn_implementation="eager", use_safetensors=True,
        )
        self.assertIs(loaded.model, runtime.model)
        self.assertEqual(runtime.model.eval_calls, 1)

    def test_missing_local_resources_raise_without_any_fallback(self):
        _, torch, transformers = self.fake_modules()
        transformers.AutoTokenizer.from_pretrained.side_effect = OSError("No local fixture")
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            with self.assertRaises(AdapterLoadError):
                load_huggingface_runtime("Qwen/Qwen3-14B", LoadConfig())
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
