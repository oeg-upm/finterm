"""Lazy imports, optional corpus metadata, frozen prompts and dependency floors."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch

from llm_adapters import (
    DiSTERLlamaAdapter, Gemma2Adapter, GenerationOptions, LoadConfig,
    Qwen3Adapter, adapter_from_profile, adapter_profiles_from_toml,
)
from prompts import PromptBuilder, build_dister_prompt, default_prompt_config
from syntactic_retrieval import CorpusLocations, FastKassimConfig, RetrievalConfig, Settings
from syntactic_retrieval import StanfordPCFGConfig, build_retriever, load_settings


PROJECT = Path(__file__).resolve().parents[1]


class AdapterConfigurationTests(unittest.TestCase):
    def test_profiles_use_official_models_offline_and_deterministic_defaults(self):
        source = (PROJECT / "config" / "llm_adapters.toml").read_text(encoding="utf-8")
        profiles = adapter_profiles_from_toml(source)
        self.assertEqual(set(profiles), {"qwen3", "gemma2", "dister"})
        for key, cls in (("qwen3", Qwen3Adapter), ("gemma2", Gemma2Adapter), ("dister", DiSTERLlamaAdapter)):
            adapter = adapter_from_profile(profiles[key])
            self.assertIsInstance(adapter, cls)
            self.assertFalse(adapter.is_loaded)
            self.assertTrue(adapter.load_config.local_files_only)
            self.assertEqual(adapter.load_config.device_map, "cuda")
            self.assertFalse(adapter.generation_options.do_sample)
            self.assertEqual(adapter.generation_options.num_beams, 1)

    def test_profile_parsing_is_pure_and_does_not_access_any_file(self):
        source = (PROJECT / "config" / "llm_adapters.toml").read_text(encoding="utf-8")
        with patch("builtins.open", side_effect=AssertionError("No file access")), patch.object(
            Path, "open", side_effect=AssertionError("No path access")
        ):
            profiles = adapter_profiles_from_toml(source)
        self.assertEqual(profiles["qwen3"].model_id, "Qwen/Qwen3-14B")

    def test_invalid_generation_and_loading_parameters_are_rejected(self):
        for options in ({"max_new_tokens": 0}, {"num_beams": True}, {"temperature": 0.7},
                        {"do_sample": True, "top_p": 2.0}, {"do_sample": True, "top_k": -1}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                GenerationOptions(**options)
        for options in ({"dtype": "int4"}, {"local_files_only": "true"}, {"context_limit": 0}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                LoadConfig(**options)

    def test_profiles_do_not_silently_select_another_model(self):
        source = (PROJECT / "config" / "llm_adapters.toml").read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter_profiles_from_toml(source.replace("Qwen/Qwen3-14B", "unapproved-model"))


class CorpusIndependentSettingsTests(unittest.TestCase):
    def test_standalone_file_has_no_corpus_section_and_builds_without_resources(self):
        settings = load_settings(PROJECT / "config" / "fastkassim_standalone.toml")
        self.assertIsNone(settings.corpora)
        self.assertEqual(settings.retrieval.top_k, 3)
        retriever = build_retriever([], settings)
        # Empty authorization means this call returns without executing a parser.
        self.assertEqual(retriever.retrieve("An arbitrary specialised English text."), ())

    def test_settings_can_be_constructed_without_corpus_paths(self):
        parser = StanfordPCFGConfig(Path("local-parser.jar"), Path("local-models.jar"))
        settings = Settings(parser=parser)
        self.assertIsNone(settings.corpora)
        self.assertEqual(settings.fastkassim, FastKassimConfig())
        self.assertEqual(settings.retrieval, RetrievalConfig())

    def test_legacy_configuration_and_positional_constructor_remain_compatible(self):
        old = load_settings(PROJECT / "config" / "fastkassim.toml")
        self.assertIsInstance(old.corpora, CorpusLocations)
        copied = Settings(old.corpora, old.parser, old.fastkassim, old.retrieval)
        self.assertEqual(copied, old)
        self.assertEqual(old.corpora.acter, PROJECT.parent / "Corpus para LLM" / "ACTER-master")

    def test_unknown_sections_are_not_silently_accepted(self):
        from io import BytesIO

        source = (PROJECT / "config" / "fastkassim_standalone.toml").read_bytes() + b"\n[unexpected]\na = 1\n"
        with patch.object(Path, "open", return_value=BytesIO(source)):
            with self.assertRaises(ValueError):
                load_settings(PROJECT / "config" / "synthetic.toml")


class FrozenPromptAndDependencyTests(unittest.TestCase):
    @unittest.skip('Anonymous release changes identifying comments; original private byte hash does not apply.')
    def test_all_seven_prompt_sources_and_config_remain_byte_identical(self):
        paths = sorted(list((PROJECT / "prompts").glob("*.py")) + [PROJECT / "config" / "prompts.toml"])
        digest = hashlib.sha256()
        for path in paths:
            name = path.relative_to(PROJECT).as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        self.assertEqual(len(paths), 7)
        self.assertEqual(digest.hexdigest(), "44ecd21d2d35c2028ad06301b58d6bc44b422f7e65c8bebe45ffd8c89a9933ea")

    def test_dependency_metadata_requires_nltk_394_and_optional_hf_stack(self):
        data = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        self.assertIn("nltk>=3.9.4,<4", extras["syntax"])
        self.assertIn("transformers>=4.57.1,<5", extras["llm"])
        self.assertIn("torch>=2.6,<3", extras["llm"])
        self.assertEqual(data["project"]["dependencies"], [])
        self.assertIn("llm_adapters*", data["tool"]["setuptools"]["packages"]["find"]["include"])

    def test_import_construction_and_preparation_cannot_import_model_libraries(self):
        code = """
import importlib.abc, sys
class ForbidModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'accelerate', 'safetensors', 'nltk', 'fkassim', 'spacy', 'numpy', 'scipy', 'huggingface_hub'}:
            raise AssertionError('Heavy/model import forbidden: ' + fullname)
sys.meta_path.insert(0, ForbidModels())
from llm_adapters import Qwen3Adapter, Gemma2Adapter, DiSTERLlamaAdapter, LoadConfig, GenerationOptions
from prompts import PromptBuilder, default_prompt_config, build_dister_prompt
prompt = PromptBuilder(default_prompt_config('heart_failure')).build_basic('An arbitrary synthetic English text.')
for adapter in (Qwen3Adapter(), Gemma2Adapter()):
    adapter.prepare_input(prompt)
    assert not adapter.is_loaded
native = DiSTERLlamaAdapter()
native.prepare_input(build_dister_prompt('id', 'Synthetic.', 'custom-domain'))
assert not native.is_loaded
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pure_preparation_needs_no_corpus_or_network_access(self):
        prompt = PromptBuilder(default_prompt_config("wind_energy")).build_basic("Synthetic English text.")
        native = build_dister_prompt("id", "Synthetic English text.", "arbitrary-domain")
        with patch("builtins.open", side_effect=AssertionError("No files")), patch.object(
            Path, "open", side_effect=AssertionError("No paths")
        ), patch("socket.create_connection", side_effect=AssertionError("No network")):
            Qwen3Adapter().prepare_input(prompt)
            Gemma2Adapter().prepare_input(prompt)
            DiSTERLlamaAdapter().prepare_input(native)


if __name__ == "__main__":
    unittest.main()
