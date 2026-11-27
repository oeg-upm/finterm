"""Configuration and isolation checks limited to the new prompt component."""

import ast
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch

from prompts import (
    PromptBuilder, PromptCondition, build_dister_prompt, default_prompt_config,
    prompt_settings_from_toml,
)
from syntactic_retrieval.models import AnnotatedExample, RetrievedExample


PROJECT = Path(__file__).resolve().parents[1]


class PromptConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only this phase's own configuration file is read, never corpus data.
        cls.source = (PROJECT / "config" / "prompts.toml").read_text(encoding="utf-8")

    def test_file_configures_all_four_conditions_and_five_domains(self):
        settings = prompt_settings_from_toml(self.source)
        self.assertEqual(set(settings.conditions), set(PromptCondition))
        self.assertEqual(set(settings.configurations), {
            "finance", "corruption", "equitation_dressage", "heart_failure", "wind_energy",
        })
        for key in settings.configurations:
            self.assertEqual(settings.for_domain(key), default_prompt_config(key))

    def test_configuration_can_be_parsed_without_accessing_files(self):
        with patch("builtins.open", side_effect=AssertionError("No file access")), patch.object(
            Path, "open", side_effect=AssertionError("No path access")
        ):
            settings = prompt_settings_from_toml(self.source)
        self.assertEqual(settings.for_domain("finance").few_shot_examples, 3)

    def test_toml_supports_custom_linguistic_structure_descriptions(self):
        source = self.source.replace(
            'structure_profile = "financial"',
            'structure_profile = "financial"\nstructure_lines = ["N: synthetic illustration"]',
            1,
        )
        config = prompt_settings_from_toml(source).for_domain("finance")
        user = PromptBuilder(config).build_patterns("Synthetic.").messages[1].content
        self.assertIn("- N: synthetic illustration", user)
        self.assertNotIn("- N: dividend", user)

    def test_financial_profile_cannot_silently_be_used_for_acter(self):
        source = self.source.replace('structure_profile = "general"', 'structure_profile = "financial"', 1)
        with self.assertRaises(ValueError):
            prompt_settings_from_toml(source)

    def test_unknown_domain_does_not_fall_back_to_finance(self):
        settings = prompt_settings_from_toml(self.source)
        with self.assertRaises(KeyError):
            settings.for_domain("unknown")

    def test_invalid_or_duplicate_conditions_and_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            prompt_settings_from_toml(self.source.replace('"basic", "patterns"', '"basic", "basic"', 1))
        with self.assertRaises(ValueError):
            prompt_settings_from_toml(self.source.replace("few_shot_examples = 3", "few_shot_examples = 0", 1))
        with self.assertRaises(TypeError):
            prompt_settings_from_toml(PROJECT / "config" / "prompts.toml")

    def test_settings_cannot_be_mutated_through_the_domain_mapping(self):
        settings = prompt_settings_from_toml(self.source)
        with self.assertRaises(TypeError):
            settings.configurations["finance"] = default_prompt_config("corruption")

    def test_packaging_preserves_the_old_package_and_includes_prompts(self):
        data = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertTrue(
            {"syntactic_retrieval*", "prompts*"}.issubset(
                data["tool"]["setuptools"]["packages"]["find"]["include"]
            )
        )
        self.assertEqual(data["project"]["dependencies"], [])


class PromptIsolationTests(unittest.TestCase):
    def test_all_builders_run_without_files_network_parser_or_kernel_calls(self):
        config = default_prompt_config("finance")
        builder = PromptBuilder(config)
        examples = tuple(
            RetrievedExample(AnnotatedExample(str(index), f"Synthetic text {index}.", (), "demo", "test"), 0.5)
            for index in range(3)
        )
        with patch("builtins.open", side_effect=AssertionError("No file access")), patch.object(
            Path, "open", side_effect=AssertionError("No path access")
        ), patch("socket.create_connection", side_effect=AssertionError("No network")), patch(
            "syntactic_retrieval.parsers.StanfordPCFGParser.parse", side_effect=AssertionError("No parser")
        ), patch(
            "syntactic_retrieval.fastkassim.FastKassimSimilarity.score", side_effect=AssertionError("No kernel")
        ), patch(
            "syntactic_retrieval.retriever.SyntacticRetriever.retrieve", side_effect=AssertionError("No retrieval")
        ):
            builder.build_basic("Synthetic.").to_json()
            builder.build_patterns("Synthetic.").to_json()
            builder.build_few_shot("Synthetic.", examples).to_json()
            builder.build_hybrid("Synthetic.", ["supplied candidate"]).to_json()
            build_dister_prompt("id", "Synthetic.", "test").to_json()

    def test_import_and_construction_work_when_model_libraries_are_forbidden(self):
        code = """
import importlib.abc, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'nltk', 'fkassim', 'spacy', 'numpy', 'scipy', 'huggingface_hub'}:
            raise AssertionError('Model/parser library import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockModels())
from prompts import PromptBuilder, default_prompt_config, build_dister_prompt
from syntactic_retrieval.models import AnnotatedExample, RetrievedExample
builder = PromptBuilder(default_prompt_config('finance'))
examples = [RetrievedExample(AnnotatedExample(str(i), 'Example ' + str(i), (), 'demo', 'test'), 0.5) for i in range(3)]
builder.build_basic('Synthetic.').to_json()
builder.build_patterns('Synthetic.').to_json()
builder.build_few_shot('Synthetic.', examples).to_json()
builder.build_hybrid('Synthetic.', ['candidate']).to_json()
build_dister_prompt('id', 'Synthetic.', 'test').to_json()
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=PROJECT,
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_has_no_corpus_io_model_loading_or_retrieval_calls(self):
        forbidden_calls = {
            "open", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir",
            "unlink", "rename", "download", "from_pretrained", "generate",
            "retrieve", "score", "parse", "eval", "exec",
        }
        for path in sorted((PROJECT / "prompts").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None
                )
                self.assertNotIn(name, forbidden_calls, (path.name, node.lineno, name))


if __name__ == "__main__":
    unittest.main()
