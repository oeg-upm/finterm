"""Configuration, no-resource execution with doubles and approved-file integrity."""

import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch

from extractor import ExtractorConfig, TermExtractor, UnsupportedModelConditionError, extractor_config_from_toml

from extractor_doubles import (
    QUERY, ExistingRetrieverFixture, adapter_fixture, example_bank_fixture,
    prompt_configuration, settings_fixture,
)


PROJECT = Path(__file__).resolve().parents[1]


class ExtractorConfigurationTests(unittest.TestCase):
    def test_minimal_configuration_and_python_defaults_match(self):
        source = (PROJECT / "config" / "extractor.toml").read_text(encoding="utf-8")
        self.assertEqual(extractor_config_from_toml(source), ExtractorConfig())

    def test_native_defaults_are_explicit_and_invalid_pairs_are_rejected(self):
        native = ExtractorConfig(default_model="dister", default_condition="native_zero_shot")
        self.assertEqual(native.default_model, "dister")
        with self.assertRaises(UnsupportedModelConditionError):
            ExtractorConfig(default_model="dister")

    def test_configuration_parses_strings_without_opening_files(self):
        source = '[extractor]\ndefault_model = "gemma2"\ndefault_condition = "patterns"\n'
        with patch("builtins.open", side_effect=AssertionError("No files")), patch.object(
            Path, "open", side_effect=AssertionError("No paths")
        ):
            self.assertEqual(extractor_config_from_toml(source).default_model, "gemma2")

    def test_configuration_does_not_accept_dataset_or_model_loading_sections(self):
        for source in ('[corpora]\nroot = "unused"', '[extractor]\n[model_loading]\nsource = "unused"'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                extractor_config_from_toml(source)

    def test_package_discovery_preserves_all_approved_packages(self):
        data = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertTrue({"syntactic_retrieval*", "prompts*", "llm_adapters*", "extractor*"}.issubset(
            data["tool"]["setuptools"]["packages"]["find"]["include"]
        ))
        self.assertEqual(data["project"]["dependencies"], [])
        self.assertIn("nltk>=3.9.4,<4", data["project"]["optional-dependencies"]["syntax"])


class ExtractorIsolationTests(unittest.TestCase):
    def test_import_and_construction_do_not_import_optional_libraries(self):
        code = """
import importlib.abc, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'nltk', 'fkassim', 'spacy', 'numpy', 'scipy', 'accelerate', 'huggingface_hub', 'safetensors'}:
            raise AssertionError('Resource library import forbidden: ' + fullname)
sys.meta_path.insert(0, NoModels())
from extractor import TermExtractor, ExtractorConfig, extractor_config_from_toml
engine = TermExtractor()
assert engine._retriever is None
assert engine._adapters == {}
assert extractor_config_from_toml('[extractor]').default_model == 'qwen3'
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_flows_execute_with_doubles_while_files_network_and_real_resources_are_forbidden(self):
        bank = example_bank_fixture()
        fixture = ExistingRetrieverFixture(bank)
        adapters = {key: adapter_fixture(key)[0] for key in ("qwen3", "gemma2", "dister")}
        engine = TermExtractor(
            examples=bank, parser=fixture.parser,
            retrieval_settings=settings_fixture({f"doc-{key}" for key in ("a", "b", "c", "d")}),
            adapters=adapters,
        )
        with patch("builtins.open", side_effect=AssertionError("No files")), patch.object(
            Path, "open", side_effect=AssertionError("No paths")
        ), patch("socket.create_connection", side_effect=AssertionError("No network")), patch(
            "syntactic_retrieval.parsers.StanfordPCFGParser.parse", side_effect=AssertionError("No real parser")
        ), patch(
            "syntactic_retrieval.fastkassim.FastKassimSimilarity.score", side_effect=AssertionError("No real kernel")
        ), patch(
            "llm_adapters.adapters.load_huggingface_runtime", side_effect=AssertionError("No real model load")
        ), patch("extractor.pipeline.build_retriever", wraps=fixture):
            for key in ("qwen3", "gemma2"):
                engine.extract(QUERY, prompt_configuration(), model=key, condition="basic")
                engine.extract(QUERY, prompt_configuration(), model=key, condition="patterns")
                engine.extract(QUERY, prompt_configuration(), model=key, condition="hybrid", candidates=["supplied"])
                engine.extract(QUERY, prompt_configuration(), model=key, condition="few_shot",
                               query_document_id="query-doc", query_id="query-id")
            engine.extract(QUERY, prompt_configuration(), model="dister", condition="native_zero_shot", query_id="native-id")

    def test_historical_lowlevel_api_contains_no_corpus_io_or_duplicate_implementations(self):
        forbidden = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir", "unlink", "rename",
                     "from_pretrained", "download", "apply_chat_template", "tokenize", "validate_terms_json"}
        # The newly authorized user CLI reads text/banks/reports. This invariant
        # still applies to the historical, corpus-independent low-level API.
        for path in [PROJECT / "extractor" / name for name in ("__init__.py", "config.py", "models.py", "pipeline.py")]:
            source = path.read_text(encoding="utf-8")
            for corpus in ("ACTER", "CELEX", "IATE"):
                self.assertNotIn(corpus, source)
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                    self.assertNotIn(name, forbidden, (path.name, node.lineno))


class ApprovedComponentsIntegrityTests(unittest.TestCase):
    @unittest.skip('Anonymous source release excludes private historical reports; original private byte hash does not apply.')
    def test_all_37_historical_sources_and_records_remain_byte_identical(self):
        paths = [p for folder in ("syntactic_retrieval", "prompts", "llm_adapters")
                 for p in (PROJECT / folder).glob("*.py")]
        paths += [PROJECT / "config" / name for name in (
            "fastkassim.toml", "fastkassim_standalone.toml", "llm_adapters.toml", "prompts.toml",
        )]
        paths += [p for p in (PROJECT / "tests").glob("test_*.py") if not p.name.startswith("test_extractor")]
        # .gitignore is intentionally extended for the functional source release;
        # all 37 other historical sources/configs/tests/reports stay frozen.
        paths += [PROJECT / "tests" / "adapter_doubles.py"]
        paths += list(PROJECT.glob("0[0-3]_*.md"))
        digest = hashlib.sha256()
        for path in sorted(paths):
            name = path.relative_to(PROJECT).as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        self.assertEqual(len(paths), 37)
        self.assertEqual(digest.hexdigest(), "718281fd4025e2e814229cad423e4272ce12221bb8cb5fe5bb35f533061120fb")

    def test_source_release_exclusions_extend_the_historical_gitignore(self):
        source = (PROJECT / ".gitignore").read_text(encoding="utf-8")
        self.assertTrue(source.startswith("__pycache__/\n*.py[cod]\n*.egg-info/\n.venv/\nbuild/\ndist/\nresources/\n"))
        for name in ("evaluation_runs/", "evaluation_runs_v2/", "extraction_unit_runs/", ".env", "*.safetensors",
                     "demonstrations/candidatos_financieros.jsonl"):
            self.assertIn(name, source.splitlines())


if __name__ == "__main__":
    unittest.main()
