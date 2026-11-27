"""Lazy initialization, adapter contract and portable configuration tests."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from syntactic_retrieval import (
    StanfordPCFGConfig,
    StanfordPCFGParser,
    build_retriever,
    load_settings,
)
from syntactic_retrieval.parsers import ConstituencyParseError, ParserUnavailableError


PROJECT = Path(__file__).resolve().parents[1]
HAS_NLTK = importlib.util.find_spec("nltk") is not None


class ConfigurationTests(unittest.TestCase):
    def test_paths_and_default_three_are_resolved_from_config_location(self):
        settings = load_settings(PROJECT / "config" / "fastkassim.toml")
        root = PROJECT.parent / "Corpus para LLM"
        self.assertEqual(settings.corpora.acter, root / "ACTER-master")
        self.assertEqual(settings.corpora.financial_texts, root / "Corpus financiero" / "Silver Standard" / "docsutf8")
        self.assertEqual(settings.corpora.financial_annotations, root / "Corpus financiero" / "Silver Standard" / "gold")
        self.assertEqual(settings.retrieval.top_k, 3)
        self.assertEqual(settings.retrieval.authorized_document_ids, frozenset())

    def test_import_configuration_and_construction_do_not_import_optional_dependencies(self):
        code = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nltk', 'fkassim', 'numpy', 'scipy', 'torch', 'transformers'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from syntactic_retrieval import load_settings, build_retriever
settings = load_settings('config/fastkassim.toml')
retriever = build_retriever([], settings)
assert retriever.retrieve('A synthetic query.') == ()
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=PROJECT,
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_bank_builds_without_loading_parser(self):
        settings = load_settings(PROJECT / "config" / "fastkassim.toml")
        with patch.object(StanfordPCFGParser, "_get_backend", side_effect=AssertionError("Parser load")):
            retriever = build_retriever([], settings)
            self.assertEqual(retriever.retrieve("A synthetic query."), ())


class ParserLifecycleTests(unittest.TestCase):
    def parser(self, **kwargs):
        return StanfordPCFGParser(StanfordPCFGConfig(Path("not-present.jar"), Path("models-not-present.jar"), **kwargs))

    def test_constructing_adapter_is_lazy(self):
        with patch.object(StanfordPCFGParser, "_get_backend", side_effect=AssertionError("Backend initialized")):
            parser = self.parser()
            self.assertIsNone(parser._backend)

    def test_missing_resource_raises_before_initializing_a_model(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(ParserUnavailableError):
                self.parser().parse("An input sentence.")

    def test_invalid_input_and_segmentation(self):
        with self.assertRaises(ValueError):
            self.parser(sentence_segmentation="regex-guess")
        with self.assertRaises(ValueError):
            self.parser().parse("  ")

    @unittest.skipUnless(HAS_NLTK, "NLTK is needed only for predefined tree objects")
    def test_adapter_converts_supplied_backend_trees_without_real_parsing(self):
        from nltk.tree import Tree

        parser = self.parser()
        backend = Mock()
        backend.raw_parse_sents.return_value = iter([iter([Tree.fromstring("(ROOT (S (NP (NN debt))))")])])
        with patch.object(parser, "_get_backend", return_value=backend):
            parsed = parser.parse("Debt\n  grows.")
        backend.raw_parse_sents.assert_called_once_with(["Debt grows."])
        self.assertEqual(parsed.parser_id, parser.parser_id)
        self.assertEqual(parsed.trees[0].label, "ROOT")
        self.assertNotIn("debt", repr(parsed.trees))

    def test_backend_failure_is_not_replaced_with_a_synthetic_tree(self):
        parser = self.parser()
        backend = Mock()
        backend.raw_parse_sents.return_value = iter([iter([])])
        with patch.object(parser, "_get_backend", return_value=backend):
            with self.assertRaises(ConstituencyParseError):
                parser.parse("An input sentence.")

    def test_result_count_must_match_segmentation(self):
        parser = self.parser()
        backend = Mock()
        backend.raw_parse_sents.return_value = iter([])
        with patch.object(parser, "_get_backend", return_value=backend):
            with self.assertRaises(ConstituencyParseError):
                parser.parse("An input sentence.")

    @unittest.skipUnless(HAS_NLTK, "NLTK is needed only for predefined tree objects")
    def test_punkt_mode_uses_supplied_tokenizer_and_backend_without_models(self):
        from nltk.tree import Tree

        parser = self.parser(sentence_segmentation="punkt")
        parser._tokenizer = Mock()
        parser._tokenizer.tokenize.return_value = ["First.\n", "Second."]
        backend = Mock()
        tree = Tree.fromstring("(ROOT (S (NP (NN example))))")
        backend.raw_parse_sents.return_value = iter([iter([tree]), iter([tree])])
        with patch.object(parser, "_get_backend", return_value=backend):
            parsed = parser.parse("First. Second.")
        self.assertEqual(len(parsed.trees), 2)
        backend.raw_parse_sents.assert_called_once_with(["First.", "Second."])

    @unittest.skipUnless(HAS_NLTK, "NLTK is needed to intercept its resource loader")
    def test_missing_punkt_does_not_download_resources(self):
        parser = self.parser(sentence_segmentation="punkt")
        with patch("nltk.data.load", side_effect=LookupError("Not installed")), patch(
            "nltk.download", side_effect=AssertionError("No downloads")
        ), patch.object(parser, "_get_backend", side_effect=AssertionError("No backend")):
            with self.assertRaises(ParserUnavailableError):
                parser.parse("First. Second.")


if __name__ == "__main__":
    unittest.main()
