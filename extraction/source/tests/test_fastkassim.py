"""Tests on literal predefined trees. No parser, models or corpus is executed."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from syntactic_retrieval import (
    AnnotatedExample,
    FastKassimConfig,
    FastKassimSimilarity,
    ParsedText,
    RetrievalConfig,
    SyntaxTree,
    SyntacticRetriever,
)
from syntactic_retrieval.fastkassim import FastKassimUnavailableError, _dependencies


HAS_NLTK = importlib.util.find_spec("nltk") is not None
HAS_UPSTREAM = importlib.util.find_spec("fkassim") is not None
SENTENCE = "(ROOT (S (NP (NN debt)) (VP (VBZ rises)) (. .)))"
OTHER_WORDS = "(ROOT (S (NP (NN wind)) (VP (VBZ blows)) (. !)))"
OTHER_SHAPE = "(ROOT (FRAG (UH hello) (. !)))"


class KernelConfigurationTests(unittest.TestCase):
    def test_ltk_only_and_valid_parameters(self):
        self.assertEqual(FastKassimConfig().metric, "LTK")
        for options in ({"metric": "FTK"}, {"sigma": 2}, {"lmbda": 0},
                        {"lmbda": float("nan")}, {"lmbda": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FastKassimConfig(**options)

    def test_missing_upstream_never_falls_back(self):
        parsed = ParsedText((SyntaxTree("NN"),), "fixture")
        original_import = __import__

        def without_upstream(name, *args, **kwargs):
            if name == "fkassim.ltk.ltk":
                raise ModuleNotFoundError("Upstream deliberately unavailable in this test")
            return original_import(name, *args, **kwargs)

        _dependencies.cache_clear()
        try:
            with patch("builtins.__import__", side_effect=without_upstream):
                with self.assertRaises(FastKassimUnavailableError):
                    FastKassimSimilarity().score(parsed, parsed)
        finally:
            _dependencies.cache_clear()

    def test_parser_identity_must_match(self):
        a = ParsedText((SyntaxTree("NN"),), "parser-a")
        b = ParsedText((SyntaxTree("NN"),), "parser-b")
        with self.assertRaises(ValueError):
            FastKassimSimilarity().score(a, b)


@unittest.skipUnless(HAS_NLTK, "NLTK is needed only to read predefined Penn trees")
class TreeRepresentationTests(unittest.TestCase):
    def test_word_leaves_are_discarded(self):
        a = SyntaxTree.from_penn(SENTENCE)
        b = SyntaxTree.from_penn(OTHER_WORDS)
        self.assertEqual(a, b)
        self.assertNotIn("debt", repr(a))
        self.assertNotIn("wind", repr(b))

    def test_nltk_conversion_retains_structure_with_constant_leaves(self):
        tree = SyntaxTree.from_penn(SENTENCE)
        self.assertEqual(SyntaxTree.from_nltk(tree.to_nltk()), tree)
        self.assertEqual(set(tree.to_nltk().leaves()), {"__FASTKASSIM_LEAF__"})

    def test_invalid_empty_or_mixed_external_trees_are_rejected(self):
        for bracketed in ("(ROOT)", "(ROOT word (NN debt))", "(NN two words)"):
            with self.subTest(tree=bracketed), self.assertRaises(ValueError):
                SyntaxTree.from_penn(bracketed)
        with self.assertRaises(ValueError):
            ParsedText((), "fixture")
        with self.assertRaises(TypeError):
            SyntaxTree("NP", ("word",))


@unittest.skipUnless(HAS_UPSTREAM and HAS_NLTK, "Original fkassim source is not installed/available")
class RealFastKassimTests(unittest.TestCase):
    @staticmethod
    def parsed(*trees):
        return ParsedText(tuple(SyntaxTree.from_penn(tree) for tree in trees), "predefined")

    def test_real_ltk_identity_and_lexical_invariance_without_models(self):
        query = self.parsed(SENTENCE)
        example = self.parsed(OTHER_WORDS)
        with patch("nltk.data.load", side_effect=AssertionError("Resource load forbidden")), patch(
            "nltk.download", side_effect=AssertionError("Download forbidden")
        ), patch("nltk.parse.stanford.StanfordParser", side_effect=AssertionError("Parser forbidden")):
            score = FastKassimSimilarity().score(query, example)
        self.assertAlmostEqual(score, 1.0)

    def test_changed_constituent_structure_changes_real_score(self):
        metric = FastKassimSimilarity()
        query = self.parsed(SENTENCE)
        self.assertLess(metric.score(query, self.parsed(OTHER_SHAPE)), metric.score(query, query))

    def test_same_shape_word_changes_do_not_reach_the_kernel(self):
        metric = FastKassimSimilarity()
        query = self.parsed(SENTENCE)
        self.assertEqual(metric.score(query, self.parsed(SENTENCE)), metric.score(query, self.parsed(OTHER_WORDS)))

    def test_document_aggregation_matches_original_public_method(self):
        from fkassim.kassim.Kassim import Kassim
        from fkassim.ltk.ltk import LabelTreeKernel

        query = self.parsed(SENTENCE, OTHER_SHAPE)
        example = self.parsed(OTHER_WORDS)
        for average in (False, True):
            with self.subTest(average=average):
                # The method needs only Kernel; never call Kassim/FastKassim.__init__.
                expected = Kassim.syntax_similarity_two_parsed_documents(
                    SimpleNamespace(Kernel=LabelTreeKernel),
                    [tree.to_nltk() for tree in example.trees],
                    [tree.to_nltk() for tree in query.trees],
                    average=average, sigma=1, lmbda=0.4, use_new_delta=True,
                )
                actual = FastKassimSimilarity(FastKassimConfig(average=average)).score(query, example)
                self.assertAlmostEqual(actual, expected)

    def test_retrieval_with_real_kernel_on_predefined_trees(self):
        mapping = {"query": self.parsed(SENTENCE), "same syntax": self.parsed(OTHER_WORDS),
                   "different syntax": self.parsed(OTHER_SHAPE)}

        class PredefinedTrees:
            def parse(self, text):
                return mapping[text]

        examples = [
            AnnotatedExample("different", "different syntax", (), "demo", "synthetic"),
            AnnotatedExample("same", "same syntax", ("known annotation",), "demo", "synthetic"),
        ]
        retriever = SyntacticRetriever(
            examples, PredefinedTrees(), FastKassimSimilarity(),
            RetrievalConfig(authorized_document_ids={"demo"}),
        )
        self.assertEqual(retriever.retrieve("query")[0].example.identifier, "same")


if __name__ == "__main__":
    unittest.main()
