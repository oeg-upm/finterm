"""Selection/filtering tests using explicit synthetic scores, not a fake LTK."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from syntactic_retrieval import (
    AnnotatedExample,
    ParsedText,
    RetrievalConfig,
    SyntaxTree,
    SyntacticRetriever,
)
from syntactic_retrieval.fastkassim import SimilarityComputationError


class FixtureParser:
    """Test fixture: returns supplied trees without analyzing language."""

    def __init__(self, trees):
        self.trees = trees
        self.calls = []

    def parse(self, text):
        self.calls.append(text)
        return self.trees[text]


class SyntheticScores:
    """Only for testing ranking logic; never used by the production factory."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, example):
        self.calls.append((query, example))
        return self.scores[example.trees]


def fixture_bank():
    examples = tuple(
        AnnotatedExample(letter, f"Synthetic sentence {letter}.", (f"term {letter}",), "demo", "test")
        for letter in ("a", "b", "c", "d", "e")
    )
    trees = {"Query.": ParsedText((SyntaxTree("QUERY"),), "fixture")}
    for example in examples:
        trees[example.text] = ParsedText((SyntaxTree(example.identifier.upper()),), "fixture")
    values = {"a": 0.2, "b": 0.9, "c": 0.8, "d": 0.9, "e": 0.1}
    scores = {trees[example.text].trees: values[example.identifier] for example in examples}
    return examples, FixtureParser(trees), SyntheticScores(scores)


class RetrieverTests(unittest.TestCase):
    def make_retriever(self, **config):
        examples, parser, metric = fixture_bank()
        settings = RetrievalConfig(authorized_document_ids={"demo"}, **config)
        return SyntacticRetriever(examples, parser, metric, settings), parser, metric

    def test_top_three_and_complete_payload(self):
        retriever, _, _ = self.make_retriever()
        results = retriever.retrieve("Query.")
        self.assertEqual([item.example.identifier for item in results], ["b", "d", "c"])
        self.assertEqual(
            results[0].to_dict(),
            {"id": "b", "text": "Synthetic sentence b.", "terms": ["term b"],
             "document_id": "demo", "domain": "test", "score": 0.9},
        )

    def test_ties_independent_of_input_order(self):
        examples, parser, metric = fixture_bank()
        config = RetrievalConfig(authorized_document_ids={"demo"})
        a = SyntacticRetriever(examples, parser, metric, config).retrieve("Query.")
        b = SyntacticRetriever(reversed(examples), parser, metric, config).retrieve("Query.")
        self.assertEqual(a, b)

    def test_configurable_k_and_small_candidate_pool(self):
        retriever, _, _ = self.make_retriever(top_k=1)
        self.assertEqual(len(retriever.retrieve("Query.")), 1)
        self.assertEqual(len(retriever.retrieve("Query.", top_k=10)), 5)

    def test_id_and_document_filters_apply_before_parsing(self):
        examples, parser, metric = fixture_bank()
        examples = (
            replace(examples[0], document_id="unauthorized"),
            replace(examples[1], document_id="query-doc"),
            replace(examples[2], document_id="blocked"),
            examples[3], examples[4],
        )
        retriever = SyntacticRetriever(
            examples, parser, metric,
            RetrievalConfig(authorized_document_ids={"demo", "query-doc", "blocked"}),
        )
        result = retriever.retrieve(
            "Query.", query_id="d", query_document_id="query-doc",
            excluded_document_ids={"blocked"}, excluded_example_ids={"e"},
        )
        self.assertEqual(result, ())
        self.assertEqual(parser.calls, [])
        self.assertEqual(metric.calls, [])

    def test_partial_exclusion_does_not_score_excluded_candidates(self):
        retriever, parser, metric = self.make_retriever()
        result = retriever.retrieve("Query.", excluded_example_ids={"b", "d"})
        self.assertEqual([item.example.identifier for item in result], ["c", "a", "e"])
        self.assertNotIn("Synthetic sentence b.", parser.calls)
        self.assertNotIn("Synthetic sentence d.", parser.calls)
        self.assertEqual(len(metric.calls), 3)

    def test_empty_authorization_authorizes_nothing(self):
        examples, parser, metric = fixture_bank()
        retriever = SyntacticRetriever(examples, parser, metric)
        self.assertEqual(retriever.retrieve("Query."), ())
        self.assertFalse(parser.calls)

    def test_self_text_is_excluded_even_under_another_id(self):
        example = AnnotatedExample("copy", "  QUERY.\n", (), "demo", "test")
        parser = FixtureParser({})
        metric = SyntheticScores({})
        retriever = SyntacticRetriever(
            [example], parser, metric, RetrievalConfig(authorized_document_ids={"demo"})
        )
        self.assertEqual(retriever.retrieve("Query."), ())
        self.assertFalse(parser.calls)

    def test_query_document_exclusion_is_configurable(self):
        retriever, _, _ = self.make_retriever(exclude_query_document=False)
        self.assertEqual(len(retriever.retrieve("Query.", query_document_id="demo")), 3)

    def test_annotations_and_domains_do_not_affect_similarity(self):
        examples, parser, metric = fixture_bank()
        config = RetrievalConfig(authorized_document_ids={"demo"})
        original = SyntacticRetriever(examples, parser, metric, config).retrieve("Query.")
        original_calls = tuple(metric.calls)
        metric.calls.clear()
        changed = tuple(replace(example, terms=("UNRELATED ANSWER",), domain="other") for example in examples)
        altered = SyntacticRetriever(changed, parser, metric, config).retrieve("Query.")
        self.assertEqual(tuple(metric.calls), original_calls)
        self.assertEqual(
            [(item.example.identifier, item.score) for item in original],
            [(item.example.identifier, item.score) for item in altered],
        )
        self.assertTrue(all(isinstance(value, ParsedText) for call in metric.calls for value in call))
        for call in metric.calls:
            for parsed in call:
                self.assertFalse(hasattr(parsed, "terms"))
                self.assertFalse(hasattr(parsed, "text"))
        with self.assertRaises(TypeError):
            SyntacticRetriever(examples, parser, metric, config).retrieve("Query.", terms=["gold"])

    def test_selection_performs_no_file_io(self):
        retriever, _, _ = self.make_retriever()
        with patch("builtins.open", side_effect=AssertionError("Unexpected file access")), patch.object(
            Path, "open", side_effect=AssertionError("Unexpected path access")
        ):
            self.assertEqual(len(retriever.retrieve("Query.")), 3)

    def test_parse_cache_is_bounded_and_can_be_cleared(self):
        retriever, parser, _ = self.make_retriever()
        retriever.retrieve("Query.")
        calls = len(parser.calls)
        retriever.retrieve("Query.")
        self.assertEqual(len(parser.calls), calls)
        retriever.clear_parse_cache()
        retriever.retrieve("Query.")
        self.assertEqual(len(parser.calls), calls * 2)
        bounded, _, _ = self.make_retriever(parse_cache_size=2)
        bounded.retrieve("Query.")
        self.assertEqual(len(bounded._cache), 2)

    def test_duplicate_ids_and_invalid_config_are_rejected(self):
        examples, parser, metric = fixture_bank()
        with self.assertRaises(ValueError):
            SyntacticRetriever([examples[0], examples[0]], parser, metric)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(top_k=invalid), self.assertRaises(ValueError):
                RetrievalConfig(top_k=invalid)
        with self.assertRaises(TypeError):
            RetrievalConfig(authorized_document_ids="demo")
        with self.assertRaises(ValueError):
            RetrievalConfig(parse_cache_size=-1)

    def test_invalid_scores_fail_without_fallback(self):
        for invalid in (float("nan"), float("inf"), -1, True):
            retriever, _, metric = self.make_retriever()
            metric.scores = {key: invalid for key in metric.scores}
            with self.subTest(score=invalid), self.assertRaises(SimilarityComputationError):
                retriever.retrieve("Query.")

    def test_scores_are_not_silently_clipped(self):
        retriever, _, metric = self.make_retriever()
        metric.scores = {key: 1.2 for key in metric.scores}
        self.assertEqual(retriever.retrieve("Query.")[0].score, 1.2)

    def test_existing_annotations_are_preserved_and_copied_on_export(self):
        terms = ["Debt", "Debt service", "Debt"]
        example = AnnotatedExample("id", "Original TEXT.", terms, "demo", "finance")
        terms.append("not part of the example")
        self.assertEqual(example.terms, ("Debt", "Debt service", "Debt"))
        from syntactic_retrieval import RetrievedExample

        exported = RetrievedExample(example, 0.5).to_dict()
        exported["terms"].clear()
        self.assertEqual(example.terms, ("Debt", "Debt service", "Debt"))


if __name__ == "__main__":
    unittest.main()
