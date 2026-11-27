"""Synthetic integration fixtures; no model, tokenizer, parser or LTK execution."""

from pathlib import Path
from unittest.mock import Mock

from adapter_doubles import runtime_double
from llm_adapters import DiSTERLlamaAdapter, Gemma2Adapter, Qwen3Adapter
from prompts import AnnotationPolicy, DomainSpec, NestedTerms, PromptConfig
from syntactic_retrieval import (
    AnnotatedExample, ParsedText, RetrievalConfig, Settings, StanfordPCFGConfig, SyntaxTree,
)
from syntactic_retrieval.config import build_retriever as existing_build_retriever


QUERY = 'Arbitrary synthetic technical input: "quoted" {data}.\nA second line.'


def prompt_configuration():
    return PromptConfig(
        domain=DomainSpec("materials_fixture", "materials science", "Materials science and manufacturing."),
        policy=AnnotationPolicy(nested_terms=NestedTerms.LONGEST_ONLY),
    )


def adapter_fixture(key="qwen3", response=' {"terms": ["fixture-A", "fixture-A"]}\n'):
    runtime = runtime_double(response)
    loader = Mock(return_value=runtime)
    cls = {"qwen3": Qwen3Adapter, "gemma2": Gemma2Adapter, "dister": DiSTERLlamaAdapter}[key]
    return cls(runtime_loader=loader), runtime, loader


def settings_fixture(authorized, **options):
    return Settings(
        parser=StanfordPCFGConfig(Path("fixture-parser.jar"), Path("fixture-models.jar")),
        retrieval=RetrievalConfig(authorized_document_ids=authorized, **options),
    )


def example_bank_fixture():
    training = [
        AnnotatedExample(
            key, f'Synthetic example {key}: {{value}} "quoted".\nSecond line.',
            (f'term "{key}"', "short", "long short", "short"),
            f"doc-{key}", "synthetic demonstration domain",
        )
        for key in ("a", "b", "c", "d")
    ]
    return tuple(training + [
        AnnotatedExample("unauthorized", "Private synthetic text.", ("must not be used",), "private-doc", "test"),
        AnnotatedExample("query-id", "Another unit in the evaluated document.", ("excluded",), "query-doc", "test"),
        AnnotatedExample("same-text", "  " + QUERY.upper() + "\n", ("excluded",), "copy-doc", "test"),
        AnnotatedExample("heldout-doc-example", "Another evaluation document.", ("excluded",), "heldout-doc", "test"),
        AnnotatedExample("heldout-id", "An excluded evaluation example.", ("excluded",), "heldout-id-doc", "test"),
    ])


class ParserDouble:
    """Returns supplied structural records; never analyzes language."""

    def __init__(self, bank):
        texts = [QUERY] + [example.text for example in bank]
        self.parses = {
            text: ParsedText((SyntaxTree(f"FIXTURE_{index}"),), "synthetic-parser-double")
            for index, text in enumerate(texts)
        }
        self.calls = []

    def parse(self, text):
        self.calls.append(text)
        return self.parses[text]


class SimilarityDouble:
    """Synthetic scores test existing selection logic, not the LTK algorithm."""

    def __init__(self, bank, parser):
        values = {"a": 0.2, "b": 0.9, "c": 0.8, "d": 0.9}
        self.scores = {
            parser.parses[example.text].trees: values.get(example.identifier, 100.0)
            for example in bank
        }
        self.calls = []

    def score(self, query, example):
        self.calls.append((query, example))
        return self.scores[example.trees]


class ExistingRetrieverFixture:
    """Use the actual old factory, filters and ranking with mocked analysis only."""

    def __init__(self, bank):
        self.parser = ParserDouble(bank)
        self.similarity = SimilarityDouble(bank, self.parser)
        self.created = []

    def __call__(self, examples, settings, *, parser=None):
        if parser is not self.parser:
            raise AssertionError("Expected the injected parser double")
        retriever = existing_build_retriever(examples, settings, parser=parser)
        retriever.similarity = self.similarity
        retriever.retrieve = Mock(wraps=retriever.retrieve)
        self.created.append(retriever)
        return retriever
