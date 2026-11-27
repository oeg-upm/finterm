"""Read initial TOML settings without initializing a parser or reading corpora."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Iterable

from .fastkassim import FastKassimConfig, FastKassimSimilarity
from .interfaces import ConstituencyParser
from .models import AnnotatedExample
from .parsers import StanfordPCFGConfig, StanfordPCFGParser
from .retriever import RetrievalConfig, SyntacticRetriever


@dataclass(frozen=True, slots=True)
class CorpusLocations:
    """Location metadata only; this component has no corpus loader or writer."""

    root: Path
    acter: Path
    financial_texts: Path
    financial_annotations: Path


@dataclass(frozen=True, slots=True)
class Settings:
    # Keep the original positional order for existing callers. New callers can
    # omit corpora entirely and provide only parser/kernel/retrieval settings.
    corpora: CorpusLocations | None = None
    parser: StanfordPCFGConfig | None = None
    fastkassim: FastKassimConfig = field(default_factory=FastKassimConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def __post_init__(self) -> None:
        if self.corpora is not None and not isinstance(self.corpora, CorpusLocations):
            raise TypeError("corpora must be CorpusLocations or None")
        if not isinstance(self.parser, StanfordPCFGConfig):
            raise TypeError("A StanfordPCFGConfig is required; corpus paths are optional")


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    required = {"parser", "fastkassim", "retrieval"}
    if not required.issubset(data) or set(data) - required - {"corpora"}:
        raise ValueError("Require parser, fastkassim and retrieval; corpora is optional")

    def local_path(value: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configured paths must be nonempty strings")
        return (config_path.parent / value).resolve()

    corpora = (
        CorpusLocations(**{key: local_path(value) for key, value in data["corpora"].items()})
        if "corpora" in data else None
    )
    parser = dict(data["parser"])
    if parser.pop("backend") != "stanford_pcfg":
        raise ValueError("Use stanford_pcfg or inject another ConstituencyParser explicitly")
    parser["parser_jar"] = local_path(parser["parser_jar"])
    parser["models_jar"] = local_path(parser["models_jar"])
    return Settings(
        corpora=corpora,
        parser=StanfordPCFGConfig(**parser),
        fastkassim=FastKassimConfig(**data["fastkassim"]),
        retrieval=RetrievalConfig(**data["retrieval"]),
    )


def build_retriever(
    examples: Iterable[AnnotatedExample],
    settings: Settings,
    *,
    parser: ConstituencyParser | None = None,
) -> SyntacticRetriever:
    return SyntacticRetriever(
        examples=examples,
        parser=parser if parser is not None else StanfordPCFGParser(settings.parser),
        similarity=FastKassimSimilarity(settings.fastkassim),
        config=settings.retrieval,
    )
