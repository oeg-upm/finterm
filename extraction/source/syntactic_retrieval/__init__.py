"""Constituency-based example retrieval for the future FINTERM extractor.

Importing this package does not import fkassim/NLTK, load models, or access corpora.
"""

from .config import CorpusLocations, Settings, build_retriever, load_settings
from .fastkassim import FastKassimConfig, FastKassimSimilarity
from .interfaces import ConstituencyParser, TreeSimilarity
from .models import AnnotatedExample, ParsedText, RetrievedExample, SyntaxTree
from .parsers import StanfordPCFGConfig, StanfordPCFGParser
from .retriever import RetrievalConfig, SyntacticRetriever

__all__ = [
    "AnnotatedExample",
    "ConstituencyParser",
    "CorpusLocations",
    "FastKassimConfig",
    "FastKassimSimilarity",
    "ParsedText",
    "RetrievedExample",
    "RetrievalConfig",
    "Settings",
    "StanfordPCFGConfig",
    "StanfordPCFGParser",
    "SyntaxTree",
    "SyntacticRetriever",
    "TreeSimilarity",
    "build_retriever",
    "load_settings",
]
