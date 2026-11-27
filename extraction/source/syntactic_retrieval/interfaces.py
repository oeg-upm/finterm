"""Structural interfaces: annotation metadata is not an input to similarity."""

from typing import Protocol

from .models import ParsedText


class ConstituencyParser(Protocol):
    def parse(self, text: str) -> ParsedText:
        """Produce constituent/POS trees for one accepted input unit."""
        ...


class TreeSimilarity(Protocol):
    def score(self, query: ParsedText, example: ParsedText) -> float:
        """Compare structures only; no text, domain, IDs or term annotations."""
        ...
