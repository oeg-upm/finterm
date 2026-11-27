"""In-memory syntactic retrieval with explicit document authorization."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable
import unicodedata

from .fastkassim import checked_score
from .interfaces import ConstituencyParser, TreeSimilarity
from .models import AnnotatedExample, ParsedText, RetrievedExample, identifier_set, require_text


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 3
    authorized_document_ids: frozenset[str] = frozenset()
    exclude_query_document: bool = True
    exclude_same_text: bool = True
    parse_cache_size: int = 1024

    def __post_init__(self) -> None:
        _check_k(self.top_k)
        if type(self.parse_cache_size) is not int or self.parse_cache_size < 0:
            raise ValueError("parse_cache_size must be a nonnegative integer")
        for name in ("exclude_query_document", "exclude_same_text"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "authorized_document_ids",
            identifier_set(self.authorized_document_ids, "authorized_document_ids"),
        )


def _check_k(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("top_k must be a positive integer")


def _same_text_key(text: str) -> str:
    # Exact-duplicate prevention only. This is never a similarity/ranking input.
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class SyntacticRetriever:
    """Select up to k existing examples; never read/write a corpus or create terms.

    An empty authorization set authorizes no document. Filtering happens before
    parsing or scoring. Only parsed structure is passed to the similarity object.
    """

    def __init__(
        self,
        examples: Iterable[AnnotatedExample],
        parser: ConstituencyParser,
        similarity: TreeSimilarity,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.config = config if config is not None else RetrievalConfig()
        self.parser = parser
        self.similarity = similarity
        self._examples = tuple(examples)
        if not all(isinstance(example, AnnotatedExample) for example in self._examples):
            raise TypeError("Every candidate must be an AnnotatedExample")
        ids = [example.identifier for example in self._examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Example identifiers must be unique within the bank")
        self._cache: OrderedDict[str, ParsedText] = OrderedDict()

    def clear_parse_cache(self) -> None:
        self._cache.clear()

    def _parse(self, text: str) -> ParsedText:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        parsed = self.parser.parse(text)
        if not isinstance(parsed, ParsedText):
            raise TypeError("ConstituencyParser.parse() must return ParsedText")
        if self.config.parse_cache_size:
            self._cache[text] = parsed
            if len(self._cache) > self.config.parse_cache_size:
                self._cache.popitem(last=False)
        return parsed

    def retrieve(
        self,
        text: str,
        *,
        query_id: str | None = None,
        query_document_id: str | None = None,
        excluded_example_ids: Iterable[str] = (),
        excluded_document_ids: Iterable[str] = (),
        top_k: int | None = None,
    ) -> tuple[RetrievedExample, ...]:
        require_text(text, "query text")
        k = self.config.top_k if top_k is None else top_k
        _check_k(k)
        excluded_ids = identifier_set(excluded_example_ids, "excluded_example_ids")
        excluded_docs = identifier_set(excluded_document_ids, "excluded_document_ids")
        if query_id is not None:
            require_text(query_id, "query_id")
            excluded_ids = excluded_ids | {query_id}
        if query_document_id is not None:
            require_text(query_document_id, "query_document_id")
            if self.config.exclude_query_document:
                excluded_docs = excluded_docs | {query_document_id}
        query_key = _same_text_key(text)
        candidates = [
            example
            for example in self._examples
            if example.document_id in self.config.authorized_document_ids
            and example.document_id not in excluded_docs
            and example.identifier not in excluded_ids
            and (
                not self.config.exclude_same_text
                or _same_text_key(example.text) != query_key
            )
        ]
        if not candidates:
            return ()
        query = self._parse(text)
        scored = [
            RetrievedExample(
                example,
                checked_score(self.similarity.score(query, self._parse(example.text))),
            )
            for example in candidates
        ]
        # IDs are unique, so ties are independent of bank insertion order.
        scored.sort(key=lambda item: (-item.score, item.example.identifier))
        return tuple(scored[:k])
