"""Build a Wikidata candidate inventory without semantic ranking or selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "FINTERM-CandidateRetriever/1.1"
REQUEST_TIMEOUT_SECONDS = 45
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 0.5
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_SERVICE_DELAY_SECONDS = 5.0
MAX_RETRY_WAIT_SECONDS = 60.0
TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
TRANSIENT_API_CODES = frozenset({"maxlag", "cirrussearch-too-busy-error", "ratelimited", "readonly"})
ENTITY_SEARCH_LIMIT = 50
MAX_ENTITY_SEARCH_RESULTS_PER_QUERY = 100
MAX_CONTEXT_ENTITY_SEARCH_RESULTS_PER_QUERY = 20
MAX_CONTEXT_LITERAL_QUERIES_PER_TERM = 2
MAX_CONTEXT_EXTENSION_WORDS = 3
MAX_CONTEXT_LITERAL_QUERY_WORDS = 8
MAX_ADDITIONAL_QUERIES_PER_TERM = 2
MAX_ADDITIONAL_ENTITY_SEARCH_RESULTS_PER_QUERY = 20
MAX_SUPPLEMENTARY_SEARCH_RESULTS_PER_TERM = 20
SCHOLARLY_ARTICLE_QID = "Q13442814"
ENTITY_BATCH_SIZE = 50
SPARQL_SEARCH_BATCH_SIZE = 12
SPARQL_BATCH_SIZE = 30

QID_PATTERN = re.compile(r"Q[1-9][0-9]*")
USEFUL_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
CONTEXT_WORD_PATTERN = re.compile(r"\b\w+(?:[-'\u2019]\w+)*\b", re.UNICODE)
DISAMBIGUATION_CLASSES = {"Q4167410", "Q22808320"}
RELATION_PROPERTIES = {
    "P31": "p31",
    "P279": "p279",
    "P1343": "p1343",
}

OUTPUT_COLUMNS = (
    "term",
    "qid",
    "label_en",
    "aliases_en",
    "description_en",
    "descriptions_other_languages",
    "p31",
    "p279",
    "p1343",
    "search_queries",
    "search_strategies",
)

T = TypeVar("T")
PROPER_NAME_CONNECTORS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
}


@dataclass
class EntitySearchAudit:
    pages: int = 0
    results_seen: int = 0
    completed: bool = False
    stopped_by_limit: bool = False
    hits: dict[str, dict[str, str | int]] = field(default_factory=dict)
    accepted_qids: set[str] = field(default_factory=set)


@dataclass
class SparqlSearchAudit:
    completed: bool = False
    qids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ContextQueryEvidence:
    term: str
    query: str
    occurrence_id: str
    matched_text: str
    context: str
    document_id: str = ""
    term_id: str = ""
    occurrence_uid: str = ""
    query_match_start: int | None = None
    query_match_end: int | None = None
    query_start: int | None = None
    query_end: int | None = None


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.responses: dict[str, Any] = {}
        self.dirty_entries = 0
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not isinstance(
                payload.get("responses"), dict
            ):
                raise ValueError(f"Unsupported Wikidata cache format: {path}")
            self.responses = payload["responses"]

    @staticmethod
    def request_key(endpoint: str, params: dict[str, str]) -> str:
        canonical = json.dumps(
            [endpoint, sorted(params.items())],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, endpoint: str, params: dict[str, str]) -> Any | None:
        return self.responses.get(self.request_key(endpoint, params))

    def set(self, endpoint: str, params: dict[str, str], payload: Any) -> None:
        self.responses[self.request_key(endpoint, params)] = payload
        self.dirty_entries += 1
        if self.dirty_entries >= 5:
            self.flush()

    def flush(self) -> None:
        if self.dirty_entries == 0:
            return
        temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        temporary_path.write_text(
            json.dumps(
                {"version": 1, "responses": self.responses},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
        self.dirty_entries = 0


class WikidataServiceError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.transient = code.casefold() in TRANSIENT_API_CODES
        super().__init__(f"Wikidata API error {code}: {message}")


def http_user_agent(contact: str | None = None, *, required: bool = False) -> str:
    contact = (contact or "").strip()
    if not contact:
        if required:
            raise ValueError("Online requests require --http-contact or FINTERM_WIKIMEDIA_CONTACT with an authorized public contact")
        return USER_AGENT
    if len(contact) > 400 or any(ord(char) < 33 or ord(char) > 126 for char in contact):
        raise ValueError("HTTP contact must be a single public URL/email without whitespace or header controls")
    address = contact.removeprefix("mailto:")
    email = re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}", address)
    url = urlparse(contact)
    public_url = url.scheme == "https" and url.hostname and not url.username and not url.password
    if not email and not public_url:
        raise ValueError("HTTP contact must be an authorized email/mailto or credential-free HTTPS contact URL")
    return f"{USER_AGENT} ({contact})"


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = max(0.0, date.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError, AttributeError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def validate_service_response(endpoint: str, params: dict[str, str], payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Service response must be a JSON object")
    if "error" in payload:
        error = payload["error"]
        if not isinstance(error, dict):
            raise ValueError("Malformed Wikidata API error object")
        raise WikidataServiceError(str(error.get("code", "unknown")), str(error.get("info", error)))
    if endpoint == WIKIDATA_SPARQL_URL:
        if not isinstance(payload.get("results"), dict) or not isinstance(payload["results"].get("bindings"), list):
            raise ValueError("Malformed SPARQL bindings")
        if any(not isinstance(row, dict) for row in payload["results"]["bindings"]):
            raise ValueError("Malformed SPARQL binding row")
        if "?spec_index" in params.get("query", ""):
            allowed = set(re.findall(r'\("(\d+)" ', params["query"]))
            if any(not isinstance(row.get("spec_index"), dict) or str(row["spec_index"].get("value", "")) not in allowed
                   for row in payload["results"]["bindings"]):
                raise ValueError("SPARQL response spec_index is outside the requested batch")
    elif params.get("action") == "wbsearchentities":
        if not isinstance(payload.get("search"), list) or any(not isinstance(row, dict) for row in payload["search"]):
            raise ValueError("Malformed EntitySearch results")
    elif params.get("action") == "wbgetentities":
        if not isinstance(payload.get("entities"), dict) or any(not isinstance(value, dict) for value in payload["entities"].values()):
            raise ValueError("Malformed entity metadata")
    elif params.get("action") == "query" and params.get("list") == "search":
        if (not isinstance(payload.get("query"), dict) or not isinstance(payload["query"].get("search"), list)
                or any(not isinstance(row, dict) for row in payload["query"]["search"])):
            raise ValueError("Malformed supplementary Cirrus search results")


class WikidataClient:
    def __init__(self, cache: JsonCache, *, http_contact: str | None = None) -> None:
        self.cache = cache
        self.user_agent = http_user_agent(http_contact)
        self.last_attempts: list[dict] = []
        self.cooldowns: dict[str, float] = {}

    def record_attempt(self, detail: dict, payload: Any = None) -> None:
        self.last_attempts.append(dict(detail))

    def get_json(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        self.last_attempts = []
        cached = self.cache.get(endpoint, params)
        if cached is not None:
            try:
                validate_service_response(endpoint, params, cached)
            except ValueError:
                pass  # Keep the original entry; replace it only with valid success.
            else:
                return cached
        remaining = self.cooldowns.get(endpoint, 0.0) - time.monotonic()
        if remaining > 0:
            if remaining > MAX_RETRY_WAIT_SECONDS:
                self.record_attempt({"attempt": 0, "outcome": "DEFERRED_ENDPOINT_COOLDOWN", "wait_seconds": 0,
                                     "retry_after_seconds": remaining, "will_retry": False})
                raise RuntimeError(f"Wikidata endpoint deferred by server cooldown ({remaining:.1f}s); no request sent: {endpoint}")
            self.record_attempt({"attempt": 0, "outcome": "WAIT_ENDPOINT_COOLDOWN", "wait_seconds": remaining,
                                 "retry_after_seconds": remaining, "will_retry": True})
            time.sleep(remaining)
        self.cooldowns.pop(endpoint, None)
        request_url = f"{endpoint}?{urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            payload, header, status = None, None, None
            request = Request(request_url, headers={"Accept": "application/json", "User-Agent": self.user_agent})
            detail = {"attempt": attempt + 1, "attempt_utc": datetime.now(timezone.utc).isoformat()}
            try:
                with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    status = getattr(response, "status", 200)
                    header = response.headers.get("Retry-After") if hasattr(response.headers, "get") else None
                    charset = response.headers.get_content_charset() or "utf-8"
                    payload = json.loads(response.read().decode(charset))
                validate_service_response(endpoint, params, payload)
            except HTTPError as error:
                status = error.code
                header = error.headers.get("Retry-After") if error.headers else None
                last_error = error
                retryable = status in TRANSIENT_HTTP_STATUS
                detail.update(error_type="HTTPError", error_message=str(error))
                error.close()
            except WikidataServiceError as error:
                last_error = error
                retryable = error.transient
                detail.update(error_type=type(error).__name__, error_message=str(error), api_error_code=error.code)
            except (URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError, UnicodeError) as error:
                last_error = error
                retryable = True
                detail.update(error_type=type(error).__name__, error_message=str(error))
            except ValueError as error:
                last_error = error
                retryable = False
                detail.update(error_type=type(error).__name__, error_message=str(error))
            else:
                detail.update(outcome="OK", http_status=status, wait_seconds=0, will_retry=False)
                self.record_attempt(detail, payload)
                # Cache writes are outside the transport retry block: a disk
                # failure must not cause duplicate HTTP requests.
                self.cache.set(endpoint, params, payload)
                time.sleep(REQUEST_DELAY_SECONDS)
                return payload
            server_delay = retry_after_seconds(header)
            fallback = RETRY_BASE_DELAY_SECONDS * 2**attempt
            if isinstance(last_error, WikidataServiceError) and last_error.transient:
                fallback = max(fallback, RETRY_SERVICE_DELAY_SECONDS * 2**attempt)
            delay = max(min(fallback, MAX_RETRY_WAIT_SECONDS), server_delay or 0.0)
            will_retry = retryable and attempt + 1 < MAX_RETRIES and delay <= MAX_RETRY_WAIT_SECONDS
            detail.update(outcome="RETRYABLE_ERROR" if retryable else "PERMANENT_ERROR", http_status=status,
                          retry_after_raw=header, retry_after_seconds=server_delay,
                          wait_seconds=delay if will_retry else 0, will_retry=will_retry)
            if retryable and not will_retry:
                self.cooldowns[endpoint] = time.monotonic() + delay
                detail["endpoint_deferred_seconds"] = delay
            self.record_attempt(detail, payload)
            if not retryable:
                raise ValueError(str(last_error)) from last_error
            if not will_retry:
                break
            time.sleep(delay)
        actual_attempts = sum(item["attempt"] > 0 for item in self.last_attempts)
        raise RuntimeError(f"Wikidata request failed after {actual_attempts} attempts: {endpoint}; {last_error}") from last_error


def get_json_safely(
    client: WikidataClient,
    endpoint: str,
    params: dict[str, str],
    operation: str,
) -> dict[str, Any] | None:
    try:
        payload = client.get_json(endpoint, params)
        if hasattr(client, "annotate_operation"):
            client.annotate_operation(operation)
        return payload
    except (RuntimeError, ValueError) as error:
        if hasattr(client, "annotate_operation"):
            client.annotate_operation(operation)
        print(f"WARNING: {operation} failed: {error}", file=sys.stderr)
        return None


def read_terms(path: Path) -> list[str]:
    terms = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    terms = [term for term in terms if term]
    if len(set(terms)) != len(terms):
        raise ValueError("Duplicate original terms in KEY file")
    return terms


def literal_pattern(text: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)", re.IGNORECASE)


def read_context_rows(
    path: Path,
    terms: Sequence[str],
) -> list[dict[str, str]]:
    required_columns = {"term", "occurrence_id", "matched_text", "context"}
    valid_terms = set(terms)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing Step 1 context columns: {sorted(missing)}")
        rows = [
            {key: str(value or "") for key, value in row.items()}
            for row in reader
        ]

    for row in rows:
        term = row["term"]
        matched_text = row["matched_text"]
        context = row["context"]
        if term not in valid_terms:
            raise ValueError(f"Context term is not an original KEY term: {term!r}")
        if not row["occurrence_id"] or not matched_text or not context:
            raise ValueError(f"Incomplete Step 1 context row for {term!r}")
        if not literal_pattern(matched_text).search(context):
            raise ValueError(f"matched_text is absent from context for {term!r}")
    return rows


def phrase_has_only_word_separators(
    context: str,
    word_spans: Sequence[tuple[int, int]],
    first_word: int,
    last_word: int,
) -> bool:
    return all(
        context[word_spans[index][1] : word_spans[index + 1][0]].isspace()
        for index in range(first_word, last_word)
    )


def proper_name_like(words: Sequence[str]) -> bool:
    content_words = [
        word for word in words if word.casefold() not in PROPER_NAME_CONNECTORS
    ]
    return bool(
        len(content_words) >= 2
        and all(word[0].isupper() for word in content_words if word)
    )


def generate_context_literal_queries(
    terms: Sequence[str],
    context_rows: Sequence[dict[str, str]],
    lemma_resolver=None,
) -> dict[str, list[ContextQueryEvidence]]:
    candidates: dict[
        str,
        dict[str, tuple[tuple[int, ...], ContextQueryEvidence]],
    ] = {term: {} for term in terms}
    rows_by_term: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in context_rows:
        rows_by_term[row["term"]].append(row)

    for term_index, term in enumerate(terms):
        lemma = (lemma_resolver or conservative_lemma)(term)
        existing_queries = {term.casefold()}
        if lemma:
            existing_queries.add(lemma.casefold())

        for occurrence_index, row in enumerate(rows_by_term[term]):
            context = row["context"]
            matched_text = row["matched_text"]
            word_matches = list(CONTEXT_WORD_PATTERN.finditer(context))
            word_spans = [(match.start(), match.end()) for match in word_matches]

            for matched_occurrence in literal_pattern(matched_text).finditer(context):
                covered_words = [
                    index
                    for index, (start, end) in enumerate(word_spans)
                    if start < matched_occurrence.end()
                    and end > matched_occurrence.start()
                ]
                if not covered_words:
                    continue
                base_first = covered_words[0]
                base_last = covered_words[-1]

                for added_left in range(MAX_CONTEXT_EXTENSION_WORDS + 1):
                    for added_right in range(MAX_CONTEXT_EXTENSION_WORDS + 1):
                        added_words = added_left + added_right
                        if not 1 <= added_words <= MAX_CONTEXT_EXTENSION_WORDS:
                            continue
                        first_word = base_first - added_left
                        last_word = base_last + added_right
                        if first_word < 0 or last_word >= len(word_spans):
                            continue
                        if last_word - first_word + 1 > MAX_CONTEXT_LITERAL_QUERY_WORDS:
                            continue
                        if not phrase_has_only_word_separators(
                            context, word_spans, first_word, last_word
                        ):
                            continue

                        phrase_start = word_spans[first_word][0]
                        phrase_end = word_spans[last_word][1]
                        query = context[phrase_start:phrase_end]
                        if query.casefold() in existing_queries:
                            continue
                        relative_match_start = matched_occurrence.start() - phrase_start
                        if (
                            query[
                                relative_match_start : relative_match_start
                                + len(matched_text)
                            ]
                            != matched_occurrence.group(0)
                        ):
                            continue

                        words = [
                            match.group(0)
                            for match in word_matches[first_word : last_word + 1]
                        ]
                        is_proper_name = proper_name_like(words)
                        outer_stopwords = int(
                            words[0].casefold() in PROPER_NAME_CONNECTORS
                        ) + int(words[-1].casefold() in PROPER_NAME_CONNECTORS)
                        if not is_proper_name and outer_stopwords:
                            continue
                        capitalized_words = sum(word[0].isupper() for word in words)
                        priority = (
                            0 if is_proper_name else 1,
                            outer_stopwords,
                            -capitalized_words,
                            added_words,
                            occurrence_index,
                            term_index,
                        )
                        evidence = ContextQueryEvidence(
                            term=term,
                            query=query,
                            occurrence_id=row["occurrence_id"],
                            matched_text=matched_text,
                            context=context,
                            document_id=row.get("document_id", ""),
                            term_id=row.get("term_id", ""),
                            occurrence_uid=row.get("occurrence_uid", ""),
                            query_match_start=matched_occurrence.start(),
                            query_match_end=matched_occurrence.end(),
                            query_start=phrase_start,
                            query_end=phrase_end,
                        )
                        normalized_query = query.casefold()
                        previous = candidates[term].get(normalized_query)
                        if previous is None or priority < previous[0]:
                            candidates[term][normalized_query] = (priority, evidence)

    return {
        term: [
            evidence
            for _priority, evidence in sorted(
                candidates[term].values(),
                key=lambda item: (
                    item[0],
                    item[1].query.casefold(),
                    item[1].query,
                    item[1].occurrence_id,
                ),
            )[
                :MAX_CONTEXT_LITERAL_QUERIES_PER_TERM
            ]
        ]
        for term in terms
    }


def validate_context_literal_queries(
    terms: Sequence[str],
    queries_by_term: dict[str, list[ContextQueryEvidence]],
) -> None:
    if set(queries_by_term) != set(terms):
        raise ValueError("Context-query terms do not match the original KEY terms")
    for term in terms:
        evidence_rows = queries_by_term[term]
        if len(evidence_rows) > MAX_CONTEXT_LITERAL_QUERIES_PER_TERM:
            raise ValueError(f"Too many context queries for {term!r}")
        normalized_queries: set[str] = set()
        for evidence in evidence_rows:
            if evidence.term != term:
                raise ValueError(f"Context-query term changed for {term!r}")
            if evidence.query not in evidence.context:
                raise ValueError(f"Invented context query for {term!r}: {evidence.query!r}")
            if evidence.matched_text not in evidence.query:
                raise ValueError(f"Context query lost matched_text for {term!r}")
            query_words = CONTEXT_WORD_PATTERN.findall(evidence.query)
            matched_words = CONTEXT_WORD_PATTERN.findall(evidence.matched_text)
            if not len(matched_words) < len(query_words) <= MAX_CONTEXT_LITERAL_QUERY_WORDS:
                raise ValueError(f"Invalid context-query size for {term!r}")
            if len(query_words) - len(matched_words) > MAX_CONTEXT_EXTENSION_WORDS:
                raise ValueError(f"Context query expands too far for {term!r}")
            normalized_query = evidence.query.casefold()
            if normalized_query in normalized_queries:
                raise ValueError(f"Duplicate context query for {term!r}")
            normalized_queries.add(normalized_query)


def chunks(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def qid_from_uri(uri: str) -> str | None:
    qid = str(uri).rsplit("/", 1)[-1]
    return qid if QID_PATTERN.fullmatch(qid) else None


def qid_sort_key(qid: str) -> int:
    return int(qid[1:])


@lru_cache(maxsize=1)
def english_morphology_pipeline() -> Any:
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def conservative_lemma(term: str) -> str | None:
    invariant_nouns = {"news", "series", "species"}
    irregular_nouns = {"people": "person"}
    replacements: list[tuple[int, int, str]] = []

    for token in english_morphology_pipeline()(term):
        if token.tag_ != "NNS":
            continue
        normalized_token = token.text.casefold()
        if normalized_token in invariant_nouns:
            continue
        lemma = irregular_nouns.get(normalized_token, token.lemma_.casefold())
        if not lemma.isalpha() or lemma == normalized_token:
            continue
        replacements.append((token.idx, token.idx + len(token.text), lemma))

    if not replacements:
        return None

    parts: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        parts.extend((term[cursor:start], replacement))
        cursor = end
    parts.append(term[cursor:])
    lemma = "".join(parts)
    return lemma if lemma.casefold() != term.casefold() else None


def search_specs(
    terms: Sequence[str],
    context_queries: dict[str, list[ContextQueryEvidence]],
    lemma_resolver=None,
) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for term in terms:
        specs.append((term, term, "original_entity_search"))
        lemma = (lemma_resolver or conservative_lemma)(term)
        if lemma:
            specs.append((term, lemma, "lemma_entity_search"))
        specs.extend(
            (term, evidence.query, "context_literal_entity_search")
            for evidence in context_queries[term]
        )
    return specs


def observed_acronym_expansions(term: str, context_rows: Sequence[dict]) -> list[dict]:
    """Only explicit parenthetical full forms whose initials match the acronym."""
    if not re.fullmatch(r"(?:[A-Z]\.){2,}|[A-Z]{2,}s?", term):
        return []
    acronym = term.replace(".", "").removesuffix("s")
    found = []
    for row in context_rows:
        if row["term"] != term:
            continue
        context = row["context"]
        for match in literal_pattern(term).finditer(context):
            ranges = []
            after = re.match(r"\s*\(([^()]+)\)", context[match.end():])
            if after:
                left, right = match.end() + after.start(1), match.end() + after.end(1)
                while left < right and context[left].isspace(): left += 1
                while right > left and context[right - 1].isspace(): right -= 1
                ranges.append((left, right))
            before = re.search(r"\(\s*$", context[:match.start()])
            if before and re.match(r"\s*\)", context[match.end():]):
                words_before = list(CONTEXT_WORD_PATTERN.finditer(context, 0, before.start()))
                if words_before and not context[words_before[-1].end():before.start()].strip():
                    for count in range(2, min(8, len(words_before)) + 1):
                        ranges.append((words_before[-count].start(), words_before[-1].end()))
            for left, right in ranges:
                words = list(CONTEXT_WORD_PATTERN.finditer(context, left, right))
                if not 2 <= len(words) <= 8 or words[0].start() != left or words[-1].end() != right:
                    continue
                if not phrase_has_only_word_separators(context, [item.span() for item in words], 0, len(words) - 1):
                    continue
                values = [item.group(0) for item in words]
                initials = "".join(value[0] for value in values if value.casefold() not in PROPER_NAME_CONNECTORS).upper()
                if initials != acronym or not proper_name_like(values):
                    continue
                found.append({"query": context[left:right], "reason": "EXPLICIT_PARENTHETICAL_ACRONYM_EXPANSION",
                              "occurrence_uid": row.get("occurrence_uid", ""), "document_id": row.get("document_id", ""),
                              "query_start": left, "query_end": right, "acronym_start": match.start(), "acronym_end": match.end()})
                break
    return found


def additional_search_specs(terms, context_rows, lemma_resolver) -> tuple[list[tuple[str, str, str]], list[dict]]:
    specs, evidence = [], []
    for term in terms:
        lemma = lemma_resolver(term)
        seen = {term} | ({lemma} if lemma else set())
        choices = [(item["query"], "acronym_expansion_entity_search", item)
                   for item in observed_acronym_expansions(term, context_rows)]
        for source in (term, lemma):
            if not source:
                continue
            canonical_surface = re.sub(r"(?<=\w)[\u2010\u2011\u2012\u2013\u2014](?=\w)", "-", source)
            canonical_surface = re.sub(r"(?<!\w)(?:[A-Z]\.){2,}(?!\w)", lambda match: match.group(0).replace(".", ""), canonical_surface)
            canonical_surface = re.sub(r"(?<!\w)([A-Z]{2,})s(?!\w)", r"\1", canonical_surface)
            for query, reason in ((canonical_surface, "DOTTED_OR_PLURAL_ACRONYM_OR_UNICODE_HYPHEN"),
                                  (re.sub(r"(?<=\w)-(?=\w)", " ", canonical_surface), "INTERWORD_HYPHEN_AS_SPACE")):
                choices.append((query, "orthographic_entity_search", {"source_query": source, "reason": reason}))
        count = 0
        for query, strategy, detail in choices:
            if query in seen or not query or len(CONTEXT_WORD_PATTERN.findall(query)) > MAX_CONTEXT_LITERAL_QUERY_WORDS:
                continue
            seen.add(query)
            specs.append((term, query, strategy))
            evidence.append({"term": term, "query": query, "strategy": strategy, **detail})
            count += 1
            if count >= MAX_ADDITIONAL_QUERIES_PER_TERM:
                break
    return specs, evidence


def supplementary_cirrus_specs(terms, lemma_resolver) -> list[tuple[str, str, str]]:
    # This is an ADDITIONAL retrieval budget, never an inventory-wide filter.
    # All primary article candidates are retained; no financial relevance is inferred.
    return [(term, sparql_literal(lemma_resolver(term) or term) + " -haswbstatement:P31=" + SCHOLARLY_ARTICLE_QID,
             "supplementary_non_article_cirrus_search") for term in terms]


def retrieve_supplementary_candidates(client, specs, candidates):
    audits = {}
    for term, query, strategy in specs:
        audit = EntitySearchAudit()
        audits[(term, query, strategy)] = audit
        continuation, seen = {}, set()
        while audit.results_seen < MAX_SUPPLEMENTARY_SEARCH_RESULTS_PER_TERM:
            params = {"action": "query", "list": "search", "srsearch": query, "srnamespace": "0", "srprop": "",
                      "srlimit": str(MAX_SUPPLEMENTARY_SEARCH_RESULTS_PER_TERM - audit.results_seen),
                      "format": "json", "maxlag": "5", **continuation}
            payload = get_json_safely(client, WIKIDATA_API_URL, params, f"Supplementary non-article search for {term!r}")
            if payload is None:
                break
            results = payload["query"]["search"]
            audit.pages += 1
            for result in results:
                qid = str(result.get("title", ""))
                audit.results_seen += 1
                if QID_PATTERN.fullmatch(qid):
                    audit.accepted_qids.add(qid)
                    audit.hits.setdefault(qid, {"position": audit.results_seen, "match_type": "cirrus_supplement"})
                    add_candidate(candidates, term, qid, query, strategy)
            raw_next = payload.get("continue")
            if raw_next is None:
                audit.completed = True
                break
            if audit.results_seen >= MAX_SUPPLEMENTARY_SEARCH_RESULTS_PER_TERM:
                audit.stopped_by_limit = True
                break
            if (not isinstance(raw_next, dict) or set(raw_next) - {"continue", "sroffset"}
                    or not results or not raw_next):
                break
            continuation = {key: str(value) for key, value in raw_next.items()}
            marker = json.dumps(continuation, sort_keys=True)
            if marker in seen:
                break
            seen.add(marker)
    return audits


def add_candidate(
    candidates: dict[str, dict[str, list[tuple[str, str]]]],
    term: str,
    qid: str,
    query: str,
    strategy: str,
) -> None:
    if not QID_PATTERN.fullmatch(qid):
        return
    provenance = (query, strategy)
    origins = candidates[term].setdefault(qid, [])
    if provenance not in origins:
        origins.append(provenance)


def retrieve_entity_search_candidates(
    client: WikidataClient,
    specs: Sequence[tuple[str, str, str]],
    candidates: dict[str, dict[str, list[tuple[str, str]]]],
) -> dict[tuple[str, str, str], EntitySearchAudit]:
    audits: dict[tuple[str, str, str], EntitySearchAudit] = {}

    for term, query, strategy in specs:
        audit = EntitySearchAudit()
        audits[(term, query, strategy)] = audit
        continuation: str | None = None
        seen_continuations: set[str] = set()
        max_results = (
            MAX_CONTEXT_ENTITY_SEARCH_RESULTS_PER_QUERY
            if strategy == "context_literal_entity_search"
            else MAX_ADDITIONAL_ENTITY_SEARCH_RESULTS_PER_QUERY
            if strategy in {"orthographic_entity_search", "acronym_expansion_entity_search"}
            else MAX_ENTITY_SEARCH_RESULTS_PER_QUERY
        )

        while audit.results_seen < max_results:
            remaining = max_results - audit.results_seen
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "uselang": "en",
                "type": "item",
                "limit": str(min(ENTITY_SEARCH_LIMIT, remaining)),
                "format": "json",
                "maxlag": "5",
            }
            if continuation is not None:
                params["continue"] = continuation

            payload = get_json_safely(
                client,
                WIKIDATA_API_URL,
                params,
                f"EntitySearch for {query!r}",
            )
            if payload is None:
                break

            raw_results = payload.get("search", [])
            results = raw_results if isinstance(raw_results, list) else []
            first_position = audit.results_seen + 1
            audit.pages += 1
            audit.results_seen += len(results)

            for offset, result in enumerate(results):
                qid = str(result.get("id", ""))
                if not QID_PATTERN.fullmatch(qid):
                    continue
                match = result.get("match", {})
                if not isinstance(match, dict):
                    match = {}
                audit.hits.setdefault(
                    qid,
                    {
                        "position": first_position + offset,
                        "match_type": str(match.get("type", "")),
                        "match_language": str(match.get("language", "")),
                        "match_text": str(match.get("text", "")),
                    },
                )

                description = str(result.get("description", "")).casefold()
                if "disambiguation page" in description:
                    continue
                audit.accepted_qids.add(qid)
                add_candidate(candidates, term, qid, query, strategy)

            next_continuation = payload.get("search-continue")
            if next_continuation is None:
                audit.completed = True
                break
            if audit.results_seen >= max_results:
                audit.stopped_by_limit = True
                break

            continuation = str(next_continuation)
            if not results or continuation in seen_continuations:
                print(
                    f"WARNING: invalid EntitySearch continuation for {query!r}: "
                    f"{continuation!r}",
                    file=sys.stderr,
                )
                break
            seen_continuations.add(continuation)

    return audits


def sparql_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def sparql_surface_variants(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            (query, query.casefold(), query.capitalize(), query.title())
        )
    )


def retrieve_exact_sparql_candidates(
    client: WikidataClient,
    specs: Sequence[tuple[str, str, str]],
    candidates: dict[str, dict[str, list[tuple[str, str]]]],
) -> dict[tuple[str, str, str], SparqlSearchAudit]:
    audits = {spec: SparqlSearchAudit() for spec in specs}
    indexed_specs = list(enumerate(specs))
    for batch in chunks(indexed_specs, SPARQL_SEARCH_BATCH_SIZE):
        values = "\n".join(
            f"({sparql_literal(str(index))} {sparql_literal(surface)}@en)"
            for index, (_term, query, _strategy) in batch
            for surface in sparql_surface_variants(query)
        )
        query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?spec_index ?item WHERE {{
  VALUES (?spec_index ?surface) {{
    {values}
  }}
  VALUES ?surface_predicate {{ rdfs:label skos:altLabel }}
  ?item ?surface_predicate ?surface .
  FILTER(REGEX(STR(?item), "/Q[1-9][0-9]*$"))
}}
""".strip()
        payload = get_json_safely(
            client,
            WIKIDATA_SPARQL_URL,
            {"query": query, "format": "json"},
            "SPARQL exact-label candidate search",
        )
        if payload is None:
            continue
        for _spec_index, spec in batch:
            audits[spec].completed = True

        exact_results: set[tuple[int, str]] = set()
        for binding in payload.get("results", {}).get("bindings", []):
            raw_index = str(binding.get("spec_index", {}).get("value", ""))
            qid = qid_from_uri(binding.get("item", {}).get("value", ""))
            if raw_index.isdigit() and int(raw_index) < len(specs) and qid:
                exact_results.add((int(raw_index), qid))

        for spec_index, qid in sorted(
            exact_results,
            key=lambda result: (result[0], qid_sort_key(result[1])),
        ):
            term, search_query, entity_strategy = specs[spec_index]
            audits[(term, search_query, entity_strategy)].qids.add(qid)
            sparql_strategy = entity_strategy.replace(
                "_entity_search", "_sparql_exact"
            )
            add_candidate(
                candidates,
                term,
                qid,
                search_query,
                sparql_strategy,
            )

    return audits


def fetch_entities(
    client: WikidataClient,
    qids: Sequence[str],
    properties: str,
    languages: str | None = None,
) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for batch in chunks(qids, ENTITY_BATCH_SIZE):
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": properties,
            "format": "json",
            "maxlag": "5",
        }
        if languages is not None:
            params["languages"] = languages
        payload = get_json_safely(
            client,
            WIKIDATA_API_URL,
            params,
            f"entity metadata batch ({properties})",
        )
        if payload is None:
            continue
        for qid, entity in payload.get("entities", {}).items():
            if QID_PATTERN.fullmatch(qid) and isinstance(entity, dict):
                entities[qid] = entity
    return entities


def fetch_relationships(
    client: WikidataClient,
    qids: Sequence[str],
) -> dict[str, dict[str, set[str]]]:
    relationships = {
        qid: {column: set() for column in RELATION_PROPERTIES.values()}
        for qid in qids
    }
    property_uri_prefix = "http://www.wikidata.org/prop/direct/"

    for batch in chunks(qids, SPARQL_BATCH_SIZE):
        item_values = " ".join(f"wd:{qid}" for qid in batch)
        query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?item ?property ?value WHERE {{
  VALUES ?item {{ {item_values} }}
  VALUES ?property {{ wdt:P31 wdt:P279 wdt:P1343 }}
  ?item ?property ?value .
  FILTER(STRSTARTS(STR(?value), "http://www.wikidata.org/entity/Q"))
}}
""".strip()
        payload = get_json_safely(
            client,
            WIKIDATA_SPARQL_URL,
            {"query": query, "format": "json"},
            "SPARQL P31/P279/P1343 batch",
        )
        if payload is None:
            fallback_entities = fetch_entities(client, list(batch), "claims")
            for item_qid, entity in fallback_entities.items():
                for property_id, column in RELATION_PROPERTIES.items():
                    for claim in entity.get("claims", {}).get(property_id, []):
                        if claim.get("rank") == "deprecated":
                            continue
                        value = (
                            claim.get("mainsnak", {})
                            .get("datavalue", {})
                            .get("value", {})
                        )
                        value_qid = str(value.get("id", ""))
                        if QID_PATTERN.fullmatch(value_qid):
                            relationships[item_qid][column].add(value_qid)
            continue
        for binding in payload.get("results", {}).get("bindings", []):
            item_qid = qid_from_uri(binding.get("item", {}).get("value", ""))
            value_qid = qid_from_uri(binding.get("value", {}).get("value", ""))
            property_uri = str(binding.get("property", {}).get("value", ""))
            property_id = property_uri.removeprefix(property_uri_prefix)
            column = RELATION_PROPERTIES.get(property_id)
            if item_qid in relationships and value_qid and column:
                relationships[item_qid][column].add(value_qid)

    return relationships


def english_value(entity: dict[str, Any], property_name: str) -> str:
    return str(entity.get(property_name, {}).get("en", {}).get("value", "")).strip()


def english_aliases(entity: dict[str, Any]) -> list[str]:
    aliases = entity.get("aliases", {}).get("en", [])
    values = [str(alias.get("value", "")).strip() for alias in aliases]
    return list(dict.fromkeys(value for value in values if value))


def other_descriptions(entity: dict[str, Any]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for language, payload in entity.get("descriptions", {}).items():
        value = str(payload.get("value", "")).strip()
        if language != "en" and value:
            descriptions[str(language)] = value
    return dict(sorted(descriptions.items()))


def is_disambiguation(
    qid: str,
    english_entities: dict[str, dict[str, Any]],
    relationships: dict[str, dict[str, set[str]]],
) -> bool:
    description = english_value(english_entities.get(qid, {}), "descriptions")
    instance_of = relationships.get(qid, {}).get("p31", set())
    return bool(
        DISAMBIGUATION_CLASSES & instance_of
        or "disambiguation page" in description.casefold()
    )


def relation_json(qids: set[str], labels: dict[str, str]) -> str:
    values = [
        {"qid": qid, "label_en": labels.get(qid, "")}
        for qid in sorted(qids, key=qid_sort_key)
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def build_rows(
    terms: Sequence[str],
    candidates: dict[str, dict[str, list[tuple[str, str]]]],
    english_entities: dict[str, dict[str, Any]],
    multilingual_entities: dict[str, dict[str, Any]],
    relationships: dict[str, dict[str, set[str]]],
    relation_labels: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for term in terms:
        for qid, provenance in candidates[term].items():
            if is_disambiguation(qid, english_entities, relationships):
                continue
            english_entity = english_entities.get(qid, {})
            other_languages = other_descriptions(multilingual_entities.get(qid, {}))
            queries = [query for query, _strategy in provenance]
            strategies = [strategy for _query, strategy in provenance]
            relations = relationships.get(
                qid, {column: set() for column in RELATION_PROPERTIES.values()}
            )
            rows.append(
                {
                    "term": term,
                    "qid": qid,
                    "label_en": english_value(english_entity, "labels"),
                    "aliases_en": json.dumps(
                        english_aliases(english_entity),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "description_en": english_value(
                        english_entity, "descriptions"
                    ),
                    "descriptions_other_languages": json.dumps(
                        other_languages,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "p31": relation_json(relations["p31"], relation_labels),
                    "p279": relation_json(relations["p279"], relation_labels),
                    "p1343": relation_json(relations["p1343"], relation_labels),
                    "search_queries": json.dumps(
                        queries, ensure_ascii=False, separators=(",", ":")
                    ),
                    "search_strategies": json.dumps(
                        strategies, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
    return rows


def validate_rows(
    rows: Sequence[dict[str, str]],
    terms: Sequence[str],
    context_queries: dict[str, list[ContextQueryEvidence]],
    lemma_resolver=None,
    *, additional_specs: Sequence[tuple[str, str, str]] = (),
) -> None:
    valid_terms = set(terms)
    valid_context_queries = {
        term: {evidence.query for evidence in evidence_rows}
        for term, evidence_rows in context_queries.items()
    }
    seen_candidates: set[tuple[str, str]] = set()
    allowed_additional = set(additional_specs) | {
        (term, query, strategy.replace("_entity_search", "_sparql_exact"))
        for term, query, strategy in additional_specs if strategy.endswith("_entity_search")}
    forbidden_columns = {
        "selected",
        "winner",
        "score",
        "similarity",
        "final_qid",
        "definition_generated",
        "terminlink",
        "defterm",
    }
    if {column.casefold() for column in OUTPUT_COLUMNS} & forbidden_columns:
        raise ValueError("A decision column leaked into the inventory")

    for row in rows:
        if tuple(row) != OUTPUT_COLUMNS:
            raise ValueError("Unexpected output columns or column order")
        term = row["term"]
        qid = row["qid"]
        if term not in valid_terms:
            raise ValueError(f"Output term is not an original KEY term: {term!r}")
        if not QID_PATTERN.fullmatch(qid):
            raise ValueError(f"Invalid entity QID: {qid!r}")
        candidate_key = (term, qid)
        if candidate_key in seen_candidates:
            raise ValueError(f"Duplicate QID inside term: {candidate_key}")
        seen_candidates.add(candidate_key)

        queries = json.loads(row["search_queries"])
        strategies = json.loads(row["search_strategies"])
        if not queries or len(queries) != len(strategies):
            raise ValueError(f"Incomplete retrieval provenance for {candidate_key}")
        provenance = list(zip(queries, strategies))
        if any(not query or not strategy for query, strategy in provenance):
            raise ValueError(f"Empty retrieval provenance for {candidate_key}")
        if len(provenance) != len(set(provenance)):
            raise ValueError(f"Duplicate retrieval provenance for {candidate_key}")
        lemma = (lemma_resolver or conservative_lemma)(term)
        for query, strategy in provenance:
            if strategy in {"original_entity_search", "original_sparql_exact"}:
                if query != term:
                    raise ValueError(f"Original-query provenance changed for {candidate_key}")
            elif strategy in {"lemma_entity_search", "lemma_sparql_exact"}:
                if lemma is None or query != lemma:
                    raise ValueError(f"Invalid lemma provenance for {candidate_key}")
            elif strategy in {
                "context_literal_entity_search",
                "context_literal_sparql_exact",
            }:
                if query not in valid_context_queries.get(term, set()):
                    raise ValueError(
                        f"Context provenance is not literal for {candidate_key}"
                    )
            elif strategy in {"orthographic_entity_search", "orthographic_sparql_exact", "acronym_expansion_entity_search",
                              "acronym_expansion_sparql_exact", "supplementary_non_article_cirrus_search"}:
                if (term, query, strategy) not in allowed_additional:
                    raise ValueError(f"Unjustified additional query provenance for {candidate_key}")
            else:
                raise ValueError(f"Unknown search strategy for {candidate_key}: {strategy}")

        descriptions = json.loads(row["descriptions_other_languages"])
        if not isinstance(descriptions, dict) or "en" in descriptions:
            raise ValueError(f"Invalid multilingual descriptions for {candidate_key}")
        for column in RELATION_PROPERTIES.values():
            values = json.loads(row[column])
            if any(not QID_PATTERN.fullmatch(value.get("qid", "")) for value in values):
                raise ValueError(f"Invalid related QID in {column} for {candidate_key}")


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def has_useful_description(text: str) -> bool:
    return len(USEFUL_WORD_PATTERN.findall(text)) > 1


def print_context_query_summary(
    terms: Sequence[str],
    context_queries: dict[str, list[ContextQueryEvidence]],
) -> None:
    print("CONTEXT_LITERAL_QUERIES")
    for term in terms:
        print(
            f"{term}: "
            + json.dumps(
                [
                    {
                        "query": evidence.query,
                        "occurrence_id": evidence.occurrence_id,
                    }
                    for evidence in context_queries[term]
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


def print_summary(rows: Sequence[dict[str, str]], terms: Sequence[str], output_path: Path | None = None) -> None:
    counts = Counter(row["term"] for row in rows)
    missing_terms = [term for term in terms if counts[term] == 0]
    useful_english = 0
    useful_other = 0
    no_useful_description = 0

    for row in rows:
        if has_useful_description(row["description_en"]):
            useful_english += 1
            continue
        descriptions = json.loads(row["descriptions_other_languages"])
        if any(has_useful_description(text) for text in descriptions.values()):
            useful_other += 1
        else:
            no_useful_description += 1

    print(f"TERMS_PROCESSED={len(terms)}")
    print(f"TOTAL_UNIQUE_CANDIDATES={len(rows)}")
    for term in terms:
        print(f"{term}: {counts[term]}")
    print(f"NO_CANDIDATE_TERMS={json.dumps(missing_terms, ensure_ascii=False)}")
    print(f"USEFUL_DESCRIPTION_EN={useful_english}")
    print(f"NO_USEFUL_EN_WITH_USEFUL_OTHER={useful_other}")
    print(f"NO_USEFUL_DESCRIPTION={no_useful_description}")
    print("FIRST_20_ROWS")
    preview_columns = (
        "term",
        "qid",
        "label_en",
        "description_en",
        "search_strategies",
    )
    for row in rows[:20]:
        preview = {column: row[column] for column in preview_columns}
        print(json.dumps(preview, ensure_ascii=False, separators=(",", ":")))
    if output_path is not None:
        print(f"OUTPUT_CSV={output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    # The reusable integration layer handles inputs, explicit search policy,
    # cache/transport trace, associations and immutable run outputs.
    from terminlink_stage02 import run_cli
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
