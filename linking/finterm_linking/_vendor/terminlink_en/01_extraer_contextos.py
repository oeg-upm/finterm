"""Entity-linker stage 01: locate supplied terms in user text and prepare contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from terminlink_inputs import CONTRACT_VERSION, DocumentInput, supplied_terms


MAX_CONTEXTS_PER_TERM = 10
WORDS_BEFORE = 20
SCHEMA_VERSION = "terminlink-stage01-contexts-v3"
TARGET_OPEN = "[TARGET]"
TARGET_CLOSE = "[/TARGET]"

HISTORICAL_COLUMNS = (
    "term",
    "occurrence_id",
    "matched_text",
    "context",
    "normalized_start",
    "original_start",
)
OUTPUT_COLUMNS = HISTORICAL_COLUMNS + (
    "document_id",
    "occurrence_uid",
    "context_target_start",
    "context_target_end",
    "context_marked",
    "normalized_end",
    "original_end",
    "context_normalized_start",
    "context_normalized_end",
    "term_id",
    "language",
    "term_source",
    "term_provenance_json",
)

LEXICAL_WORD_PATTERN = re.compile(r"\b\w+(?:[-'\u2019]\w+)*\b", re.UNICODE)
PERIOD_POLICY = "DECIMAL_AND_ABBREVIATION_PERIOD_HEURISTICS_V1"
CONTINUING_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "vs", "cf", "e.g", "i.e"})
CONTEXTUAL_ABBREVIATIONS = frozenset({
    "inc", "ltd", "corp", "co", "etc", "approx", "no", "nos", "fig", "figs",
    "art", "arts", "sec", "secs", "para", "paras", "pp", "vol", "dept",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})
SENTENCE_STARTERS = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "It", "They", "We", "He", "She",
    "However", "Nevertheless", "Meanwhile", "Therefore", "Moreover", "Furthermore", "Subsequently",
    "In", "On", "At", "As", "For", "But", "Yet",
})
XML_FILENAME_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*\.xml$",
    re.IGNORECASE,
)
PAGE_NUMBER_PATTERN = re.compile(r"^[A-Z]{1,3}\s+\d+(?:/\d+)+$")
HEADER_DATE_PATTERN = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")
JOURNAL_HEADER_PATTERN = re.compile(
    r"^Official\s+Journal\s+of\s+the\s+European\s+(?:Union|Communities)$",
    re.IGNORECASE,
)
EU_LANGUAGE_CODES = frozenset(
    {
        "BG",
        "CS",
        "DA",
        "DE",
        "EL",
        "EN",
        "ES",
        "ET",
        "FI",
        "FR",
        "GA",
        "HR",
        "HU",
        "IT",
        "LT",
        "LV",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SL",
        "SV",
    }
)


@dataclass(frozen=True)
class NormalizedDocument:
    text: str
    normalized_to_original: tuple[int, ...]


@dataclass(frozen=True)
class ValidationSummary:
    max_words_before: int
    words_before_by_occurrence: dict[str, int]


def canonicalize_newlines(original_text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    offsets: list[int] = []
    index = 0

    while index < len(original_text):
        character = original_text[index]
        if character == "\r":
            characters.append("\n")
            offsets.append(index)
            index += (
                2
                if index + 1 < len(original_text)
                and original_text[index + 1] == "\n"
                else 1
            )
            continue
        characters.append(character)
        offsets.append(index)
        index += 1

    return "".join(characters), offsets


def is_structural_noise_line(line: str, is_first_nonempty: bool) -> bool:
    value = line.strip()
    if not value:
        return False
    if is_first_nonempty and XML_FILENAME_PATTERN.fullmatch(value):
        return True
    return bool(
        JOURNAL_HEADER_PATTERN.fullmatch(value)
        or PAGE_NUMBER_PATTERN.fullmatch(value)
        or HEADER_DATE_PATTERN.fullmatch(value)
        or value in EU_LANGUAGE_CODES
    )


def contains_any_literal_term(text: str, terms: Sequence[str]) -> bool:
    return any(literal_pattern(term).search(text) for term in terms)


def remove_structural_lines(
    text: str,
    offsets: Sequence[int],
    terms: Sequence[str],
) -> tuple[list[str], list[int]]:
    kept_characters: list[str] = []
    kept_offsets: list[int] = []
    line_start = 0
    seen_nonempty = False

    while line_start < len(text):
        newline_position = text.find("\n", line_start)
        if newline_position == -1:
            line_end = len(text)
            next_line_start = len(text)
        else:
            line_end = newline_position
            next_line_start = newline_position + 1

        line = text[line_start:line_end]
        is_first_nonempty = bool(line.strip()) and not seen_nonempty
        is_noise = is_structural_noise_line(line, is_first_nonempty)
        if is_noise and contains_any_literal_term(line, terms):
            is_noise = False
        if line.strip():
            seen_nonempty = True

        if not is_noise:
            kept_characters.extend(text[line_start:next_line_start])
            kept_offsets.extend(offsets[line_start:next_line_start])
        line_start = next_line_start

    return kept_characters, kept_offsets


def validate_structural_filter_preserves_terms(
    text_before_filtering: str,
    text_after_filtering: str,
    terms: Sequence[str],
) -> None:
    lost_terms = [
        term
        for term in terms
        if literal_pattern(term).search(text_before_filtering)
        and not literal_pattern(term).search(text_after_filtering)
    ]
    if lost_terms:
        raise ValueError(
            "Structural filtering removed every literal occurrence of KEY terms: "
            f"{lost_terms}"
        )


def collapse_whitespace(
    characters: Sequence[str],
    offsets: Sequence[int],
) -> tuple[list[str], list[int]]:
    collapsed_characters: list[str] = []
    collapsed_offsets: list[int] = []
    pending_space_offset: int | None = None

    for character, original_offset in zip(characters, offsets):
        if character.isspace():
            if collapsed_characters and pending_space_offset is None:
                pending_space_offset = original_offset
            continue

        if pending_space_offset is not None:
            collapsed_characters.append(" ")
            collapsed_offsets.append(pending_space_offset)
            pending_space_offset = None
        collapsed_characters.append(character)
        collapsed_offsets.append(original_offset)

    return collapsed_characters, collapsed_offsets


def tighten_punctuation_spacing(
    characters: Sequence[str],
    offsets: Sequence[int],
) -> tuple[list[str], list[int]]:
    tightened_characters: list[str] = []
    tightened_offsets: list[int] = []

    for index, (character, original_offset) in enumerate(zip(characters, offsets)):
        if character == " ":
            previous_character = characters[index - 1] if index else ""
            next_character = (
                characters[index + 1] if index + 1 < len(characters) else ""
            )
            if previous_character in "([{\u00ab" or next_character in ".,;:!?%)]}\u00bb":
                continue
        tightened_characters.append(character)
        tightened_offsets.append(original_offset)

    return tightened_characters, tightened_offsets


def normalize_text(original_text: str, terms: Sequence[str]) -> NormalizedDocument:
    canonical_text, canonical_offsets = canonicalize_newlines(original_text)
    filtered_characters, filtered_offsets = remove_structural_lines(
        canonical_text, canonical_offsets, terms
    )
    validate_structural_filter_preserves_terms(
        canonical_text, "".join(filtered_characters), terms
    )
    collapsed_characters, collapsed_offsets = collapse_whitespace(
        filtered_characters, filtered_offsets
    )
    normalized_characters, normalized_offsets = tighten_punctuation_spacing(
        collapsed_characters, collapsed_offsets
    )
    normalized_text = "".join(normalized_characters)
    if len(normalized_text) != len(normalized_offsets):
        raise AssertionError("Normalized offset map is inconsistent")
    if any(
        previous >= current
        for previous, current in zip(normalized_offsets, normalized_offsets[1:])
    ):
        raise AssertionError("Normalized offsets must be strictly increasing")
    for character, original_offset in zip(normalized_text, normalized_offsets):
        if character != " " and original_text[original_offset] != character:
            raise AssertionError("Normalized character lost its original offset")
    return NormalizedDocument(normalized_text, tuple(normalized_offsets))


def read_terms(path: Path) -> list[str]:
    terms = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return [term for term in terms if term]


def literal_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def lexical_word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in LEXICAL_WORD_PATTERN.finditer(text)]


def is_context_terminating_period(text: str, position: int) -> bool:
    """Classify dots only; retain the historical 20-word/next-period method.

    This deterministic rule handles numeric dots and common abbreviations. It
    is deliberately not a sentence tokenizer or a universal abbreviation model.
    """
    following = position + 1
    if following < len(text) and text[following].isdigit():
        return False  # 0.5, .5, dates and dotted numeric references.
    if (position > 0 and text[position - 1].isalpha() and following + 1 < len(text)
            and text[following].isalpha() and text[following + 1] == "."):
        return False  # An internal dot in U.S., U.K., e.g., p.m., ...

    start = position
    while start and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    word = text[start:position]
    while following < len(text) and (text[following].isspace() or text[following] in ")]}'\"\u2019\u201d"):
        following += 1
    if following == len(text):
        return True
    if word.casefold() in CONTINUING_ABBREVIATIONS:
        return False
    acronym = re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]", word, re.UNICODE) is not None
    initial = len(word) == 1 and word.isupper()
    if word.casefold() in CONTEXTUAL_ABBREVIATIONS or acronym or initial:
        next_word = LEXICAL_WORD_PATTERN.match(text, following)
        return next_word is not None and next_word.group(0) in SENTENCE_STARTERS
    return True


def following_context_end(text: str, occurrence_end: int) -> int:
    position = text.find(".", occurrence_end)
    while position != -1:
        if is_context_terminating_period(text, position):
            return position + 1
        position = text.find(".", position + 1)
    return len(text)


def context_bounds(
    normalized_text: str,
    occurrence_start: int,
    occurrence_end: int,
    word_spans: Sequence[tuple[int, int]],
    word_ends: Sequence[int],
) -> tuple[int, int]:
    previous_word_count = bisect_right(word_ends, occurrence_start)
    if previous_word_count <= WORDS_BEFORE:
        context_start = 0
    else:
        context_start = word_spans[previous_word_count - WORDS_BEFORE][0]

    context_end = following_context_end(normalized_text, occurrence_end)
    return context_start, context_end


def occurrence_prefix(term: str) -> str:
    prefix = re.sub(r"[^0-9A-Za-z]+", "_", term).strip("_")
    return prefix or "term"


def extract_rows(
    terms: Sequence[str],
    document: NormalizedDocument,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    word_spans = lexical_word_spans(document.text)
    word_ends = [end for _, end in word_spans]

    for term in terms:
        seen_contexts: set[str] = set()
        term_rows: list[dict[str, str | int]] = []

        for match in literal_pattern(term).finditer(document.text):
            context_start, context_end = context_bounds(
                document.text,
                match.start(),
                match.end(),
                word_spans,
                word_ends,
            )
            context = document.text[context_start:context_end]
            if context in seen_contexts:
                continue

            seen_contexts.add(context)
            term_rows.append(
                {
                    "term": term,
                    "occurrence_id": f"{occurrence_prefix(term)}_{len(term_rows) + 1}",
                    "matched_text": match.group(0),
                    "context": context,
                    "normalized_start": match.start(),
                    "original_start": document.normalized_to_original[match.start()],
                }
            )
            if len(term_rows) == MAX_CONTEXTS_PER_TERM:
                break

        rows.extend(term_rows)

    return rows


def validate_rows(
    rows: Sequence[dict[str, str | int]],
    terms: Sequence[str],
    document: NormalizedDocument,
    original_text: str,
) -> ValidationSummary:
    valid_terms = set(terms)
    contexts_by_term: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    occurrence_keys: set[tuple[str, str]] = set()
    words_before_by_occurrence: dict[str, int] = {}
    word_spans = lexical_word_spans(document.text)
    word_ends = [end for _, end in word_spans]

    for row in rows:
        term = str(row["term"])
        occurrence_id = str(row["occurrence_id"])
        matched_text = str(row["matched_text"])
        context = str(row["context"])
        normalized_start = int(row["normalized_start"])
        original_start = int(row["original_start"])

        if term not in valid_terms:
            raise ValueError(f"Unknown output term: {term!r}")
        occurrence_key = (term, occurrence_id)
        if occurrence_key in occurrence_keys:
            raise ValueError(f"Duplicate occurrence_id: {occurrence_id!r}")
        if not context:
            raise ValueError(f"Empty context for {term!r}")
        if matched_text not in context:
            raise ValueError(f"matched_text is absent from context for {term!r}")
        if (
            document.text[normalized_start : normalized_start + len(matched_text)]
            != matched_text
        ):
            raise ValueError(f"Invalid normalized_start for {term!r}")
        if document.normalized_to_original[normalized_start] != original_start:
            raise ValueError(f"Invalid original_start mapping for {term!r}")
        if not 0 <= original_start < len(original_text):
            raise ValueError(f"original_start is outside the original TXT for {term!r}")
        if original_text[original_start] != matched_text[0]:
            raise ValueError(f"original_start does not identify the matched text for {term!r}")

        match = literal_pattern(term).match(document.text, normalized_start)
        if match is None or match.group(0) != matched_text:
            raise ValueError(f"normalized_start is not a literal term match for {term!r}")

        expected_start, expected_end = context_bounds(
            document.text,
            normalized_start,
            normalized_start + len(matched_text),
            word_spans,
            word_ends,
        )
        if context != document.text[expected_start:expected_end]:
            raise ValueError(f"Context boundaries are invalid for {term!r}")

        relative_start = normalized_start - expected_start
        if (
            context[relative_start : relative_start + len(matched_text)]
            != matched_text
        ):
            raise ValueError(f"matched_text is not at its expected context offset for {term!r}")
        # Count the same complete lexical spans used by context_bounds. A literal
        # match may start inside a hyphenated word (resident in non-resident):
        # regexing a sliced prefix would wrongly count "non-" as a 21st word.
        words_before = bisect_right(word_ends, normalized_start) - bisect_right(
            word_ends, expected_start
        )
        if words_before > WORDS_BEFORE:
            raise ValueError(
                f"More than {WORDS_BEFORE} lexical words before {occurrence_id!r}: "
                f"{words_before}"
            )

        required_end = following_context_end(document.text, normalized_start + len(matched_text))
        if expected_start + len(context) != required_end:
            raise ValueError(f"Context does not end at the first following terminating period for {term!r}")
        if context in contexts_by_term[term]:
            raise ValueError(f"Duplicate exact context for {term!r}")

        occurrence_keys.add(occurrence_key)
        contexts_by_term[term].add(context)
        counts[term] += 1
        words_before_by_occurrence[occurrence_id] = words_before

    for term, count in counts.items():
        if count > MAX_CONTEXTS_PER_TERM:
            raise ValueError(f"More than 10 contexts for {term!r}")

    return ValidationSummary(
        max(words_before_by_occurrence.values(), default=0),
        words_before_by_occurrence,
    )


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jsonl_line(value: object) -> str:
    # Preserve decoded values while keeping one physical record for consumers
    # using splitlines(). Identity/provenance hashes still use canonical().
    serialized = canonical(value)
    for separator in ("\u0085", "\u2028", "\u2029"):
        serialized = serialized.replace(separator, f"\\u{ord(separator):04x}")
    return serialized + "\n"


def decode_input(payload: bytes) -> str:
    # Match the historical read_text(encoding="utf-8-sig") offset convention.
    with io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline=None) as stream:
        return stream.read()


def enrich_rows(
    rows: Sequence[dict[str, str | int]], document_id: str, document: NormalizedDocument,
) -> list[dict[str, str | int]]:
    """Keep all six legacy fields; add a global key and mark ONLY the target span."""
    enriched = []
    spans = lexical_word_spans(document.text)
    ends = [end for _, end in spans]
    for source in rows:
        start = int(source["normalized_start"])
        end = start + len(str(source["matched_text"]))
        context_start, context_end = context_bounds(document.text, start, end, spans, ends)
        context = str(source["context"])
        if TARGET_OPEN in context or TARGET_CLOSE in context:
            raise ValueError("Source context contains reserved target markers")
        relative_start, relative_end = start - context_start, end - context_start
        if context[relative_start:relative_end] != source["matched_text"]:
            raise ValueError("Target span does not identify matched_text")
        identity = {"document_id": document_id, "term": source["term"],
                    "normalized_start": start, "normalized_end": end}
        row = dict(source)
        row.update({
            "document_id": document_id,
            "occurrence_uid": "occ_" + sha256(canonical(identity).encode("utf-8")).lower(),
            "context_target_start": relative_start, "context_target_end": relative_end,
            "context_marked": context[:relative_start] + TARGET_OPEN + context[relative_start:relative_end]
                              + TARGET_CLOSE + context[relative_end:],
            "normalized_end": end,
            "original_end": document.normalized_to_original[end - 1] + 1,
            "context_normalized_start": context_start, "context_normalized_end": context_end,
        })
        enriched.append(row)
    if len({row["occurrence_uid"] for row in enriched}) != len(enriched):
        raise ValueError("Duplicate global occurrence UID")
    return enriched


def audit_mentions(rows: Sequence[dict], document_id: str, document: NormalizedDocument,
                   term_records: Sequence) -> dict:
    """Preserve literal positions without changing distinct CSV contexts or IDs.

    Coordinates refer to the unmarked context; a duplicate mention can be marked
    using its own target span and the representative CSV row's context. Matches
    beyond the ten-context cap keep spans, but do not create extra context rows.
    """
    representatives = {(str(row["term"]), str(row["context"])): row for row in rows}
    spans = lexical_word_spans(document.text)
    ends = [end for _, end in spans]
    mentions = []
    counts: Counter[str] = Counter()
    for record in term_records:
        term = record.term
        for match in literal_pattern(term).finditer(document.text):
            start, end = match.span()
            context_start, context_end = context_bounds(document.text, start, end, spans, ends)
            representative = representatives.get((term, document.text[context_start:context_end]))
            selection = ("CONTEXT_LIMIT" if representative is None else
                         "REPRESENTATIVE" if int(representative["normalized_start"]) == start else "DUPLICATE_CONTEXT")
            identity = {"document_id": document_id, "term": term,
                        "normalized_start": start, "normalized_end": end}
            mentions.append({
                "term": term, "term_id": record.term_id,
                "occurrence_uid": "occ_" + sha256(canonical(identity).encode("utf-8")).lower(),
                "matched_text": match.group(0), "normalized_start": start, "normalized_end": end,
                "original_start": document.normalized_to_original[start],
                "original_end": document.normalized_to_original[end - 1] + 1,
                "context_normalized_start": context_start, "context_normalized_end": context_end,
                "context_target_start": start - context_start, "context_target_end": end - context_start,
                "representative_occurrence_uid": representative["occurrence_uid"] if representative else None,
                "selection": selection,
            })
            counts[selection] += 1
    return {"schema_version": "stage01-mention-audit-v1", "total_mentions": len(mentions),
            "selected_contexts": len(rows), "duplicate_context_mentions": counts["DUPLICATE_CONTEXT"],
            "context_limit_mentions": counts["CONTEXT_LIMIT"], "mentions": mentions}


def discover_pairs(
    txt: Path | None = None, key: Path | None = None,
    texts_dir: Path | None = None, keys_dir: Path | None = None,
) -> tuple[list[tuple[str, Path, Path]], list[dict], list[dict]]:
    """Pair flat directories by case-insensitive stem; list orphans deterministically."""
    if txt is not None:
        txt, key = txt.resolve(), key.resolve()
        if not txt.is_file() or not key.is_file():
            raise ValueError("Both TXT and KEY must be existing files")
        if txt.suffix.casefold() != ".txt" or key.suffix.casefold() != ".key":
            raise ValueError("Single-document inputs must have .txt and .key extensions")
        if txt.stem.casefold() != key.stem.casefold():
            raise ValueError("TXT/KEY filenames must have the same document stem")
        return [(txt.stem, txt, key)], [], []
    texts_dir, keys_dir = texts_dir.resolve(), keys_dir.resolve()
    if not texts_dir.is_dir() or not keys_dir.is_dir():
        raise ValueError("Both corpus input directories must exist")
    buckets = []
    for directory, suffix in ((texts_dir, ".txt"), (keys_dir, ".key")):
        indexed = defaultdict(list)
        for path in sorted(directory.iterdir(), key=lambda value: (value.name.casefold(), value.name)):
            if path.is_file() and path.suffix.casefold() == suffix:
                indexed[path.stem.casefold()].append(path.resolve())
        buckets.append(indexed)
    texts, keys = buckets
    pairs, unpaired, errors = [], [], []
    for stem in sorted(set(texts) | set(keys)):
        text_paths, key_paths = texts[stem], keys[stem]
        if len(text_paths) > 1 or len(key_paths) > 1:
            errors.append({"stage": "pairing", "kind": "AMBIGUOUS_DOCUMENT_ID", "document_id": stem,
                           "paths": [str(path) for path in text_paths + key_paths]})
        elif text_paths and key_paths:
            pairs.append((text_paths[0].stem, text_paths[0], key_paths[0]))
        else:
            for path in text_paths + key_paths:
                unpaired.append({"document_id": path.stem, "path": str(path),
                                 "missing": ".key" if text_paths else ".txt"})
    return pairs, unpaired, errors


def process_document(
    document_id: str | None, txt: Path | None, key: Path | None = None, *,
    text: str | None = None, terms_file: Path | None = None, terms: Sequence[str] | None = None,
    term_origin: str = "user", language: str = "und",
) -> tuple[list[dict], bytes | None, dict, dict | None, list[dict]]:
    info = {"document_id": document_id, "txt_path": str(txt) if txt else None,
            "key_path": str(key) if key else None, "status": "ERROR", "inputs": {}, "errors": [],
            "contexts": 0}
    document_record, records = None, []
    try:
        source = (DocumentInput.from_text(text, document_id, language) if text is not None
                  else DocumentInput.from_file(txt, document_id, language))
        document_id = source.document_id
        info["document_id"] = document_id
        document_record = source.to_dict()
        info["inputs"]["txt_sha256" if txt else "inline_text_sha256"] = source.source_sha256
        term_records, term_meta = supplied_terms(source, path=key or terms_file, terms=terms,
                                                 origin=term_origin, legacy=key is not None)
        info["inputs"]["key_sha256" if key else "terms_sha256"] = term_meta["source_sha256"]
        records = [record.to_dict() for record in term_records]
        labels = [record.term for record in term_records]
        info.update(term_provider=term_meta["provider"], term_input=term_meta,
                    supplied_terms_count=term_meta["supplied_terms_count"],
                    reference_terms=term_meta["supplied_terms_count"], unique_reference_terms=len(labels),
                    duplicate_reference_terms=term_meta["duplicate_terms"])
        # Inline strings retain their original BOM/newlines for source offsets.
        # Keep the historical normalized view/UIDs by omitting only an initial
        # inline BOM here and translating its offsets back to the supplied str.
        prefix = int(source.source_kind == "inline" and source.text.startswith("\ufeff"))
        document = normalize_text(source.text[prefix:], labels)
        if prefix:
            document = NormalizedDocument(document.text, tuple(offset + prefix for offset in document.normalized_to_original))
        legacy_rows = extract_rows(labels, document)
        validation = validate_rows(legacy_rows, labels, document, source.text)
        rows = enrich_rows(legacy_rows, document_id, document)
        record_by_term = {record.term: record for record in term_records}
        for row in rows:
            term_record = record_by_term[str(row["term"])]
            row.update(term_id=term_record.term_id, language=source.language,
                       term_source="+".join(sorted({item["provider"] for item in term_record.provenance})),
                       term_provenance_json=canonical(list(term_record.provenance)))
        counts = Counter(str(row["term"]) for row in rows)
        normalized_name = f"normalized/{document_id}.txt"
        normalized = document.text.encode("utf-8")
        info.update(status="OK", contexts=len(rows), terms_found=sum(counts[term] > 0 for term in labels),
                    contexts_by_term={term: counts[term] for term in labels},
                    terms_not_found=[term for term in labels if not counts[term]],
                    max_words_before=validation.max_words_before,
                     normalized_file=normalized_name, normalized_sha256=sha256(normalized),
                     mention_audit=audit_mentions(rows, document_id, document, term_records))
        return rows, normalized, info, document_record, records
    except (OSError, UnicodeError, ValueError, AssertionError) as exception:
        info["errors"].append({"stage": "document", "error_type": type(exception).__name__, "message": str(exception)})
        return [], None, info, document_record, records


def contexts_payload(rows: Sequence[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def run_stage01(
    output_dir: Path, *, txt: Path | None = None, key: Path | None = None,
    texts_dir: Path | None = None, keys_dir: Path | None = None, max_documents: int | None = None,
    text: str | None = None, terms_file: Path | None = None, terms: Sequence[str] | None = None,
    document_id: str | None = None, term_origin: str = "user", language: str = "und",
) -> dict:
    if sum(value is not None for value in (txt, text, texts_dir)) != 1:
        raise ValueError("Use exactly one of --txt, --text or --texts-dir")
    sources = (key, keys_dir, terms_file, terms)
    if sum(value is not None for value in sources) != 1:
        raise ValueError("Supply --terms-file or --term; --key/--keys-dir are optional legacy modes")
    if texts_dir is not None and (keys_dir is None or document_id is not None):
        raise ValueError("The legacy directory mode requires --keys-dir and no document ID override")
    if texts_dir is None and keys_dir is not None:
        raise ValueError("--keys-dir requires --texts-dir")
    if key is not None and (txt is None or text is not None):
        raise ValueError("--key is a legacy TXT/KEY pair; use --terms-file or --term for direct text")
    if max_documents is not None and max_documents < 1:
        raise ValueError("--max-documents must be positive")
    output_dir = output_dir.resolve()
    if output_dir == Path(__file__).resolve().parent:
        raise ValueError("Use a separate output directory, never the historical project root")
    for directory in (texts_dir, keys_dir):
        if directory and (output_dir == directory.resolve() or directory.resolve() in output_dir.parents):
            raise ValueError("Output must be outside the corpus input directories")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be new or empty; previous results are never overwritten")
    legacy = key is not None or keys_dir is not None
    if legacy:
        pairs, unpaired, errors = discover_pairs(txt, key, texts_dir, keys_dir)
        if document_id is not None:
            pairs = [(document_id, pairs[0][1], pairs[0][2])]
    else:
        if txt is not None and not txt.is_file():
            raise ValueError("--txt must point to an existing user text file")
        if terms_file is not None and not terms_file.is_file():
            raise ValueError("--terms-file must point to an existing UTF-8 file, one term per line")
        pairs, unpaired, errors = [(document_id, txt.resolve() if txt else None, None)], [], []
    available_count = len(pairs)
    selected = pairs[:max_documents] if max_documents is not None else pairs
    if not pairs:
        errors.append({"stage": "pairing", "kind": "NO_PAIRED_INPUTS"})
    errors.extend({"stage": "pairing", "kind": "MISSING_PAIR", **item} for item in unpaired)
    rows, documents, payloads, document_records, term_records = [], [], {}, [], []
    for selected_id, txt_path, key_path in selected:
        doc_rows, normalized, info, doc_record, term_record_list = process_document(
            selected_id, txt_path, key_path, text=text, terms_file=terms_file, terms=terms,
            term_origin=term_origin, language=language,
        )
        documents.append(info)
        if doc_record is not None:
            document_records.append(doc_record)
        term_records.extend(term_record_list)
        if normalized is not None:
            rows.extend(doc_rows)
            payloads[info["normalized_file"]] = normalized
        errors.extend({"document_id": info["document_id"], **item} for item in info["errors"])
    if len({row["occurrence_uid"] for row in rows}) != len(rows):
        raise ValueError("Global UID collision across documents")
    payloads["01_contextos.csv"] = contexts_payload(rows)
    payloads["01_documents.jsonl"] = "".join(jsonl_line(record) for record in document_records).encode("utf-8")
    payloads["01_terms.jsonl"] = "".join(jsonl_line(record) for record in term_records).encode("utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION, "script_sha256": sha256(Path(__file__).read_bytes()),
        "component_sha256": {name: sha256((Path(__file__).resolve().parent / name).read_bytes())
                             for name in ("01_extraer_contextos.py", "terminlink_inputs.py")},
        "python_version": platform.python_version(),
        "input_mode": "paired_directories" if texts_dir else "inline_text" if text is not None else "single_pair" if key else "user_text_file",
        "linking_contract_version": CONTRACT_VERSION,
        "term_input_mode": "legacy_key" if legacy else "supplied_terms",
        "execution_scope": "STAGE01_CONTEXT_PREPARATION_ONLY", "linking_executed": False,
        "methodology": {"literal_case_insensitive": True, "word_boundaries": "(?<!\\w)/(?!\\w)",
                         "words_before": WORDS_BEFORE, "end": "FIRST_TERMINATING_PERIOD_AFTER_MATCH_OR_SEGMENT_END",
                         "period_boundary_policy": PERIOD_POLICY,
                         "deduplication": "EXACT_CONTEXT_PER_TERM_PER_DOCUMENT", "max_contexts_per_term_per_document": MAX_CONTEXTS_PER_TERM,
                         "mention_audit": "ALL_LITERAL_SPANS_IN_DOCUMENT_METADATA_NO_EXTRA_CONTEXT_ROWS"},
        "offset_convention": ("ZERO_BASED_HALF_OPEN_UNICODE_CHARACTERS_ORIGINAL_INLINE_STRING_NOT_BYTES" if text is not None
                              else "ZERO_BASED_HALF_OPEN_UNICODE_CHARACTERS_UTF8_SIG_UNIVERSAL_NEWLINES_NOT_BYTES"),
        "markers": {"open": TARGET_OPEN, "close": TARGET_CLOSE, "used_in_embeddings_or_prompts": False},
        "identity": {"occurrence_id": "LEGACY_LOCAL_ID_UNCHANGED", "primary_key": "occurrence_uid",
                     "document_id": "EXPLICIT_OR_CONTENT_IDENTIFIED_USER_DOCUMENT; LEGACY_STEM_FOR_TXT_KEY_PAIRS",
                     "uid_components": ["document_id", "term", "normalized_start", "normalized_end"]},
        "columns": list(OUTPUT_COLUMNS), "discovery": {"paired_documents_available": available_count,
                    "max_documents": max_documents, "selected_documents": len(selected),
                    "processing_order": [info["document_id"] for info in documents], "order": "CASEFOLDED_FILENAME_STEM_LEXICOGRAPHIC_FLAT_DIRECTORIES" if texts_dir else "SINGLE_USER_DOCUMENT"},
        "documents": documents, "unpaired_inputs": unpaired, "errors": errors,
        "totals": {"documents_processed": len(documents), "documents_ok": sum(item["status"] == "OK" for item in documents),
                    "documents_failed": sum(item["status"] == "ERROR" for item in documents),
                   "terms": len(term_records), "contexts": len(rows)},
        "status": "PARTIAL_OR_FAILED" if errors else "COMPLETED",
        "output_sha256": {name: sha256(data) for name, data in sorted(payloads.items())},
    }
    payloads["01_manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(payloads.items()):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--txt", type=Path)
    mode.add_argument("--text", help="Direct text; no corpus directory or reference file required")
    mode.add_argument("--texts-dir", type=Path)
    term_mode = parser.add_mutually_exclusive_group(required=True)
    term_mode.add_argument("--terms-file", type=Path, help="Any UTF-8 filename, one term per line")
    term_mode.add_argument("--term", dest="terms", action="append", help="Repeat for an inline list of user terms")
    term_mode.add_argument("--key", type=Path, help="Legacy TXT/KEY regression compatibility")
    term_mode.add_argument("--keys-dir", type=Path, help="Legacy paired-directory compatibility")
    parser.add_argument("--document-id", help="Optional user document identifier; text input otherwise gets a content-derived ID")
    parser.add_argument("--term-origin", choices=("user", "external"), default="user", help="Provenance label only; does not change matching")
    parser.add_argument("--language", default="und", help="Language metadata; no automatic detection or universal-language performance claim")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args(argv)
    try:
        manifest = run_stage01(**vars(args))
    except (OSError, ValueError) as exception:
        print(f"STAGE01_ERROR={exception}", file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], **manifest["totals"],
                      "unpaired_inputs": len(manifest["unpaired_inputs"]), "errors": len(manifest["errors"])}, sort_keys=True))
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
