#!/usr/bin/env python3
"""Prepare contextualized Wikidata-only inputs for a future translation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = BASE_DIR / "03_candidatos_representaciones.csv"
PENDING_PATH = BASE_DIR / "03_traducciones_pendientes.csv"
OUTPUT_PATH = BASE_DIR / "03_traducciones_contextualizadas_entradas.csv"

EXPECTED_CANDIDATES_SHA256 = (
    "AB134E71DC9CA4F62297C2C7823EBACBC0E32D7190055E336C0AA16A928A9D9A"
)
EXPECTED_PENDING_SHA256 = (
    "63945729ED853F15C3676231C7D7AE88A931028D456A3C08DB6006FB2555E6CE"
)
EXPECTED_TOTAL_CANDIDATES = 3474
EXPECTED_SOURCE_CANDIDATES = 328
EXPECTED_PENDING_TRANSLATIONS = 2036
EXPECTED_UNIQUE_QIDS = 327
EXPECTED_SOURCE_LANGUAGES = 157
EXPECTED_AVAILABILITY_COUNTS = {
    "label": 2030,
    "aliases": 1092,
    "other_description": 1978,
    "p31": 1976,
    "p279": 195,
    "p1343": 21,
}

SOURCE_ROUTE = "OTHER_LANGUAGE_DESCRIPTION"
SOURCE_TRANSLATION_STATUS = "PENDING"
TARGET_LANGUAGE = "en"
CONTEXT_POLICY_VERSION = "wikidata-candidate-translation-context-v1"

CANDIDATE_FIELDS = (
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
    "representation_route",
    "description_en_word_count",
    "available_other_descriptions",
    "has_aliases",
    "has_p31",
    "has_p279",
    "has_p1343",
)

PENDING_FIELDS = (
    "representation_id",
    "term",
    "qid",
    "label_en",
    "source_language",
    "source_description",
    "translation_status",
)

OUTPUT_FIELDS = (
    "contextual_translation_input_id",
    "representation_id",
    "qid",
    "source_language",
    "source_description",
    "target_language",
    "source_translation_status",
    "label_en",
    "aliases_en",
    "description_en",
    "other_descriptions",
    "p31",
    "p279",
    "p1343",
    "representation_route",
    "context_policy_version",
    "translation_evidence_text",
    "pending_input_sha256",
    "candidate_input_sha256",
)

FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "term",
        "context",
        "corpus_context",
        "occurrence_id",
        "search_queries",
        "search_strategies",
        "embedding",
        "embedding_score",
        "score_label",
        "score_alias",
        "score_context_def",
        "llm_decision",
        "expected_relation",
        "evidence_sufficiency",
        "grounding",
        "overall_decision",
    }
)

QID_PATTERN = re.compile(r"Q[1-9][0-9]*")
REPRESENTATION_ID_PATTERN = re.compile(r"rep_[0-9a-f]{64}")
CONTEXTUAL_INPUT_ID_PATTERN = re.compile(r"ctxtrans_[0-9a-f]{64}")
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


class PreparationError(RuntimeError):
    """Raised when an input or generated contextual row violates its contract."""


@dataclass(frozen=True)
class Relation:
    qid: str
    label_en: str

    def as_dict(self) -> dict[str, str]:
        return {"qid": self.qid, "label_en": self.label_en}


@dataclass(frozen=True)
class CandidateEvidence:
    qid: str
    label_en: str
    aliases_en: tuple[str, ...]
    description_en: str
    descriptions: tuple[tuple[str, str], ...]
    p31: tuple[Relation, ...]
    p279: tuple[Relation, ...]
    p1343: tuple[Relation, ...]
    representation_route: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_sha256(path: Path, expected_hash: str) -> str:
    require(path.is_file(), f"Required CSV does not exist: {path.name}")
    actual_hash = sha256_file(path)
    require(
        actual_hash == expected_hash,
        f"{path.name}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}",
    )
    return actual_hash


def read_csv_exact(
    path: Path, expected_fields: Sequence[str]
) -> list[dict[str, str]]:
    require(path.is_file(), f"Required CSV does not exist: {path.name}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require(
                tuple(reader.fieldnames or ()) == tuple(expected_fields),
                f"{path.name}: unexpected schema {reader.fieldnames!r}",
            )
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                require(
                    None not in row and all(value is not None for value in row.values()),
                    f"{path.name}: malformed row {row_number}",
                )
                rows.append({field: row[field] for field in expected_fields})
    except (OSError, UnicodeError, csv.Error) as error:
        raise PreparationError(f"Could not read {path.name}: {error}") from error
    return rows


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_json_value(raw: str, description: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise PreparationError(f"Invalid JSON in {description}") from error


def parse_aliases(raw: str, candidate_key: tuple[str, str]) -> tuple[str, ...]:
    value = parse_json_value(raw, f"aliases_en for {candidate_key!r}")
    require(
        isinstance(value, list)
        and all(isinstance(alias, str) and alias.strip() for alias in value),
        f"Malformed aliases_en for {candidate_key!r}",
    )
    require(
        len(value) == len(set(value)),
        f"Duplicate English aliases for {candidate_key!r}",
    )
    return tuple(value)


def parse_relations(
    raw: str, field: str, candidate_key: tuple[str, str]
) -> tuple[Relation, ...]:
    value = parse_json_value(raw, f"{field} for {candidate_key!r}")
    require(isinstance(value, list), f"Malformed {field} for {candidate_key!r}")
    relations: list[Relation] = []
    seen_qids: set[str] = set()
    for relation in value:
        require(
            isinstance(relation, dict)
            and set(relation) == {"qid", "label_en"}
            and isinstance(relation["qid"], str)
            and isinstance(relation["label_en"], str)
            and QID_PATTERN.fullmatch(relation["qid"]) is not None,
            f"Malformed {field} relation for {candidate_key!r}",
        )
        require(
            relation["qid"] not in seen_qids,
            f"Duplicate {field} relation for {candidate_key!r}",
        )
        seen_qids.add(relation["qid"])
        relations.append(Relation(relation["qid"], relation["label_en"]))
    return tuple(relations)


def parse_descriptions(
    row: dict[str, str], candidate_key: tuple[str, str]
) -> tuple[tuple[str, str], ...]:
    original = parse_json_value(
        row["descriptions_other_languages"],
        f"descriptions_other_languages for {candidate_key!r}",
    )
    require(
        isinstance(original, dict)
        and all(
            isinstance(language, str)
            and language
            and language == language.strip()
            and isinstance(text, str)
            for language, text in original.items()
        ),
        f"Malformed multilingual descriptions for {candidate_key!r}",
    )
    expected_available = [
        {"language": language, "text": text}
        for language, text in original.items()
        if text.strip()
    ]
    available = parse_json_value(
        row["available_other_descriptions"],
        f"available_other_descriptions for {candidate_key!r}",
    )
    require(
        available == expected_available,
        f"Available descriptions differ from source data for {candidate_key!r}",
    )
    require(
        all(
            isinstance(item, dict)
            and set(item) == {"language", "text"}
            and isinstance(item["language"], str)
            and item["language"]
            and item["language"] == item["language"].strip()
            and isinstance(item["text"], str)
            and item["text"].strip()
            for item in available
        ),
        f"Malformed available descriptions for {candidate_key!r}",
    )
    languages = [item["language"] for item in available]
    require(
        len(languages) == len(set(languages)),
        f"Duplicate description languages for {candidate_key!r}",
    )
    return tuple(
        sorted(
            ((item["language"], item["text"]) for item in available),
            key=lambda item: (item[0], item[1]),
        )
    )


def parse_boolean(raw: str, field: str, candidate_key: tuple[str, str]) -> bool:
    require(
        raw in {"true", "false"},
        f"Invalid {field} flag for {candidate_key!r}",
    )
    return raw == "true"


def parse_candidate_evidence(row: dict[str, str]) -> CandidateEvidence:
    candidate_key = (row["term"], row["qid"])
    require(row["term"].strip() != "", f"Empty term for {candidate_key!r}")
    require(
        QID_PATTERN.fullmatch(row["qid"]) is not None,
        f"Invalid candidate QID for {candidate_key!r}",
    )
    require(
        row["label_en"] == "" or row["label_en"].strip() != "",
        f"Whitespace-only English label for {candidate_key!r}",
    )
    aliases = parse_aliases(row["aliases_en"], candidate_key)
    descriptions = parse_descriptions(row, candidate_key)
    p31 = parse_relations(row["p31"], "p31", candidate_key)
    p279 = parse_relations(row["p279"], "p279", candidate_key)
    p1343 = parse_relations(row["p1343"], "p1343", candidate_key)

    require(
        parse_boolean(row["has_aliases"], "has_aliases", candidate_key)
        == bool(aliases),
        f"has_aliases mismatch for {candidate_key!r}",
    )
    require(
        parse_boolean(row["has_p31"], "has_p31", candidate_key) == bool(p31),
        f"has_p31 mismatch for {candidate_key!r}",
    )
    require(
        parse_boolean(row["has_p279"], "has_p279", candidate_key) == bool(p279),
        f"has_p279 mismatch for {candidate_key!r}",
    )
    require(
        parse_boolean(row["has_p1343"], "has_p1343", candidate_key)
        == bool(p1343),
        f"has_p1343 mismatch for {candidate_key!r}",
    )
    require(
        row["description_en_word_count"]
        == str(len(WORD_PATTERN.findall(row["description_en"]))),
        f"description_en_word_count mismatch for {candidate_key!r}",
    )
    return CandidateEvidence(
        qid=row["qid"],
        label_en=row["label_en"],
        aliases_en=aliases,
        description_en=row["description_en"],
        descriptions=descriptions,
        p31=p31,
        p279=p279,
        p1343=p1343,
        representation_route=row["representation_route"],
    )


def load_candidate_index() -> dict[tuple[str, str], CandidateEvidence]:
    rows = read_csv_exact(CANDIDATES_PATH, CANDIDATE_FIELDS)
    require(
        len(rows) == EXPECTED_TOTAL_CANDIDATES,
        f"Expected {EXPECTED_TOTAL_CANDIDATES} candidates, found {len(rows)}",
    )
    seen_keys: set[tuple[str, str]] = set()
    source_index: dict[tuple[str, str], CandidateEvidence] = {}
    for row in rows:
        key = (row["term"], row["qid"])
        require(key not in seen_keys, f"Duplicate candidate key: {key!r}")
        seen_keys.add(key)
        if row["representation_route"] == SOURCE_ROUTE:
            evidence = parse_candidate_evidence(row)
            require(
                evidence.descriptions,
                f"Fallback candidate has no descriptions: {key!r}",
            )
            source_index[key] = evidence
    require(
        len(source_index) == EXPECTED_SOURCE_CANDIDATES,
        f"Expected {EXPECTED_SOURCE_CANDIDATES} fallback candidates, "
        f"found {len(source_index)}",
    )
    return source_index


def is_english_variant(language: str) -> bool:
    return (
        language.strip().casefold().replace("_", "-").split("-", 1)[0]
        == "en"
    )


def expected_translation_units(
    source_index: dict[tuple[str, str], CandidateEvidence],
) -> set[tuple[str, str, str, str]]:
    return {
        (*candidate_key, language, text)
        for candidate_key, evidence in source_index.items()
        for language, text in evidence.descriptions
        if not is_english_variant(language)
    }


def expected_representation_id(row: dict[str, str]) -> str:
    identity = json.dumps(
        [
            row["term"],
            row["qid"],
            row["source_language"],
            row["source_description"],
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "rep_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_pending_rows(
    source_index: dict[tuple[str, str], CandidateEvidence],
) -> list[dict[str, str]]:
    rows = read_csv_exact(PENDING_PATH, PENDING_FIELDS)
    require(
        len(rows) == EXPECTED_PENDING_TRANSLATIONS,
        f"Expected {EXPECTED_PENDING_TRANSLATIONS} translation units, "
        f"found {len(rows)}",
    )
    seen_ids: set[str] = set()
    seen_units: set[tuple[str, str, str, str]] = set()
    covered_candidates: set[tuple[str, str]] = set()
    languages: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        key = (row["term"], row["qid"])
        unit = (*key, row["source_language"], row["source_description"])
        require(key in source_index, f"Pending row {row_number} has no candidate")
        evidence = source_index[key]
        require(
            REPRESENTATION_ID_PATTERN.fullmatch(row["representation_id"])
            is not None,
            f"Invalid representation_id at pending row {row_number}",
        )
        require(
            row["representation_id"] not in seen_ids,
            f"Duplicate representation_id at pending row {row_number}",
        )
        require(unit not in seen_units, f"Duplicate translation unit: {unit!r}")
        require(
            row["representation_id"] == expected_representation_id(row),
            f"representation_id mismatch at pending row {row_number}",
        )
        require(
            row["qid"] == evidence.qid and row["label_en"] == evidence.label_en,
            f"Candidate traceability mismatch at pending row {row_number}",
        )
        require(
            row["source_language"].strip() != ""
            and row["source_language"] == row["source_language"].strip()
            and row["source_description"].strip() != "",
            f"Missing target language or description at pending row {row_number}",
        )
        require(
            not is_english_variant(row["source_language"]),
            f"English source language at pending row {row_number}",
        )
        require(
            (row["source_language"], row["source_description"])
            in evidence.descriptions,
            f"Target description is absent from candidate evidence at row {row_number}",
        )
        require(
            row["translation_status"] == SOURCE_TRANSLATION_STATUS,
            f"Unexpected source translation status at row {row_number}",
        )
        seen_ids.add(row["representation_id"])
        seen_units.add(unit)
        covered_candidates.add(key)
        languages.add(row["source_language"])

    require(
        covered_candidates == set(source_index),
        "Pending translations do not cover every fallback candidate",
    )
    require(
        seen_units == expected_translation_units(source_index),
        "Pending translations do not exactly cover all non-English descriptions",
    )
    require(
        len({row["qid"] for row in rows}) == EXPECTED_UNIQUE_QIDS,
        "Unexpected unique QID count in pending translations",
    )
    require(
        len(languages) == EXPECTED_SOURCE_LANGUAGES,
        "Unexpected source-language count in pending translations",
    )
    return rows


def relation_dicts(relations: Sequence[Relation]) -> list[dict[str, str]]:
    return [relation.as_dict() for relation in relations]


def other_descriptions(
    evidence: CandidateEvidence, source_language: str, source_description: str
) -> list[dict[str, str]]:
    return [
        {"language": language, "text": text}
        for language, text in evidence.descriptions
        if (language, text) != (source_language, source_description)
    ]


def render_relations(
    lines: list[str], heading: str, relations: Sequence[Relation]
) -> None:
    if not relations:
        return
    lines.extend(["", heading])
    for relation in relations:
        if relation.label_en:
            lines.append(f"- {relation.label_en} (QID: {relation.qid})")
        else:
            lines.append(f"- QID: {relation.qid}")


def render_translation_evidence(
    evidence: CandidateEvidence,
    source_language: str,
    source_description: str,
    sibling_descriptions: Sequence[dict[str, str]],
) -> str:
    lines = ["Wikidata candidate:", f"QID: {evidence.qid}"]
    if evidence.label_en:
        lines.append(f"English label: {evidence.label_en}")
    if evidence.aliases_en:
        lines.append("English aliases: " + "; ".join(evidence.aliases_en))
    if evidence.description_en:
        lines.append(
            "English description (auxiliary evidence, not translation target): "
            + evidence.description_en
        )

    if sibling_descriptions:
        lines.extend(
            [
                "",
                "Other available candidate descriptions "
                "(context only, not translation target):",
            ]
        )
        for description in sibling_descriptions:
            lines.append(
                f"- [{description['language']}] {description['text']}"
            )

    render_relations(lines, "Instance of (P31):", evidence.p31)
    render_relations(lines, "Subclass of (P279):", evidence.p279)
    render_relations(lines, "Described by source (P1343):", evidence.p1343)

    lines.extend(
        [
            "",
            "Target text to translate:",
            f"Source language: {source_language}",
            "Target language: English (en)",
            f"Source description: {source_description}",
        ]
    )
    return "\n".join(lines)


def contextual_input_id(payload: dict[str, Any]) -> str:
    serialized = canonical_json(payload)
    return "ctxtrans_" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_output_row(
    pending: dict[str, str],
    evidence: CandidateEvidence,
    pending_sha256: str,
    candidate_sha256: str,
) -> dict[str, str]:
    siblings = other_descriptions(
        evidence, pending["source_language"], pending["source_description"]
    )
    aliases = list(evidence.aliases_en)
    p31 = relation_dicts(evidence.p31)
    p279 = relation_dicts(evidence.p279)
    p1343 = relation_dicts(evidence.p1343)
    id_payload = {
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "representation_id": pending["representation_id"],
        "qid": evidence.qid,
        "source_language": pending["source_language"],
        "source_description": pending["source_description"],
        "target_language": TARGET_LANGUAGE,
        "label_en": evidence.label_en,
        "aliases_en": aliases,
        "description_en": evidence.description_en,
        "other_descriptions": siblings,
        "p31": p31,
        "p279": p279,
        "p1343": p1343,
        "representation_route": evidence.representation_route,
    }
    return {
        "contextual_translation_input_id": contextual_input_id(id_payload),
        "representation_id": pending["representation_id"],
        "qid": evidence.qid,
        "source_language": pending["source_language"],
        "source_description": pending["source_description"],
        "target_language": TARGET_LANGUAGE,
        "source_translation_status": pending["translation_status"],
        "label_en": evidence.label_en,
        "aliases_en": canonical_json(aliases),
        "description_en": evidence.description_en,
        "other_descriptions": canonical_json(siblings),
        "p31": canonical_json(p31),
        "p279": canonical_json(p279),
        "p1343": canonical_json(p1343),
        "representation_route": evidence.representation_route,
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "translation_evidence_text": render_translation_evidence(
            evidence,
            pending["source_language"],
            pending["source_description"],
            siblings,
        ),
        "pending_input_sha256": pending_sha256,
        "candidate_input_sha256": candidate_sha256,
    }


def build_output_rows(
    pending_rows: Sequence[dict[str, str]],
    source_index: dict[tuple[str, str], CandidateEvidence],
    pending_sha256: str,
    candidate_sha256: str,
) -> list[dict[str, str]]:
    return [
        build_output_row(
            pending,
            source_index[(pending["term"], pending["qid"])],
            pending_sha256,
            candidate_sha256,
        )
        for pending in pending_rows
    ]


def availability_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    return {
        "label": sum(bool(row["label_en"]) for row in rows),
        "aliases": sum(row["aliases_en"] != "[]" for row in rows),
        "other_description": sum(
            row["other_descriptions"] != "[]" for row in rows
        ),
        "p31": sum(row["p31"] != "[]" for row in rows),
        "p279": sum(row["p279"] != "[]" for row in rows),
        "p1343": sum(row["p1343"] != "[]" for row in rows),
    }


def validate_output_rows(
    output_rows: Sequence[dict[str, str]],
    pending_rows: Sequence[dict[str, str]],
    source_index: dict[tuple[str, str], CandidateEvidence],
    pending_sha256: str,
    candidate_sha256: str,
) -> dict[str, int]:
    require(
        len(output_rows) == EXPECTED_PENDING_TRANSLATIONS,
        f"Expected {EXPECTED_PENDING_TRANSLATIONS} output rows, "
        f"found {len(output_rows)}",
    )
    require(
        not FORBIDDEN_OUTPUT_FIELDS.intersection(OUTPUT_FIELDS),
        "Output schema exposes forbidden corpus, evaluation, or score fields",
    )

    seen_context_ids: set[str] = set()
    seen_representation_ids: set[str] = set()
    for row_number, (output, pending) in enumerate(
        zip(output_rows, pending_rows), start=2
    ):
        require(
            tuple(output) == OUTPUT_FIELDS,
            f"Unexpected output schema at row {row_number}",
        )
        require(
            CONTEXTUAL_INPUT_ID_PATTERN.fullmatch(
                output["contextual_translation_input_id"]
            )
            is not None,
            f"Invalid contextual input ID at row {row_number}",
        )
        require(
            output["contextual_translation_input_id"] not in seen_context_ids,
            f"Duplicate contextual input ID at row {row_number}",
        )
        require(
            output["representation_id"] not in seen_representation_ids,
            f"Duplicate representation ID at row {row_number}",
        )
        require(
            output["representation_id"] == pending["representation_id"]
            and output["qid"] == pending["qid"]
            and output["source_language"] == pending["source_language"]
            and output["source_description"] == pending["source_description"],
            f"Target traceability changed at row {row_number}",
        )
        require(
            output["source_language"].strip() != ""
            and output["source_description"].strip() != "",
            f"Target language or description missing at row {row_number}",
        )
        require(
            output["context_policy_version"] == CONTEXT_POLICY_VERSION
            and output["representation_route"] == SOURCE_ROUTE,
            f"Context policy or route mismatch at row {row_number}",
        )
        target_block = (
            "Target text to translate:\n"
            f"Source language: {output['source_language']}\n"
            "Target language: English (en)\n"
            f"Source description: {output['source_description']}"
        )
        require(
            target_block in output["translation_evidence_text"],
            f"Target is not explicit in contextual text at row {row_number}",
        )
        require(
            output["pending_input_sha256"] == pending_sha256
            and output["candidate_input_sha256"] == candidate_sha256,
            f"Input provenance mismatch at row {row_number}",
        )
        expected = build_output_row(
            pending,
            source_index[(pending["term"], pending["qid"])],
            pending_sha256,
            candidate_sha256,
        )
        require(
            output == expected,
            f"Non-deterministic or altered contextual row {row_number}",
        )
        evidence = source_index[(pending["term"], pending["qid"])]
        expected_siblings = [
            {"language": language, "text": text}
            for language, text in evidence.descriptions
            if (language, text)
            != (pending["source_language"], pending["source_description"])
        ]
        require(
            parse_json_value(output["aliases_en"], "output aliases")
            == list(evidence.aliases_en)
            and parse_json_value(
                output["other_descriptions"], "output other descriptions"
            )
            == expected_siblings
            and parse_json_value(output["p31"], "output p31")
            == relation_dicts(evidence.p31)
            and parse_json_value(output["p279"], "output p279")
            == relation_dicts(evidence.p279)
            and parse_json_value(output["p1343"], "output p1343")
            == relation_dicts(evidence.p1343),
            f"Structured candidate evidence changed at row {row_number}",
        )
        require(
            output["label_en"] == evidence.label_en
            and output["description_en"] == evidence.description_en
            and output["target_language"] == TARGET_LANGUAGE
            and output["source_translation_status"]
            == SOURCE_TRANSLATION_STATUS,
            f"Scalar candidate evidence changed at row {row_number}",
        )
        seen_context_ids.add(output["contextual_translation_input_id"])
        seen_representation_ids.add(output["representation_id"])

    require(
        len(seen_context_ids) == len(output_rows)
        and len(seen_representation_ids) == len(output_rows),
        "Unexpected contextual translation duplicates",
    )
    counts = availability_counts(output_rows)
    require(
        counts == EXPECTED_AVAILABILITY_COUNTS,
        f"Unexpected evidence availability counts: {counts!r}",
    )
    return counts


def write_temporary_csv(
    path: Path, rows: Sequence[dict[str, str]]
) -> Path:
    temporary_path = path.with_name(f".{path.name}.tmp")
    require(
        not temporary_path.exists(),
        f"Refusing to overwrite stale temporary file: {temporary_path.name}",
    )
    try:
        with temporary_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(OUTPUT_FIELDS),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def synthetic_candidate(
    *, with_optional_evidence: bool = True
) -> CandidateEvidence:
    return CandidateEvidence(
        qid="Q1",
        label_en="Synthetic candidate" if with_optional_evidence else "",
        aliases_en=("Synthetic alias",) if with_optional_evidence else (),
        description_en="brief description" if with_optional_evidence else "",
        descriptions=(
            ("de", "Zielbeschreibung"),
            ("en-gb", "target-language auxiliary description"),
            ("fr", "description de contexte"),
        ),
        p31=(Relation("Q2", "synthetic class"),)
        if with_optional_evidence
        else (),
        p279=(Relation("Q3", "broader class"),)
        if with_optional_evidence
        else (),
        p1343=(Relation("Q4", "reference work"),)
        if with_optional_evidence
        else (),
        representation_route=SOURCE_ROUTE,
    )


def synthetic_pending() -> dict[str, str]:
    row = {
        "representation_id": "",
        "term": "CORPUS_TERM_SENTINEL",
        "qid": "Q1",
        "label_en": "Synthetic candidate",
        "source_language": "de",
        "source_description": "Zielbeschreibung",
        "translation_status": SOURCE_TRANSLATION_STATUS,
    }
    row["representation_id"] = expected_representation_id(row)
    return row


def test_context_rendering() -> None:
    pending = synthetic_pending()
    evidence = synthetic_candidate()
    row = build_output_row(pending, evidence, "A" * 64, "B" * 64)
    text = row["translation_evidence_text"]
    require("English label: Synthetic candidate" in text, "English label missing")
    require("English aliases: Synthetic alias" in text, "Aliases missing")
    require("[fr] description de contexte" in text, "Sibling description missing")
    require(
        "[en-gb] target-language auxiliary description" in text,
        "English regional sibling description missing",
    )
    require("Instance of (P31):" in text, "P31 missing")
    require("Subclass of (P279):" in text, "P279 missing")
    require("Described by source (P1343):" in text, "P1343 missing")
    require(
        "Target text to translate:\nSource language: de\n"
        "Target language: English (en)\n"
        "Source description: Zielbeschreibung" in text,
        "Target block missing",
    )
    require(
        text.count("Zielbeschreibung") == 1,
        "Target description was duplicated in contextual text",
    )
    print("SYNTHETIC_CONTEXT_RENDERING_TEST=OK")


def test_sense_isolation() -> None:
    pending = synthetic_pending()
    row = build_output_row(
        pending, synthetic_candidate(), "A" * 64, "B" * 64
    )
    require("term" not in OUTPUT_FIELDS, "Corpus term leaked into output schema")
    require(
        pending["term"] not in row["translation_evidence_text"],
        "Corpus term leaked into model input",
    )
    require(
        not FORBIDDEN_OUTPUT_FIELDS.intersection(OUTPUT_FIELDS),
        "Forbidden corpus, evaluation, or score field leaked",
    )
    print("SYNTHETIC_SENSE_ISOLATION_TEST=OK")


def test_determinism() -> None:
    pending = synthetic_pending()
    evidence = synthetic_candidate()
    first = build_output_row(pending, evidence, "A" * 64, "B" * 64)
    second = build_output_row(pending, evidence, "A" * 64, "B" * 64)
    require(first == second, "Contextual output is not deterministic")
    require(
        first["contextual_translation_input_id"]
        == second["contextual_translation_input_id"],
        "Contextual IDs are not deterministic",
    )
    print("SYNTHETIC_CONTEXT_DETERMINISM_TEST=OK")


def test_translation_unit_coverage() -> None:
    source_index = {
        ("CORPUS_TERM_SENTINEL", "Q1"): synthetic_candidate()
    }
    units = expected_translation_units(source_index)
    require(
        units
        == {
            ("CORPUS_TERM_SENTINEL", "Q1", "de", "Zielbeschreibung"),
            (
                "CORPUS_TERM_SENTINEL",
                "Q1",
                "fr",
                "description de contexte",
            ),
        },
        "Translation-unit coverage included or omitted the wrong descriptions",
    )
    print("SYNTHETIC_TRANSLATION_UNIT_COVERAGE_TEST=OK")


def test_empty_section_omission() -> None:
    pending = synthetic_pending()
    evidence = synthetic_candidate(with_optional_evidence=False)
    siblings = other_descriptions(
        evidence, pending["source_language"], pending["source_description"]
    )
    text = render_translation_evidence(
        evidence,
        pending["source_language"],
        pending["source_description"],
        siblings,
    )
    for absent_heading in (
        "English label:",
        "English aliases:",
        "English description (auxiliary evidence, not translation target):",
        "Instance of (P31):",
        "Subclass of (P279):",
        "Described by source (P1343):",
    ):
        require(absent_heading not in text, f"Unexpected empty section: {absent_heading}")
    require("Target text to translate:" in text, "Target section was omitted")
    print("SYNTHETIC_EMPTY_SECTION_OMISSION_TEST=OK")


def run_self_tests() -> None:
    test_context_rendering()
    test_sense_isolation()
    test_determinism()
    test_translation_unit_coverage()
    test_empty_section_omission()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare contextualized, Wikidata-only description translation inputs."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic tests without reading or writing project CSVs",
    )
    return parser.parse_args()


def run_preparation() -> None:
    candidate_sha256 = verify_sha256(
        CANDIDATES_PATH, EXPECTED_CANDIDATES_SHA256
    )
    pending_sha256 = verify_sha256(PENDING_PATH, EXPECTED_PENDING_SHA256)
    source_index = load_candidate_index()
    pending_rows = load_pending_rows(source_index)
    output_rows = build_output_rows(
        pending_rows, source_index, pending_sha256, candidate_sha256
    )
    repeated_rows = build_output_rows(
        pending_rows, source_index, pending_sha256, candidate_sha256
    )
    require(output_rows == repeated_rows, "Repeated preparation is not deterministic")
    counts = validate_output_rows(
        output_rows,
        pending_rows,
        source_index,
        pending_sha256,
        candidate_sha256,
    )

    temporary_path = write_temporary_csv(OUTPUT_PATH, output_rows)
    try:
        written_rows = read_csv_exact(temporary_path, OUTPUT_FIELDS)
        require(written_rows == output_rows, "CSV round-trip changed contextual data")
        validate_output_rows(
            written_rows,
            pending_rows,
            source_index,
            pending_sha256,
            candidate_sha256,
        )
        require(
            verify_sha256(CANDIDATES_PATH, EXPECTED_CANDIDATES_SHA256)
            == candidate_sha256
            and verify_sha256(PENDING_PATH, EXPECTED_PENDING_SHA256)
            == pending_sha256,
            "An input changed during preparation",
        )
        temporary_path.replace(OUTPUT_PATH)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise
    require(
        verify_sha256(CANDIDATES_PATH, EXPECTED_CANDIDATES_SHA256)
        == candidate_sha256
        and verify_sha256(PENDING_PATH, EXPECTED_PENDING_SHA256)
        == pending_sha256,
        "An input changed after output publication",
    )

    candidate_keys = {(row["term"], row["qid"]) for row in pending_rows}
    print(f"CANDIDATE_INPUT_SHA256={candidate_sha256}")
    print(f"PENDING_INPUT_SHA256={pending_sha256}")
    print(f"CONTEXT_POLICY_VERSION={CONTEXT_POLICY_VERSION}")
    print(f"TOTAL_ROWS={len(written_rows)}")
    print(f"UNIQUE_QIDS={len({row['qid'] for row in written_rows})}")
    print(f"AFFECTED_CANDIDATES={len(candidate_keys)}")
    print(f"ROWS_WITH_LABEL={counts['label']}")
    print(f"ROWS_WITH_ALIASES={counts['aliases']}")
    print(f"ROWS_WITH_OTHER_DESCRIPTION={counts['other_description']}")
    print(f"ROWS_WITH_P31={counts['p31']}")
    print(f"ROWS_WITH_P279={counts['p279']}")
    print(f"ROWS_WITH_P1343={counts['p1343']}")
    print(f"OUTPUT_SHA256={sha256_file(OUTPUT_PATH)}")
    print(f"OUTPUT={OUTPUT_PATH.name}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    run_preparation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
