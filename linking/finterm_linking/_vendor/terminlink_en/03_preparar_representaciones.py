"""Classify existing Wikidata candidates by available representation evidence."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "02_wikidata_candidatos.csv"
OUTPUT_PATH = SCRIPT_DIR / "03_candidatos_representaciones.csv"
NO_CANDIDATE_OUTPUT_PATH = SCRIPT_DIR / "03_terminos_sin_candidato.csv"

ORIGINAL_COLUMNS = (
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
ADDED_COLUMNS = (
    "representation_route",
    "description_en_word_count",
    "available_other_descriptions",
    "has_aliases",
    "has_p31",
    "has_p279",
    "has_p1343",
)
OUTPUT_COLUMNS = ORIGINAL_COLUMNS + ADDED_COLUMNS

EN_DESCRIPTION = "EN_DESCRIPTION"
OTHER_LANGUAGE_DESCRIPTION = "OTHER_LANGUAGE_DESCRIPTION"
STRUCTURED_EVIDENCE = "STRUCTURED_EVIDENCE"
NO_REPRESENTATION_EVIDENCE = "NO_REPRESENTATION_EVIDENCE"
ROUTES = (
    EN_DESCRIPTION,
    OTHER_LANGUAGE_DESCRIPTION,
    STRUCTURED_EVIDENCE,
    NO_REPRESENTATION_EVIDENCE,
)

EXPECTED_TOTAL_CANDIDATES = 3474
EXPECTED_EN_DESCRIPTION = 2987
EXPECTED_OTHER_LANGUAGE_DESCRIPTION = 328
EXPECTED_STRUCTURED_EVIDENCE = 125
EXPECTED_NO_REPRESENTATION_EVIDENCE = 34
ENGLISH_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

NO_WIKIDATA_CANDIDATE_TERMS = (
    "tool for analysing",
    "integrated economic accounts",
    "European system of national and regional accounts in the European Union",
    "European System of National and Regional Accounts in the Community",
)


def english_word_count(text: str) -> int:
    return len(ENGLISH_WORD_PATTERN.findall(text))


def candidate_key(row: dict[str, str]) -> tuple[str, str]:
    return row["term"], row["qid"]


def read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_columns):
            raise ValueError(
                f"Unexpected columns in {path.name}: {reader.fieldnames!r}"
            )
        rows = [dict(row) for row in reader]

    for row_number, row in enumerate(rows, start=2):
        if any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {row_number} in {path.name}")
    return rows


def parse_json_list(row: dict[str, str], column: str) -> list[Any]:
    try:
        value = json.loads(row[column])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {column} for {candidate_key(row)}") from exc
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list in {column} for {candidate_key(row)}")
    return value


def parse_other_descriptions(row: dict[str, str]) -> dict[str, str]:
    try:
        value = json.loads(row["descriptions_other_languages"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid descriptions JSON for {candidate_key(row)}"
        ) from exc
    if not isinstance(value, dict) or any(
        not isinstance(language, str) or not isinstance(text, str)
        for language, text in value.items()
    ):
        raise ValueError(f"Invalid descriptions object for {candidate_key(row)}")
    return value


def available_other_descriptions(row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"language": language, "text": text}
        for language, text in parse_other_descriptions(row).items()
        if text.strip()
    ]


def enrich_candidate(row: dict[str, str]) -> dict[str, str]:
    description_word_count = english_word_count(row["description_en"])
    other_descriptions = available_other_descriptions(row)
    has_aliases = bool(parse_json_list(row, "aliases_en"))
    has_p31 = bool(parse_json_list(row, "p31"))
    has_p279 = bool(parse_json_list(row, "p279"))
    has_p1343 = bool(parse_json_list(row, "p1343"))

    if description_word_count > 1:
        route = EN_DESCRIPTION
    elif other_descriptions:
        route = OTHER_LANGUAGE_DESCRIPTION
    elif has_aliases or has_p31 or has_p279 or has_p1343:
        route = STRUCTURED_EVIDENCE
    else:
        route = NO_REPRESENTATION_EVIDENCE

    enriched = dict(row)
    enriched.update(
        {
            "representation_route": route,
            "description_en_word_count": str(description_word_count),
            "available_other_descriptions": json.dumps(
                other_descriptions,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "has_aliases": json.dumps(has_aliases),
            "has_p31": json.dumps(has_p31),
            "has_p279": json.dumps(has_p279),
            "has_p1343": json.dumps(has_p1343),
        }
    )
    return enriched


def validate_route_counts(rows: Sequence[dict[str, str]]) -> Counter[str]:
    route_counts = Counter(row["representation_route"] for row in rows)
    unexpected_routes = set(route_counts) - set(ROUTES)
    if unexpected_routes:
        raise ValueError(f"Unexpected representation routes: {unexpected_routes!r}")
    if len(rows) != EXPECTED_TOTAL_CANDIDATES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_CANDIDATES} candidates, found {len(rows)}"
        )
    if route_counts[EN_DESCRIPTION] != EXPECTED_EN_DESCRIPTION:
        raise ValueError(
            f"Expected {EXPECTED_EN_DESCRIPTION} EN descriptions, found "
            f"{route_counts[EN_DESCRIPTION]}"
        )
    if route_counts[OTHER_LANGUAGE_DESCRIPTION] != EXPECTED_OTHER_LANGUAGE_DESCRIPTION:
        raise ValueError(
            f"Expected {EXPECTED_OTHER_LANGUAGE_DESCRIPTION} other-language "
            f"descriptions, found {route_counts[OTHER_LANGUAGE_DESCRIPTION]}"
        )
    if route_counts[STRUCTURED_EVIDENCE] != EXPECTED_STRUCTURED_EVIDENCE:
        raise ValueError(
            f"Expected {EXPECTED_STRUCTURED_EVIDENCE} structured-evidence "
            f"candidates, found {route_counts[STRUCTURED_EVIDENCE]}"
        )
    if route_counts[NO_REPRESENTATION_EVIDENCE] != EXPECTED_NO_REPRESENTATION_EVIDENCE:
        raise ValueError(
            f"Expected {EXPECTED_NO_REPRESENTATION_EVIDENCE} candidates without "
            f"representation evidence, found {route_counts[NO_REPRESENTATION_EVIDENCE]}"
        )
    if sum(route_counts[route] for route in ROUTES) != EXPECTED_TOTAL_CANDIDATES:
        raise ValueError("Representation routes do not cover every candidate exactly once")
    return route_counts


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def validate_candidate_integrity(
    input_rows: Sequence[dict[str, str]],
    output_rows: Sequence[dict[str, str]],
) -> dict[str, int | str]:
    input_counts = Counter(candidate_key(row) for row in input_rows)
    output_counts = Counter(candidate_key(row) for row in output_rows)
    input_duplicates = sum(count - 1 for count in input_counts.values() if count > 1)
    output_duplicates = sum(count - 1 for count in output_counts.values() if count > 1)
    if input_duplicates or output_duplicates:
        raise ValueError(
            f"Duplicate candidate keys: input={input_duplicates}, "
            f"output={output_duplicates}"
        )
    if set(input_counts) != set(output_counts):
        raise ValueError("Candidate term/QID keys changed")

    input_by_key = {candidate_key(row): row for row in input_rows}
    modified_original_fields = 0
    available_description_mismatches = 0
    route_mismatches = 0
    for output_row in output_rows:
        input_row = input_by_key[candidate_key(output_row)]
        for column in ORIGINAL_COLUMNS:
            if output_row[column] != input_row[column]:
                modified_original_fields += 1
        try:
            stored_descriptions = json.loads(
                output_row["available_other_descriptions"]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid available descriptions for {candidate_key(output_row)}"
            ) from exc
        if stored_descriptions != available_other_descriptions(input_row):
            available_description_mismatches += 1
        if output_row["representation_route"] != enrich_candidate(input_row)[
            "representation_route"
        ]:
            route_mismatches += 1
    if modified_original_fields:
        raise ValueError(
            f"Modified original fields detected: {modified_original_fields}"
        )
    if available_description_mismatches:
        raise ValueError(
            "Available multilingual descriptions differ from the original data: "
            f"{available_description_mismatches}"
        )
    if route_mismatches:
        raise ValueError(f"Representation-route mismatches: {route_mismatches}")

    input_terms = {row["term"] for row in input_rows}
    output_terms = {row["term"] for row in output_rows}
    input_qids = {row["qid"] for row in input_rows}
    output_qids = {row["qid"] for row in output_rows}
    if input_terms != output_terms or input_qids != output_qids:
        raise ValueError("Terms or QIDs changed independently of candidate keys")

    return {
        "rows": len(output_rows),
        "duplicate_term_qid": output_duplicates,
        "new_terms": len(output_terms - input_terms),
        "new_qids": len(output_qids - input_qids),
        "removed_original_columns": 0,
        "modified_original_fields": modified_original_fields,
        "available_description_mismatches": available_description_mismatches,
        "route_mismatches": route_mismatches,
        "status": "PASS",
    }


def validate_no_candidate_rows(
    rows: Sequence[dict[str, str]],
    candidate_rows: Sequence[dict[str, str]],
) -> None:
    expected = [
        {"term": term, "status": "NO_WIKIDATA_CANDIDATE"}
        for term in NO_WIKIDATA_CANDIDATE_TERMS
    ]
    if list(rows) != expected:
        raise ValueError("Unexpected no-candidate term output")
    candidate_terms = {row["term"] for row in candidate_rows}
    overlap = candidate_terms & set(NO_WIKIDATA_CANDIDATE_TERMS)
    if overlap:
        raise ValueError(f"No-candidate terms unexpectedly have candidates: {overlap!r}")


def boolean_value(row: dict[str, str], column: str) -> bool:
    if row[column] not in {"true", "false"}:
        raise ValueError(f"Invalid boolean {column} for {candidate_key(row)}")
    return row[column] == "true"


def audit_without_descriptions(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    rows_without_descriptions = [
        row
        for row in rows
        if row["representation_route"]
        in {STRUCTURED_EVIDENCE, NO_REPRESENTATION_EVIDENCE}
    ]
    source_columns = (
        ("aliases", "has_aliases"),
        ("p31", "has_p31"),
        ("p279", "has_p279"),
        ("p1343", "has_p1343"),
    )
    source_counts = {
        source: sum(boolean_value(row, column) for row in rows_without_descriptions)
        for source, column in source_columns
    }
    source_counts["at_least_one"] = sum(
        any(boolean_value(row, column) for _, column in source_columns)
        for row in rows_without_descriptions
    )
    source_counts["none"] = len(rows_without_descriptions) - source_counts["at_least_one"]
    source_counts["total"] = len(rows_without_descriptions)

    combinations = Counter(
        " + ".join(
            source
            for source, column in source_columns
            if boolean_value(row, column)
        )
        or "none"
        for row in rows_without_descriptions
    )
    sorted_combinations = dict(
        sorted(combinations.items(), key=lambda item: (-item[1], item[0]))
    )
    return source_counts, sorted_combinations


def relevant_example(row: dict[str, str]) -> dict[str, Any]:
    example: dict[str, Any] = {
        "term": row["term"],
        "qid": row["qid"],
        "label_en": row["label_en"],
        "representation_route": row["representation_route"],
    }
    route = row["representation_route"]
    if route == EN_DESCRIPTION:
        example.update(
            {
                "description_en": row["description_en"],
                "description_en_word_count": int(row["description_en_word_count"]),
            }
        )
    elif route == OTHER_LANGUAGE_DESCRIPTION:
        example.update(
            {
                "description_en": row["description_en"],
                "available_other_descriptions": json.loads(
                    row["available_other_descriptions"]
                ),
            }
        )
    elif route == STRUCTURED_EVIDENCE:
        example.update(
            {
                "aliases_en": json.loads(row["aliases_en"]),
                "p31": json.loads(row["p31"]),
                "p279": json.loads(row["p279"]),
                "p1343": json.loads(row["p1343"]),
            }
        )
    else:
        example.update(
            {
                "description_en": row["description_en"],
                "available_other_descriptions": json.loads(
                    row["available_other_descriptions"]
                ),
                "has_aliases": boolean_value(row, "has_aliases"),
                "has_p31": boolean_value(row, "has_p31"),
                "has_p279": boolean_value(row, "has_p279"),
                "has_p1343": boolean_value(row, "has_p1343"),
            }
        )
    return example


def print_audit(
    rows: Sequence[dict[str, str]],
    route_counts: Counter[str],
    integrity: dict[str, int | str],
) -> None:
    source_counts, combinations = audit_without_descriptions(rows)
    other_language_rows = [
        row
        for row in rows
        if row["representation_route"] == OTHER_LANGUAGE_DESCRIPTION
    ]
    available_lists = [
        json.loads(row["available_other_descriptions"])
        for row in other_language_rows
    ]
    language_counts = Counter(
        description["language"]
        for descriptions in available_lists
        for description in descriptions
    )
    english_variant_counts = Counter(
        {
            language: count
            for language, count in language_counts.items()
            if language.casefold().split("-", 1)[0] == "en"
        }
    )
    translatable_language_counts = Counter(
        {
            language: count
            for language, count in language_counts.items()
            if language.casefold().split("-", 1)[0] != "en"
        }
    )
    total_available_other = sum(map(len, available_lists))
    average_available_other = total_available_other / len(other_language_rows)
    max_available_other = max(map(len, available_lists), default=0)

    p1343_rows = [row for row in rows if boolean_value(row, "has_p1343")]
    structured_p1343_rows = [
        row
        for row in p1343_rows
        if row["representation_route"] == STRUCTURED_EVIDENCE
    ]
    p1343_examples = [
        {
            "term": row["term"],
            "qid": row["qid"],
            "label_en": row["label_en"],
            "p1343": json.loads(row["p1343"]),
        }
        for row in p1343_rows[:10]
    ]
    route_examples = {
        route: [
            relevant_example(row)
            for row in rows
            if row["representation_route"] == route
        ][:10]
        for route in ROUTES
    }

    print(f"TOTAL_CANDIDATES={len(rows)}")
    for route in ROUTES:
        print(f"{route}={route_counts[route]}")
    print(
        "ROUTE_SUM="
        + str(sum(route_counts[route] for route in ROUTES))
    )
    print(
        "NO_AVAILABLE_DESCRIPTION_AUDIT="
        + json.dumps(source_counts, ensure_ascii=False, separators=(",", ":"))
    )
    print(
        "NO_AVAILABLE_DESCRIPTION_COMBINATIONS="
        + json.dumps(combinations, ensure_ascii=False, separators=(",", ":"))
    )
    print(
        "OTHER_LANGUAGE_AUDIT="
        + json.dumps(
            {
                "candidates": len(other_language_rows),
                "total_available_descriptions": total_available_other,
                "total_requiring_translation": sum(
                    translatable_language_counts.values()
                ),
                "average_available_descriptions": round(
                    average_available_other, 6
                ),
                "max_available_descriptions": max_available_other,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    print(
        "TRANSLATABLE_LANGUAGE_DISTRIBUTION="
        + json.dumps(
            dict(
                sorted(
                    translatable_language_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    print(
        "ENGLISH_VARIANT_AUDIT="
        + json.dumps(
            dict(
                sorted(
                    english_variant_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    print(
        "P1343_AUDIT="
        + json.dumps(
            {
                "all_candidates": len(p1343_rows),
                "structured_evidence": len(structured_p1343_rows),
            },
            separators=(",", ":"),
        )
    )
    print(
        "P1343_EXAMPLES="
        + json.dumps(p1343_examples, ensure_ascii=False, separators=(",", ":"))
    )
    print(
        "ROUTE_EXAMPLES="
        + json.dumps(route_examples, ensure_ascii=False, separators=(",", ":"))
    )
    print(
        "NO_WIKIDATA_CANDIDATE_TERMS="
        + json.dumps(NO_WIKIDATA_CANDIDATE_TERMS, ensure_ascii=False)
    )
    print(
        "INTEGRITY="
        + json.dumps(integrity, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"OUTPUT_CSV={OUTPUT_PATH}")
    print(f"NO_CANDIDATE_OUTPUT_CSV={NO_CANDIDATE_OUTPUT_PATH}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_rows = read_csv(INPUT_PATH, ORIGINAL_COLUMNS)
    input_counts = Counter(candidate_key(row) for row in input_rows)
    duplicate_input_rows = sum(
        count - 1 for count in input_counts.values() if count > 1
    )
    if duplicate_input_rows:
        raise ValueError(f"Input contains {duplicate_input_rows} duplicate candidates")

    output_rows = [enrich_candidate(row) for row in input_rows]
    route_counts = validate_route_counts(output_rows)
    validate_candidate_integrity(input_rows, output_rows)

    no_candidate_rows = read_csv(
        NO_CANDIDATE_OUTPUT_PATH,
        ("term", "status"),
    )
    validate_no_candidate_rows(no_candidate_rows, input_rows)

    write_csv(OUTPUT_PATH, OUTPUT_COLUMNS, output_rows)

    written_rows = read_csv(OUTPUT_PATH, OUTPUT_COLUMNS)
    written_no_candidate_rows = read_csv(
        NO_CANDIDATE_OUTPUT_PATH,
        ("term", "status"),
    )
    written_route_counts = validate_route_counts(written_rows)
    integrity = validate_candidate_integrity(input_rows, written_rows)
    validate_no_candidate_rows(written_no_candidate_rows, written_rows)
    print_audit(written_rows, written_route_counts, integrity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
