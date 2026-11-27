#!/usr/bin/env python3
"""Prepare and audit contextual V2 DefTerm inputs without running any model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent

NEW_PENDING_FILE = "03_defterm_pendientes_contextual_v2.csv"
OLD_PENDING_FILE = "03_defterm_pendientes.csv"
OLD_PREPARER_FILE = "04_preparar_defterm.py"
OLD_INVENTORY_FILE = "04_defterm_entradas.csv"
OLD_SAMPLE_FILE = "04_defterm_muestra_evaluacion.csv"
SUFFICIENCY_REFERENCE_FILE = "05_evaluacion_suficiencia_evidencia_COMPLETADA.csv"
FEWSHOT_FILE = "05_fewshot_ejemplos_aprobados.csv"
FROZEN_EXPERIMENT_FILE = "05_defterm_qwen14_few_shot_abstention.py"

INVENTORY_OUTPUT_FILE = "04_defterm_entradas_contextual_v2.csv"
SAMPLE_AUDIT_OUTPUT_FILE = "04_auditoria_muestra_defterm_contextual_v2.csv"
FEWSHOT_AUDIT_OUTPUT_FILE = "04_auditoria_fewshot_contextual_v2.csv"
COMPARISON_OUTPUT_FILE = "04_comparacion_defterm_nllb_vs_contextual_v2.csv"
FORBIDDEN_SAMPLE_OUTPUT_FILE = "04_defterm_muestra_evaluacion_contextual_v2.csv"
LOCK_FILE = ".04_preparar_defterm_contextual_v2.lock"

EXPECTED_HASHES = {
    NEW_PENDING_FILE: "F5B846EC9EC351765E49841A56F5E921706C2A4CF727440B74DE25E1558B95C8",
    OLD_PENDING_FILE: "650E670E8F64D3E03015CE876928A0767B7B3487181E6AB90642DBBA2D1F3EDB",
    OLD_PREPARER_FILE: "67CB9BEF127EEF430588CAABC558A3C22B5B48DF841ECD62B0BF92046E30EA25",
    OLD_INVENTORY_FILE: "F9CDC47AA4F99095E5A7355A3049AF798A0703C1A4D81BE45ECDE5A0A71D9BE1",
    OLD_SAMPLE_FILE: "A77C2D635CF8F0793ED550C1ABAB363BB2DBA5F37DD72A4EB7B1623374A4358C",
    SUFFICIENCY_REFERENCE_FILE: "758E22B00EA31C2A0F92DBA5A48B8165C161B3F4F7F3800092E8E8238915B6E2",
    FEWSHOT_FILE: "AC83C1996824E1D2645FCBCC56FBA14282572498DA8000238C3A77D50B088026",
    FROZEN_EXPERIMENT_FILE: "F71C302B197672289A655962616545FDDF3108EA9CCB422E6CE10395819EA3F6",
}

EXPECTED_NEW_COUNT = 150
EXPECTED_OLD_COUNT = 136
EXPECTED_SAMPLE_COUNT = 20
EXPECTED_SUFFICIENCY_COUNT = 20
EXPECTED_FEWSHOT_COUNT = 6
EXPECTED_NEW_ROUTE_COUNTS = {
    "STRUCTURED_EVIDENCE": 125,
    "OTHER_LANGUAGE_DESCRIPTION": 25,
}

QID_PATTERN = re.compile(r"Q[1-9][0-9]*")
DEFTERM_INPUT_ID_PATTERN = re.compile(r"defterm_input_[0-9a-f]{64}")
FEWSHOT_ID_PATTERN = re.compile(r"fewshot_example_[0-9a-f]{64}")

MODEL_LOADED = False
INFERENCE_EXECUTED = False
FORBIDDEN_MODEL_MODULES = frozenset(
    {
        "accelerate",
        "bitsandbytes",
        "ctranslate2",
        "huggingface_hub",
        "llama_cpp",
        "onnxruntime",
        "sentence_transformers",
        "sentencepiece",
        "torch",
        "transformers",
    }
)

REQUIRED_TESTS = (
    "INPUT_HASH_TEST",
    "DEFTERM_COUNT_TEST",
    "TERM_QID_UNIQUENESS_TEST",
    "EVIDENCE_STRUCTURE_TEST",
    "EVIDENCE_TEXT_DETERMINISM_TEST",
    "NO_SENSE1_CONTAMINATION_TEST",
    "OLD_INVENTORY_IMMUTABILITY_TEST",
    "OLD_SAMPLE_IMMUTABILITY_TEST",
    "FEWSHOT_IMMUTABILITY_TEST",
    "OLD_NEW_MEMBERSHIP_TEST",
    "OLD_SAMPLE_OVERLAP_TEST",
    "FEWSHOT_OVERLAP_TEST",
    "EVALUATION_ISOLATION_TEST",
    "DETERMINISM_TEST",
)

PENDING_FIELDS = (
    "term",
    "qid",
    "label_en",
    "aliases_en",
    "p31",
    "p279",
    "p1343",
    "representation_route",
    "sense2_status",
)

OLD_INVENTORY_FIELDS = (
    "defterm_input_id",
    "term",
    "qid",
    "label_en",
    "aliases_en",
    "p31",
    "p279",
    "p1343",
    "has_aliases",
    "has_p31",
    "has_p279",
    "has_p1343",
    "aliases_count",
    "p31_count",
    "p279_count",
    "p1343_count",
    "evidence_profile",
    "evidence_sparsity",
    "defterm_evidence_text",
)

INVENTORY_FIELDS = (
    "defterm_input_id",
    "term",
    "qid",
    "label_en",
    "aliases_en",
    "p31",
    "p279",
    "p1343",
    "representation_route",
    "has_aliases",
    "has_p31",
    "has_p279",
    "has_p1343",
    "aliases_count",
    "p31_count",
    "p279_count",
    "p1343_count",
    "evidence_profile",
    "evidence_sparsity",
    "defterm_evidence_text",
)

OLD_SAMPLE_FIELDS = (*OLD_INVENTORY_FIELDS, "evaluation_selection_reason")

SUFFICIENCY_FIELDS = (
    "defterm_input_id",
    "term",
    "qid",
    "label_en",
    "evidence_profile",
    "evidence_sparsity",
    "defterm_evidence_text",
    "evaluation_selection_reason",
    "evidence_sufficiency",
    "evidence_sufficiency_comment",
)

FEWSHOT_FIELDS = (
    "fewshot_example_id",
    "defterm_input_id",
    "term",
    "qid",
    "label_en",
    "evidence_profile",
    "evidence_sparsity",
    "defterm_evidence_text",
    "manual_definition",
    "manual_status",
    "manual_notes",
)

SAMPLE_AUDIT_FIELDS = (
    "historical_defterm_input_id",
    "new_defterm_input_id",
    "term",
    "qid",
    "present_in_new_inventory",
    "evidence_identical",
    "old_evidence_profile",
    "new_evidence_profile",
    "old_evidence_sparsity",
    "new_evidence_sparsity",
    "old_defterm_evidence_text",
    "new_defterm_evidence_text",
    "historical_evaluation_selection_reason",
    "old_representation_route",
    "new_representation_route",
    "sufficiency_reference_present",
    "sufficiency_reference_evidence_matches_historical_sample",
    "sufficiency_reference_status",
)

FEWSHOT_AUDIT_FIELDS = (
    "fewshot_example_id",
    "historical_defterm_input_id",
    "new_defterm_input_id",
    "term",
    "qid",
    "present_in_new_inventory",
    "evidence_identical",
    "old_evidence_profile",
    "new_evidence_profile",
    "old_evidence_sparsity",
    "new_evidence_sparsity",
    "old_defterm_evidence_text",
    "new_defterm_evidence_text",
    "old_representation_route",
    "new_representation_route",
    "overlap_with_historical_evaluation_sample",
    "future_evaluation_eligibility",
    "representativeness_status",
)

COMPARISON_FIELDS = (
    "term",
    "qid",
    "membership_status",
    "old_defterm_input_id",
    "new_defterm_input_id",
    "evidence_identical",
    "old_label_en",
    "new_label_en",
    "old_evidence_profile",
    "new_evidence_profile",
    "old_evidence_sparsity",
    "new_evidence_sparsity",
    "old_defterm_evidence_text",
    "new_defterm_evidence_text",
    "old_representation_route",
    "new_representation_route",
)

PROFILE_COMPONENTS = (
    ("ALIASES", "has_aliases"),
    ("P31", "has_p31"),
    ("P279", "has_p279"),
    ("P1343", "has_p1343"),
)

SPARSITY_ALIASES_ONLY = "ALIASES_ONLY"
SPARSITY_SINGLE_RELATION = "SINGLE_STRUCTURED_RELATION"
SPARSITY_MULTIPLE = "MULTIPLE_EVIDENCE"

EVIDENCE_SIGNATURE_FIELDS = (
    "label_en",
    "aliases_en",
    "p31",
    "p279",
    "p1343",
    "has_aliases",
    "has_p31",
    "has_p279",
    "has_p1343",
    "aliases_count",
    "p31_count",
    "p279_count",
    "p1343_count",
    "evidence_profile",
    "evidence_sparsity",
    "defterm_evidence_text",
)

FORBIDDEN_INPUT_INFLUENCE_FIELDS = frozenset(
    {
        "context",
        "corpus_context",
        "sentido1",
        "sense1",
        "translation",
        "translated_description_en",
        "translation_clean",
        "termlink",
        "embedding",
        "gold",
        "evidence_sufficiency",
        "evidence_sufficiency_comment",
        "manual_definition",
        "manual_status",
        "manual_notes",
        "llm_decision",
    }
)


class ValidationError(RuntimeError):
    """Raised when a fixed input or derived artifact violates its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def pass_test(tests: dict[str, str], name: str, condition: bool, message: str) -> None:
    require(name in REQUIRED_TESTS, f"Unknown self-test {name}")
    require(condition, f"{name}: {message}")
    tests[name] = "PASS"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def snapshot_hashes(filenames: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = BASE_DIR / filename
        require(path.is_file(), f"{filename}: required file does not exist")
        actual = sha256_bytes(path.read_bytes())
        expected = EXPECTED_HASHES[filename]
        require(
            actual == expected,
            f"{filename}: SHA-256 mismatch; expected {expected}, got {actual}",
        )
        hashes[filename] = actual
    return hashes


def parse_csv_bytes(
    filename: str, payload: bytes, expected_fields: Sequence[str]
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{filename}: input is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    actual_fields = tuple(reader.fieldnames or ())
    require(
        actual_fields == tuple(expected_fields),
        f"{filename}: unexpected schema {actual_fields!r}",
    )
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        require(None not in row, f"{filename}:{row_number}: surplus CSV fields")
        require(
            all(value is not None for value in row.values()),
            f"{filename}:{row_number}: missing CSV field",
        )
        rows.append({field: row[field] for field in expected_fields})
    return rows


def read_fixed_csv(
    filename: str, expected_fields: Sequence[str]
) -> tuple[list[dict[str, str]], str]:
    path = BASE_DIR / filename
    require(path.is_file(), f"{filename}: required input does not exist")
    payload = path.read_bytes()
    actual_hash = sha256_bytes(payload)
    require(
        actual_hash == EXPECTED_HASHES[filename],
        f"{filename}: SHA-256 mismatch; expected {EXPECTED_HASHES[filename]}, "
        f"got {actual_hash}",
    )
    return parse_csv_bytes(filename, payload, expected_fields), actual_hash


def parse_json(raw: str, filename: str, row_number: int, field: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"{filename}:{row_number}: invalid JSON in {field}: {exc.msg}"
        ) from exc


def validate_aliases(
    value: Any, filename: str, row_number: int
) -> list[str]:
    require(
        isinstance(value, list),
        f"{filename}:{row_number}: aliases_en must be a JSON list",
    )
    require(
        all(isinstance(alias, str) and alias.strip() for alias in value),
        f"{filename}:{row_number}: aliases_en contains an invalid alias",
    )
    return value


def validate_relations(
    value: Any, filename: str, row_number: int, field: str
) -> list[dict[str, str]]:
    require(
        isinstance(value, list),
        f"{filename}:{row_number}: {field} must be a JSON list",
    )
    for item_number, item in enumerate(value, start=1):
        require(
            isinstance(item, dict) and set(item) == {"qid", "label_en"},
            f"{filename}:{row_number}: malformed {field} item {item_number}",
        )
        require(
            isinstance(item["qid"], str)
            and QID_PATTERN.fullmatch(item["qid"]) is not None,
            f"{filename}:{row_number}: invalid QID in {field} item {item_number}",
        )
        require(
            isinstance(item["label_en"], str)
            and (item["label_en"] == "" or item["label_en"].strip()),
            f"{filename}:{row_number}: invalid label in {field} item {item_number}",
        )
    return value


def evidence_profile(flags: dict[str, bool]) -> str:
    components = [name for name, flag in PROFILE_COMPONENTS if flags[flag]]
    require(bool(components), "candidate has no DefTerm evidence")
    if len(components) == 1:
        return f"{components[0]}_ONLY"
    return "_".join(components)


def evidence_sparsity(
    aliases_count: int, p31_count: int, p279_count: int, p1343_count: int
) -> str:
    relation_count = p31_count + p279_count + p1343_count
    if aliases_count > 0 and relation_count == 0:
        return SPARSITY_ALIASES_ONLY
    if aliases_count == 0 and relation_count == 1:
        return SPARSITY_SINGLE_RELATION
    return SPARSITY_MULTIPLE


def build_evidence_text(
    label_en: str,
    aliases: list[str],
    p31: list[dict[str, str]],
    p279: list[dict[str, str]],
    p1343: list[dict[str, str]],
) -> str:
    sections: list[str] = []
    if label_en:
        sections.append(f"Label: {label_en}")
    if aliases:
        sections.append(f"Aliases: {'; '.join(aliases)}")
    for heading, relations in (
        ("Instance of", p31),
        ("Subclass of", p279),
        ("Described by source", p1343),
    ):
        labels = [relation["label_en"] for relation in relations if relation["label_en"]]
        if labels:
            sections.append(f"{heading}: {'; '.join(labels)}")
    return "\n".join(sections)


def make_defterm_input_id(
    term: str,
    qid: str,
    label_en: str,
    aliases: list[str],
    p31: list[dict[str, str]],
    p279: list[dict[str, str]],
    p1343: list[dict[str, str]],
) -> str:
    canonical = json.dumps(
        {
            "aliases_en": aliases,
            "label_en": label_en,
            "p1343": p1343,
            "p279": p279,
            "p31": p31,
            "qid": qid,
            "term": term,
            "version": "defterm-input-v1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "defterm_input_" + hashlib.sha256(canonical).hexdigest()


def prepare_inventory(
    input_rows: list[dict[str, str]], filename: str, expected_count: int
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    require(
        len(input_rows) == expected_count,
        f"{filename}: expected {expected_count} rows, got {len(input_rows)}",
    )
    prepared_rows: list[dict[str, str]] = []
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    ids: set[str] = set()
    for row_number, row in enumerate(input_rows, start=2):
        term = row["term"]
        qid = row["qid"]
        label_en = row["label_en"]
        key = (term, qid)
        require(term.strip(), f"{filename}:{row_number}: empty term")
        require(
            QID_PATTERN.fullmatch(qid) is not None,
            f"{filename}:{row_number}: invalid QID {qid!r}",
        )
        require(
            label_en == "" or label_en.strip(),
            f"{filename}:{row_number}: whitespace-only label",
        )
        require(
            row["sense2_status"] == "DEFTERM_PENDING",
            f"{filename}:{row_number}: status is not DEFTERM_PENDING",
        )
        require(key not in metadata, f"{filename}:{row_number}: duplicate term+qid {key!r}")
        route = row["representation_route"]
        require(
            route in {"STRUCTURED_EVIDENCE", "OTHER_LANGUAGE_DESCRIPTION"},
            f"{filename}:{row_number}: unexpected representation route {route!r}",
        )
        aliases = validate_aliases(
            parse_json(row["aliases_en"], filename, row_number, "aliases_en"),
            filename,
            row_number,
        )
        p31 = validate_relations(
            parse_json(row["p31"], filename, row_number, "p31"),
            filename,
            row_number,
            "p31",
        )
        p279 = validate_relations(
            parse_json(row["p279"], filename, row_number, "p279"),
            filename,
            row_number,
            "p279",
        )
        p1343 = validate_relations(
            parse_json(row["p1343"], filename, row_number, "p1343"),
            filename,
            row_number,
            "p1343",
        )
        counts = {
            "aliases_count": len(aliases),
            "p31_count": len(p31),
            "p279_count": len(p279),
            "p1343_count": len(p1343),
        }
        flags = {
            "has_aliases": counts["aliases_count"] > 0,
            "has_p31": counts["p31_count"] > 0,
            "has_p279": counts["p279_count"] > 0,
            "has_p1343": counts["p1343_count"] > 0,
        }
        profile = evidence_profile(flags)
        sparsity = evidence_sparsity(**counts)
        require(
            not flags["has_p1343"]
            or flags["has_aliases"]
            or flags["has_p31"]
            or flags["has_p279"],
            f"{filename}:{row_number}: candidate depends only on exploratory P1343",
        )
        evidence_text = build_evidence_text(label_en, aliases, p31, p279, p1343)
        require(evidence_text, f"{filename}:{row_number}: empty DefTerm evidence text")
        evidence_qids = [
            relation["qid"] for relations in (p31, p279, p1343) for relation in relations
        ]
        require(
            all(evidence_qid not in evidence_text for evidence_qid in [qid, *evidence_qids]),
            f"{filename}:{row_number}: QID leaked into evidence text",
        )
        candidate_id = make_defterm_input_id(
            term, qid, label_en, aliases, p31, p279, p1343
        )
        require(candidate_id not in ids, f"{filename}:{row_number}: duplicate DefTerm ID")
        ids.add(candidate_id)
        prepared = {
            "defterm_input_id": candidate_id,
            "term": term,
            "qid": qid,
            "label_en": label_en,
            "aliases_en": row["aliases_en"],
            "p31": row["p31"],
            "p279": row["p279"],
            "p1343": row["p1343"],
            "representation_route": route,
            "has_aliases": bool_text(flags["has_aliases"]),
            "has_p31": bool_text(flags["has_p31"]),
            "has_p279": bool_text(flags["has_p279"]),
            "has_p1343": bool_text(flags["has_p1343"]),
            "aliases_count": str(counts["aliases_count"]),
            "p31_count": str(counts["p31_count"]),
            "p279_count": str(counts["p279_count"]),
            "p1343_count": str(counts["p1343_count"]),
            "evidence_profile": profile,
            "evidence_sparsity": sparsity,
            "defterm_evidence_text": evidence_text,
        }
        prepared_rows.append(prepared)
        metadata[key] = {
            "source": row,
            "prepared": prepared,
            "aliases": aliases,
            "p31": p31,
            "p279": p279,
            "p1343": p1343,
            "counts": counts,
            "flags": flags,
        }
    return prepared_rows, metadata


def without_route(row: dict[str, str]) -> dict[str, str]:
    return {field: row[field] for field in OLD_INVENTORY_FIELDS}


def evidence_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in EVIDENCE_SIGNATURE_FIELDS)


def validate_historical_inventory(
    old_pending_rows: list[dict[str, str]],
    old_inventory_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    rebuilt, metadata = prepare_inventory(
        old_pending_rows, OLD_PENDING_FILE, EXPECTED_OLD_COUNT
    )
    require(
        [without_route(row) for row in rebuilt] == old_inventory_rows,
        f"{OLD_INVENTORY_FILE}: bytes decode to content different from historical algorithm",
    )
    return rebuilt, metadata


def index_unique(
    rows: list[dict[str, str]], filename: str
) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = (row["term"], row["qid"])
        require(key not in indexed, f"{filename}:{row_number}: duplicate term+qid {key!r}")
        indexed[key] = row
    return indexed


def validate_old_sample(
    rows: list[dict[str, str]], old_inventory: dict[tuple[str, str], dict[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    require(
        len(rows) == EXPECTED_SAMPLE_COUNT,
        f"{OLD_SAMPLE_FILE}: expected {EXPECTED_SAMPLE_COUNT} rows, got {len(rows)}",
    )
    indexed = index_unique(rows, OLD_SAMPLE_FILE)
    ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        key = (row["term"], row["qid"])
        require(key in old_inventory, f"{OLD_SAMPLE_FILE}:{row_number}: unknown candidate")
        require(
            {field: row[field] for field in OLD_INVENTORY_FIELDS} == old_inventory[key],
            f"{OLD_SAMPLE_FILE}:{row_number}: evidence differs from old inventory",
        )
        require(
            row["defterm_input_id"] not in ids
            and row["evaluation_selection_reason"].strip(),
            f"{OLD_SAMPLE_FILE}:{row_number}: duplicate ID or missing selection reason",
        )
        ids.add(row["defterm_input_id"])
    return indexed


def validate_sufficiency_reference(
    rows: list[dict[str, str]], sample: dict[tuple[str, str], dict[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    require(
        len(rows) == EXPECTED_SUFFICIENCY_COUNT,
        f"{SUFFICIENCY_REFERENCE_FILE}: expected {EXPECTED_SUFFICIENCY_COUNT} rows, "
        f"got {len(rows)}",
    )
    indexed = index_unique(rows, SUFFICIENCY_REFERENCE_FILE)
    require(
        set(indexed) == set(sample),
        f"{SUFFICIENCY_REFERENCE_FILE}: evaluated keys differ from historical sample",
    )
    for row_number, row in enumerate(rows, start=2):
        historical = sample[(row["term"], row["qid"])]
        for field in (
            "defterm_input_id",
            "term",
            "qid",
            "label_en",
            "evidence_profile",
            "evidence_sparsity",
            "defterm_evidence_text",
            "evaluation_selection_reason",
        ):
            require(
                row[field] == historical[field],
                f"{SUFFICIENCY_REFERENCE_FILE}:{row_number}: {field} differs from sample",
            )
        require(
            row["evidence_sufficiency"].strip()
            and row["evidence_sufficiency_comment"].strip(),
            f"{SUFFICIENCY_REFERENCE_FILE}:{row_number}: incomplete assisted evaluation",
        )
    return indexed


def validate_fewshot(
    rows: list[dict[str, str]], old_inventory: dict[tuple[str, str], dict[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    require(
        len(rows) == EXPECTED_FEWSHOT_COUNT,
        f"{FEWSHOT_FILE}: expected {EXPECTED_FEWSHOT_COUNT} rows, got {len(rows)}",
    )
    indexed = index_unique(rows, FEWSHOT_FILE)
    seen_example_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        key = (row["term"], row["qid"])
        require(key in old_inventory, f"{FEWSHOT_FILE}:{row_number}: unknown candidate")
        historical = old_inventory[key]
        require(
            FEWSHOT_ID_PATTERN.fullmatch(row["fewshot_example_id"]) is not None
            and row["fewshot_example_id"] not in seen_example_ids,
            f"{FEWSHOT_FILE}:{row_number}: invalid or duplicate example ID",
        )
        seen_example_ids.add(row["fewshot_example_id"])
        for field in (
            "defterm_input_id",
            "term",
            "qid",
            "label_en",
            "evidence_profile",
            "evidence_sparsity",
            "defterm_evidence_text",
        ):
            require(
                row[field] == historical[field],
                f"{FEWSHOT_FILE}:{row_number}: {field} differs from old inventory",
            )
        require(
            row["manual_status"] == "APPROVED_MANUAL"
            and row["manual_definition"].strip()
            and row["manual_notes"].strip(),
            f"{FEWSHOT_FILE}:{row_number}: approved manual example is incomplete",
        )
    return indexed


def build_sample_audit(
    sample_rows: list[dict[str, str]],
    old_inventory: dict[tuple[str, str], dict[str, str]],
    old_prepared: dict[tuple[str, str], dict[str, str]],
    new_inventory: dict[tuple[str, str], dict[str, str]],
    sufficiency: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []
    for sample in sample_rows:
        key = (sample["term"], sample["qid"])
        old = old_inventory[key]
        new = new_inventory.get(key)
        reference = sufficiency.get(key)
        present = new is not None
        identical = present and evidence_signature(old) == evidence_signature(new)
        reference_matches_old = reference is not None and all(
            reference[field] == sample[field]
            for field in (
                "defterm_input_id",
                "term",
                "qid",
                "label_en",
                "evidence_profile",
                "evidence_sparsity",
                "defterm_evidence_text",
                "evaluation_selection_reason",
            )
        )
        if not present:
            reference_status = "CANDIDATE_LEFT_DEFTERM"
        elif not identical:
            reference_status = "EVIDENCE_CHANGED_DO_NOT_TRANSFER_LABEL"
        else:
            reference_status = "CORRESPONDING_CANDIDATE_UNCHANGED"
        audit_rows.append(
            {
                "historical_defterm_input_id": old["defterm_input_id"],
                "new_defterm_input_id": new["defterm_input_id"] if new else "",
                "term": key[0],
                "qid": key[1],
                "present_in_new_inventory": bool_text(present),
                "evidence_identical": bool_text(identical) if present else "",
                "old_evidence_profile": old["evidence_profile"],
                "new_evidence_profile": new["evidence_profile"] if new else "",
                "old_evidence_sparsity": old["evidence_sparsity"],
                "new_evidence_sparsity": new["evidence_sparsity"] if new else "",
                "old_defterm_evidence_text": old["defterm_evidence_text"],
                "new_defterm_evidence_text": new["defterm_evidence_text"] if new else "",
                "historical_evaluation_selection_reason": sample[
                    "evaluation_selection_reason"
                ],
                "old_representation_route": old_prepared[key]["representation_route"],
                "new_representation_route": new["representation_route"] if new else "",
                "sufficiency_reference_present": bool_text(reference is not None),
                "sufficiency_reference_evidence_matches_historical_sample": bool_text(
                    reference_matches_old
                ),
                "sufficiency_reference_status": reference_status,
            }
        )
    return audit_rows


def build_fewshot_audit(
    fewshot_rows: list[dict[str, str]],
    old_inventory: dict[tuple[str, str], dict[str, str]],
    old_prepared: dict[tuple[str, str], dict[str, str]],
    new_inventory: dict[tuple[str, str], dict[str, str]],
    historical_sample_keys: set[tuple[str, str]],
) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []
    for fewshot in fewshot_rows:
        key = (fewshot["term"], fewshot["qid"])
        old = old_inventory[key]
        new = new_inventory.get(key)
        present = new is not None
        identical = present and evidence_signature(old) == evidence_signature(new)
        overlap = key in historical_sample_keys
        if not present:
            representativeness = "NO_LONGER_IN_DEFTERM"
        elif not identical:
            representativeness = "REVIEW_REQUIRED_EVIDENCE_CHANGED"
        else:
            representativeness = "STILL_REPRESENTATIVE_UNCHANGED_EVIDENCE"
        audit_rows.append(
            {
                "fewshot_example_id": fewshot["fewshot_example_id"],
                "historical_defterm_input_id": old["defterm_input_id"],
                "new_defterm_input_id": new["defterm_input_id"] if new else "",
                "term": key[0],
                "qid": key[1],
                "present_in_new_inventory": bool_text(present),
                "evidence_identical": bool_text(identical) if present else "",
                "old_evidence_profile": old["evidence_profile"],
                "new_evidence_profile": new["evidence_profile"] if new else "",
                "old_evidence_sparsity": old["evidence_sparsity"],
                "new_evidence_sparsity": new["evidence_sparsity"] if new else "",
                "old_defterm_evidence_text": old["defterm_evidence_text"],
                "new_defterm_evidence_text": new["defterm_evidence_text"] if new else "",
                "old_representation_route": old_prepared[key]["representation_route"],
                "new_representation_route": new["representation_route"] if new else "",
                "overlap_with_historical_evaluation_sample": bool_text(overlap),
                "future_evaluation_eligibility": "EXCLUDE_APPROVED_FEWSHOT",
                "representativeness_status": representativeness,
            }
        )
    return audit_rows


def build_comparison(
    old_rows: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    old_by_key = {(row["term"], row["qid"]): row for row in old_rows}
    new_by_key = {(row["term"], row["qid"]): row for row in new_rows}
    ordered_keys = list(old_by_key)
    ordered_keys.extend(key for key in new_by_key if key not in old_by_key)
    comparison: list[dict[str, str]] = []
    for key in ordered_keys:
        old = old_by_key.get(key)
        new = new_by_key.get(key)
        if old is not None and new is not None:
            membership = "STILL_PENDING"
            identical = bool_text(evidence_signature(old) == evidence_signature(new))
        elif old is None:
            membership = "ENTERED_DEFTERM"
            identical = ""
        else:
            membership = "LEFT_DEFTERM"
            identical = ""
        comparison.append(
            {
                "term": key[0],
                "qid": key[1],
                "membership_status": membership,
                "old_defterm_input_id": old["defterm_input_id"] if old else "",
                "new_defterm_input_id": new["defterm_input_id"] if new else "",
                "evidence_identical": identical,
                "old_label_en": old["label_en"] if old else "",
                "new_label_en": new["label_en"] if new else "",
                "old_evidence_profile": old["evidence_profile"] if old else "",
                "new_evidence_profile": new["evidence_profile"] if new else "",
                "old_evidence_sparsity": old["evidence_sparsity"] if old else "",
                "new_evidence_sparsity": new["evidence_sparsity"] if new else "",
                "old_defterm_evidence_text": old["defterm_evidence_text"] if old else "",
                "new_defterm_evidence_text": new["defterm_evidence_text"] if new else "",
                "old_representation_route": old["representation_route"] if old else "",
                "new_representation_route": new["representation_route"] if new else "",
            }
        )
    return comparison


def validate_new_inventory(
    input_rows: list[dict[str, str]],
    prepared_rows: list[dict[str, str]],
    metadata: dict[tuple[str, str], dict[str, Any]],
) -> None:
    require(len(prepared_rows) == EXPECTED_NEW_COUNT, "new inventory row count differs")
    require(
        [(row["term"], row["qid"]) for row in prepared_rows]
        == [(row["term"], row["qid"]) for row in input_rows],
        f"{INVENTORY_OUTPUT_FILE}: candidate coverage or order differs",
    )
    ids: set[str] = set()
    for row_number, row in enumerate(prepared_rows, start=2):
        key = (row["term"], row["qid"])
        item = metadata[key]
        source = item["source"]
        require(
            DEFTERM_INPUT_ID_PATTERN.fullmatch(row["defterm_input_id"]) is not None
            and row["defterm_input_id"] not in ids,
            f"{INVENTORY_OUTPUT_FILE}:{row_number}: invalid or duplicate ID",
        )
        ids.add(row["defterm_input_id"])
        require(
            row == item["prepared"],
            f"{INVENTORY_OUTPUT_FILE}:{row_number}: row differs from validated build",
        )
        require(
            row["term"] == source["term"]
            and row["qid"] == source["qid"]
            and row["label_en"] == source["label_en"]
            and row["aliases_en"] == source["aliases_en"]
            and row["p31"] == source["p31"]
            and row["p279"] == source["p279"]
            and row["p1343"] == source["p1343"]
            and row["representation_route"] == source["representation_route"],
            f"{INVENTORY_OUTPUT_FILE}:{row_number}: source evidence changed",
        )
        expected_text = build_evidence_text(
            source["label_en"],
            item["aliases"],
            item["p31"],
            item["p279"],
            item["p1343"],
        )
        require(
            row["defterm_evidence_text"] == expected_text,
            f"{INVENTORY_OUTPUT_FILE}:{row_number}: evidence text is not deterministic",
        )
        expected_id = make_defterm_input_id(
            source["term"],
            source["qid"],
            source["label_en"],
            item["aliases"],
            item["p31"],
            item["p279"],
            item["p1343"],
        )
        require(
            row["defterm_input_id"] == expected_id,
            f"{INVENTORY_OUTPUT_FILE}:{row_number}: ID is not deterministic",
        )
        allowed_prefixes = (
            "Label: ",
            "Aliases: ",
            "Instance of: ",
            "Subclass of: ",
            "Described by source: ",
        )
        require(
            all(line.startswith(allowed_prefixes) for line in expected_text.splitlines()),
            f"{INVENTORY_OUTPUT_FILE}:{row_number}: unexpected evidence-text section",
        )
    route_counts = Counter(row["representation_route"] for row in prepared_rows)
    require(
        dict(route_counts) == EXPECTED_NEW_ROUTE_COUNTS,
        f"{INVENTORY_OUTPUT_FILE}: unexpected route counts {dict(route_counts)!r}",
    )


def validate_audits(
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
    sample_audit: list[dict[str, str]],
    fewshot_rows: list[dict[str, str]],
    fewshot_audit: list[dict[str, str]],
    comparison: list[dict[str, str]],
) -> None:
    old_by_key = {(row["term"], row["qid"]): row for row in old_rows}
    new_by_key = {(row["term"], row["qid"]): row for row in new_rows}
    old_keys = set(old_by_key)
    new_keys = set(new_by_key)
    sample_keys = {(row["term"], row["qid"]) for row in sample_rows}
    require(
        len(sample_audit) == len(sample_rows) == EXPECTED_SAMPLE_COUNT
        and [(row["term"], row["qid"]) for row in sample_audit]
        == [(row["term"], row["qid"]) for row in sample_rows],
        f"{SAMPLE_AUDIT_OUTPUT_FILE}: sample coverage or order differs",
    )
    require(
        len(fewshot_audit) == len(fewshot_rows) == EXPECTED_FEWSHOT_COUNT
        and [(row["term"], row["qid"]) for row in fewshot_audit]
        == [(row["term"], row["qid"]) for row in fewshot_rows],
        f"{FEWSHOT_AUDIT_OUTPUT_FILE}: few-shot coverage or order differs",
    )
    for row_number, (sample, audit) in enumerate(
        zip(sample_rows, sample_audit), start=2
    ):
        key = (sample["term"], sample["qid"])
        old = old_by_key[key]
        new = new_by_key.get(key)
        present = new is not None
        identical = present and evidence_signature(old) == evidence_signature(new)
        if not present:
            reference_status = "CANDIDATE_LEFT_DEFTERM"
        elif not identical:
            reference_status = "EVIDENCE_CHANGED_DO_NOT_TRANSFER_LABEL"
        else:
            reference_status = "CORRESPONDING_CANDIDATE_UNCHANGED"
        expected = {
            "historical_defterm_input_id": old["defterm_input_id"],
            "new_defterm_input_id": new["defterm_input_id"] if new else "",
            "term": key[0],
            "qid": key[1],
            "present_in_new_inventory": bool_text(present),
            "evidence_identical": bool_text(identical) if present else "",
            "old_evidence_profile": old["evidence_profile"],
            "new_evidence_profile": new["evidence_profile"] if new else "",
            "old_evidence_sparsity": old["evidence_sparsity"],
            "new_evidence_sparsity": new["evidence_sparsity"] if new else "",
            "old_defterm_evidence_text": old["defterm_evidence_text"],
            "new_defterm_evidence_text": new["defterm_evidence_text"] if new else "",
            "historical_evaluation_selection_reason": sample[
                "evaluation_selection_reason"
            ],
            "old_representation_route": old["representation_route"],
            "new_representation_route": new["representation_route"] if new else "",
            "sufficiency_reference_present": "true",
            "sufficiency_reference_evidence_matches_historical_sample": "true",
            "sufficiency_reference_status": reference_status,
        }
        require(
            audit == expected,
            f"{SAMPLE_AUDIT_OUTPUT_FILE}:{row_number}: calculated audit fields differ",
        )

    for row_number, (fewshot, audit) in enumerate(
        zip(fewshot_rows, fewshot_audit), start=2
    ):
        key = (fewshot["term"], fewshot["qid"])
        old = old_by_key[key]
        new = new_by_key.get(key)
        present = new is not None
        identical = present and evidence_signature(old) == evidence_signature(new)
        overlap = key in sample_keys
        if not present:
            representativeness = "NO_LONGER_IN_DEFTERM"
        elif not identical:
            representativeness = "REVIEW_REQUIRED_EVIDENCE_CHANGED"
        else:
            representativeness = "STILL_REPRESENTATIVE_UNCHANGED_EVIDENCE"
        expected = {
            "fewshot_example_id": fewshot["fewshot_example_id"],
            "historical_defterm_input_id": old["defterm_input_id"],
            "new_defterm_input_id": new["defterm_input_id"] if new else "",
            "term": key[0],
            "qid": key[1],
            "present_in_new_inventory": bool_text(present),
            "evidence_identical": bool_text(identical) if present else "",
            "old_evidence_profile": old["evidence_profile"],
            "new_evidence_profile": new["evidence_profile"] if new else "",
            "old_evidence_sparsity": old["evidence_sparsity"],
            "new_evidence_sparsity": new["evidence_sparsity"] if new else "",
            "old_defterm_evidence_text": old["defterm_evidence_text"],
            "new_defterm_evidence_text": new["defterm_evidence_text"] if new else "",
            "old_representation_route": old["representation_route"],
            "new_representation_route": new["representation_route"] if new else "",
            "overlap_with_historical_evaluation_sample": bool_text(overlap),
            "future_evaluation_eligibility": "EXCLUDE_APPROVED_FEWSHOT",
            "representativeness_status": representativeness,
        }
        require(
            audit == expected,
            f"{FEWSHOT_AUDIT_OUTPUT_FILE}:{row_number}: calculated audit fields differ",
        )

    comparison_by_status = Counter(row["membership_status"] for row in comparison)
    require(
        set(comparison_by_status) <= {
            "STILL_PENDING",
            "ENTERED_DEFTERM",
            "LEFT_DEFTERM",
        },
        f"{COMPARISON_OUTPUT_FILE}: unexpected membership status",
    )
    comparison_keys = {(row["term"], row["qid"]) for row in comparison}
    require(
        len(comparison_keys) == len(comparison)
        and comparison_keys == old_keys | new_keys,
        f"{COMPARISON_OUTPUT_FILE}: union coverage differs",
    )
    require(
        comparison_by_status["STILL_PENDING"] == len(old_keys & new_keys)
        and comparison_by_status["ENTERED_DEFTERM"] == len(new_keys - old_keys)
        and comparison_by_status["LEFT_DEFTERM"] == len(old_keys - new_keys),
        f"{COMPARISON_OUTPUT_FILE}: membership counts differ",
    )
    for row_number, row in enumerate(comparison, start=2):
        key = (row["term"], row["qid"])
        old = old_by_key.get(key)
        new = new_by_key.get(key)
        if old is not None and new is not None:
            membership = "STILL_PENDING"
            identical = bool_text(evidence_signature(old) == evidence_signature(new))
        elif old is None:
            membership = "ENTERED_DEFTERM"
            identical = ""
        else:
            membership = "LEFT_DEFTERM"
            identical = ""
        expected = {
            "term": key[0],
            "qid": key[1],
            "membership_status": membership,
            "old_defterm_input_id": old["defterm_input_id"] if old else "",
            "new_defterm_input_id": new["defterm_input_id"] if new else "",
            "evidence_identical": identical,
            "old_label_en": old["label_en"] if old else "",
            "new_label_en": new["label_en"] if new else "",
            "old_evidence_profile": old["evidence_profile"] if old else "",
            "new_evidence_profile": new["evidence_profile"] if new else "",
            "old_evidence_sparsity": old["evidence_sparsity"] if old else "",
            "new_evidence_sparsity": new["evidence_sparsity"] if new else "",
            "old_defterm_evidence_text": old["defterm_evidence_text"] if old else "",
            "new_defterm_evidence_text": new["defterm_evidence_text"] if new else "",
            "old_representation_route": old["representation_route"] if old else "",
            "new_representation_route": new["representation_route"] if new else "",
        }
        require(
            row == expected,
            f"{COMPARISON_OUTPUT_FILE}:{row_number}: calculated comparison fields differ",
        )
    output_fields = (
        set(INVENTORY_FIELDS)
        | set(SAMPLE_AUDIT_FIELDS)
        | set(FEWSHOT_AUDIT_FIELDS)
        | set(COMPARISON_FIELDS)
    )
    require(
        not FORBIDDEN_INPUT_INFLUENCE_FIELDS.intersection(output_fields),
        "an evaluation, manual-decision, corpus, translation, or Sense 1 field leaked",
    )


def serialize_csv(
    filename: str, fields: Sequence[str], rows: list[dict[str, str]]
) -> bytes:
    expected = set(fields)
    for row_number, row in enumerate(rows, start=2):
        require(
            set(row) == expected,
            f"{filename}:{row_number}: output row has an unexpected schema",
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_payloads(
    inventory: list[dict[str, str]],
    sample_audit: list[dict[str, str]],
    fewshot_audit: list[dict[str, str]],
    comparison: list[dict[str, str]],
) -> dict[str, bytes]:
    return {
        INVENTORY_OUTPUT_FILE: serialize_csv(
            INVENTORY_OUTPUT_FILE, INVENTORY_FIELDS, inventory
        ),
        SAMPLE_AUDIT_OUTPUT_FILE: serialize_csv(
            SAMPLE_AUDIT_OUTPUT_FILE, SAMPLE_AUDIT_FIELDS, sample_audit
        ),
        FEWSHOT_AUDIT_OUTPUT_FILE: serialize_csv(
            FEWSHOT_AUDIT_OUTPUT_FILE, FEWSHOT_AUDIT_FIELDS, fewshot_audit
        ),
        COMPARISON_OUTPUT_FILE: serialize_csv(
            COMPARISON_OUTPUT_FILE, COMPARISON_FIELDS, comparison
        ),
    }


def acquire_lock() -> int:
    lock_path = BASE_DIR / LOCK_FILE
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValidationError(f"{LOCK_FILE}: another publication is active") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return descriptor


def publish_without_overwrite(payloads: dict[str, bytes]) -> tuple[dict[str, str], str]:
    descriptor = acquire_lock()
    lock_path = BASE_DIR / LOCK_FILE
    temporary_paths: dict[str, Path] = {}
    linked_paths: dict[str, Path] = {}
    try:
        existing = {
            filename: os.path.lexists(BASE_DIR / filename) for filename in payloads
        }
        if any(existing.values()):
            require(
                all(existing.values()),
                "Refusing to continue with a partial set of contextual V2 audit outputs",
            )
            for filename, payload in payloads.items():
                path = BASE_DIR / filename
                require(
                    path.is_file()
                    and not path.is_symlink()
                    and path.read_bytes() == payload,
                    f"{filename}: existing output differs; refusing to overwrite",
                )
            publication_status = "REUSED_IDENTICAL_OUTPUTS"
        else:
            for filename, payload in payloads.items():
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{filename}.", suffix=".tmp", dir=BASE_DIR
                )
                temporary_path = Path(temporary_name)
                temporary_paths[filename] = temporary_path
                with os.fdopen(file_descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            for filename in payloads:
                output_path = BASE_DIR / filename
                try:
                    os.link(temporary_paths[filename], output_path)
                except FileExistsError as exc:
                    raise ValidationError(
                        f"{filename}: appeared during publication; refusing to overwrite"
                    ) from exc
                linked_paths[filename] = output_path
            publication_status = "CREATED_NEW_OUTPUTS"
        output_hashes = {
            filename: sha256_bytes((BASE_DIR / filename).read_bytes())
            for filename in payloads
        }
        for filename, payload in payloads.items():
            require(
                output_hashes[filename] == sha256_bytes(payload),
                f"{filename}: published bytes differ from validated payload",
            )
        return output_hashes, publication_status
    except BaseException:
        # The lock coordinates this publisher; remove only links to our staged inodes.
        for filename in reversed(tuple(linked_paths)):
            output_path = linked_paths[filename]
            temporary_path = temporary_paths[filename]
            try:
                if os.path.samefile(output_path, temporary_path):
                    output_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_path in temporary_paths.values():
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def inventory_statistics(rows: list[dict[str, str]]) -> dict[str, Any]:
    profile_counts = Counter(row["evidence_profile"] for row in rows)
    sparsity_counts = Counter(row["evidence_sparsity"] for row in rows)
    route_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        route_rows[row["representation_route"]].append(row)
    by_route: dict[str, dict[str, Any]] = {}
    for route, selected in route_rows.items():
        by_route[route] = {
            "count": len(selected),
            "profiles": Counter(row["evidence_profile"] for row in selected),
            "sparsities": Counter(row["evidence_sparsity"] for row in selected),
            "has_aliases": sum(row["has_aliases"] == "true" for row in selected),
            "has_p31": sum(row["has_p31"] == "true" for row in selected),
            "has_p279": sum(row["has_p279"] == "true" for row in selected),
            "has_p1343": sum(row["has_p1343"] == "true" for row in selected),
            "aliases_total": sum(int(row["aliases_count"]) for row in selected),
            "p31_total": sum(int(row["p31_count"]) for row in selected),
            "p279_total": sum(int(row["p279_count"]) for row in selected),
            "p1343_total": sum(int(row["p1343_count"]) for row in selected),
            "single_relation": sum(
                row["evidence_sparsity"] == SPARSITY_SINGLE_RELATION
                for row in selected
            ),
            "structured_relation_count_eq_1": sum(
                int(row["p31_count"])
                + int(row["p279_count"])
                + int(row["p1343_count"])
                == 1
                for row in selected
            ),
            "aliases_only": sum(row["evidence_profile"] == "ALIASES_ONLY" for row in selected),
            "without_label": sum(row["label_en"] == "" for row in selected),
        }
    return {
        "terms": len({row["term"] for row in rows}),
        "qids": len({row["qid"] for row in rows}),
        "pairs": len({(row["term"], row["qid"]) for row in rows}),
        "profile_counts": profile_counts,
        "sparsity_counts": sparsity_counts,
        "has_aliases": sum(row["has_aliases"] == "true" for row in rows),
        "has_p31": sum(row["has_p31"] == "true" for row in rows),
        "has_p279": sum(row["has_p279"] == "true" for row in rows),
        "has_p1343": sum(row["has_p1343"] == "true" for row in rows),
        "single_relation": sparsity_counts[SPARSITY_SINGLE_RELATION],
        "structured_relation_count_eq_1": sum(
            int(row["p31_count"])
            + int(row["p279_count"])
            + int(row["p1343_count"])
            == 1
            for row in rows
        ),
        "aliases_only": profile_counts["ALIASES_ONLY"],
        "without_label": sum(row["label_en"] == "" for row in rows),
        "by_route": by_route,
    }


def assert_no_model_runtime() -> None:
    loaded_roots = {name.partition(".")[0] for name in sys.modules}
    require(
        not FORBIDDEN_MODEL_MODULES.intersection(loaded_roots),
        "A forbidden model library was imported into the preparation process",
    )
    require(not MODEL_LOADED and not INFERENCE_EXECUTED, "model/inference guard changed")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every validation without publishing the four outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    csv.field_size_limit(10_000_000)
    tests: dict[str, str] = {}
    assert_no_model_runtime()
    require(
        not os.path.lexists(BASE_DIR / FORBIDDEN_SAMPLE_OUTPUT_FILE),
        f"{FORBIDDEN_SAMPLE_OUTPUT_FILE}: forbidden new evaluation sample already exists",
    )

    immutable_files = (
        OLD_PENDING_FILE,
        OLD_PREPARER_FILE,
        OLD_INVENTORY_FILE,
        OLD_SAMPLE_FILE,
        SUFFICIENCY_REFERENCE_FILE,
        FEWSHOT_FILE,
        FROZEN_EXPERIMENT_FILE,
    )
    new_pending_rows, new_pending_hash = read_fixed_csv(
        NEW_PENDING_FILE, PENDING_FIELDS
    )

    # Build the model-facing inventory before reading any evaluation or manual data.
    new_rows, new_metadata = prepare_inventory(
        new_pending_rows, NEW_PENDING_FILE, EXPECTED_NEW_COUNT
    )
    validate_new_inventory(new_pending_rows, new_rows, new_metadata)
    new_inventory = index_unique(new_rows, INVENTORY_OUTPUT_FILE)
    main_payload_before_audits = serialize_csv(
        INVENTORY_OUTPUT_FILE, INVENTORY_FIELDS, new_rows
    )

    immutable_before = snapshot_hashes(immutable_files)
    old_pending_rows, old_pending_hash = read_fixed_csv(
        OLD_PENDING_FILE, PENDING_FIELDS
    )
    old_inventory_rows, old_inventory_hash = read_fixed_csv(
        OLD_INVENTORY_FILE, OLD_INVENTORY_FIELDS
    )
    old_sample_rows, old_sample_hash = read_fixed_csv(
        OLD_SAMPLE_FILE, OLD_SAMPLE_FIELDS
    )
    sufficiency_rows, sufficiency_hash = read_fixed_csv(
        SUFFICIENCY_REFERENCE_FILE, SUFFICIENCY_FIELDS
    )
    fewshot_rows, fewshot_hash = read_fixed_csv(FEWSHOT_FILE, FEWSHOT_FIELDS)
    pass_test(tests, "INPUT_HASH_TEST", True, "one or more fixed hashes were not verified")

    old_prepared_rows, _ = validate_historical_inventory(
        old_pending_rows, old_inventory_rows
    )
    old_inventory = index_unique(old_inventory_rows, OLD_INVENTORY_FILE)
    old_prepared = index_unique(old_prepared_rows, OLD_PENDING_FILE)
    historical_sample = validate_old_sample(old_sample_rows, old_inventory)
    sufficiency = validate_sufficiency_reference(sufficiency_rows, historical_sample)
    validate_fewshot(fewshot_rows, old_inventory)

    pass_test(
        tests,
        "DEFTERM_COUNT_TEST",
        len(new_rows) == EXPECTED_NEW_COUNT,
        "new inventory does not contain exactly 150 candidates",
    )
    pass_test(
        tests,
        "TERM_QID_UNIQUENESS_TEST",
        len(new_inventory) == len(new_rows),
        "term+qid keys are not unique",
    )
    pass_test(
        tests,
        "EVIDENCE_STRUCTURE_TEST",
        all(
            row["has_aliases"] == bool_text(int(row["aliases_count"]) > 0)
            and row["has_p31"] == bool_text(int(row["p31_count"]) > 0)
            and row["has_p279"] == bool_text(int(row["p279_count"]) > 0)
            and row["has_p1343"] == bool_text(int(row["p1343_count"]) > 0)
            for row in new_rows
        ),
        "evidence flags and counts disagree",
    )
    pass_test(
        tests,
        "EVIDENCE_TEXT_DETERMINISM_TEST",
        all(row == new_metadata[(row["term"], row["qid"])]["prepared"] for row in new_rows),
        "prepared evidence differs from its deterministic rebuild",
    )
    pass_test(
        tests,
        "NO_SENSE1_CONTAMINATION_TEST",
        not FORBIDDEN_INPUT_INFLUENCE_FIELDS.intersection(INVENTORY_FIELDS)
        and all(
            row["defterm_evidence_text"]
            == build_evidence_text(
                row["label_en"],
                new_metadata[(row["term"], row["qid"])]["aliases"],
                new_metadata[(row["term"], row["qid"])]["p31"],
                new_metadata[(row["term"], row["qid"])]["p279"],
                new_metadata[(row["term"], row["qid"])]["p1343"],
            )
            for row in new_rows
        ),
        "DefTerm evidence contains a non-candidate source",
    )

    sample_audit = build_sample_audit(
        old_sample_rows,
        old_inventory,
        old_prepared,
        new_inventory,
        sufficiency,
    )
    fewshot_audit = build_fewshot_audit(
        fewshot_rows,
        old_inventory,
        old_prepared,
        new_inventory,
        set(historical_sample),
    )
    comparison = build_comparison(old_prepared_rows, new_rows)
    validate_audits(
        old_prepared_rows,
        new_rows,
        old_sample_rows,
        sample_audit,
        fewshot_rows,
        fewshot_audit,
        comparison,
    )

    membership_counts = Counter(row["membership_status"] for row in comparison)
    pass_test(
        tests,
        "OLD_NEW_MEMBERSHIP_TEST",
        membership_counts["STILL_PENDING"]
        + membership_counts["LEFT_DEFTERM"]
        == EXPECTED_OLD_COUNT
        and membership_counts["STILL_PENDING"]
        + membership_counts["ENTERED_DEFTERM"]
        == EXPECTED_NEW_COUNT,
        "old/new membership partition is inconsistent",
    )
    pass_test(
        tests,
        "OLD_SAMPLE_OVERLAP_TEST",
        len(sample_audit) == EXPECTED_SAMPLE_COUNT
        and all(row["sufficiency_reference_present"] == "true" for row in sample_audit)
        and all(
            row["present_in_new_inventory"]
            == bool_text((row["term"], row["qid"]) in new_inventory)
            for row in sample_audit
        ),
        "historical sample audit is incomplete",
    )
    pass_test(
        tests,
        "FEWSHOT_OVERLAP_TEST",
        len(fewshot_audit) == EXPECTED_FEWSHOT_COUNT
        and all(
            row["future_evaluation_eligibility"] == "EXCLUDE_APPROVED_FEWSHOT"
            for row in fewshot_audit
        )
        and all(
            row["overlap_with_historical_evaluation_sample"]
            == bool_text((row["term"], row["qid"]) in historical_sample)
            for row in fewshot_audit
        ),
        "few-shot audit or future exclusion policy is incomplete",
    )
    pass_test(
        tests,
        "EVALUATION_ISOLATION_TEST",
        main_payload_before_audits
        == serialize_csv(INVENTORY_OUTPUT_FILE, INVENTORY_FIELDS, new_rows)
        and not {"evidence_sufficiency", "evidence_sufficiency_comment"}.intersection(
            INVENTORY_FIELDS
        )
        and not {"manual_definition", "manual_status", "manual_notes"}.intersection(
            INVENTORY_FIELDS
        ),
        "evaluation or manual labels influenced the new inventory",
    )

    payloads = build_payloads(new_rows, sample_audit, fewshot_audit, comparison)
    second_new_rows, second_metadata = prepare_inventory(
        new_pending_rows, NEW_PENDING_FILE, EXPECTED_NEW_COUNT
    )
    second_inventory = index_unique(second_new_rows, INVENTORY_OUTPUT_FILE)
    second_sample_audit = build_sample_audit(
        old_sample_rows,
        old_inventory,
        old_prepared,
        second_inventory,
        sufficiency,
    )
    second_fewshot_audit = build_fewshot_audit(
        fewshot_rows,
        old_inventory,
        old_prepared,
        second_inventory,
        set(historical_sample),
    )
    second_comparison = build_comparison(old_prepared_rows, second_new_rows)
    second_payloads = build_payloads(
        second_new_rows,
        second_sample_audit,
        second_fewshot_audit,
        second_comparison,
    )
    pass_test(
        tests,
        "DETERMINISM_TEST",
        payloads == second_payloads
        and set(second_metadata) == set(new_metadata),
        "two independent builds produced different bytes",
    )

    immutable_prepublication = snapshot_hashes(immutable_files)
    pass_test(
        tests,
        "OLD_INVENTORY_IMMUTABILITY_TEST",
        immutable_prepublication[OLD_INVENTORY_FILE]
        == immutable_before[OLD_INVENTORY_FILE]
        and immutable_prepublication[OLD_PENDING_FILE]
        == immutable_before[OLD_PENDING_FILE]
        and immutable_prepublication[OLD_PREPARER_FILE]
        == immutable_before[OLD_PREPARER_FILE],
        "historical inventory, source, or preparer changed",
    )
    pass_test(
        tests,
        "OLD_SAMPLE_IMMUTABILITY_TEST",
        immutable_prepublication[OLD_SAMPLE_FILE] == immutable_before[OLD_SAMPLE_FILE],
        "historical sample changed",
    )
    pass_test(
        tests,
        "FEWSHOT_IMMUTABILITY_TEST",
        immutable_prepublication[FEWSHOT_FILE] == immutable_before[FEWSHOT_FILE],
        "approved few-shot file changed",
    )
    require(
        immutable_prepublication[SUFFICIENCY_REFERENCE_FILE]
        == immutable_before[SUFFICIENCY_REFERENCE_FILE]
        and immutable_prepublication[FROZEN_EXPERIMENT_FILE]
        == immutable_before[FROZEN_EXPERIMENT_FILE],
        "sufficiency reference or frozen experiment script changed",
    )
    require(
        not os.path.lexists(BASE_DIR / FORBIDDEN_SAMPLE_OUTPUT_FILE),
        f"{FORBIDDEN_SAMPLE_OUTPUT_FILE}: forbidden evaluation sample was created",
    )
    assert_no_model_runtime()
    require(
        set(tests) == set(REQUIRED_TESTS) and all(value == "PASS" for value in tests.values()),
        "not all required self-tests completed",
    )

    statistics = inventory_statistics(new_rows)
    sample_present = sum(row["present_in_new_inventory"] == "true" for row in sample_audit)
    sample_identical = sum(row["evidence_identical"] == "true" for row in sample_audit)
    sample_changed = sum(row["evidence_identical"] == "false" for row in sample_audit)
    fewshot_present = sum(row["present_in_new_inventory"] == "true" for row in fewshot_audit)
    fewshot_identical = sum(row["evidence_identical"] == "true" for row in fewshot_audit)
    fewshot_changed = sum(row["evidence_identical"] == "false" for row in fewshot_audit)
    fewshot_overlap = sum(
        row["overlap_with_historical_evaluation_sample"] == "true"
        for row in fewshot_audit
    )
    script_hash = sha256_bytes(Path(__file__).read_bytes())

    # Publication is the final mutating operation after every required check passes.
    if args.validate_only:
        output_hashes = {
            filename: sha256_bytes(payload) for filename, payload in payloads.items()
        }
        publication_status = "VALIDATED_NOT_PUBLISHED"
    else:
        output_hashes, publication_status = publish_without_overwrite(payloads)

    print("PREPARATION_STATUS=SUCCESS")
    print(f"PUBLICATION_STATUS={publication_status}")
    print("INCONSISTENCIES=0")
    print("MODEL_LOADED=NO")
    print("INFERENCE_EXECUTED=NO")
    print("NEW_EVALUATION_SAMPLE_CREATED=NO")
    for test in REQUIRED_TESTS:
        print(f"{test}={tests[test]}")
    print(f"INPUT_SHA256_{NEW_PENDING_FILE}={new_pending_hash}")
    print(f"INPUT_SHA256_{OLD_PENDING_FILE}={old_pending_hash}")
    print(f"INPUT_SHA256_{OLD_INVENTORY_FILE}={old_inventory_hash}")
    print(f"INPUT_SHA256_{OLD_SAMPLE_FILE}={old_sample_hash}")
    print(f"INPUT_SHA256_{SUFFICIENCY_REFERENCE_FILE}={sufficiency_hash}")
    print(f"INPUT_SHA256_{FEWSHOT_FILE}={fewshot_hash}")
    print(f"TOTAL_DEFTERM_INPUTS={len(new_rows)}")
    print(f"UNIQUE_TERMS={statistics['terms']}")
    print(f"UNIQUE_QIDS={statistics['qids']}")
    print(f"UNIQUE_TERM_QID_PAIRS={statistics['pairs']}")
    for status in ("STILL_PENDING", "ENTERED_DEFTERM", "LEFT_DEFTERM"):
        print(f"MEMBERSHIP_{status}={membership_counts[status]}")
    print(f"HISTORICAL_SAMPLE_PRESENT={sample_present}")
    print(f"HISTORICAL_SAMPLE_LEFT={EXPECTED_SAMPLE_COUNT - sample_present}")
    print(f"HISTORICAL_SAMPLE_EVIDENCE_IDENTICAL={sample_identical}")
    print(f"HISTORICAL_SAMPLE_EVIDENCE_CHANGED={sample_changed}")
    print(f"FEWSHOT_PRESENT={fewshot_present}")
    print(f"FEWSHOT_LEFT={EXPECTED_FEWSHOT_COUNT - fewshot_present}")
    print(f"FEWSHOT_EVIDENCE_IDENTICAL={fewshot_identical}")
    print(f"FEWSHOT_EVIDENCE_CHANGED={fewshot_changed}")
    print(f"FEWSHOT_HISTORICAL_SAMPLE_OVERLAP={fewshot_overlap}")
    print(f"HAS_ALIASES={statistics['has_aliases']}")
    print(f"HAS_P31={statistics['has_p31']}")
    print(f"HAS_P279={statistics['has_p279']}")
    print(f"HAS_P1343={statistics['has_p1343']}")
    print(f"SINGLE_STRUCTURED_RELATION={statistics['single_relation']}")
    print(
        "STRUCTURED_RELATION_COUNT_EQ_1="
        f"{statistics['structured_relation_count_eq_1']}"
    )
    print(f"ALIASES_ONLY={statistics['aliases_only']}")
    print(f"WITHOUT_LABEL_EN={statistics['without_label']}")
    for profile in sorted(statistics["profile_counts"]):
        print(
            f"EVIDENCE_PROFILE_{profile}="
            f"{statistics['profile_counts'][profile]}"
        )
    for sparsity in sorted(statistics["sparsity_counts"]):
        print(
            f"EVIDENCE_SPARSITY_{sparsity}="
            f"{statistics['sparsity_counts'][sparsity]}"
        )
    for route in sorted(statistics["by_route"]):
        route_stats = statistics["by_route"][route]
        print(f"ROUTE_{route}_COUNT={route_stats['count']}")
        for metric in (
            "has_aliases",
            "has_p31",
            "has_p279",
            "has_p1343",
            "aliases_total",
            "p31_total",
            "p279_total",
            "p1343_total",
            "single_relation",
            "structured_relation_count_eq_1",
            "aliases_only",
            "without_label",
        ):
            print(f"ROUTE_{route}_{metric.upper()}={route_stats[metric]}")
        for profile in sorted(route_stats["profiles"]):
            print(
                f"ROUTE_{route}_EVIDENCE_PROFILE_{profile}="
                f"{route_stats['profiles'][profile]}"
            )
        for sparsity in sorted(route_stats["sparsities"]):
            print(
                f"ROUTE_{route}_EVIDENCE_SPARSITY_{sparsity}="
                f"{route_stats['sparsities'][sparsity]}"
            )
    print("SENSE1_IN_DEFTERM_EVIDENCE_TEXT=NO")
    print("SUFFICIENCY_REFERENCE_USED_TO_BUILD_INPUTS=NO")
    print(f"SCRIPT_SHA256={script_hash}")
    for filename, output_hash in output_hashes.items():
        print(f"OUTPUT_SHA256_{filename}={output_hash}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        loaded_roots = {name.partition(".")[0] for name in sys.modules}
        model_guard_clear = not FORBIDDEN_MODEL_MODULES.intersection(loaded_roots)
        print("PREPARATION_STATUS=FAILED", file=sys.stderr)
        print(
            f"MODEL_LOADED={'NO' if model_guard_clear and not MODEL_LOADED else 'DETECTED'}",
            file=sys.stderr,
        )
        print(
            f"INFERENCE_EXECUTED={'YES' if INFERENCE_EXECUTED else 'NO'}",
            file=sys.stderr,
        )
        print(f"INCONSISTENCY={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
