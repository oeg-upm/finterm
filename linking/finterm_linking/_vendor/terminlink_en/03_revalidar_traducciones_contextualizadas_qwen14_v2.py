#!/usr/bin/env python3
"""Deterministically revalidate existing contextual Qwen V2 translations."""

from __future__ import annotations

import argparse
import ast
import codecs
import csv
import hashlib
import io
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_RESULTS_PATH = (
    BASE_DIR / "03_traducciones_contextualizadas_qwen14_v2_completo_results.csv"
)
ORIGINAL_REVIEW_PATH = (
    BASE_DIR / "03_traducciones_contextualizadas_qwen14_v2_completo_review.csv"
)
REVALIDATED_RESULTS_PATH = (
    BASE_DIR
    / "03_traducciones_contextualizadas_qwen14_v2_completo_results_revalidated.csv"
)
REVALIDATION_AUDIT_PATH = (
    BASE_DIR / "03_traducciones_contextualizadas_qwen14_v2_revalidation_audit.csv"
)
REVALIDATED_REVIEW_PATH = (
    BASE_DIR
    / "03_traducciones_contextualizadas_qwen14_v2_completo_review_revalidated.csv"
)
RUN_LOCK_PATH = BASE_DIR / ".03_revalidar_traducciones_qwen14_v2.lock"

EXPECTED_ORIGINAL_RESULTS_SHA256 = (
    "A1C95B8BA93C1C82D8A44622BBF75A536660E1F79A71497204FA7D7CCD5E1548"
)
EXPECTED_ORIGINAL_REVIEW_SHA256 = (
    "93D912D25DB953134A75463E837DF085DA0D0541731AF820D7D1193A9CB294BB"
)
EXPECTED_ROWS = 2036
EXPECTED_ORIGINAL_OK = 2024
EXPECTED_ORIGINAL_ERROR = 12
EXPECTED_ORIGINAL_REVIEW_ROWS = 149
EXPECTED_REVALIDATED_OK = 2036
EXPECTED_REVALIDATED_ERROR = 0
EXPECTED_REVALIDATED_REVIEW_ROWS = 137
EXPECTED_ORIGINAL_ERROR_MESSAGES = Counter(
    {
        "MalformedGenerationError: Generation contains quotation marks": 8,
        "MalformedGenerationError: Generation contains a translation header": 3,
        "MalformedGenerationError: Generation contains alternative separators": 1,
    }
)
EXPECTED_ORIGINAL_ALERT_COUNTS = Counter(
    {
        "RARE_SOURCE_LANGUAGE": 125,
        "POSSIBLY_UNTRANSLATED_NORMALIZED_EQUAL": 30,
        "GENERATION_ERROR": 12,
    }
)
EXPECTED_REVALIDATED_ALERT_COUNTS = Counter(
    {
        "RARE_SOURCE_LANGUAGE": 125,
        "POSSIBLY_UNTRANSLATED_NORMALIZED_EQUAL": 30,
    }
)
VALIDATION_POLICY_VERSION = "contextual-qwen14-v2-posthoc-format-revalidation-v1"

RESULT_FIELDS = (
    "contextual_translation_input_id",
    "representation_id",
    "qid",
    "source_language",
    "source_description",
    "target_language",
    "source_translation_status",
    "representation_route",
    "context_policy_version",
    "pending_input_sha256",
    "candidate_input_sha256",
    "translation_evidence_sha256",
    "model_id",
    "experiment_type",
    "prompt_version",
    "model_revision",
    "system_prompt_sha256",
    "user_prompt_template_sha256",
    "input_sha256",
    "inference_script_sha256",
    "runtime_module_sha256",
    "inference_seed",
    "quantization_config",
    "generation_parameters",
    "python_version",
    "transformers_version",
    "tokenizers_version",
    "huggingface_hub_version",
    "accelerate_version",
    "bitsandbytes_version",
    "safetensors_version",
    "torch_version",
    "cuda_version",
    "device_name",
    "generation_attempts",
    "translation_raw",
    "translation_clean",
    "generation_status",
    "error_message",
)

REVIEW_DISCLAIMER = (
    "Heuristic review aid only; these alerts are advisory, are not a semantic "
    "evaluation, and do not determine translation correctness."
)
RARE_LANGUAGE_MAX_FREQUENCY = 2
MIN_EXTREME_LENGTH_RATIO = 0.25
MAX_EXTREME_LENGTH_RATIO = 4.0

REVIEW_FIELDS = (
    "contextual_translation_input_id",
    "representation_id",
    "qid",
    "source_language",
    "source_description",
    "translation_raw",
    "translation_clean",
    "generation_status",
    "error_message",
    "source_language_inventory_frequency",
    "source_character_count",
    "translation_character_count",
    "translation_to_source_length_ratio",
    "translation_word_count",
    "source_script_profile",
    "translation_script_profile",
    "alert_generation_error",
    "alert_possibly_untranslated",
    "alert_extreme_length_ratio",
    "alert_rare_source_language",
    "alert_non_latin_or_mixed_translation_script",
    "alert_count",
    "alert_codes",
    "review_recommended",
    "alerts_are_advisory",
    "disclaimer_alerts_are_not_semantic_evaluation",
)

AUDIT_FIELDS = (
    "contextual_translation_input_id",
    "representation_id",
    "qid",
    "source_language",
    "source_description",
    "translation_raw",
    "translation_raw_sha256",
    "original_generation_status",
    "original_error_message",
    "revalidated_generation_status",
    "revalidation_reason",
    "translation_clean_revalidated",
    "revalidation_error_message",
    "status_changed",
    "validation_policy_version",
    "original_results_sha256",
    "original_review_sha256",
    "revalidation_script_sha256",
)

THINK_PATTERN = re.compile(
    r"</?think\b|<think>.*?</think>", re.IGNORECASE | re.DOTALL
)
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
MARKDOWN_HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
LIST_ITEM_PATTERN = re.compile(
    r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE
)
MARKDOWN_INLINE_PATTERN = re.compile(
    r"(?:\*\*[^*\r\n]+\*\*|__[^_\r\n]+__|"
    r"(?<!\w)\*[^*\r\n]+\*(?!\w)|"
    r"(?<!\w)_[^_\r\n]+_(?!\w)|"
    r"~~[^~\r\n]+~~|`[^`\r\n]+`|"
    r"\[[^\]\r\n]+\]\([^\)\r\n]+\))"
)
MARKDOWN_BLOCKQUOTE_PATTERN = re.compile(r"^\s*>", re.MULTILINE)
LINE_BREAK_PATTERN = re.compile(r"[\r\n\u0085\u2028\u2029]")
HEADER_SEPARATOR = (
    r"(?:\s*:|\s+[\-\u2013\u2014]\s*|\s*[\-\u2013\u2014]\s+)"
)
PRESENTATION_HEADER_PATTERN = re.compile(
    r"^\s*(?:"
    r"here(?:'s|\u2019s|\s+is)\s+(?:the\s+)?(?:english\s+)?translation"
    + HEADER_SEPARATOR
    + r"|"
    r"(?:the\s+)?(?:english\s+)?translation"
    r"(?:\s+into\s+english)?\s+(?:is|would\s+be)\b\s*|"
    r"(?:the\s+)?(?:english\s+)?translation"
    r"(?:\s+into\s+english)?"
    + HEADER_SEPARATOR
    + r"|english"
    + HEADER_SEPARATOR
    + r"|(?:answer|output|result)"
    + HEADER_SEPARATOR
    + r"|(?:answer|output|result)\s+(?:is|would\s+be)\b\s*:?[\t ]*"
    + r"|"
    r"sure\s*[,!:]"
    r")",
    re.IGNORECASE,
)
COMMENTARY_PATTERN = re.compile(
    r"\b(?:this\s+translates?\s+(?:to|as)|which\s+means|"
    r"literally\s+means|(?:note|explanation|reasoning|alternatives?|option)"
    r"\s*[:\-\u2013\u2014]|alternatively\s*[:\-\u2013\u2014]|"
    r"another\s+translation\s+(?:is|would\s+be)|"
    r"(?:it|this)\s+means\b|meaning\s*:)",
    re.IGNORECASE,
)
EXPLICIT_ALTERNATIVE_SEPARATOR_PATTERN = re.compile(r"(?:\s/|/\s|\|)")
WRAPPING_QUOTE_PAIRS = frozenset(
    {
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u201e", "\u201d"),
        ("\u201e", "\u201c"),
        ("\u201f", "\u201d"),
        ("\u00ab", "\u00bb"),
        ("\u2018", "\u2019"),
        ("\u201a", "\u2018"),
        ("\u2039", "\u203a"),
    }
)

WORD_PATTERN = re.compile(
    r"[^\W_]+(?:['\u2019\u2010\u2011\u2012\u2013\u2014\-][^\W_]+)*",
    re.UNICODE,
)
SCRIPT_MARKERS = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARABIC",
    "HEBREW",
    "DEVANAGARI",
    "BENGALI",
    "GURMUKHI",
    "GUJARATI",
    "ORIYA",
    "TAMIL",
    "TELUGU",
    "KANNADA",
    "MALAYALAM",
    "SINHALA",
    "THAI",
    "LAO",
    "MYANMAR",
    "GEORGIAN",
    "ARMENIAN",
    "HANGUL",
    "HIRAGANA",
    "KATAKANA",
    "CJK",
    "ETHIOPIC",
    "KHMER",
    "MONGOLIAN",
)

MODEL_LIBRARY_ROOTS = frozenset(
    {
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
        "sentence_transformers",
    }
)
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "argparse",
        "ast",
        "codecs",
        "csv",
        "hashlib",
        "io",
        "os",
        "re",
        "sys",
        "unicodedata",
        "collections",
        "pathlib",
        "typing",
    }
)


class RevalidationError(RuntimeError):
    """Raised when a post-hoc revalidation invariant fails."""


class MalformedGenerationError(ValueError):
    """Raised when existing raw output is structurally invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RevalidationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RevalidationError(f"Could not hash {path.name}: {error}") from error
    return digest.hexdigest().upper()


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def model_libraries_loaded() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == root or name.startswith(root + ".")
            for root in MODEL_LIBRARY_ROOTS
        )
    )


def require_no_model_usage() -> None:
    loaded = model_libraries_loaded()
    require(
        not loaded,
        "Model libraries were imported during deterministic revalidation: "
        + ", ".join(loaded),
    )
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
            if node.module.split(".")[0] == "os":
                forbidden_calls.append("from-os-import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in {
                    "generate",
                    "from_pretrained",
                    "load_model_and_tokenizer",
                    "infer_once",
                    "cuda",
                    "urlopen",
                    "system",
                    "popen",
                } or node.func.attr.startswith(("spawn", "exec")):
                    forbidden_calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                if node.func.id in {
                    "pipeline",
                    "__import__",
                    "eval",
                    "exec",
                    "compile",
                    "getattr",
                }:
                    forbidden_calls.append(node.func.id)
    require(
        imported_roots.issubset(ALLOWED_IMPORT_ROOTS),
        "The revalidation script imports a non-whitelisted module: "
        + ", ".join(sorted(imported_roots - ALLOWED_IMPORT_ROOTS)),
    )
    require(
        not forbidden_calls,
        "The revalidation script contains a forbidden dynamic, process, network, "
        "or inference call: " + ", ".join(sorted(set(forbidden_calls))),
    )


def parse_csv_payload(
    payload: bytes, source_name: str, expected_fields: Sequence[str]
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            require(
                tuple(reader.fieldnames or ()) == tuple(expected_fields),
                f"{source_name}: unexpected schema {reader.fieldnames!r}",
            )
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                require(
                    None not in row
                    and all(value is not None for value in row.values()),
                    f"{source_name}: malformed row {row_number}",
                )
                normalized = {field: row[field] for field in expected_fields}
                require(
                    all("\x00" not in value for value in normalized.values()),
                    f"{source_name}: NUL character at row {row_number}",
                )
                rows.append(normalized)
    except (UnicodeError, csv.Error) as error:
        raise RevalidationError(f"Could not parse {source_name}: {error}") from error
    return rows


def read_csv_snapshot(
    path: Path, expected_fields: Sequence[str], expected_sha256: str | None
) -> tuple[list[dict[str, str]], str]:
    require(path.is_file(), f"Required input does not exist: {path.name}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RevalidationError(f"Could not read {path.name}: {error}") from error
    actual_sha256 = sha256_bytes(payload)
    if expected_sha256 is not None:
        require(
            actual_sha256 == expected_sha256,
            f"{path.name}: SHA-256 mismatch; expected {expected_sha256}, "
            f"got {actual_sha256}",
        )
    return parse_csv_payload(payload, path.name, expected_fields), actual_sha256


def format_error(error: Exception) -> str:
    detail = re.sub(r"\s+", " ", str(error)).strip()
    return f"{type(error).__name__}: {detail or 'No error detail provided'}"


def normalize_and_validate_generation(raw_text: str) -> str:
    """Reject structure, then perform whitespace-only normalization."""
    raw = str(raw_text)
    stripped = raw.strip()
    if not stripped:
        raise MalformedGenerationError("Generation is empty")
    if "\x00" in raw:
        raise MalformedGenerationError("Generation contains a NUL character")
    if THINK_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains reasoning tags")
    if "```" in raw:
        raise MalformedGenerationError("Generation contains a fenced block")
    if HTML_TAG_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains HTML markup")
    if LINE_BREAK_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains line breaks")
    if (
        MARKDOWN_HEADER_PATTERN.search(raw)
        or LIST_ITEM_PATTERN.search(raw)
        or MARKDOWN_INLINE_PATTERN.search(raw)
        or MARKDOWN_BLOCKQUOTE_PATTERN.search(raw)
    ):
        raise MalformedGenerationError("Generation contains Markdown structure")
    if PRESENTATION_HEADER_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains a presentation header")
    if COMMENTARY_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains explanation or commentary")
    if EXPLICIT_ALTERNATIVE_SEPARATOR_PATTERN.search(raw):
        raise MalformedGenerationError(
            "Generation contains an explicit alternative separator"
        )
    if ";" in raw:
        raise MalformedGenerationError(
            "Generation contains a semicolon-separated alternative"
        )
    if len(stripped) >= 2 and (stripped[0], stripped[-1]) in WRAPPING_QUOTE_PAIRS:
        raise MalformedGenerationError(
            "Generation is wrapped in presentation quotation marks"
        )
    return re.sub(r"\s+", " ", raw).strip()


def validate_original_rows(rows: Sequence[dict[str, str]]) -> None:
    require(
        len(rows) == EXPECTED_ROWS,
        f"Expected {EXPECTED_ROWS} original rows, found {len(rows)}",
    )
    contextual_ids: set[str] = set()
    representation_ids: set[str] = set()
    statuses: Counter[str] = Counter()
    error_messages: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        require(
            tuple(row) == RESULT_FIELDS,
            f"Unexpected original result schema at row {row_number}",
        )
        contextual_id = row["contextual_translation_input_id"]
        representation_id = row["representation_id"]
        require(
            contextual_id and contextual_id not in contextual_ids,
            f"Missing or duplicate contextual ID at row {row_number}",
        )
        require(
            representation_id and representation_id not in representation_ids,
            f"Missing or duplicate representation ID at row {row_number}",
        )
        require(
            row["generation_attempts"] == "1"
            and row["generation_status"] in {"OK", "ERROR"},
            f"Invalid generation policy metadata at row {row_number}",
        )
        if row["generation_status"] == "OK":
            try:
                normalized = normalize_and_validate_generation(
                    row["translation_raw"]
                )
            except MalformedGenerationError as error:
                raise RevalidationError(
                    f"Previously OK row fails corrected validation at row "
                    f"{row_number}: {error}"
                ) from error
            require(
                row["translation_clean"] == normalized
                and not row["error_message"],
                f"Previously OK row fields disagree at row {row_number}",
            )
        else:
            require(
                bool(row["translation_raw"])
                and not row["translation_clean"]
                and bool(row["error_message"]),
                f"Original ERROR row is not the expected validator rejection at "
                f"row {row_number}",
            )
            error_messages[row["error_message"]] += 1
        statuses[row["generation_status"]] += 1
        contextual_ids.add(contextual_id)
        representation_ids.add(representation_id)
    require(
        statuses
        == Counter(
            {"OK": EXPECTED_ORIGINAL_OK, "ERROR": EXPECTED_ORIGINAL_ERROR}
        ),
        f"Unexpected original status counts: {statuses!r}",
    )
    require(
        error_messages == EXPECTED_ORIGINAL_ERROR_MESSAGES,
        f"Unexpected original error-message counts: {error_messages!r}",
    )


def audit_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized)
        .split()
    )


def audit_character_count(text: str) -> int:
    return sum(
        character.isalnum() for character in unicodedata.normalize("NFKC", text)
    )


def audit_word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def character_script(character: str) -> str:
    name = unicodedata.name(character, "")
    for marker in SCRIPT_MARKERS:
        if marker in name:
            return marker
    return "OTHER"


def script_profile(text: str) -> str:
    scripts = {
        character_script(character)
        for character in unicodedata.normalize("NFKC", text)
        if unicodedata.category(character).startswith("L")
    }
    if not scripts:
        return "NONE"
    ordered = sorted(scripts)
    if len(ordered) > 1:
        return "MIXED:" + "+".join(ordered)
    script = ordered[0]
    return script if script == "LATIN" else f"NON_LATIN:{script}"


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def build_review_rows(
    result_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    language_frequencies = Counter(row["source_language"] for row in result_rows)
    review_rows: list[dict[str, str]] = []
    for result in result_rows:
        source_text = result["source_description"]
        translation = result["translation_clean"]
        source_characters = audit_character_count(source_text)
        translation_characters = audit_character_count(translation)
        ratio = (
            translation_characters / source_characters
            if source_characters and translation_characters
            else None
        )
        source_words = audit_word_count(source_text)
        translation_words = audit_word_count(translation)
        source_profile = script_profile(source_text)
        translation_profile = script_profile(translation)
        generation_error = result["generation_status"] == "ERROR"
        possibly_untranslated = (
            result["generation_status"] == "OK"
            and bool(audit_normalize(source_text))
            and audit_normalize(source_text) == audit_normalize(translation)
        )
        extreme_ratio = (
            result["generation_status"] == "OK"
            and ratio is not None
            and source_words >= 2
            and translation_words >= 2
            and (
                ratio < MIN_EXTREME_LENGTH_RATIO
                or ratio > MAX_EXTREME_LENGTH_RATIO
            )
        )
        rare_language = (
            language_frequencies[result["source_language"]]
            <= RARE_LANGUAGE_MAX_FREQUENCY
        )
        unusual_script = (
            result["generation_status"] == "OK"
            and translation_profile not in {"LATIN", "NONE"}
        )
        alert_values = (
            ("GENERATION_ERROR", generation_error),
            ("POSSIBLY_UNTRANSLATED_NORMALIZED_EQUAL", possibly_untranslated),
            ("EXTREME_LENGTH_RATIO", extreme_ratio),
            ("RARE_SOURCE_LANGUAGE", rare_language),
            ("NON_LATIN_OR_MIXED_TRANSLATION_SCRIPT", unusual_script),
        )
        alerts = [code for code, active in alert_values if active]
        if not alerts:
            continue
        review_rows.append(
            {
                "contextual_translation_input_id": result[
                    "contextual_translation_input_id"
                ],
                "representation_id": result["representation_id"],
                "qid": result["qid"],
                "source_language": result["source_language"],
                "source_description": source_text,
                "translation_raw": result["translation_raw"],
                "translation_clean": translation,
                "generation_status": result["generation_status"],
                "error_message": result["error_message"],
                "source_language_inventory_frequency": str(
                    language_frequencies[result["source_language"]]
                ),
                "source_character_count": str(source_characters),
                "translation_character_count": str(translation_characters),
                "translation_to_source_length_ratio": (
                    f"{ratio:.6f}" if ratio is not None else ""
                ),
                "translation_word_count": str(translation_words),
                "source_script_profile": source_profile,
                "translation_script_profile": translation_profile,
                "alert_generation_error": yes_no(generation_error),
                "alert_possibly_untranslated": yes_no(possibly_untranslated),
                "alert_extreme_length_ratio": yes_no(extreme_ratio),
                "alert_rare_source_language": yes_no(rare_language),
                "alert_non_latin_or_mixed_translation_script": yes_no(
                    unusual_script
                ),
                "alert_count": str(len(alerts)),
                "alert_codes": ";".join(alerts),
                "review_recommended": "YES",
                "alerts_are_advisory": "YES",
                "disclaimer_alerts_are_not_semantic_evaluation": (
                    REVIEW_DISCLAIMER
                ),
            }
        )
    return review_rows


def count_alerts(review_rows: Sequence[dict[str, str]]) -> Counter[str]:
    return Counter(
        code
        for row in review_rows
        for code in row["alert_codes"].split(";")
        if code
    )


def validate_original_review(
    original_rows: Sequence[dict[str, str]],
    review_rows: Sequence[dict[str, str]],
) -> None:
    require(
        len(review_rows) == EXPECTED_ORIGINAL_REVIEW_ROWS,
        f"Expected {EXPECTED_ORIGINAL_REVIEW_ROWS} historical review rows, "
        f"found {len(review_rows)}",
    )
    for row_number, row in enumerate(review_rows, start=2):
        require(
            tuple(row) == REVIEW_FIELDS,
            f"Unexpected historical review schema at row {row_number}",
        )
        require(
            row["review_recommended"] == "YES"
            and row["alerts_are_advisory"] == "YES"
            and row["disclaimer_alerts_are_not_semantic_evaluation"]
            == REVIEW_DISCLAIMER,
            f"Historical review disclaimer changed at row {row_number}",
        )
    require(
        count_alerts(review_rows) == EXPECTED_ORIGINAL_ALERT_COUNTS,
        f"Historical review alert counts changed: {count_alerts(review_rows)!r}",
    )
    reconstructed = build_review_rows(original_rows)
    require(
        reconstructed == list(review_rows),
        "Historical review differs from the deterministic heuristic reconstruction",
    )


def acceptance_reason(original_error: str) -> str:
    if original_error.endswith("Generation contains quotation marks"):
        return "ACCEPTED_INTERNAL_QUOTES_NOT_WRAPPING_PRESENTATION"
    if original_error.endswith("Generation contains a translation header"):
        return "ACCEPTED_LEXICAL_ENGLISH_PREFIX_NOT_PRESENTATION_HEADER"
    if original_error.endswith("Generation contains alternative separators"):
        return "ACCEPTED_LEXICALIZED_SLASH_NOT_EXPLICIT_ALTERNATIVES"
    return "ACCEPTED_BY_CORRECTED_STRUCTURAL_VALIDATOR"


def revalidate_rows(
    original_rows: Sequence[dict[str, str]],
    original_results_sha256: str,
    original_review_sha256: str,
    script_sha256: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    revalidated_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for original in original_rows:
        output = dict(original)
        if original["generation_status"] == "OK":
            normalized = normalize_and_validate_generation(
                original["translation_raw"]
            )
            require(
                normalized == original["translation_clean"],
                "Previously OK translation changed under whitespace normalization",
            )
        else:
            try:
                normalized = normalize_and_validate_generation(
                    original["translation_raw"]
                )
            except MalformedGenerationError as error:
                revalidated_status = "ERROR"
                revalidated_error = format_error(error)
                revalidation_reason = f"REJECTED: {revalidated_error}"
                normalized = ""
            else:
                revalidated_status = "OK"
                revalidated_error = ""
                revalidation_reason = acceptance_reason(
                    original["error_message"]
                )
            output["translation_clean"] = normalized
            output["generation_status"] = revalidated_status
            output["error_message"] = revalidated_error
            audit_rows.append(
                {
                    "contextual_translation_input_id": original[
                        "contextual_translation_input_id"
                    ],
                    "representation_id": original["representation_id"],
                    "qid": original["qid"],
                    "source_language": original["source_language"],
                    "source_description": original["source_description"],
                    "translation_raw": original["translation_raw"],
                    "translation_raw_sha256": sha256_text(
                        original["translation_raw"]
                    ),
                    "original_generation_status": original[
                        "generation_status"
                    ],
                    "original_error_message": original["error_message"],
                    "revalidated_generation_status": revalidated_status,
                    "revalidation_reason": revalidation_reason,
                    "translation_clean_revalidated": normalized,
                    "revalidation_error_message": revalidated_error,
                    "status_changed": yes_no(
                        original["generation_status"] != revalidated_status
                    ),
                    "validation_policy_version": VALIDATION_POLICY_VERSION,
                    "original_results_sha256": original_results_sha256,
                    "original_review_sha256": original_review_sha256,
                    "revalidation_script_sha256": script_sha256,
                }
            )
        revalidated_rows.append({field: output[field] for field in RESULT_FIELDS})
    return revalidated_rows, audit_rows


def validate_revalidated_rows(
    original_rows: Sequence[dict[str, str]],
    revalidated_rows: Sequence[dict[str, str]],
    audit_rows: Sequence[dict[str, str]],
    original_results_sha256: str,
    original_review_sha256: str,
    script_sha256: str,
) -> None:
    require(
        len(revalidated_rows) == len(original_rows) == EXPECTED_ROWS,
        "Revalidation changed the result row count",
    )
    require(
        len(audit_rows) == EXPECTED_ORIGINAL_ERROR,
        "Audit must contain exactly the 12 original ERROR rows",
    )
    original_errors = [
        row for row in original_rows if row["generation_status"] == "ERROR"
    ]
    revalidated_by_id = {
        row["contextual_translation_input_id"]: row for row in revalidated_rows
    }
    for row_number, (original, revalidated) in enumerate(
        zip(original_rows, revalidated_rows), start=2
    ):
        require(
            tuple(revalidated) == RESULT_FIELDS,
            f"Revalidated output schema changed at row {row_number}",
        )
        require(
            revalidated["contextual_translation_input_id"]
            == original["contextual_translation_input_id"]
            and revalidated["representation_id"]
            == original["representation_id"],
            f"Result identity changed at row {row_number}",
        )
        require(
            revalidated["translation_raw"] == original["translation_raw"],
            f"Raw translation changed at row {row_number}",
        )
        if original["generation_status"] == "OK":
            require(
                revalidated == original,
                f"Previously OK row changed at row {row_number}",
            )
        else:
            changed_fields = {
                field
                for field in RESULT_FIELDS
                if original[field] != revalidated[field]
            }
            require(
                changed_fields.issubset(
                    {"translation_clean", "generation_status", "error_message"}
                ),
                f"Non-validation fields changed at row {row_number}: "
                f"{sorted(changed_fields)!r}",
            )
            if revalidated["generation_status"] == "OK":
                require(
                    revalidated["translation_clean"]
                    == normalize_and_validate_generation(
                        revalidated["translation_raw"]
                    )
                    and not revalidated["error_message"],
                    f"Accepted row fields disagree at row {row_number}",
                )
            else:
                require(
                    not revalidated["translation_clean"]
                    and bool(revalidated["error_message"]),
                    f"Rejected row fields disagree at row {row_number}",
                )
    statuses = Counter(row["generation_status"] for row in revalidated_rows)
    require(
        statuses
        == Counter(
            {
                "OK": EXPECTED_REVALIDATED_OK,
                "ERROR": EXPECTED_REVALIDATED_ERROR,
            }
        ),
        f"Unexpected revalidated status counts: {statuses!r}",
    )
    for audit, original in zip(audit_rows, original_errors):
        revalidated = revalidated_by_id[original["contextual_translation_input_id"]]
        require(
            tuple(audit) == AUDIT_FIELDS,
            "Unexpected revalidation audit schema",
        )
        require(
            audit["contextual_translation_input_id"]
            == original["contextual_translation_input_id"]
            and audit["representation_id"] == original["representation_id"]
            and audit["translation_raw"] == original["translation_raw"]
            and audit["original_generation_status"] == "ERROR"
            and audit["original_error_message"] == original["error_message"],
            "Revalidation audit lost original traceability",
        )
        require(
            audit["qid"] == original["qid"]
            and audit["source_language"] == original["source_language"]
            and audit["source_description"] == original["source_description"]
            and audit["translation_raw_sha256"]
            == sha256_text(original["translation_raw"]),
            "Revalidation audit source or raw hash changed",
        )
        require(
            audit["revalidated_generation_status"]
            == revalidated["generation_status"]
            == "OK"
            and audit["translation_clean_revalidated"]
            == revalidated["translation_clean"]
            and not audit["revalidation_error_message"]
            and not revalidated["error_message"]
            and audit["status_changed"] == "YES"
            and audit["revalidation_reason"]
            == acceptance_reason(original["error_message"]),
            "Revalidation audit decision differs from the published row",
        )
        require(
            audit["validation_policy_version"] == VALIDATION_POLICY_VERSION
            and audit["original_results_sha256"] == original_results_sha256
            and audit["original_review_sha256"] == original_review_sha256
            and audit["revalidation_script_sha256"] == script_sha256,
            "Revalidation audit provenance changed",
        )


def validate_review_rows(review_rows: Sequence[dict[str, str]]) -> None:
    require(
        len(review_rows) == EXPECTED_REVALIDATED_REVIEW_ROWS,
        f"Expected {EXPECTED_REVALIDATED_REVIEW_ROWS} revalidated review rows, "
        f"found {len(review_rows)}",
    )
    seen_ids: set[str] = set()
    for row_number, row in enumerate(review_rows, start=2):
        require(
            tuple(row) == REVIEW_FIELDS,
            f"Unexpected revalidated review schema at row {row_number}",
        )
        contextual_id = row["contextual_translation_input_id"]
        require(
            contextual_id not in seen_ids,
            f"Duplicate revalidated review ID at row {row_number}",
        )
        require(
            row["review_recommended"] == "YES"
            and row["alerts_are_advisory"] == "YES"
            and row["disclaimer_alerts_are_not_semantic_evaluation"]
            == REVIEW_DISCLAIMER,
            f"Revalidated review disclaimer changed at row {row_number}",
        )
        seen_ids.add(contextual_id)
    require(
        count_alerts(review_rows) == EXPECTED_REVALIDATED_ALERT_COUNTS,
        f"Unexpected revalidated review alert counts: "
        f"{count_alerts(review_rows)!r}",
    )


def serialize_csv(
    fields: Sequence[str], rows: Sequence[dict[str, str]]
) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return codecs.BOM_UTF8 + text.getvalue().encode("utf-8")


def fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | (os.O_DIRECTORY if os.name != "nt" else 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as error:
        raise RevalidationError(
            f"Could not open parent directory for durability sync: {error}"
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise RevalidationError(
            f"Could not durability-sync parent directory: {error}"
        ) from error
    finally:
        os.close(descriptor)


def write_temporary(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"Refusing to overwrite temporary file {path.name}")
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_parent_directory(path)
    except Exception:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def publish_without_overwrite(temporary: Path, destination: Path) -> None:
    require(temporary.is_file(), f"Missing temporary file {temporary.name}")
    require(
        not destination.exists(),
        f"Refusing to overwrite existing output {destination.name}",
    )
    linked = False
    try:
        os.link(temporary, destination)
        linked = True
        fsync_parent_directory(destination)
    except FileExistsError as error:
        raise RevalidationError(
            f"Refusing to overwrite concurrently created {destination.name}"
        ) from error
    except OSError as error:
        raise RevalidationError(
            f"Atomic no-overwrite publication failed for {destination.name}: {error}"
        ) from error
    except Exception:
        if linked:
            try:
                destination.unlink()
                fsync_parent_directory(destination)
            except (OSError, RevalidationError):
                pass
        raise
    try:
        temporary.unlink()
        fsync_parent_directory(temporary)
    except (OSError, RevalidationError) as error:
        print(
            f"WARNING_TEMPORARY_CLEANUP={temporary.name}: {error}",
            file=sys.stderr,
        )


def acquire_run_lock(script_sha256: str) -> str:
    token = f"pid={os.getpid()};script_sha256={script_sha256}\n"
    created = False
    try:
        with RUN_LOCK_PATH.open("x", encoding="ascii", newline="") as handle:
            created = True
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_parent_directory(RUN_LOCK_PATH)
    except FileExistsError as error:
        raise RevalidationError(
            f"Revalidation lock already exists: {RUN_LOCK_PATH.name}"
        ) from error
    except OSError as error:
        if created:
            try:
                RUN_LOCK_PATH.unlink()
            except OSError:
                pass
        raise RevalidationError(f"Could not create revalidation lock: {error}") from error
    except Exception:
        if created:
            try:
                RUN_LOCK_PATH.unlink()
            except OSError:
                pass
        raise
    return token


def release_run_lock(token: str) -> None:
    try:
        current = RUN_LOCK_PATH.read_text(encoding="ascii")
        require(current == token, "Revalidation lock ownership changed")
        RUN_LOCK_PATH.unlink()
        fsync_parent_directory(RUN_LOCK_PATH)
    except FileNotFoundError:
        raise RevalidationError("Revalidation lock disappeared")
    except OSError as error:
        raise RevalidationError(f"Could not release revalidation lock: {error}") from error


def load_and_validate_inputs() -> tuple[
    list[dict[str, str]],
    str,
    list[dict[str, str]],
    str,
]:
    require_no_model_usage()
    validate_policy_boundaries()
    original_rows, original_sha256 = read_csv_snapshot(
        ORIGINAL_RESULTS_PATH,
        RESULT_FIELDS,
        EXPECTED_ORIGINAL_RESULTS_SHA256,
    )
    review_rows, review_sha256 = read_csv_snapshot(
        ORIGINAL_REVIEW_PATH,
        REVIEW_FIELDS,
        EXPECTED_ORIGINAL_REVIEW_SHA256,
    )
    validate_original_rows(original_rows)
    validate_original_review(original_rows, review_rows)
    return original_rows, original_sha256, review_rows, review_sha256


def assert_raises(
    action: Callable[[], object], expected_fragment: str
) -> None:
    try:
        action()
    except MalformedGenerationError as error:
        require(
            expected_fragment.lower() in str(error).lower(),
            f"Expected error containing {expected_fragment!r}, got {str(error)!r}",
        )
        return
    raise RevalidationError(
        f"Expected MalformedGenerationError containing {expected_fragment!r}"
    )


def validate_policy_boundaries() -> None:
    valid_outputs = (
        'degree of the Ukrainian Order "For Merit"',
        'Order "Palmes academiques"',
        'Names of weapons, for example, tanks "Pantsir" and others.',
        "English-language periodical journal",
        "English-speaking population",
        "Answer-based classification",
        "multiplication of 2 * 3",
        "highest cooperation body between the EU and EEA/EFTA countries",
    )
    for value in valid_outputs:
        require(
            normalize_and_validate_generation(value) == value,
            f"Valid structural boundary case was rejected: {value!r}",
        )
    for header in (
        "Translation: station",
        "English translation: station",
        "Translation into English: station",
        "English - station",
        "Answer: station",
        "Answer - station",
        "Answer is station",
        "Output is: station",
        "Result would be station",
        "Here\u2019s the translation: station",
    ):
        assert_raises(
            lambda value=header: normalize_and_validate_generation(value),
            "presentation header",
        )
    for alternatives in (
        "railway station / train station",
        "railway station /train station",
        "railway station/ train station",
        "railway station|train station",
    ):
        assert_raises(
            lambda value=alternatives: normalize_and_validate_generation(value),
            "alternative separator",
        )
    for wrapped in (
        '"station"',
        "\u201cstation\u201d",
        "\u201estation\u201c",
        "'station'",
    ):
        assert_raises(
            lambda value=wrapped: normalize_and_validate_generation(value),
            "wrapped in presentation quotation marks",
        )
    for markdown in (
        "text **bold** here",
        "_station_",
        "[station](https://example.invalid)",
        "> station",
        ">station",
        "-\tstation",
    ):
        assert_raises(
            lambda value=markdown: normalize_and_validate_generation(value),
            "Markdown structure",
        )
    for multiline in (
        "station\n",
        "station\n\n",
        "station\ntranslation",
        "station\u0085translation",
        "station\u2028translation",
        "station\u2029translation",
    ):
        assert_raises(
            lambda value=multiline: normalize_and_validate_generation(value),
            "line breaks",
        )


def synthetic_result(
    index: int, raw: str, status: str, error_message: str = ""
) -> dict[str, str]:
    row = {field: f"synthetic-{field}" for field in RESULT_FIELDS}
    row.update(
        {
            "contextual_translation_input_id": f"ctxtrans_{index:064x}",
            "representation_id": f"rep_{index:064x}",
            "qid": f"Q{index + 100}",
            "source_language": "xx",
            "source_description": f"synthetic source {index}",
            "generation_attempts": "1",
            "translation_raw": raw,
            "translation_clean": (
                normalize_and_validate_generation(raw) if status == "OK" else ""
            ),
            "generation_status": status,
            "error_message": error_message,
        }
    )
    return {field: row[field] for field in RESULT_FIELDS}


def run_self_tests() -> None:
    require_no_model_usage()
    validate_policy_boundaries()
    internal_quotes = (
        'degree of the Ukrainian Order "For Merit"',
        'Order "Palmes academiques"',
        'Names of weapons, for example, tanks "Pantsir" and others.',
    )
    for value in internal_quotes:
        require(
            normalize_and_validate_generation(value) == value,
            "Internal quotation marks were changed or rejected",
        )
    english_language = "English-language periodical journal"
    require(
        normalize_and_validate_generation(english_language) == english_language,
        "English-language was treated as a presentation header",
    )
    acronym_slash = "highest cooperation body between the EU and EEA/EFTA countries"
    require(
        normalize_and_validate_generation(acronym_slash) == acronym_slash,
        "A lexicalized acronym slash was treated as alternatives",
    )
    for header in (
        "Translation: station",
        "English translation: station",
        "Translation into English: station",
        "Answer: station",
    ):
        assert_raises(
            lambda value=header: normalize_and_validate_generation(value),
            "presentation header",
        )
    assert_raises(
        lambda: normalize_and_validate_generation(
            "railway station / train station"
        ),
        "alternative separator",
    )
    for wrapped in ('"station"', "\u201cstation\u201d", "'station'"):
        assert_raises(
            lambda value=wrapped: normalize_and_validate_generation(value),
            "wrapped in presentation quotation marks",
        )

    synthetic_rows = [
        synthetic_result(1, "station", "OK"),
        synthetic_result(
            2,
            'degree of the Ukrainian Order "For Merit"',
            "ERROR",
            "MalformedGenerationError: Generation contains quotation marks",
        ),
    ]
    first_rows, first_audit = revalidate_rows(
        synthetic_rows, "A" * 64, "B" * 64, "C" * 64
    )
    second_rows, second_audit = revalidate_rows(
        synthetic_rows, "A" * 64, "B" * 64, "C" * 64
    )
    require(
        first_rows == second_rows
        and first_audit == second_audit
        and serialize_csv(RESULT_FIELDS, first_rows)
        == serialize_csv(RESULT_FIELDS, second_rows),
        "Synthetic revalidation is not deterministic",
    )
    require(
        first_rows[0] == synthetic_rows[0],
        "A previously OK synthetic row changed",
    )
    require(
        first_rows[1]["translation_raw"]
        == synthetic_rows[1]["translation_raw"],
        "Synthetic raw output changed",
    )
    require(
        tuple(first_rows[0]) == RESULT_FIELDS
        and tuple(first_audit[0]) == AUDIT_FIELDS,
        "Synthetic output schema changed",
    )
    require_no_model_usage()
    print("INTERNAL_QUOTES_ACCEPTANCE_TEST=OK")
    print("ENGLISH_LANGUAGE_NOT_HEADER_TEST=OK")
    print("ACRONYM_SLASH_ACCEPTANCE_TEST=OK")
    print("REAL_HEADER_REJECTION_TEST=OK")
    print("REAL_ALTERNATIVES_REJECTION_TEST=OK")
    print("WRAPPING_PRESENTATION_QUOTES_TEST=OK")
    print("ALREADY_OK_ROWS_STABILITY_TEST=OK")
    print("RAW_TRANSLATION_IMMUTABILITY_TEST=OK")
    print("OUTPUT_SCHEMA_TEST=OK")
    print("DETERMINISM_TEST=OK")
    print("NO_MODEL_USAGE_TEST=OK")
    print("MODEL_LOADED=NO")
    print("INFERENCE_EXECUTED=NO")


def compute_revalidation(
    original_rows: Sequence[dict[str, str]],
    original_sha256: str,
    review_sha256: str,
    script_sha256: str,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    first_rows, first_audit = revalidate_rows(
        original_rows, original_sha256, review_sha256, script_sha256
    )
    second_rows, second_audit = revalidate_rows(
        original_rows, original_sha256, review_sha256, script_sha256
    )
    require(
        first_rows == second_rows and first_audit == second_audit,
        "Repeated in-memory revalidation produced different data",
    )
    validate_revalidated_rows(
        original_rows,
        first_rows,
        first_audit,
        original_sha256,
        review_sha256,
        script_sha256,
    )
    first_review = build_review_rows(first_rows)
    second_review = build_review_rows(second_rows)
    require(
        first_review == second_review,
        "Repeated heuristic review reconstruction produced different data",
    )
    validate_review_rows(first_review)
    require(
        serialize_csv(RESULT_FIELDS, first_rows)
        == serialize_csv(RESULT_FIELDS, second_rows)
        and serialize_csv(AUDIT_FIELDS, first_audit)
        == serialize_csv(AUDIT_FIELDS, second_audit)
        and serialize_csv(REVIEW_FIELDS, first_review)
        == serialize_csv(REVIEW_FIELDS, second_review),
        "Repeated serialization produced different bytes",
    )
    return first_rows, first_audit, first_review


def print_validation_summary(
    original_rows: Sequence[dict[str, str]],
    revalidated_rows: Sequence[dict[str, str]],
    audit_rows: Sequence[dict[str, str]],
    review_rows: Sequence[dict[str, str]],
) -> None:
    before = Counter(row["generation_status"] for row in original_rows)
    after = Counter(row["generation_status"] for row in revalidated_rows)
    print(f"TOTAL_ROWS={len(revalidated_rows)}")
    print(f"OK_BEFORE={before['OK']}")
    print(f"ERROR_BEFORE={before['ERROR']}")
    print(f"OK_AFTER={after['OK']}")
    print(f"ERROR_AFTER={after['ERROR']}")
    print(f"AUDIT_ROWS={len(audit_rows)}")
    print(f"REVALIDATED_REVIEW_ROWS={len(review_rows)}")
    print(
        "REVALIDATED_REVIEW_ALERT_COUNTS="
        + repr(dict(sorted(count_alerts(review_rows).items())))
    )


def validate_inputs_only() -> None:
    require_no_model_usage()
    original_rows, original_sha256, _, review_sha256 = load_and_validate_inputs()
    absent = [
        path.name
        for path in (
            REVALIDATED_RESULTS_PATH,
            REVALIDATION_AUDIT_PATH,
            REVALIDATED_REVIEW_PATH,
            temporary_path(REVALIDATED_RESULTS_PATH),
            temporary_path(REVALIDATION_AUDIT_PATH),
            temporary_path(REVALIDATED_REVIEW_PATH),
            RUN_LOCK_PATH,
        )
        if path.exists()
    ]
    require(
        not absent,
        "Revalidation outputs, temporaries, or lock already exist: "
        + ", ".join(absent),
    )
    script_sha256 = sha256_file(Path(__file__).resolve())
    revalidated_rows, audit_rows, review_rows = compute_revalidation(
        original_rows, original_sha256, review_sha256, script_sha256
    )
    require(
        sha256_file(ORIGINAL_RESULTS_PATH) == original_sha256
        and sha256_file(ORIGINAL_REVIEW_PATH) == review_sha256,
        "An original input changed during validation",
    )
    print(f"ORIGINAL_RESULTS_SHA256={original_sha256}")
    print(f"ORIGINAL_REVIEW_SHA256={review_sha256}")
    print(f"REVALIDATION_SCRIPT_SHA256={script_sha256}")
    print_validation_summary(
        original_rows, revalidated_rows, audit_rows, review_rows
    )
    print("ORIGINAL_RESULTS_HASH_TEST=OK")
    print("ROW_COUNT_PRESERVATION_TEST=OK")
    print("REPRESENTATION_ID_PRESERVATION_TEST=OK")
    print("RAW_TRANSLATION_IMMUTABILITY_TEST=OK")
    print("ALREADY_OK_ROWS_STABILITY_TEST=OK")
    print("OUTPUT_SCHEMA_TEST=OK")
    print("DETERMINISM_TEST=OK")
    print("HISTORICAL_REVIEW_RECONSTRUCTION_TEST=OK")
    print("OUTPUTS_WRITTEN=NO")
    require_no_model_usage()
    print("NO_MODEL_USAGE_TEST=OK")
    print("MODEL_LOADED=NO")
    print("INFERENCE_EXECUTED=NO")


def run_revalidation() -> None:
    require_no_model_usage()
    destinations = (
        REVALIDATED_RESULTS_PATH,
        REVALIDATION_AUDIT_PATH,
        REVALIDATED_REVIEW_PATH,
    )
    temporary_paths = tuple(temporary_path(path) for path in destinations)
    existing = [
        path.name
        for path in (*destinations, *temporary_paths, RUN_LOCK_PATH)
        if path.exists()
    ]
    require(
        not existing,
        "Refusing to overwrite existing revalidation artifacts: "
        + ", ".join(existing),
    )
    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    lock_token = acquire_run_lock(script_sha256)
    primary_error: BaseException | None = None
    payloads: tuple[bytes, ...] = ()
    published: list[Path] = []
    publication_complete = False
    try:
        original_rows, original_sha256, _, review_sha256 = load_and_validate_inputs()
        revalidated_rows, audit_rows, review_rows = compute_revalidation(
            original_rows, original_sha256, review_sha256, script_sha256
        )
        payloads = (
            serialize_csv(RESULT_FIELDS, revalidated_rows),
            serialize_csv(AUDIT_FIELDS, audit_rows),
            serialize_csv(REVIEW_FIELDS, review_rows),
        )
        for temporary, payload in zip(temporary_paths, payloads):
            write_temporary(temporary, payload)
        roundtrip_results = parse_csv_payload(
            temporary_paths[0].read_bytes(),
            temporary_paths[0].name,
            RESULT_FIELDS,
        )
        roundtrip_audit = parse_csv_payload(
            temporary_paths[1].read_bytes(),
            temporary_paths[1].name,
            AUDIT_FIELDS,
        )
        roundtrip_review = parse_csv_payload(
            temporary_paths[2].read_bytes(),
            temporary_paths[2].name,
            REVIEW_FIELDS,
        )
        require(
            roundtrip_results == revalidated_rows
            and roundtrip_audit == audit_rows
            and roundtrip_review == review_rows,
            "Temporary CSV round-trip changed revalidation data",
        )
        require(
            sha256_file(ORIGINAL_RESULTS_PATH) == original_sha256
            and sha256_file(ORIGINAL_REVIEW_PATH) == review_sha256
            and sha256_file(script_path) == script_sha256,
            "Input or revalidation code changed before publication",
        )
        require_no_model_usage()
        for temporary, destination in zip(temporary_paths, destinations):
            publish_without_overwrite(temporary, destination)
            published.append(destination)
        for destination, payload in zip(destinations, payloads):
            require(
                destination.read_bytes() == payload,
                f"Published bytes differ from canonical payload: {destination.name}",
            )
        require(
            sha256_file(ORIGINAL_RESULTS_PATH) == original_sha256
            and sha256_file(ORIGINAL_REVIEW_PATH) == review_sha256
            and sha256_file(script_path) == script_sha256,
            "An original input changed during publication",
        )
        require_no_model_usage()
        publication_complete = True
        print(f"ORIGINAL_RESULTS_SHA256={original_sha256}")
        print(f"ORIGINAL_REVIEW_SHA256={review_sha256}")
        print(f"REVALIDATION_SCRIPT_SHA256={script_sha256}")
        print(
            f"REVALIDATED_RESULTS_SHA256={sha256_bytes(payloads[0])}"
        )
        print(
            f"REVALIDATION_AUDIT_SHA256={sha256_bytes(payloads[1])}"
        )
        print(
            f"REVALIDATED_REVIEW_SHA256={sha256_bytes(payloads[2])}"
        )
        print_validation_summary(
            original_rows, revalidated_rows, audit_rows, review_rows
        )
        print("ORIGINAL_RESULTS_HASH_TEST=OK")
        print("ROW_COUNT_PRESERVATION_TEST=OK")
        print("REPRESENTATION_ID_PRESERVATION_TEST=OK")
        print("RAW_TRANSLATION_IMMUTABILITY_TEST=OK")
        print("ALREADY_OK_ROWS_STABILITY_TEST=OK")
        print("OUTPUT_SCHEMA_TEST=OK")
        print("DETERMINISM_TEST=OK")
        print("ORIGINAL_RESULTS_MODIFIED=NO")
        print("NO_MODEL_USAGE_TEST=OK")
        print("MODEL_LOADED=NO")
        print("INFERENCE_EXECUTED=NO")
    except BaseException as error:
        primary_error = error
        if not publication_complete:
            for destination, payload in reversed(
                list(zip(destinations, payloads))
            ):
                if destination not in published or not destination.exists():
                    continue
                try:
                    require(
                        destination.read_bytes() == payload,
                        f"Refusing to remove non-owned partial output "
                        f"{destination.name}",
                    )
                    destination.unlink()
                    fsync_parent_directory(destination)
                except Exception as cleanup_error:
                    print(
                        f"WARNING_PARTIAL_OUTPUT_CLEANUP={destination.name}: "
                        f"{cleanup_error}",
                        file=sys.stderr,
                    )
            for temporary in temporary_paths:
                if temporary.exists():
                    try:
                        temporary.unlink()
                        fsync_parent_directory(temporary)
                    except Exception as cleanup_error:
                        print(
                            f"WARNING_TEMPORARY_CLEANUP={temporary.name}: "
                            f"{cleanup_error}",
                            file=sys.stderr,
                        )
        raise
    finally:
        try:
            release_run_lock(lock_token)
        except Exception as lock_error:
            if primary_error is None:
                raise
            print(
                f"WARNING_RUN_LOCK_RELEASE={lock_error}", file=sys.stderr
            )


def verify_outputs() -> None:
    require_no_model_usage()
    original_rows, original_sha256, _, review_sha256 = load_and_validate_inputs()
    script_sha256 = sha256_file(Path(__file__).resolve())
    expected_results, expected_audit, expected_review = compute_revalidation(
        original_rows, original_sha256, review_sha256, script_sha256
    )
    expected_payloads = (
        serialize_csv(RESULT_FIELDS, expected_results),
        serialize_csv(AUDIT_FIELDS, expected_audit),
        serialize_csv(REVIEW_FIELDS, expected_review),
    )
    output_paths = (
        REVALIDATED_RESULTS_PATH,
        REVALIDATION_AUDIT_PATH,
        REVALIDATED_REVIEW_PATH,
    )
    for path in output_paths:
        require(path.is_file(), f"Required output does not exist: {path.name}")
    actual_payloads = tuple(path.read_bytes() for path in output_paths)
    require(
        actual_payloads == expected_payloads,
        "Published output bytes differ from canonical deterministic serialization",
    )
    actual_results = parse_csv_payload(
        actual_payloads[0], REVALIDATED_RESULTS_PATH.name, RESULT_FIELDS
    )
    actual_audit = parse_csv_payload(
        actual_payloads[1], REVALIDATION_AUDIT_PATH.name, AUDIT_FIELDS
    )
    actual_review = parse_csv_payload(
        actual_payloads[2], REVALIDATED_REVIEW_PATH.name, REVIEW_FIELDS
    )
    results_sha256, audit_sha256, review_output_sha256 = (
        sha256_bytes(payload) for payload in actual_payloads
    )
    require(
        actual_results == expected_results,
        "Published revalidated result differs from deterministic reconstruction",
    )
    require(
        actual_audit == expected_audit,
        "Published audit differs from deterministic reconstruction",
    )
    require(
        actual_review == expected_review,
        "Published review differs from deterministic reconstruction",
    )
    require(
        sha256_file(ORIGINAL_RESULTS_PATH) == original_sha256
        and sha256_file(ORIGINAL_REVIEW_PATH) == review_sha256,
        "An original input changed after revalidation",
    )
    print(f"ORIGINAL_RESULTS_SHA256={original_sha256}")
    print(f"REVALIDATION_SCRIPT_SHA256={script_sha256}")
    print(f"REVALIDATED_RESULTS_SHA256={results_sha256}")
    print(f"REVALIDATION_AUDIT_SHA256={audit_sha256}")
    print(f"REVALIDATED_REVIEW_SHA256={review_output_sha256}")
    print_validation_summary(
        original_rows, actual_results, actual_audit, actual_review
    )
    print("ORIGINAL_RESULTS_HASH_TEST=OK")
    print("ROW_COUNT_PRESERVATION_TEST=OK")
    print("REPRESENTATION_ID_PRESERVATION_TEST=OK")
    print("RAW_TRANSLATION_IMMUTABILITY_TEST=OK")
    print("ALREADY_OK_ROWS_STABILITY_TEST=OK")
    print("OUTPUT_SCHEMA_TEST=OK")
    print("DETERMINISM_TEST=OK")
    print("PUBLISHED_OUTPUT_VERIFICATION=OK")
    print("ORIGINAL_RESULTS_MODIFIED=NO")
    require_no_model_usage()
    print("NO_MODEL_USAGE_TEST=OK")
    print("MODEL_LOADED=NO")
    print("INFERENCE_EXECUTED=NO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically revalidate existing contextual Qwen V2 output "
            "without loading or calling any model."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic format-policy tests without reading project CSVs",
    )
    mode.add_argument(
        "--validate-inputs",
        action="store_true",
        help="validate inputs and compute the revalidation without writing outputs",
    )
    mode.add_argument(
        "--verify-outputs",
        action="store_true",
        help="verify existing outputs against deterministic reconstruction",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        if args.self_test:
            run_self_tests()
        elif args.validate_inputs:
            validate_inputs_only()
        elif args.verify_outputs:
            verify_outputs()
        else:
            run_revalidation()
    except RevalidationError as error:
        print(f"REVALIDATION_FAILURE={error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
