#!/usr/bin/env python3
"""Blind Qwen3-14B few-shot sufficiency classification of the fixed 20 cases.

Inference reads only the evaluation sample and the frozen 10-example CSV.
--self-test, --validate-inputs, and --audit-example-sources never import an ML
runtime or generate real decisions. Source auditing is a separate local mode.
The standalone runtime preserves the validated zero-shot gate configuration,
strict parsing, atomic persistence, kernel locking, and crash/no-retry policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "04_defterm_muestra_evaluacion.csv"
FEWSHOT_PATH = BASE_DIR / "05_sufficiency_gate_fewshot_examples.csv"
OUTPUT_PATH = BASE_DIR / "05_defterm_qwen14_sufficiency_gate_fewshot_results.csv"
RUN_LOCK_PATH = BASE_DIR / ".05_defterm_qwen14_sufficiency_gate_fewshot.lock"
INFLIGHT_PATH = BASE_DIR / ".05_defterm_qwen14_sufficiency_gate_fewshot.inflight.json"
MODEL_INPUT_PATHS = (INPUT_PATH, FEWSHOT_PATH)

MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
EXPERIMENT_TYPE = "SUFFICIENCY_GATE_FEW_SHOT"
PROMPT_VERSION = "defterm-sufficiency-gate-en-fewshot-v1"
EXPECTED_INPUT_SHA256 = "A77C2D635CF8F0793ED550C1ABAB363BB2DBA5F37DD72A4EB7B1623374A4358C"
EXPECTED_FEWSHOT_SHA256 = "70F1F056F7C8DF92994F1474FF059BE9C90218D5BE0E214B3837157CA4079E73"
EXPECTED_SAMPLE_SIZE = 20
EXPECTED_FEWSHOT_COUNT = 10

SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
VALID_DECISIONS = frozenset({SUFFICIENT_EVIDENCE, INSUFFICIENT_EVIDENCE})
EXPECTED_CLASS_BALANCE = {SUFFICIENT_EVIDENCE: 6, INSUFFICIENT_EVIDENCE: 4}
POSITIVE_PROVENANCE = "EXISTING_APPROVED_DEFTERM_DEFINITION"
NEGATIVE_PROVENANCE = "CHATGPT_ASSISTED_REVIEW"
POSITIVE_NOTES = (
    "Existing APPROVED_MANUAL DefTerm definition provides the external justification "
    "for this positive example. The definition is not included in the gate prompt."
)

# Fixed, interleaved order chosen before inference; it is not tuned to results.
EXPECTED_EXAMPLE_ORDER = (
    ("European Central Bank", "Q76161279", SUFFICIENT_EVIDENCE),
    ("Timely", "Q124407807", INSUFFICIENT_EVIDENCE),
    ("regulation", "Q18845817", SUFFICIENT_EVIDENCE),
    ("solely", "Q112204255", INSUFFICIENT_EVIDENCE),
    ("Commission", "Q2986626", SUFFICIENT_EVIDENCE),
    ("transparency", "Q4380181", INSUFFICIENT_EVIDENCE),
    ("ESA", "Q5358555", SUFFICIENT_EVIDENCE),
    ("EEA", "Q7887783", INSUFFICIENT_EVIDENCE),
    ("structure", "Q28659665", SUFFICIENT_EVIDENCE),
    ("covering", "Q134100366", SUFFICIENT_EVIDENCE),
)
EXCLUDED_POOL_QIDS = frozenset(
    {
        "Q48739775", "Q4048730", "Q11649773", "Q3775266",
        "Q108583230", "Q19065086", "Q6814224", "Q25515502",
    }
)
NEGATIVE_ANNOTATIONS = {
    "Q124407807": (
        "Timely",
        "Label: disclosure in accounting\n"
        "Aliases: timely disclosure policy; timely disclosure information\n"
        "Subclass of: accounting",
        "`accounting` is too broad to provide the semantic type/hypernym required "
        "to define the disclosure faithfully without completing unsupported information.",
    ),
    "Q112204255": (
        "solely",
        "Instance of: organization",
        "The evidence only states the extremely general type `organization`, "
        "with no label or distinguishing information.",
    ),
    "Q4380181": (
        "transparency",
        "Label: Transparency (in the public sphere)\nInstance of: concept",
        "The generic type `concept` does not provide enough definitional "
        "structure to characterize the candidate faithfully.",
    ),
    "Q7887783": (
        "EEA",
        "Label: environment of evolutionary adaptedness\nInstance of: term",
        "The generic type `term` does not explain the concept denoted by the label; "
        "a meaningful intensional definition would require unsupported semantic completion.",
    ),
}

GENERATION_ATTEMPTS = 1
INFERENCE_SEED = 42
MAX_NEW_TOKENS = 16
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
ENABLE_THINKING = False
DO_SAMPLE = False
NUM_BEAMS = 1
NUM_RETURN_SEQUENCES = 1
REPETITION_PENALTY = 1.1
TRUST_REMOTE_CODE = False
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_RUNTIME_VERSIONS = {
    "accelerate": "1.0.0", "bitsandbytes": "0.45.0",
    "torch": "2.1.0", "transformers": "4.51.0",
}
EXCLUSIVE_MAXIMUM_RUNTIME_VERSIONS = {
    "accelerate": "2.0.0", "bitsandbytes": "1.0.0",
    "torch": "3.0.0", "transformers": "5.0.0",
}

# Identical to the validated zero-shot system prompt and evaluation question.
SYSTEM_MSG = """You are an expert terminologist evaluating whether the supplied semantic evidence is sufficient to support a faithful intensional definition.

Do NOT write a definition.

Decide only whether a meaningful minimal definition could be written using exclusively the supplied evidence, without introducing unsupported information or external knowledge.

A candidate has SUFFICIENT_EVIDENCE only when the evidence supports the semantic type or immediate hypernym needed for a minimal definition and any essential distinguishing information required to characterize the candidate faithfully.

A candidate has INSUFFICIENT_EVIDENCE when a definition would require guessing a semantic type, inventing a hypernym, adding external facts, completing information from general world knowledge, relying only on names or aliases that do not provide enough definitional structure, using a relation that is too general to characterize the candidate minimally, or resolving contradictory evidence with information that was not supplied.

Do not assume that a P31, P279, alias, or label is automatically sufficient or insufficient. Judge the concrete semantic content of the evidence provided. A sufficiently specific relation may support a minimal definition, while an excessively general relation may not.

Output exactly one of:

SUFFICIENT_EVIDENCE
INSUFFICIENT_EVIDENCE

Output nothing else."""

FEWSHOT_USER_MSG_TEMPLATE = """Semantic evidence about the Wikidata candidate:

{defterm_evidence_text}

Decision:"""

USER_MSG_TEMPLATE = """Semantic evidence about the Wikidata candidate:

{defterm_evidence_text}

Is this evidence sufficient to write a faithful minimal intensional definition without adding unsupported information?

Output exactly SUFFICIENT_EVIDENCE or INSUFFICIENT_EVIDENCE.

Decision:"""

EXPECTED_SYSTEM_PROMPT_SHA256 = "8B25CEE520C97DFE8326871140BD3FF9DA9CEB20CD5A05A76E5531E953948200"
EXPECTED_USER_PROMPT_TEMPLATE_SHA256 = "C2D2EECCE531BED72BBCAACBA31A565CEDE6EE51515DB120E3181151B7972EEE"
EXPECTED_FEWSHOT_USER_TEMPLATE_SHA256 = "FC5C85FD40DF37F58F9EA8E2580CF0D8A2D4E3879C69F35E9B314AB7F1091A76"
EXPECTED_FEWSHOT_MESSAGES_SHA256 = "826ACCCB0F60638A8AD2D49424EB126A72379F1E6D4478FC7BAF4DA8BED7A645"

INPUT_FIELDS = (
    "defterm_input_id", "term", "qid", "label_en", "aliases_en",
    "p31", "p279", "p1343", "has_aliases", "has_p31", "has_p279", "has_p1343",
    "aliases_count", "p31_count", "p279_count", "p1343_count",
    "evidence_profile", "evidence_sparsity", "defterm_evidence_text",
    "evaluation_selection_reason",
)
FEWSHOT_FIELDS = (
    "sufficiency_example_id", "defterm_input_id", "term", "qid",
    "evidence_profile", "evidence_sparsity", "defterm_evidence_text",
    "sufficiency_label", "annotation_provenance", "annotation_notes",
    "source_file", "source_sha256",
)
TRACEABILITY_FIELDS = (
    "defterm_input_id", "term", "qid", "label_en",
    "evidence_profile", "evidence_sparsity", "defterm_evidence_text",
)
EXAMPLE_IDENTITY_FIELDS = tuple(field for field in TRACEABILITY_FIELDS if field != "label_en")

# Used ONLY by --audit-example-sources, never by inference or normal self-tests.
APPROVED_SOURCE_FILE = "05_fewshot_ejemplos_aprobados.csv"
POOL_SOURCE_FILE = "05_sufficiency_gate_fewshot_candidates.csv"
CURRENT_SOURCE_FILE = "04_defterm_entradas_contextual_v2.csv"
SOURCE_HASHES = {
    APPROVED_SOURCE_FILE: "AC83C1996824E1D2645FCBCC56FBA14282572498DA8000238C3A77D50B088026",
    POOL_SOURCE_FILE: "8CF3BF0AEF5D76077FA03D3731AA92D920D449BEC9569EA0594437F5692D8884",
    CURRENT_SOURCE_FILE: "BE5C2C87EC41D475B42875B6B665CCB7F09F2E8A51A39EF57717A6D0F5339CBD",
}
APPROVED_SOURCE_FIELDS = (
    "fewshot_example_id", *TRACEABILITY_FIELDS,
    "manual_definition", "manual_status", "manual_notes",
)
POOL_SOURCE_FIELDS = (
    "candidate_group", *TRACEABILITY_FIELDS, "selection_reason",
    "proposed_sufficiency_label", "review_status", "review_notes",
    "manual_definition", "fewshot_example_id", "manual_status", "manual_notes",
    "present_in_current_inventory", "evidence_identical", "evaluation_id_overlap",
    "evaluation_qid_overlap", "approved_defterm_fewshot_id_overlap",
    "approved_defterm_fewshot_qid_overlap",
)
INVENTORY_FIELDS = INPUT_FIELDS[:-1]
CURRENT_SOURCE_FIELDS = (*INVENTORY_FIELDS[:8], "representation_route", *INVENTORY_FIELDS[8:])

STATIC_METADATA_FIELDS = (
    "model_id", "model_revision", "experiment_type", "metadata_mode", "prompt_version",
    "input_sha256", "fewshot_sha256", "fewshot_count", "fewshot_class_balance",
    "fewshot_example_order", "system_prompt_sha256", "user_prompt_template_sha256",
    "fewshot_user_template_sha256", "fewshot_messages_sha256",
    "inference_script_sha256", "quantization_config",
)
RUNTIME_METADATA_FIELDS = (
    *STATIC_METADATA_FIELDS, "generation_parameters", "runtime_versions", "device_name",
)
RESULT_FIELDS = (
    *TRACEABILITY_FIELDS, *RUNTIME_METADATA_FIELDS, "generation_attempts",
    "generation_call_status", "sufficiency_decision", "generation_raw",
    "generation_clean", "generation_status", "error_message",
)
RUNTIME_VERSION_KEYS = frozenset(
    {
        "accelerate", "attention_implementation", "bitsandbytes", "chat_template_sha256",
        "cuda", "cuda_driver", "cuda_device_capability", "cuda_matmul_allow_tf32",
        "cudnn", "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic",
        "huggingface_hub", "device_multi_processor_count", "device_total_memory",
        "device_uuid", "model_class", "model_config_dtype", "model_loaded_in_4bit",
        "model_parameter_dtype", "model_primary_device", "resolved_model_commit",
        "platform", "python", "safetensors", "tokenizer_class", "tokenizers", "torch",
        "torch_deterministic_algorithms", "transformers",
    }
)
RUNTIME_BOOLEAN_FIELDS = frozenset(
    {
        "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
        "cudnn_deterministic", "model_loaded_in_4bit", "torch_deterministic_algorithms",
    }
)
RUNTIME_INTEGER_FIELDS = frozenset({"device_multi_processor_count", "device_total_memory"})
REFERENCE_FILENAME = "05_evaluacion_suficiencia_evidencia_COMPLETADA.csv"
PREVIOUS_RESULT_FILENAMES = frozenset(
    {
        "05_defterm_qwen14_sufficiency_gate_results.csv",
        "05_defterm_qwen14_few_shot_abstention_results.csv",
        "05_defterm_qwen14_few_shot_results.csv",
        "05_defterm_qwen32_few_shot_results.csv",
        "05_defterm_qwen14_zero_shot_results.csv",
    }
)
QID_PATTERN = re.compile(r"Q[1-9][0-9]*")
QID_IN_TEXT_PATTERN = re.compile(r"(?<![A-Za-z0-9])Q[1-9][0-9]*(?![A-Za-z0-9])")
DEFTERM_INPUT_ID_PATTERN = re.compile(r"defterm_input_[0-9a-f]{64}")
MODEL_LOADED = False
INFERENCE_EXECUTED = False
FORBIDDEN_MODEL_MODULES = frozenset(
    {
        "accelerate", "bitsandbytes", "ctranslate2", "huggingface_hub", "llama_cpp",
        "onnxruntime", "sentence_transformers", "torch", "transformers",
    }
)
INITIAL_MODEL_MODULES = {name.partition(".")[0] for name in sys.modules} & FORBIDDEN_MODEL_MODULES
REQUIRED_SELF_TESTS = (
    "INPUT_HASH_TEST", "SAMPLE_SIZE_TEST",
    "FEWSHOT_COUNT_TEST", "FEWSHOT_CLASS_BALANCE_TEST",
    "EVALUATION_DISJOINTNESS_TEST", "QID_DISJOINTNESS_TEST",
    "EVIDENCE_DISJOINTNESS_TEST", "POSITIVE_PROVENANCE_TEST",
    "NEGATIVE_PROVENANCE_TEST", "EXCLUDED_POOL_EXAMPLES_TEST",
    "NO_MANUAL_DEFINITION_IN_PROMPT_TEST", "PROMPT_ISOLATION_TEST",
    "EXACT_DECISION_PARSE_TEST", "REFERENCE_EXCLUSION_TEST",
    "PREVIOUS_RESULTS_EXCLUSION_TEST", "SINGLE_GENERATION_POLICY_TEST",
    "NO_RETRY_POLICY_TEST", "OUTPUT_SCHEMA_TEST", "DETERMINISM_TEST",
)


class ValidationError(RuntimeError):
    """An input, prompt, runtime, or output violates the frozen contract."""


@dataclass(frozen=True)
class PromptExample:
    """The complete model-visible projection of one example; no metadata."""

    evidence: str
    decision: str


@dataclass
class Inputs:
    sample: list[dict[str, str]]
    examples: list[dict[str, str]]
    input_hash: str
    fewshot_hash: str


@dataclass(frozen=True)
class StatePaths:
    output: Path
    lock: Path
    inflight: Path


RUN_PATHS = StatePaths(OUTPUT_PATH, RUN_LOCK_PATH, INFLIGHT_PATH)
Generator = Callable[[Any, Any, str], str]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def validate_static_contract() -> None:
    for text, expected, name in (
        (SYSTEM_MSG, EXPECTED_SYSTEM_PROMPT_SHA256, "system"),
        (USER_MSG_TEMPLATE, EXPECTED_USER_PROMPT_TEMPLATE_SHA256, "evaluation user"),
        (FEWSHOT_USER_MSG_TEMPLATE, EXPECTED_FEWSHOT_USER_TEMPLATE_SHA256, "few-shot user"),
    ):
        require(sha256_text(text) == expected, f"Frozen {name} prompt hash differs")
    require(USER_MSG_TEMPLATE.count("{defterm_evidence_text}") == 1, "Evaluation placeholder differs")
    require(FEWSHOT_USER_MSG_TEMPLATE.count("{defterm_evidence_text}") == 1, "Example placeholder differs")


def parse_csv_payload(
    filename: str, payload: bytes, expected_fields: Sequence[str]
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{filename}: invalid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    require(tuple(reader.fieldnames or ()) == tuple(expected_fields), f"{filename}: wrong schema")
    rows = []
    for row in reader:
        require(
            None not in row and all(value is not None for value in row.values()),
            f"{filename}: malformed record near line {reader.line_num}",
        )
        rows.append({field: row[field] for field in expected_fields})
    return rows


def read_fixed_csv(
    path: Path, fields: Sequence[str], expected_hash: str
) -> tuple[list[dict[str, str]], str]:
    require(path.is_file(), f"Required input is missing: {path.name}")
    payload = path.read_bytes()
    actual_hash = sha256_bytes(payload)
    require(actual_hash == expected_hash, f"{path.name}: SHA-256 mismatch ({actual_hash})")
    return parse_csv_payload(path.name, payload, fields), actual_hash


def validate_unique_candidates(rows: list[dict[str, str]], filename: str) -> None:
    ids = [row["defterm_input_id"] for row in rows]
    keys = [(row["term"], row["qid"]) for row in rows]
    require(len(ids) == len(set(ids)), f"{filename}: duplicate DefTerm ID")
    require(len(keys) == len(set(keys)), f"{filename}: duplicate term+QID")
    for row in rows:
        require(DEFTERM_INPUT_ID_PATTERN.fullmatch(row["defterm_input_id"]) is not None, "Invalid DefTerm ID")
        require(QID_PATTERN.fullmatch(row["qid"]) is not None, "Invalid candidate QID")
        require(bool(row["term"].strip()), "Empty term metadata")
        require(bool(row["evidence_profile"].strip()) and bool(row["evidence_sparsity"].strip()), "Empty evidence metadata")
        require(bool(row["defterm_evidence_text"].strip()), "Empty model evidence")
        require(QID_IN_TEXT_PATTERN.search(row["defterm_evidence_text"]) is None, "QID leaked into evidence")


def disjointness_counts(
    sample: list[dict[str, str]], examples: list[dict[str, str]]
) -> dict[str, int]:
    return {
        "defterm_input_id": len({row["defterm_input_id"] for row in sample} & {row["defterm_input_id"] for row in examples}),
        "term_qid": len({(row["term"], row["qid"]) for row in sample} & {(row["term"], row["qid"]) for row in examples}),
        "qid": len({row["qid"] for row in sample} & {row["qid"] for row in examples}),
        "evidence": len({row["defterm_evidence_text"] for row in sample} & {row["defterm_evidence_text"] for row in examples}),
    }


def validate_disjointness(sample: list[dict[str, str]], examples: list[dict[str, str]]) -> None:
    counts = disjointness_counts(sample, examples)
    require(all(count == 0 for count in counts.values()), f"Evaluation contamination: {counts}")


def validate_examples(examples: list[dict[str, str]]) -> None:
    require(len(examples) == EXPECTED_FEWSHOT_COUNT, "Expected exactly 10 examples")
    validate_unique_candidates(examples, FEWSHOT_PATH.name)
    require(len({row["qid"] for row in examples}) == 10, "Duplicate few-shot QID")
    require(len({row["defterm_evidence_text"] for row in examples}) == 10, "Duplicate few-shot evidence")
    require(Counter(row["sufficiency_label"] for row in examples) == EXPECTED_CLASS_BALANCE, "Expected 6 positive and 4 negative examples")
    require(
        tuple((row["term"], row["qid"], row["sufficiency_label"]) for row in examples) == EXPECTED_EXAMPLE_ORDER,
        "Example membership, labels, or fixed order changed",
    )
    require(not EXCLUDED_POOL_QIDS.intersection(row["qid"] for row in examples), "An excluded pool example was included")
    for index, row in enumerate(examples, 1):
        require(row["sufficiency_example_id"] == f"sufficiency_example_{index:02d}", "Example IDs/order changed")
        positive = row["sufficiency_label"] == SUFFICIENT_EVIDENCE
        source_file = APPROVED_SOURCE_FILE if positive else POOL_SOURCE_FILE
        require(row["source_file"] == source_file, "Wrong example source")
        require(row["source_sha256"] == SOURCE_HASHES[source_file], "Wrong example source hash")
        require(row["annotation_provenance"] == (POSITIVE_PROVENANCE if positive else NEGATIVE_PROVENANCE), "Wrong annotation provenance")
        if positive:
            require(row["annotation_notes"] == POSITIVE_NOTES, "Positive justification changed")
        else:
            term, evidence, notes = NEGATIVE_ANNOTATIONS[row["qid"]]
            require((row["term"], row["defterm_evidence_text"], row["annotation_notes"]) == (term, evidence, notes), "Assisted negative annotation changed")


def project_examples(rows: Sequence[dict[str, str]]) -> tuple[PromptExample, ...]:
    # Deliberately access ONLY these two fields at the model boundary.
    return tuple(PromptExample(row["defterm_evidence_text"], row["sufficiency_label"]) for row in rows)


def build_demo_messages(examples: Sequence[PromptExample]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for example in examples:
        require(example.decision in VALID_DECISIONS, "Invalid demonstration decision")
        messages.extend(
            (
                {"role": "user", "content": FEWSHOT_USER_MSG_TEMPLATE.format(defterm_evidence_text=example.evidence)},
                {"role": "assistant", "content": example.decision},
            )
        )
    return messages


def build_messages(evidence: str, examples: Sequence[PromptExample]) -> list[dict[str, str]]:
    return [
        *build_demo_messages(examples),
        {"role": "user", "content": USER_MSG_TEMPLATE.format(defterm_evidence_text=evidence)},
    ]


def load_inputs() -> Inputs:
    validate_static_contract()
    sample, input_hash = read_fixed_csv(INPUT_PATH, INPUT_FIELDS, EXPECTED_INPUT_SHA256)
    examples, fewshot_hash = read_fixed_csv(FEWSHOT_PATH, FEWSHOT_FIELDS, EXPECTED_FEWSHOT_SHA256)
    require(len(sample) == EXPECTED_SAMPLE_SIZE, "The evaluation sample must contain exactly 20 cases")
    validate_unique_candidates(sample, INPUT_PATH.name)
    validate_examples(examples)
    validate_disjointness(sample, examples)
    projected = project_examples(examples)
    require(
        sha256_text(canonical_json(build_demo_messages(projected)))
        == EXPECTED_FEWSHOT_MESSAGES_SHA256,
        "Frozen system/few-shot message content or role order changed",
    )
    for row in sample:
        messages = build_messages(row["defterm_evidence_text"], projected)
        require(len(messages) == 22, "Expected system + 10 user/assistant pairs + evaluation user")
        require(messages[0] == {"role": "system", "content": SYSTEM_MSG}, "System message changed")
        require(messages[-1]["role"] == "user", "Evaluation user message is missing")
    return Inputs(sample, examples, input_hash, fewshot_hash)


def audit_example_sources(inputs: Inputs) -> None:
    """Local provenance/membership audit; not called by the inference route."""
    schemas = {
        APPROVED_SOURCE_FILE: APPROVED_SOURCE_FIELDS,
        POOL_SOURCE_FILE: POOL_SOURCE_FIELDS,
        CURRENT_SOURCE_FILE: CURRENT_SOURCE_FIELDS,
    }
    sources = {
        filename: read_fixed_csv(BASE_DIR / filename, schemas[filename], expected)[0]
        for filename, expected in SOURCE_HASHES.items()
    }
    approved = sources[APPROVED_SOURCE_FILE]
    pool = sources[POOL_SOURCE_FILE]
    current = sources[CURRENT_SOURCE_FILE]
    require((len(approved), len(pool), len(current)) == (6, 18, 150), "Source counts changed")
    for filename, rows in sources.items():
        validate_unique_candidates(rows, filename)
    approved_by_id = {row["defterm_input_id"]: row for row in approved}
    pool_by_id = {row["defterm_input_id"]: row for row in pool}
    current_by_id = {row["defterm_input_id"]: row for row in current}
    positives = [row for row in inputs.examples if row["sufficiency_label"] == SUFFICIENT_EVIDENCE]
    require({row["defterm_input_id"] for row in positives} == set(approved_by_id), "Not exactly the six historical approved examples")
    prompt_text = canonical_json(build_demo_messages(project_examples(inputs.examples)))
    for row in inputs.examples:
        candidate_id = row["defterm_input_id"]
        require(candidate_id in current_by_id and candidate_id in pool_by_id, "Example is absent from current inventory or pool")
        for source in (current_by_id[candidate_id], pool_by_id[candidate_id]):
            require(all(row[field] == source[field] for field in EXAMPLE_IDENTITY_FIELDS), "Example identity/evidence differs from source")
        if row["sufficiency_label"] == SUFFICIENT_EVIDENCE:
            original = approved_by_id[candidate_id]
            require(all(row[field] == original[field] for field in EXAMPLE_IDENTITY_FIELDS), "Approved example changed")
            require(original["manual_status"] == "APPROVED_MANUAL" and bool(original["manual_definition"].strip()), "Missing approved definition")
            require(original["manual_definition"] not in prompt_text, "Historical definition appeared in the gate prompt")
        else:
            require(pool_by_id[candidate_id]["candidate_group"] == "NEGATIVE_REVIEW_CANDIDATE", "Negative is not from the review pool")
            require(pool_by_id[candidate_id]["proposed_sufficiency_label"] == "", "Historical review pool was relabeled")
    for filename, expected in SOURCE_HASHES.items():
        require(sha256_bytes((BASE_DIR / filename).read_bytes()) == expected, "Source changed during audit")
    print("EXAMPLE_SOURCE_AUDIT=OK")
    print("EXACT_HISTORICAL_POSITIVE_SET=6")
    print("EXACT_CHATGPT_ASSISTED_NEGATIVE_SET=4")
    print("CURRENT_INVENTORY_MEMBERSHIP=10")
    print("SOURCE_EVIDENCE_IDENTICAL=10")
    print("MANUAL_DEFINITIONS_IN_PROMPT=0")
    print("HISTORICAL_SOURCES_UNCHANGED=YES")


def parse_sufficiency_output(raw: str) -> tuple[str, str]:
    require(isinstance(raw, str), "Generated output is not a string")
    require(raw in VALID_DECISIONS, "Generated output is not an exact allowed sufficiency label")
    return raw, raw


def quantization_config_json() -> str:
    return canonical_json(
        {
            "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE,
            "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE,
            "bnb_4bit_use_double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
            "load_in_4bit": LOAD_IN_4BIT,
        }
    )


def static_run_metadata(inputs: Inputs, mode: str = "PRODUCTION") -> dict[str, str]:
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "experiment_type": EXPERIMENT_TYPE,
        "metadata_mode": mode,
        "prompt_version": PROMPT_VERSION,
        "input_sha256": inputs.input_hash,
        "fewshot_sha256": inputs.fewshot_hash,
        "fewshot_count": str(len(inputs.examples)),
        "fewshot_class_balance": canonical_json(EXPECTED_CLASS_BALANCE),
        "fewshot_example_order": canonical_json([row["sufficiency_example_id"] for row in inputs.examples]),
        "system_prompt_sha256": sha256_text(SYSTEM_MSG),
        "user_prompt_template_sha256": sha256_text(USER_MSG_TEMPLATE),
        "fewshot_user_template_sha256": sha256_text(FEWSHOT_USER_MSG_TEMPLATE),
        "fewshot_messages_sha256": sha256_text(canonical_json(build_demo_messages(project_examples(inputs.examples)))),
        "inference_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "quantization_config": quantization_config_json(),
    }
    require(tuple(metadata) == STATIC_METADATA_FIELDS, "Static metadata schema differs")
    return metadata


def normalize_token_ids(value: Any, name: str) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    require(bool(values), f"{name} is empty")
    require(all(type(token) is int and token >= 0 for token in values), f"Invalid {name}")
    return tuple(dict.fromkeys(values))


def resolve_generation_token_ids(tokenizer: Any, model: Any) -> tuple[tuple[int, ...], int]:
    config = getattr(model, "generation_config", None)
    eos_value = getattr(config, "eos_token_id", None)
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    eos_ids = normalize_token_ids(eos_value, "effective EOS token IDs")
    pad_id = getattr(config, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]
    require(type(pad_id) is int and pad_id >= 0, "Invalid effective PAD token ID")
    return eos_ids, pad_id


def remove_single_terminal_eos(tokens: Sequence[int], eos_ids: Sequence[int]) -> list[int]:
    content = list(tokens)
    if content and content[-1] in eos_ids:
        content.pop()
    return content


def generation_parameters_json(eos_ids: Sequence[int], pad_id: int) -> str:
    return canonical_json(
        {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "device_map": {"": 0},
            "do_sample": DO_SAMPLE,
            "enable_thinking": ENABLE_THINKING,
            "eos_token_ids": list(eos_ids),
            "inference_seed": INFERENCE_SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "model_torch_dtype": BNB_4BIT_COMPUTE_DTYPE,
            "num_beams": NUM_BEAMS,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
            "pad_token_id": pad_id,
            "repetition_penalty": REPETITION_PENALTY,
            "torch_deterministic_algorithms": True,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "use_cache": True,
        }
    )


def installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValidationError(f"Required runtime package is missing: {package}") from exc


def validate_supported_version(package: str, actual: str) -> None:
    from packaging.version import InvalidVersion, Version

    minimum = MINIMUM_RUNTIME_VERSIONS[package]
    maximum = EXCLUSIVE_MAXIMUM_RUNTIME_VERSIONS[package]
    try:
        version = Version(actual)
        compatible = (
            not version.is_prerelease and not version.is_devrelease
            and Version(minimum) <= version < Version(maximum)
        )
    except InvalidVersion as exc:
        raise ValidationError(f"Invalid {package} version: {actual}") from exc
    require(compatible, f"{package}>={minimum},<{maximum} is required; found {actual}")


def load_cuda_driver_library() -> tuple[Any, Any]:
    import ctypes

    library_name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
    try:
        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        library = loader(library_name)
        library.cuInit.argtypes = (ctypes.c_uint,)
        library.cuInit.restype = ctypes.c_int
        result = library.cuInit(0)
    except (AttributeError, OSError) as exc:
        raise ValidationError("Cannot initialize the CUDA driver API") from exc
    require(result == 0, f"CUDA driver initialization failed: {result}")
    return ctypes, library


def cuda_driver_version(torch_module: Any) -> str:
    for getter in (
        getattr(torch_module.cuda, "driver_version", None),
        getattr(torch_module._C, "_cuda_getDriverVersion", None),
    ):
        if not callable(getter):
            continue
        try:
            value = int(getter())
        except (TypeError, ValueError, RuntimeError):
            continue
        if value > 0:
            return str(value)
    ctypes, library = load_cuda_driver_library()
    try:
        library.cuDriverGetVersion.argtypes = (ctypes.POINTER(ctypes.c_int),)
        library.cuDriverGetVersion.restype = ctypes.c_int
        value = ctypes.c_int()
        result = library.cuDriverGetVersion(ctypes.byref(value))
    except AttributeError as exc:
        raise ValidationError("Cannot query the CUDA driver version") from exc
    require(result == 0 and value.value > 0, "CUDA driver query failed")
    return str(value.value)


def cuda_device_uuid(torch_module: Any, index: int) -> str:
    properties = torch_module.cuda.get_device_properties(index)
    property_uuid = getattr(properties, "uuid", None)
    if property_uuid is not None:
        text = str(property_uuid).strip()
        if text and text.lower() != "none":
            return text
    ctypes, library = load_cuda_driver_library()

    class CuUuid(ctypes.Structure):
        _fields_ = (("bytes", ctypes.c_ubyte * 16),)

    try:
        library.cuDeviceGet.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
        library.cuDeviceGet.restype = ctypes.c_int
        device = ctypes.c_int()
        result = library.cuDeviceGet(ctypes.byref(device), index)
        require(result == 0, f"CUDA device lookup failed: {result}")
        getter = getattr(library, "cuDeviceGetUuid_v2", getattr(library, "cuDeviceGetUuid", None))
        require(getter is not None, "CUDA driver has no UUID function")
        getter.argtypes = (ctypes.POINTER(CuUuid), ctypes.c_int)
        getter.restype = ctypes.c_int
        value = CuUuid()
        result = getter(ctypes.byref(value), device)
    except AttributeError as exc:
        raise ValidationError("Cannot query the CUDA device UUID") from exc
    require(result == 0, f"CUDA UUID query failed: {result}")
    hex_value = bytes(value.bytes).hex().upper()
    return f"{hex_value[:8]}-{hex_value[8:12]}-{hex_value[12:16]}-{hex_value[16:20]}-{hex_value[20:]}"


def configure_determinism(torch_module: Any) -> None:
    torch_module.manual_seed(INFERENCE_SEED)
    torch_module.cuda.manual_seed_all(INFERENCE_SEED)
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.allow_tf32 = False


def load_model_and_tokenizer() -> tuple[Any, Any, Any]:
    global MODEL_LOADED

    validate_static_contract()
    for package in MINIMUM_RUNTIME_VERSIONS:
        validate_supported_version(package, installed_version(package))
    loaded = {name.partition(".")[0] for name in sys.modules}
    require(not FORBIDDEN_MODEL_MODULES.intersection(loaded), "Inference requires a fresh Python process before any ML runtime import")
    require(bool(os.environ.get("HF_TOKEN", "").strip()), "A nonempty HF_TOKEN is required")
    require(not (Path.cwd() / Path(MODEL_ID)).exists(), "A local model path would shadow the pinned Hub revision")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {None, CUBLAS_WORKSPACE_CONFIG}, "Conflicting CUBLAS configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    require(torch.cuda.is_available(), "CUDA is required for this gate configuration")
    require(torch.cuda.is_bf16_supported(), "The configured device lacks bfloat16 support")
    configure_determinism(torch)
    cuda_driver_version(torch)
    cuda_device_uuid(torch, 0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=os.environ["HF_TOKEN"], revision=MODEL_REVISION,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=os.environ["HF_TOKEN"], revision=MODEL_REVISION,
        device_map={"": 0}, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, trust_remote_code=TRUST_REMOTE_CODE,
    )
    MODEL_LOADED = True
    require(getattr(model.config, "_commit_hash", None) == MODEL_REVISION, "Resolved model commit differs")
    require(getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") in {None, MODEL_REVISION}, "Resolved tokenizer commit differs")
    model.eval()
    resolve_generation_token_ids(tokenizer, model)
    return tokenizer, model, torch


def collect_runtime_metadata(
    static: dict[str, str], tokenizer: Any, model: Any, torch_module: Any
) -> dict[str, str]:
    versions = {
        name: installed_version(name) for name in
        ("transformers", "tokenizers", "huggingface_hub", "accelerate", "bitsandbytes", "safetensors")
    }
    versions.update(
        {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch_module.__version__), "cuda": str(torch_module.version.cuda or ""),
            "cuda_driver": cuda_driver_version(torch_module),
            "cudnn": str(torch_module.backends.cudnn.version() or ""),
        }
    )
    properties = torch_module.cuda.get_device_properties(0)
    versions.update(
        {
            "cuda_device_capability": f"{properties.major}.{properties.minor}",
            "device_multi_processor_count": int(properties.multi_processor_count),
            "device_total_memory": int(properties.total_memory),
            "device_uuid": cuda_device_uuid(torch_module, 0),
            "torch_deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        }
    )
    template = getattr(tokenizer, "chat_template", None)
    require(isinstance(template, str) and bool(template), "Chat template is missing")
    parameter = next(model.parameters())
    dtype = getattr(model.config, "dtype", None)
    if dtype is None:
        dtype = getattr(model.config, "torch_dtype", None)
    require(dtype is not None, "Model config dtype is missing")
    versions.update(
        {
            "chat_template_sha256": sha256_text(template),
            "attention_implementation": str(getattr(model.config, "_attn_implementation", "")),
            "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
            "resolved_model_commit": str(getattr(model.config, "_commit_hash", "")),
            "model_primary_device": str(parameter.device),
            "model_parameter_dtype": str(parameter.dtype), "model_config_dtype": str(dtype),
            "model_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        }
    )
    require(versions["model_primary_device"] == "cuda:0", "Model is not on configured GPU 0")
    require(versions["resolved_model_commit"] == MODEL_REVISION, "Runtime commit differs")
    require(versions["model_loaded_in_4bit"], "Model was not loaded in 4-bit mode")
    eos_ids, pad_id = resolve_generation_token_ids(tokenizer, model)
    return {
        **static,
        "generation_parameters": generation_parameters_json(eos_ids, pad_id),
        "runtime_versions": canonical_json(versions),
        "device_name": str(torch_module.cuda.get_device_name(0)),
    }


def synthetic_runtime_metadata(static: dict[str, str]) -> dict[str, str]:
    versions: dict[str, Any] = {name: "synthetic-self-test" for name in RUNTIME_VERSION_KEYS}
    versions.update(
        {
            "accelerate": "1.0.0", "bitsandbytes": "0.45.0", "torch": "2.1.0", "transformers": "4.51.0",
            "chat_template_sha256": sha256_text("synthetic-chat-template"),
            "cuda_device_capability": "8.0", "cuda_driver": "12000",
            "device_multi_processor_count": 1, "device_total_memory": 1,
            "device_uuid": "00000000-0000-0000-0000-000000000000",
            "model_config_dtype": "torch.bfloat16", "model_primary_device": "cuda:0",
            "resolved_model_commit": MODEL_REVISION,
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "model_loaded_in_4bit": True, "torch_deterministic_algorithms": True,
        }
    )
    return {
        **static,
        "generation_parameters": generation_parameters_json((90, 91), 98),
        "runtime_versions": canonical_json(versions),
        "device_name": "SYNTHETIC_NO_DEVICE",
    }


def generate_once(
    tokenizer: Any, model: Any, evidence: str, examples: Sequence[PromptExample],
    torch_module: Any, *, real_inference: bool = True,
) -> str:
    global INFERENCE_EXECUTED

    rendered = tokenizer.apply_chat_template(
        build_messages(evidence, examples), tokenize=False,
        add_generation_prompt=True, enable_thinking=ENABLE_THINKING,
    )
    model_inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    input_count = model_inputs["input_ids"].shape[-1]
    eos_ids, pad_id = resolve_generation_token_ids(tokenizer, model)
    eos_argument: int | list[int] = eos_ids[0] if len(eos_ids) == 1 else list(eos_ids)
    if real_inference:
        INFERENCE_EXECUTED = True
    with torch_module.inference_mode():
        generated = model.generate(
            **model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
            num_beams=NUM_BEAMS, num_return_sequences=NUM_RETURN_SEQUENCES,
            repetition_penalty=REPETITION_PENALTY, eos_token_id=eos_argument,
            pad_token_id=pad_id, use_cache=True,
        )
    content = remove_single_terminal_eos(generated[0, input_count:].tolist(), eos_ids)
    return tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def process_candidate(
    source: dict[str, str], tokenizer: Any, model: Any,
    metadata: dict[str, str], generator: Generator,
) -> dict[str, str]:
    require(tuple(metadata) == RUNTIME_METADATA_FIELDS, "Runtime metadata schema differs")
    raw = clean = decision = error = ""
    status = "ERROR"
    call_status = "FAILED_OR_UNKNOWN"
    try:
        raw = generator(tokenizer, model, source["defterm_evidence_text"])
        call_status = "RETURNED"
        decision, clean = parse_sufficiency_output(raw)
        status = "OK"
    except Exception as exc:
        detail = re.sub(r"\s+", " ", str(exc)).strip()
        error = f"{type(exc).__name__}: {detail or 'No error detail provided'}"
    return {
        **{field: source[field] for field in TRACEABILITY_FIELDS}, **metadata,
        "generation_attempts": str(GENERATION_ATTEMPTS), "generation_call_status": call_status,
        "sufficiency_decision": decision, "generation_raw": raw, "generation_clean": clean,
        "generation_status": status, "error_message": error,
    }


def validate_result_row(
    row: dict[str, str], source: dict[str, str], static: dict[str, str],
    runtime: dict[str, str] | None = None, *, allow_self_test: bool = False,
) -> None:
    require(tuple(row) == RESULT_FIELDS, "Unexpected result schema")
    require(all(row[field] == source[field] for field in TRACEABILITY_FIELDS), "Result traceability differs")
    require(static["metadata_mode"] == ("SELF_TEST" if allow_self_test else "PRODUCTION"), "Synthetic metadata is not allowed in production")
    require({field: row[field] for field in STATIC_METADATA_FIELDS} == static, "Static result metadata differs")
    try:
        parameters = json.loads(row["generation_parameters"])
        versions = json.loads(row["runtime_versions"])
    except json.JSONDecodeError as exc:
        raise ValidationError("Invalid result metadata JSON") from exc
    require(isinstance(parameters, dict) and isinstance(versions, dict), "Runtime JSON must contain objects")
    eos_ids = parameters.get("eos_token_ids")
    pad_id = parameters.get("pad_token_id")
    require(isinstance(eos_ids, list), "Invalid persisted EOS IDs")
    normalized_eos = normalize_token_ids(eos_ids, "persisted EOS IDs")
    require(type(pad_id) is int and pad_id >= 0, "Invalid persisted PAD ID")
    require(row["generation_parameters"] == generation_parameters_json(normalized_eos, pad_id), "Generation configuration differs")
    require(set(versions) == RUNTIME_VERSION_KEYS, "Runtime version keys differ")
    require(canonical_json(versions) == row["runtime_versions"], "Runtime JSON is not canonical")
    for field in RUNTIME_BOOLEAN_FIELDS:
        require(type(versions[field]) is bool, f"Invalid runtime boolean: {field}")
    for field in RUNTIME_INTEGER_FIELDS:
        require(type(versions[field]) is int and versions[field] > 0, f"Invalid runtime integer: {field}")
    for field in RUNTIME_VERSION_KEYS - RUNTIME_BOOLEAN_FIELDS - RUNTIME_INTEGER_FIELDS:
        require(type(versions[field]) is str and bool(versions[field].strip()), f"Invalid runtime string: {field}")
    for package in MINIMUM_RUNTIME_VERSIONS:
        validate_supported_version(package, versions[package])
    require(
        versions["torch_deterministic_algorithms"] is True
        and versions["cudnn_benchmark"] is False
        and versions["cudnn_deterministic"] is True
        and versions["cuda_matmul_allow_tf32"] is False
        and versions["cudnn_allow_tf32"] is False
        and versions["model_loaded_in_4bit"] is True
        and versions["model_primary_device"] == "cuda:0"
        and versions["resolved_model_commit"] == MODEL_REVISION,
        "Effective runtime differs from deterministic/model contract",
    )
    require(re.fullmatch(r"[0-9A-F]{64}", versions["chat_template_sha256"]) is not None, "Invalid template hash")
    require(re.fullmatch(r"[0-9]+\.[0-9]+", versions["cuda_device_capability"]) is not None, "Invalid GPU capability")
    require(versions["cuda_driver"].isdigit() and int(versions["cuda_driver"]) > 0, "Invalid CUDA driver version")
    require(versions["device_uuid"].lower() != "none" and versions["model_config_dtype"] != "None", "Invalid device/dtype metadata")
    require(bool(row["device_name"].strip()), "Device name is missing")
    if runtime is not None:
        require({field: row[field] for field in RUNTIME_METADATA_FIELDS} == runtime, "Active runtime metadata differs")
    require(row["generation_attempts"] == "1", "Generation attempt budget is not exactly one")
    require(row["generation_call_status"] in {"RETURNED", "FAILED_OR_UNKNOWN", "INTERRUPTED_OR_UNKNOWN"}, "Invalid call status")
    require(row["generation_status"] in {"OK", "ERROR"}, "Invalid generation status")
    if row["generation_status"] == "OK":
        require(
            row["generation_call_status"] == "RETURNED"
            and row["generation_raw"] == row["generation_clean"] == row["sufficiency_decision"]
            and row["sufficiency_decision"] in VALID_DECISIONS
            and row["error_message"] == "",
            "Successful result is not an exact label",
        )
    else:
        require(row["sufficiency_decision"] == row["generation_clean"] == "" and bool(row["error_message"].strip()), "Failed result is inconsistent")


def serialize_result_rows(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RESULT_FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def load_existing_results(
    sample: list[dict[str, str]], static: dict[str, str], paths: StatePaths = RUN_PATHS,
    *, allow_self_test: bool = False,
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    if not paths.output.exists():
        return [], None
    payload = paths.output.read_bytes()
    require(bool(payload) and payload.endswith(b"\n"), "Existing output is empty or incomplete")
    rows = parse_csv_payload(paths.output.name, payload, RESULT_FIELDS)
    require(len(rows) <= len(sample), "Output has more rows than the fixed sample")
    require(
        [row["defterm_input_id"] for row in rows] == [row["defterm_input_id"] for row in sample[:len(rows)]],
        "Output is not the exact sample prefix",
    )
    common = None
    for row, source in zip(rows, sample):
        validate_result_row(row, source, static, allow_self_test=allow_self_test)
        runtime = {field: row[field] for field in RUNTIME_METADATA_FIELDS}
        if common is None:
            common = runtime
        else:
            require(runtime == common, "Output mixes incompatible runtimes")
    return rows, common


def sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_file_durably(source: Path, destination: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        if not move(str(source), str(destination), 0x1 | 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    os.replace(source, destination)


def write_bytes_atomically(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file_durably(temporary, path)
        sync_directory(path.parent)
        require(path.read_bytes() == payload, "Committed file bytes differ")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def acquire_run_lock(path: Path = RUN_LOCK_PATH) -> tuple[int, str]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return descriptor, "nt"
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor, "posix"
    except OSError as exc:
        os.close(descriptor)
        raise ValidationError(f"Another run holds {path.name}") from exc


def release_run_lock(lock: tuple[int, str]) -> None:
    descriptor, kind = lock
    try:
        if kind == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def write_inflight_marker(
    source: dict[str, str], metadata: dict[str, str], paths: StatePaths = RUN_PATHS,
) -> None:
    require(not paths.inflight.exists(), "An unresolved attempt marker exists")
    require(tuple(metadata) == RUNTIME_METADATA_FIELDS, "Invalid journal metadata schema")
    marker = {"defterm_input_id": source["defterm_input_id"], "runtime_metadata": metadata}
    write_bytes_atomically(paths.inflight, (canonical_json(marker) + "\n").encode("utf-8"))


def read_inflight_marker(paths: StatePaths = RUN_PATHS) -> dict[str, Any] | None:
    if not paths.inflight.exists():
        return None
    payload = paths.inflight.read_bytes()
    require(payload.endswith(b"\n"), "Attempt marker is incomplete")
    try:
        marker = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Attempt marker is invalid") from exc
    require(
        isinstance(marker, dict) and set(marker) == {"defterm_input_id", "runtime_metadata"}
        and type(marker["defterm_input_id"]) is str
        and isinstance(marker["runtime_metadata"], dict)
        and set(marker["runtime_metadata"]) == set(RUNTIME_METADATA_FIELDS)
        and all(type(value) is str for value in marker["runtime_metadata"].values()),
        "Attempt marker schema differs",
    )
    return {
        "defterm_input_id": marker["defterm_input_id"],
        "runtime_metadata": {field: marker["runtime_metadata"][field] for field in RUNTIME_METADATA_FIELDS},
    }


def clear_inflight_marker(candidate_id: str, paths: StatePaths = RUN_PATHS) -> None:
    marker = read_inflight_marker(paths)
    require(marker is not None and marker["defterm_input_id"] == candidate_id, "Attempt marker changed or disappeared")
    paths.inflight.unlink()
    sync_directory(paths.inflight.parent)
    require(not paths.inflight.exists(), "Attempt marker cleanup failed")


def interrupted_result(source: dict[str, str], metadata: dict[str, str]) -> dict[str, str]:
    return {
        **{field: source[field] for field in TRACEABILITY_FIELDS}, **metadata,
        "generation_attempts": "1", "generation_call_status": "INTERRUPTED_OR_UNKNOWN",
        "sufficiency_decision": "", "generation_raw": "", "generation_clean": "",
        "generation_status": "ERROR",
        "error_message": (
            "InterruptedGeneration: a durable attempt marker existed without a "
            "committed result; the candidate was not retried"
        ),
    }


def recover_interrupted_attempt(
    sample: list[dict[str, str]], rows: list[dict[str, str]], static: dict[str, str],
    common: dict[str, str] | None, paths: StatePaths = RUN_PATHS,
    *, allow_self_test: bool = False,
) -> tuple[list[dict[str, str]], dict[str, str] | None, bool]:
    marker = read_inflight_marker(paths)
    if marker is None:
        return rows, common, False
    candidate_id = marker["defterm_input_id"]
    metadata = marker["runtime_metadata"]
    if rows and rows[-1]["defterm_input_id"] == candidate_id:
        require({field: rows[-1][field] for field in RUNTIME_METADATA_FIELDS} == metadata, "Committed result and marker disagree")
        clear_inflight_marker(candidate_id, paths)
        return rows, common, False
    require(len(rows) < len(sample), "Marker remains after the complete sample")
    source = sample[len(rows)]
    require(source["defterm_input_id"] == candidate_id, "Marker is not for the next fixed candidate")
    if common is not None:
        require(common == metadata, "Marker and partial-output runtimes differ")
    result = interrupted_result(source, metadata)
    validate_result_row(result, source, static, metadata, allow_self_test=allow_self_test)
    recovered = [*rows, result]
    write_bytes_atomically(paths.output, serialize_result_rows(recovered))
    clear_inflight_marker(candidate_id, paths)
    return recovered, metadata, True


def require_no_recorded_errors(rows: list[dict[str, str]]) -> None:
    require(
        all(row["generation_status"] == "OK" for row in rows),
        "Output contains a terminal ERROR; refusing another generation or retry",
    )


def assert_no_model_runtime() -> None:
    loaded = {name.partition(".")[0] for name in sys.modules} & FORBIDDEN_MODEL_MODULES
    require(not (loaded - INITIAL_MODEL_MODULES), "Local validation imported an ML runtime")
    require(not MODEL_LOADED and not INFERENCE_EXECUTED, "Local validation ran a model")


def print_input_audit(inputs: Inputs) -> None:
    counts = disjointness_counts(inputs.sample, inputs.examples)
    print("INPUT_VALIDATION=OK")
    print(f"INPUT_FILE={INPUT_PATH.name}")
    print(f"INPUT_SHA256={inputs.input_hash}")
    print(f"FEWSHOT_FILE={FEWSHOT_PATH.name}")
    print(f"FEWSHOT_SHA256={inputs.fewshot_hash}")
    print(f"TOTAL_SAMPLE={len(inputs.sample)}")
    print(f"TOTAL_FEWSHOT={len(inputs.examples)}")
    print("FEWSHOT_SUFFICIENT=6")
    print("FEWSHOT_INSUFFICIENT=4")
    for name, value in counts.items():
        print(f"EVALUATION_{name.upper()}_OVERLAP={value}")
    print("EXCLUDED_POOL_EXAMPLES_INCLUDED=0")
    print(f"POSITIVE_ANNOTATION_PROVENANCE={POSITIVE_PROVENANCE}")
    print(f"NEGATIVE_ANNOTATION_PROVENANCE={NEGATIVE_PROVENANCE}")
    for key, value in static_run_metadata(inputs).items():
        print(f"{key.upper()}={value}")


def print_completion(rows: list[dict[str, str]], sample_size: int) -> None:
    # Operational completeness only; no reference reading or evaluation metrics.
    print(f"TOTAL_RESULTS={len(rows)}")
    print(f"GENERATION_OK={sum(row['generation_status'] == 'OK' for row in rows)}")
    print(f"GENERATION_ERROR={sum(row['generation_status'] == 'ERROR' for row in rows)}")
    print(f"MODEL_LOADED={'YES' if MODEL_LOADED else 'NO'}")
    print(f"INFERENCE_EXECUTED={'YES' if INFERENCE_EXECUTED else 'NO'}")
    require(len(rows) == sample_size, "Not all 20 cases have durable results")
    require_no_recorded_errors(rows)


def run_experiment() -> None:
    inputs = load_inputs()
    static = static_run_metadata(inputs)
    examples = project_examples(inputs.examples)
    lock = acquire_run_lock()
    try:
        rows, common = load_existing_results(inputs.sample, static)
        rows, common, recovered = recover_interrupted_attempt(inputs.sample, rows, static, common)
        print_input_audit(inputs)
        print(f"ALREADY_PROCESSED={len(rows)}")
        print(f"PENDING={len(inputs.sample) - len(rows)}")
        print(f"INTERRUPTED_ATTEMPT_RECOVERED={bool_text(recovered)}")
        require_no_recorded_errors(rows)
        if len(rows) == len(inputs.sample):
            print_completion(rows, len(inputs.sample))
            return

        tokenizer, model, torch_module = load_model_and_tokenizer()
        metadata = collect_runtime_metadata(static, tokenizer, model, torch_module)
        if common is not None:
            require(common == metadata, "Active runtime differs from partial results")
        print(f"GENERATION_PARAMETERS={metadata['generation_parameters']}")
        print(f"RUNTIME_VERSIONS={metadata['runtime_versions']}")
        print(f"DEVICE_NAME={metadata['device_name']}")

        def generator(tok: Any, mdl: Any, evidence: str) -> str:
            return generate_once(tok, mdl, evidence, examples, torch_module)

        pending = inputs.sample[len(rows):]
        for source in pending:
            write_inflight_marker(source, metadata)
            # Any exception leaves the marker intact until a result is durable.
            result = process_candidate(source, tokenizer, model, metadata, generator)
            validate_result_row(result, source, static, metadata)
            rows = [*rows, result]
            write_bytes_atomically(OUTPUT_PATH, serialize_result_rows(rows))
            clear_inflight_marker(source["defterm_input_id"])
            require(result["generation_status"] == "OK", "ERROR row committed; this candidate will never be retried")
        verified, _ = load_existing_results(inputs.sample, static)
        print_completion(verified, len(inputs.sample))
    finally:
        release_run_lock(lock)


class FakeTokens:
    """CPU-only Python fixture; it is not a Torch tensor."""

    def __init__(self, values: Sequence[int]) -> None:
        self.values = list(values)
        self.shape = (1, len(values))

    def to(self, device: Any) -> FakeTokens:
        require(device == "SYNTHETIC_NO_DEVICE", "Fake tensor received a real device")
        return self

    def tolist(self) -> list[int]:
        return list(self.values)

    def __getitem__(self, key: tuple[int, slice]) -> FakeTokens:
        return FakeTokens(self.values[key[1]])


class FakeTokenizer:
    eos_token_id = 89
    pad_token_id = 88

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        require(kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}, "Chat-template configuration changed")
        self.messages = [dict(message) for message in messages]
        return "SYNTHETIC_RENDERED_PROMPT"

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, FakeTokens]:
        require(prompt == "SYNTHETIC_RENDERED_PROMPT" and kwargs == {"return_tensors": "pt"}, "Tokenization configuration changed")
        return {"input_ids": FakeTokens((10, 11, 12))}

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        require(kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}, "Decoding repaired or discarded output")
        mapping = {
            50: SUFFICIENT_EVIDENCE, 60: INSUFFICIENT_EVIDENCE,
            90: "<|EOS|>", 91: "<|EOS_ALT|>", 98: "<|PAD|>", 97: " ",
        }
        return "".join(mapping[token] for token in tokens)


class FakeModel:
    def __init__(self, content: Sequence[int] = (50, 90), *, fail: bool = False) -> None:
        self.content = list(content)
        self.fail = fail
        self.calls = 0
        self.kwargs: dict[str, Any] = {}
        self.generation_config = SimpleNamespace(eos_token_id=[90, 91], pad_token_id=98)

    def parameters(self) -> Any:
        return iter((SimpleNamespace(device="SYNTHETIC_NO_DEVICE"),))

    def generate(self, **kwargs: Any) -> FakeTokens:
        self.calls += 1
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("synthetic generation failure")
        return FakeTokens((10, 11, 12, *self.content))


def expect_validation_error(operation: Callable[[], Any]) -> None:
    try:
        operation()
    except ValidationError:
        return
    raise ValidationError("A deliberately invalid operation was accepted")


def synthetic_source() -> dict[str, str]:
    return {
        "defterm_input_id": "defterm_input_" + "a" * 64,
        "term": "FORBIDDEN_EVALUATION_TERM_SENTINEL", "qid": "Q999999999",
        "label_en": "FORBIDDEN_EVALUATION_LABEL_SENTINEL",
        "evidence_profile": "FORBIDDEN_EVALUATION_PROFILE_SENTINEL",
        "evidence_sparsity": "FORBIDDEN_EVALUATION_SPARSITY_SENTINEL",
        "defterm_evidence_text": "Label: Synthetic evidence-only candidate\nInstance of: synthetic class",
    }


def run_self_tests(inputs: Inputs) -> None:
    assert_no_model_runtime()
    tests: dict[str, str] = {}

    def passed(name: str, condition: bool) -> None:
        require(name in REQUIRED_SELF_TESTS, f"Unknown test: {name}")
        require(condition, f"{name} failed")
        tests[name] = "PASS"

    passed("INPUT_HASH_TEST", inputs.input_hash == EXPECTED_INPUT_SHA256 and inputs.fewshot_hash == EXPECTED_FEWSHOT_SHA256)
    passed("SAMPLE_SIZE_TEST", len(inputs.sample) == 20)
    passed("FEWSHOT_COUNT_TEST", len(inputs.examples) == 10)
    passed("FEWSHOT_CLASS_BALANCE_TEST", Counter(row["sufficiency_label"] for row in inputs.examples) == EXPECTED_CLASS_BALANCE)
    overlaps = disjointness_counts(inputs.sample, inputs.examples)
    passed("EVALUATION_DISJOINTNESS_TEST", overlaps["defterm_input_id"] == overlaps["term_qid"] == 0)
    passed("QID_DISJOINTNESS_TEST", overlaps["qid"] == 0)
    passed("EVIDENCE_DISJOINTNESS_TEST", overlaps["evidence"] == 0)
    # Exercise rejection as well as acceptance of the actual frozen examples.
    contaminated = [dict(row) for row in inputs.examples]
    for field in ("defterm_input_id", "term", "qid", "defterm_evidence_text"):
        contaminated[0][field] = inputs.sample[0][field]
    expect_validation_error(lambda: validate_disjointness(inputs.sample, contaminated))
    positives = [row for row in inputs.examples if row["sufficiency_label"] == SUFFICIENT_EVIDENCE]
    negatives = [row for row in inputs.examples if row["sufficiency_label"] == INSUFFICIENT_EVIDENCE]
    passed("POSITIVE_PROVENANCE_TEST", len(positives) == 6 and all(row["annotation_provenance"] == POSITIVE_PROVENANCE for row in positives))
    passed("NEGATIVE_PROVENANCE_TEST", len(negatives) == 4 and all(row["annotation_provenance"] == NEGATIVE_PROVENANCE for row in negatives))
    passed("EXCLUDED_POOL_EXAMPLES_TEST", not EXCLUDED_POOL_QIDS.intersection(row["qid"] for row in inputs.examples))

    examples = project_examples(inputs.examples)
    source = synthetic_source()
    messages = build_messages(source["defterm_evidence_text"], examples)
    rich_rows = [
        {
            **row, "term": "FORBIDDEN_EXAMPLE_TERM_SENTINEL", "qid": "Q888888888",
            "evidence_profile": "FORBIDDEN_EXAMPLE_PROFILE_SENTINEL",
            "evidence_sparsity": "FORBIDDEN_EXAMPLE_SPARSITY_SENTINEL",
            "annotation_notes": "FORBIDDEN_ANNOTATION_NOTES_SENTINEL",
            "annotation_provenance": "FORBIDDEN_ANNOTATION_PROVENANCE_SENTINEL",
            "manual_definition": "FORBIDDEN_MANUAL_DEFINITION_SENTINEL",
        }
        for row in inputs.examples
    ]
    serialized = canonical_json(messages)
    passed(
        "NO_MANUAL_DEFINITION_IN_PROMPT_TEST",
        "manual_definition" not in FEWSHOT_FIELDS
        and build_messages(source["defterm_evidence_text"], project_examples(rich_rows)) == messages
        and "FORBIDDEN_MANUAL_DEFINITION_SENTINEL" not in serialized,
    )
    forbidden = [
        value for field, value in source.items() if field != "defterm_evidence_text"
    ] + [value for field, value in rich_rows[0].items() if field in {
        "term", "qid", "evidence_profile", "evidence_sparsity", "annotation_notes", "annotation_provenance",
    }]
    passed(
        "PROMPT_ISOLATION_TEST",
        len(messages) == 22
        and [message["role"] for message in messages] == ["system", *(["user", "assistant"] * 10), "user"]
        and all(messages[2 * index + 2]["content"] == example.decision for index, example in enumerate(examples))
        and messages[-1]["content"] == USER_MSG_TEMPLATE.format(defterm_evidence_text=source["defterm_evidence_text"])
        and all(value not in serialized for value in forbidden)
        and QID_IN_TEXT_PATTERN.search(serialized) is None
        and all(row["annotation_notes"] not in serialized for row in inputs.examples),
    )
    for label in VALID_DECISIONS:
        require(parse_sufficiency_output(label) == (label, label), "Exact label rejected")
    for invalid in (
        "", " SUFFICIENT_EVIDENCE", "SUFFICIENT_EVIDENCE ", "SUFFICIENT_EVIDENCE\n",
        "INSUFFICIENT_EVIDENCE\n", "sufficient_evidence", "Decision: SUFFICIENT_EVIDENCE",
        "SUFFICIENT_EVIDENCE because the type is specific.",
        "INSUFFICIENT_EVIDENCE\nExplanation: broad type.",
        "SUFFICIENT_EVIDENCE\nINSUFFICIENT_EVIDENCE", "A synthetic definition.",
    ):
        expect_validation_error(lambda text=invalid: parse_sufficiency_output(text))
    passed("EXACT_DECISION_PARSE_TEST", True)
    input_names = {path.name for path in MODEL_INPUT_PATHS}
    passed("REFERENCE_EXCLUSION_TEST", REFERENCE_FILENAME not in input_names and input_names == {INPUT_PATH.name, FEWSHOT_PATH.name})
    passed("PREVIOUS_RESULTS_EXCLUSION_TEST", input_names.isdisjoint(PREVIOUS_RESULT_FILENAMES)
           and OUTPUT_PATH.name not in PREVIOUS_RESULT_FILENAMES)

    static = static_run_metadata(inputs, "SELF_TEST")
    metadata = synthetic_runtime_metadata(static)
    tokenizer = FakeTokenizer()
    model = FakeModel()
    fake_torch = SimpleNamespace(inference_mode=nullcontext)

    def generator(tok: Any, mdl: Any, evidence: str) -> str:
        return generate_once(tok, mdl, evidence, examples, fake_torch, real_inference=False)

    result = process_candidate(source, tokenizer, model, metadata, generator)
    passed(
        "SINGLE_GENERATION_POLICY_TEST",
        model.calls == 1 and result["generation_attempts"] == "1"
        and result["generation_status"] == "OK" and tokenizer.messages == messages
        and model.kwargs["max_new_tokens"] == 16
        and model.kwargs["do_sample"] is False
        and model.kwargs["num_beams"] == model.kwargs["num_return_sequences"] == 1
        and model.kwargs["eos_token_id"] == [90, 91]
        and model.kwargs["pad_token_id"] == 98,
    )
    failing_model = FakeModel(fail=True)
    failed = process_candidate(source, FakeTokenizer(), failing_model, metadata, generator)
    malformed = process_candidate(source, FakeTokenizer(), FakeModel((50, 98, 90)), metadata, generator)
    require(malformed["generation_raw"] == "SUFFICIENT_EVIDENCE<|PAD|>" and malformed["generation_status"] == "ERROR", "Special PAD token was hidden")
    require(remove_single_terminal_eos((50, 90, 90), (90, 91)) == [50, 90], "Multiple EOS tokens were hidden")
    require(failing_model.calls == 1 and failed["generation_call_status"] == "FAILED_OR_UNKNOWN", "Failure was retried or falsely marked returned")
    for row in (result, failed, malformed):
        validate_result_row(row, source, static, metadata, allow_self_test=True)

    # Test both crash windows against synthetic files, not the future output.
    temp_parent = Path(tempfile.gettempdir()) / "opencode"
    with tempfile.TemporaryDirectory(prefix="sufficiency-fewshot-self-test-", dir=temp_parent if temp_parent.is_dir() else None) as name:
        root = Path(name)
        paths = StatePaths(root / "synthetic_results.csv", root / "synthetic.lock", root / "synthetic.inflight.json")
        lock = acquire_run_lock(paths.lock)
        release_run_lock(lock)
        lock = acquire_run_lock(paths.lock)
        release_run_lock(lock)
        write_inflight_marker(source, metadata, paths)
        recovered, common, recovery_happened = recover_interrupted_attempt(
            [source], [], static, None, paths, allow_self_test=True,
        )
        loaded, _ = load_existing_results([source], static, paths, allow_self_test=True)
        expect_validation_error(lambda: require_no_recorded_errors(loaded))
        require(loaded == recovered and recovery_happened and not paths.inflight.exists(), "Interrupted attempt was not durably recovered")
        write_inflight_marker(source, metadata, paths)
        same, _, second_recovery = recover_interrupted_attempt(
            [source], recovered, static, common, paths, allow_self_test=True,
        )
        require(same == recovered and not second_recovery and not paths.inflight.exists(), "Post-commit marker caused a repeat")
        passed(
            "NO_RETRY_POLICY_TEST",
            failing_model.calls == 1 and failed["generation_status"] == "ERROR"
            and recovered[0]["generation_call_status"] == "INTERRUPTED_OR_UNKNOWN"
            and recovered[0]["sufficiency_decision"] == ""
            and "not retried" in recovered[0]["error_message"],
        )
    decoded = parse_csv_payload("synthetic.csv", serialize_result_rows([result]), RESULT_FIELDS)
    expect_validation_error(lambda: validate_result_row(result, source, static_run_metadata(inputs)))
    passed(
        "OUTPUT_SCHEMA_TEST",
        tuple(result) == RESULT_FIELDS and decoded == [result]
        and {"definition_raw", "definition_clean"}.isdisjoint(RESULT_FIELDS),
    )
    duplicate_result = process_candidate(source, FakeTokenizer(), FakeModel(), metadata, generator)
    passed(
        "DETERMINISM_TEST",
        build_messages(source["defterm_evidence_text"], examples) == messages
        and duplicate_result == result
        and serialize_result_rows([duplicate_result]) == serialize_result_rows([result])
        and DO_SAMPLE is False and NUM_BEAMS == NUM_RETURN_SEQUENCES == 1
        and INFERENCE_SEED == 42,
    )
    assert_no_model_runtime()
    require(set(tests) == set(REQUIRED_SELF_TESTS), "Not all required self-tests ran")
    print("SELF_TEST_STATUS=SUCCESS")
    for test in REQUIRED_SELF_TESTS:
        print(f"{test}={tests[test]}")
    print("SELF_TEST_SCOPE=LOCAL_CONTRACTS_AND_SYNTHETIC_RUNTIME_ONLY")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true", help="local tests without loading or downloading any model")
    modes.add_argument("--validate-inputs", action="store_true", help="validate frozen examples/sample and overlap without inference")
    modes.add_argument("--audit-example-sources", action="store_true", help="local-only audit against historical example sources and current inventory")
    args = parser.parse_args(argv)
    if args.self_test or args.validate_inputs or args.audit_example_sources:
        assert_no_model_runtime()
        inputs = load_inputs()
        if args.self_test:
            run_self_tests(inputs)
        elif args.audit_example_sources:
            audit_example_sources(inputs)
        assert_no_model_runtime()
        print_input_audit(inputs)
        print("MODEL_LOADED=NO")
        print("INFERENCE_EXECUTED=NO")
    else:
        run_experiment()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError) as exc:
        print("SUFFICIENCY_GATE_FEWSHOT_STATUS=FAILED", file=sys.stderr)
        print(f"INCONSISTENCY={exc}", file=sys.stderr)
        print(f"MODEL_LOADED={'YES' if MODEL_LOADED else 'NO'}", file=sys.stderr)
        print(f"INFERENCE_EXECUTED={'YES' if INFERENCE_EXECUTED else 'NO'}", file=sys.stderr)
        raise SystemExit(1) from exc
