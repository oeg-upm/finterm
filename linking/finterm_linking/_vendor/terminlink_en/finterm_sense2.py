"""Internal dynamic Sense-2 evidence/task preparation. Standard library only.

The historical entrypoints are immutable utility dependencies, never runners.
Document evidence travels in associations; only CandidateEvidence reaches Qwen.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any


BASE = Path(__file__).resolve().parent
SCHEMA = "finterm-sense2-v1"
STAGE02_SCHEMA = "terminlink-stage02-candidates-v1"
POLICIES = {"historical": "historical-priority-english-gt1-v1",
            "multilingual": "multilingual-all-nonenglish-v1"}
LEXICAL_POLICY = "historical-unicode-regex-gt1-v1"
INVENTORY_COLUMNS = ("term", "qid", "label_en", "aliases_en", "description_en",
                     "descriptions_other_languages", "p31", "p279", "p1343",
                     "search_queries", "search_strategies")
INPUT_FILES = ("02_wikidata_candidatos.csv", "02_candidatos_relaciones.csv",
               "02_terminos_estado.csv", "02_ocurrencias_estado.csv")
TERM_FIELDS = {"document_id", "term_id", "term", "language", "term_provenance_json",
               "occurrence_count", "candidate_count", "retrieval_status", "retrieval_complete",
               "absence_of_entity_proven", "error_details_json"}
OCC_FIELDS = {"document_id", "term_id", "occurrence_uid", "occurrence_id", "term", "context",
              "candidate_count", "retrieval_status", "absence_of_entity_proven"}
ASSOC_FIELDS = {"document_id", "term_id", "occurrence_uid", "occurrence_id", "term", "context",
                "candidate_id", "qid", "wikidata_url", "candidate_role", "validation_status",
                "retrieval_status", "search_queries", "search_strategies", "data_provenance"}
RETRIEVAL_STATES = {"CANDIDATES_AVAILABLE", "CANDIDATES_AVAILABLE_WITH_ERRORS", "NO_CANDIDATES",
                    "TERM_NOT_FOUND_IN_TEXT", "RETRIEVAL_ERROR"}
LEGACY_FILES = ("03_preparar_representaciones.py", "03_preparar_traducciones_contextualizadas.py",
                "qwen14_contextual_v2_runtime.py", "03_revalidar_traducciones_contextualizadas_qwen14_v2.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_hash(value: Any) -> str:
    return digest(canonical(value).encode("utf-8"))


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@lru_cache(maxsize=4)
def legacy(filename: str):
    """Load only reviewed, model-lazy utilities; no main(), frozen CSV or manifest."""
    require(filename in LEGACY_FILES, "Unreviewed historical utility")
    name = "_finterm_sense2_" + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    require(spec is not None and spec.loader is not None, "Historical utility is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by the historical frozen dataclasses.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def component_hashes() -> dict[str, str]:
    names = (*LEGACY_FILES, "finterm_sense2.py", "finterm_sense2_qwen.py", "finterm_sense2_store.py")
    return {name: file_hash(BASE / name) for name in names}


def candidate_id(row: dict[str, str]) -> str:
    # Exact existing stage-02 candidate identity; no document-dependent replacement.
    return "cand_" + object_hash([row["term"], row["qid"]])


def lexical_count(text: str) -> int:
    return legacy("03_preparar_representaciones.py").english_word_count(text)


def is_english(language: str) -> bool:
    return language.casefold().replace("_", "-").split("-", 1)[0] == "en"


def _csv(payload: bytes, name: str) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    columns = tuple(reader.fieldnames or ())
    require(bool(columns) and len(columns) == len(set(columns)), "Invalid CSV header: " + name)
    rows = tuple(dict(row) for row in reader)
    require(all(None not in row and all(isinstance(value, str) for value in row.values()) for row in rows),
            "Malformed CSV row: " + name)
    return columns, rows


def _safe_member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    require(name and not relative.is_absolute() and ".." not in relative.parts
            and "\\" not in name and ":" not in name, "Unsafe stage-02 artifact path")
    path = root.joinpath(*relative.parts).resolve()
    require(root in path.parents, "Stage-02 artifact escapes its directory")
    return path


@dataclass(frozen=True)
class Stage02Input:
    candidates: tuple[dict[str, str], ...]
    associations: tuple[dict[str, str], ...]
    terms: tuple[dict[str, str], ...]
    occurrences: tuple[dict[str, str], ...]
    provenance: dict[str, Any]
    content_sha256: str
    source_directory: Path | None = None

    @classmethod
    def from_records(cls, candidates, associations, terms, occurrences, *, provenance):
        """In-memory version of the same four CSV contracts; no invented provenance."""
        value = json.loads(canonical({"candidates": list(candidates), "associations": list(associations),
                                      "terms": list(terms), "occurrences": list(occurrences),
                                      "provenance": provenance}))
        require(isinstance(provenance, dict) and bool(provenance), "Explicit input provenance is required")
        source = cls(*(tuple(value[key]) for key in ("candidates", "associations", "terms", "occurrences")),
                     value["provenance"], object_hash(value))
        source.validate()
        return source

    def content(self):
        return {"candidates": self.candidates, "associations": self.associations, "terms": self.terms,
                "occurrences": self.occurrences, "provenance": self.provenance}

    def verify_files(self) -> None:
        if self.source_directory is not None:
            for name, checksum in self.provenance["files_sha256"].items():
                require(file_hash(_safe_member(self.source_directory, name)) == checksum,
                        "Stage-02 input changed: " + name)

    def validate(self) -> None:
        require(object_hash(self.content()) == self.content_sha256, "In-memory stage-02 input changed")
        for rows, fields in ((self.candidates, set(INVENTORY_COLUMNS)), (self.associations, ASSOC_FIELDS),
                             (self.terms, TERM_FIELDS), (self.occurrences, OCC_FIELDS)):
            for row in rows:
                require(fields <= set(row) and all(isinstance(value, str) for value in row.values()),
                        "Stage-02 record does not satisfy its string-column contract")
        pairs = {(row["term"], row["qid"]): row for row in self.candidates}
        require(len(pairs) == len(self.candidates), "Duplicate stage-02 term/QID pair")
        terms = {row["term_id"]: row for row in self.terms}
        occurrences = {row["occurrence_uid"]: row for row in self.occurrences}
        require(len(terms) == len(self.terms) and len(occurrences) == len(self.occurrences), "Duplicate source identifiers")
        for row in self.terms:
            require(row["term"] and row["document_id"] and row["term_id"] == "term_" + object_hash([row["document_id"], row["term"]]),
                    "Stage-02 document/term identity differs")
            require(row["retrieval_status"] in RETRIEVAL_STATES and row["absence_of_entity_proven"].lower() == "false",
                    "Unsupported retrieval state or asserted entity absence")
            require(row["retrieval_complete"].lower() in {"true", "false"}
                    and row["occurrence_count"].isdecimal() and row["candidate_count"].isdecimal(), "Invalid retrieval flags/counts")
            found, count = int(row["occurrence_count"]), int(row["candidate_count"])
            status = row["retrieval_status"]
            require((status == "TERM_NOT_FOUND_IN_TEXT") == (found == 0)
                    and (status in {"CANDIDATES_AVAILABLE", "CANDIDATES_AVAILABLE_WITH_ERRORS"}) == (count > 0),
                    "Retrieval status contradicts source coverage")
            require((row["retrieval_complete"].lower() == "true") == (status in {"CANDIDATES_AVAILABLE", "NO_CANDIDATES"}),
                    "Retrieval completeness contradicts source status")
        for row in self.occurrences:
            term = terms.get(row["term_id"])
            require(term is not None and all(row[key] == term[key] for key in ("document_id", "term", "retrieval_status")),
                    "Stage-02 occurrence/term reference differs")
            require(row["absence_of_entity_proven"].lower() == "false", "Occurrence asserts unsupported entity absence")
        seen = set()
        linked = set()
        by_occurrence = Counter()
        by_term = {}
        for row in self.associations:
            key = (row["term"], row["qid"])
            occurrence = occurrences.get(row["occurrence_uid"])
            require(key in pairs and occurrence is not None, "Association references an absent candidate or occurrence")
            require(all(row[field] == occurrence[field] for field in ("document_id", "term_id", "term", "occurrence_id", "context", "retrieval_status")),
                    "Association changed its original occurrence")
            require(row["candidate_id"] == candidate_id(row) and row["candidate_role"] == "RETRIEVED_CANDIDATE"
                    and row["validation_status"] == "NOT_VALIDATED_NOT_MANUALLY_VERIFIED", "Invalid candidate identity/role")
            require(all(row[field] == pairs[key][field] for field in ("search_queries", "search_strategies")), "Candidate search provenance differs")
            unit = (row["occurrence_uid"], row["qid"])
            require(unit not in seen, "Duplicate occurrence/QID association")
            seen.add(unit)
            linked.add(key)
            by_occurrence[row["occurrence_uid"]] += 1
            by_term.setdefault(row["term_id"], set()).add(row["qid"])
        require(linked == set(pairs), "Inventory has candidates without source associations")
        occurrence_counts = Counter(row["term_id"] for row in self.occurrences)
        for row in self.occurrences:
            require(str(by_occurrence[row["occurrence_uid"]]) == row["candidate_count"], "Occurrence candidate count differs")
        for row in self.terms:
            require(str(len(by_term.get(row["term_id"], ()))) == row["candidate_count"]
                    and str(occurrence_counts[row["term_id"]]) == row["occurrence_count"], "Term coverage/count differs")
        for row in self.candidates:
            require(set(row) == set(INVENTORY_COLUMNS) and re.fullmatch(r"Q[1-9][0-9]*", row["qid"]) is not None,
                    "Invalid inventory schema or QID")
            queries, strategies = (json.loads(row[field]) for field in ("search_queries", "search_strategies"))
            require(isinstance(queries, list) and isinstance(strategies, list) and len(queries) == len(strategies)
                    and all(isinstance(item, str) for item in queries + strategies), "Invalid search provenance")


def load_stage02(folder: Path | str) -> Stage02Input:
    """Read/verify an existing dynamic stage-02 run, including partial runs."""
    root = Path(folder).resolve()
    manifest_path = root / "02_manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    require(manifest.get("schema_version") == STAGE02_SCHEMA
            and manifest.get("status") in {"COMPLETED", "PARTIAL_OR_FAILED"}, "Unsupported stage-02 manifest")
    checksums = manifest.get("output_sha256", {})
    require(set(INPUT_FILES) <= set(checksums) and "02_manifest.json" not in checksums, "Stage-02 manifest omits required artifacts")
    files = {}
    payloads = {}
    for name, checksum in checksums.items():
        path = _safe_member(root, name)
        actual = file_hash(path)
        require(isinstance(checksum, str) and actual == checksum.lower(), "Stage-02 artifact hash differs: " + name)
        files[name] = actual
        if name in INPUT_FILES:
            payloads[name] = path.read_bytes()
            require(digest(payloads[name]) == actual, "Input changed while being read")
    files["02_manifest.json"] = digest(manifest_payload)
    tables = [_csv(payloads[name], name) for name in INPUT_FILES]
    require(tables[0][0] == INVENTORY_COLUMNS, "Stage-02 inventory must have its exact eleven columns")
    provenance = {"schema_version": STAGE02_SCHEMA, "status": manifest["status"], "files_sha256": files,
                  "data_provenance": manifest.get("data_provenance"), "source_directory": str(root),
                  "source_component_sha256": manifest.get("component_sha256", {})}
    value = Stage02Input.from_records(*(table[1] for table in tables), provenance=provenance)
    return Stage02Input(value.candidates, value.associations, value.terms, value.occurrences,
                        value.provenance, value.content_sha256, root)


@dataclass(frozen=True)
class Sense2Plan:
    source: Stage02Input
    payload: dict[str, Any]
    sha256: str

    def validate(self):
        self.source.validate()
        require(object_hash(self.payload) == self.sha256, "Prepared Sense-2 plan changed")

    @property
    def counts(self):
        return {name: len(self.payload[name]) for name in
                ("candidates", "evidence", "representations", "tasks", "associations", "terms", "occurrences")}


def _selection(requested, available: set[str], label: str):
    if requested is None:
        return None
    require(not isinstance(requested, (str, bytes)), label + " must be a sequence")
    selected = list(requested)
    require(selected and all(isinstance(item, str) and item for item in selected)
            and len(set(selected)) == len(selected), "Empty/duplicate/invalid " + label)
    require(set(selected) <= available, "Unknown " + label)
    return set(selected)


def prepare_sense2(source: Stage02Input, *, policy="historical", candidate_ids=None, terms=None,
                   generation_contract=None) -> Sense2Plan:
    """Pure planning: all original sources survive; policies only select tasks."""
    from finterm_sense2_qwen import generation_contract as default_contract, messages_for

    source.validate()
    require(policy in POLICIES, "Unknown Sense-2 policy")
    term_selection = _selection(terms, {row["term"] for row in source.terms}, "term filter")
    possible = [row for row in source.candidates if term_selection is None or row["term"] in term_selection]
    selected = _selection(candidate_ids, {candidate_id(row) for row in possible}, "candidate filter")
    contract = json.loads(canonical(generation_contract if generation_contract is not None else default_contract()))
    utility = legacy("03_preparar_traducciones_contextualizadas.py")
    evidence, representations, tasks, candidates = {}, {}, {}, []
    for index, row in enumerate(source.candidates, 1):
        cid = candidate_id(row)
        if (term_selection is not None and row["term"] not in term_selection) or (selected is not None and cid not in selected):
            continue
        key = (row["term"], row["qid"])
        descriptions = json.loads(row["descriptions_other_languages"])
        require(isinstance(descriptions, dict) and all(isinstance(lang, str) and lang == lang.strip() and lang
                and lang != "en" and isinstance(text, str) for lang, text in descriptions.items()), "Invalid multilingual descriptions")
        aliases = utility.parse_aliases(row["aliases_en"], key)
        relations = {field: utility.parse_relations(row[field], field, key) for field in ("p31", "p279", "p1343")}
        available = tuple(sorted((language, text) for language, text in descriptions.items() if text.strip()))
        candidate_evidence = utility.CandidateEvidence(row["qid"], row["label_en"], aliases, row["description_en"], available,
                                                       relations["p31"], relations["p279"], relations["p1343"], "DYNAMIC")
        data = {"qid": row["qid"], "label_en": row["label_en"], "aliases_en": list(aliases),
                "description_en": row["description_en"], "descriptions_other_languages": dict(sorted(descriptions.items())),
                **{field: utility.relation_dicts(value) for field, value in relations.items()}}
        eid = "evidence_" + object_hash(data)
        if eid not in evidence:
            evidence[eid] = {"evidence_id": eid, **data, "source_refs": []}
        reference = {"candidate_id": cid, "inventory_record_number": index,
                     "inventory_row_sha256": object_hash(row), "input_content_sha256": source.content_sha256}
        evidence[eid]["source_refs"].append(reference)
        ids = []
        originals = ([("en", row["description_en"])] if row["description_en"] != "" else []) + sorted(descriptions.items())
        for language, text in originals:
            english = is_english(language)
            source_id = "description_" + object_hash([row["qid"], language, text])
            rid = "s2_" + object_hash([eid, source_id, "original" if english else contract])
            ids.append(rid)
            if rid in representations:
                continue
            translated = not english and bool(text.strip()) and (policy == "multilingual" or lexical_count(row["description_en"]) <= 1)
            task_id = None
            if translated:
                siblings = utility.other_descriptions(candidate_evidence, language, text)
                rendered = utility.render_translation_evidence(candidate_evidence, language, text, siblings)
                request = {"schema_version": SCHEMA, "qid": row["qid"], "source_language": language,
                           "source_description": text, "context_policy_version": utility.CONTEXT_POLICY_VERSION,
                           "translation_evidence_text": rendered, "messages": messages_for(rendered),
                           "generation_contract": contract}
                task_id = "translation_" + object_hash(request)
                tasks[task_id] = {"task_id": task_id, "request": request}
            representations[rid] = {"representation_id": rid, "evidence_id": eid, "qid": row["qid"],
                                    "description_id": source_id, "source_language": language, "source_text": text,
                                    "origin": "WIKIDATA_DESCRIPTION_EN" if language == "en" else
                                              "WIKIDATA_ENGLISH_VARIANT" if english else "WIKIDATA_CONTEXTUAL_TRANSLATION_V2",
                                    "task_id": task_id, "is_original_english": english,
                                    "planned_status": "ORIGINAL_ENGLISH" if english else "EMPTY_SOURCE" if not text.strip() else
                                                      "PENDING" if translated else "NOT_SELECTED_BY_POLICY"}
        candidates.append({"candidate_id": cid, "term": row["term"], "qid": row["qid"], "evidence_id": eid,
                           "original": dict(row), "source_ref": reference, "representation_ids": ids})
    included = {row["candidate_id"] for row in candidates}
    payload = {"schema_version": SCHEMA, "policy": {"name": policy, "version": POLICIES[policy]},
               "lexical_policy_version": LEXICAL_POLICY, "generation_contract": contract,
               "selection": {"terms": sorted(term_selection) if term_selection is not None else None,
                             "candidate_ids": sorted(selected) if selected is not None else None,
                             "source_candidate_count": len(source.candidates), "selected_candidate_count": len(candidates)},
               "candidates": candidates, "evidence": list(evidence.values()), "representations": list(representations.values()),
               "tasks": list(tasks.values()),
               "associations": [dict(row) for row in source.associations if row["candidate_id"] in included],
               "terms": [dict(row) for row in source.terms if term_selection is None or row["term"] in term_selection],
               "occurrences": [dict(row) for row in source.occurrences if term_selection is None or row["term"] in term_selection]}
    return Sense2Plan(source, payload, object_hash(payload))
