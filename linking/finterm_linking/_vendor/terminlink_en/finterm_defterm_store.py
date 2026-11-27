"""Dynamic DefTerm from committed Sense-2 runs, with bounded two-stage execution.

No translation result or upstream file is changed. The cache uses immutable
per-request claims/results and the historical durable atomic writer, not a
repeated rewrite of a complete result prefix.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from finterm_sense2 import (BASE, SCHEMA as SENSE2_SCHEMA, _safe_member, _selection, canonical,
                               candidate_id, digest, file_hash, lexical_count, object_hash, require)
from finterm_sense2_store import (_task_view as translation_view, _write_json, _write_jsonl,
                                     consolidate as consolidate_text, utc_now, validate_record)
from finterm_defterm import (SCHEMA, PRODUCTION, CONTROLLED, HISTORICAL_FILES, FINAL_FILE, _project,
                                definition_request, gate_request, generation_failure, load_fewshots,
                                methods, process_response, structured_identity, validate_task)


CACHE_SCHEMA = "finterm-defterm-cache-v1"
SOURCE_TABLES = {"candidates": "03_candidates.jsonl", "evidence": "03_evidence.jsonl",
                 "representations": "03_representations.jsonl", "translations": "03_translations.jsonl",
                 "associations": "03_associations.jsonl", "terms": "03_terms.jsonl", "occurrences": "03_occurrences.jsonl"}
PLAN_REP_FIELDS = ("representation_id", "evidence_id", "qid", "description_id", "source_language", "source_text",
                   "origin", "task_id", "is_original_english", "planned_status")
PLAN_CAND_FIELDS = ("candidate_id", "term", "qid", "evidence_id", "original", "source_ref", "representation_ids")
PENDING = {"PENDING_BUDGET", "BLOCKED_ADAPTER", "INFLIGHT_OR_INTERRUPTED"}
TRANSLATION_ERRORS = {"GENERATION_ERROR", "INVALID_STRUCTURE", "INCOMPLETE_GENERATION"}
SELECTION_POLICY = "HISTORICAL_LEXICAL_GT1_FALLBACK_AFTER_TRANSLATIONS_V1"
PRESELECTION_POLICIES = {"conservative": "DIRECT_NAMES_AND_QUERY_TEXT_COVERAGE_V1",
                        "direct_only": "DIRECT_NAMES_WITHOUT_QUERY_SUPPRESSION_V1"}
NAME_NORMALIZATION = "NFKC_CASEFOLD_WHITESPACE_UNICODE_DASH_APOSTROPHE_V1"
VARIANT_STRATEGIES = {
    "original_entity_search": "original", "original_sparql_exact": "original",
    "lemma_entity_search": "lemma", "lemma_sparql_exact": "lemma",
    "orthographic_entity_search": "orthographic", "orthographic_sparql_exact": "orthographic",
    "acronym_expansion_entity_search": "acronym_expansion", "acronym_expansion_sparql_exact": "acronym_expansion",
}


@dataclass(frozen=True)
class Sense2Input:
    source_directory: Path
    manifest: dict
    tables: dict
    files_sha256: dict
    content_sha256: str

    def verify_files(self):
        for name, checksum in self.files_sha256.items():
            require(file_hash(_safe_member(self.source_directory, name)) == checksum, "Sense-2 input changed: " + name)

    def validate(self):
        require(object_hash({"manifest": self.manifest, "tables": self.tables}) == self.content_sha256, "In-memory Sense-2 input changed")


def load_sense2(folder):
    root = Path(folder).resolve()
    payload = (root / "03_manifest.json").read_bytes()
    manifest = json.loads(payload)
    require(manifest.get("schema_version") == SENSE2_SCHEMA and manifest.get("status") in
            {"PREPARED", "COMPLETED", "PARTIAL", "INTERRUPTED"}, "Unsupported Sense-2 manifest")
    hashes = manifest.get("output_sha256", {})
    require(set(SOURCE_TABLES.values()) | {"03_source.json"} <= set(hashes) and "03_manifest.json" not in hashes,
            "Sense-2 manifest omits required contracts")
    fingerprints, tables = {}, {}
    for name, checksum in hashes.items():
        path = _safe_member(root, name)
        actual = file_hash(path)
        require(isinstance(checksum, str) and actual == checksum.lower(), "Sense-2 artifact hash differs: " + name)
        fingerprints[name] = actual
        if name in SOURCE_TABLES.values():
            raw = path.read_bytes()
            require(digest(raw) == actual, "Sense-2 input changed while reading")
            key = next(key for key, filename in SOURCE_TABLES.items() if filename == name)
            tables[key] = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
    fingerprints["03_manifest.json"] = digest(payload)
    _validate_sense2(tables, manifest)
    value = {"manifest": manifest, "tables": tables}
    return Sense2Input(root, manifest, tables, fingerprints, object_hash(value))


def _index(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), "Duplicate source identifier: " + key)
    return result


def _validate_sense2(tables, manifest):
    candidates = _index(tables["candidates"], "candidate_id")
    evidence = _index(tables["evidence"], "evidence_id")
    reps = _index(tables["representations"], "representation_id")
    tasks = _index(tables["translations"], "task_id")
    for name, rows in tables.items():
        count_name = "tasks" if name == "translations" else name
        require(manifest["counts"][count_name] == len(rows), "Sense-2 manifest count differs: " + name)
    for row in evidence.values():
        data = {key: value for key, value in row.items() if key not in {"evidence_id", "source_refs"}}
        require(row["evidence_id"] == "evidence_" + object_hash(data), "Source evidence identity differs")
    verified_tasks = []
    for row in tasks.values():
        task = {"task_id": row["task_id"], "request": row["request"]}
        require(task["task_id"] == "translation_" + object_hash(task["request"]), "Translation request identity differs")
        if row["generation"] is not None:
            require(row["generation_sha256"] == object_hash(row["generation"]), "Source translation generation hash differs")
            validate_record(row["generation"], task)
            entry = {key: row[key] for key in ("state", "generation", "generation_sha256", "claimed_utc", "finished_utc")}
            require(entry["state"] == "DONE", "Source result is not committed")
        elif row["processing_status"] == "INFLIGHT_OR_INTERRUPTED":
            entry = {"state": "INFLIGHT", "generation": None, "generation_sha256": None,
                     "claimed_utc": row.get("claimed_utc"), "finished_utc": None}
        else:
            require(row["processing_status"] in {"PENDING_BUDGET", "BLOCKED_ADAPTER"}, "Unknown unresolved translation status")
            entry = None
        expected = translation_view(task, entry, blocked=row["processing_status"] == "BLOCKED_ADAPTER")
        require(all(row.get(key) == value for key, value in expected.items()), "Source translation validation/state differs")
        verified_tasks.append(expected)
    for row in candidates.values():
        require(row["candidate_id"] == candidate_id(row) and re.fullmatch(r"Q[1-9][0-9]*", row["qid"]), "Invalid source candidate identity")
        data = evidence.get(row["evidence_id"])
        require(data is not None and data["qid"] == row["qid"], "Missing/mismatched candidate evidence")
        original = row["original"]
        require(all(original[key] == row[key] for key in ("term", "qid")), "Original candidate identity differs")
        for key in ("label_en", "description_en", "aliases_en", "descriptions_other_languages", "p31", "p279", "p1343"):
            value = original[key] if key in {"label_en", "description_en"} else json.loads(original[key])
            require(value == data[key], "Candidate structured evidence differs from original")
        for rid in row["representation_ids"]:
            require(rid in reps and reps[rid]["evidence_id"] == row["evidence_id"] and reps[rid]["qid"] == row["qid"], "Foreign/missing source representation")
    for row in reps.values():
        require(row["evidence_id"] in evidence and isinstance(row["source_text"], str), "Invalid source representation")
        if row["task_id"] is not None:
            require(row["task_id"] in tasks, "Representation refers to a missing translation")
            request = tasks[row["task_id"]]["request"]
            require(all(request[key] == row[field] for key, field in
                        (("qid", "qid"), ("source_language", "source_language"), ("source_description", "source_text"))), "Representation translation target differs")
    # Reuse the current Sense-2 consolidator to check summaries and availability;
    # never infer permanent absence from a mutable summary field alone.
    plan = SimpleNamespace(payload={"evidence": tables["evidence"],
        "candidates": [{key: row[key] for key in PLAN_CAND_FIELDS} for row in tables["candidates"]],
        "representations": [{key: row[key] for key in PLAN_REP_FIELDS} for row in tables["representations"]]})
    expected_candidates, expected_reps = consolidate_text(plan, verified_tasks)
    require(expected_candidates == tables["candidates"] and expected_reps == tables["representations"], "Sense-2 consolidation changed or is inconsistent")
    for row in tables["associations"]:
        require(row["candidate_id"] in candidates and row["qid"] == candidates[row["candidate_id"]]["qid"], "Foreign source association")


def _normalized_name(text):
    require(isinstance(text, str), "Candidate names/recorded variants must be strings")
    # No stemming, singularization, token overlap or removal of qualifiers.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.translate(str.maketrans({character: "-" for character in "\u2010\u2011\u2012\u2013\u2014\u2212"}
                                      | {character: "'" for character in "\u2018\u2019\u02bc"}))
    return " ".join(normalized.split())


def _usable_text(candidate, reps):
    return any(reps[rid]["available_for_later_evaluation"] and reps[rid]["lexical_word_count"] > 1
               for rid in candidate["representation_ids"])


def _preselection(source, reps, policy):
    """Name evidence only, using existing phase-02 variants; never a link decision."""
    require(policy in PRESELECTION_POLICIES, "Unknown DefTerm preselection policy")
    groups = {}
    for candidate in source.tables["candidates"]:
        term = candidate["term"]  # Keep retrieval cohorts separate, including across different supplied labels.
        group = groups.setdefault(term, {"variants": {}, "ignored_strategies": Counter(), "anchors": []})
        group["variants"].setdefault(("original", term), {"text": term, "normalized": _normalized_name(term),
            "kind": "original", "provenance": [{"source": "supplied_term"}]})
        original = candidate["original"]
        queries, strategies = json.loads(original["search_queries"]), json.loads(original["search_strategies"])
        require(isinstance(queries, list) and isinstance(strategies, list) and len(queries) == len(strategies)
                and all(isinstance(value, str) and value.strip() for value in queries + strategies), "Invalid phase-02 variant provenance")
        for query, strategy in zip(queries, strategies):
            kind = VARIANT_STRATEGIES.get(strategy)
            if kind is None:
                group["ignored_strategies"][strategy] += 1
                continue  # Context extensions/Cirrus expressions are not term variants.
            require(kind != "original" or query == term, "Original-query provenance differs from supplied term")
            variant = group["variants"].setdefault((kind, query), {"text": query, "normalized": _normalized_name(query),
                                                                  "kind": kind, "provenance": []})
            variant["provenance"].append({"source": "stage02_search_queries", "candidate_id": candidate["candidate_id"],
                                           "query": query, "strategy": strategy})
    records = {}
    for candidate in source.tables["candidates"]:
        group = groups[candidate["term"]]
        matches = []
        original = candidate["original"]
        names = [("label_en", original["label_en"]), *(("aliases_en", alias) for alias in json.loads(original["aliases_en"]))]
        for field, value in names:
            normalized = _normalized_name(value)
            if normalized:
                matches.extend({"field": field, "value": value, "normalized": normalized, "variant": variant["text"],
                                "variant_kind": variant["kind"]} for variant in group["variants"].values()
                               if normalized == variant["normalized"])
        own_text = _usable_text(candidate, reps)
        records[candidate["candidate_id"]] = {"policy": policy, "policy_version": PRESELECTION_POLICIES[policy],
            "normalization": NAME_NORMALIZATION, "query_term": candidate["term"], "direct_name_match": bool(matches),
            "name_matches": matches, "own_usable_text": own_text, "variants": list(group["variants"].values()),
            "lemma_recorded": any(variant["kind"] == "lemma" for variant in group["variants"].values()),
            "ignored_query_strategies": dict(group["ignored_strategies"]), "semantic_identity_verified": False}
        if matches and own_text:
            group["anchors"].append({"candidate_id": candidate["candidate_id"], "qid": candidate["qid"]})
    for record in records.values():
        anchors = groups[record["query_term"]]["anchors"]
        record["query_textual_anchor_candidate_ids"] = [row["candidate_id"] for row in anchors]
        record["query_textual_anchor_qids"] = list(dict.fromkeys(row["qid"] for row in anchors))
        record["secondary_suppressed"] = policy == "conservative" and bool(anchors) and not record["own_usable_text"]
        record["reason"] = ("OWN_USABLE_TEXT" if record["own_usable_text"] else "QUERY_HAS_DIRECT_TEXT" if record["secondary_suppressed"]
                            else "DIRECT_NAME_MATCH" if record["direct_name_match"] else "RELEVANCE_NOT_ESTABLISHED")
    return records


def prepare_defterm(source, *, examples=None, candidate_ids=None, backend_kind=PRODUCTION, preselection_policy="conservative"):
    source = source if isinstance(source, Sense2Input) else load_sense2(source)
    source.validate()
    examples = examples or load_fewshots()
    examples.validate()
    selected = _selection(candidate_ids, {row["candidate_id"] for row in source.tables["candidates"]}, "DefTerm candidate filter")
    reps = _index(source.tables["representations"], "representation_id")
    preselection = _preselection(source, reps, preselection_policy)  # Entire input, before the pilot candidate filter.
    targets = []
    m = methods()
    for candidate in source.tables["candidates"]:
        cid = candidate["candidate_id"]
        rows = [reps[rid] for rid in candidate["representation_ids"]]
        translations = Counter(row["processing_status"] for row in rows if not row["is_original_english"])
        state = "ELIGIBLE"
        if selected is not None and cid not in selected:
            state = "NOT_SELECTED"
        elif _usable_text(candidate, reps):
            state = "TEXTUAL_REPRESENTATION_AVAILABLE"
        elif any(translations[key] for key in PENDING):
            state = "DEFERRED_TRANSLATION_PENDING"
        elif any(translations[key] for key in TRANSLATION_ERRORS):
            state = "DEFERRED_TRANSLATION_ERROR"
        elif translations["NOT_SELECTED_BY_POLICY"]:
            state = "DEFERRED_TRANSLATION_NOT_SELECTED"
        elif any(row["retrieval_status"] in {"RETRIEVAL_ERROR", "CANDIDATES_AVAILABLE_WITH_ERRORS"}
                 for row in source.tables["associations"] if row["candidate_id"] == cid):
            state = "DEFERRED_WIKIDATA_INCOMPLETE"
        original = candidate["original"]
        core = any(json.loads(original[key]) for key in ("aliases_en", "p31", "p279"))
        prepared, task = None, None
        if state == "ELIGIBLE" and not core:
            state = "NO_CORE_STRUCTURED_EVIDENCE"
        if state == "ELIGIBLE":
            pending = {key: original[key] for key in ("term", "qid", "label_en", "aliases_en", "p31", "p279", "p1343")}
            pending.update(sense2_status="DEFTERM_PENDING", representation_route="OTHER_LANGUAGE_DESCRIPTION" if
                           json.loads(original["descriptions_other_languages"]) else "STRUCTURED_EVIDENCE")
            try:
                inventory, _ = m.prep.prepare_inventory([pending], "dynamic_sense2", expected_count=1)
            except m.prep.ValidationError:
                state = "STRUCTURED_EVIDENCE_NOT_RENDERABLE"
            else:
                prepared = inventory[0]
                if preselection[cid]["secondary_suppressed"]:
                    state = "DEFERRED_QUERY_HAS_DIRECT_TEXT"
                elif not preselection[cid]["direct_name_match"]:
                    state = "RELEVANCE_NOT_ESTABLISHED"
                else:
                    task = gate_request(prepared, examples, backend_kind=backend_kind)
        targets.append({"candidate_id": cid, "eligibility": state, "translation_states": dict(translations),
                        "prepared": prepared, "gate_task": task, "preselection": preselection[cid]})
    return {"source": source, "examples": examples, "targets": targets,
            "selection": {"candidate_ids": sorted(selected) if selected is not None else None, "policy": SELECTION_POLICY,
                          "preselection_policy": preselection_policy, "preselection_version": PRESELECTION_POLICIES[preselection_policy],
                          "name_normalization": NAME_NORMALIZATION},
            "counts": dict(Counter(row["eligibility"] for row in targets))}


class DefTermCache:
    """One immutable request claim and one atomically committed result per task."""
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        require(self.folder.parent.is_dir(), "Cache parent must exist")
        self.folder.mkdir(exist_ok=True)
        schema = self.folder / "schema.json"
        expected = {"schema_version": CACHE_SCHEMA}
        if not schema.exists():
            require(not any(self.folder.iterdir()), "Refusing an unrecognized nonempty cache directory")
            _write_json(schema, expected)
        require(json.loads(schema.read_text(encoding="utf-8")) == expected, "Unsupported DefTerm cache")

    def _folder(self, task):
        validate_task(task)
        return self.folder / task["task_id"]

    def lookup(self, task):
        folder = self._folder(task)
        claim_path, result_path = folder / "claim.json", folder / "result.json"
        if not claim_path.exists():
            require(not result_path.exists(), "Cached result has no committed request claim")
            return None
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        require(claim["task_id"] == task["task_id"] and claim["request"] == task["request"]
                and claim["request_sha256"] == object_hash(task["request"]) and claim["claim_id"], "Cached DefTerm request hash differs")
        if not result_path.exists():
            return {"state": "INFLIGHT_OR_INTERRUPTED", "response": None, "processed": None, "result_sha256": None}
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        record = payload["record"]
        require(payload["record_sha256"] == object_hash(record) and record["task_id"] == task["task_id"]
                and record["request_sha256"] == claim["request_sha256"], "Cached DefTerm result hash differs")
        require(record["processed"] == process_response(task, record["response"]), "Cached DefTerm validation differs")
        return {"state": "DONE", "response": record["response"], "processed": record["processed"],
                "result_sha256": payload["record_sha256"]}

    def claim(self, task):
        folder = self._folder(task)
        folder.mkdir(exist_ok=True)
        if self.lookup(task) is not None:
            return None
        owner = uuid.uuid4().hex
        try:
            _write_json(folder / "claim.json", {"task_id": task["task_id"], "request": task["request"],
                "request_sha256": object_hash(task["request"]), "claim_id": owner, "claimed_utc": utc_now()})
        except FileExistsError:
            return None
        return owner

    def finish(self, task, owner, response):
        folder = self._folder(task)
        existing = self.lookup(task)
        require(existing is not None and existing["state"] == "INFLIGHT_OR_INTERRUPTED", "DefTerm claim is not unfinished")
        claim = json.loads((folder / "claim.json").read_text(encoding="utf-8"))
        require(claim["claim_id"] == owner, "DefTerm claim belongs to another writer")
        record = {"task_id": task["task_id"], "request_sha256": object_hash(task["request"]), "response": response,
                  "processed": process_response(task, response), "finished_utc": utc_now()}
        destination = folder / "result.json"
        require(not destination.exists(), "Refusing to overwrite a committed DefTerm result")
        methods().gate.write_bytes_atomically(destination, (canonical({"record": record, "record_sha256": object_hash(record)}) + "\n").encode("utf-8"))


def gate_certificate(task, entry):
    require(entry is not None and entry["state"] == "DONE"
            and entry["processed"]["accepted_decision"] == methods().gate.SUFFICIENT_EVIDENCE, "No verified sufficient gate result")
    data = {"gate_task_id": task["task_id"], "gate_result_id": "gate_result_" + entry["result_sha256"],
            "gate_result_sha256": entry["result_sha256"], "defterm_evidence_id": task["request"]["defterm_evidence_id"],
            "decision": entry["processed"]["accepted_decision"]}
    return {**data, "certificate_id": "post_gate_" + object_hash(data)}


def _execute(task, cache, backend, budget, execution, stage, proof_check=None):
    entry = cache.lookup(task)
    if entry is not None:
        execution[stage]["cache_hits" if entry["state"] == "DONE" else "inflight_not_retried"] += 1
        return entry
    if execution[stage]["new_attempts"] >= budget or execution["adapter_error"] is not None:
        return None
    try:
        backend.prepare()
    except Exception:
        execution["adapter_error"] = "DEFTERM_LOCAL_ADAPTER_UNAVAILABLE"
        return None
    if proof_check is not None:
        proof_check()
    owner = cache.claim(task)
    if owner is not None:
        execution[stage]["new_attempts"] += 1
        if proof_check is not None:
            proof_check()  # Last-mile certificate, immediately before generation.
        try:
            response = backend.generate(task)
        except Exception:
            response = generation_failure(task)
        cache.finish(task, owner, response)
    return cache.lookup(task)


def _view(task, entry, adapter_error=False):
    return {**task, **(entry or {"state": "BLOCKED_ADAPTER" if adapter_error else "PENDING_BUDGET", "response": None,
                               "processed": None, "result_sha256": None})}


def _components():
    names = (*HISTORICAL_FILES, "finterm_defterm.py", "finterm_defterm_store.py",
             "finterm_sense2.py", "finterm_sense2_qwen.py", "finterm_sense2_store.py")
    return {name: file_hash(BASE / name) for name in names}


def run_defterm(sense2_dir, output_dir, cache_dir, *, candidate_ids=None, max_gate_calls=0, max_definition_calls=0,
                backend=None, gate_examples=None, definition_examples=None, preselection_policy="conservative"):
    """Internal composition for the future linker. Default budgets: no inference."""
    require(type(max_gate_calls) is int and max_gate_calls >= 0 and type(max_definition_calls) is int and max_definition_calls >= 0,
            "DefTerm budgets must be finite nonnegative integers")
    require(backend is not None or max_gate_calls == max_definition_calls == 0, "Explicit backend required for inference")
    kind = backend.backend_kind if backend is not None else PRODUCTION
    require(kind in {PRODUCTION, CONTROLLED}, "Unsupported DefTerm backend namespace")
    source = sense2_dir if isinstance(sense2_dir, Sense2Input) else load_sense2(sense2_dir)
    source.verify_files()
    examples = load_fewshots(gate_path=gate_examples, definition_path=definition_examples)
    plan = prepare_defterm(source, examples=examples, candidate_ids=candidate_ids, backend_kind=kind,
                           preselection_policy=preselection_policy)
    output, cache_dir = Path(output_dir).resolve(), Path(cache_dir).resolve()
    require(output.parent.is_dir() and cache_dir.parent.is_dir(), "Run/cache parents must exist")
    require(output != BASE and cache_dir != BASE and output != cache_dir and output not in cache_dir.parents and cache_dir not in output.parents,
            "Use independent new output/cache directories")
    require(source.source_directory not in output.parents and output != source.source_directory
            and source.source_directory not in cache_dir.parents and cache_dir != source.source_directory
            and output not in source.source_directory.parents and cache_dir not in source.source_directory.parents,
            "Do not write into Sense-2 inputs")
    upstream_cache = source.manifest.get("cache", {}).get("path")
    require(upstream_cache is None or cache_dir != Path(upstream_cache).resolve(), "Do not reuse or modify the translation cache")
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), "DefTerm output must be new or empty")
    output.mkdir(exist_ok=True)
    _write_json(output / ".defterm_in_progress.json", {"input_sha256": source.content_sha256, "started_utc": utc_now()})
    before = _components()
    cache = DefTermCache(cache_dir)
    execution = {stage: {"new_attempts": 0, "cache_hits": 0, "inflight_not_retried": 0} for stage in ("gate", "definition")}
    execution["adapter_error"] = None
    gate_tasks = {row["gate_task"]["task_id"]: row["gate_task"] for row in plan["targets"] if row["gate_task"] is not None}
    definition_tasks, target_tasks, interruption = {}, {}, None
    try:
        for task in gate_tasks.values():
            _execute(task, cache, backend, max_gate_calls, execution, "gate")
        for target in plan["targets"]:
            gate_task = target["gate_task"]
            if gate_task is None:
                continue
            entry = cache.lookup(gate_task)
            if entry is None or entry["state"] != "DONE" or entry["processed"]["accepted_decision"] != methods().gate.SUFFICIENT_EVIDENCE:
                continue
            certificate = gate_certificate(gate_task, entry)
            task = definition_request(target["prepared"], examples, certificate, backend_kind=kind)
            target_tasks[target["candidate_id"]] = task["task_id"]
            if task["task_id"] not in definition_tasks:
                definition_tasks[task["task_id"]] = task
                def proof_check(gt=gate_task, expected=certificate):
                    require(gate_certificate(gt, cache.lookup(gt)) == expected, "Gate certificate changed before definition")
                _execute(task, cache, backend, max_definition_calls, execution, "definition", proof_check)
    except BaseException as exc:
        if isinstance(exc, Exception):
            raise
        interruption = exc
    gate_views = [_view(task, cache.lookup(task), execution["adapter_error"]) for task in gate_tasks.values()]
    definition_views = [_view(task, cache.lookup(task), execution["adapter_error"]) for task in definition_tasks.values()]
    source.validate()
    source.verify_files()
    examples.validate()
    examples.verify_files()
    require(_components() == before, "Code changed during DefTerm execution")
    hashes = {"05_defterm_gate.jsonl": _write_jsonl(output / "05_defterm_gate.jsonl", gate_views),
              "05_defterm_definitions.jsonl": _write_jsonl(output / "05_defterm_definitions.jsonl", definition_views)}
    partitions = []
    for target in plan["targets"]:
        task = target["gate_task"]
        entry = cache.lookup(task) if task is not None else None
        decision = entry["processed"]["accepted_decision"] if entry is not None and entry["state"] == "DONE" else None
        partitions.append({"candidate_id": target["candidate_id"], "eligibility": target["eligibility"],
                           "preselection": target["preselection"],
                           "gate_task_id": task["task_id"] if task else None, "sufficiency_decision": decision,
                           "partition": "SUFFICIENT" if decision == methods().gate.SUFFICIENT_EVIDENCE else
                                        "INSUFFICIENT" if decision == methods().gate.INSUFFICIENT_EVIDENCE else "UNCLASSIFIED",
                           "certificate": gate_certificate(task, entry) if decision == methods().gate.SUFFICIENT_EVIDENCE else None})
    hashes["05_defterm_partition.jsonl"] = _write_jsonl(output / "05_defterm_partition.jsonl", partitions)
    final = _consolidate(source, plan, gate_views, definition_views, target_tasks, hashes)
    for name, rows in final.items():
        hashes[name] = _write_jsonl(output / name, rows)
    _write_json(output / "05_defterm_source.json", {"source_directory": str(source.source_directory), "files_sha256": source.files_sha256,
                "input_content_sha256": source.content_sha256, "fewshots": examples.provenance})
    hashes["05_defterm_source.json"] = file_hash(output / "05_defterm_source.json")
    states = Counter(row["defterm"]["state"] for row in final["05_defterm_candidates.jsonl"])
    terminal = {"TEXTUAL_REPRESENTATION_AVAILABLE", "NO_CORE_STRUCTURED_EVIDENCE", "NOT_SELECTED",
                "DEFTERM_AVAILABLE_UNVERIFIED", "GATE_INSUFFICIENT_EVIDENCE", "DEFERRED_QUERY_HAS_DIRECT_TEXT", "RELEVANCE_NOT_ESTABLISHED"}
    unfinished = set(states) - terminal
    prepared_only = max_gate_calls == max_definition_calls == 0 and unfinished <= {"GATE_PENDING_BUDGET", "DEFINITION_PENDING_BUDGET"}
    status = "INTERRUPTED" if interruption else "PREPARED" if unfinished and prepared_only else "PARTIAL" if unfinished else "COMPLETED"
    manifest = {"schema_version": SCHEMA, "status": status,
                "execution_scope": "DEFTERM_ENRICHMENT_ONLY_NO_LINK_DECISION", "created_utc": utc_now(),
                "source_schema_version": SENSE2_SCHEMA, "source_status": source.manifest["status"], "source_content_sha256": source.content_sha256,
                "selection": plan["selection"], "eligibility": plan["counts"], "counts": {"candidates": len(plan["targets"]),
                  "gate_tasks": len(gate_tasks), "definition_tasks": len(definition_tasks), "states": dict(states),
                  "source_representations": len(source.tables["representations"]), "final_representations": len(final["05_defterm_representations.jsonl"])},
                "execution": {**execution, "backend_kind": kind, "max_gate_calls": max_gate_calls, "max_definition_calls": max_definition_calls},
                "component_sha256": before, "output_sha256": hashes, "fidelity_verified": False}
    _write_json(output / "05_defterm_manifest.json", manifest)
    (output / ".defterm_in_progress.json").unlink()
    if interruption:
        raise interruption
    return manifest


def _consolidate(source, plan, gate_views, definition_views, target_tasks, hashes):
    expected = {**source.files_sha256, **hashes}
    historical = _project(FINAL_FILE, {"canonical_json", "sha256_bytes", "row_hash", "lineage", "defterm_representation_id", "DEFTERM_ROUTE"},
        {"EXPECTED_INPUT_HASHES": expected, "SCHEMA_VERSION": SCHEMA, "DEFTERM_FILE": "05_defterm_definitions.jsonl"})
    gates = {row["task_id"]: row for row in gate_views}
    definitions = {row["task_id"]: (index, row) for index, row in enumerate(definition_views, 1)}
    partitions_sha = hashes["05_defterm_partition.jsonl"]
    representations = list(source.tables["representations"])
    lineage = [{"representation_id": row["representation_id"], **historical.lineage("03_representations.jsonl", index, row)}
               for index, row in enumerate(representations, 1)]
    candidates = []
    for candidate, target in zip(source.tables["candidates"], plan["targets"]):
        state = target["eligibility"]
        gate_view = gates.get(target["gate_task"]["task_id"]) if target["gate_task"] else None
        definition = definitions.get(target_tasks.get(candidate["candidate_id"]))
        decision = None
        new_id = None
        if gate_view is not None:
            state = "GATE_" + gate_view["state"]
            if gate_view["state"] == "DONE":
                decision = gate_view["processed"]["accepted_decision"]
                state = ("GATE_INSUFFICIENT_EVIDENCE" if decision == methods().gate.INSUFFICIENT_EVIDENCE else
                         "DEFINITION_PENDING_BUDGET" if decision else "GATE_INCOMPLETE_GENERATION" if
                         gate_view["processed"]["generation_status"] == "OK" and gate_view["processed"]["completion_status"] != "COMPLETE" else "GATE_ERROR")
        if definition is not None:
            index, row = definition
            state = "DEFINITION_" + row["state"]
            if row["state"] == "DONE":
                processed = row["processed"]
                state = ("DEFTERM_AVAILABLE_UNVERIFIED" if processed["available_for_later_evaluation"] else
                         "DEFINITION_INCOMPLETE_GENERATION" if processed["generation_status"] == "OK" and processed["completion_status"] != "COMPLETE" else
                         "DEFINITION_CIRCULARITY_REVIEW" if processed["circularity_flag_final"] == "true" else "DEFINITION_ERROR")
                identity = {**processed, "term": candidate["term"], "defterm_input_id": target["prepared"]["defterm_input_id"],
                            "post_gate_manifest_sha256": partitions_sha}
                new_id = historical.defterm_representation_id(identity)
                representation = {"representation_id": new_id, "evidence_id": candidate["evidence_id"], "qid": candidate["qid"],
                    "defterm_input_id": target["prepared"]["defterm_input_id"], "defterm_evidence_id": processed["defterm_evidence_id"],
                    "candidate_id": candidate["candidate_id"], "origin": historical.DEFTERM_ROUTE, "source_language": "en",
                    "source_text": target["prepared"]["defterm_evidence_text"], "text_en": processed["definition_clean"],
                    "task_id": row["task_id"], "processing_status": state, "fidelity_status": "NOT_VERIFIED",
                    "evidence_sufficiency": decision, "available_for_later_evaluation": processed["available_for_later_evaluation"],
                    "circularity_flag_final": processed["circularity_flag_final"], "result_sha256": row["result_sha256"],
                    "gate_certificate": row["request"]["gate_certificate"], "lexical_word_count": lexical_count(processed["definition_clean"])}
                # Even unsuccessful/circular outputs are retained explicitly.
                representations.append(representation)
                lineage.append({"representation_id": new_id, **historical.lineage("05_defterm_definitions.jsonl", index, row)})
        candidates.append({**candidate, "source_candidate": candidate, "defterm": {"state": state, "eligibility": target["eligibility"],
                           "preselection": target["preselection"],
                           "defterm_input_id": target["prepared"]["defterm_input_id"] if target["prepared"] else None,
                           "defterm_evidence_id": structured_identity(target["prepared"]) if target["prepared"] else None,
                           "evidence_profile": target["prepared"]["evidence_profile"] if target["prepared"] else None,
                           "evidence_sparsity": target["prepared"]["evidence_sparsity"] if target["prepared"] else None,
                           "translation_states": target["translation_states"], "sufficiency_decision": decision,
                           "gate_task_id": gate_view["task_id"] if gate_view else None,
                           "definition_task_id": definition[1]["task_id"] if definition else None},
                           "all_representation_ids": [*candidate["representation_ids"], *([new_id] if new_id else [])]})
    return {"05_defterm_candidates.jsonl": candidates, "05_defterm_representations.jsonl": representations,
            "05_defterm_lineage.jsonl": lineage, "05_defterm_associations.jsonl": source.tables["associations"],
            "05_defterm_terms.jsonl": source.tables["terms"], "05_defterm_occurrences.jsonl": source.tables["occurrences"]}
