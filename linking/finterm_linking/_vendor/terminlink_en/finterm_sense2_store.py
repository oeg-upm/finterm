"""Internal Sense-2 transactional cache, bounded execution and consolidation.

SQLite commits one claim/result at a time. A committed unfinished claim is never
automatically regenerated, including after a process crash. Run exports are
immutable snapshots, published once; their manifest is the final commit point.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from finterm_sense2 import (BASE, SCHEMA, Stage02Input, canonical, component_hashes, digest,
                               file_hash, lexical_count, load_stage02, object_hash, prepare_sense2, require)
from finterm_sense2_qwen import (PRODUCTION_BACKEND, TEST_BACKEND, AdapterUnavailable, exception_record,
                                   generation_contract, validate_contract, validate_generation)


CACHE_SCHEMA = "finterm-sense2-generations-sqlite-v1"
GENERATION_FIELDS = {"task_id", "generation_call_status", "translation_raw", "finish_reason",
                     "input_token_count", "generated_token_count", "error", "metadata"}
METADATA_FIELDS = {"backend_kind", "model_id", "model_revision", "snapshot_path", "resource_json_sha256",
                   "python_version", "torch_version", "transformers_version", "cuda_version", "device_name",
                   "runtime_versions", "quantization", "generation", "local_files_only", "test_only"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_record(record, task):
    require(isinstance(record, dict) and set(record) == GENERATION_FIELDS, "Invalid generation record schema")
    require(record["task_id"] == task["task_id"] and isinstance(record["translation_raw"], str), "Generation/task identity differs")
    require(record["generation_call_status"] in {"RETURNED", "RAISED"}, "Invalid generation call status")
    require(record["finish_reason"] in {"eos_token", "max_new_tokens", "unknown_stop", "generation_exception", "input_context_limit"},
            "Unknown generation finish reason")
    for field in ("input_token_count", "generated_token_count"):
        require(record[field] is None or type(record[field]) is int and record[field] >= 0, "Invalid token count")
    if record["finish_reason"] == "eos_token":
        require(record["generated_token_count"] is not None and record["generated_token_count"] > 0,
                "EOS claim lacks an observed generated token")
    require((record["error"] is None) == (record["generation_call_status"] == "RETURNED"), "Generation error/status differs")
    if record["error"] is not None:
        require(isinstance(record["error"], dict) and set(record["error"]) == {"type", "code"}
                and all(isinstance(value, str) for value in record["error"].values()), "Unexpected error metadata")
    metadata = record["metadata"]
    require(isinstance(metadata, dict) and set(metadata) <= METADATA_FIELDS, "Unexpected generation metadata (no auth/config secrets)")
    contract = task["request"]["generation_contract"]
    require(all(metadata.get(key) == contract[key] for key in ("backend_kind", "model_id", "model_revision")),
            "Generation model provenance differs")
    if contract["backend_kind"] == TEST_BACKEND:
        require(metadata.get("test_only") is True, "Synthetic backend output must be explicitly marked")
    canonical(record)  # Reject non-JSON/nonfinite metadata before a transaction.


class TranslationStore:
    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        require(self.path.suffix in {".sqlite", ".sqlite3", ".db"} and self.path.parent.is_dir(),
                "Use an explicit SQLite cache file in an existing directory")
        self.connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        try:
            tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                self.connection.execute("BEGIN IMMEDIATE")
                try:
                    self.connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    self.connection.execute("INSERT INTO meta VALUES ('schema_version', ?)", (CACHE_SCHEMA,))
                    self.connection.execute("""CREATE TABLE generations (
                        task_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('INFLIGHT','DONE')), claim_id TEXT NOT NULL,
                        claimed_utc TEXT NOT NULL, finished_utc TEXT,
                        generation_json TEXT, generation_sha256 TEXT)""")
                    self.connection.execute("COMMIT")
                except BaseException:
                    self.connection.execute("ROLLBACK")
                    raise
            require({row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"meta", "generations"},
                    "Unrecognized existing SQLite cache")
            row = self.connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            require(row is not None and row[0] == CACHE_SCHEMA, "Unsupported translation cache schema")
            expected = {"task_id", "request_json", "request_sha256", "state", "claim_id", "claimed_utc",
                        "finished_utc", "generation_json", "generation_sha256"}
            require({row[1] for row in self.connection.execute("PRAGMA table_info(generations)")} == expected, "Translation cache columns differ")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.verify()
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def verify(self):
        require([row[0] for row in self.connection.execute("PRAGMA quick_check")] == ["ok"], "SQLite integrity check failed")

    def lookup(self, task):
        expected = canonical(task["request"])
        checksum = digest(expected.encode("utf-8"))
        require(task["task_id"] == "translation_" + checksum, "Invalid planned translation identity")
        row = self.connection.execute("SELECT * FROM generations WHERE task_id=?", (task["task_id"],)).fetchone()
        if row is None:
            return None
        require(row["request_sha256"] == checksum and row["request_json"] == expected, "Cached request/evidence hash differs")
        require(row["state"] in {"INFLIGHT", "DONE"} and bool(row["claim_id"]) and bool(row["claimed_utc"]), "Invalid cached claim")
        record = None
        if row["state"] == "DONE":
            require(row["generation_json"] is not None and row["finished_utc"] is not None
                    and digest(row["generation_json"].encode("utf-8")) == row["generation_sha256"], "Cached raw generation hash differs")
            record = json.loads(row["generation_json"])
            require(canonical(record) == row["generation_json"], "Cached generation is not canonical")
            validate_record(record, task)
        else:
            require(row["generation_json"] is None and row["generation_sha256"] is None and row["finished_utc"] is None,
                    "Uncommitted cache entry contains a result")
        return {"state": row["state"], "generation": record, "generation_sha256": row["generation_sha256"],
                "claimed_utc": row["claimed_utc"], "finished_utc": row["finished_utc"]}

    def claim(self, task):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.lookup(task) is not None:
                self.connection.execute("COMMIT")
                return None
            claim = uuid.uuid4().hex
            self.connection.execute("INSERT INTO generations VALUES (?, ?, ?, 'INFLIGHT', ?, ?, NULL, NULL, NULL)",
                (task["task_id"], canonical(task["request"]), object_hash(task["request"]), claim, utc_now()))
            self.connection.execute("COMMIT")
            return claim
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def finish(self, task, claim, record):
        validate_record(record, task)
        payload = canonical(record)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            # Validate the durable request again immediately before committing.
            existing = self.lookup(task)
            require(existing is not None and existing["state"] == "INFLIGHT", "No unfinished translation claim")
            cursor = self.connection.execute("""UPDATE generations SET state='DONE', finished_utc=?,
                generation_json=?, generation_sha256=? WHERE task_id=? AND state='INFLIGHT' AND claim_id=?""",
                (utc_now(), payload, digest(payload.encode("utf-8")), task["task_id"], claim))
            require(cursor.rowcount == 1, "Translation claim belongs to another writer")
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise


def _task_view(task, entry, *, blocked=False):
    if entry is None:
        return {**task, "processing_status": "BLOCKED_ADAPTER" if blocked else "PENDING_BUDGET",
                "generation": None, "generation_sha256": None, "validation": None}
    if entry["state"] == "INFLIGHT":
        return {**task, **entry, "processing_status": "INFLIGHT_OR_INTERRUPTED", "validation": None}
    validation = validate_generation(entry["generation"], task["request"])
    status = ("TRANSLATED" if validation["available_for_later_evaluation"] else
              "GENERATION_ERROR" if entry["generation"]["generation_call_status"] == "RAISED" else
              "INVALID_STRUCTURE" if validation["structural_status"] != "VALID" else "INCOMPLETE_GENERATION")
    return {**task, **entry, "processing_status": status, "validation": validation}


def consolidate(plan, task_views):
    """All source alternatives survive, independently of task/word-count status."""
    by_task = {row["task_id"]: row for row in task_views}
    representations = []
    for source in plan.payload["representations"]:
        row = {**source, "text_en": None, "processing_status": source["planned_status"],
               "structural_validity": "NOT_APPLICABLE", "fidelity_status": "NOT_VERIFIED",
               "evidence_sufficiency": "NOT_ASSESSED", "available_for_later_evaluation": False,
               "lexical_word_count": 0, "validation": None, "generation_sha256": None}
        if source["is_original_english"]:
            row.update(text_en=source["source_text"], available_for_later_evaluation=bool(source["source_text"].strip()),
                       fidelity_status="NOT_APPLICABLE_ORIGINAL_ENGLISH")
        elif source["task_id"] is not None:
            translated = by_task[source["task_id"]]
            row["processing_status"] = translated["processing_status"]
            row["generation_sha256"] = translated["generation_sha256"]
            validation = translated["validation"]
            if validation is not None:
                row.update(text_en=validation["translation_clean"], structural_validity=validation["structural_status"],
                           validation=validation, available_for_later_evaluation=validation["available_for_later_evaluation"])
        row["lexical_word_count"] = lexical_count(row["text_en"]) if row["text_en"] else 0
        row["lexical_information"] = ("MULTIWORD_HEURISTIC" if row["lexical_word_count"] > 1 else
                                      "SINGLE_WORD" if row["lexical_word_count"] == 1 else "NO_LEXICAL_CONTENT")
        representations.append(row)
    by_id = {row["representation_id"]: row for row in representations}
    by_evidence = {row["evidence_id"]: row for row in plan.payload["evidence"]}
    candidates = []
    for candidate in plan.payload["candidates"]:
        reps = [by_id[rid] for rid in candidate["representation_ids"]]
        available = [row for row in reps if row["available_for_later_evaluation"]]
        states = Counter(row["processing_status"] for row in reps)
        data = by_evidence[candidate["evidence_id"]]
        has_evidence = any(data[field] for field in ("label_en", "aliases_en", "p31", "p279", "p1343"))
        state = ("TEXT_AVAILABLE_UNVERIFIED" if available else "ORIGINAL_NONENGLISH_EVIDENCE_ONLY" if
                 any(row["source_text"].strip() for row in reps) else "STRUCTURED_OR_LABEL_EVIDENCE_ONLY" if has_evidence else "NO_NONEMPTY_EVIDENCE")
        candidates.append({**candidate, "sense2_state": state, "available_representation_ids": [row["representation_id"] for row in available],
                           "representation_states": dict(states), "evidence_sufficiency": "NOT_ASSESSED", "enrichment_state": "NOT_RUN"})
    return candidates, representations


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path, rows):
    value = __import__("hashlib").sha256()
    with path.open("xb") as stream:
        for row in rows:
            line = canonical(row)
            for separator in ("\u0085", "\u2028", "\u2029"):
                line = line.replace(separator, "\\u" + format(ord(separator), "04x"))
            raw = (line + "\n").encode("utf-8")
            value.update(raw)
            stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return value.hexdigest()


def _check_paths(source, output, cache):
    require(output != BASE and cache.parent != BASE, "Use dedicated run/cache locations, not the historical code root")
    require(output.parent.is_dir() and cache.parent.is_dir(), "Output/cache parent directories must already exist")
    require(output != cache and output not in cache.parents and cache not in output.parents, "Keep output and cache separate")
    if source.source_directory is not None:
        root = source.source_directory
        require(output != root and root not in output.parents and output not in root.parents
                and cache != root and root not in cache.parents, "Do not write into or over stage-02 input directories")
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), "Output directory must be new or empty")


def run_stage03(stage02: Stage02Input | Path | str, output_dir: Path | str, cache_path: Path | str, *,
                policy="historical", max_translations=0, candidate_ids=None, terms=None, adapter=None):
    """One internal call from verified 02 to bounded 03. Default: zero inference.

    A budget counts newly claimed attempts, including failed/interrupted ones;
    hits do not consume it. Re-run into a new output directory using the same
    cache. DONE errors and ambiguous INFLIGHT claims are not retried implicitly.
    """
    require(type(max_translations) is int and max_translations >= 0, "Translation budget must be a finite nonnegative integer")
    source = stage02 if isinstance(stage02, Stage02Input) else load_stage02(stage02)
    source.validate()
    source.verify_files()
    contract = generation_contract() if adapter is None else json.loads(canonical(adapter.contract))
    validate_contract(contract)
    plan = prepare_sense2(source, policy=policy, candidate_ids=candidate_ids, terms=terms, generation_contract=contract)
    require(max_translations == 0 or adapter is not None, "An explicit adapter is required for a nonzero inference budget")
    output, cache = Path(output_dir).resolve(), Path(cache_path).resolve()
    _check_paths(source, output, cache)
    output.mkdir(exist_ok=True)
    marker = output / ".stage03_in_progress.json"
    _write_json(marker, {"plan_sha256": plan.sha256, "started_utc": utc_now()})
    components_before = component_hashes()
    counts = {"new_attempts": 0, "cache_hits": 0, "inflight_not_retried": 0}
    blocked, prepared, interruption = None, False, None
    task_views = []
    with TranslationStore(cache) as store:
        try:
            for task in plan.payload["tasks"]:
                entry = store.lookup(task)
                if entry is not None:
                    counts["cache_hits" if entry["state"] == "DONE" else "inflight_not_retried"] += 1
                elif counts["new_attempts"] < max_translations and blocked is None:
                    if not prepared:
                        try:
                            adapter.prepare()
                            require(adapter.contract == contract, "Adapter identity changed during initialization")
                        except Exception as exc:
                            blocked = {"type": type(exc).__name__, "code": str(exc) if isinstance(exc, AdapterUnavailable) else "ADAPTER_INITIALIZATION_FAILED_DETAILS_NOT_PERSISTED"}
                        else:
                            prepared = True
                    if blocked is None:
                        claim = store.claim(task)
                        if claim is not None:
                            counts["new_attempts"] += 1
                            try:
                                record = adapter.generate(task)
                                validate_record(record, task)
                            except Exception as exc:
                                metadata = {key: contract[key] for key in ("backend_kind", "model_id", "model_revision")}
                                if contract["backend_kind"] == TEST_BACKEND:
                                    metadata["test_only"] = True
                                record = exception_record(task["task_id"], exc, metadata=metadata)
                            store.finish(task, claim, record)
                        entry = store.lookup(task)
                        if claim is None and entry is not None:
                            counts["cache_hits" if entry["state"] == "DONE" else "inflight_not_retried"] += 1
                task_views.append(_task_view(task, entry, blocked=blocked is not None))
        except BaseException as exc:
            if isinstance(exc, Exception):
                raise  # Integrity/programming errors never become valid run snapshots.
            interruption = exc
            task_views = [_task_view(task, store.lookup(task)) for task in plan.payload["tasks"]]
        store.verify()
    plan.validate()
    source.verify_files()
    require(component_hashes() == components_before, "Code changed during the Sense-2 run")
    candidates, representations = consolidate(plan, task_views)
    task_states = Counter(row["processing_status"] for row in task_views)
    pending = sum(count for state, count in task_states.items() if state in {"PENDING_BUDGET", "BLOCKED_ADAPTER", "INFLIGHT_OR_INTERRUPTED"})
    failures = sum(count for state, count in task_states.items() if state in {"GENERATION_ERROR", "INVALID_STRUCTURE", "INCOMPLETE_GENERATION"})
    upstream_partial = source.provenance.get("status") == "PARTIAL_OR_FAILED" or any(
        row["retrieval_status"] in {"RETRIEVAL_ERROR", "CANDIDATES_AVAILABLE_WITH_ERRORS"} for row in plan.payload["terms"])
    status = ("INTERRUPTED" if interruption is not None else "PARTIAL" if upstream_partial or failures or blocked or task_states["INFLIGHT_OR_INTERRUPTED"] else
              "PREPARED" if pending and max_translations == 0 else "PARTIAL" if pending else "COMPLETED")
    tables = {"03_candidates.jsonl": candidates, "03_evidence.jsonl": plan.payload["evidence"],
              "03_representations.jsonl": representations, "03_translations.jsonl": task_views,
              "03_associations.jsonl": plan.payload["associations"], "03_terms.jsonl": plan.payload["terms"],
              "03_occurrences.jsonl": plan.payload["occurrences"]}
    output_hashes = {name: _write_jsonl(output / name, rows) for name, rows in tables.items()}
    _write_json(output / "03_source.json", source.provenance)
    output_hashes["03_source.json"] = file_hash(output / "03_source.json")
    manifest = {"schema_version": SCHEMA, "status": status, "created_utc": utc_now(),
                "execution_scope": "SENSE2_BUILD_ONLY", "policy": plan.payload["policy"],
                "lexical_policy_version": plan.payload["lexical_policy_version"], "selection": plan.payload["selection"],
                "input_content_sha256": source.content_sha256, "plan_sha256": plan.sha256,
                "generation_contract": contract, "component_sha256": components_before,
                "execution": {"backend_kind": contract["backend_kind"], "max_translations": max_translations,
                              **counts, "adapter_error": blocked, "task_states": dict(task_states)},
                "counts": {**plan.counts, "available_representations": sum(row["available_for_later_evaluation"] for row in representations),
                           "pending_translations": pending, "failed_translations": failures},
                "cache": {"schema_version": CACHE_SCHEMA, "path": str(cache), "sqlite_integrity": "ok",
                          "reuse_identity": "EXACT_CANDIDATE_EVIDENCE_MESSAGES_AND_GENERATION_CONTRACT"},
                "output_sha256": output_hashes, "fidelity_verified": False, "evidence_sufficiency_assessed": False}
    _write_json(output / "03_manifest.json", manifest)
    marker.unlink()
    if interruption is not None:
        raise interruption
    return manifest
