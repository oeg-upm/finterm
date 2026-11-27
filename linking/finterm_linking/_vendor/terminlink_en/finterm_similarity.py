"""Independent semantic-similarity stage: per-occurrence cosine and QID MAX.

Consumes immutable DefTerm contracts (including inherited phase-01 contexts and
phase-02 associations). No linking decision, score threshold or second reranker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
from collections import Counter, defaultdict

from finterm_sense2 import BASE, _safe_member, _selection, canonical, candidate_id, file_hash, object_hash, require
from finterm_sense2_store import _write_json, _write_jsonl, utc_now
from finterm_similarity_qwen import (BACKEND_KIND, TEST_BACKEND, InputTooLong, contract, exclusive_lock, official_cosine)


SCHEMA = "finterm-semantic-similarity-v1"
CACHE_SCHEMA = "finterm-normalized-embeddings-sqlite-v1"
TABLES = {"candidates": "05_defterm_candidates.jsonl", "representations": "05_defterm_representations.jsonl",
          "associations": "05_defterm_associations.jsonl", "terms": "05_defterm_terms.jsonl",
          "occurrences": "05_defterm_occurrences.jsonl"}
PENDING_TRANSLATIONS = {"PENDING_BUDGET", "BLOCKED_ADAPTER", "INFLIGHT_OR_INTERRUPTED"}


def _index(rows, field):
    result = {row[field]: row for row in rows}
    require(len(result) == len(rows), "Duplicate similarity input identifier: " + field)
    return result


@dataclass(frozen=True)
class SimilarityInput:
    folder: Path
    manifest: dict
    tables: dict
    fingerprints: dict
    crosschecks: dict
    sha256: str

    def verify(self):
        require(object_hash({"manifest": self.manifest, "tables": self.tables, "crosschecks": self.crosschecks}) == self.sha256, "Similarity in-memory input changed")
        for name, checksum in self.fingerprints.items():
            require(file_hash(_safe_member(self.folder, name)) == checksum, "Similarity input file changed: " + name)
        for absolute, checksum in self.crosschecks.get("files_sha256", {}).items():
            require(file_hash(Path(absolute)) == checksum.lower(), "Authoritative phase-01/02 file changed")


def load_similarity_input(folder, *, stage01_dir=None, stage02_dir=None):
    root = Path(folder).resolve()
    payload = (root / "05_defterm_manifest.json").read_bytes()
    manifest = json.loads(payload)
    require(manifest.get("schema_version") == "finterm-defterm-v1" and manifest.get("status") in
            {"PREPARED", "PARTIAL", "COMPLETED", "INTERRUPTED"}, "Unsupported consolidated DefTerm contract")
    hashes = manifest.get("output_sha256", {})
    require(set(TABLES.values()) <= set(hashes) and "05_defterm_manifest.json" not in hashes, "DefTerm manifest omits similarity contracts")
    tables, fingerprints = {}, {}
    for name, expected in hashes.items():
        path = _safe_member(root, name)
        checksum = file_hash(path)
        require(isinstance(expected, str) and checksum == expected.lower(), "DefTerm input hash differs: " + name)
        fingerprints[name] = checksum
        if name in TABLES.values():
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == checksum, "DefTerm input changed while reading")
            key = next(key for key, member in TABLES.items() if member == name)
            tables[key] = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
    fingerprints["05_defterm_manifest.json"] = hashlib.sha256(payload).hexdigest()
    _validate_tables(tables)
    require(manifest["counts"]["candidates"] == len(tables["candidates"]) and manifest["counts"]["final_representations"] == len(tables["representations"]),
            "DefTerm manifest counts differ")
    crosschecks = {"sense1_authority": "IMMUTABLE_PHASE01_CONTEXTS_COPIED_IN_DEFTERM",
                   "candidate_authority": "IMMUTABLE_PHASE02_INVENTORY_AND_ASSOCIATIONS_COPIED_IN_DEFTERM", "files_sha256": {}, "directories": []}
    if stage01_dir is not None:
        from terminlink_stage02 import load_stage01
        phase01 = load_stage01(Path(stage01_dir).resolve())
        contexts = _index(phase01["contexts"], "occurrence_uid")
        for occurrence in tables["occurrences"]:
            original = contexts.get(occurrence["occurrence_uid"])
            require(original is not None and all(occurrence[key] == original[key] for key in
                    ("document_id", "term_id", "term", "context", "occurrence_id")), "Sense-1 context/identity differs from phase 01")
        for name, checksum in phase01["input_sha256"].items():
            crosschecks["files_sha256"][str(Path(stage01_dir).resolve() / name)] = checksum.lower()
        crosschecks["sense1_authority"] = "CROSSCHECKED_ORIGINAL_STAGE01"
        crosschecks["directories"].append(str(Path(stage01_dir).resolve()))
    if stage02_dir is not None:
        from finterm_sense2 import load_stage02
        phase02 = load_stage02(stage02_dir)
        originals = {candidate_id(row): row for row in phase02.candidates}
        for row in tables["candidates"]:
            require(originals.get(row["candidate_id"]) == row["original"], "Candidate inventory differs from phase 02")
        associations = {(row["occurrence_uid"], row["candidate_id"]): row for row in phase02.associations}
        require(all(associations.get((row["occurrence_uid"], row["candidate_id"])) == row for row in tables["associations"]),
                "Association differs from phase 02")
        crosschecks["files_sha256"].update({str(phase02.source_directory / name): checksum for name, checksum in phase02.provenance["files_sha256"].items()})
        crosschecks["candidate_authority"] = "CROSSCHECKED_ORIGINAL_STAGE02"
        crosschecks["directories"].append(str(Path(stage02_dir).resolve()))
    value = {"manifest": manifest, "tables": tables, "crosschecks": crosschecks}
    return SimilarityInput(root, manifest, tables, fingerprints, crosschecks, object_hash(value))


def _representation_ids(candidate):
    return candidate.get("all_representation_ids", candidate["representation_ids"])


def _validate_tables(tables):
    candidates = _index(tables["candidates"], "candidate_id")
    representations = _index(tables["representations"], "representation_id")
    terms = _index(tables["terms"], "term_id")
    occurrences = _index(tables["occurrences"], "occurrence_uid")
    referenced = set()
    for row in candidates.values():
        require(row["candidate_id"] == candidate_id(row) and row["qid"] == row["original"]["qid"]
                and row["term"] == row["original"]["term"], "Candidate identity differs")
        ids = _representation_ids(row)
        require(isinstance(ids, list) and len(ids) == len(set(ids)), "Invalid candidate representation list")
        for rid in ids:
            rep = representations.get(rid)
            require(rep is not None and rep["qid"] == row["qid"] and rep["evidence_id"] == row["evidence_id"], "Foreign/missing candidate representation")
            require(rep.get("candidate_id", row["candidate_id"]) == row["candidate_id"], "Generated representation belongs to another candidate")
            referenced.add(rid)
    require(referenced == set(representations), "Unassociated representation must not be silently dropped")
    for row in representations.values():
        require(type(row["available_for_later_evaluation"]) is bool, "Invalid representation availability flag")
        if row["available_for_later_evaluation"]:
            require(isinstance(row["text_en"], str) and row["text_en"].strip(), "Available English representation is empty")
    for row in occurrences.values():
        term = terms.get(row["term_id"])
        require(term is not None and all(row[key] == term[key] for key in ("document_id", "term")) and isinstance(row["context"], str),
                "Occurrence differs from its term/document")
    seen = set()
    for row in tables["associations"]:
        occurrence, candidate = occurrences.get(row["occurrence_uid"]), candidates.get(row["candidate_id"])
        require(occurrence is not None and candidate is not None and row["qid"] == candidate["qid"] and row["term"] == candidate["term"],
                "Association refers to an absent/wrong candidate or occurrence")
        require(all(row[key] == occurrence[key] for key in ("context", "document_id", "term_id", "term", "occurrence_id")), "Associated Sense-1 context changed")
        pair = (row["occurrence_uid"], row["candidate_id"])
        require(pair not in seen, "Duplicate occurrence/candidate association")
        seen.add(pair)


def embedding_request(text, identity):
    request = {"model": identity, "text": text}
    return {"embedding_id": "embedding_" + object_hash(request), "request": request}


def prepare_similarity(source, *, terms=None, identity=None):
    source = source if isinstance(source, SimilarityInput) else load_similarity_input(source)
    source.verify()
    identity = json.loads(canonical(contract() if identity is None else identity))
    require(identity.get("backend_kind") in {BACKEND_KIND, TEST_BACKEND} and type(identity.get("embedding_dimension")) is int
            and identity["embedding_dimension"] > 0 and identity.get("normalize_embeddings") is True and identity.get("similarity") == "cosine",
            "Unsupported embedding identity")
    if identity["backend_kind"] == BACKEND_KIND:
        require(identity == contract(), "Embedding production contract differs")
    selected = _selection(terms, {row["term"] for row in source.tables["terms"]}, "similarity term filter")
    candidates = _index(source.tables["candidates"], "candidate_id")
    reps = _index(source.tables["representations"], "representation_id")
    occurrences = _index(source.tables["occurrences"], "occurrence_uid")
    requests, pairs = {}, []
    for association in source.tables["associations"]:
        occurrence = occurrences[association["occurrence_uid"]]
        if selected is not None and occurrence["term"] not in selected:
            continue
        candidate = candidates[association["candidate_id"]]
        available = [reps[rid] for rid in _representation_ids(candidate) if reps[rid]["available_for_later_evaluation"]]
        if not available:
            continue  # Preserve the candidate in rankings with a null score, never embed labels as a fallback.
        require(occurrence["context"].strip(), "An evaluable occurrence has an empty context")
        context_request = embedding_request(occurrence["context"], identity)
        requests[context_request["embedding_id"]] = context_request
        for representation in available:
            rep_request = embedding_request(representation["text_en"], identity)
            requests[rep_request["embedding_id"]] = rep_request
            pairs.append({"pair_id": "similarity_" + object_hash([occurrence["occurrence_uid"], candidate["candidate_id"],
                          representation["representation_id"], context_request["embedding_id"], rep_request["embedding_id"]]),
                          "document_id": occurrence["document_id"], "term_id": occurrence["term_id"], "term": occurrence["term"],
                          "occurrence_uid": occurrence["occurrence_uid"], "occurrence_id": occurrence["occurrence_id"],
                          "candidate_id": candidate["candidate_id"], "qid": candidate["qid"],
                          "representation_id": representation["representation_id"], "representation_origin": representation["origin"],
                          "source_language": representation["source_language"], "context_embedding_id": context_request["embedding_id"],
                          "representation_embedding_id": rep_request["embedding_id"],
                          "context_sha256": hashlib.sha256(occurrence["context"].encode("utf-8")).hexdigest(), "representation_sha256": object_hash(representation),
                          "representation_text_sha256": hashlib.sha256(representation["text_en"].encode("utf-8")).hexdigest()})
    return {"source": source, "identity": identity, "requests": list(requests.values()), "pairs": pairs,
            "selection": {"terms": sorted(selected) if selected is not None else None},
            "counts": {"input_candidates": len(candidates), "input_occurrences": len(occurrences), "input_representations": len(reps),
                       "individual_pairs": len(pairs), "unique_embedding_requests": len(requests)}}


def _unit_vector(vector, dimension):
    require(isinstance(vector, (list, tuple)) and len(vector) == dimension and all(type(value) in {int, float} and math.isfinite(value) for value in vector),
            "Embedding has invalid dimension or nonfinite components")
    norm = math.sqrt(sum(value * value for value in vector))
    require(norm > 0 and abs(norm - 1.0) <= 1e-4, "Embedding is zero or not L2-normalized")  # Numerical tolerance, never a link threshold.
    return [float(value) for value in vector]


class EmbeddingCache:
    def __init__(self, path):
        self.path = Path(path).resolve()
        require(self.path.suffix in {".sqlite", ".sqlite3", ".db"} and self.path.parent.is_dir(), "Use an embedding SQLite cache in an existing parent")

    def __enter__(self):
        self.lock = exclusive_lock(self.path.with_suffix(self.path.suffix + ".lock"))
        self.lock.__enter__()
        try:
            self.connection = sqlite3.connect(self.path)
            self.connection.row_factory = sqlite3.Row
            tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                self.connection.execute("CREATE TABLE meta (schema_version TEXT NOT NULL)")
                self.connection.execute("INSERT INTO meta VALUES (?)", (CACHE_SCHEMA,))
                self.connection.execute("CREATE TABLE vectors (embedding_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, dimension INTEGER NOT NULL, data BLOB NOT NULL, checksum TEXT NOT NULL, runtime_json TEXT NOT NULL, record_sha256 TEXT NOT NULL)")
                self.connection.commit()
            require({row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"meta", "vectors"}, "Foreign embedding cache database")
            require([row[0] for row in self.connection.execute("SELECT schema_version FROM meta")] == [CACHE_SCHEMA], "Embedding cache schema differs")
            require({row[1] for row in self.connection.execute("PRAGMA table_info(vectors)")} ==
                    {"embedding_id", "request_json", "dimension", "data", "checksum", "runtime_json", "record_sha256"}, "Embedding cache columns differ")
            require([row[0] for row in self.connection.execute("PRAGMA quick_check")] == ["ok"], "Embedding cache integrity failed")
            return self
        except BaseException:
            if hasattr(self, "connection"):
                self.connection.close()
            self.lock.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self.connection.close()
        self.lock.__exit__(*args)

    def get(self, item):
        require(item["embedding_id"] == "embedding_" + object_hash(item["request"]), "Embedding request identity changed")
        row = self.connection.execute("SELECT * FROM vectors WHERE embedding_id=?", (item["embedding_id"],)).fetchone()
        if row is None:
            return None
        require(row["request_json"] == canonical(item["request"]) and row["dimension"] == item["request"]["model"]["embedding_dimension"], "Cached embedding request/dimension differs")
        require(len(row["data"]) == row["dimension"] * 4 and hashlib.sha256(row["data"]).hexdigest() == row["checksum"], "Cached embedding payload hash differs")
        runtime = json.loads(row["runtime_json"])
        require(row["record_sha256"] == object_hash([item["embedding_id"], row["request_json"], row["checksum"], runtime]),
                "Cached embedding provenance hash differs")
        require(runtime.get("backend_kind") == item["request"]["model"]["backend_kind"] and
                (runtime["backend_kind"] != TEST_BACKEND or runtime.get("test_only") is True), "Cached embedding runtime differs")
        values = list(struct.unpack("<" + str(row["dimension"]) + "f", row["data"]))
        _unit_vector(values, row["dimension"])
        return {"vector": values, "sha256": row["checksum"], "runtime": runtime}

    def put(self, item, vector, runtime):
        dimension = item["request"]["model"]["embedding_dimension"]
        values = _unit_vector(vector, dimension)
        require(runtime.get("backend_kind") == item["request"]["model"]["backend_kind"], "Embedding runtime/model namespace differs")
        if runtime["backend_kind"] == TEST_BACKEND:
            require(runtime.get("test_only") is True, "Controlled embeddings must be marked")
        data = struct.pack("<" + str(dimension) + "f", *values)
        checksum = hashlib.sha256(data).hexdigest()
        existing = self.get(item)
        require(existing is None or existing["sha256"] == checksum, "Refusing an incompatible embedding overwrite")
        if existing is None:
            request_json = canonical(item["request"])
            self.connection.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?,?)", (item["embedding_id"], request_json,
                dimension, data, checksum, canonical(runtime), object_hash([item["embedding_id"], request_json, checksum, runtime])))
            self.connection.commit()  # One successful vector, not a rewrite of the result prefix.


def _paths(source, output, cache):
    require(output.parent.is_dir() and cache.parent.is_dir() and output != BASE and cache.parent != BASE, "Use explicit independent run/cache locations")
    require(output != source.folder and source.folder not in output.parents and output not in source.folder.parents
            and source.folder not in cache.parents and output not in cache.parents and cache not in output.parents, "Do not write into inputs or combine output/cache")
    for checked in source.crosschecks.get("files_sha256", {}):
        require(output not in Path(checked).parents and cache != Path(checked), "Output/cache overlaps authoritative input")
    for directory in source.crosschecks.get("directories", []):
        root = Path(directory)
        require(root != output and root not in output.parents and root not in cache.parents and output not in root.parents,
                "Do not write inside authoritative phase-01/02 directories")
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), "Similarity output must be new or empty")


def run_similarity(defterm_dir, output_dir, cache_path, *, backend=None, max_new_embeddings=0, terms=None,
                   stage01_dir=None, stage02_dir=None):
    require(type(max_new_embeddings) is int and max_new_embeddings >= 0, "Embedding budget must be a finite nonnegative integer")
    require(max_new_embeddings == 0 or backend is not None, "Explicit embedding backend required for new inference")
    source = defterm_dir if isinstance(defterm_dir, SimilarityInput) else load_similarity_input(defterm_dir, stage01_dir=stage01_dir, stage02_dir=stage02_dir)
    identity = contract() if backend is None else backend.identity
    plan = prepare_similarity(source, terms=terms, identity=identity)
    output, cache_path = Path(output_dir).resolve(), Path(cache_path).resolve()
    _paths(source, output, cache_path)
    output.mkdir(exist_ok=True)
    _write_json(output / ".similarity_in_progress.json", {"source_sha256": source.sha256, "created_utc": utc_now()})
    components = {name: file_hash(BASE / name) for name in ("finterm_similarity.py", "finterm_similarity_qwen.py")}
    vectors, errors = {}, {}
    execution = {"new_embedding_attempts": 0, "new_embeddings_cached": 0, "encoding_batches": 0, "cache_hits": 0,
                 "backend_error": None, "scoring_errors": 0}
    interruption = None
    with EmbeddingCache(cache_path) as cache:
        missing = []
        for item in plan["requests"]:
            entry = cache.get(item)
            if entry is None:
                missing.append(item)
            else:
                vectors[item["embedding_id"]] = entry
                execution["cache_hits"] += 1
        allowed = missing[:max_new_embeddings]
        if allowed:
            try:
                backend.prepare()
                require(backend.identity == plan["identity"], "Embedding backend identity changed during initialization")
            except Exception:
                execution["backend_error"] = "EMBEDDING_BACKEND_UNAVAILABLE_CHECK_ENVIRONMENT_RESOURCES_AND_GPU"
            else:
                batch_size = getattr(backend, "batch_size", 1)
                require(type(batch_size) is int and batch_size > 0, "Invalid embedding batch size")
                try:
                    for start in range(0, len(allowed), batch_size):
                        batch = allowed[start:start + batch_size]
                        execution["new_embedding_attempts"] += len(batch)
                        _encode_batch(batch, backend, cache, vectors, errors, execution)
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        raise
                    interruption = exc
    scorer = backend.similarity if backend is not None else official_cosine
    pairs = _score_pairs(plan, vectors, errors, scorer, execution)
    rankings, occurrence_status = _rank(plan, pairs)
    source.verify()
    require(components == {name: file_hash(BASE / name) for name in components}, "Similarity code changed during run")
    encoding = [{"embedding_id": item["embedding_id"], "text_sha256": hashlib.sha256(item["request"]["text"].encode("utf-8")).hexdigest(),
                 "state": "AVAILABLE" if item["embedding_id"] in vectors else errors.get(item["embedding_id"], "BLOCKED_BACKEND" if execution["backend_error"] else "PENDING_BUDGET"),
                 "vector_sha256": vectors[item["embedding_id"]]["sha256"] if item["embedding_id"] in vectors else None}
                for item in plan["requests"]]
    tables = {"06_similarity_scores.jsonl": pairs, "06_candidate_rankings.jsonl": rankings, "06_occurrence_status.jsonl": occurrence_status,
              "06_embedding_status.jsonl": encoding, "06_input_candidates.jsonl": source.tables["candidates"],
              "06_input_representations.jsonl": source.tables["representations"], "06_input_terms.jsonl": source.tables["terms"],
              "06_input_associations.jsonl": source.tables["associations"]}
    hashes = {name: _write_jsonl(output / name, rows) for name, rows in tables.items()}
    _write_json(output / "06_similarity_source.json", {"defterm_directory": str(source.folder), "files_sha256": source.fingerprints, "crosschecks": source.crosschecks})
    hashes["06_similarity_source.json"] = file_hash(output / "06_similarity_source.json")
    pending = sum(row["state"] != "AVAILABLE" for row in encoding)
    status = "INTERRUPTED" if interruption else "PARTIAL" if errors or execution["backend_error"] or execution["scoring_errors"] else "PREPARED" if pending and max_new_embeddings == 0 else "PARTIAL" if pending else "COMPLETED"
    manifest = {"schema_version": SCHEMA, "status": status, "created_utc": utc_now(), "source_status": source.manifest["status"],
                "execution_scope": "SEMANTIC_SCORES_AND_MAX_RANKINGS_ONLY", "model": plan["identity"], "selection": plan["selection"],
                "similarity": "COSINE_OFFICIAL_SENTENCE_TRANSFORMERS", "aggregation": "MAX_PER_OCCURRENCE_AND_CANDIDATE",
                "tie_breaking": "NUMERIC_QID_THEN_CANDIDATE_ID_FOR_DISPLAY_ONLY", "input_sha256": source.sha256,
                "counts": {**plan["counts"], "scored_pairs": sum(row["score"] is not None for row in pairs), "pending_embeddings": pending,
                           "scored_rank_rows": sum(row["score"] is not None for row in rankings)},
                "execution": {**execution, "max_new_embeddings": max_new_embeddings}, "component_sha256": components, "output_sha256": hashes,
                "embedding_runtime_records": list({object_hash(entry["runtime"]): entry["runtime"] for entry in vectors.values()}.values()),
                "cache": {"schema_version": CACHE_SCHEMA, "path": str(cache_path)}, "threshold_applied": False, "final_link_selected": False}
    _write_json(output / "06_similarity_manifest.json", manifest)
    (output / ".similarity_in_progress.json").unlink()
    if interruption:
        raise interruption
    return manifest


def _encode_batch(batch, backend, cache, vectors, errors, execution):
    try:
        execution["encoding_batches"] += 1
        encoded = backend.encode([item["request"]["text"] for item in batch])
    except InputTooLong as exc:
        require(exc.indices and all(type(index) is int and 0 <= index < len(batch) for index in exc.indices), "Invalid oversized-text indices")
        for index in exc.indices:
            errors[batch[index]["embedding_id"]] = "INPUT_TOO_LONG"
        remaining = [item for index, item in enumerate(batch) if index not in exc.indices]
        if remaining:
            _encode_batch(remaining, backend, cache, vectors, errors, execution)
        return
    except Exception:
        for item in batch:
            errors[item["embedding_id"]] = "ENCODING_ERROR_DETAILS_NOT_PERSISTED"
        return
    require(isinstance(encoded, list) and len(encoded) == len(batch), "Embedding result cardinality differs")
    for item, vector in zip(batch, encoded):
        cache.put(item, vector, backend.runtime)
        vectors[item["embedding_id"]] = cache.get(item)
        execution["new_embeddings_cached"] += 1


def _score_pairs(plan, vectors, errors, scorer, execution):
    rows = [{**pair, "score": None, "scoring_status": "PENDING_EMBEDDING"} for pair in plan["pairs"]]
    groups = defaultdict(list)
    for row in rows:
        ids = (row["context_embedding_id"], row["representation_embedding_id"])
        if any(eid in errors for eid in ids):
            row["scoring_status"] = "EMBEDDING_ERROR"
        elif all(eid in vectors for eid in ids):
            groups[row["occurrence_uid"]].append(row)
    for group in groups.values():
        context = vectors[group[0]["context_embedding_id"]]["vector"]
        try:
            values = scorer(context, [vectors[row["representation_embedding_id"]]["vector"] for row in group])
            require(len(values) == len(group) and all(type(value) in {float, int} and math.isfinite(value) and abs(value) <= 1.0001 for value in values),
                    "Invalid official cosine result")
        except Exception:
            execution["scoring_errors"] += len(group)
            for row in group:
                row["scoring_status"] = "SIMILARITY_ERROR"
            continue
        for row, score in zip(group, values):
            row.update(score=float(score), scoring_status="SCORED", context_vector_sha256=vectors[row["context_embedding_id"]]["sha256"],
                       representation_vector_sha256=vectors[row["representation_embedding_id"]]["sha256"])
    return rows


def _rank(plan, pairs):
    source = plan["source"]
    by_pair = defaultdict(list)
    for row in pairs:
        by_pair[(row["occurrence_uid"], row["candidate_id"])].append(row)
    candidates = _index(source.tables["candidates"], "candidate_id")
    reps = _index(source.tables["representations"], "representation_id")
    selected = set(plan["selection"]["terms"]) if plan["selection"]["terms"] is not None else None
    groups, rankings = defaultdict(list), []
    for association in source.tables["associations"]:
        candidate = candidates[association["candidate_id"]]
        rows = by_pair[(association["occurrence_uid"], candidate["candidate_id"])]
        scored = [row for row in rows if row["score"] is not None]
        available = [reps[rid] for rid in _representation_ids(candidate) if reps[rid]["available_for_later_evaluation"]]
        pending = sum(reps[rid]["processing_status"] in PENDING_TRANSLATIONS for rid in _representation_ids(candidate))
        maximum = max(row["score"] for row in scored) if scored else None
        maximizing = sorted(row["representation_id"] for row in scored if row["score"] == maximum)
        status = ("NOT_IN_PILOT_SCOPE" if selected is not None and association["term"] not in selected else
                  "NO_AVAILABLE_ENGLISH_TEXT" if not available else "SCORED_COMPLETE" if len(scored) == len(available) else
                  "SCORED_PARTIAL" if scored else "EMBEDDING_OR_SIMILARITY_ERROR" if any(row["scoring_status"] not in {"SCORED", "PENDING_EMBEDDING"} for row in rows) else "PENDING_EMBEDDING")
        row = {key: association[key] for key in ("document_id", "term_id", "term", "occurrence_uid", "occurrence_id", "candidate_id", "qid", "retrieval_status")}
        row.update(score=maximum, rank_position=None, score_rank=None, ranking_status=status, max_representation_ids=maximizing,
                   representative_representation_id=maximizing[0] if maximizing else None, available_representation_count=len(available),
                   scored_representation_count=len(scored), source_representation_count=len(_representation_ids(candidate)),
                   pending_translation_count=pending, source_sense2_state=candidate["sense2_state"], source_defterm=candidate["defterm"],
                   aggregation="MAX", complete_over_available_texts=len(scored) == len(available) and bool(available))
        groups[association["occurrence_uid"]].append(row)
    occurrences = []
    for occurrence in source.tables["occurrences"]:
        group = groups[occurrence["occurrence_uid"]]
        scored = sorted([row for row in group if row["score"] is not None], key=lambda row: (-row["score"], int(row["qid"][1:]), row["candidate_id"]))
        previous_score, score_rank = None, None
        for position, row in enumerate(scored, 1):
            if previous_score is None or row["score"] != previous_score:
                score_rank = position
            row.update(rank_position=position, score_rank=score_rank)
            previous_score = row["score"]
        rankings.extend(scored + sorted([row for row in group if row["score"] is None], key=lambda row: (int(row["qid"][1:]), row["candidate_id"])))
        occurrences.append({**occurrence, "similarity_scope": "SELECTED" if selected is None or occurrence["term"] in selected else "NOT_IN_PILOT_SCOPE",
                            "associated_candidate_count_in_this_input": len(group), "scored_candidate_count": len(scored),
                            "similarity_status": "NO_ASSOCIATED_CANDIDATES" if not group else "SCORED" if scored else "NO_SCORES_AVAILABLE"})
    return rankings, occurrences


def reference_positions(output_dir, references):
    """Observational diagnostic only. References never enter encodings/rankings."""
    root = Path(output_dir)
    manifest = json.loads((root / "06_similarity_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["output_sha256"].items():
        require(file_hash(_safe_member(root.resolve(), name)) == checksum, "Similarity output hash differs")
    ranks = [json.loads(line) for line in (root / "06_candidate_rankings.jsonl").read_text(encoding="utf-8").splitlines() if line]
    occurrences = [json.loads(line) for line in (root / "06_occurrence_status.jsonl").read_text(encoding="utf-8").splitlines() if line]
    index = {(row["occurrence_uid"], row["qid"]): row for row in ranks}
    report = []
    for occurrence in occurrences:
        if occurrence["term"] not in references:
            continue
        qid = references[occurrence["term"]]
        row = index.get((occurrence["occurrence_uid"], qid))
        report.append({"term": occurrence["term"], "occurrence_uid": occurrence["occurrence_uid"], "qid": qid,
                       "status": row["ranking_status"] if row else "NOT_IN_RETRIEVED_CANDIDATES", "rank_position": row["rank_position"] if row else None,
                       "score": row["score"] if row else None, "max_representation_ids": row["max_representation_ids"] if row else [],
                       "reference_role": "USER_KNOWN_REFERENCE_NOT_COMPLETE_GOLD"})
    return report
