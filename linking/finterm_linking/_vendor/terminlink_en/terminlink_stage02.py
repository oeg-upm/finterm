"""Stage-02 contracts/trace with historical retrieval and explicit coverage extras."""
from __future__ import annotations

import argparse
import csv
import io
import importlib.metadata
import json
import os
import platform
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from terminlink_inputs import CONTRACT_VERSION, TermRecord, canonical, digest


VERSION = "terminlink-stage02-candidates-v1"
BASE = Path(__file__).resolve().parent
LIMIT_NAMES = ("ENTITY_SEARCH_LIMIT", "MAX_ENTITY_SEARCH_RESULTS_PER_QUERY", "MAX_CONTEXT_ENTITY_SEARCH_RESULTS_PER_QUERY",
               "MAX_CONTEXT_LITERAL_QUERIES_PER_TERM", "MAX_CONTEXT_EXTENSION_WORDS", "MAX_CONTEXT_LITERAL_QUERY_WORDS",
                "ENTITY_BATCH_SIZE", "SPARQL_SEARCH_BATCH_SIZE", "SPARQL_BATCH_SIZE", "MAX_RETRIES", "REQUEST_TIMEOUT_SECONDS", "REQUEST_DELAY_SECONDS",
                "RETRY_BASE_DELAY_SECONDS", "RETRY_SERVICE_DELAY_SECONDS", "MAX_RETRY_WAIT_SECONDS", "MAX_ADDITIONAL_QUERIES_PER_TERM",
                "MAX_ADDITIONAL_ENTITY_SEARCH_RESULTS_PER_QUERY", "MAX_SUPPLEMENTARY_SEARCH_RESULTS_PER_TERM")
TERM_STATUS_COLUMNS = ("document_id", "term_id", "term", "language", "term_provenance_json", "occurrence_count",
                       "candidate_count", "retrieval_status", "retrieval_complete", "absence_of_entity_proven", "error_details_json")
OCC_STATUS_COLUMNS = ("document_id", "term_id", "occurrence_uid", "occurrence_id", "term", "context",
                      "candidate_count", "retrieval_status", "absence_of_entity_proven")
REL_EXTRA_COLUMNS = ("candidate_id", "qid", "wikidata_url", "candidate_role", "validation_status",
                     "retrieval_status", "search_queries", "search_strategies", "data_provenance")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(values):
    lines = []
    for value in values:
        line = canonical(value)
        for separator in ("\u0085", "\u2028", "\u2029"):
            line = line.replace(separator, f"\\u{ord(separator):04x}")
        lines.append(line + "\n")
    return "".join(lines).encode("utf-8")


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\r\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Invalid CSV header")
        rows = list(reader)
        require(all(None not in row and all(value is not None for value in row.values()) for row in rows), "Malformed CSV")
        return rows


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relative_path(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name,
            "Unsafe artifact path")
    return path


def load_stage01(folder):
    folder = folder.resolve()
    manifest_path = folder / "01_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "COMPLETED" and not manifest.get("errors"), "Stage 01 must be complete without errors")
    require(manifest.get("linking_contract_version") == CONTRACT_VERSION, "Unsupported stage-01 linking contract")
    required = {"01_contextos.csv", "01_documents.jsonl", "01_terms.jsonl"}
    require(required <= set(manifest["output_sha256"]), "Stage-01 manifest omits contract files")
    fingerprints = {"01_manifest.json": digest(manifest_path.read_bytes())}
    for name, checksum in manifest["output_sha256"].items():
        relative_path(name)
        require(digest((folder / name).read_bytes()) == checksum, "Stage-01 file changed: " + name)
        fingerprints[name] = checksum
    documents = read_jsonl(folder / "01_documents.jsonl")
    terms = read_jsonl(folder / "01_terms.jsonl")
    contexts = read_csv(folder / "01_contextos.csv")
    with (folder / "01_contextos.csv").open(encoding="utf-8-sig", newline="") as stream:
        context_columns = tuple(csv.DictReader(stream).fieldnames)
    require(list(context_columns) == manifest["columns"], "Stage-01 CSV columns differ from its manifest")
    docs = {item["document_id"]: item for item in documents}
    term_index = {item["term_id"]: item for item in terms}
    require(len(docs) == len(documents) and len(term_index) == len(terms), "Duplicate document/term IDs")
    for doc in documents:
        require(doc["contract_version"] == CONTRACT_VERSION and digest(doc["text"].encode("utf-8")) == doc["decoded_text_sha256"], "Document text/hash differs")
    for term in terms:
        require(term["document_id"] in docs and term["contract_version"] == CONTRACT_VERSION, "Term/document reference differs")
        expected_id = "term_" + digest(canonical([term["document_id"], term["term"]]).encode("utf-8")).lower()
        require(term["term_id"] == expected_id and term["provenance"], "Term identity/provenance differs")
        require(all(item.get("provider") for item in term["provenance"]), "Term provider missing")
    seen = set()
    normalized = {item["document_id"]: (folder / item["normalized_file"]).read_text(encoding="utf-8")
                  for item in manifest["documents"]}
    for row in contexts:
        require(row["occurrence_uid"] not in seen, "Duplicate occurrence UID")
        seen.add(row["occurrence_uid"])
        term = term_index[row["term_id"]]
        require((row["document_id"], row["term"]) == (term["document_id"], term["term"]), "Context/term join differs")
        require(json.loads(row["term_provenance_json"]) == term["provenance"], "Context term provenance differs")
        a, b = int(row["context_target_start"]), int(row["context_target_end"])
        nstart, nend = int(row["normalized_start"]), int(row["normalized_end"])
        cstart, cend = int(row["context_normalized_start"]), int(row["context_normalized_end"])
        text = normalized[row["document_id"]]
        require(row["context"][a:b] == row["matched_text"] == text[nstart:nend]
                and text[cstart:cend] == row["context"], "Context offsets/text differ")
        identity = {"document_id": row["document_id"], "term": row["term"], "normalized_start": nstart, "normalized_end": nend}
        require(row["occurrence_uid"] == "occ_" + digest(canonical(identity).encode("utf-8")).lower(), "Occurrence UID derivation differs")
    return {"documents": documents, "terms": terms, "contexts": contexts, "context_columns": context_columns, "input_sha256": fingerprints,
            "input_mode": "stage01_contract", "input_directory": str(folder)}


def load_legacy(api, contexts_path, key_path, document_id=None):
    terms = api.read_terms(key_path)
    rows = api.read_context_rows(contexts_path, terms)
    with contexts_path.open(encoding="utf-8-sig", newline="") as stream:
        old_columns = tuple(csv.DictReader(stream).fieldnames)
    require("occurrence_uid" not in old_columns, "Use --stage01-dir for the generic contract; legacy mode expects the historical CSV")
    doc_id = document_id or contexts_path.stem
    source_sha = digest(key_path.read_bytes())
    records = [TermRecord(doc_id, term, "en", ({"provider": "legacy_key", "source_uri": str(key_path.resolve()),
                "source_sha256": source_sha, "record_number": i},)).to_dict() for i, term in enumerate(terms, 1)]
    by_term = {record["term"]: record for record in records}
    contexts = []
    for row in rows:
        record = by_term[row["term"]]
        identity = [doc_id, row["term"], row["occurrence_id"], row["context"], row.get("normalized_start")]
        contexts.append({**row, "document_id": doc_id, "term_id": record["term_id"], "language": "en",
                         "occurrence_uid": "legacy_occ_" + digest(canonical(identity).encode("utf-8")).lower(),
                         "term_provenance_json": canonical(record["provenance"])})
    return {"documents": [{"document_id": doc_id, "text": None, "source_kind": "legacy_contexts_only"}],
            "terms": records, "contexts": contexts, "input_mode": "legacy_explicit_files",
            "context_columns": tuple(contexts[0]) if contexts else (*old_columns, "document_id", "term_id", "language", "occurrence_uid", "term_provenance_json"),
            "input_sha256": {str(contexts_path.resolve()): digest(contexts_path.read_bytes()), str(key_path.resolve()): source_sha},
            "input_directory": str(contexts_path.resolve().parent)}


class RecordedLemmas:
    def __init__(self, path, terms):
        payload = path.read_bytes()
        value = json.loads(payload)
        require(value.get("schema_version") == "terminlink-recorded-lemmas-v1", "Unsupported lemma-map schema")
        self.entries = value["entries"]
        require(set(terms) <= set(self.entries), "Offline lemma map is incomplete; no silent lemma fallback")
        require(all(item is None or isinstance(item, str) and item.strip() for item in self.entries.values()), "Invalid recorded lemma")
        self.provenance = {"file": str(path.resolve()), "sha256": digest(payload),
                           "data_provenance": value.get("data_provenance", "RECORDED_VALUES"),
                           "note": value.get("note"), "recorded_source": value.get("source"),
                           "model_executed_in_this_run": False}

    def __call__(self, term):
        return self.entries[term]


def cache_and_client(api, cache_dir, offline, seed_cache=None, *, http_contact=None):
    path = cache_dir / "wikidata_responses.json"
    if seed_cache is not None:
        require(seed_cache.resolve() != path.resolve()
                and not (seed_cache.exists() and path.exists() and seed_cache.samefile(path)),
                "Read-only seed and mutable cache must be different files")
    before_sha = digest(path.read_bytes()) if path.exists() else None
    cache = api.JsonCache(path)
    origins = {key: "configured_cache" for key in cache.responses}
    provenance = {}
    fixture = False
    if path.exists():
        header = json.loads(path.read_text(encoding="utf-8"))
        fixture = "FIXTURE" in str(header.get("data_provenance", ""))
        provenance["configured_cache_data_provenance"] = header.get("data_provenance", "PREEXISTING_CACHE")
    if seed_cache:
        raw = seed_cache.read_bytes()
        source = json.loads(raw)
        require(source.get("version") == 1 and isinstance(source.get("responses"), dict), "Invalid read-only seed cache")
        for key, payload in source["responses"].items():
            if key in cache.responses:
                configured = cache.responses[key]
                configured_error = not isinstance(configured, dict) or "error" in configured
                seed_error = not isinstance(payload, dict) or "error" in payload
                if canonical(configured) != canonical(payload) and (configured_error or seed_error):
                    prefer_seed = configured_error and not seed_error
                    provenance.setdefault("seed_error_resolutions", []).append({
                        "cache_key": key, "configured_sha256": digest(canonical(configured).encode("utf-8")),
                        "seed_sha256": digest(canonical(payload).encode("utf-8")),
                        "choice": "seed_non_error_then_validate" if prefer_seed else "keep_configured_then_validate"})
                    if prefer_seed:
                        cache.responses[key] = payload
                        origins[key] = "seed_cache_over_error"
                    continue
                require(canonical(cache.responses[key]) == canonical(payload), "Seed/configured cache conflict")
            else:
                cache.responses[key] = payload
                origins[key] = "seed_cache"
        fixture = fixture or "FIXTURE" in str(source.get("data_provenance", ""))
        provenance.update(seed_cache_path=str(seed_cache.resolve()), seed_cache_sha256=digest(raw),
                          seed_data_provenance=source.get("data_provenance", "PREEXISTING_CACHE_UNKNOWN_RETRIEVAL_TIME"))
    require(offline or not fixture, "Controlled fixture responses cannot be mixed with an online run")
    provenance.update(cache_path=str(path.resolve()), before_sha256=before_sha,
                      controlled_fixture=fixture, offline=offline)

    class TracedClient(api.WikidataClient):
        def __init__(self):
            super().__init__(cache, http_contact=http_contact)
            self.events, self.snapshots = [], {}
            self.online_operations = 0
            self.response_times = {}

        def annotate_operation(self, operation):
            if self.events:
                self.events[-1]["operation"] = operation

        def snapshot_payload(self, payload):
            raw = canonical(payload).encode("utf-8")
            checksum = digest(raw)
            name = "responses/" + checksum + ".json"
            self.snapshots[name] = raw
            return {"response_sha256": checksum, "response_path": name}

        def record_attempt(self, detail, payload=None):
            super().record_attempt(detail, payload)
            event = dict(detail)
            if payload is not None:
                event.update(self.snapshot_payload(payload))
            self.events[-1]["attempts"].append(event)

        def get_json(self, endpoint, params):
            key = api.JsonCache.request_key(endpoint, params)
            event = {"request_number": len(self.events) + 1, "endpoint": endpoint, "params": dict(params),
                      "cache_key": key, "outcome": "ERROR", "response_sha256": None,
                      "response_path": None, "original_response_retrieved_utc": None, "attempts": []}
            self.events.append(event)
            try:
                payload = cache.get(endpoint, params)
                if payload is not None:
                    try:
                        api.validate_service_response(endpoint, params, payload)
                    except ValueError as error:
                        event["invalid_cached_response"] = {**self.snapshot_payload(payload), "error_message": str(error)}
                        if not offline:
                            payload = None
                if payload is None:
                    if offline:
                        event["response_source"] = "offline_cache_miss"
                        raise RuntimeError("OFFLINE_CACHE_MISS: " + key)
                    self.online_operations += 1
                    event["response_source"] = "online_after_invalid_cache" if "invalid_cached_response" in event else "online_wikidata"
                    payload = super().get_json(endpoint, params)
                    event["original_response_retrieved_utc"] = datetime.now(timezone.utc).isoformat()
                    self.response_times[key] = event["original_response_retrieved_utc"]
                    origins[key] = "online_cache_this_run"
                else:
                    event["response_source"] = origins.get(key, "configured_cache")
                    event["original_response_retrieved_utc"] = self.response_times.get(key)
                event.update(self.snapshot_payload(payload))
                api.validate_service_response(endpoint, params, payload)
                event["outcome"] = "OK"
                return payload
            except Exception as exception:
                event.update(error_type=type(exception).__name__, error_message=str(exception))
                if isinstance(exception, (ValueError, RuntimeError)):
                    raise
                raise RuntimeError("Wikidata transport/cache error: " + type(exception).__name__) from exception

    return cache, TracedClient(), provenance


def collect(api, terms, contexts, resolver, client, *, search_policy="historical"):
    """Keep historical requests intact; append explicitly requested bounded coverage."""
    require(search_policy in {"historical", "coverage"}, "Unknown search policy")
    context_queries = api.generate_context_literal_queries(terms, contexts, lemma_resolver=resolver)
    api.validate_context_literal_queries(terms, context_queries)
    candidates = {term: {} for term in terms}
    specs = api.search_specs(terms, context_queries, lemma_resolver=resolver)
    entity_audits = api.retrieve_entity_search_candidates(client, specs, candidates)
    sparql_audits = api.retrieve_exact_sparql_candidates(client, specs, candidates)
    primary_qids = sorted({qid for group in candidates.values() for qid in group}, key=api.qid_sort_key)
    additional_specs, variant_evidence, supplementary_specs, supplementary_audits = [], [], [], {}
    if search_policy == "coverage":
        additional_specs, variant_evidence = api.additional_search_specs(terms, contexts, resolver)
        # Separate calls preserve ALL original SPARQL query strings/cache keys.
        entity_audits.update(api.retrieve_entity_search_candidates(client, additional_specs, candidates))
        sparql_audits.update(api.retrieve_exact_sparql_candidates(client, additional_specs, candidates))
        supplementary_specs = api.supplementary_cirrus_specs(terms, resolver)
        supplementary_audits = api.retrieve_supplementary_candidates(client, supplementary_specs, candidates)
    else:
        require(search_policy == "historical", "Unknown search policy")
    specs = [*specs, *additional_specs]
    qids = sorted({qid for group in candidates.values() for qid in group}, key=api.qid_sort_key)
    cohorts = [primary_qids, sorted(set(qids) - set(primary_qids), key=api.qid_sort_key)]
    english, multilingual, relationships, labels = {}, {}, {}, {}
    queried_related = set()
    for cohort in cohorts:
        # New candidates must not invalidate cached metadata batches for the
        # original inventory. Fetch only the additional cohort separately.
        english.update(api.fetch_entities(client, cohort, "labels|aliases|descriptions", languages="en"))
        multilingual.update(api.fetch_entities(client, cohort, "descriptions"))
        relationships.update(api.fetch_relationships(client, cohort))
        related_qids = sorted({related for qid in cohort for values in relationships.get(qid, {}).values()
                              for related in values} - queried_related, key=api.qid_sort_key)
        related = api.fetch_entities(client, related_qids, "labels", languages="en")
        labels.update({qid: api.english_value(entity, "labels") for qid, entity in related.items()})
        queried_related.update(related_qids)
    rows = api.build_rows(terms, candidates, english, multilingual, relationships, labels)
    api.validate_rows(rows, terms, context_queries, lemma_resolver=resolver, additional_specs=[*additional_specs, *supplementary_specs])
    return {"rows": rows, "context_queries": context_queries, "specs": specs, "entity_audits": entity_audits,
            "sparql_audits": sparql_audits, "raw_candidates": candidates, "english": english,
             "multilingual": multilingual, "relationships": relationships, "variant_evidence": variant_evidence,
             "supplementary_specs": supplementary_specs, "supplementary_audits": supplementary_audits,
             "metadata_cohorts": {"historical": cohorts[0], "additional_only": cohorts[1]}}


def term_errors(api, term, collected, client):
    problems = []
    for spec in collected["specs"]:
        if spec[0] != term:
            continue
        audit = collected["entity_audits"][spec]
        if not (audit.completed or audit.stopped_by_limit):
            problems.append({"kind": "ENTITY_SEARCH_INCOMPLETE", "query": spec[1], "strategy": spec[2]})
        if not collected["sparql_audits"][spec].completed:
            problems.append({"kind": "SPARQL_SEARCH_INCOMPLETE", "query": spec[1], "strategy": spec[2]})
    for spec, audit in collected["supplementary_audits"].items():
        if spec[0] == term and not (audit.completed or audit.stopped_by_limit):
            problems.append({"kind": "SUPPLEMENTARY_SEARCH_INCOMPLETE", "query": spec[1], "strategy": spec[2]})
    qids = set(collected["raw_candidates"][term])
    for qid in sorted(qids, key=api.qid_sort_key):
        if qid not in collected["english"] or qid not in collected["multilingual"]:
            problems.append({"kind": "CANDIDATE_METADATA_INCOMPLETE", "qid": qid})
    affected_ids = set(qids)
    for qid in qids:
        for values in collected["relationships"].get(qid, {}).values():
            affected_ids.update(values)
    for event in client.events:
        if event["outcome"] == "OK":
            continue
        ids = set(event["params"].get("ids", "").split("|"))
        query = event["params"].get("query", "")
        if affected_ids & ids or any(("wd:" + qid + " ") in query or ("wd:" + qid + " }") in query for qid in qids):
            problems.append({"kind": "RELATED_METADATA_REQUEST_ERROR", "request_number": event["request_number"]})
    return problems


def publish(folder, payloads):
    for name in payloads:
        relative_path(name)
        require(not (folder / name).exists(), "Refusing to overwrite prior run artifact")
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(payloads.items()):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)


def run_stage02(api, input_dir, output_dir, cache_dir, *, offline=True, lemma_map=None, seed_cache=None,
                 legacy_key=None, legacy_contexts=None, document_id=None, http_contact=None,
                 search_policy="historical", term_filter=None):
    output_dir, cache_dir = output_dir.resolve(), cache_dir.resolve()
    require(output_dir != BASE and cache_dir != BASE, "Use separate run/cache directories, not the historical project root")
    require(output_dir != cache_dir and output_dir not in cache_dir.parents and cache_dir not in output_dir.parents,
            "Output and mutable cache directories must be separate")
    require(not output_dir.exists() or output_dir.is_dir() and not any(output_dir.iterdir()), "Output directory must be new or empty")
    if input_dir is not None:
        require(legacy_key is None and legacy_contexts is None, "Choose contract input or explicit legacy files")
        require(input_dir.resolve() != output_dir and input_dir.resolve() not in output_dir.parents, "Do not write into stage-01 inputs")
        require(input_dir.resolve() != cache_dir and input_dir.resolve() not in cache_dir.parents, "Do not cache inside stage-01 inputs")
        inputs = load_stage01(input_dir)
    else:
        require(legacy_key is not None and legacy_contexts is not None, "Explicit legacy files are required")
        inputs = load_legacy(api, legacy_contexts, legacy_key, document_id)
    require(search_policy in {"historical", "coverage"}, "Unknown search policy")
    if term_filter is not None:
        require(not isinstance(term_filter, (str, bytes)), "Term filter must be a list of exact labels")
        requested = list(term_filter)
        require(requested and all(isinstance(term, str) and term for term in requested), "Term filter must contain exact supplied labels")
        require(len(set(requested)) == len(requested), "Duplicate pilot term label")
        available = {record["term"] for record in inputs["terms"]}
        require(set(requested) <= available, "Term filter contains labels absent from stage 01: " + repr(sorted(set(requested) - available)))
        inputs = {**inputs, "terms": [item for item in inputs["terms"] if item["term"] in requested],
                  "contexts": [item for item in inputs["contexts"] if item["term"] in requested],
                  "term_selection": {"requested_exact_labels": requested, "available_labels": len(available),
                                     "order": "ORIGINAL_STAGE01_TERM_ORDER", "stage01_files_modified": False}}
    http_contact = http_contact or os.environ.get("FINTERM_WIKIMEDIA_CONTACT")
    active_labels = {row["term"] for row in inputs["contexts"]}
    terms = list(dict.fromkeys(record["term"] for record in inputs["terms"] if record["term"] in active_labels))
    identity = api.http_user_agent(http_contact, required=not offline and bool(terms))
    source_lemma_map = lemma_map or cache_dir / "02_lemmas.json"
    if not terms:
        resolver = {}.__getitem__
        morphology = {"data_provenance": "NO_ACTIVE_TERMS_NO_MORPHOLOGY_NEEDED", "model_executed_in_this_run": False}
    elif source_lemma_map.is_file():
        resolver = RecordedLemmas(source_lemma_map, terms)
        morphology = resolver.provenance
    else:
        require(not offline, "Offline mode requires --lemma-map or cache-dir/02_lemmas.json; no NLP model or heuristic fallback is run")
        # Original spaCy lemmatization, before any Wikidata request. Not executed in offline tests.
        lemmas = {term: api.conservative_lemma(term) for term in terms}
        resolver = lemmas.__getitem__
        nlp = api.english_morphology_pipeline()
        morphology = {"data_provenance": "ORIGINAL_CONSERVATIVE_SPACY_LEMMATIZATION", "model_executed_in_this_run": True,
                      "spacy_version": importlib.metadata.version("spacy"),
                      "model_name": "en_core_web_sm", "model_metadata": {key: nlp.meta.get(key) for key in ("lang", "name", "version")}}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache, client, cache_provenance = cache_and_client(api, cache_dir, offline, seed_cache, http_contact=http_contact)
    try:
        collected = collect(api, terms, inputs["contexts"], resolver, client, search_policy=search_policy)
    finally:
        if not offline:
            cache.flush()
    rows = collected["rows"]
    candidates_by_term = defaultdict(list)
    for row in rows:
        candidates_by_term[row["term"]].append(row)
    contexts_by_term_id = defaultdict(list)
    for row in inputs["contexts"]:
        contexts_by_term_id[row["term_id"]].append(row)
    states, state_by_term_id, errors_by_label = [], {}, {}
    for term in terms:
        errors_by_label[term] = term_errors(api, term, collected, client)
    for record in inputs["terms"]:
        occurrences = contexts_by_term_id[record["term_id"]]
        count = len(candidates_by_term[record["term"]]) if occurrences else 0
        problems = errors_by_label.get(record["term"], []) if occurrences else []
        status = ("TERM_NOT_FOUND_IN_TEXT" if not occurrences else "CANDIDATES_AVAILABLE_WITH_ERRORS" if count and problems
                  else "RETRIEVAL_ERROR" if problems else "CANDIDATES_AVAILABLE" if count else "NO_CANDIDATES")
        state = {"document_id": record["document_id"], "term_id": record["term_id"], "term": record["term"],
                 "language": record["language"], "term_provenance_json": canonical(record["provenance"]),
                 "occurrence_count": len(occurrences), "candidate_count": count, "retrieval_status": status,
                 "retrieval_complete": bool(occurrences) and not problems, "absence_of_entity_proven": False,
                 "error_details_json": canonical(problems)}
        states.append(state)
        state_by_term_id[record["term_id"]] = state
    associations, occurrence_states = [], []
    origin = "CONTROLLED_FIXTURE_NOT_LIVE_WIKIDATA" if cache_provenance["controlled_fixture"] else "WIKIDATA_RESPONSES_UNVALIDATED"
    for context in inputs["contexts"]:
        state = state_by_term_id[context["term_id"]]
        occurrence_states.append({key: context[key] for key in ("document_id", "term_id", "occurrence_uid", "occurrence_id", "term", "context")}
                                 | {"candidate_count": state["candidate_count"], "retrieval_status": state["retrieval_status"], "absence_of_entity_proven": False})
        for candidate in candidates_by_term[context["term"]]:
            candidate_id = "cand_" + digest(canonical([candidate["term"], candidate["qid"]]).encode("utf-8")).lower()
            associations.append({**context, "candidate_id": candidate_id, "qid": candidate["qid"],
                "wikidata_url": "https://www.wikidata.org/entity/" + candidate["qid"],
                "candidate_role": "RETRIEVED_CANDIDATE", "validation_status": "NOT_VALIDATED_NOT_MANUALLY_VERIFIED",
                "retrieval_status": state["retrieval_status"], "search_queries": candidate["search_queries"],
                "search_strategies": candidate["search_strategies"], "data_provenance": origin})
    require(len(associations) == len({(row["occurrence_uid"], row["qid"]) for row in associations}), "Duplicate occurrence/candidate association")
    query_audit = {"context_query_policy": "UNCHANGED_ALL_LITERAL_MATCHES_IN_SOURCE_CONTEXT_MAX_2_PER_UNIQUE_TERM",
                    "search_policy": search_policy, "additional_query_evidence": collected["variant_evidence"],
                    "metadata_cohorts": collected["metadata_cohorts"],
                    "supplementary_searches": [{"term": spec[0], "query": spec[1], "strategy": spec[2],
                        "audit": {**asdict(audit), "accepted_qids": sorted(audit.accepted_qids, key=api.qid_sort_key)}}
                        for spec, audit in collected["supplementary_audits"].items()],
                    "bibliographic_diagnostics": {"direct_instance_qid": api.SCHOLARLY_ARTICLE_QID,
                        "not_a_financial_relevance_classification": True, "candidates_removed_by_this_diagnostic": 0,
                        "terms": {term: {"total_candidates": len(candidates_by_term[term]),
                            "direct_scholarly_articles": sum(api.SCHOLARLY_ARTICLE_QID in collected["relationships"].get(row["qid"], {}).get("p31", set())
                                                            for row in candidates_by_term[term])} for term in terms}},
                   "context_queries": {term: [asdict(item) for item in collected["context_queries"][term]] for term in terms},
                   "search_specs": [{"term": spec[0], "query": spec[1], "strategy": spec[2],
                         "entity_search": {**asdict(collected["entity_audits"][spec]), "accepted_qids": sorted(collected["entity_audits"][spec].accepted_qids, key=api.qid_sort_key)},
                         "sparql": {"completed": collected["sparql_audits"][spec].completed, "qids": sorted(collected["sparql_audits"][spec].qids, key=api.qid_sort_key)}} for spec in collected["specs"]],
                   "filtered_disambiguation_pairs": [{"term": term, "qid": qid} for term in terms for qid in collected["raw_candidates"][term]
                         if api.is_disambiguation(qid, collected["english"], collected["relationships"])]}
    context_columns = inputs["context_columns"]
    payloads = {"02_wikidata_candidatos.csv": csv_bytes(rows, api.OUTPUT_COLUMNS),
                "02_candidatos_relaciones.csv": csv_bytes(associations, (*context_columns, *REL_EXTRA_COLUMNS)),
                "02_terminos_estado.csv": csv_bytes(states, TERM_STATUS_COLUMNS),
                "02_ocurrencias_estado.csv": csv_bytes(occurrence_states, OCC_STATUS_COLUMNS),
                "02_query_audit.json": json_bytes(query_audit),
                 "02_requests.jsonl": jsonl_bytes(client.events),
                "02_lemmas.json": json_bytes({"schema_version": "terminlink-recorded-lemmas-v1", "language": "en",
                      "data_provenance": morphology["data_provenance"], "entries": {term: resolver(term) for term in terms}, "source": morphology}),
                 "02_input_documents.jsonl": jsonl_bytes(inputs["documents"]),
                 "02_input_terms.jsonl": jsonl_bytes(inputs["terms"])}
    payloads.update(client.snapshots)
    errors = [event for event in client.events if event["outcome"] != "OK"]
    retrieval_errors = [{"term": term, **problem} for term, problems in errors_by_label.items() for problem in problems]
    cache_path = cache_dir / "wikidata_responses.json"
    cache_provenance["after_sha256"] = digest(cache_path.read_bytes()) if cache_path.exists() else None
    if seed_cache:
        require(digest(seed_cache.read_bytes()) == cache_provenance["seed_cache_sha256"], "Read-only seed cache was changed")
    manifest = {"schema_version": VERSION, "status": "PARTIAL_OR_FAILED" if errors or retrieval_errors else "COMPLETED",
                "execution_scope": "CANDIDATE_RETRIEVAL_ONLY_NO_LINK_SELECTION", "input": inputs | {"documents": None, "terms": None, "contexts": None},
                "input_sha256": inputs["input_sha256"], "offline": offline, "online_operations": client.online_operations,
                "data_provenance": origin, "morphology": morphology, "cache": cache_provenance,
                 "search_parameters": {name: getattr(api, name) for name in LIMIT_NAMES}, "search_language": "en",
                 "search_policy": search_policy, "http_identification": {"user_agent": identity, "contact_supplied": bool(http_contact)},
                "warnings": ["English retrieval/morphology inherited; no equivalent multilingual/domain performance is asserted"],
                "counts": {"documents": len(inputs["documents"]), "term_records": len(inputs["terms"]),
                           "unique_term_labels_searched": len(terms), "occurrences": len(inputs["contexts"]),
                           "term_qid_pairs": len(rows), "occurrence_candidate_relations": len(associations),
                           "unique_qids": len({row["qid"] for row in rows}), "requests": len(client.events),
                            "request_errors": len(errors), "retrieval_errors": len(retrieval_errors),
                            "http_attempts": sum(attempt["attempt"] > 0 for event in client.events for attempt in event["attempts"]),
                            "retry_wait_seconds": sum(attempt.get("wait_seconds", 0) for event in client.events for attempt in event["attempts"]),
                            "term_states": dict(Counter(row["retrieval_status"] for row in states))},
                 "errors": errors, "retrieval_errors": retrieval_errors, "output_sha256": {name: digest(data) for name, data in sorted(payloads.items())},
                "component_sha256": {"02_wikidata_candidatos.py": digest(Path(api.__file__).read_bytes()),
                    "terminlink_stage02.py": digest(Path(__file__).read_bytes()), "terminlink_inputs.py": digest((BASE / "terminlink_inputs.py").read_bytes())},
                "python_version": platform.python_version(), "candidates_are_proposed_or_gold_links": False,
                "embedding_or_validator_models_executed": False,
                "morphology_model_executed": morphology["model_executed_in_this_run"]}
    payloads["02_manifest.json"] = json_bytes(manifest)
    publish(output_dir, payloads)
    return manifest


def run_cli(api, argv=None):
    parser = argparse.ArgumentParser(description="Stage 02: reuse Wikidata retrieval for generic stage-01 inputs; offline by default")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage01-dir", type=Path)
    source.add_argument("--legacy-contexts", type=Path)
    parser.add_argument("--legacy-key", type=Path)
    parser.add_argument("--document-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed-cache", type=Path, help="Read-only version-1 historical cache, never overwritten")
    parser.add_argument("--lemma-map", type=Path, help="Recorded/controlled lemmas for offline execution, without running spaCy")
    parser.add_argument("--http-contact", help="Authorized public email/mailto/HTTPS contact; otherwise FINTERM_WIKIMEDIA_CONTACT")
    parser.add_argument("--search-policy", choices=("historical", "coverage"), default="historical",
                        help="coverage adds bounded justified surface/acronym and supplementary non-article retrieval; preserves original candidates")
    parser.add_argument("--term", action="append", dest="term_filter", help="Repeat exact stage-01 labels for a small pilot; never rewrites stage 01")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--online", action="store_true", help="Explicitly permit real Wikidata requests; not used in offline acceptance")
    args = parser.parse_args(argv)
    try:
        require(not args.legacy_key or args.legacy_contexts, "--legacy-key requires --legacy-contexts")
        result = run_stage02(api, args.stage01_dir, args.output_dir, args.cache_dir, offline=not args.online,
                              lemma_map=args.lemma_map, seed_cache=args.seed_cache, legacy_key=args.legacy_key,
                              legacy_contexts=args.legacy_contexts, document_id=args.document_id,
                              http_contact=args.http_contact, search_policy=args.search_policy, term_filter=args.term_filter)
    except (OSError, ValueError, ImportError) as exception:
        print(json.dumps({"status": "FAILED_BEFORE_RUN", "error_type": type(exception).__name__, "message": str(exception)}, ensure_ascii=True), file=__import__('sys').stderr)
        return 1
    print(json.dumps({"status": result["status"], "offline": result["offline"], **result["counts"],
                      "online_operations": result["online_operations"], "links_selected": 0}, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 1
