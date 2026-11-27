"""Explicit replay of archived observations. Never synthesizes model vectors."""
import json
import math
from pathlib import Path

from ._runtime import (file_hash, frozen_runtime, guard_destination, protected_roots, read_jsonl,
                       safe_member, verify_outputs, write_json)

REPLAY_KIND = "ARCHIVED_SCORE_REPLAY_NO_INFERENCE"


def _candidate_content(row, expected_input_hash):
    """Exclude only the run-dependent 02 fingerprint, retaining all other fields.

    load_stage02 hashes provenance (paths/manifests) as well as data. A fresh
    offline replay necessarily has a new fingerprint. Never rewrite that hash
    in the actual candidates, or ignore other source-ref/eligibility fields.
    """
    result = dict(row)
    reference = dict(result["source_ref"])
    if reference.pop("input_content_sha256") != expected_input_hash:
        raise ValueError("Archived replay has inconsistent candidate source fingerprints")
    result["source_ref"] = reference
    if "source_candidate" in result:
        result["source_candidate"] = _candidate_content(result["source_candidate"], expected_input_hash)
    return result


def rebuild_wikidata_seed(stage02_dir, destination, *, code_dir):
    """Rebuild a v1 read-only seed from verified successful request snapshots."""
    root, destination = Path(stage02_dir).resolve(), Path(destination).resolve()
    guard_destination(destination, protected_roots(root, code_dir) | {root, Path(code_dir).resolve()})
    manifest = verify_outputs(root, "02_manifest.json")
    if manifest["schema_version"] != "terminlink-stage02-candidates-v1":
        raise ValueError("Unsupported Wikidata snapshot contract")
    responses = {}
    events = read_jsonl(root / "02_requests.jsonl")
    with frozen_runtime(code_dir) as runtime:
        request_key = runtime.api.JsonCache.request_key
    for event in events:
        if event["outcome"] != "OK" or not event.get("response_path"):
            raise ValueError("A complete replay seed requires successful recorded responses")
        path = safe_member(root, event["response_path"])
        if event["response_path"] not in manifest["output_sha256"]:
            raise ValueError("Request refers to an unmanifested snapshot")
        if file_hash(path) != event["response_sha256"].lower():
            raise ValueError("Wikidata snapshot digest differs")
        key = request_key(event["endpoint"], event["params"])
        if key != event["cache_key"]:
            raise ValueError("Wikidata recorded cache key differs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if key in responses and responses[key] != payload:
            raise ValueError("Conflicting archived Wikidata responses")
        responses[key] = payload
    value = {"version": 1, "data_provenance": "ARCHIVED_REAL_WIKIDATA_RESPONSES_OFFLINE_REPLAY",
             "source": {"directory": str(root), "manifest_sha256": file_hash(root / "02_manifest.json"),
                        "request_ledger_sha256": file_hash(root / "02_requests.jsonl"), "request_count": len(events)},
             "responses": responses}
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != value:
            raise ValueError("Refusing to overwrite a different replay seed")
    else:
        if not destination.parent.is_dir():
            raise ValueError("Replay seed parent must already exist")
        write_json(destination, value)
    return destination


def replay_scores(runtime, defterm_dir, output_dir, archive_dir, *, stage01_dir, stage02_dir):
    """Bind an archived ledger to the current exact input, then call frozen MAX.

    This deliberately writes an integration-specific replay manifest, never a
    production inference manifest. No encoder or embedding SQLite is involved.
    """
    archive, output = Path(archive_dir).resolve(), Path(output_dir).resolve()
    guard_destination(output, protected_roots(archive) | {archive})
    manifest = verify_outputs(archive, "06_similarity_manifest.json")
    if manifest.get("schema_version") != runtime.similarity.SCHEMA or manifest.get("model") != runtime.native.contract():
        raise ValueError("Archived replay model/contract differs from the authorized runtime")
    source = runtime.similarity.load_similarity_input(defterm_dir, stage01_dir=stage01_dir, stage02_dir=stage02_dir)
    plan = runtime.similarity.prepare_similarity(source)
    archived_candidates = read_jsonl(archive / "06_input_candidates.jsonl")
    current_input_hash = runtime.sense2.load_stage02(stage02_dir).content_sha256
    archived_input_hashes = {row["source_ref"]["input_content_sha256"] for row in archived_candidates}
    if len(archived_input_hashes) > 1:
        raise ValueError("Archived replay has inconsistent input fingerprints")
    archived_input_hash = next(iter(archived_input_hashes), None)
    if ([_candidate_content(row, current_input_hash) for row in source.tables["candidates"]] !=
            [_candidate_content(row, archived_input_hash) for row in archived_candidates]):
        raise ValueError("Archived replay input differs: candidates")
    for key, name in (("representations", "06_input_representations.jsonl"),
                      ("terms", "06_input_terms.jsonl"), ("associations", "06_input_associations.jsonl")):
        if source.tables[key] != read_jsonl(archive / name):
            raise ValueError("Archived replay input differs: " + key)
    if any(plan["counts"][key] != manifest["counts"][key] for key in plan["counts"]):
        raise ValueError("Archived replay input counts differ")
    scores = read_jsonl(archive / "06_similarity_scores.jsonl")
    by_pair = {row["pair_id"]: row for row in scores}
    if len(by_pair) != len(scores) or set(by_pair) != {row["pair_id"] for row in plan["pairs"]}:
        raise ValueError("Archived replay pairs differ")
    for pair in plan["pairs"]:
        row = by_pair[pair["pair_id"]]
        if any(row.get(key) != value for key, value in pair.items()):
            raise ValueError("Archived replay pair identity/content differs")
        score = row["score"]
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)
                                  or abs(score) > 1.0001 or row["scoring_status"] != "SCORED"):
            raise ValueError("Archived replay score is invalid")
    # MAX, ties and ordering remain exclusively the original H100 implementation.
    rankings, occurrences = runtime.similarity._rank(plan, scores)
    if rankings != read_jsonl(archive / "06_candidate_rankings.jsonl"):
        raise ValueError("Archived replay MAX/rankings differ")
    if occurrences != read_jsonl(archive / "06_occurrence_status.jsonl"):
        raise ValueError("Archived replay occurrence states differ")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Replay output must be new or empty")
    output.mkdir()
    names = {"scores": "06_replayed_scores.jsonl", "rankings": "06_replayed_rankings.jsonl",
             "occurrences": "06_replayed_occurrences.jsonl"}
    hashes = {names[key]: runtime.sense2_store._write_jsonl(output / names[key], rows)
              for key, rows in (("scores", scores), ("rankings", rankings), ("occurrences", occurrences))}
    source.verify()
    verify_outputs(archive, "06_similarity_manifest.json")
    replay = {"schema_version": "finterm-baseline-archived-score-replay-v1", "status": manifest["status"],
              "execution_kind": REPLAY_KIND, "real_embedding_inference_performed": False,
              "new_embedding_attempts": 0, "aggregation": "MAX_PER_OCCURRENCE_AND_CANDIDATE",
              "model": manifest["model"], "counts": manifest["counts"], "tables": names,
              "source_defterm_sha256": source.sha256, "source_crosschecks": source.crosschecks,
              "lineage_binding": {"field": "candidate.source_ref.input_content_sha256",
                  "current_stage02_sha256": current_input_hash, "archived_stage02_sha256": archived_input_hash,
                  "verification": "EXACT_CONTENT_EXCEPT_RUN_FINGERPRINT_AND_EXACT_PAIRS_AND_MAX"},
              "archive": {"directory": str(archive), "manifest_sha256": file_hash(archive / "06_similarity_manifest.json"),
                          "output_sha256": manifest["output_sha256"]},
              "output_sha256": hashes, "threshold_applied": False, "final_link_selected": False}
    path = output / "06_archived_replay_manifest.json"
    write_json(path, replay)
    return path
