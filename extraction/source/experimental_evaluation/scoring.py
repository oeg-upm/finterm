"""Offline scoring ONLY. This module alone reads the evaluation .key files."""

from pathlib import Path

from corpus_readers import CelexSegment, read_celex_segment

from .common import atomic_json, digest, read_json, utc_now
from .formats import FORMAT_POLICY, interpret
from .metrics import NORMALIZATION, aggregate, compare
from .provenance import verify_code


def read_reference(query, corpus_root):
    root = Path(corpus_root).resolve()
    txt, key = ((root / query[name]).resolve() for name in ("text_path", "key_path"))
    if root not in txt.parents or root not in key.parents:
        raise ValueError("Reference outside the declared original corpus")
    stem = "32013R0575_seg_" + str(query["segment_number"])
    reading = read_celex_segment(CelexSegment(stem, "32013R0575", str(query["segment_number"]), txt, key))
    if not reading.units or reading.units[0].example is None or any(d.severity == "error" for d in reading.diagnostics):
        raise ValueError(f"Reference withheld, not repaired: {reading.diagnostics}")
    if reading.units[0].text != query["text"]:
        raise ValueError("Original text changed before offline scoring")
    return {"terms": list(reading.units[0].example.terms),
            "sources": [{"path": str(s.path), "sha256": s.sha256, "size_bytes": s.size_bytes} for s in reading.sources],
            "diagnostics": [d.to_dict() for d in reading.diagnostics]}


def score_run(store, *, corpus_root, require_complete=True):
    """No inference, no repairs to files, no overwriting historical analyses.

    End-to-end views count syntax/generation failures and unreadable output as
    empty predictions, retaining all reference FN. Not-yet-run jobs are missing,
    NOT invented failures; partial analyses are explicitly not comparable runs.
    """
    manifest = store.manifest
    verify_code(Path(__file__).resolve().parent.parent, manifest["code"])
    if manifest["format_policy"] != FORMAT_POLICY or manifest["normalization"] != NORMALIZATION:
        raise ValueError("Scoring policy changed; version a new analysis explicitly")
    with store.writer():
        jobs = {}
        missing = []
        for condition in manifest["conditions"]:
            jobs[condition] = {}
            for query in manifest["queries"]:
                identifier = query["identifier"]
                raw = store.generation(condition, identifier)
                errors = store.failed(condition, identifier)
                if raw is None and not errors:
                    missing.append([condition, identifier])
                jobs[condition][identifier] = {"raw": raw, "errors": [read_json(p) for p in errors]}
        if missing and require_complete:
            raise ValueError(f"Frozen grid incomplete: {len(missing)} jobs not run; no comparable summary produced")
        references = {q["identifier"]: read_reference(q, corpus_root) for q in manifest["queries"]}
        output = {"schema": "silver-scoring-v1", "manifest_sha256": store.fingerprint,
                  "reference_kind": "automatic_silver_not_human_quality", "purpose": manifest["purpose"],
                  "bank_human_reviewed": manifest["bank"]["all_human_reviewed"],
                  "normalization": NORMALIZATION, "format_policy": FORMAT_POLICY,
                  "grid_complete": not missing, "missing_jobs": missing, "references": references,
                  "conditions": {}, "segments": {}}
        for condition, items in jobs.items():
            model = manifest["conditions"][condition][0]
            views = {name: [] for name in ("strict_end_to_end", "recovered_end_to_end", "strict_conditional", "recovered_conditional")}
            coverage = {name: 0 for name in ("generated", "execution_failed", "missing", "strict_valid", "recovered_valid", "recovery_applied", "unrecoverable_output")}
            details = {}
            for query in manifest["queries"]:
                identifier = query["identifier"]; job = items[identifier]; raw = job["raw"]
                if raw is None and not job["errors"]:
                    coverage["missing"] += 1
                    continue
                if raw is not None:
                    coverage["generated"] += 1
                    formats = interpret(raw["generation"]["response_text"],
                                        "native_json_array" if model == "dister" else "terms_object")
                    coverage["unrecoverable_output"] += not formats["recovered"]["accepted"]
                    coverage["recovery_applied"] += formats["recovered"]["accepted"] and not formats["strict"]["accepted"]
                else:
                    coverage["execution_failed"] += 1
                    formats = {name: {"accepted": False, "terms": None} for name in ("strict", "recovered")}
                entry = {"query_id": identifier, "prior_exposure": query["prior_exposure"],
                         "formats": formats, "execution_errors": job["errors"], "views": {}}
                for label in ("strict", "recovered"):
                    view = formats[label]; coverage[label + "_valid"] += view["accepted"]
                    compared = compare(view["terms"] if view["accepted"] else [],
                                       references[identifier]["terms"], query["text"])
                    entry["views"][label] = compared
                    views[label + "_end_to_end"].append(compared)
                    if view["accepted"]:
                        views[label + "_conditional"].append(compared)
                details[identifier] = entry
            output["segments"][condition] = details
            summary = {name: aggregate(rows) for name, rows in views.items()}
            for name, rows in views.items():
                summary[name]["by_type"] = {kind: aggregate([r["by_type"][kind] for r in rows])
                                              for kind in ("simple", "multiword")}
            output["conditions"][condition] = {"planned": len(manifest["queries"]), "coverage": coverage, "views": summary}
        output["common_interpretable_subsets"] = {}
        for label in ("strict", "recovered"):
            common = [q["identifier"] for q in manifest["queries"]
                      if all(q["identifier"] in output["segments"][c]
                             and output["segments"][c][q["identifier"]]["formats"][label]["accepted"]
                             for c in manifest["conditions"])]
            output["common_interpretable_subsets"][label] = {
                "identifiers": common, "conditions": {
                    c: aggregate([output["segments"][c][i]["views"][label] for i in common])
                    for c in manifest["conditions"]}}
        analysis_id = digest(output)
        path = store.root / "analyses" / (analysis_id + ".json")
        if path.exists():
            return path, read_json(path)
        output["created_at"] = utc_now()
        atomic_json(path, output)
        return path, output
