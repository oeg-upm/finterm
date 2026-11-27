"""Aggregate actual saved production evidence; never declares mocks GPU-valid."""
import json
from pathlib import Path

from .verification import verify_run


def validate_h100_acceptance(run_dirs):
    paths = [Path(path).resolve() for path in run_dirs]
    reports = [verify_run(path) for path in paths]
    production = [r for r in reports if r["mode"] == "full" and
                  r["execution_kind"] == "H100_ORIGINAL_MODELS_OR_CACHE" and not r["controlled_test"]]
    checks = {"at_least_one_full_production_run": bool(production),
              "all_inputs_are_full_production_runs": bool(reports) and len(production) == len(reports),
              "all_run_integrities_verified": bool(reports) and all(r["integrity_verified"] for r in reports)}
    requirements = (("real_contextual_translation", "production_translations_available", "new_translation_attempts"),
                    ("real_sufficient_gate", "production_sufficient_gates", "new_gate_attempts"),
                    ("real_definition", "production_definitions_available", "new_definition_attempts"),
                    ("real_embedding_rankings", "production_scored_candidates", "new_embeddings_computed"))
    for label, available, attempts in requirements:
        checks[label] = any(r["gpu_evidence"][available] > 0 and r["gpu_evidence"][attempts] > 0 for r in production)
    views = [json.loads((path / "top_candidates.json").read_text(encoding="utf-8")) for path in paths]
    for k in (3, 5):
        checks["top_" + str(k)] = any(len(view["candidates"]) == k for group in views for view in group.get(str(k), []))
    checks["no_final_qid_selection"] = all(not view["final_link_selected"] for group in views for rows in group.values() for view in rows)
    return {"schema_version": "finterm-full-h100-acceptance-v0.1", "full_runtime_acceptance_passed": all(checks.values()),
            "checks": checks, "runs": [{"directory": str(path), **report} for path, report in zip(paths, reports)],
            "all_requested_tasks_completed": bool(reports) and all(r["status"] == "COMPLETED" for r in reports),
            "semantic_fidelity_or_link_accuracy_validated": False, "final_link_selected": False}
