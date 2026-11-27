"""Model-free verification and honest GPU-evidence summaries for saved runs."""
import math
from pathlib import Path

from ._runtime import file_hash, frozen_runtime, read_jsonl, verify_outputs


def verify_run(run_dir, *, code_dir=None):
    root = Path(run_dir).resolve()
    manifest = verify_outputs(root, "integration_manifest.json")
    phases = {"01": root / "01_sentido1", "02": root / "02_candidatos", "03": root / "03_sentido2",
              "05": root / "05_defterm", "06": root / "06_similitud"}
    stages = {}
    for key, record in manifest["stages"].items():
        name = record["path"].replace("\\", "/").rsplit("/", 1)[-1]
        if file_hash(phases[key] / name) != record["sha256"]:
            raise ValueError("Recorded stage manifest changed: " + key)
        stages[key] = verify_outputs(phases[key], name)
    with frozen_runtime(code_dir) as runtime:
        runtime.stage02.load_stage01(phases["01"])
        runtime.sense2.load_stage02(phases["02"])
        runtime.defterm_store.load_sense2(phases["03"])
        source = runtime.similarity.load_similarity_input(phases["05"], stage01_dir=phases["01"], stage02_dir=phases["02"])
        m06 = stages["06"]
        names = m06.get("tables", {"scores": "06_similarity_scores.jsonl", "rankings": "06_candidate_rankings.jsonl"})
        scores = read_jsonl(phases["06"] / names["scores"])
        plan = runtime.similarity.prepare_similarity(source, identity=m06["model"])
        expected_pairs = {row["pair_id"]: row for row in plan["pairs"]}
        if len(scores) != len(expected_pairs) or {r["pair_id"] for r in scores} != set(expected_pairs):
            raise ValueError("Similarity pair inventory differs")
        for row in scores:
            if any(row.get(k) != v for k, v in expected_pairs[row["pair_id"]].items()):
                raise ValueError("Similarity pair identity differs")
            if row["score"] is not None and (type(row["score"]) not in (int, float) or
                    not math.isfinite(row["score"]) or abs(row["score"]) > 1.0001):
                raise ValueError("Invalid scalar cosine")
        rankings, _ = runtime.similarity._rank(plan, scores)
        if rankings != read_jsonl(phases["06"] / names["rankings"]):
            raise ValueError("Original MAX/rankings differ")
    controlled = manifest["execution_kind"] == "CONTROLLED_TEST_NO_REAL_MODEL"
    translations = read_jsonl(phases["03"] / "03_translations.jsonl")
    definitions = read_jsonl(phases["05"] / "05_defterm_definitions.jsonl")
    gates = read_jsonl(phases["05"] / "05_defterm_gate.jsonl")
    evidence = {
        "production_translations_available": sum(r["processing_status"] == "TRANSLATED" and
            r["request"]["generation_contract"]["backend_kind"] == "QWEN_CONTEXTUAL_V2_LOCAL" and
            "H100" in (r.get("generation") or {}).get("metadata", {}).get("device_name", "").upper() for r in translations),
        "production_sufficient_gates": sum(r.get("state") == "DONE" and
            r["request"]["recipe"]["backend_kind"] == "QWEN_DEFTERM_FEWSHOT_LOCAL" and
            r["processed"]["accepted_decision"] == "SUFFICIENT_EVIDENCE" for r in gates),
        "production_definitions_available": sum(r.get("state") == "DONE" and
            r["request"]["recipe"]["backend_kind"] == "QWEN_DEFTERM_FEWSHOT_LOCAL" and
            r["processed"]["available_for_later_evaluation"] for r in definitions),
        "production_scored_candidates": sum(r["score"] is not None for r in rankings)
            if m06["model"]["backend_kind"] == "QWEN3_EMBEDDING4B_SENTENCE_TRANSFORMERS" and
               manifest["execution_kind"] == "H100_ORIGINAL_MODELS_OR_CACHE" else 0,
        "new_translation_attempts": stages["03"]["execution"]["new_attempts"],
        "new_gate_attempts": stages["05"]["execution"]["gate"]["new_attempts"],
        "new_definition_attempts": stages["05"]["execution"]["definition"]["new_attempts"],
        "new_embedding_attempts": m06.get("execution", {}).get("new_embedding_attempts", 0),
        "new_embeddings_computed": m06.get("execution", {}).get("new_embeddings_cached", 0),
    }
    return {"integrity_verified": True, "mode": manifest["mode"], "status": manifest["status"],
            "execution_kind": manifest["execution_kind"], "controlled_test": controlled,
            "gpu_evidence": evidence, "inferences": manifest["inferences"],
            "full_gpu_validation_claimed": False, "final_link_selected": False}
