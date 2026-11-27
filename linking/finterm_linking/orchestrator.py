"""Extends the existing BASELINE composition; all NLP/LLM/ranking stays frozen."""
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .config import LinkerConfig
from ._runtime import (file_hash, frozen_runtime, guard_destination, protected_roots,
                       read_jsonl, verify_outputs, write_json)
from .replay import REPLAY_KIND, replay_scores


@dataclass(frozen=True)
class LinkResult:
    run_dir: Path
    mode: str
    top_k: int
    status: str
    execution_kind: str
    real_embedding_inference_performed: bool
    occurrences: list[dict]
    terms: list[dict]
    candidates: list[dict]
    representations: list[dict]
    scores: list[dict]
    errors: list[dict]
    trace: dict

    @property
    def rankings(self):
        return [row for occurrence in self.occurrences for row in occurrence["rankings"]]

    def candidate_views(self, *, top_k=None):
        """Presentation join only: no new scores, filtering rule or ranking."""
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        candidates = {row["candidate_id"]: row for row in self.candidates}
        representations = {row["representation_id"]: row for row in self.representations}
        output = []
        for occurrence in self.occurrences:
            rows = occurrence["rankings"]
            if top_k is not None:
                rows = [row for row in rows if row["score"] is not None][:top_k]
            presented = []
            for row in rows:
                candidate = candidates[row["candidate_id"]]
                reps = [representations[rid] for rid in row["max_representation_ids"]]
                presented.append({**row, "context": occurrence["context"], "label_en": candidate["original"]["label_en"],
                    "description_en": candidate["original"]["description_en"],
                    "representation_text_en": reps[0]["text_en"] if reps else None,
                    "max_representations": reps, "wikidata_url": "https://www.wikidata.org/entity/" + row["qid"],
                    "validated_final_link": False})
            output.append({"term": occurrence["term"], "context": occurrence["context"],
                "document_id": occurrence["document_id"], "occurrence_uid": occurrence["occurrence_uid"],
                "candidates": presented, "scored_candidates_available": sum(r["score"] is not None for r in occurrence["rankings"]),
                "unscored_candidates_preserved": sum(r["score"] is None for r in occurrence["rankings"]),
                "top_k": top_k, "final_link_selected": False})
        return output

    def top(self, k=5):
        return self.candidate_views(top_k=k)

    def to_dict(self):
        return {"run_dir": str(self.run_dir), "mode": self.mode, "top_k": self.top_k, "status": self.status,
                "execution_kind": self.execution_kind, "real_embedding_inference_performed": self.real_embedding_inference_performed,
                "occurrences": self.occurrences, "terms": self.terms, "candidates": self.candidates,
                "representations": self.representations, "scores": self.scores, "errors": self.errors, "trace": self.trace,
                "top_candidates": self.top(self.top_k), "threshold_applied": False, "final_link_selected": False}


class Linker:
    def __init__(self, config=None):
        self.config = config or LinkerConfig()
        if not isinstance(self.config, LinkerConfig):
            raise TypeError("config must be LinkerConfig")

    def link(self, text: str, terms: list[str], *, mode=None, top_k=5, document_id=None):
        if mode is not None and mode != self.config.mode:
            return Linker(replace(self.config, mode=mode)).link(text, terms, top_k=top_k, document_id=document_id)
        if not isinstance(text, str) or type(terms) is not list:
            raise TypeError("Supply original text as str and terms as list[str]")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        c = self.config
        roots = protected_roots(c.code_dir, c.lemma_map, c.wikidata_seed_cache, c.archived_similarity_dir)
        roots.update(path for path in (c.code_dir, c.archived_similarity_dir, c.qwen14_hub_cache, c.embedding_resource_dir) if path)
        paths_to_write = (c.work_dir, c.work_dir / "runs", c.work_dir / ".pipeline.lock", c.work_dir / "cache/wikidata",
                          c.work_dir / "cache/wikidata/wikidata_responses.json", c.translation_cache or c.work_dir / "cache/translations.sqlite",
                          c.defterm_cache or c.work_dir / "cache/defterm", c.embedding_cache or c.work_dir / "cache/embeddings.sqlite",
                          c.gpu_lock or c.work_dir / ".gpu.lock")
        for path in paths_to_write:
            guard_destination(path, roots)
        with frozen_runtime(c.code_dir) as runtime:
            from terminlink_inputs import DocumentInput, supplied_terms
            document = DocumentInput.from_text(text, document_id, "en")
            supplied_terms(document, terms=terms, origin=c.term_origin)
            c.validate_recipes(runtime)
            c.work_dir.mkdir(parents=True, exist_ok=True)
            with runtime.native.exclusive_lock(c.work_dir / ".pipeline.lock"):
                return self._run(runtime, text, terms, document_id, top_k)

    def _run(self, runtime, text, terms, document_id, top_k):
        c = self.config
        runs = c.work_dir / "runs"
        runs.mkdir(exist_ok=True)
        run = runs / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex)
        run.mkdir()
        paths = {key: run / name for key, name in (("01", "01_sentido1"), ("02", "02_candidatos"),
                 ("03", "03_sentido2"), ("05", "05_defterm"), ("06", "06_similitud"))}
        paths["run"] = run
        enrichment = run / "enrichment_cache" if c.mode == "quick" else c.work_dir / "cache"
        enrichment.mkdir(parents=True, exist_ok=True)
        tcache = enrichment / "translations.sqlite" if c.mode == "quick" else c.translation_cache or enrichment / "translations.sqlite"
        dcache = enrichment / "defterm" if c.mode == "quick" else c.defterm_cache or enrichment / "defterm"
        for cache in (tcache, dcache):
            cache.parent.mkdir(parents=True, exist_ok=True)
        stage = "01"
        try:
            runtime.stage01.run_stage01(paths["01"], text=text, terms=terms, document_id=document_id,
                                       language="en", term_origin=c.term_origin)
            runtime.stage02.load_stage01(paths["01"])
            stage = "02"
            runtime.stage02.run_stage02(runtime.api, paths["01"], paths["02"], c.work_dir / "cache/wikidata",
                offline=c.offline, lemma_map=c.lemma_map, seed_cache=c.wikidata_seed_cache,
                http_contact=c.http_contact, search_policy=c.search_policy)
            runtime.sense2.load_stage02(paths["02"])
            stage = "03"
            if c.mode == "full":
                self._execute_gpu_stage(runtime, paths, "03", tcache)
            else:
                if tcache.exists():
                    raise ValueError("quick enrichment caches must be new and empty")
                m03 = runtime.sense2_store.run_stage03(paths["02"], paths["03"], tcache,
                    policy=c.translation_policy, max_translations=0, adapter=None)
                self._assert_no_enrichment(m03, "03")
            runtime.defterm_store.load_sense2(paths["03"])
            stage = "05"
            if c.mode == "full":
                self._execute_gpu_stage(runtime, paths, "05", dcache)
            else:
                if dcache.exists():
                    raise ValueError("quick enrichment caches must be new and empty")
                m05 = runtime.defterm_store.run_defterm(paths["03"], paths["05"], dcache,
                    max_gate_calls=0, max_definition_calls=0, backend=None, preselection_policy=c.preselection_policy)
                self._assert_no_enrichment(m05, "05")
            stage = "06"
            manifest_path = self._execute_similarity(runtime, paths)
            result = self._collect(runtime, paths, manifest_path, top_k)
            write_json(run / "result.json", result.to_dict())
            write_json(run / "top_candidates.json", {str(k): result.top(k) for k in sorted({1, 3, 5, top_k})})
            self._write_csv(result)
            write_json(run / "integration_manifest.json", {
                "schema_version": "finterm-linking-integration-v0.1", "mode": c.mode, "status": result.status,
                "execution_kind": result.execution_kind, "config": c.to_dict(), "budgets": c.budgets(),
                "source_export": runtime.integrity, "integration_component_sha256": result.trace["integration_component_sha256"],
                "enrichment_cache_policy": "NEW_EMPTY_PER_QUICK_RUN" if c.mode == "quick" else "PERSISTENT_EXACT_REQUEST_IDENTITY",
                "stages": result.trace["stages"], "coverage": result.trace["coverage"],
                "inferences": result.trace["inferences"], "source_stage_status": result.trace["source_stage_status"],
                "full_gpu_validation": "NOT_CLAIMED_REQUIRES_EXTERNAL_ACCEPTANCE", "errors": result.errors,
                "output_sha256": {name: file_hash(run / name) for name in ("result.json", "top_candidates.json", "rankings.csv")},
                "threshold_applied": False, "final_link_selected": False})
            return result
        except BaseException as error:
            write_json(run / "integration_failure.json", {"schema_version": "finterm-linking-failure-v0.1",
                "mode": c.mode, "status": "FAILED" if isinstance(error, Exception) else "INTERRUPTED",
                "stage": stage, "error_type": type(error).__name__, "run_directory": str(run),
                "source_export": runtime.integrity, "final_link_selected": False})
            raise

    @staticmethod
    def _assert_no_enrichment(manifest, phase):
        execution = manifest["execution"]
        records = [execution] if phase == "03" else [execution["gate"], execution["definition"]]
        if any(row[key] != 0 for row in records for key in ("new_attempts", "cache_hits", "inflight_not_retried")):
            raise ValueError("Strict quick detected reused or newly generated enrichment")

    def _execute_similarity(self, runtime, paths):
        c = self.config
        if c.similarity_mode == "archived_replay":
            return replay_scores(runtime, paths["05"], paths["06"], c.archived_similarity_dir,
                                 stage01_dir=paths["01"], stage02_dir=paths["02"])
        if c.similarity_mode == "prepare":
            runtime.similarity.run_similarity(paths["05"], paths["06"], paths["run"] / "enrichment_cache/prepared_embeddings.sqlite",
                max_new_embeddings=0, stage01_dir=paths["01"], stage02_dir=paths["02"])
            return paths["06"] / "06_similarity_manifest.json"
        return self._execute_gpu_stage(runtime, paths, "06", c.embedding_cache or c.work_dir / "cache/embeddings.sqlite")

    def _execute_gpu_stage(self, runtime, paths, phase, cache):
        c = self.config
        cache.parent.mkdir(parents=True, exist_ok=True)
        lock = c.gpu_lock or c.work_dir / ".gpu.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        job = paths["run"] / ("gpu_job_" + phase + ".json")
        write_json(job, {"schema_version": "finterm-gpu-stage-job-v0.1", "stage": phase, "mode": c.mode,
            "code_dir": str(c.code_dir), "input_dir": str(paths[{"03": "02", "05": "03", "06": "05"}[phase]]),
            "output_dir": str(paths[phase]), "cache_path": str(cache), "gpu_lock": str(lock),
            "stage01_dir": str(paths["01"]), "stage02_dir": str(paths["02"]),
            "hub_cache_dir": str(c.qwen14_hub_cache) if c.qwen14_hub_cache else None,
            "resource_dir": str(c.embedding_resource_dir), "budgets": c.budgets(),
            "translation_policy": c.translation_policy, "preselection_policy": c.preselection_policy})
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        completed = subprocess.run([sys.executable, "-B", "-m", "finterm_linking.gpu_worker", "--job", str(job)],
            cwd=paths["run"], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for suffix, content in (("stdout", completed.stdout), ("stderr", completed.stderr)):
            with (paths["run"] / ("gpu_" + phase + "_" + suffix + ".log")).open("x", encoding="utf-8") as stream:
                stream.write(content)
        if completed.returncode:
            raise RuntimeError("GPU stage " + phase + " failed; inspect this run's stage log")
        name = {"03": "03_manifest.json", "05": "05_defterm_manifest.json", "06": "06_similarity_manifest.json"}[phase]
        return paths[phase] / name

    def _collect(self, runtime, paths, manifest_path, top_k):
        m06 = verify_outputs(paths["06"], manifest_path.name)
        names = m06.get("tables", {"scores": "06_similarity_scores.jsonl", "rankings": "06_candidate_rankings.jsonl",
                                   "occurrences": "06_occurrence_status.jsonl"})
        scores, rankings, occurrences = (read_jsonl(paths["06"] / names[key]) for key in ("scores", "rankings", "occurrences"))
        contexts = {r["occurrence_uid"]: r for r in runtime.stage02.load_stage01(paths["01"])["contexts"]}
        grouped = {uid: [] for uid in contexts}
        for row in rankings:
            grouped[row["occurrence_uid"]].append(row)
        occurrences = [{**contexts[r["occurrence_uid"]], **r, "rankings": grouped[r["occurrence_uid"]]} for r in occurrences]
        names_by_stage = {"01": "01_manifest.json", "02": "02_manifest.json", "03": "03_manifest.json",
                          "05": "05_defterm_manifest.json", "06": manifest_path.name}
        manifests = {key: verify_outputs(paths[key], name) for key, name in names_by_stage.items()}
        stages = {key: {"path": str(paths[key] / name), "sha256": file_hash(paths[key] / name)} for key, name in names_by_stage.items()}
        controlled = (m06.get("model", {}).get("backend_kind") == runtime.native.TEST_BACKEND or
                      manifests["03"]["execution"]["backend_kind"] == "CONTROLLED_TEST_NO_MODEL" or
                      manifests["05"]["execution"]["backend_kind"] == "CONTROLLED_TEST_NO_MODEL")
        kind = ("CONTROLLED_TEST_NO_REAL_MODEL" if controlled else REPLAY_KIND if m06.get("execution_kind") == REPLAY_KIND
                else "PREPARE_ONLY_NO_INFERENCE" if self.config.similarity_mode == "prepare" else "H100_ORIGINAL_MODELS_OR_CACHE")
        inferences = {"translation": manifests["03"]["execution"], "gate": manifests["05"]["execution"]["gate"],
                      "definition": manifests["05"]["execution"]["definition"], "embedding": m06.get("execution", {"new_embedding_attempts": 0})}
        errors = [{"stage": "02", "detail": e} for e in (*manifests["02"].get("errors", []), *manifests["02"].get("retrieval_errors", []))]
        for phase, name, field, normal in (("03", "03_translations.jsonl", "processing_status", {"TRANSLATED", "PENDING_BUDGET"}),
                                         ("05", "05_defterm_gate.jsonl", "state", {"DONE", "PENDING_BUDGET"}),
                                         ("05", "05_defterm_definitions.jsonl", "state", {"DONE", "PENDING_BUDGET"})):
            for row in read_jsonl(paths[phase] / name):
                state = row.get(field)
                if state not in normal:
                    errors.append({"stage": phase, "task_id": row["task_id"], "state": state})
                elif field == "state" and state == "DONE" and row.get("processed", {}).get("generation_status") == "ERROR":
                    errors.append({"stage": phase, "task_id": row["task_id"], "state": "GENERATION_ERROR"})
        for phase in ("03", "05"):
            if manifests[phase]["execution"].get("adapter_error"):
                errors.append({"stage": phase, "state": "ADAPTER_UNAVAILABLE"})
        errors += [{"stage": "06", "pair_id": r["pair_id"], "state": r["scoring_status"]}
                   for r in scores if r["scoring_status"] not in {"SCORED", "PENDING_EMBEDDING"}]
        if m06.get("execution", {}).get("backend_error"):
            errors.append({"stage": "06", "state": m06["execution"]["backend_error"]})
        status = m06["status"]
        if errors or manifests["02"]["status"] != "COMPLETED" or self.config.mode == "full" and any(
                manifests[key]["status"] != "COMPLETED" for key in ("03", "05", "06")):
            status = "PARTIAL"
        candidates = read_jsonl(paths["05"] / "05_defterm_candidates.jsonl")
        reps = read_jsonl(paths["05"] / "05_defterm_representations.jsonl")
        components = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        trace = {"stages": stages, "source_export": runtime.integrity, "integration_component_sha256": components,
                 "source_stage_status": {key: m["status"] for key, m in manifests.items()}, "inferences": inferences,
                 "coverage": {"occurrences": len(occurrences), "candidates": len(candidates), "candidate_rank_rows": len(rankings),
                     "scored_rank_rows": sum(r["score"] is not None for r in rankings), "source_representations": len(reps),
                     "pending_translations": manifests["03"]["counts"]["pending_translations"],
                     "defterm_states": manifests["05"]["counts"]["states"]}, "similarity_manifest": m06}
        real = not controlled and kind == "H100_ORIGINAL_MODELS_OR_CACHE" and inferences["embedding"].get("new_embeddings_cached", 0) > 0
        return LinkResult(paths["run"], self.config.mode, top_k, status, kind, real, occurrences,
            read_jsonl(paths["05"] / "05_defterm_terms.jsonl"), candidates, reps, scores, errors, trace)

    @staticmethod
    def _write_csv(result):
        columns = ("document_id", "occurrence_uid", "term", "context", "qid", "label_en", "description_en",
                   "representation_text_en", "score", "rank_position", "ranking_status", "max_representation_ids")
        with (result.run_dir / "rankings.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for occurrence in result.candidate_views():
                writer.writerows({**row, "max_representation_ids": json.dumps(row["max_representation_ids"])} for row in occurrence["candidates"])
