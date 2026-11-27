from contextlib import redirect_stderr
import csv
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

from finterm_linking import Linker, LinkerConfig, link
from finterm_linking._runtime import file_hash, frozen_runtime
from finterm_linking.verification import verify_run
from tests.support import CODE, WorkingTest, ControlledDefTerm, jsonl


class PipelineTests(WorkingTest):
    def test_packaged_imports_contracts_and_original_fewshots(self):
        from finterm_linking.cli import doctor
        report = doctor()
        self.assertTrue(report["imports_and_contracts_valid"])
        self.assertFalse(report["models_imported"])
        self.assertEqual(report["source_export"]["file_count"], 26)
        self.assertEqual(report["models"]["translation"]["generation"]["max_new_tokens"], 128)
        self.assertEqual(report["models"]["gate"]["generation"]["max_new_tokens"], 16)
        self.assertEqual(report["models"]["definition"]["generation"]["max_new_tokens"], 150)

    def test_quick_prepares_original_chain_and_preserves_pending_and_ids(self):
        text, terms, seed = self.fixture(full=True)
        result = link(text, terms, config=self.config(seed), mode="quick", document_id="synthetic_demo")
        self.assertEqual(result.mode, "quick")
        self.assertEqual(result.execution_kind, "PREPARE_ONLY_NO_INFERENCE")
        self.assertEqual(result.status, "PREPARED")
        self.assertTrue(all(r["score"] is None for r in result.rankings))
        self.assertEqual(result.trace["inferences"]["translation"]["new_attempts"], 0)
        self.assertEqual(result.trace["inferences"]["gate"]["new_attempts"], 0)
        self.assertGreater(result.trace["coverage"]["pending_translations"], 0)
        with frozen_runtime() as runtime:
            original = runtime.stage02.read_csv(result.run_dir / "02_candidatos/02_candidatos_relaciones.csv")
            self.assertEqual(original, jsonl(result.run_dir / "03_sentido2/03_associations.jsonl"))
            self.assertEqual(original, jsonl(result.run_dir / "05_defterm/05_defterm_associations.jsonl"))
        self.assertTrue(verify_run(result.run_dir)["integrity_verified"])

    def test_full_is_not_quick_and_calls_all_three_original_stage_paths_in_order(self):
        text, terms, seed = self.fixture(full=True)
        execute, order, translation, definition, encoder = self.controlled_executor()
        with patch.object(sys, "platform", "linux"):
            config = self.config(seed, mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "TEST_ONLY_RESOURCES")
        linker = Linker(config)
        with patch.object(linker, "_execute_gpu_stage", side_effect=execute):
            result = linker.link(text, terms, document_id="synthetic_demo")
        self.assertEqual(order, ["03", "05", "06"])
        self.assertGreater(len(translation.calls), 0)
        self.assertEqual([t["request"]["stage"] for t in definition.calls], ["gate", "definition"])
        self.assertGreater(len(encoder.calls), 0)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.execution_kind, "CONTROLLED_TEST_NO_REAL_MODEL")
        self.assertFalse(result.real_embedding_inference_performed)
        checked = verify_run(result.run_dir)
        self.assertTrue(checked["controlled_test"])
        self.assertEqual(checked["gpu_evidence"]["production_definitions_available"], 0)
        self.assertFalse(checked["full_gpu_validation_claimed"])
        from finterm_linking.acceptance import validate_h100_acceptance
        self.assertFalse(validate_h100_acceptance([result.run_dir])["full_runtime_acceptance_passed"])

    def test_saved_run_can_be_verified_after_relocation(self):
        import shutil
        text, terms, seed = self.fixture()
        result = Linker(self.config(seed)).link(text, terms)
        relocated = self.root / "relocated_run"
        shutil.copytree(result.run_dir, relocated)
        self.assertTrue(verify_run(relocated)["integrity_verified"])

    def test_candidate_only_messages_and_original_structured_evidence(self):
        text, terms, seed = self.fixture(full=True)
        execute, _, translation, definition, _ = self.controlled_executor()
        with patch.object(sys, "platform", "linux"):
            config = self.config(seed, mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "TEST_ONLY")
        linker = Linker(config)
        with patch.object(linker, "_execute_gpu_stage", side_effect=execute):
            linker.link(text, terms, document_id="synthetic_demo")
        tasks = translation.calls + definition.calls
        for task in tasks:
            messages = json.dumps(task["request"]["messages"])
            self.assertNotIn("USER_CORPUS_LEAK_SENTINEL", messages)
            self.assertNotIn("synthetic_demo", messages)
        evidence = translation.calls[0]["request"]["translation_evidence_text"]
        for expected in ("lunar rover", "synthetic alias", "SYNTHETIC TEST ONLY type", "SYNTHETIC TEST ONLY parent", "SYNTHETIC TEST ONLY source"):
            self.assertIn(expected, evidence)

    def test_full_persistent_cache_reuses_translation_gate_definition_and_embeddings(self):
        text, terms, seed = self.fixture(full=True)
        execute, _, translation, definition, encoder = self.controlled_executor()
        with patch.object(sys, "platform", "linux"):
            config = self.config(seed, mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "TEST_ONLY")
        linker = Linker(config)
        with patch.object(linker, "_execute_gpu_stage", side_effect=execute):
            first = linker.link(text, terms, document_id="synthetic_demo")
            counts = (len(translation.calls), len(definition.calls), len(encoder.calls))
            second = linker.link(text, terms, document_id="synthetic_demo")
        self.assertEqual(counts, (len(translation.calls), len(definition.calls), len(encoder.calls)))
        self.assertEqual(first.rankings, second.rankings)
        self.assertGreater(second.trace["inferences"]["translation"]["cache_hits"], 0)
        self.assertGreater(second.trace["inferences"]["gate"]["cache_hits"], 0)
        self.assertGreater(second.trace["inferences"]["definition"]["cache_hits"], 0)

    def test_convenience_api_preserves_full_mode_from_configuration(self):
        text, terms, seed = self.fixture(full=True)
        execute, order, _, _, _ = self.controlled_executor()
        with patch.object(sys, "platform", "linux"):
            config = self.config(seed, mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "TEST_ONLY")
        with patch.object(Linker, "_execute_gpu_stage", side_effect=execute):
            result = link(text, terms, config=config, document_id="synthetic_demo")
        self.assertEqual(result.mode, "full")
        self.assertEqual(order, ["03", "05", "06"])

    def test_insufficient_gate_abstains_without_definition(self):
        text, terms, seed = self.fixture(full=True)
        execute, _, _, definition, _ = self.controlled_executor(definition=ControlledDefTerm("INSUFFICIENT_EVIDENCE"))
        with patch.object(sys, "platform", "linux"):
            config = self.config(seed, mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "TEST_ONLY")
        linker = Linker(config)
        with patch.object(linker, "_execute_gpu_stage", side_effect=execute):
            result = linker.link(text, terms, document_id="synthetic_demo")
        self.assertEqual([t["request"]["stage"] for t in definition.calls], ["gate"])
        self.assertIn("GATE_INSUFFICIENT_EVIDENCE", result.trace["coverage"]["defterm_states"])

    def test_top_1_3_5_views_and_csv_preserve_native_order_and_scores(self):
        text, terms, seed = self.fixture()
        execute, _, _, _, _ = self.controlled_executor()
        linker = Linker(self.config(seed))
        def controlled(runtime, paths):
            return execute(runtime, paths, "06", self.root / "controlled.sqlite")
        with patch.object(linker, "_execute_similarity", side_effect=controlled):
            result = linker.link(text, terms, document_id="synthetic_demo", top_k=3)
        for k in (1, 3, 5):
            for view, occurrence in zip(result.top(k), result.occurrences):
                expected = [r for r in occurrence["rankings"] if r["score"] is not None][:k]
                self.assertEqual([r["qid"] for r in view["candidates"]], [r["qid"] for r in expected])
                self.assertTrue(all("label_en" in r and "description_en" in r and "context" in r for r in view["candidates"]))
                self.assertTrue(all(not r["validated_final_link"] for r in view["candidates"]))
        with (result.run_dir / "rankings.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), len(result.rankings))
        self.assertEqual(set(json.loads((result.run_dir / "top_candidates.json").read_text(encoding="utf-8"))), {"1", "3", "5"})

    def test_no_candidates_missing_terms_and_empty_list_are_explicit(self):
        text, terms, seed = self.fixture()
        result = Linker(self.config(seed)).link(text, terms, document_id="synthetic_demo")
        states = {r["term"]: r["retrieval_status"] for r in result.terms}
        self.assertEqual(states["basalt"], "NO_CANDIDATES")
        self.assertEqual(states["missing mineral"], "TERM_NOT_FOUND_IN_TEXT")
        empty = Linker(LinkerConfig(work_dir=self.root / "empty", search_policy="coverage")).link("Original text.", [])
        self.assertEqual(empty.rankings, [])
        self.assertEqual(empty.occurrences, [])

    def test_offline_miss_and_incomplete_lemma_map_remain_errors(self):
        text, terms, seed = self.fixture(empty=True)
        with redirect_stderr(io.StringIO()):
            result = Linker(self.config(seed)).link(text, terms)
        self.assertEqual(result.status, "PARTIAL")
        self.assertTrue(result.errors)

    def test_invalid_inputs_policies_budgets_and_recipe_overrides_fail_explicitly(self):
        for values in ({"search_policy": "invented"}, {"translation_policy": "invented"}, {"max_translations": True},
                       {"max_gate_calls": -1}, {"mode": "full"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LinkerConfig(**values)
        for terms in ("account", ("account",), [None], [" account"], ["a\nb"]):
            with self.subTest(terms=terms), self.assertRaises((ValueError, TypeError)):
                Linker(LinkerConfig(work_dir=self.root / "invalid")).link("Original account.", terms)
        with self.assertRaisesRegex(ValueError, "original fixed contract"):
            Linker(LinkerConfig(work_dir=self.root / "bad_limit", translation_token_limit=1)).link("A term.", ["term"])

    def test_protected_vendor_and_tampered_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            Linker(LinkerConfig(work_dir=CODE / "forbidden")).link("text", [])
        text, terms, seed = self.fixture()
        result = Linker(self.config(seed)).link(text, terms)
        path = result.run_dir / "rankings.csv"
        path.write_bytes(path.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verify_run(result.run_dir)

    def test_windows_refuses_real_inference_before_resources_or_models(self):
        with patch.object(sys, "platform", "win32"), self.assertRaisesRegex(ValueError, "Windows"):
            LinkerConfig(mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "missing")

    def test_attempt_without_computed_vectors_is_not_reported_as_real_inference(self):
        text, terms, seed = self.fixture()
        prepared = Linker(self.config(seed)).link(text, terms)
        path = prepared.run_dir / "06_similitud/06_similarity_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # Protocol-only failure record: an attempt is not a completed model result.
        manifest["execution"]["new_embedding_attempts"] = 1
        manifest["execution"]["new_embeddings_cached"] = 0
        manifest["status"] = "PARTIAL"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(sys, "platform", "linux"):
            config = self.config(seed, similarity_mode="h100", embedding_resource_dir=self.root / "TEST_ONLY")
        paths = {key: prepared.run_dir / name for key, name in (("01", "01_sentido1"), ("02", "02_candidatos"),
            ("03", "03_sentido2"), ("05", "05_defterm"), ("06", "06_similitud"))}
        paths["run"] = prepared.run_dir
        with frozen_runtime() as runtime:
            result = Linker(config)._collect(runtime, paths, path, 5)
        self.assertFalse(result.real_embedding_inference_performed)
