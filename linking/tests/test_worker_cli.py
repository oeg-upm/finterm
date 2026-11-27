from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from finterm_linking import Linker
from finterm_linking._runtime import frozen_runtime
from finterm_linking.cli import main
from finterm_linking.gpu_worker import run_job
from tests.support import CODE, ROOT, WorkingTest


class WorkerCLITests(WorkingTest):
    def job(self, phase="06"):
        return {"schema_version": "finterm-gpu-stage-job-v0.1", "stage": phase, "mode": "full",
                "code_dir": str(CODE), "input_dir": str(self.root / "input"), "output_dir": str(self.root / "output"),
                "stage01_dir": str(self.root / "01"), "stage02_dir": str(self.root / "02"),
                "cache_path": str(self.root / "cache"), "gpu_lock": str(self.root / "gpu.lock"),
                "hub_cache_dir": str(self.root / "hub"), "resource_dir": str(self.root / "resources"),
                "budgets": {"translations": 2, "gate": 3, "definitions": 1, "embeddings": 81},
                "translation_policy": "historical", "preselection_policy": "conservative"}

    def test_windows_worker_rejects_model_execution_immediately(self):
        with patch.object(sys, "platform", "win32"), self.assertRaisesRegex(ValueError, "Windows"):
            run_job(self.job())

    def test_each_worker_delegates_to_original_backend_with_exact_budgets(self):
        with frozen_runtime() as runtime:
            import finterm_sense2_qwen as translation
            import finterm_defterm as definition
            for phase, owner, cls, operation, module in (
                ("03", translation, "QwenContextualV2", "run_stage03", runtime.sense2_store),
                ("05", definition, "QwenDefTerm", "run_defterm", runtime.defterm_store),
                ("06", runtime.native, "QwenEmbedding4B", "run_similarity", runtime.similarity)):
                backend = SimpleNamespace(model=None, tokenizer=None, close=Mock())
                with self.subTest(phase=phase), patch.object(sys, "platform", "linux"), \
                        patch("finterm_linking.gpu_worker.frozen_runtime", return_value=nullcontext(runtime)), \
                        patch.object(runtime.native, "check_environment", return_value={"compatible": True}), \
                        patch.object(owner, cls, return_value=backend) as constructor, \
                        patch.object(module, operation, return_value={"status": "CONTROLLED_PROTOCOL_TEST"}) as call:
                    run_job(self.job(phase))
                backend.close.assert_called_once()
                kwargs = call.call_args.kwargs
                if phase == "03":
                    self.assertEqual(kwargs["max_translations"], 2)
                    self.assertIs(kwargs["adapter"], backend)
                elif phase == "05":
                    self.assertEqual((kwargs["max_gate_calls"], kwargs["max_definition_calls"]), (3, 1))
                else:
                    self.assertEqual(kwargs["max_new_embeddings"], 81)
                    self.assertEqual(constructor.call_args.kwargs, {"batch_size": 1, "gpu_reserve_gib": 8.0, "allow_shared_gpu": False})

    def test_gpu_lock_and_close_on_failure(self):
        with frozen_runtime() as runtime:
            backend = SimpleNamespace(close=Mock())
            with patch.object(sys, "platform", "linux"), \
                    patch("finterm_linking.gpu_worker.frozen_runtime", return_value=nullcontext(runtime)), \
                    patch.object(runtime.native, "check_environment", return_value={"compatible": True}), \
                    patch.object(runtime.native, "QwenEmbedding4B", return_value=backend), \
                    patch.object(runtime.similarity, "run_similarity", side_effect=ValueError("CONTROLLED_FAILURE")), \
                    self.assertRaisesRegex(ValueError, "CONTROLLED_FAILURE"):
                run_job(self.job())
            backend.close.assert_called_once()
            with runtime.native.exclusive_lock(Path(self.job()["gpu_lock"])), patch.object(sys, "platform", "linux"), \
                    self.assertRaises(OSError):
                run_job(self.job())

    def test_parent_starts_synchronous_local_only_clean_process(self):
        from finterm_linking import LinkerConfig
        with patch.object(sys, "platform", "linux"):
            config = LinkerConfig(work_dir=self.root / "work", mode="full", similarity_mode="h100", embedding_resource_dir=self.root / "resources")
        run = config.work_dir / "run"
        run.mkdir(parents=True)
        paths = {k: run / k for k in ("01", "02", "03", "05", "06")}
        paths["run"] = run
        with frozen_runtime() as runtime, patch("finterm_linking.orchestrator.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as process:
            Linker(config)._execute_gpu_stage(runtime, paths, "03", run / "cache.sqlite")
        args, kwargs = process.call_args
        self.assertEqual(args[0][:4], [sys.executable, "-B", "-m", "finterm_linking.gpu_worker"])
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(kwargs["cwd"], run)

    def test_cli_prepare_verify_and_relative_config_paths(self):
        text, terms, seed = self.fixture()
        config = self.root / "settings.json"
        config.write_text(json.dumps({"work_dir": "runs_here", "wikidata_seed_cache": str(seed),
            "lemma_map": str(ROOT / "tests/fixtures/stage02_lemmas.json")}), encoding="utf-8")
        args = ["run", "--config", str(config), "--text", text, "--top-k", "3"]
        for term in terms:
            args.extend(["--term", term])
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(args), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["execution_kind"], "PREPARE_ONLY_NO_INFERENCE")
        self.assertTrue(Path(report["run_dir"]).is_relative_to(self.root / "runs_here"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["verify", report["run_dir"]]), 0)

    def test_cli_imports_in_clean_process_without_ml_or_project_paths(self):
        program = ("import sys; from finterm_linking.cli import doctor; r=doctor(); "
                   "assert r['imports_and_contracts_valid']; "
                   "assert not ({'torch','transformers','sentence_transformers','spacy','external_extractor'} & set(sys.modules)); "
                   "print('CLEAN_IMPORT_OK')")
        result = subprocess.run([sys.executable, "-B", "-c", program], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLEAN_IMPORT_OK", result.stdout)
