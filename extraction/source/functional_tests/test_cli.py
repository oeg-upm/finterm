"""Standalone CLI and distributable-wheel tests; no model or Java execution."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch
import zipfile

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "evaluation_tests"))
from fixtures import Fixture

from extractor.cli import main
from extractor.financial import FinancialExtractionResult
from experimental_evaluation.common import read_json


class CliTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture(); self.addCleanup(self.fixture.close)
        self.path = self.fixture.profiles["qwen3"].load.source
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.result = FinancialExtractionResult("Input.", "qwen3", ("Aster Bank", "credit risk"), (), True, {}, {}, "basic")
        self.engine = Mock(extract_detailed=Mock(return_value=self.result))
        self.factory = Mock(return_value=self.engine)

    def invoke(self, args, *, platform=True):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr), patch("extractor.cli.require_execution_platform") if platform else patch("extractor.cli.sys.platform", "win32"):
            return main(args, engine_factory=self.factory)

    def test_one_final_list_output_and_new_detailed_report(self):
        report = self.fixture.root / "new-report.json"
        code = self.invoke(["--model-path", self.path, "--text", "Input.", "--report", str(report)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.stdout.getvalue()), {"terms": ["Aster Bank", "credit risk"]})
        self.assertEqual(read_json(report)["prompt_version"], "finance_entities_v2")
        self.engine.extract_detailed.assert_called_once_with("Input.", query_document_id=None)

    def test_original_utf8_file_not_rewritten(self):
        source = self.fixture.root / "input.txt"; value = "Équity rises.\r\nDebt falls."
        source.write_bytes(value.encode("utf-8"))
        self.invoke(["--model-path", self.path, "--input", str(source)])
        self.engine.extract_detailed.assert_called_once_with(value, query_document_id=None)
        self.assertEqual(source.read_bytes(), value.encode("utf-8"))

    def test_existing_report_never_overwritten_or_model_constructed(self):
        report = self.fixture.root / "old.json"; report.write_bytes(b"old result")
        self.assertEqual(self.invoke(["--model-path", self.path, "--text", "Input.", "--report", str(report)]), 2)
        self.assertEqual(report.read_bytes(), b"old result")
        self.factory.assert_not_called()

    def test_key_is_not_an_extraction_input(self):
        source = self.fixture.root / "original.key"; source.write_bytes(b"auxiliary reference")
        self.assertEqual(self.invoke(["--model-path", self.path, "--input", str(source)]), 2)
        self.factory.assert_not_called()

    def test_dister_fastkassim_rejected_before_loading(self):
        self.assertEqual(self.invoke(["--model", "dister", "--model-path", self.path, "--text", "Input.", "--fastkassim"]), 2)
        self.factory.assert_not_called()

    def test_windows_real_execution_guard_before_model_factory(self):
        self.assertEqual(self.invoke(["--model-path", self.path, "--text", "Input."], platform=False), 2)
        self.factory.assert_not_called()

    def test_incomplete_extraction_is_nonzero_no_partial_success_stdout(self):
        failed = {"number": 1, "error": {"message": "synthetic failure"}}
        self.engine.extract_detailed.return_value = FinancialExtractionResult("Input.", "qwen3", ("partial",), (failed,), False, {}, {}, "basic")
        report = self.fixture.root / "failure.json"
        code = self.invoke(["--model-path", self.path, "--text", "Input.", "--report", str(report)])
        self.assertEqual(code, 2)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertFalse(read_json(report)["complete"])

    def test_model_selector_and_explicit_budget_forwarded(self):
        self.assertEqual(self.invoke(["--model", "gemma2", "--model-path", self.path, "--text", "Input.", "--max-new-tokens", "700"]), 0)
        self.assertEqual(self.factory.call_args.args[0], "gemma2")
        self.assertEqual(self.factory.call_args.kwargs["max_new_tokens"], 700)

    def test_help_and_version_run_without_optional_libraries(self):
        for flag in ("--help", "--version"):
            code = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','nltk','fkassim','numpy','scipy','spacy'}:
            raise AssertionError('heavy import: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['extractor',sys.argv[1]]
runpy.run_module('extractor',run_name='__main__')
"""
            result = subprocess.run([sys.executable, "-B", "-c", code, flag], cwd=PROJECT,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FINTERM", result.stdout)


class PackagingTests(unittest.TestCase):
    def test_metadata_includes_runtime_helpers_and_explicit_cli(self):
        data = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["project"]["version"], "0.2.0")
        self.assertEqual(data["project"]["scripts"]["finterm-finance"], "extractor.cli:main")
        self.assertIn("nltk>=3.10.4,<4", data["project"]["optional-dependencies"]["text"])
        self.assertTrue({"experimental_evaluation*", "demonstrations*"}.issubset(data["tool"]["setuptools"]["packages"]["find"]["include"]))

    def test_wheel_build_offline_in_temporary_copy_contains_working_cli(self):
        # Build only a disposable source copy: no build/egg-info in the project,
        # no dependency installation, network, models or original corpus reads.
        with tempfile.TemporaryDirectory(dir=os.environ.get("FINTERM_TEST_TMP")) as temporary:
            root = Path(temporary); source = root / "source"; source.mkdir()
            for name in ("pyproject.toml", "README.md"):
                shutil.copyfile(PROJECT / name, source / name)
            for name in ("extractor", "prompts", "llm_adapters", "syntactic_retrieval", "corpus_readers", "experimental_evaluation", "demonstrations"):
                shutil.copytree(PROJECT / name, source / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.json", "*.jsonl", "*.md", "candidate_tests"))
            (source / "demonstrations/candidatos_financieros.jsonl").write_bytes(b"synthetic dataset must not ship")
            wheels = root / "wheels"; wheels.mkdir()
            command = [sys.executable, "-B", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
                       "--wheel-dir", str(wheels), str(source)]
            build = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            wheel, = wheels.glob("*.whl")
            installed = root / "installed"; installed.mkdir()
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                for name in ("extractor/financial.py", "extractor/cli.py", "experimental_evaluation/syntax_worker.py", "demonstrations/matching.py"):
                    self.assertIn(name, names)
                self.assertNotIn("demonstrations/candidatos_financieros.jsonl", names)
                self.assertFalse(any(n.endswith((".safetensors", ".bin", ".jar")) for n in names))
                entries = [n for n in names if n.endswith("entry_points.txt")]
                self.assertIn("finterm-finance = extractor.cli:main", archive.read(entries[0]).decode())
                archive.extractall(installed)  # our own locally built wheel only
            code = "import sys,runpy; sys.path.insert(0," + repr(str(installed)) + "); sys.argv=['extractor','--help']; runpy.run_module('extractor',run_name='__main__')"
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=root, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--model-path", result.stdout)


if __name__ == "__main__":
    unittest.main()
