"""Operating modes with real coordinator/adapters and simulated runtimes only."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "evaluation_tests"))
from fixtures import Fixture
from adapter_doubles import runtime_double

from extractor import FinancialExtractor, MissingRetrievalResourcesError, SegmentationError, create_financial_extractor
from extractor.cli import main
from extractor.financial import ADAPTERS
from extractor.financial_prompts import FINANCIAL_ENTITIES
from extractor.modes import DEFAULT_BANK, DEMONSTRATION_DOCUMENTS
from extractor.retrieval import SentenceWorkerParser
from extractor.segmentation import PunktSegmenter
from experimental_evaluation.common import read_json
from experimental_evaluation.syntax import OriginalTimedSimilarity
from llm_adapters import MODEL_IDS
from syntactic_retrieval import ParsedText, SyntaxTree


FIRST = "The European Central Bank assessed CDS credit risk."
SECOND = "The European Central Bank reviewed CDS credit risk."
TEXT = FIRST + " " + SECOND
TERMS = ["European Central Bank", "CDS", "credit risk"]


@contextmanager
def fake_punkt(root):
    """Exercise the actual PunktSegmenter boundary, replacing NLTK's tokenizer."""
    package, data = ModuleType("nltk"), ModuleType("nltk.data")
    tokenize, punkt = ModuleType("nltk.tokenize"), ModuleType("nltk.tokenize.punkt")
    package.__path__ = []; tokenize.__path__ = []
    package.data, package.tokenize, tokenize.punkt = data, tokenize, punkt
    data.path = ["synthetic-original-search-path"]
    data.find = Mock(return_value=Path(root) / "tokenizers/punkt_tab/english")
    punkt.PunktTokenizer = Mock()
    punkt.PunktTokenizer.return_value.span_tokenize.side_effect = lambda text: (
        [(0, len(FIRST)), (len(FIRST) + 1, len(TEXT))] if text == TEXT else [(0, len(text))]
    )
    modules = {"nltk": package, "nltk.data": data, "nltk.tokenize": tokenize, "nltk.tokenize.punkt": punkt}
    with patch.dict(sys.modules, modules), patch("experimental_evaluation.resources.private_path", side_effect=lambda p: Path(p).resolve()), patch(
        "extractor.segmentation.metadata.version", return_value="3.10.4"
    ):
        yield punkt.PunktTokenizer, data


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture(); self.addCleanup(self.fixture.close)
        self.runtime = runtime_double(json.dumps({"terms": TERMS + ["CDS"]}))
        self.loader = self.enterContext(patch("llm_adapters.adapters.load_huggingface_runtime", return_value=self.runtime))
        self.tokenizer, self.nltk_data = self.enterContext(fake_punkt(self.fixture.resources.nltk_data))
        self.preflight = self.enterContext(patch("extractor.retrieval.resource_inventory"))
        self.parsed = []
        def parse(parser, text):
            self.parsed.append(text)
            return ParsedText((SyntaxTree("ROOT", (SyntaxTree("NN"),)),), parser.parser_id)
        self.enterContext(patch.object(SentenceWorkerParser, "parse", parse))
        self.kernel = self.enterContext(patch.object(OriginalTimedSimilarity, "score", return_value=0.5))
        self.factory = Mock(wraps=FinancialExtractor.from_local)

    def options(self, mode="high_recall"):
        result = {"model_path": self.fixture.profiles["gemma2" if mode == "high_recall" else "qwen3"].load.source,
                  "punkt_data": self.fixture.resources.nltk_data}
        if mode == "high_recall":
            result.update(java_bin=self.fixture.resources.java_bin, parser_jar=self.fixture.resources.parser_jar,
                          models_jar=self.fixture.resources.models_jar)
        return result

    def create(self, mode="balanced", **overrides):
        options = self.options(mode); options.update(overrides)
        return create_financial_extractor(mode, **options, _engine_factory=self.factory)

    def cli(self, mode="balanced", *, manual=False):
        options = self.options(mode)
        argv = ["--model-path", options["model_path"], "--punkt-data", options["punkt_data"], "--text", TEXT]
        if not manual:
            argv += ["--mode", mode]
        if mode == "high_recall":
            argv += ["--java-bin", options["java_bin"], "--parser-jar", options["parser_jar"], "--models-jar", options["models_jar"]]
        return argv

    def invoke(self, argv, *, platform=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch("extractor.cli.require_execution_platform") if platform else patch(
            "extractor.cli.sys.platform", "win32"
        ):
            code = main(argv, engine_factory=self.factory)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_balanced_selects_only_lazy_qwen_basic_without_retrieval(self):
        qwen, gemma, dister = (Mock(wraps=ADAPTERS[key]) for key in ("qwen3", "gemma2", "dister"))
        with patch.dict(ADAPTERS, qwen3=qwen, gemma2=gemma, dister=dister), patch(
            "extractor.modes.load_financial_bank", side_effect=AssertionError("balanced must not read a bank")
        ):
            engine = self.create()
        self.assertIsInstance(engine, FinancialExtractor)
        self.assertEqual(engine.model, "qwen3")
        self.assertIsNone(engine.retriever)
        self.assertFalse(engine.adapter.is_loaded)
        self.loader.assert_not_called(); self.preflight.assert_not_called()
        gemma.assert_not_called(); dister.assert_not_called()
        self.assertTrue(qwen.called)

    def test_python_default_is_balanced(self):
        engine = create_financial_extractor(**self.options("balanced"), _engine_factory=self.factory)
        self.assertEqual(engine.mode, "balanced")
        self.assertEqual(engine.model, "qwen3")

    def test_high_recall_selects_only_lazy_gemma_original_retriever_and_ten_example_bank(self):
        before = DEFAULT_BANK.read_bytes()
        qwen, gemma, dister = (Mock(wraps=ADAPTERS[key]) for key in ("qwen3", "gemma2", "dister"))
        with patch.dict(ADAPTERS, qwen3=qwen, gemma2=gemma, dister=dister):
            engine = self.create("high_recall")
        self.assertEqual(engine.model, "gemma2")
        self.assertFalse(engine.adapter.is_loaded)
        qwen.assert_not_called(); dister.assert_not_called(); self.assertTrue(gemma.called)
        self.loader.assert_not_called()
        self.assertIsInstance(engine.retriever.parser, SentenceWorkerParser)
        self.assertIsInstance(engine.retriever.similarity, OriginalTimedSimilarity)
        self.assertEqual(engine.retriever.config.top_k, 3)
        self.assertEqual(engine.retriever.config.authorized_document_ids, frozenset(DEMONSTRATION_DOCUMENTS))
        self.assertTrue(engine.retriever.config.exclude_query_document and engine.retriever.config.exclude_same_text)
        self.preflight.assert_called_once_with(engine.retriever.parser.resources)
        actual = [(e.identifier, e.text, list(e.terms)) for e in engine.retriever._examples]
        original = [(r["identifier"], r["text"], r["terms"]) for r in map(json.loads, before.decode("utf-8").splitlines()) if r]
        self.assertEqual(actual, original); self.assertEqual(len(actual), 10)
        self.assertEqual(DEFAULT_BANK.read_bytes(), before)
        retrieval = engine.mode_config["retrieval"]
        self.assertEqual(retrieval["bank_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(retrieval["bank_status"], "provisional_or_review_not_declared")

    def test_both_modes_use_punkt_sentence_calls_policy_acronyms_and_exact_union(self):
        for mode in ("balanced", "high_recall"):
            with self.subTest(mode=mode):
                self.loader.reset_mock(); self.runtime.model.calls.clear()
                engine = self.create(mode)
                self.assertIsInstance(engine.segmenter, PunktSegmenter)
                result = engine.extract_detailed(TEXT, query_document_id="manual-mode-test")
                self.assertTrue(result.complete)
                self.assertEqual(result.mode, mode)
                self.assertEqual(result.model, "qwen3" if mode == "balanced" else "gemma2")
                self.assertEqual(result.condition, "basic" if mode == "balanced" else "few_shot")
                self.assertEqual(result.prompt_version, "finance_entities_v2")
                self.assertEqual(result.terms, tuple(TERMS))
                self.assertEqual([s["text"] for s in result.sentences], [FIRST, SECOND])
                self.assertEqual(len(self.runtime.model.calls), 2)
                self.loader.assert_called_once_with(MODEL_IDS[result.model], engine.adapter.load_config)
                self.assertEqual(result.profile["generation"]["max_new_tokens"], 512)
                self.assertFalse(result.profile["generation"]["do_sample"])
                self.assertTrue(result.profile["load"]["local_files_only"])
                self.assertEqual(result.segmentation["policy"], "english_punkt_original_slices_v1")
                for sentence in result.sentences:
                    self.assertIn(FINANCIAL_ENTITIES, sentence["prompt"][1]["content"])
                    self.assertTrue(sentence["prompt_metadata"]["policy"]["include_acronyms"])
                    self.assertEqual(len(sentence["retrieved_examples"]), 0 if mode == "balanced" else 3)
                    self.assertIn("raw_response", sentence["generation"])
                if mode == "high_recall":
                    self.assertIn(FIRST, self.parsed); self.assertIn(SECOND, self.parsed)
                    self.assertTrue(self.kernel.called)
                    self.assertTrue(all("retrieval_diagnostics" in s for s in result.sentences))
                self.assertEqual(self.nltk_data.path, ["synthetic-original-search-path"])
        self.assertTrue(all(c.args == ("english",) for c in self.tokenizer.call_args_list))

    def test_python_selector_rejects_manual_options_even_when_redundant(self):
        for mode in ("balanced", "high_recall"):
            for conflict in ({"model": "qwen3"}, {"model": "gemma2"}, {"fastkassim": True}, {"fastkassim": False},
                             {"authorized_documents": DEMONSTRATION_DOCUMENTS}, {"authorized_documents": ()}):
                with self.subTest(mode=mode, conflict=conflict), self.assertRaisesRegex(ValueError, "manual selectors"):
                    self.create(mode, **conflict)
        self.factory.assert_not_called(); self.loader.assert_not_called()

    def test_unknown_modes_fail_before_resource_or_adapter_loading(self):
        for mode in ("", "BALANCED", "dister", None, []):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Select mode"):
                create_financial_extractor(mode, model_path="unused", _engine_factory=self.factory)
        self.factory.assert_not_called(); self.preflight.assert_not_called()

    def test_balanced_rejects_retrieval_resource_options(self):
        for key in ("bank", "java_bin", "parser_jar", "models_jar"):
            with self.subTest(option=key), self.assertRaisesRegex(ValueError, "balanced is basic"):
                self.create("balanced", **{key: "unused"})
        self.factory.assert_not_called()

    def test_missing_high_recall_options_are_explicit_before_adapter_construction(self):
        for key in ("punkt_data", "java_bin", "parser_jar", "models_jar"):
            with self.subTest(option=key), patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
                MissingRetrievalResourcesError, "high_recall requires FastKASSIM resources"
            ):
                self.create("high_recall", **{key: None})
        self.factory.assert_not_called()

    def test_missing_files_or_punkt_directory_do_not_fallback(self):
        for key in ("bank", "java_bin", "parser_jar", "models_jar", "punkt_data"):
            with self.subTest(option=key), self.assertRaisesRegex(MissingRetrievalResourcesError, "high_recall needs"):
                self.create("high_recall", **{key: self.fixture.root / ("missing-" + key)})
        self.factory.assert_not_called(); self.loader.assert_not_called()

    def test_empty_resource_file_is_rejected(self):
        empty = self.fixture.root / "empty.jar"; empty.write_bytes(b"")
        with self.assertRaisesRegex(MissingRetrievalResourcesError, "nonempty --parser-jar"):
            self.create("high_recall", parser_jar=empty)
        self.factory.assert_not_called()

    def test_unavailable_punkt_fails_explicitly_before_adapter_construction(self):
        self.tokenizer.side_effect = LookupError("synthetic missing Punkt parameters")
        with self.assertRaises(SegmentationError):
            self.create("high_recall")
        self.factory.assert_not_called(); self.loader.assert_not_called()

    def test_preflight_or_dependency_failure_never_substitutes_basic(self):
        for failure in (ValueError("synthetic broken Stanford JAR"), ModuleNotFoundError("synthetic missing fkassim")):
            with self.subTest(failure=failure), self.assertRaises(type(failure)):
                self.preflight.side_effect = failure
                self.create("high_recall")
        self.factory.assert_not_called(); self.loader.assert_not_called()

    def test_bank_location_is_configurable_and_annotations_are_preserved(self):
        copy = self.fixture.root / "relocated-pilot.jsonl"; before = DEFAULT_BANK.read_bytes(); copy.write_bytes(before)
        engine = self.create("high_recall", bank=copy)
        self.assertEqual(engine.mode_config["retrieval"]["bank_path"], str(copy))
        self.assertEqual(engine.mode_config["retrieval"]["bank_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(copy.read_bytes(), before)

    def test_missing_default_bank_is_an_explicit_error(self):
        with patch("extractor.modes.DEFAULT_BANK", self.fixture.root / "not-installed.jsonl"), self.assertRaisesRegex(
            MissingRetrievalResourcesError, "--bank"
        ):
            self.create("high_recall")
        self.factory.assert_not_called()

    def test_punkt_environment_and_explicit_profile_overrides_remain_configurable(self):
        with patch.dict(os.environ, {"FINTERM_PUNKT_ROOT": self.fixture.resources.nltk_data}):
            engine = self.create("high_recall", punkt_data=None, revision="c" * 40, max_new_tokens=700)
        self.assertEqual(engine.adapter.load_config.revision, "c" * 40)
        self.assertEqual(engine.adapter.generation_options.max_new_tokens, 700)
        self.assertEqual(engine.segmenter.data_root, Path(self.fixture.resources.nltk_data))

    def test_runtime_retrieval_failure_keeps_mode_diagnostics_and_never_loads_model(self):
        engine = self.create("high_recall")
        engine.retriever.retrieve = Mock(side_effect=RuntimeError("synthetic Stanford failure"))
        result = engine.extract_detailed(TEXT)
        self.assertFalse(result.complete)
        self.assertEqual(result.mode, "high_recall"); self.assertEqual(result.condition, "few_shot")
        self.assertEqual(result.terms, ())
        self.assertTrue(all(s["error"]["phase"] == "retrieval" for s in result.sentences))
        self.assertTrue(all("synthetic Stanford failure" in s["error"]["traceback"] for s in result.sentences))
        self.loader.assert_not_called()

    def test_mode_report_configuration_is_a_snapshot(self):
        engine = self.create("high_recall")
        result = engine.extract_detailed(TEXT)
        engine.mode_config["retrieval"]["top_k"] = 9
        self.assertEqual(result.mode_config["retrieval"]["top_k"], 3)

    def test_insufficient_demonstrations_are_an_error_in_high_recall(self):
        engine = self.create("high_recall")
        selected = engine.retriever.retrieve(FIRST, top_k=3)
        engine.retriever.retrieve = Mock(return_value=selected[:2])
        result = engine.extract_detailed(TEXT)
        self.assertFalse(result.complete)
        self.assertEqual(result.mode, "high_recall"); self.assertEqual(result.condition, "few_shot")
        self.assertTrue(all(s["error"] for s in result.sentences))
        self.loader.assert_not_called()

    def test_cli_explicit_modes_write_effective_mode_model_and_configuration(self):
        for mode in ("balanced", "high_recall"):
            with self.subTest(mode=mode):
                report = self.fixture.root / (mode + ".json")
                code, stdout, stderr = self.invoke(self.cli(mode) + ["--report", str(report)])
                self.assertEqual(code, 0, stderr)
                self.assertEqual(json.loads(stdout), {"terms": TERMS})
                payload = read_json(report)
                self.assertEqual(payload["mode"], mode)
                self.assertEqual(payload["model"], "qwen3" if mode == "balanced" else "gemma2")
                self.assertEqual(payload["condition"], "basic" if mode == "balanced" else "few_shot")
                self.assertEqual(payload["profile"]["model_id"], MODEL_IDS[payload["model"]])
                self.assertEqual(payload["mode_config"]["retrieval"]["top_k"], 0 if mode == "balanced" else 3)
                self.assertEqual(payload["prompt_version"], "finance_entities_v2")
                self.assertEqual(len(payload["sentences"]), 2)

    def test_cli_without_selectors_defaults_to_balanced(self):
        report = self.fixture.root / "implicit-balanced.json"
        code, _, stderr = self.invoke(self.cli(manual=True) + ["--report", str(report)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(read_json(report)["mode"], "balanced")

    def test_cli_rejects_conflicting_manual_flags_before_constructing_adapter(self):
        for mode in ("balanced", "high_recall"):
            for extra in (["--model", "qwen3"], ["--model", "gemma2"], ["--fastkassim"],
                          ["--authorized-document", "CELEX:32013R0549"]):
                with self.subTest(mode=mode, extra=extra):
                    code, stdout, stderr = self.invoke(self.cli(mode) + extra)
                    self.assertEqual(code, 2); self.assertEqual(stdout, "")
                    self.assertIn("manual selectors", json.loads(stderr)["message"])
        self.factory.assert_not_called()

    def test_cli_missing_high_recall_resources_is_explicit(self):
        code, stdout, stderr = self.invoke(["--mode", "high_recall", "--model-path", "unused", "--text", TEXT])
        self.assertEqual(code, 2); self.assertEqual(stdout, "")
        self.assertIn("high_recall requires FastKASSIM resources", json.loads(stderr)["message"])
        self.factory.assert_not_called()

    def test_cli_balanced_rejects_retrieval_resources(self):
        code, _, stderr = self.invoke(self.cli() + ["--java-bin", "unused"])
        self.assertEqual(code, 2); self.assertIn("balanced is basic", json.loads(stderr)["message"])
        self.factory.assert_not_called()

    def test_cli_runtime_retrieval_failure_preserves_report_and_is_nonzero(self):
        report = self.fixture.root / "failed-high-recall.json"
        self.kernel.side_effect = RuntimeError("synthetic LTK failure")
        code, stdout, stderr = self.invoke(self.cli("high_recall") + ["--report", str(report)])
        self.assertEqual(code, 2); self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"], "incomplete_extraction")
        result = read_json(report)
        self.assertFalse(result["complete"])
        self.assertEqual(result["mode"], "high_recall"); self.assertEqual(result["condition"], "few_shot")
        self.assertTrue(all(s["error"]["phase"] == "retrieval" for s in result["sentences"]))
        self.assertTrue(all("synthetic LTK failure" in s["error"]["traceback"] for s in result["sentences"]))
        self.loader.assert_not_called()

    def test_cli_legacy_manual_models_including_dister_remain_available(self):
        for model in ("qwen3", "gemma2", "dister"):
            with self.subTest(model=model):
                self.runtime.tokenizer.response = json.dumps(TERMS if model == "dister" else {"terms": TERMS})
                report = self.fixture.root / ("manual-" + model + ".json")
                code, stdout, stderr = self.invoke([
                    "--model", model, "--model-path", self.fixture.profiles[model].load.source,
                    "--punkt-data", self.fixture.resources.nltk_data, "--text", TEXT, "--report", str(report),
                ])
                self.assertEqual(code, 0, stderr); self.assertEqual(json.loads(stdout), {"terms": TERMS})
                payload = read_json(report)
                self.assertIsNone(payload["mode"]); self.assertIsNone(payload["mode_config"])
                self.assertEqual(payload["model"], model)
        self.preflight.assert_not_called()

    def test_cli_legacy_manual_fastkassim_still_uses_explicit_bank_and_authorization(self):
        report = self.fixture.root / "manual-few-shot.json"
        code, _, stderr = self.invoke(self.cli("high_recall", manual=True) + [
            "--model", "gemma2", "--fastkassim", "--bank", str(self.fixture.bank_path),
            "--authorized-document", "CELEX:32021B0417", "--report", str(report),
        ])
        self.assertEqual(code, 0, stderr)
        result = read_json(report)
        self.assertIsNone(result["mode"]); self.assertEqual(result["condition"], "few_shot")
        self.assertTrue(all(len(s["retrieved_examples"]) == 3 for s in result["sentences"]))

    def test_cli_windows_execution_guard_applies_to_modes(self):
        code, _, stderr = self.invoke(self.cli(), platform=False)
        self.assertEqual(code, 2); self.assertIn("Linux/CUDA", json.loads(stderr)["message"])
        self.factory.assert_not_called(); self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
