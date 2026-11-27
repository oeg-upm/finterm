import builtins
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from finterm_linking import LinkerConfig
from finterm_linking._runtime import frozen_runtime
from tests.recovered_support import ControlledEncoder, FixtureRecorder

ROOT = Path(__file__).resolve().parents[1]
CODE = LinkerConfig().code_dir
FIXTURES = Path(__file__).parent / "fixtures"
TEST_WORK = Path(os.environ.get("FINTERM_TEST_TMP", tempfile.gettempdir()))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


class ControlledTranslation:
    def __init__(self):
        from finterm_sense2_qwen import generation_contract, TEST_BACKEND
        self.contract = generation_contract(backend_kind=TEST_BACKEND)
        self.calls = []

    def prepare(self):
        pass

    def generate(self, task):
        self.calls.append(task)
        return {"task_id": task["task_id"], "generation_call_status": "RETURNED", "translation_raw": "An exploratory lunar vehicle.",
                "finish_reason": "eos_token", "input_token_count": 30, "generated_token_count": 6, "error": None,
                "metadata": {"backend_kind": self.contract["backend_kind"], "model_id": self.contract["model_id"],
                             "model_revision": self.contract["model_revision"], "test_only": True}}


class ControlledDefTerm:
    backend_kind = "CONTROLLED_TEST_NO_MODEL"

    def __init__(self, gate="SUFFICIENT_EVIDENCE"):
        self.calls, self.gate = [], gate

    def prepare(self):
        pass

    def generate(self, task):
        self.calls.append(task)
        raw = self.gate if task["request"]["stage"] == "gate" else "An instrument used for a synthetic operation."
        return {"task_id": task["task_id"], "raw": raw, "call_status": "RETURNED", "finish_reason": "eos_token",
                "input_token_count": 30, "generated_token_count": 9,
                "runtime": {"backend_kind": self.backend_kind, "test_only": True}}


class FullFixtureRecorder(FixtureRecorder):
    """Only tests: one translation and one naturally eligible synthetic gate."""
    def entity(self, qid):
        from tests.recovered_support import Q_ROVER, Q_SAMPLES
        value = super().entity(qid)
        if qid == Q_ROVER:
            value["labels"] = {"en": {"value": "lunar rover"}}
            value["descriptions"] = {"es": {"value": "vehículo explorador lunar"}}
        if qid == Q_SAMPLES:
            value["labels"] = {"en": {"value": "samples"}}
            value["descriptions"] = {}
        return value


class WorkingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="finterm-distribution-", dir=TEST_WORK)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("OFFLINE TEST: network forbidden")).start()
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "sentence_transformers", "spacy", "external_extractor"}:
                raise AssertionError("OFFLINE TEST: importing a real model/runtime is forbidden: " + name)
            return original(name, *args, **kwargs)
        patch.object(builtins, "__import__", side_effect=guarded).start()

    def fixture(self, *, full=False, empty=False):
        text = "USER_CORPUS_LEAK_SENTINEL. " + (FIXTURES / "user_document.txt").read_text(encoding="utf-8")
        terms = (FIXTURES / "user_terms.txt").read_text(encoding="utf-8").splitlines()
        with frozen_runtime() as runtime:
            phase01 = self.root / "fixture01"
            runtime.stage01.run_stage01(phase01, text=text, terms=terms, document_id="synthetic_demo", language="en")
            contexts = runtime.stage02.load_stage01(phase01)["contexts"]
            lemmas = json.loads((FIXTURES / "stage02_lemmas.json").read_text(encoding="utf-8"))["entries"]
            recorder = (FullFixtureRecorder if full else FixtureRecorder)(runtime.api, ["lunar rover", "basalt", "samples"], contexts, lemmas)
            runtime.stage02.collect(runtime.api, ["lunar rover", "basalt", "samples"], contexts, lemmas.__getitem__, recorder)
        seed = self.root / "seed.json"
        seed.write_text(json.dumps({"version": 1, "data_provenance": "CONTROLLED_FIXTURE_NOT_LIVE_WIKIDATA",
                                   "responses": {} if empty else recorder.responses}), encoding="utf-8")
        return text, terms, seed

    def config(self, seed, **changes):
        values = dict(work_dir=self.root / "work", offline=True, wikidata_seed_cache=seed,
                      lemma_map=FIXTURES / "stage02_lemmas.json")
        values.update(changes)
        return LinkerConfig(**values)

    def controlled_executor(self, translation=None, definition=None, encoder=None):
        order = []
        with frozen_runtime():
            translation = translation or ControlledTranslation()
            definition = definition or ControlledDefTerm()
        encoder = encoder or ControlledEncoder()
        def execute(runtime, paths, phase, cache):
            order.append(phase)
            cache.parent.mkdir(parents=True, exist_ok=True)
            if phase == "03":
                runtime.sense2_store.run_stage03(paths["02"], paths["03"], cache,
                    adapter=translation, max_translations=32)
                return paths["03"] / "03_manifest.json"
            if phase == "05":
                runtime.defterm_store.run_defterm(paths["03"], paths["05"], cache,
                    backend=definition, max_gate_calls=16, max_definition_calls=8)
                return paths["05"] / "05_defterm_manifest.json"
            runtime.similarity.run_similarity(paths["05"], paths["06"], cache, backend=encoder,
                max_new_embeddings=256, stage01_dir=paths["01"], stage02_dir=paths["02"])
            return paths["06"] / "06_similarity_manifest.json"
        return execute, order, translation, definition, encoder
