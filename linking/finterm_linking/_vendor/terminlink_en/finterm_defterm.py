"""Internal DefTerm method adapter: original evidence, prompts and few-shot code.

Historical runners are not executed. Pure functions embedded in the full-run
wrappers are projected unchanged, avoiding their experiment-specific imports.
Gate and definition engines have separate globals, as in the historical chain.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

from finterm_sense2 import BASE, canonical, digest, file_hash, object_hash, require


SCHEMA = "finterm-defterm-v1"
PRODUCTION = "QWEN_DEFTERM_FEWSHOT_LOCAL"
CONTROLLED = "CONTROLLED_TEST_NO_MODEL"
GATE_FILE = "05_defterm_qwen14_sufficiency_gate_fewshot.py"
PREP_FILE = "04_preparar_defterm_contextual_v2.py"
LOO_FILE = "05_defterm_qwen14_sufficiency_gate_fewshot_full_v2.py"
DEF_FILE = "05_defterm_qwen14_few_shot_contextual_v2_full.py"
POST_FILE = "05_preparar_defterm_post_gate.py"
FINAL_FILE = "05_consolidar_sentidos2_final.py"
HISTORICAL_FILES = (PREP_FILE, GATE_FILE, LOO_FILE, POST_FILE, DEF_FILE, FINAL_FILE)
TRACE_FIELDS = ("defterm_evidence_id", "qid", "label_en", "defterm_evidence_text")
CERTIFICATE_FIELDS = ("sufficiency_decision", "sufficiency_gate_result_id", "post_gate_partition_id")
META_FIELDS = ("model_id", "model_revision", "metadata_mode", "prompt_version", "recipe_sha256", "runtime_metadata")


def _module(name, path, body=None, namespace=None):
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    try:
        payload = path.read_bytes()
        code = compile(ast.Module(body=body, type_ignores=[]), str(path), "exec") if body is not None else compile(payload, str(path), "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _project(filename, names, namespace=None):
    """Compile named original AST nodes, without rewriting their function bodies."""
    require(filename in HISTORICAL_FILES, "Unreviewed historical DefTerm utility")
    path = BASE / filename
    tree = ast.parse(path.read_bytes())
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)  # These reviewed wrappers import only stdlib here.
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & set(names):
                body.append(node)
                found.update(targets)
    require(set(names) <= found, "Historical DefTerm utility changed its exported nodes")
    return _module("_finterm_defterm_projection_" + filename.removesuffix(".py"), path, body, namespace)


@lru_cache(maxsize=1)
def methods():
    prep = _module("_finterm_defterm_preparation", BASE / PREP_FILE)
    gate = _module("_finterm_defterm_gate", BASE / GATE_FILE)
    definition_engine = _module("_finterm_defterm_definition_engine", BASE / GATE_FILE)
    gate.validate_static_contract()
    gate.TRACEABILITY_FIELDS = TRACE_FIELDS
    gate.RUNTIME_METADATA_FIELDS = META_FIELDS
    definition_engine.TRACEABILITY_FIELDS = (*TRACE_FIELDS, *CERTIFICATE_FIELDS)
    definition_engine.RUNTIME_METADATA_FIELDS = META_FIELDS
    definition = _project(DEF_FILE, {
        "SYSTEM_MSG", "USER_MSG_TEMPLATE", "PROMPT_VERSION", "SYSTEM_SHA256", "USER_SHA256",
        "FEWSHOT_MESSAGES_SHA256", "FEWSHOT_FIELDS", "FEWSHOT_QIDS", "DefinitionExample",
        "project_examples", "build_demo_messages", "build_messages", "ensure_sufficient",
        "normalize_definition", "circularity_flag", "process_candidate",
    }, {"ENGINE": definition_engine, "GATE": SimpleNamespace(ENGINE=gate),
        "POST": SimpleNamespace(PARTITION_FIELDS=(*TRACE_FIELDS, *CERTIFICATE_FIELDS)), "require": gate.require})
    # Read the approved override, instead of maintaining a second token limit.
    limits = [ast.literal_eval(node.value) for node in ast.parse((BASE / DEF_FILE).read_bytes()).body
              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
                  and isinstance(target.value, ast.Name) and target.value.id == "ENGINE" and target.attr == "MAX_NEW_TOKENS"
                  for target in node.targets)]
    require(len(limits) == 1, "Missing approved DefTerm token limit")
    definition_engine.MAX_NEW_TOKENS = limits[0]
    definition_engine.SYSTEM_MSG = definition.SYSTEM_MSG
    definition_engine.USER_MSG_TEMPLATE = definition.USER_MSG_TEMPLATE
    definition_engine.FEWSHOT_USER_MSG_TEMPLATE = definition.USER_MSG_TEMPLATE
    definition_engine.EXPECTED_SYSTEM_PROMPT_SHA256 = definition.SYSTEM_SHA256
    definition_engine.EXPECTED_USER_PROMPT_TEMPLATE_SHA256 = definition.USER_SHA256
    definition_engine.EXPECTED_FEWSHOT_USER_TEMPLATE_SHA256 = definition.USER_SHA256
    definition_engine.build_messages = definition.build_messages
    definition_engine.build_demo_messages = definition.build_demo_messages
    definition_engine.validate_static_contract()
    loo = _project(LOO_FILE, {"FULL_MODE", "LOO_MODE", "EffectiveContext", "effective_context"},
                   {"ENGINE": gate, "require": gate.require, "canonical_json": gate.canonical_json})
    return SimpleNamespace(prep=prep, gate=gate, definition=definition, definition_engine=definition_engine, loo=loo)


@dataclass(frozen=True)
class FewShots:
    gate: tuple[dict, ...]
    definition: tuple[dict, ...]
    provenance: dict

    def validate(self):
        m = methods()
        m.gate.validate_examples(list(self.gate))
        gate_prefix = m.gate.build_demo_messages(m.gate.project_examples(self.gate))
        require(object_hash(gate_prefix) == m.gate.EXPECTED_FEWSHOT_MESSAGES_SHA256.lower(), "Approved gate examples/messages changed")
        require(len(self.definition) == 6 and tuple(row["qid"] for row in self.definition) == m.definition.FEWSHOT_QIDS,
                "Approved definition example membership/order changed")
        require(all(row["manual_status"] == "APPROVED_MANUAL" and row["manual_definition"].strip() for row in self.definition),
                "Definition example is not approved")
        prefix = m.definition.build_demo_messages(m.definition.project_examples(self.definition))
        require(object_hash(prefix) == m.definition.FEWSHOT_MESSAGES_SHA256.lower(), "Approved definition examples/messages changed")

    def verify_files(self):
        for path, checksum in self.provenance["files_sha256"].items():
            require(file_hash(Path(path)) == checksum, "Few-shot resource changed during the run")


def load_fewshots(*, gate_path=None, definition_path=None):
    m = methods()
    gate_path = Path(gate_path) if gate_path is not None else BASE / "05_sufficiency_gate_fewshot_examples.csv"
    definition_path = Path(definition_path) if definition_path is not None else BASE / "05_fewshot_ejemplos_aprobados.csv"
    rows, fingerprints = [], {}
    for path, columns in ((gate_path, m.gate.FEWSHOT_FIELDS), (definition_path, m.definition.FEWSHOT_FIELDS)):
        raw = path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        require(tuple(reader.fieldnames or ()) == tuple(columns), "Unexpected few-shot CSV schema")
        values = tuple(dict(row) for row in reader)
        require(all(None not in row and all(isinstance(value, str) for value in row.values()) for row in values), "Malformed few-shot CSV")
        rows.append(values)
        fingerprints[str(path.resolve())] = digest(raw)
    result = FewShots(rows[0], rows[1], {"files_sha256": fingerprints})
    result.validate()
    return result


def recipe(stage, *, backend_kind=PRODUCTION):
    require(stage in {"gate", "definition"} and backend_kind in {PRODUCTION, CONTROLLED}, "Unknown DefTerm stage/backend")
    m = methods()
    engine = m.gate if stage == "gate" else m.definition_engine
    parameters = json.loads(engine.generation_parameters_json((0,), 0))
    parameters.pop("eos_token_ids")
    parameters.pop("pad_token_id")
    return {"stage": stage, "backend_kind": backend_kind, "model_id": engine.MODEL_ID, "model_revision": engine.MODEL_REVISION,
            "prompt_version": m.gate.PROMPT_VERSION if stage == "gate" else m.definition.PROMPT_VERSION,
            "system_prompt_sha256": engine.sha256_text(engine.SYSTEM_MSG).lower(),
            "user_prompt_sha256": engine.sha256_text(engine.USER_MSG_TEMPLATE).lower(),
            "quantization": json.loads(engine.quantization_config_json()), "generation": parameters,
            "generation_attempts": engine.GENERATION_ATTEMPTS, "eos_pad_policy": "PINNED_MODEL_GENERATION_CONFIG",
            "adapter_sha256": file_hash(Path(__file__))}


def structured_identity(row):
    return "defterm_evidence_" + object_hash({key: row[key] if key in {"qid", "label_en"} else json.loads(row[key])
        for key in ("qid", "label_en", "aliases_en", "p31", "p279", "p1343")})


def gate_request(prepared, examples, *, backend_kind=PRODUCTION):
    examples.validate()
    m = methods()
    selector = prepared
    matches = [row for row in examples.gate if row["defterm_evidence_text"] == prepared["defterm_evidence_text"]]
    require(len(matches) <= 1, "Ambiguous self demonstration")
    if matches:
        # Model-visible self identity survives a different corpus term/document.
        # This projection is used ONLY by the unchanged historical LOO selector.
        selector = {**prepared, **{key: matches[0][key] for key in ("defterm_input_id", "term", "qid")}}
    context = m.loo.effective_context(selector, examples.gate)
    effective = [row for row in examples.gate if row["sufficiency_example_id"] in context.retained_example_ids]
    request = {"stage": "gate", "defterm_evidence_id": structured_identity(prepared), "qid": prepared["qid"],
               "label_en": prepared["label_en"], "defterm_evidence_text": prepared["defterm_evidence_text"],
               "recipe": recipe("gate", backend_kind=backend_kind),
               "examples": [{"evidence": row["defterm_evidence_text"], "decision": row["sufficiency_label"]} for row in effective],
               "messages": m.gate.build_messages(prepared["defterm_evidence_text"], context.examples),
               "fewshot": {**context.provenance(), "retained_example_ids": list(context.retained_example_ids),
                           "self_identity_policy": "LEAVE_SELF_OUT_BY_EQUIVALENT_MODEL_EVIDENCE_V1"}}
    return {"task_id": "defterm_gate_" + object_hash(request), "request": request}


def definition_request(prepared, examples, certificate, *, backend_kind=PRODUCTION):
    examples.validate()
    m = methods()
    request = {"stage": "definition", "defterm_evidence_id": structured_identity(prepared), "qid": prepared["qid"],
               "label_en": prepared["label_en"], "defterm_evidence_text": prepared["defterm_evidence_text"],
               "recipe": recipe("definition", backend_kind=backend_kind), "gate_certificate": certificate,
               "examples": [{"evidence": row["defterm_evidence_text"], "definition": row["manual_definition"]} for row in examples.definition],
               "messages": m.definition.build_messages(prepared["defterm_evidence_text"], m.definition.project_examples(examples.definition)),
               "fewshot": {"effective_fewshot_count": 6, "retained_example_ids": [row["fewshot_example_id"] for row in examples.definition],
                           "messages_prefix_sha256": m.definition.FEWSHOT_MESSAGES_SHA256.lower()}}
    require(certificate["decision"] == m.gate.SUFFICIENT_EVIDENCE and certificate["defterm_evidence_id"] == request["defterm_evidence_id"],
            "Definition requires matching sufficient gate evidence")
    return {"task_id": "defterm_definition_" + object_hash(request), "request": request}


def validate_task(task):
    request = task["request"]
    stage = request["stage"]
    require(task["task_id"] == "defterm_" + stage + "_" + object_hash(request), "DefTerm task identity changed")
    require(request["recipe"] == recipe(stage, backend_kind=request["recipe"]["backend_kind"]), "DefTerm recipe differs from original method")
    m = methods()
    projected = (tuple(m.gate.PromptExample(row["evidence"], row["decision"]) for row in request["examples"]) if stage == "gate" else
                 tuple(m.definition.DefinitionExample(row["evidence"], row["definition"]) for row in request["examples"]))
    engine = m.gate if stage == "gate" else m.definition_engine
    require(request["messages"] == engine.build_messages(request["defterm_evidence_text"], projected), "DefTerm message boundary differs")
    return engine, projected


def process_response(task, response):
    """Reapply ORIGINAL process_candidate with a raw-response callback, not an LLM."""
    engine, _ = validate_task(task)
    request = task["request"]
    require(response["task_id"] == task["task_id"] and isinstance(response["raw"], str)
            and response["call_status"] in {"RETURNED", "FAILED_OR_UNKNOWN"}, "Invalid DefTerm response")
    require(response["finish_reason"] in {"eos_token", "max_new_tokens", "unknown_stop", "generation_exception", "input_context_limit"},
            "Invalid DefTerm finish reason")
    for key in ("input_token_count", "generated_token_count"):
        require(response[key] is None or type(response[key]) is int and response[key] >= 0, "Invalid DefTerm token count")
    if response["finish_reason"] == "eos_token":
        require(type(response["generated_token_count"]) is int and response["generated_token_count"] > 0, "EOS was not observed")
    runtime = response["runtime"]
    require(isinstance(runtime, dict) and runtime.get("backend_kind") == request["recipe"]["backend_kind"], "DefTerm runtime identity differs")
    if request["recipe"]["backend_kind"] == CONTROLLED:
        require(runtime.get("test_only") is True, "Synthetic response is not explicitly marked")
    elif response["call_status"] == "RETURNED":
        require(runtime.get("local_files_only") is True and "H100" in runtime.get("device_name", "").upper(),
                "DefTerm runtime does not identify local H100 execution")
        actual = json.loads(runtime["generation_parameters"])
        eos, pad = actual["eos_token_ids"], actual["pad_token_id"]
        require(runtime["generation_parameters"] == engine.generation_parameters_json(engine.normalize_token_ids(eos, "cached EOS"), pad),
                "Effective DefTerm generation parameters differ")
        versions = json.loads(runtime["runtime_versions"])
        require(versions["resolved_model_commit"] == request["recipe"]["model_revision"] and versions["model_loaded_in_4bit"] is True
                and versions["model_primary_device"] == "cuda:0" and versions["torch_deterministic_algorithms"] is True
                and versions["cuda_matmul_allow_tf32"] is False and versions["cudnn_allow_tf32"] is False
                and versions["cudnn_benchmark"] is False and versions["cudnn_deterministic"] is True,
                "Effective DefTerm model/determinism differs")
        for package in engine.MINIMUM_RUNTIME_VERSIONS:
            engine.validate_supported_version(package, versions[package])
    source = {key: request[key] for key in TRACE_FIELDS}
    metadata = {"model_id": request["recipe"]["model_id"], "model_revision": request["recipe"]["model_revision"],
                "metadata_mode": request["recipe"]["backend_kind"], "prompt_version": request["recipe"]["prompt_version"],
                "recipe_sha256": object_hash(request["recipe"]), "runtime_metadata": runtime}
    if request["stage"] == "definition":
        cert = request["gate_certificate"]
        require(cert["defterm_evidence_id"] == request["defterm_evidence_id"], "Gate certificate targets different evidence")
        source.update(sufficiency_decision=cert["decision"], sufficiency_gate_result_id=cert["gate_result_id"],
                      post_gate_partition_id=cert["certificate_id"])

    def raw_generator(*_):
        if response["call_status"] != "RETURNED":
            raise RuntimeError("GENERATION_FAILED_DETAILS_NOT_PERSISTED")
        return response["raw"]

    operation = engine.process_candidate if request["stage"] == "gate" else methods().definition.process_candidate
    result = operation(source, None, None, metadata, raw_generator)
    if result["error_message"]:
        result["error_message"] = "ORIGINAL_DEFTERM_VALIDATION_OR_GENERATION_ERROR"
    complete = response["call_status"] == "RETURNED" and response["finish_reason"] == "eos_token"
    result.update(completion_status="COMPLETE" if complete else "INCOMPLETE_OR_UNKNOWN", fidelity_status="NOT_VERIFIED")
    if request["stage"] == "gate":
        result["accepted_decision"] = result["sufficiency_decision"] if complete and result["generation_status"] == "OK" else None
    else:
        result["available_for_later_evaluation"] = complete and result["generation_status"] == "OK" and result["circularity_flag_final"] == "false"
    return result


def generation_failure(task, *, finish_reason="generation_exception", runtime=None):
    return {"task_id": task["task_id"], "raw": "", "call_status": "FAILED_OR_UNKNOWN", "finish_reason": finish_reason,
            "input_token_count": None, "generated_token_count": None,
            "runtime": runtime or {"backend_kind": task["request"]["recipe"]["backend_kind"],
                                   **({"test_only": True} if task["request"]["recipe"]["backend_kind"] == CONTROLLED else {})}}


class _ObservedModel:
    def __init__(self, model):
        self.model = model
        self.generation_config = model.generation_config
        self.input_tokens = self.generated_tokens = None
        self.finish_reason = "generation_exception"

    def parameters(self):
        return self.model.parameters()

    def generate(self, **kwargs):
        self.input_tokens = int(kwargs["input_ids"].shape[-1])
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(maximum, int) and self.input_tokens + kwargs["max_new_tokens"] > maximum:
            self.finish_reason = "input_context_limit"
            raise ValueError("Input exceeds model context")
        output = self.model.generate(**kwargs)
        tokens = output[0, self.input_tokens:].tolist()
        self.generated_tokens = len(tokens)
        eos = kwargs["eos_token_id"]
        if tokens and tokens[-1] in (eos if isinstance(eos, list) else [eos]):
            self.finish_reason = "eos_token"
        elif len(tokens) >= kwargs["max_new_tokens"]:
            self.finish_reason = "max_new_tokens"
        else:
            self.finish_reason = "unknown_stop"
        return output


class QwenDefTerm:
    """Local-only pinned weights, shared lazily by gate and definition stages."""
    backend_kind = PRODUCTION

    def __init__(self, *, cache_dir=None):
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.model = self.tokenizer = self.torch = None
        self.runtime = {}

    def prepare(self):
        if self.model is not None:
            return
        import os
        from finterm_sense2_qwen import QwenContextualV2

        m = methods()
        try:
            resources = QwenContextualV2(cache_dir=self.cache_dir)
            snapshot, resource_hashes = resources._snapshot()  # Existing resource checks, no model load.
            require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {None, m.gate.CUBLAS_WORKSPACE_CONFIG}, "Conflicting CUBLAS configuration")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = m.gate.CUBLAS_WORKSPACE_CONFIG
            for package in m.gate.MINIMUM_RUNTIME_VERSIONS:
                m.gate.validate_supported_version(package, m.gate.installed_version(package))
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "H100" in torch.cuda.get_device_name(0).upper(), "H100/bfloat16 required")
            require(not (Path.cwd() / m.gate.MODEL_ID).exists(), "Local path shadows pinned model")
            m.gate.configure_determinism(torch)
            quantization = BitsAndBytesConfig(**{**json.loads(m.gate.quantization_config_json()), "bnb_4bit_compute_dtype": torch.bfloat16})
            kwargs = {"revision": m.gate.MODEL_REVISION, "cache_dir": self.cache_dir, "local_files_only": True, "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(m.gate.MODEL_ID, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(m.gate.MODEL_ID, device_map={"": 0},
                       quantization_config=quantization, torch_dtype=torch.bfloat16, **kwargs)
            model.eval()
            runtime = {}
            for stage, engine in (("gate", m.gate), ("definition", m.definition_engine)):
                actual = engine.collect_runtime_metadata({}, tokenizer, model, torch)
                runtime[stage] = {"backend_kind": PRODUCTION, "snapshot_path": str(snapshot), "resource_json_sha256": resource_hashes,
                                  "local_files_only": True, **actual}
        except Exception:
            raise RuntimeError("DEFTERM_LOCAL_MODEL_UNAVAILABLE_CHECK_PINNED_CACHE_AND_H100_ENVIRONMENT") from None
        self.model, self.tokenizer, self.torch, self.runtime = model, tokenizer, torch, runtime

    def generate(self, task):
        self.prepare()
        engine, projected = validate_task(task)
        stage = task["request"]["stage"]
        observer = _ObservedModel(self.model)
        try:
            # Original renderer, generation, EOS handling and decode, unchanged.
            raw = engine.generate_once(self.tokenizer, observer, task["request"]["defterm_evidence_text"], projected, self.torch)
        except Exception:
            response = generation_failure(task, finish_reason=observer.finish_reason, runtime=self.runtime[stage])
        else:
            response = {"task_id": task["task_id"], "raw": raw, "call_status": "RETURNED", "finish_reason": observer.finish_reason,
                        "input_token_count": observer.input_tokens, "generated_token_count": observer.generated_tokens,
                        "runtime": self.runtime[stage]}
        response["input_token_count"], response["generated_token_count"] = observer.input_tokens, observer.generated_tokens
        return response
