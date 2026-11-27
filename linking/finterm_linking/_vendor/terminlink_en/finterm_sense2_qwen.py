"""Internal Qwen contextual V2 adapter: approved prompts/generation, local files only.

Imports no ML library until prepare(). The historical generator itself is reused
through a transparent observer so token counts and EOS/limit stops are retained.
"""
from __future__ import annotations

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

from finterm_sense2 import BASE, canonical, digest, file_hash, legacy, lexical_count, require


PRODUCTION_BACKEND = "QWEN_CONTEXTUAL_V2_LOCAL"
TEST_BACKEND = "CONTROLLED_TEST_NO_MODEL"
VALIDATION_POLICY = "contextual-qwen14-v2-posthoc-format-revalidation-v1"


def generation_contract(*, backend_kind=PRODUCTION_BACKEND):
    require(backend_kind in {PRODUCTION_BACKEND, TEST_BACKEND}, "Unsupported generation backend")
    runtime = legacy("qwen14_contextual_v2_runtime.py")
    runtime.validate_prompt_contract()
    return {"backend_kind": backend_kind, "model_id": runtime.MODEL_ID, "model_revision": runtime.MODEL_REVISION,
            "prompt_version": runtime.PROMPT_VERSION, "system_prompt_sha256": runtime.EXPECTED_SYSTEM_MESSAGE_SHA256.lower(),
            "user_prompt_template_sha256": runtime.EXPECTED_USER_TEMPLATE_SHA256.lower(),
            "quantization": dict(runtime.FIXED_QUANTIZATION_PARAMETERS),
            "generation": {**runtime.FIXED_GENERATION_PARAMETERS, "enable_thinking": runtime.ENABLE_THINKING,
                           "inference_seed": runtime.INFERENCE_SEED, "generation_attempts": runtime.GENERATION_ATTEMPTS,
                           "torch_deterministic_algorithms": True},
            "runtime_module_sha256": file_hash(BASE / "qwen14_contextual_v2_runtime.py"),
            "context_renderer_sha256": file_hash(BASE / "03_preparar_traducciones_contextualizadas.py"),
            "adapter_sha256": file_hash(Path(__file__))}


def validate_contract(contract):
    require(isinstance(contract, dict) and contract.get("backend_kind") in {PRODUCTION_BACKEND, TEST_BACKEND},
            "Missing explicit production/test generation identity")
    require(contract == generation_contract(backend_kind=contract["backend_kind"]), "Generation contract differs from approved contextual V2")


def messages_for(evidence_text):
    return legacy("qwen14_contextual_v2_runtime.py").build_messages(evidence_text)


def validate_generation(record, request):
    """Reapply the corrected historical structural validator; never repair text."""
    utility = legacy("03_revalidar_traducciones_contextualizadas_qwen14_v2.py")
    raw = record["translation_raw"]
    require(isinstance(raw, str), "Generation raw output must be text")
    cleaned, error, reason = None, None, None
    structural = "NO_OUTPUT"
    if record["generation_call_status"] == "RETURNED":
        try:
            cleaned = utility.normalize_and_validate_generation(raw)
        except utility.MalformedGenerationError as exc:
            structural, error = "INVALID", "NON_TRANSLATION_STRUCTURE"
            reason = str(exc)  # The reviewed validator emits fixed diagnostics only.
        else:
            structural = "VALID"
    complete = record["generation_call_status"] == "RETURNED" and record["finish_reason"] == "eos_token"
    alerts = []
    if cleaned:
        if utility.audit_normalize(request["source_description"]) == utility.audit_normalize(cleaned):
            alerts.append("POSSIBLY_UNTRANSLATED_NORMALIZED_EQUAL")
        if utility.script_profile(cleaned) not in {"LATIN", "NONE"}:
            alerts.append("NON_LATIN_OR_MIXED_TRANSLATION_SCRIPT")
    return {"policy_version": VALIDATION_POLICY,
            "validator_sha256": file_hash(BASE / "03_revalidar_traducciones_contextualizadas_qwen14_v2.py"),
            "structural_status": structural, "structural_error_code": error, "structural_error_reason": reason, "translation_clean": cleaned,
            "completion_status": "COMPLETE" if complete else "INCOMPLETE_OR_UNKNOWN",
            "fidelity_status": "NOT_VERIFIED", "evidence_sufficiency": "NOT_ASSESSED",
            "lexical_word_count": lexical_count(cleaned) if cleaned else 0,
            "review_flags": alerts, "review_flags_are_semantic_validation": False,
            "available_for_later_evaluation": structural == "VALID" and complete}


def exception_record(task_id, exc, *, metadata=None, finish_reason="generation_exception"):
    # Arbitrary exceptions may contain auth arguments. Their messages are never
    # persisted, nor are environment dictionaries/token values serialized.
    return {"task_id": task_id, "generation_call_status": "RAISED", "translation_raw": "",
            "finish_reason": finish_reason, "input_token_count": None, "generated_token_count": None,
            "error": {"type": type(exc).__name__, "code": "GENERATION_EXCEPTION_DETAILS_NOT_PERSISTED"},
            "metadata": metadata or {}}


class AdapterUnavailable(RuntimeError):
    """Safe, fixed diagnostics; no external exception string or credentials."""


class _ObservedModel:
    def __init__(self, model, eos_token_id, max_new_tokens):
        self.model = model
        self.eos = eos_token_id
        self.limit = max_new_tokens
        self.input_tokens = self.generated_tokens = None
        self.finish_reason = "generation_exception"

    def parameters(self):
        return self.model.parameters()

    def generate(self, **kwargs):
        self.input_tokens = int(kwargs["input_ids"].shape[-1])
        positions = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(positions, int) and self.input_tokens + self.limit > positions:
            self.finish_reason = "input_context_limit"
            raise ValueError("Input and reserved output exceed the model context window")
        output = self.model.generate(**kwargs)
        tokens = output[0, self.input_tokens:].tolist()
        self.generated_tokens = len(tokens)
        if tokens and tokens[-1] == self.eos:
            self.finish_reason = "eos_token"
        elif len(tokens) >= self.limit:
            self.finish_reason = "max_new_tokens"
        else:
            self.finish_reason = "unknown_stop"
        return output


class QwenContextualV2:
    """Lazy, pinned H100/NF4 adapter; reuses an existing HF cache, never downloads.

    cache_dir is the Hugging Face hub cache (containing models--Qwen--Qwen3-14B),
    not the translation SQLite cache. None respects the existing HF configuration.
    Authentication, when needed to populate resources externally, is external.
    """
    def __init__(self, *, cache_dir: Path | str | None = None):
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.contract = generation_contract()
        self.tokenizer = self.model = None
        self.metadata = None

    def _snapshot(self):
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(self.contract["model_id"], "config.json",
                                       revision=self.contract["model_revision"], cache_dir=self.cache_dir)
        if not isinstance(config, str):
            raise AdapterUnavailable("PINNED_QWEN_SNAPSHOT_NOT_CACHED")
        snapshot = Path(config).parent
        if snapshot.name != self.contract["model_revision"] or snapshot.parent.name != "snapshots":
            raise AdapterUnavailable("CACHE_PATH_DOES_NOT_IDENTIFY_PINNED_REVISION")
        required = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if not isinstance(weights, dict) or not weights:
                raise AdapterUnavailable("INVALID_WEIGHT_INDEX")
            names = sorted(set(weights.values()))
            if not all(isinstance(name, str) and Path(name).name == name for name in names):
                raise AdapterUnavailable("INVALID_WEIGHT_INDEX_PATH")
            required.extend(["model.safetensors.index.json", *names])
        else:
            required.append("model.safetensors")
        if not all((snapshot / name).is_file() and (snapshot / name).stat().st_size for name in required):
            raise AdapterUnavailable("INCOMPLETE_PINNED_MODEL_OR_TOKENIZER")
        return snapshot, {name: file_hash(snapshot / name) for name in required if name.endswith(".json")}

    def prepare(self):
        validate_contract(self.contract)
        if self.model is not None:
            return
        runtime = legacy("qwen14_contextual_v2_runtime.py")
        try:
            snapshot, resource_hashes = self._snapshot()
            # This check precedes ML initialization. The original V2 determinism
            # requirement must also hold in an already-running Jupyter kernel.
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise AdapterUnavailable("SET_CUBLAS_WORKSPACE_CONFIG_AND_RESTART_KERNEL")
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0).upper() or not torch.cuda.is_bf16_supported():
                raise AdapterUnavailable("CONTEXTUAL_V2_REQUIRES_H100_BFLOAT16")
            torch.manual_seed(runtime.INFERENCE_SEED)
            torch.cuda.manual_seed_all(runtime.INFERENCE_SEED)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            kwargs = {"local_files_only": True, "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), **kwargs)
            model = AutoModelForCausalLM.from_pretrained(str(snapshot), device_map={"": 0},
                       quantization_config=quantization, torch_dtype=torch.bfloat16, **kwargs)
            model.eval()
            require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
            metadata = {"backend_kind": PRODUCTION_BACKEND, "model_id": runtime.MODEL_ID,
                        "model_revision": runtime.MODEL_REVISION, "snapshot_path": str(snapshot),
                        "resource_json_sha256": resource_hashes, "python_version": platform.python_version(),
                        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                        "cuda_version": str(torch.version.cuda), "device_name": torch.cuda.get_device_name(0),
                        "runtime_versions": {name: version(name) for name in
                            ("tokenizers", "huggingface-hub", "accelerate", "bitsandbytes", "safetensors")},
                        "quantization": dict(runtime.FIXED_QUANTIZATION_PARAMETERS),
                        "generation": {**self.contract["generation"], "eos_token_id": tokenizer.eos_token_id,
                                       "pad_token_id": tokenizer.eos_token_id},
                        "local_files_only": True}
        except AdapterUnavailable:
            raise
        except Exception:
            raise AdapterUnavailable("LOCAL_QWEN_INITIALIZATION_FAILED_CHECK_RESOURCES_AND_ENVIRONMENT") from None
        self.tokenizer, self.model, self.metadata = tokenizer, model, metadata

    def generate(self, task):
        self.prepare()
        require(task["request"]["generation_contract"] == self.contract, "Task/backend contract differs")
        runtime = legacy("qwen14_contextual_v2_runtime.py")
        observer = _ObservedModel(self.model, self.tokenizer.eos_token_id, runtime.MAX_NEW_TOKENS)
        try:
            # Exact approved rendering/tokenization/greedy call, with observation
            # only. No retry, semantic repair or first-sentence truncation.
            raw = runtime.generate_once(self.tokenizer, observer, task["request"]["translation_evidence_text"])
            require(isinstance(raw, str), "Model output is not text")
        except Exception as exc:
            result = exception_record(task["task_id"], exc, metadata=self.metadata, finish_reason=observer.finish_reason)
        else:
            result = {"task_id": task["task_id"], "generation_call_status": "RETURNED", "translation_raw": raw,
                      "finish_reason": observer.finish_reason, "input_token_count": observer.input_tokens,
                      "generated_token_count": observer.generated_tokens, "error": None, "metadata": self.metadata}
        result["input_token_count"] = observer.input_tokens
        result["generated_token_count"] = observer.generated_tokens
        return result
