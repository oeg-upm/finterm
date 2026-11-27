"""Default financial product API: arbitrary text -> Punkt -> one exact union."""

from dataclasses import dataclass, replace
from copy import deepcopy
import hashlib
from pathlib import Path
import time
import traceback

from llm_adapters import Qwen3Adapter, Gemma2Adapter, DiSTERLlamaAdapter, AdapterProfile
from experimental_evaluation.common import plain
from experimental_evaluation.formats import interpret
from experimental_evaluation.runner import observe_runtime, termination
from experimental_evaluation.provenance import model_inventory

from .financial_prompts import FinancialPromptBuilder, PROMPT_VERSION, build_financial_dister
from .models import model_key, UnsupportedModelConditionError
from .pipeline import TermExtractor
from .segmentation import PunktSegmenter, sentence_slices


VERSION = "financial_sentence_extractor_v1"
MODEL_REVISIONS = {
    "qwen3": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "gemma2": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "dister": "89aaca22eac19ea7b617e28de05dbce651b1a955",
}
ADAPTERS = {"qwen3": Qwen3Adapter, "gemma2": Gemma2Adapter, "dister": DiSTERLlamaAdapter}


def exact_unique(collections):
    result, seen = [], set()
    for values in collections:
        for value in values:
            if value not in seen:
                seen.add(value); result.append(value)
    return result


@dataclass(frozen=True)
class FinancialExtractionResult:
    text: str
    model: str
    terms: tuple[str, ...]
    sentences: tuple[dict, ...]
    complete: bool
    segmentation: dict
    profile: dict
    condition: str
    prompt_version: str = PROMPT_VERSION
    program_version: str = VERSION
    mode: str | None = None
    mode_config: dict | None = None

    def to_dict(self):
        return plain(self)


class FinancialExtractionError(RuntimeError):
    def __init__(self, result):
        super().__init__("Extraction is incomplete; inspect sentence errors in the detailed result")
        self.result = result


class FinancialExtractor:
    """Reusable standard API. Construction does not load Punkt, Java or an LLM.

    `extract` returns ONE list, or raises rather than silently returning a
    partial extraction. `extract_detailed` retains originals and all errors.
    """
    def __init__(self, model="qwen3", *, adapter=None, punkt_data=None, segmenter=None, retriever=None):
        self.model = model_key(model)
        self.adapter = adapter if adapter is not None else ADAPTERS[self.model]()
        TermExtractor._validate_adapter(self.model, self.adapter)
        if self.model == "dister" and retriever is not None:
            raise UnsupportedModelConditionError("DiSTER uses its native conversation without additional demonstrations")
        self.segmenter = segmenter if segmenter is not None else PunktSegmenter(punkt_data)
        self.retriever = retriever
        self.builder = FinancialPromptBuilder()
        self.model_files = None
        self.mode = None
        self.mode_config = None

    @classmethod
    def from_local(cls, model, model_path, *, punkt_data=None, revision=None,
                   max_new_tokens=None, retriever=None):
        key = model_key(model)
        if ".." in Path(model_path).parts:
            raise ValueError("Model path must not contain '..' components")
        source = Path(model_path).expanduser().resolve()
        if not source.is_dir() or not (source / "config.json").is_file():
            raise ValueError("Provide an existing, complete local model snapshot directory")
        base = ADAPTERS[key]()
        load = replace(base.load_config, source=str(source), revision=revision or MODEL_REVISIONS[key],
                       local_files_only=True)
        generation = base.generation_options
        if max_new_tokens is not None:
            generation = replace(generation, max_new_tokens=max_new_tokens)
        # Opt in ONLY for the functional local-checkpoint API. Historical
        # evaluators keep model_inventory's snapshot-only default policy.
        inventory = model_inventory(AdapterProfile(key, base.model_id, load, generation), allow_hf_cache_blobs=True)
        instance = cls(key, adapter=ADAPTERS[key](load=load, generation=generation),
                       punkt_data=punkt_data, retriever=retriever)
        instance.model_files = inventory
        return instance

    def extract(self, text, *, query_document_id=None):
        result = self.extract_detailed(text, query_document_id=query_document_id)
        if not result.complete:
            raise FinancialExtractionError(result)
        return list(result.terms)

    def extract_detailed(self, text, *, query_document_id=None):
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        TermExtractor._validate_adapter(self.model, self.adapter)
        document = query_document_id if query_document_id is not None else "user-input:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not isinstance(document, str) or not document.strip():
            raise ValueError("query_document_id must be a nonempty identity")
        segmented = self.segmenter.segment(text)
        # Validate even injected segmenters: never silently omit input content.
        sentences = sentence_slices(text, [(s.start, s.end) for s in segmented])
        records = []
        for sentence in sentences:
            identifier = document + f":sentence:{sentence.number:06d}"
            record = {"identifier": identifier, "number": sentence.number, "start": sentence.start,
                      "end": sentence.end, "text": sentence.text, "terms": [], "error": None,
                      "prompt": None, "generation": None, "runtime": None, "retrieved_examples": []}
            started, phase = time.monotonic(), "prompt"
            try:
                if self.model == "dister":
                    prompt = build_financial_dister(identifier, sentence.text)
                    record["prompt"] = prompt.to_dict()
                else:
                    if self.retriever is None:
                        prompt = self.builder.build_basic(sentence.text)
                    else:
                        phase = "retrieval"
                        selected = self.retriever.retrieve(sentence.text, query_id=identifier,
                                                           query_document_id=document, top_k=3)
                        prompt = self.builder.build_few_shot(sentence.text, selected)
                        record["retrieved_examples"] = [item.to_dict() for item in selected]
                        record["retrieval_diagnostics"] = deepcopy({
                            "parse_reports": getattr(self.retriever.parser, "reports", {}),
                            "score_reports": getattr(self.retriever.similarity, "reports", []),
                            "bank_status": getattr(self.retriever, "bank_status", "externally_supplied")})
                    record["prompt"] = prompt.to_messages()
                    record["prompt_metadata"] = prompt.metadata.to_dict()
                phase = "load"; began = time.monotonic()
                record["runtime"] = observe_runtime(self.adapter)
                record["load_seconds"] = time.monotonic() - began
                phase = "generation"; began = time.monotonic()
                generated = self.adapter.generate(prompt)
                record["generation_seconds"] = time.monotonic() - began
                record["generation"] = generated.to_dict()
                record["termination"] = termination(record["generation"], self.adapter.generation_options.max_new_tokens,
                                                     record["runtime"])
                phase = "format"
                formats = interpret(generated.response_text, "native_json_array" if self.model == "dister" else "terms_object")
                record["formats"] = formats
                accepted = formats["recovered"]
                if not accepted["accepted"]:
                    raise ValueError("Uninterpretable output under the predefined strict/recovery policy")
                record["terms"] = list(accepted["terms"])
                # Observational only: no silent term rewriting/removal or acronym injection.
                record["nonliteral_in_sentence"] = [term for term in record["terms"] if term not in sentence.text]
                record["nonliteral_in_input"] = [term for term in record["terms"] if term not in text]
            except Exception as exc:
                record["error"] = {"phase": phase, "type": type(exc).__name__, "message": str(exc),
                                   "traceback": traceback.format_exc(), "syntax_report": getattr(exc, "report", None),
                                   "worker_diagnostics": getattr(exc, "diagnostics", None)}
            finally:
                record["seconds"] = time.monotonic() - started
            records.append(record)
        return FinancialExtractionResult(
            text, self.model, tuple(exact_unique(r["terms"] for r in records)), tuple(records),
            all(r["error"] is None for r in records), dict(getattr(self.segmenter, "info", {})),
            {"model_id": self.adapter.model_id, "load": plain(self.adapter.load_config),
             "generation": plain(self.adapter.generation_options), "local_files": self.model_files},
            "native" if self.model == "dister" else ("few_shot" if self.retriever is not None else "basic"),
            mode=self.mode, mode_config=deepcopy(self.mode_config),
        )
