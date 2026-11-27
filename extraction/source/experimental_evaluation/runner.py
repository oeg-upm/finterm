"""Generation ONLY. Reuses original prompts, adapters, filtering and ranking.

The runner deliberately composes these public components rather than copying
model/tokenizer logic or modifying TermExtractor: retrieval must be durable and
identical across Qwen/Gemma, and generation must commit before interpretation.
"""

import gc
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

from llm_adapters import adapter_from_profile
from prompts import PromptBuilder, build_dister_prompt, prompt_settings_from_toml
from syntactic_retrieval import AnnotatedExample, RetrievedExample

from .common import atomic_json, digest, outside, plain, read_json, utc_now
from .data import read_query
from .formats import FORMAT_POLICY, interpret
from .provenance import check_versions, environment, model_inventory, profile_from_dict, verify_code
from .resources import Resources, resource_inventory
from .syntax import ForestConfig, ForestParser, OriginalTimedSimilarity, make_retriever


def retrieval_diagnostics(retriever):
    return {"parse_reports": getattr(retriever.parser, "reports", {}),
            "score_reports": getattr(retriever.similarity, "reports", [])}


def observe_runtime(adapter):
    runtime = adapter.load()
    # Observe the SAME source-frozen config builder used by generate(); never
    # mutate the runtime, model config, options, tokenization or EOS settings.
    effective = adapter._generation_config(runtime)
    value = effective.to_dict() if hasattr(effective, "to_dict") else vars(effective)
    config = runtime.model.config
    hardware = {}
    if hasattr(runtime.torch, "cuda") and runtime.torch.cuda.is_available():
        hardware = {"name": runtime.torch.cuda.get_device_name(0),
                    "capability": runtime.torch.cuda.get_device_capability(0),
                    "visible_memory_bytes": runtime.torch.cuda.get_device_properties(0).total_memory}
    return json.loads(json.dumps({"effective_generation_config": value, "gpu": hardware,
                                 "model_context_limit": getattr(config, "max_position_embeddings", None),
                                 "model_type": getattr(config, "model_type", None),
                                 "runtime_model_dtype": str(getattr(runtime.model, "dtype", None))}, default=str))


def termination(generation, budget, runtime):
    eos = runtime["effective_generation_config"].get("eos_token_id")
    eos = [eos] if isinstance(eos, int) else (eos or [])
    tokens = generation["generated_token_ids"]
    end_seen = bool(tokens and tokens[-1] in eos)
    return {"new_token_count": len(tokens), "max_new_tokens": budget,
            "budget_reached": len(tokens) >= budget, "effective_eos_token_ids": eos,
            "last_token_is_eos": end_seen,
            "budget_exhaustion_suspected_not_proven": len(tokens) >= budget and not end_seen,
            "finish_reason": "not_exposed_by_existing_adapter"}


class ExperimentRunner:
    def __init__(self, store, *, project, corpus_root, adapter_factory=adapter_from_profile,
                 retriever_factory=None):
        self.store, self.project, self.corpus_root = store, Path(project), Path(corpus_root)
        self.manifest = store.manifest
        self.profiles = {k: profile_from_dict(v) for k, v in self.manifest["profiles"].items()}
        self.examples = tuple(AnnotatedExample(**{**e, "terms": tuple(e["terms"])})
                              for e in self.manifest["bank"]["examples"])
        self.example_by_id = {e.identifier: e for e in self.examples}
        self.prompt_config = prompt_settings_from_toml(self.manifest["prompt_toml"]).for_domain("finance")
        self.adapter_factory, self.retriever_factory = adapter_factory, retriever_factory
        self.retriever = None

    def _verify(self, model):
        check_versions()
        verify_code(self.project, self.manifest["code"])
        if environment() != self.manifest["environment"]:
            raise ValueError("Runtime environment changed; use a different run, not silent resume")
        if resource_inventory(Resources(**self.manifest["resources"])) != self.manifest["resource_inventory"]:
            raise ValueError("Stanford/Java/Punkt resources changed")
        if model_inventory(self.profiles[model], hash_weights=self.manifest["model_files"][model]["weight_content_hashes_verified"]) != self.manifest["model_files"][model]:
            raise ValueError("Model/tokenizer metadata or shard inventory changed")

    def _selection(self, query):
        path = self.store.root / "retrieval" / (digest(query["identifier"]) + ".json")
        if path.exists():
            value = read_json(path)
            if value["manifest_sha256"] != self.store.fingerprint or value["query_sha256"] != query["text_sha256"]:
                raise ValueError("Retrieval cache belongs to different inputs/configuration")
        else:
            if self.retriever is None:
                if self.retriever_factory is not None:
                    self.retriever = self.retriever_factory(self.examples)
                else:
                    config = ForestConfig(**self.manifest["forest"])
                    self.retriever = make_retriever(
                        self.examples, ForestParser(Resources(**self.manifest["resources"]), config),
                        OriginalTimedSimilarity(config),
                    )
            began = time.monotonic()
            retrieved = self.retriever.retrieve(query["text"], query_id=query["identifier"],
                                                query_document_id=query["document_id"], top_k=3)
            if len(retrieved) != 3:
                raise ValueError("Fewer than three authorized demonstrations survived; no padding")
            value = {"manifest_sha256": self.store.fingerprint, "query_sha256": query["text_sha256"],
                     "selected": [{"identifier": item.example.identifier, "score": item.score} for item in retrieved],
                     "seconds": time.monotonic() - began, **retrieval_diagnostics(self.retriever)}
            atomic_json(path, value)
        selected = tuple(RetrievedExample(self.example_by_id[item["identifier"]], item["score"])
                         for item in value["selected"])
        if len(selected) != 3 or len({r.example.identifier for r in selected}) != 3:
            raise ValueError("Invalid frozen retrieval selection")
        return selected, value

    def _interpret_saved(self, condition, query, receipt):
        path = self.store.job_path(condition, query["identifier"]) / "formats_v1.json"
        if not path.exists():
            model = self.manifest["conditions"][condition][0]
            atomic_json(path, interpret(receipt["generation"]["response_text"],
                                        "native_json_array" if model == "dister" else "terms_object"))

    def run_configuration(self, condition, *, segment_ids, retry_errors=False):
        if condition not in self.manifest["conditions"]:
            raise ValueError("Unknown condition")
        return self._run([condition], segment_ids, retry_errors)

    def run_model(self, model, *, segment_ids, retry_errors=False):
        """At most one LLM resident; basic/few-shot share that model instance."""
        conditions = [c for c, (key, _) in self.manifest["conditions"].items() if key == model]
        if not conditions:
            raise ValueError("Unknown model")
        return self._run(conditions, segment_ids, retry_errors)

    def _run(self, conditions, segment_ids, retry_errors):
        ids = tuple(segment_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Provide the explicit small subset to execute; there is no run-all default")
        query_map = {q["identifier"]: q for q in self.manifest["queries"]}
        if set(ids) - query_map.keys():
            raise ValueError("Selected IDs are outside the frozen sample")
        model = self.manifest["conditions"][conditions[0]][0]
        self._verify(model)
        profile = self.profiles[model]
        adapter = None
        counts = {"generated": 0, "resumed_without_inference": 0, "previous_errors_skipped": 0, "errors": 0}
        with self.store.writer():
            try:
                for condition in conditions:
                    for identifier in ids:
                        query = query_map[identifier]
                        previous = self.store.generation(condition, identifier)
                        if previous is not None:
                            self._interpret_saved(condition, query, previous)
                            counts["resumed_without_inference"] += 1
                            continue
                        if self.store.failed(condition, identifier) and not retry_errors:
                            counts["previous_errors_skipped"] += 1
                            continue
                        began = time.monotonic(); phase = "input"; request = None
                        try:
                            text = read_query(query, self.corpus_root)
                            builder = PromptBuilder(self.prompt_config)
                            _, method = self.manifest["conditions"][condition]
                            selection = None
                            if method == "few_shot":
                                phase = "syntax"
                                retrieved, selection = self._selection(query)
                                prompt = builder.build_few_shot(text, retrieved)
                            elif method == "basic":
                                prompt = builder.build_basic(text)
                            else:
                                prompt = build_dister_prompt(identifier, text, self.prompt_config.domain.name)
                            request = {"manifest_sha256": self.store.fingerprint, "condition": condition,
                                       "query": query, "profile": plain(profile), "time": utc_now(),
                                       "prompt": prompt.to_dict() if method == "native_zero_shot" else prompt.to_messages(),
                                       "prompt_metadata": None if method == "native_zero_shot" else prompt.metadata.to_dict(),
                                       "retrieval": selection}
                            job = self.store.job_path(condition, identifier)
                            request_id = uuid.uuid4().hex
                            atomic_json(job / "requests" / (request_id + ".json"), request)
                            phase = "load"
                            if adapter is None:
                                adapter = self.adapter_factory(profile)
                            load_start = time.monotonic(); runtime = observe_runtime(adapter)
                            load_seconds = time.monotonic() - load_start
                            phase = "generation"; gen_start = time.monotonic()
                            result = adapter.generate(prompt)
                            generation_seconds = time.monotonic() - gen_start
                            generation = result.to_dict()
                            receipt = {"request_id": request_id, "request": request, "generation": generation,
                                       "runtime": runtime, "generation_seconds": generation_seconds,
                                       "load_seconds": load_seconds, "total_seconds": time.monotonic() - began,
                                       "termination": termination(generation, profile.generation.max_new_tokens, runtime),
                                       "completed_at": utc_now()}
                            # Commit the complete original response BEFORE any new parser/scorer.
                            self.store.save_generation(condition, identifier, receipt)
                            counts["generated"] += 1
                            phase = "interpretation"
                            self._interpret_saved(condition, query, receipt)
                        except Exception as exc:
                            self.store.save_error(condition, identifier,
                                {"manifest_sha256": self.store.fingerprint, "condition": condition, "query": query,
                                 "profile": plain(profile), "phase": phase, "request": request,
                                 "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                                 "syntax_report": getattr(exc, "report", None),
                                 "retrieval_diagnostics": retrieval_diagnostics(self.retriever) if self.retriever else None,
                                 "seconds": time.monotonic() - began, "time": utc_now()})
                            counts["errors"] += 1
            finally:
                adapter = None
                gc.collect()
                # No torch import here; only release caches if the real run loaded it.
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return counts


def syntax_probe(*, queries, bank, resources, forest_config, output_root, corpus_root,
                 project, expected_code):
    """Explicit syntax-only rehearsal. Never constructs an LLM adapter."""
    check_versions(); verify_code(project, expected_code)
    root = outside(output_root, [corpus_root]); root.mkdir(parents=True, exist_ok=False)
    inventory = resource_inventory(resources)
    atomic_json(root / "manifest.json", {"strategy": plain(forest_config), "resources": inventory,
                                         "queries": queries, "bank": bank.snapshot(), "time": utc_now()})
    parser, similarity = ForestParser(resources, forest_config), OriginalTimedSimilarity(forest_config)
    retriever = make_retriever(bank.examples, parser, similarity)
    results = []
    for query in queries:
        began = time.monotonic()
        result = {"query": query}
        try:
            text = read_query(query, corpus_root)
            selected = retriever.retrieve(text, query_id=query["identifier"],
                                          query_document_id=query["document_id"], top_k=3)
            if len(selected) != 3:
                raise ValueError("Insufficient authorized demonstrations")
            result.update(status="ok", selected=[r.to_dict() for r in selected])
        except Exception as exc:
            result.update(status="error", type=type(exc).__name__, message=str(exc),
                          syntax_report=getattr(exc, "report", None), traceback=traceback.format_exc())
        result.update(seconds=time.monotonic() - began, **retrieval_diagnostics(retriever))
        atomic_json(root / (digest(query["identifier"]) + ".json"), result)
        results.append(result)
    return results
