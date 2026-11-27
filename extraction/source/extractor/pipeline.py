"""Coordinate approved retrieval, prompts and adapters for a single input."""

from typing import Iterable, Mapping

from llm_adapters import (
    AdapterProfile, DiSTERLlamaAdapter, Gemma2Adapter, HuggingFaceAdapter,
    MODEL_IDS, Qwen3Adapter, adapter_from_profile,
)
from prompts import PromptBuilder, PromptConfig, build_dister_prompt
from syntactic_retrieval import AnnotatedExample, Settings, build_retriever
from syntactic_retrieval.interfaces import ConstituencyParser
from syntactic_retrieval.models import identifier_set

from .config import ExtractorConfig
from .models import (
    ExtractionCondition, ExtractionRequest, ExtractionResult,
    InvalidExtractionRequest, MissingRetrievalResourcesError, model_key,
)


class TermExtractor:
    """Corpus-independent single-text extraction; execution is explicit.

    Construction does not initialize a parser, retrieve examples or load an LLM.
    The supplied bank contains existing AnnotatedExample objects, never generated
    annotations. Only few_shot constructs/calls the existing retriever.
    """

    def __init__(
        self,
        config: ExtractorConfig | None = None,
        *,
        examples: Iterable[AnnotatedExample] = (),
        retrieval_settings: Settings | None = None,
        parser: ConstituencyParser | None = None,
        adapters: Mapping[str, HuggingFaceAdapter] | None = None,
        adapter_profiles: Mapping[str, AdapterProfile] | None = None,
        evaluation_example_ids: Iterable[str] = (),
        evaluation_document_ids: Iterable[str] = (),
    ) -> None:
        self.config = config if config is not None else ExtractorConfig()
        if not isinstance(self.config, ExtractorConfig):
            raise TypeError("Expected ExtractorConfig")
        self._examples = tuple(examples)
        if not all(isinstance(example, AnnotatedExample) for example in self._examples):
            raise TypeError("Provide an external bank of existing AnnotatedExample objects")
        if retrieval_settings is not None and not isinstance(retrieval_settings, Settings):
            raise TypeError("retrieval_settings must be existing syntactic_retrieval.Settings")
        self._retrieval_settings = retrieval_settings
        self._parser = parser
        self._retriever = None
        self._evaluation_ids = identifier_set(evaluation_example_ids, "evaluation_example_ids")
        self._evaluation_docs = identifier_set(evaluation_document_ids, "evaluation_document_ids")
        self._adapters = {}
        for name, adapter in (adapters or {}).items():
            key = model_key(name)
            if key in self._adapters:
                raise ValueError("An adapter was supplied twice through model aliases")
            self._validate_adapter(key, adapter)
            self._adapters[key] = adapter
        self._profiles = {}
        for name, profile in (adapter_profiles or {}).items():
            key = model_key(name)
            if key in self._profiles or key in self._adapters:
                raise ValueError("Choose an adapter instance or one profile per model")
            if not isinstance(profile, AdapterProfile) or profile.key != key:
                raise TypeError("Each profile must identify its selected adapter")
            self._validate_decoding(profile.generation)
            self._profiles[key] = profile

    @staticmethod
    def _validate_decoding(options) -> None:
        if options.do_sample is not False or options.num_beams != 1:
            raise ValueError("The integrated baseline requires do_sample=False and num_beams=1")

    @classmethod
    def _validate_adapter(cls, key: str, adapter: HuggingFaceAdapter) -> None:
        if not isinstance(adapter, HuggingFaceAdapter):
            raise TypeError("Reuse an existing HuggingFaceAdapter")
        if adapter.key != key or adapter.model_id != MODEL_IDS[key]:
            raise ValueError("The provided adapter does not match the selected model")
        cls._validate_decoding(adapter.generation_options)

    def _adapter(self, key: str) -> HuggingFaceAdapter:
        if key not in self._adapters:
            if key in self._profiles:
                adapter = adapter_from_profile(self._profiles[key])
            else:
                constructors = {"qwen3": Qwen3Adapter, "gemma2": Gemma2Adapter, "dister": DiSTERLlamaAdapter}
                adapter = constructors[key]()
            self._validate_adapter(key, adapter)
            self._adapters[key] = adapter
        # Also detect mutation of a supplied adapter's settings before execution.
        self._validate_adapter(key, self._adapters[key])
        return self._adapters[key]

    def _few_shot_retriever(self):
        settings = self._retrieval_settings
        if settings is None:
            raise MissingRetrievalResourcesError(
                "few_shot requires external examples and syntactic retrieval settings"
            )
        if not settings.retrieval.exclude_same_text or not settings.retrieval.exclude_query_document:
            raise InvalidExtractionRequest(
                "few_shot requires the existing same-text and query-document exclusions"
            )
        if self._retriever is None:
            self._retriever = build_retriever(self._examples, settings, parser=self._parser)
        return self._retriever

    def extract(
        self,
        text: str,
        prompt_config: PromptConfig,
        *,
        model: str | None = None,
        condition: ExtractionCondition | str | None = None,
        candidates: Iterable[str] | None = None,
        query_id: str | None = None,
        query_document_id: str | None = None,
        excluded_example_ids: Iterable[str] = (),
        excluded_document_ids: Iterable[str] = (),
    ) -> ExtractionResult:
        """Execute one selected flow; model/parser execution must be authorized.

        Tests supply doubles. This method is not called with real resources in
        phase 2.4. Evaluation term annotations are deliberately not an argument.
        """
        request = ExtractionRequest(
            text=text, prompt_config=prompt_config,
            model=self.config.default_model if model is None else model,
            condition=self.config.default_condition if condition is None else condition,
            query_id=query_id, query_document_id=query_document_id, candidates=candidates,
            excluded_example_ids=excluded_example_ids,
            excluded_document_ids=excluded_document_ids,
        )
        if request.condition is ExtractionCondition.NATIVE_ZERO_SHOT:
            prompt = build_dister_prompt(request.query_id, request.text, request.domain.name)
        else:
            builder = PromptBuilder(request.prompt_config)
            if request.condition is ExtractionCondition.BASIC:
                prompt = builder.build_basic(request.text)
            elif request.condition is ExtractionCondition.PATTERNS:
                prompt = builder.build_patterns(request.text)
            elif request.condition is ExtractionCondition.HYBRID:
                prompt = builder.build_hybrid(request.text, request.candidates)
            else:
                retrieved = self._few_shot_retriever().retrieve(
                    request.text, query_id=request.query_id,
                    query_document_id=request.query_document_id,
                    excluded_example_ids=self._evaluation_ids | request.excluded_example_ids,
                    excluded_document_ids=self._evaluation_docs | request.excluded_document_ids,
                    top_k=3,
                )
                # Existing builder raises InsufficientExamplesError before any
                # adapter execution; no padding/relabeling of demonstrations.
                prompt = builder.build_few_shot(request.text, retrieved)
        generation = self._adapter(request.model).generate(prompt)
        return ExtractionResult(request=request, prompt=prompt, generation=generation)
