"""Request/result contracts for coordination, not annotation or model algorithms."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from llm_adapters.config import MODEL_IDS
from llm_adapters.responses import FormatError, GenerationResult, JsonTermsResponse
from prompts.dister import DiSTERPrompt
from prompts.models import ChatPrompt, DomainSpec, PromptCondition, PromptConfig
from syntactic_retrieval.models import RetrievedExample, identifier_set, require_text


class ExtractionCondition(str, Enum):
    BASIC = PromptCondition.BASIC.value
    PATTERNS = PromptCondition.PATTERNS.value
    FEW_SHOT = PromptCondition.FEW_SHOT.value
    HYBRID = PromptCondition.HYBRID.value
    NATIVE_ZERO_SHOT = "native_zero_shot"


class InterpretationStatus(str, Enum):
    JSON_TERMS = "validated_json_terms"
    FORMAT_ERROR = "format_error"
    NATIVE_PENDING = "native_terms_pending_validation"


class UnsupportedModelConditionError(ValueError):
    """The requested model/condition pair is not implemented."""


class InvalidExtractionRequest(ValueError):
    """Required inputs or the approved experimental contract are missing."""


class MissingRetrievalResourcesError(ValueError):
    """Few-shot needs externally supplied examples and retrieval settings."""


def model_key(value: str) -> str:
    """Accept an existing adapter key or its official ID, without a fallback."""
    if isinstance(value, str):
        for key, identifier in MODEL_IDS.items():
            if value == key or value == identifier:
                return key
    raise ValueError("Select one of the existing qwen3, gemma2 or dister adapters")


def validate_combination(model: str, condition: ExtractionCondition) -> None:
    native = condition is ExtractionCondition.NATIVE_ZERO_SHOT
    if (model == "dister") != native:
        raise UnsupportedModelConditionError(
            "DiSTER supports only native_zero_shot; Qwen/Gemma support "
            "basic, patterns, few_shot and hybrid. No condition was substituted."
        )


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    text: str
    prompt_config: PromptConfig
    model: str
    condition: ExtractionCondition
    query_id: str | None = None
    query_document_id: str | None = None
    candidates: tuple[str, ...] | None = None
    excluded_example_ids: frozenset[str] = frozenset()
    excluded_document_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        require_text(self.text, "text")
        if not isinstance(self.prompt_config, PromptConfig):
            raise TypeError("Provide the existing PromptConfig with its domain and policy")
        key = model_key(self.model)
        condition = ExtractionCondition(self.condition)
        validate_combination(key, condition)
        object.__setattr__(self, "model", key)
        object.__setattr__(self, "condition", condition)
        for name in ("query_id", "query_document_id"):
            if getattr(self, name) is not None:
                require_text(getattr(self, name), name)
        for name in ("excluded_example_ids", "excluded_document_ids"):
            object.__setattr__(self, name, identifier_set(getattr(self, name), name))
        if self.candidates is not None:
            if isinstance(self.candidates, (str, bytes)):
                raise TypeError("candidates must be an external collection, not a string")
            object.__setattr__(self, "candidates", tuple(self.candidates))
        if condition is ExtractionCondition.HYBRID:
            if self.candidates is None:
                raise InvalidExtractionRequest("hybrid requires externally supplied candidates")
        elif self.candidates is not None:
            raise InvalidExtractionRequest("Candidates are accepted only by the hybrid condition")
        if condition is ExtractionCondition.FEW_SHOT:
            if self.query_document_id is None:
                raise InvalidExtractionRequest(
                    "few_shot requires query_document_id to exclude the entire source document"
                )
            if self.prompt_config.few_shot_examples != 3:
                raise InvalidExtractionRequest(
                    "This integrated few_shot condition requires a PromptConfig with exactly 3 examples"
                )
        if condition is ExtractionCondition.NATIVE_ZERO_SHOT:
            if self.query_id is None:
                raise InvalidExtractionRequest("native_zero_shot requires query_id for DiSTER's native record")

    @property
    def domain(self) -> DomainSpec:
        return self.prompt_config.domain


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    request: ExtractionRequest
    prompt: ChatPrompt | DiSTERPrompt
    generation: GenerationResult

    def __post_init__(self) -> None:
        if not isinstance(self.generation, GenerationResult):
            raise TypeError("Adapters must return the existing GenerationResult")
        if self.generation.model_id != MODEL_IDS[self.request.model]:
            raise ValueError("The adapter returned a result from a different model")
        native = self.request.condition is ExtractionCondition.NATIVE_ZERO_SHOT
        if native:
            if not isinstance(self.prompt, DiSTERPrompt) or self.generation.json_response is not None:
                raise TypeError("Native DiSTER must preserve its prompt and uninterpreted response")
        elif not isinstance(self.prompt, ChatPrompt) or self.generation.json_response is None:
            raise TypeError("Generalist adapters must return their JSON format validation")

    @property
    def interpretation(self) -> InterpretationStatus:
        if self.request.condition is ExtractionCondition.NATIVE_ZERO_SHOT:
            return InterpretationStatus.NATIVE_PENDING
        if self.generation.json_response.schema_valid:
            return InterpretationStatus.JSON_TERMS
        return InterpretationStatus.FORMAT_ERROR

    @property
    def terms(self) -> tuple[str, ...] | None:
        return self.generation.terms

    @property
    def raw_response(self) -> str:
        return self.generation.raw_response

    @property
    def json_response(self) -> JsonTermsResponse | None:
        return self.generation.json_response

    @property
    def format_errors(self) -> tuple[FormatError, ...]:
        response = self.json_response
        return response.errors if response is not None else ()

    @property
    def retrieved_examples(self) -> tuple[RetrievedExample, ...]:
        return self.prompt.metadata.retrieved_examples if isinstance(self.prompt, ChatPrompt) else ()

    def to_dict(self) -> dict[str, Any]:
        generalist = isinstance(self.prompt, ChatPrompt)
        return {
            "model": self.request.model,
            "condition": self.request.condition.value,
            "domain": {
                "key": self.request.domain.key, "name": self.request.domain.name,
                "description": self.request.domain.description,
            },
            "query_id": self.request.query_id,
            "query_document_id": self.request.query_document_id,
            "interpretation": self.interpretation.value,
            "terms": list(self.terms) if self.terms is not None else None,
            "format_errors": [error.to_dict() for error in self.format_errors],
            "prompt": self.prompt.to_messages() if generalist else self.prompt.to_dict(),
            "prompt_metadata": self.prompt.metadata.to_dict() if generalist else None,
            "generation": self.generation.to_dict(),
        }
