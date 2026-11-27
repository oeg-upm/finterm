"""Configuration and text-only prompt records; reuse the existing retrieval data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

from syntactic_retrieval.models import RetrievedExample, require_text


class PromptCondition(str, Enum):
    BASIC = "basic"
    PATTERNS = "patterns"
    FEW_SHOT = "few_shot"
    HYBRID = "hybrid"


class NestedTerms(str, Enum):
    WHEN_PERMITTED = "when_permitted"
    LONGEST_ONLY = "longest_only"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class AnnotationPolicy:
    exclude_named_entities: bool = True
    nested_terms: NestedTerms = NestedTerms.WHEN_PERMITTED
    include_acronyms: bool = True

    def __post_init__(self) -> None:
        for name in ("exclude_named_entities", "include_acronyms"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(self, "nested_terms", NestedTerms(self.nested_terms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exclude_named_entities": self.exclude_named_entities,
            "nested_terms": self.nested_terms.value,
            "include_acronyms": self.include_acronyms,
        }


@dataclass(frozen=True, slots=True)
class DomainSpec:
    key: str
    name: str
    description: str
    boundary_example: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("key", "name", "description"):
            require_text(getattr(self, name), name)
        if self.boundary_example is not None:
            if isinstance(self.boundary_example, (str, bytes)):
                raise TypeError("boundary_example must contain an incorrect/correct pair")
            pair = tuple(self.boundary_example)
            if len(pair) != 2:
                raise ValueError("boundary_example must contain exactly two expressions")
            for expression in pair:
                require_text(expression, "boundary expression")
            object.__setattr__(self, "boundary_example", pair)


@dataclass(frozen=True, slots=True)
class StructureGuide:
    """Linguistic illustrations for a prompt, NOT executable POS/spaCy rules."""

    title: str
    introduction: str
    structures: tuple[str, ...]
    conclusion: str
    acronym_structure: str | None = None
    domain_key: str | None = None

    def __post_init__(self) -> None:
        for name in ("title", "introduction", "conclusion"):
            require_text(getattr(self, name), name)
        if isinstance(self.structures, (str, bytes)):
            raise TypeError("structures must be a collection of linguistic descriptions")
        structures = tuple(self.structures)
        if not structures:
            raise ValueError("At least one grammatical structure is required")
        for structure in structures:
            require_text(structure, "grammatical structure")
        object.__setattr__(self, "structures", structures)
        if self.acronym_structure is not None:
            require_text(self.acronym_structure, "acronym_structure")
        if self.domain_key is not None:
            require_text(self.domain_key, "structure guide domain_key")


@dataclass(frozen=True, slots=True)
class PromptConfig:
    domain: DomainSpec
    policy: AnnotationPolicy
    few_shot_examples: int = 3
    structure_guide: StructureGuide | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, DomainSpec) or not isinstance(self.policy, AnnotationPolicy):
            raise TypeError("PromptConfig requires a DomainSpec and an AnnotationPolicy")
        if type(self.few_shot_examples) is not int or self.few_shot_examples < 1:
            raise ValueError("few_shot_examples must be a positive integer")
        if self.structure_guide is not None:
            if not isinstance(self.structure_guide, StructureGuide):
                raise TypeError("structure_guide must be a StructureGuide")
            scope = self.structure_guide.domain_key
            if scope is not None and scope != self.domain.key:
                raise ValueError("Structure examples belong to a different domain")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user"):
            raise ValueError("Generalist prompts use system and user messages only")
        require_text(self.content, "message content")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class PromptMetadata:
    condition: PromptCondition
    config: PromptConfig
    retrieved_examples: tuple[RetrievedExample, ...] = ()
    available_examples: int = 0

    @property
    def complete_few_shot(self) -> bool:
        return (
            self.condition is PromptCondition.FEW_SHOT
            and len(self.retrieved_examples) == self.config.few_shot_examples
        )

    def to_dict(self) -> dict[str, Any]:
        """Retrieval scores stay here, separate from the messages sent to an LLM."""
        return {
            "condition": self.condition.value,
            "domain": self.config.domain.key,
            "policy": self.config.policy.to_dict(),
            "requested_examples": (
                self.config.few_shot_examples
                if self.condition is PromptCondition.FEW_SHOT else 0
            ),
            "available_examples": self.available_examples,
            "used_examples": len(self.retrieved_examples),
            "complete_few_shot": self.complete_few_shot,
            "retrieved_examples": [item.to_dict() for item in self.retrieved_examples],
        }


@dataclass(frozen=True, slots=True)
class ChatPrompt:
    messages: tuple[ChatMessage, ChatMessage]
    metadata: PromptMetadata

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if (
            len(messages) != 2
            or not all(isinstance(message, ChatMessage) for message in messages)
            or tuple(message.role for message in messages) != ("system", "user")
        ):
            raise ValueError("Expected one system message followed by one user message")
        object.__setattr__(self, "messages", messages)

    def to_messages(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]

    def to_json(self) -> str:
        """Serialize messages only; metadata and scores are never included."""
        return json.dumps(self.to_messages(), ensure_ascii=False)
