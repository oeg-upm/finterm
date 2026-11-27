"""Pure prompt construction: no retrieval, model calls, tokenization or corpus I/O."""

import json
from typing import Iterable

from syntactic_retrieval.models import RetrievedExample, require_text

from .models import (
    ChatMessage, ChatPrompt, NestedTerms, PromptCondition, PromptConfig,
    PromptMetadata, StructureGuide,
)
from .templates import (
    ALL_NESTING, BASIC_OUTPUT, CONDITIONAL_NESTING, DEFINITION,
    FINANCIAL_STRUCTURES, GENERAL_STRUCTURES, HYBRID_OUTPUT, HYBRID_TASK,
    LONGEST_NESTING,
)


class InsufficientExamplesError(ValueError):
    def __init__(self, required: int, available: int) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"Few-shot condition requires {required} retrieved examples; received "
            f"{available}. No complete few-shot prompt was built."
        )


class PromptBuilder:
    def __init__(self, config: PromptConfig) -> None:
        if not isinstance(config, PromptConfig):
            raise TypeError("PromptBuilder requires a PromptConfig")
        self.config = config

    def _basic_rules(self) -> str:
        policy = self.config.policy
        nesting = {
            NestedTerms.WHEN_PERMITTED: CONDITIONAL_NESTING,
            NestedTerms.LONGEST_ONLY: LONGEST_NESTING,
            NestedTerms.ALL: ALL_NESTING,
        }[policy.nested_terms]
        exclusions = (
            "named entities (companies, people, places, institutions, products), "
            if policy.exclude_named_entities else ""
        )
        rules = [
            "- Extract only terms relevant to the specified domain.",
            "- Extract terms exactly as they appear in the text (same word form and spelling).",
            nesting,
            "- Do NOT extract: " + exclusions + (
                "numbers, dates, monetary amounts, percentages, or general-language "
                "words without a specialised meaning in the specified domain."
            ),
        ]
        if not policy.include_acronyms:
            rules.append("- Do NOT extract acronyms or abbreviations.")
        rules.append("- If there are no terms, return an empty list.")
        return "### Rules\n" + "\n".join(rules)

    def _base_sections(self) -> list[str]:
        return [DEFINITION, f"### Domain\n{self.config.domain.description}", self._basic_rules()]

    def _result(
        self,
        condition: PromptCondition,
        sections: list[str],
        *,
        examples: tuple[RetrievedExample, ...] = (),
        available_examples: int = 0,
    ) -> ChatPrompt:
        return ChatPrompt(
            messages=(
                ChatMessage(
                    "system",
                    f"You are an expert terminologist specialised in the {self.config.domain.name} domain.",
                ),
                ChatMessage("user", "\n\n".join(sections)),
            ),
            metadata=PromptMetadata(
                condition=condition, config=self.config, retrieved_examples=examples,
                available_examples=available_examples,
            ),
        )

    def build_basic(self, text: str) -> ChatPrompt:
        require_text(text, "text")
        return self._result(
            PromptCondition.BASIC,
            self._base_sections() + [f"### Text\n{text}", BASIC_OUTPUT],
        )

    def _structure_block(self, guide: StructureGuide) -> str:
        structures = list(guide.structures)
        if self.config.policy.include_acronyms and guide.acronym_structure is not None:
            structures.append(guide.acronym_structure)
        return "\n\n".join([
            f"### {guide.title}", guide.introduction,
            "\n".join(f"- {structure}" for structure in structures), guide.conclusion,
        ])

    def build_patterns(self, text: str) -> ChatPrompt:
        require_text(text, "text")
        guide = self.config.structure_guide
        if guide is None:
            guide = FINANCIAL_STRUCTURES if self.config.domain.key == "finance" else GENERAL_STRUCTURES
        return self._result(
            PromptCondition.PATTERNS,
            self._base_sections() + [self._structure_block(guide), f"### Text\n{text}", BASIC_OUTPUT],
        )

    def build_few_shot(
        self, text: str, retrieved_examples: Iterable[RetrievedExample]
    ) -> ChatPrompt:
        require_text(text, "text")
        if isinstance(retrieved_examples, (str, bytes)):
            raise TypeError("retrieved_examples must contain existing RetrievedExample objects")
        supplied = tuple(retrieved_examples)
        if not all(isinstance(item, RetrievedExample) for item in supplied):
            raise TypeError("Every demonstration must be an existing RetrievedExample")
        required = self.config.few_shot_examples
        if len(supplied) < required:
            raise InsufficientExamplesError(required, len(supplied))
        # Consume retrieval order; never sort by scores, recompute similarity or
        # inspect evaluation answers. A longer supplied list is truncated only.
        selected = supplied[:required]
        demonstrations = []
        for retrieved in selected:
            example = retrieved.example
            terms_json = json.dumps({"terms": list(example.terms)}, ensure_ascii=False)
            demonstrations.append(f"Text: {example.text}\nOutput: {terms_json}")
        examples_block = "### Examples\n\n" + "\n\n".join(demonstrations)
        return self._result(
            PromptCondition.FEW_SHOT,
            self._base_sections() + [examples_block, f"### Text\n{text}", BASIC_OUTPUT],
            examples=selected, available_examples=len(supplied),
        )

    def _hybrid_rules(self) -> str:
        policy = self.config.policy
        excluded_entities = "named entities, " if policy.exclude_named_entities else ""
        boundary_rule = "- Reject candidates with incorrect boundaries"
        pair = self.config.domain.boundary_example
        if pair is not None:
            incorrect, correct = (json.dumps(value, ensure_ascii=False) for value in pair)
            boundary_rule += (
                f" (e.g., {incorrect} → reject; the correct term would be "
                f"{correct} if it is in the list)"
            )
        rules = [
            "- Accept a candidate only if it designates a specialised concept in this context.",
            "- Reject " + excluded_entities + (
                "numbers, dates, monetary amounts, and general-language expressions."
            ),
            boundary_rule + ".",
        ]
        if policy.nested_terms is NestedTerms.LONGEST_ONLY:
            rules.append(
                "- For nested terms, accept only the longest term span. Accept a shorter "
                "candidate only if it also occurs independently in the text."
            )
        elif policy.nested_terms is NestedTerms.ALL:
            rules.append(
                "- Accept both a multiword candidate and shorter candidates contained "
                "in it when they are also terms on their own."
            )
        if not policy.include_acronyms:
            rules.append("- Reject acronyms and abbreviations.")
        rules.append("- Do not add expressions that are not in the candidate list.")
        return "### Rules\n" + "\n".join(rules)

    def build_hybrid(self, text: str, candidates: Iterable[str]) -> ChatPrompt:
        require_text(text, "text")
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be a collection of expressions, not a string")
        candidate_list = list(candidates)
        for candidate in candidate_list:
            require_text(candidate, "candidate")
        candidates_json = json.dumps(candidate_list, ensure_ascii=False)
        return self._result(
            PromptCondition.HYBRID,
            [
                DEFINITION, f"### Domain\n{self.config.domain.description}", HYBRID_TASK,
                self._hybrid_rules(), f"### Text\n{text}",
                f"### Candidates\n{candidates_json}", HYBRID_OUTPUT,
            ],
        )
