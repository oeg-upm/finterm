"""Production finance_entities_v2 policy. Historical prompts/ stays untouched."""

from dataclasses import replace

from prompts import PromptBuilder, default_prompt_config
from prompts.dister import DiSTERPrompt


PROMPT_VERSION = "finance_entities_v2"
FINANCIAL_ENTITIES = (
    "- For this task, output specialised financial terms AND relevant named financial entities "
    "in the same terms list. Include a name only when it designates a financial institution "
    "or a directly relevant financial organisation in this sentence, such as a bank, central "
    "bank, insurer, investment fund, financial supervisor or financial-market infrastructure.\n"
    "- Do not include person names, standalone geographic names or unrelated companies and "
    "organisations merely because they occur in a financial text. Keep geographic words when "
    "they are part of the complete name of a relevant financial institution.\n"
    "- Preserve each selected expression exactly as it appears in the text; do not invent, "
    "translate, expand or abbreviate entity names."
)


class FinancialPromptBuilder(PromptBuilder):
    """Same established tasks/structures, with a NEW, scoped entity policy."""
    def __init__(self):
        base = default_prompt_config("finance")
        super().__init__(replace(base, policy=replace(base.policy, exclude_named_entities=False)))

    def _basic_rules(self):
        return super()._basic_rules() + "\n" + FINANCIAL_ENTITIES

    def _hybrid_rules(self):
        # The task remains candidate validation; the new scope never authorizes
        # additions outside the externally supplied list.
        rules = super()._hybrid_rules()
        rules = rules.replace("- Accept a candidate only if it designates a specialised concept in this context.",
                              "- Accept a candidate only if it designates a specialised financial concept or a relevant named financial entity in this context.")
        return rules + "\n" + FINANCIAL_ENTITIES


class FinancialDiSTERPromptV2(DiSTERPrompt):
    """Native id/conversations/from/value structure, no system role or examples.

    This is explicitly NOT the historical native baseline: its third human turn
    adds the user-authorized financial-entity scope, without a generalist schema.
    """
    def to_dict(self):
        payload = super().to_dict()
        # Do not impose the generalists' {terms: [...]} response object. DiSTER
        # keeps its native list; only the task's semantic scope is versioned.
        native_scope = FINANCIAL_ENTITIES.replace("in the same terms list", "in the extracted list")
        payload["conversations"][2]["value"] += "\n" + native_scope
        return payload


def build_financial_dister(identifier, text):
    return FinancialDiSTERPromptV2(identifier, text, "Finance")
