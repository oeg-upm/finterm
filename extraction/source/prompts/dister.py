"""The model-card-native DiSTER JSON conversation, zero-shot and text-only."""

from dataclasses import dataclass
import json
from typing import Any, ClassVar

from syntactic_retrieval.models import require_text


@dataclass(frozen=True, slots=True)
class DiSTERPrompt:
    identifier: str
    text: str
    domain: str
    model_id: ClassVar[str] = "ElenaSenger/DiSTER-Llama-3-8B-Instruct"

    def __post_init__(self) -> None:
        for name in ("identifier", "text", "domain"):
            require_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "conversations": [
                {"from": "human", "value": f"Text: {self.text}"},
                {"from": "gpt", "value": "I've read this text."},
                {
                    "from": "human",
                    "value": (
                        "What describes (technical or scientific) terms in the text, "
                        f"that are relevant to the domain {self.domain}?"
                    ),
                },
                {"from": "gpt", "value": ""},
            ],
        }

    def to_json(self) -> str:
        # The last gpt turn is empty inside a valid, closed JSON object, as in
        # the model card. No chat template or generalist output schema is added.
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4) + "\n"


def build_dister_prompt(identifier: str, text: str, domain: str) -> DiSTERPrompt:
    return DiSTERPrompt(identifier=identifier, text=text, domain=domain)
