"""Transport adaptation of frozen prompts; no model or tokenizer is loaded here."""

from dataclasses import dataclass
from typing import Any

from prompts.models import ChatMessage, ChatPrompt
from prompts.dister import DiSTERPrompt


@dataclass(frozen=True, slots=True)
class PreparedInput:
    kind: str
    messages: tuple[ChatMessage, ...] = ()
    native_text: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("qwen3", "gemma2", "dister"):
            raise ValueError("Unsupported input format")
        messages = tuple(self.messages)
        if self.kind == "dister":
            if messages or not isinstance(self.native_text, str) or not self.native_text:
                raise ValueError("Native DiSTER input requires its JSON string only")
        else:
            expected = ("system", "user") if self.kind == "qwen3" else ("user",)
            if (
                self.native_text is not None
                or not all(isinstance(item, ChatMessage) for item in messages)
                or tuple(item.role for item in messages) != expected
            ):
                raise ValueError("Invalid messages for the model's conversation format")
        object.__setattr__(self, "messages", messages)

    def render(self, tokenizer: Any) -> str:
        """Use a supplied tokenizer, never instantiate/download one implicitly."""
        if self.kind == "dister":
            return self.native_text
        options = {"tokenize": False, "add_generation_prompt": True}
        if self.kind == "qwen3":
            options["enable_thinking"] = False
        text = tokenizer.apply_chat_template(
            [message.to_dict() for message in self.messages], **options
        )
        if not isinstance(text, str) or not text:
            raise ValueError("The official chat template did not return input text")
        return text

    def tokenize(self, tokenizer: Any) -> tuple[str, Any]:
        text = self.render(tokenizer)
        # Chat templates already insert BOS/turn tokens. Native DiSTER follows
        # the model-card tokenizer(prompt) call, whose default adds BOS.
        encoded = tokenizer(
            text, return_tensors="pt", truncation=False,
            add_special_tokens=self.kind == "dister",
        )
        return text, encoded


def prepare_qwen3(prompt: ChatPrompt) -> PreparedInput:
    if not isinstance(prompt, ChatPrompt):
        raise TypeError("Qwen3 requires an existing generalist ChatPrompt")
    return PreparedInput("qwen3", messages=prompt.messages)


def prepare_gemma2(prompt: ChatPrompt) -> PreparedInput:
    if not isinstance(prompt, ChatPrompt):
        raise TypeError("Gemma-2 requires an existing generalist ChatPrompt")
    system, user = prompt.messages
    # Only change transport: preserve both contents verbatim, in their order.
    merged = ChatMessage("user", system.content + "\n\n" + user.content)
    return PreparedInput("gemma2", messages=(merged,))


def prepare_dister(prompt: DiSTERPrompt) -> PreparedInput:
    if not isinstance(prompt, DiSTERPrompt):
        raise TypeError("DiSTER requires its existing native DiSTERPrompt")
    return PreparedInput("dister", native_text=prompt.to_json())
