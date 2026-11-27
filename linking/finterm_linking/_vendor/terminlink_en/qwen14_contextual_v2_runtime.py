#!/usr/bin/env python3
"""Shared audited runtime for contextual-translation V2 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
PROMPT_VERSION = "contextual-translation-en-qwen14-v2"
MAX_NEW_TOKENS = 128
GENERATION_ATTEMPTS = 1
INFERENCE_SEED = 20260929
ENABLE_THINKING = False

SYSTEM_MESSAGE = """You are an expert multilingual translator.

Your task is to translate ONLY the source description explicitly identified in the input into English.

The input also contains auxiliary semantic evidence about the same Wikidata candidate. Use this evidence ONLY to resolve lexical, grammatical, or semantic ambiguity in the source description.

The source description is the text to translate. Candidate labels, aliases, other descriptions, and structured properties are auxiliary evidence only. They may be incomplete, incorrect, or contradictory.

Do not translate, summarize, paraphrase, or reproduce the auxiliary evidence itself.

Do not add any information that is expressed only in the candidate label, aliases, other descriptions, or structured properties.

Do not add proper names, institutions, relations, dates, places, or more specific categories unless they are explicitly expressed in the source description.

If the auxiliary evidence conflicts with the source description, preserve the linguistic meaning of the source description. Do not reconcile the conflict by inventing details.

If the source description is ambiguous and the auxiliary evidence does not justify a specific interpretation, preserve the ambiguity and do not invent a more specific interpretation.

If you do not understand the source language or source text well enough to produce a faithful translation, do not guess or invent content.

Do not add facts, properties, relations, hypernyms, explanations, definitions, or semantic information that is not expressed in the source description.

A one-word English translation is valid when the source meaning is correctly expressed by one word.

Preserve the meaning, grammatical category, and level of specificity of the source text as closely as English permits.

Do not turn the translation into a definition, explanation, or summary of the candidate.

Output only the English translation of the source description.

Do not include quotation marks, labels, explanations, alternatives, notes, reasoning, or commentary."""

USER_MESSAGE_TEMPLATE = """Semantic information about the Wikidata candidate and the source text to translate:

{translation_evidence_text}

Translate ONLY the text identified as "Source description" into English.

Use the other information only to disambiguate its meaning.

Return only the English translation and nothing else.

Translation:"""

EXPECTED_SYSTEM_MESSAGE_SHA256 = (
    "931DF301DD05912EE59ED12EC5F9984A506E97338DAB7FF7989A3594D818EFE5"
)
EXPECTED_USER_TEMPLATE_SHA256 = (
    "A72080B7AB99A616864310F9D92B2B970B2E4E233874F7E3E45C743443033FC3"
)

RUNTIME_METADATA_FIELDS = (
    "model_revision",
    "system_prompt_sha256",
    "user_prompt_template_sha256",
    "input_sha256",
    "inference_script_sha256",
    "runtime_module_sha256",
    "inference_seed",
    "quantization_config",
    "generation_parameters",
    "python_version",
    "transformers_version",
    "tokenizers_version",
    "huggingface_hub_version",
    "accelerate_version",
    "bitsandbytes_version",
    "safetensors_version",
    "torch_version",
    "cuda_version",
    "device_name",
)

FIXED_QUANTIZATION_PARAMETERS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
}

FIXED_GENERATION_PARAMETERS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "use_cache": True,
}

RUNTIME_GENERATION_PARAMETER_FIELDS = frozenset(
    {
        *FIXED_GENERATION_PARAMETERS,
        "enable_thinking",
        "pad_token_id",
        "eos_token_id",
        "torch_deterministic_algorithms",
    }
)

THINK_PATTERN = re.compile(r"</?think\b|<think>.*?</think>", re.IGNORECASE | re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
MARKDOWN_HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+] |\d+[.)]\s+)", re.MULTILINE)
MARKDOWN_INLINE_PATTERN = re.compile(
    r"(?:^\s*(?:\*+|_+|~~|`+)|(?:\*+|_+|~~|`+)\s*$)"
)
META_PREFIX_PATTERN = re.compile(
    r"^\s*(?:here(?:'s|\s+is)\s+(?:the\s+)?(?:english\s+)?"
    r"translation\s*[:\-–—]|(?:the\s+)?(?:english\s+)?translation\s*"
    r"(?:(?:is|would\s+be)\b|[:\-–—])|english\s*[:\-–—]|"
    r"answer\s*[:\-–—]|output\s*[:\-–—]|result\s*[:\-–—]|"
    r"sure\s*[,!:])",
    re.IGNORECASE,
)
COMMENTARY_PATTERN = re.compile(
    r"\b(?:this\s+translates?\s+(?:to|as)|which\s+means|"
    r"literally\s+means|(?:note|explanation|reasoning|alternatives?|option)"
    r"\s*[:\-–—]|alternatively\s*[:\-–—]|"
    r"another\s+translation\s+(?:is|would\s+be)|"
    r"(?:it|this)\s+means\b|meaning\s*:)",
    re.IGNORECASE,
)
ALTERNATIVE_SEPARATOR_PATTERN = re.compile(r"[/|]")
FORBIDDEN_QUOTATION_CHARS = frozenset('"“”«»„‟')


class ContextualV2Error(RuntimeError):
    """Raised when a shared V2 runtime invariant fails."""


class MalformedGenerationError(ValueError):
    """Raised when a generation contains anything beyond one translation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContextualV2Error(message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def validate_prompt_contract() -> None:
    require(
        sha256_text(SYSTEM_MESSAGE) == EXPECTED_SYSTEM_MESSAGE_SHA256,
        "The canonical V2 system prompt changed",
    )
    require(
        sha256_text(USER_MESSAGE_TEMPLATE) == EXPECTED_USER_TEMPLATE_SHA256,
        "The canonical V2 user prompt changed",
    )


def build_user_message(translation_evidence_text: str) -> str:
    require(bool(translation_evidence_text.strip()), "Translation evidence is empty")
    return USER_MESSAGE_TEMPLATE.format(
        translation_evidence_text=translation_evidence_text
    )


def build_messages(translation_evidence_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(translation_evidence_text)},
    ]


def render_prompt(tokenizer: Any, translation_evidence_text: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(translation_evidence_text),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


def normalize_and_validate_generation(raw_text: str) -> str:
    """Normalize whitespace only after rejecting non-translation output."""
    raw = str(raw_text)
    stripped = raw.strip()
    if not stripped:
        raise MalformedGenerationError("Generation is empty")
    if "\x00" in raw:
        raise MalformedGenerationError("Generation contains a NUL character")
    if THINK_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains reasoning tags")
    if "```" in raw:
        raise MalformedGenerationError("Generation contains a fenced block")
    if HTML_TAG_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains HTML markup")
    if (
        MARKDOWN_HEADER_PATTERN.search(raw)
        or LIST_ITEM_PATTERN.search(raw)
        or MARKDOWN_INLINE_PATTERN.search(raw)
    ):
        raise MalformedGenerationError("Generation contains Markdown structure")
    if len([line for line in raw.splitlines() if line.strip()]) > 1:
        raise MalformedGenerationError("Generation contains multiple non-empty lines")
    if META_PREFIX_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains a translation header")
    if COMMENTARY_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains explanation or commentary")
    if ALTERNATIVE_SEPARATOR_PATTERN.search(raw):
        raise MalformedGenerationError("Generation contains alternative separators")
    if ";" in raw:
        raise MalformedGenerationError("Generation contains a semicolon-separated alternative")
    if any(character in raw for character in FORBIDDEN_QUOTATION_CHARS):
        raise MalformedGenerationError("Generation contains quotation marks")
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == "'":
        raise MalformedGenerationError("Generation is wrapped in quotation marks")
    if len(stripped) >= 2 and (stripped[0], stripped[-1]) in {
        ("‘", "’"),
        ("‚", "‘"),
        ("‹", "›"),
    }:
        raise MalformedGenerationError("Generation is wrapped in quotation marks")
    return re.sub(r"\s+", " ", raw).strip()


def load_model_and_tokenizer() -> tuple[Any, Any, dict[str, str]]:
    import platform
    from importlib.metadata import version

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    require(torch.cuda.is_available(), "CUDA is required for Qwen3-14B inference")
    device_name = torch.cuda.get_device_name(0)
    require(
        "H100" in device_name.upper(),
        f"This audited V2 runtime requires an H100; detected {device_name}",
    )
    require(
        torch.cuda.is_bf16_supported(),
        "The active CUDA device does not support bfloat16",
    )
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    token = os.environ.get("HF_TOKEN", "").strip()
    auth_kwargs = {"token": token} if token else {}
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=FIXED_QUANTIZATION_PARAMETERS["load_in_4bit"],
        bnb_4bit_quant_type=FIXED_QUANTIZATION_PARAMETERS[
            "bnb_4bit_quant_type"
        ],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=FIXED_QUANTIZATION_PARAMETERS[
            "bnb_4bit_use_double_quant"
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        **auth_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map={"": 0},
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        **auth_kwargs,
    )
    model.eval()
    require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token ID")
    runtime_generation_parameters = {
        **FIXED_GENERATION_PARAMETERS,
        "enable_thinking": ENABLE_THINKING,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "torch_deterministic_algorithms": True,
    }
    environment_metadata = {
        "model_revision": MODEL_REVISION,
        "system_prompt_sha256": EXPECTED_SYSTEM_MESSAGE_SHA256,
        "user_prompt_template_sha256": EXPECTED_USER_TEMPLATE_SHA256,
        "input_sha256": "",
        "inference_script_sha256": "",
        "runtime_module_sha256": "",
        "inference_seed": str(INFERENCE_SEED),
        "quantization_config": json.dumps(
            FIXED_QUANTIZATION_PARAMETERS,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "generation_parameters": json.dumps(
            runtime_generation_parameters,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "python_version": platform.python_version(),
        "transformers_version": transformers.__version__,
        "tokenizers_version": version("tokenizers"),
        "huggingface_hub_version": version("huggingface-hub"),
        "accelerate_version": version("accelerate"),
        "bitsandbytes_version": version("bitsandbytes"),
        "safetensors_version": version("safetensors"),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda or ""),
        "device_name": device_name,
    }
    return tokenizer, model, environment_metadata


def generate_once(
    tokenizer: Any, model: Any, translation_evidence_text: str
) -> str:
    import torch

    rendered_prompt = render_prompt(tokenizer, translation_evidence_text)
    model_inputs = tokenizer(rendered_prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    model_inputs = {
        name: tensor.to(input_device) for name, tensor in model_inputs.items()
    }
    input_token_count = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **FIXED_GENERATION_PARAMETERS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_only = generated_ids[0, input_token_count:]
    return tokenizer.decode(generated_only, skip_special_tokens=True)


Generator = Callable[[Any, Any, str], str]


def format_error(error: Exception) -> str:
    detail = re.sub(r"\s+", " ", str(error)).strip()
    return f"{type(error).__name__}: {detail or 'No error detail provided'}"


def complete_runtime_metadata(
    environment_metadata: dict[str, str],
    input_sha256: str,
    inference_script_sha256: str,
    runtime_module_sha256: str,
) -> dict[str, str]:
    metadata = {
        **environment_metadata,
        "input_sha256": input_sha256,
        "inference_script_sha256": inference_script_sha256,
        "runtime_module_sha256": runtime_module_sha256,
    }
    validate_runtime_metadata(metadata)
    return metadata


def validate_runtime_metadata(runtime_metadata: dict[str, str]) -> None:
    require(
        len(runtime_metadata) == len(RUNTIME_METADATA_FIELDS)
        and set(runtime_metadata) == set(RUNTIME_METADATA_FIELDS),
        "Unexpected V2 runtime metadata schema",
    )
    require(
        runtime_metadata["model_revision"] == MODEL_REVISION,
        "Runtime model revision differs from the pinned revision",
    )
    require(
        runtime_metadata["system_prompt_sha256"]
        == EXPECTED_SYSTEM_MESSAGE_SHA256
        and runtime_metadata["user_prompt_template_sha256"]
        == EXPECTED_USER_TEMPLATE_SHA256,
        "Runtime prompt provenance differs from the V2 prompt",
    )
    for field in (
        "input_sha256",
        "inference_script_sha256",
        "runtime_module_sha256",
    ):
        require(
            re.fullmatch(r"[0-9A-F]{64}", runtime_metadata[field]) is not None,
            f"Runtime {field} is missing or invalid",
        )
    require(
        runtime_metadata["inference_seed"] == str(INFERENCE_SEED),
        "Runtime inference seed differs from the fixed seed",
    )
    try:
        quantization = json.loads(runtime_metadata["quantization_config"])
        generation = json.loads(runtime_metadata["generation_parameters"])
    except (json.JSONDecodeError, TypeError) as error:
        raise ContextualV2Error(
            f"Runtime technical parameters are not valid JSON: {error}"
        ) from error
    require(
        type(quantization) is dict
        and set(quantization) == set(FIXED_QUANTIZATION_PARAMETERS)
        and quantization["load_in_4bit"] is True
        and quantization["bnb_4bit_quant_type"] == "nf4"
        and quantization["bnb_4bit_compute_dtype"] == "bfloat16"
        and quantization["bnb_4bit_use_double_quant"] is True,
        "Runtime quantization parameters differ from the fixed NF4 contract",
    )
    require(
        type(generation) is dict
        and set(generation) == RUNTIME_GENERATION_PARAMETER_FIELDS,
        "Runtime generation parameter schema differs from the fixed contract",
    )
    require(
        type(generation["max_new_tokens"]) is int
        and generation["max_new_tokens"] == MAX_NEW_TOKENS
        and generation["do_sample"] is False
        and type(generation["num_beams"]) is int
        and generation["num_beams"] == 1
        and type(generation["num_return_sequences"]) is int
        and generation["num_return_sequences"] == 1
        and generation["use_cache"] is True
        and generation["enable_thinking"] is False
        and generation["torch_deterministic_algorithms"] is True,
        "Runtime generation parameters differ from the fixed greedy contract",
    )
    eos_token_id = generation["eos_token_id"]
    pad_token_id = generation["pad_token_id"]
    require(
        type(eos_token_id) is int
        and eos_token_id >= 0
        and type(pad_token_id) is int
        and pad_token_id == eos_token_id,
        "Runtime EOS/padding provenance is invalid",
    )
    require(
        "H100" in runtime_metadata["device_name"].upper(),
        "Runtime device provenance does not identify the required H100",
    )
    require(
        bool(runtime_metadata["cuda_version"].strip()),
        "Runtime CUDA version provenance is empty",
    )
    require(
        all(runtime_metadata[field] for field in RUNTIME_METADATA_FIELDS),
        "Runtime provenance contains an empty value",
    )


def infer_once(
    translation_evidence_text: str,
    tokenizer: Any,
    model: Any,
    generator: Generator,
) -> dict[str, str]:
    translation_raw = ""
    translation_clean = ""
    generation_status = "ERROR"
    error_message = ""
    try:
        translation_raw = generator(
            tokenizer, model, translation_evidence_text
        )
        translation_clean = normalize_and_validate_generation(translation_raw)
        generation_status = "OK"
    except Exception as error:
        error_message = format_error(error)
    return {
        "generation_attempts": str(GENERATION_ATTEMPTS),
        "translation_raw": translation_raw,
        "translation_clean": translation_clean,
        "generation_status": generation_status,
        "error_message": error_message,
    }


def synthetic_runtime_metadata() -> dict[str, str]:
    generation_parameters = {
        **FIXED_GENERATION_PARAMETERS,
        "enable_thinking": ENABLE_THINKING,
        "pad_token_id": 151643,
        "eos_token_id": 151643,
        "torch_deterministic_algorithms": True,
    }
    return {
        "model_revision": MODEL_REVISION,
        "system_prompt_sha256": EXPECTED_SYSTEM_MESSAGE_SHA256,
        "user_prompt_template_sha256": EXPECTED_USER_TEMPLATE_SHA256,
        "input_sha256": "A" * 64,
        "inference_script_sha256": "B" * 64,
        "runtime_module_sha256": "C" * 64,
        "inference_seed": str(INFERENCE_SEED),
        "quantization_config": json.dumps(
            FIXED_QUANTIZATION_PARAMETERS, sort_keys=True, separators=(",", ":")
        ),
        "generation_parameters": json.dumps(
            generation_parameters, sort_keys=True, separators=(",", ":")
        ),
        "python_version": "synthetic",
        "transformers_version": "synthetic",
        "tokenizers_version": "synthetic",
        "huggingface_hub_version": "synthetic",
        "accelerate_version": "synthetic",
        "bitsandbytes_version": "synthetic",
        "safetensors_version": "synthetic",
        "torch_version": "synthetic",
        "cuda_version": "synthetic",
        "device_name": "synthetic-H100",
    }


def run_self_tests() -> None:
    validate_prompt_contract()
    evidence = "ALLOWED_CONTEXTUAL_EVIDENCE_SENTINEL"
    messages = build_messages(evidence)
    require(
        messages
        == [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_user_message(evidence)},
        ]
        and evidence in messages[1]["content"],
        "V2 prompt does not contain exactly the supplied candidate evidence",
    )
    forbidden_values = (
        "FORBIDDEN_CORPUS_TERM_SENTINEL",
        "FORBIDDEN_CORPUS_CONTEXT_SENTINEL",
        "FORBIDDEN_SENSE1_SENTINEL",
        "FORBIDDEN_EMBEDDING_SENTINEL",
        "FORBIDDEN_TERMINLINK_SENTINEL",
        "FORBIDDEN_EVALUATION_SENTINEL",
        "FORBIDDEN_NLLB_TRANSLATION_SENTINEL",
        "FORBIDDEN_PREVIOUS_QWEN_TRANSLATION_SENTINEL",
    )
    serialized = json.dumps(messages, ensure_ascii=False)
    require(
        all(value not in serialized for value in forbidden_values),
        "Forbidden external evidence reached V2 messages",
    )

    class CapturingTokenizer:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] | None = None

        def apply_chat_template(
            self,
            captured_messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            self.messages = captured_messages
            require(tokenize is False, "Prompt renderer unexpectedly tokenized")
            require(add_generation_prompt is True, "Generation marker is missing")
            require(enable_thinking is False, "Thinking was enabled")
            return "SYNTHETIC_RENDERED_PROMPT"

    capturing_tokenizer = CapturingTokenizer()
    require(
        render_prompt(capturing_tokenizer, evidence)
        == "SYNTHETIC_RENDERED_PROMPT"
        and capturing_tokenizer.messages == messages,
        "Tokenizer boundary did not receive the exact V2 messages",
    )

    successful_calls: list[str] = []
    malformed_calls: list[str] = []

    def successful_generator(_: Any, __: Any, supplied_evidence: str) -> str:
        successful_calls.append(supplied_evidence)
        return "station"

    def malformed_generator(_: Any, __: Any, supplied_evidence: str) -> str:
        malformed_calls.append(supplied_evidence)
        return "Translation: station"

    successful = infer_once(evidence, None, None, successful_generator)
    malformed = infer_once(evidence, None, None, malformed_generator)
    require(
        successful_calls == [evidence]
        and successful["generation_status"] == "OK"
        and successful["translation_clean"] == "station",
        "Successful V2 generation policy changed",
    )
    require(
        malformed_calls == [evidence]
        and malformed["generation_status"] == "ERROR"
        and malformed["translation_raw"] == "Translation: station"
        and not malformed["translation_clean"],
        "Malformed V2 output was repaired or retried",
    )
    require(
        normalize_and_validate_generation("  one\tword  ") == "one word",
        "V2 cleaning performs more than whitespace normalization",
    )
    validate_runtime_metadata(synthetic_runtime_metadata())
    for field, invalid_value in (
        ("quantization_config", "not-json"),
        (
            "quantization_config",
            json.dumps(
                {
                    **FIXED_QUANTIZATION_PARAMETERS,
                    "load_in_4bit": 1,
                    "bnb_4bit_use_double_quant": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
        ("generation_parameters", "{}"),
        ("device_name", "CPU"),
    ):
        invalid_metadata = synthetic_runtime_metadata()
        invalid_metadata[field] = invalid_value
        try:
            validate_runtime_metadata(invalid_metadata)
        except ContextualV2Error:
            pass
        else:
            raise ContextualV2Error(
                f"Invalid runtime provenance was accepted: {field}"
            )
    require(
        FIXED_QUANTIZATION_PARAMETERS
        == {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }
        and FIXED_GENERATION_PARAMETERS
        == {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "use_cache": True,
        }
        and ENABLE_THINKING is False
        and GENERATION_ATTEMPTS == 1,
        "V2 technical parameters changed",
    )
    print("V2_PROMPT_CONTRACT_TEST=OK")
    print("V2_CONTEXT_EVIDENCE_ONLY_TEST=OK")
    print("V2_TOKENIZER_BOUNDARY_TEST=OK")
    print("V2_SINGLE_GENERATION_TEST=OK")
    print("V2_NO_RETRY_TEST=OK")
    print("V2_NO_SEMANTIC_REPAIR_TEST=OK")
    print("V2_TECHNICAL_PARAMETERS_TEST=OK")
    print("V2_RUNTIME_METADATA_CONTRACT_TEST=OK")
