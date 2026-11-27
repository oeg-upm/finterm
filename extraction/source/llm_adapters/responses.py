"""Strict output-format interpretation, without repairing or normalizing terms."""

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class FormatError:
    code: str
    message: str
    location: str = "$"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "location": self.location}


@dataclass(frozen=True, slots=True)
class JsonTermsResponse:
    response: str
    json_valid: bool
    schema_valid: bool
    terms: tuple[str, ...] | None
    errors: tuple[FormatError, ...]
    parsed_json: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response, "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "terms": list(self.terms) if self.terms is not None else None,
            "errors": [error.to_dict() for error in self.errors],
            "parsed_json": self.parsed_json,
        }


def validate_terms_json(response: str) -> JsonTermsResponse:
    """Validate the entire decoded response against {\"terms\": [str, ...]}.

    No markdown removal, substring extraction, deduplication, case conversion,
    whitespace repair, term filtering, or conversion of non-string values.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a decoded string")
    duplicates = []

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    try:
        parsed = json.loads(
            response, object_pairs_hook=object_pairs, parse_constant=reject_constant
        )
    except json.JSONDecodeError as exc:
        error = FormatError("invalid_json", exc.msg, f"line {exc.lineno}, column {exc.colno}")
        return JsonTermsResponse(response, False, False, None, (error,))
    except (ValueError, RecursionError) as exc:
        return JsonTermsResponse(
            response, False, False, None, (FormatError("invalid_json", str(exc)),)
        )

    errors = [FormatError("duplicate_key", f"Duplicate JSON key: {key}") for key in duplicates]
    if not isinstance(parsed, dict):
        errors.append(FormatError("not_object", "The top-level JSON value must be an object"))
    else:
        if "terms" not in parsed:
            errors.append(FormatError("missing_terms", "Missing terms property"))
        if set(parsed) - {"terms"}:
            errors.append(FormatError("extra_properties", "Only the terms property is allowed"))
        if "terms" in parsed:
            if not isinstance(parsed["terms"], list):
                errors.append(FormatError("not_array", "terms must be a JSON array", "$.terms"))
            else:
                for index, term in enumerate(parsed["terms"]):
                    if not isinstance(term, str) or not term.strip():
                        errors.append(FormatError(
                            "invalid_term", "Each term must be a nonempty string",
                            f"$.terms[{index}]",
                        ))
    terms = tuple(parsed["terms"]) if not errors else None
    return JsonTermsResponse(response, True, not errors, terms, tuple(errors), parsed)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    model_id: str
    source: str
    revision: str
    rendered_input: str
    # raw_response also retains generated special tokens; response_text removes
    # only tokenizer-defined special tokens. Neither is stripped or corrected.
    raw_response: str
    response_text: str
    generated_token_ids: tuple[int, ...]
    input_token_count: int
    json_response: JsonTermsResponse | None

    @property
    def terms(self) -> tuple[str, ...] | None:
        return self.json_response.terms if self.json_response is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "source": self.source, "revision": self.revision,
            "rendered_input": self.rendered_input, "raw_response": self.raw_response,
            "response_text": self.response_text,
            "generated_token_ids": list(self.generated_token_ids),
            "input_token_count": self.input_token_count,
            "json_response": self.json_response.to_dict() if self.json_response is not None else None,
        }
