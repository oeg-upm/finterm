"""Predefined, model-neutral wrapper recovery; original responses stay intact."""

import json
import re

from llm_adapters import validate_terms_json


FORMAT_POLICY = "whole_json_then_full_fence_then_unambiguous_json_prefix_v1"
FENCE = re.compile(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", re.I | re.S)


def reject_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def strict(text, schema):
    if schema == "terms_object":
        validated = validate_terms_json(text)
        return {"accepted": validated.schema_valid,
                "terms": list(validated.terms) if validated.terms is not None else None,
                "errors": [error.to_dict() for error in validated.errors]}
    if schema != "native_json_array":
        raise ValueError("The output contract must be explicit")
    try:
        value = json.loads(text, parse_constant=reject_constant)
        if not isinstance(value, list) or any(not isinstance(t, str) or not t.strip() for t in value):
            raise ValueError("Expected a complete JSON array of nonempty strings")
        return {"accepted": True, "terms": value, "errors": []}
    except (ValueError, RecursionError) as exc:
        return {"accepted": False, "terms": None, "errors": [{"code": "invalid_native_array", "message": str(exc)}]}


def interpret(text, schema):
    original = strict(text, schema)
    if original["accepted"]:
        return {"policy": FORMAT_POLICY, "schema": schema, "strict": original,
                "recovered": {**original, "rules": [], "discarded_prefix": "", "discarded_suffix": ""}}
    candidate, rules, prefix, suffix = text, [], "", ""
    wrapped = FENCE.fullmatch(candidate)
    if wrapped:
        prefix, suffix = candidate[:wrapped.start(1)], candidate[wrapped.end(1):]
        candidate = wrapped.group(1)
        rules.append("full_markdown_fence")
    accepted = strict(candidate, schema)
    if not accepted["accepted"]:
        try:
            stripped = candidate.lstrip()
            _, end = json.JSONDecoder(parse_constant=reject_constant).raw_decode(stripped)
            remainder = stripped[end:]
            # Never select between several containers or complete a broken one.
            if not remainder.strip() or "[" in remainder or "{" in remainder:
                raise ValueError("No unique JSON prefix with a simple residual suffix")
            accepted = strict(stripped[:end], schema)
            if accepted["accepted"]:
                prefix += candidate[:len(candidate) - len(stripped)]
                suffix = remainder + suffix
                rules.append("complete_json_prefix_with_recorded_suffix")
        except (ValueError, RecursionError):
            pass
    return {"policy": FORMAT_POLICY, "schema": schema, "strict": original,
            "recovered": {**accepted, "rules": rules, "discarded_prefix": prefix,
                          "discarded_suffix": suffix}}
