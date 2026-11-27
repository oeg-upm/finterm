"""Minimal routing defaults, supplied as Python objects or TOML contents."""

from dataclasses import dataclass
import tomllib

from .models import ExtractionCondition, model_key, validate_combination


@dataclass(frozen=True, slots=True)
class ExtractorConfig:
    default_model: str = "qwen3"
    default_condition: ExtractionCondition = ExtractionCondition.BASIC

    def __post_init__(self) -> None:
        key = model_key(self.default_model)
        condition = ExtractionCondition(self.default_condition)
        validate_combination(key, condition)
        object.__setattr__(self, "default_model", key)
        object.__setattr__(self, "default_condition", condition)


def extractor_config_from_toml(source: str) -> ExtractorConfig:
    """Parse a supplied string; no configuration/corpus file is opened."""
    if not isinstance(source, str):
        raise TypeError("Supply TOML contents as a string")
    data = tomllib.loads(source)
    if set(data) != {"extractor"}:
        raise ValueError("Expected only an extractor table")
    return ExtractorConfig(**data["extractor"])
