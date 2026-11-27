"""Pure configuration from arguments/TOML strings; never open a corpus or file."""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping
import tomllib

from .models import AnnotationPolicy, DomainSpec, NestedTerms, PromptCondition, PromptConfig
from .templates import FINANCIAL_STRUCTURES, GENERAL_STRUCTURES


_DOMAINS = {
    "finance": DomainSpec(
        "finance", "Finance",
        "Finance (banking, accounting, financial markets, financial regulation).",
        ("the higher interest rate", "interest rate"),
    ),
    "corruption": DomainSpec("corruption", "corruption", "Corruption."),
    "equitation_dressage": DomainSpec(
        "equitation_dressage", "equitation/dressage", "Equitation/dressage."
    ),
    "heart_failure": DomainSpec("heart_failure", "heart failure", "Heart failure."),
    "wind_energy": DomainSpec("wind_energy", "wind energy", "Wind energy."),
}


def default_prompt_config(domain_key: str) -> PromptConfig:
    """Explicit domain selection, with no silent fallback to finance."""
    domain = _DOMAINS[domain_key]
    nesting = NestedTerms.WHEN_PERMITTED if domain_key == "finance" else NestedTerms.LONGEST_ONLY
    return PromptConfig(
        domain=domain,
        policy=AnnotationPolicy(nested_terms=nesting),
        structure_guide=(FINANCIAL_STRUCTURES if domain_key == "finance" else GENERAL_STRUCTURES),
    )


@dataclass(frozen=True, slots=True)
class PromptSettings:
    configurations: Mapping[str, PromptConfig]
    conditions: tuple[PromptCondition, ...]

    def __post_init__(self) -> None:
        configurations = dict(self.configurations)
        if not configurations or not all(
            key == config.domain.key for key, config in configurations.items()
        ):
            raise ValueError("Every configuration key must match its domain key")
        conditions = tuple(PromptCondition(value) for value in self.conditions)
        if len(conditions) != 4 or set(conditions) != set(PromptCondition):
            raise ValueError("Declare each of the four prompt conditions exactly once")
        object.__setattr__(self, "configurations", MappingProxyType(configurations))
        object.__setattr__(self, "conditions", conditions)

    def for_domain(self, domain_key: str) -> PromptConfig:
        return self.configurations[domain_key]


def prompt_settings_from_toml(source: str) -> PromptSettings:
    """The caller supplies TOML contents. This function performs no file access."""
    if not isinstance(source, str):
        raise TypeError("Supply TOML contents as a string, not a file or path object")
    data = tomllib.loads(source)
    if set(data) != {"conditions", "few_shot_examples", "domains"}:
        raise ValueError("Expected conditions, few_shot_examples and domains sections")
    count = data["few_shot_examples"]
    configurations = {}
    for key, entry in data["domains"].items():
        options = dict(entry)
        name = options.pop("name")
        description = options.pop("description")
        profile = options.pop("structure_profile")
        if profile not in ("financial", "general"):
            raise ValueError("structure_profile must be 'financial' or 'general'")
        guide = FINANCIAL_STRUCTURES if profile == "financial" else GENERAL_STRUCTURES
        guide = replace(
            guide,
            structures=options.pop("structure_lines", guide.structures),
            acronym_structure=options.pop("acronym_structure", guide.acronym_structure),
        )
        default_boundary = _DOMAINS[key].boundary_example if key in _DOMAINS else None
        boundary = options.pop("boundary_example", default_boundary)
        policy = AnnotationPolicy(**options)
        configurations[key] = PromptConfig(
            domain=DomainSpec(key, name, description, boundary),
            policy=policy, few_shot_examples=count, structure_guide=guide,
        )
    return PromptSettings(configurations, tuple(data["conditions"]))
