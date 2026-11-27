"""Model execution settings, independent of corpus locations and model imports."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping
import tomllib


MODEL_IDS = MappingProxyType({
    "qwen3": "Qwen/Qwen3-14B",
    "gemma2": "google/gemma-2-9b-it",
    "dister": "ElenaSenger/DiSTER-Llama-3-8B-Instruct",
})


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class LoadConfig:
    # A local mirror may be supplied; None uses the adapter's official model ID.
    source: str | None = None
    revision: str = "main"
    device_map: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    local_files_only: bool = True
    cache_dir: str | None = None
    context_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("revision", "device_map", "attn_implementation"):
            _nonempty(getattr(self, name), name)
        for name in ("source", "cache_dir"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        if self.dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError("Unsupported floating-point dtype")
        if type(self.local_files_only) is not bool:
            raise TypeError("local_files_only must be a boolean")
        if self.context_limit is not None and (
            type(self.context_limit) is not int or self.context_limit < 1
        ):
            raise ValueError("context_limit must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int = 512
    do_sample: bool = False
    num_beams: int = 1
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "num_beams"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.do_sample) is not bool:
            raise TypeError("do_sample must be a boolean")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.top_p is not None and self.top_p > 1:
            raise ValueError("top_p must not exceed 1")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 0):
            raise ValueError("top_k must be a nonnegative integer")
        if not self.do_sample and any(
            value is not None for value in (self.temperature, self.top_p, self.top_k)
        ):
            raise ValueError("Sampling parameters require do_sample=True")


@dataclass(frozen=True, slots=True)
class AdapterProfile:
    key: str
    model_id: str
    load: LoadConfig
    generation: GenerationOptions

    def __post_init__(self) -> None:
        if self.key not in MODEL_IDS or self.model_id != MODEL_IDS[self.key]:
            raise ValueError("The profile must identify one of the three approved models")


def adapter_profiles_from_toml(source: str) -> Mapping[str, AdapterProfile]:
    """Parse supplied TOML contents without opening files or loading resources."""
    if not isinstance(source, str):
        raise TypeError("Supply TOML contents as a string")
    data = tomllib.loads(source)
    if set(data) != set(MODEL_IDS):
        raise ValueError("Declare qwen3, gemma2 and dister profiles")
    profiles = {}
    for key, entry in data.items():
        if set(entry) != {"model_id", "load", "generation"}:
            raise ValueError("Each profile needs model_id, load and generation sections")
        profiles[key] = AdapterProfile(
            key, entry["model_id"], LoadConfig(**entry["load"]),
            GenerationOptions(**entry["generation"]),
        )
    return MappingProxyType(profiles)
