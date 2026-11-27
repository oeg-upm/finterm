"""Lazy Qwen3, Gemma-2 and native DiSTER adapters, with no import-time inference."""

from .adapters import (
    AdapterExecutionError, DiSTERLlamaAdapter, Gemma2Adapter, HuggingFaceAdapter,
    Qwen3Adapter, adapter_from_profile,
)
from .config import AdapterProfile, GenerationOptions, LoadConfig, MODEL_IDS, adapter_profiles_from_toml
from .inputs import PreparedInput
from .responses import FormatError, GenerationResult, JsonTermsResponse, validate_terms_json
from .runtime import AdapterLoadError, HuggingFaceRuntime

__all__ = [
    "AdapterExecutionError", "AdapterLoadError", "AdapterProfile", "DiSTERLlamaAdapter",
    "FormatError", "Gemma2Adapter", "GenerationOptions", "GenerationResult",
    "HuggingFaceAdapter", "HuggingFaceRuntime", "JsonTermsResponse", "LoadConfig",
    "MODEL_IDS", "PreparedInput", "Qwen3Adapter", "adapter_from_profile",
    "adapter_profiles_from_toml", "validate_terms_json",
]
