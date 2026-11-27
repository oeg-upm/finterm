"""The sole optional-dependency/weights loading boundary, called explicitly."""

from dataclasses import dataclass
from typing import Any

from .config import LoadConfig


class AdapterLoadError(RuntimeError):
    """The real Hugging Face runtime could not be loaded; no fallback response."""


@dataclass(slots=True)
class HuggingFaceRuntime:
    tokenizer: Any
    model: Any
    torch: Any


def load_huggingface_runtime(model_id: str, config: LoadConfig) -> HuggingFaceRuntime:
    """Load the real tokenizer and CausalLM only when explicitly requested.

    This function is not invoked by imports, configurations, constructors, or
    pure prepare_input(). Tests replace this boundary with lightweight doubles.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise AdapterLoadError("Prepare the optional llm dependencies on the H100 server") from exc

    source = config.source if config.source is not None else model_id
    common = {
        "revision": config.revision, "local_files_only": config.local_files_only,
        "cache_dir": config.cache_dir, "trust_remote_code": False,
    }
    dtype = "auto" if config.dtype == "auto" else getattr(torch, config.dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        model = AutoModelForCausalLM.from_pretrained(
            source, **common, device_map=config.device_map, torch_dtype=dtype,
            attn_implementation=config.attn_implementation, use_safetensors=True,
        )
        model.eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise AdapterLoadError(f"Could not load the requested Hugging Face model: {model_id}") from exc
    return HuggingFaceRuntime(tokenizer, model, torch)
