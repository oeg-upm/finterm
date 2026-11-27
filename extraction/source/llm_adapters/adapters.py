"""Single-input Hugging Face adapters; no corpus reader or experiment runner."""

from copy import deepcopy
from typing import Callable

from prompts.dister import DiSTERPrompt
from prompts.models import ChatPrompt

from .config import AdapterProfile, GenerationOptions, LoadConfig, MODEL_IDS
from .inputs import PreparedInput, prepare_dister, prepare_gemma2, prepare_qwen3
from .responses import GenerationResult, validate_terms_json
from .runtime import HuggingFaceRuntime, load_huggingface_runtime


class AdapterExecutionError(RuntimeError):
    """Input/generation failure; no synthetic terms or repaired response is returned."""


class HuggingFaceAdapter:
    """Shared functional execution path with explicit lazy loading.

    A runtime_loader may be injected to test call contracts without real weights.
    Normal construction always selects load_huggingface_runtime.
    """

    def __init__(
        self,
        key: str,
        load: LoadConfig,
        generation: GenerationOptions,
        *,
        runtime_loader: Callable[[str, LoadConfig], HuggingFaceRuntime] | None = None,
    ) -> None:
        if key not in MODEL_IDS:
            raise ValueError("Unsupported model adapter")
        if not isinstance(load, LoadConfig) or not isinstance(generation, GenerationOptions):
            raise TypeError("Expected LoadConfig and GenerationOptions")
        self.key = key
        self.model_id = MODEL_IDS[key]
        self.load_config = load
        self.generation_options = generation
        self._runtime_loader = runtime_loader if runtime_loader is not None else load_huggingface_runtime
        self._runtime: HuggingFaceRuntime | None = None

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def prepare_input(self, prompt: ChatPrompt | DiSTERPrompt) -> PreparedInput:
        """Prepare transport only; no imports, model/tokenizer loading or execution."""
        preparers = {"qwen3": prepare_qwen3, "gemma2": prepare_gemma2, "dister": prepare_dister}
        return preparers[self.key](prompt)

    def load(self) -> HuggingFaceRuntime:
        """Explicit execution boundary. Never called automatically at construction."""
        if self._runtime is None:
            loaded = self._runtime_loader(self.model_id, self.load_config)
            if not isinstance(loaded, HuggingFaceRuntime):
                raise TypeError("The runtime loader must return HuggingFaceRuntime")
            self._runtime = loaded
        return self._runtime

    def _generation_config(self, runtime: HuggingFaceRuntime):
        # Preserve model-specific EOS/turn termination and copy, never mutate
        # the model's config. Override decoding even if repo defaults sample.
        result = deepcopy(runtime.model.generation_config)
        options = self.generation_options
        result.do_sample = options.do_sample
        result.num_beams = options.num_beams
        result.num_beam_groups = 1
        result.num_return_sequences = 1
        result.max_new_tokens = options.max_new_tokens
        result.return_dict_in_generate = False
        result.output_scores = False
        result.output_hidden_states = False
        result.output_attentions = False
        result.penalty_alpha = None
        if not options.do_sample:
            for name, value in {
                "temperature": 1.0, "top_p": 1.0, "top_k": 50, "min_p": None,
                "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
            }.items():
                setattr(result, name, value)
        else:
            for name in ("temperature", "top_p", "top_k"):
                if getattr(options, name) is not None:
                    setattr(result, name, getattr(options, name))
        # Some causal tokenizers have no pad ID; use an existing EOS for this
        # single-input call only, without changing the tokenizer or its vocabulary.
        if getattr(result, "pad_token_id", None) is None:
            pad_id = getattr(runtime.tokenizer, "pad_token_id", None)
            if pad_id is None:
                eos_id = getattr(result, "eos_token_id", None)
                pad_id = eos_id[0] if isinstance(eos_id, (list, tuple)) and eos_id else eos_id
            if pad_id is not None:
                result.pad_token_id = pad_id
        return result

    def _check_context(self, runtime: HuggingFaceRuntime, input_length: int) -> None:
        limits = [self.load_config.context_limit, getattr(runtime.model.config, "max_position_embeddings", None)]
        limits = [value for value in limits if type(value) is int and value > 0]
        if limits and input_length + self.generation_options.max_new_tokens > min(limits):
            raise AdapterExecutionError(
                "Input plus requested generation exceeds the context limit; "
                "the frozen prompt has not been truncated or rewritten"
            )

    def generate(self, prompt: ChatPrompt | DiSTERPrompt) -> GenerationResult:
        """Execute the real model ONLY on this explicit call (mocked in tests)."""
        prepared = self.prepare_input(prompt)  # Reject wrong prompt types before loading.
        runtime = self.load()
        if getattr(runtime.model.config, "is_encoder_decoder", False):
            raise AdapterExecutionError("These adapters require a decoder-only CausalLM")
        rendered, inputs = prepared.tokenize(runtime.tokenizer)
        input_ids = inputs["input_ids"]
        if len(input_ids.shape) != 2 or input_ids.shape[0] != 1 or input_ids.shape[-1] < 1:
            raise AdapterExecutionError("Expected one nonempty tokenized input")
        input_length = input_ids.shape[-1]
        self._check_context(runtime, input_length)

        embeddings = runtime.model.get_input_embeddings()
        hook = getattr(embeddings, "_hf_hook", None)
        device = getattr(hook, "execution_device", None)
        if device is None:
            device = embeddings.weight.device
        if str(device) == "meta":
            raise AdapterExecutionError("Cannot send input to a meta device")
        inputs = inputs.to(device)
        generation_config = self._generation_config(runtime)
        with runtime.torch.inference_mode():
            output = runtime.model.generate(**inputs, generation_config=generation_config)
        if len(output.shape) != 2 or output.shape[0] != 1 or output.shape[-1] < input_length:
            raise AdapterExecutionError("Unexpected generated sequence shape")
        # Decoder-only outputs include the prompt. Slice by token position, never
        # by a decoded-character prefix (particularly important for DiSTER).
        new_ids = output[0, input_length:]
        token_ids = tuple(int(token) for token in new_ids.tolist())
        raw = runtime.tokenizer.decode(
            new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        response_text = runtime.tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return GenerationResult(
            model_id=self.model_id,
            source=self.load_config.source if self.load_config.source is not None else self.model_id,
            revision=self.load_config.revision, rendered_input=rendered,
            raw_response=raw, response_text=response_text,
            generated_token_ids=token_ids, input_token_count=input_length,
            json_response=None if self.key == "dister" else validate_terms_json(response_text),
        )


class Qwen3Adapter(HuggingFaceAdapter):
    def __init__(self, load: LoadConfig | None = None, generation: GenerationOptions | None = None, *, runtime_loader=None):
        super().__init__(
            "qwen3", load if load is not None else LoadConfig(context_limit=32768),
            generation if generation is not None else GenerationOptions(),
            runtime_loader=runtime_loader,
        )


class Gemma2Adapter(HuggingFaceAdapter):
    def __init__(self, load: LoadConfig | None = None, generation: GenerationOptions | None = None, *, runtime_loader=None):
        super().__init__(
            "gemma2", load if load is not None else LoadConfig(context_limit=8192),
            generation if generation is not None else GenerationOptions(),
            runtime_loader=runtime_loader,
        )


class DiSTERLlamaAdapter(HuggingFaceAdapter):
    def __init__(self, load: LoadConfig | None = None, generation: GenerationOptions | None = None, *, runtime_loader=None):
        super().__init__(
            "dister", load if load is not None else LoadConfig(context_limit=8192),
            generation if generation is not None else GenerationOptions(max_new_tokens=128),
            runtime_loader=runtime_loader,
        )


def adapter_from_profile(profile: AdapterProfile) -> HuggingFaceAdapter:
    adapters = {"qwen3": Qwen3Adapter, "gemma2": Gemma2Adapter, "dister": DiSTERLlamaAdapter}
    return adapters[profile.key](load=profile.load, generation=profile.generation)
