"""Protocol doubles ONLY for unit tests: no torch tensors or real inference."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

from llm_adapters.runtime import HuggingFaceRuntime


class TensorDouble:
    def __init__(self, values):
        self.values = values
        self.shape = (
            (len(values), len(values[0]))
            if values and isinstance(values[0], list) else (len(values),)
        )

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, columns = key
            return TensorDouble(self.values[row][columns])
        return TensorDouble(self.values[key])

    def tolist(self):
        return deepcopy(self.values)


class EncodingDouble(dict):
    def __init__(self):
        super().__init__(input_ids=TensorDouble([[11, 12, 13]]), attention_mask=TensorDouble([[1, 1, 1]]))
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class TokenizerDouble:
    def __init__(self, response=' {"terms": ["fixture-A", "fixture-A"]}\n'):
        self.response = response
        self.pad_token_id = None
        self.template_calls = []
        self.tokenization_calls = []
        self.decode_calls = []
        self.last_encoding = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((deepcopy(messages), kwargs))
        # Deliberately NOT a reproduction of a model template. Tests inspect
        # the arguments; the actual implementation delegates to HF tokenizers.
        return "TEMPLATE_TEST_DOUBLE\n" + "\n".join(item["content"] for item in messages)

    def __call__(self, text, **kwargs):
        self.tokenization_calls.append((text, kwargs))
        self.last_encoding = EncodingDouble()
        return self.last_encoding

    def decode(self, token_ids, **kwargs):
        self.decode_calls.append((token_ids.tolist(), kwargs))
        if kwargs["skip_special_tokens"]:
            return self.response
        return self.response + "<fixture_eos>"


class TorchDouble:
    def __init__(self):
        self.inference_active = False
        self.inference_entries = 0

    @contextmanager
    def inference_mode(self):
        self.inference_entries += 1
        self.inference_active = True
        try:
            yield
        finally:
            self.inference_active = False


class ModelDouble:
    def __init__(self, torch):
        self.torch = torch
        self.config = SimpleNamespace(is_encoder_decoder=False, max_position_embeddings=8192)
        self.generation_config = SimpleNamespace(
            do_sample=True, num_beams=4, num_return_sequences=2,
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.1,
            typical_p=0.8, epsilon_cutoff=0.1, eta_cutoff=0.1,
            eos_token_id=[20, 21], pad_token_id=None,
        )
        self.embeddings = SimpleNamespace(weight=SimpleNamespace(device="cuda:0"))
        self.calls = []
        self.eval_calls = 0
        self.output = TensorDouble([[11, 12, 13, 31, 32, 21]])

    def eval(self):
        self.eval_calls += 1
        return self

    def get_input_embeddings(self):
        return self.embeddings

    def generate(self, **kwargs):
        if not self.torch.inference_active:
            raise AssertionError("Expected inference_mode context")
        self.calls.append(kwargs)
        return self.output


def runtime_double(response=' {"terms": ["fixture-A", "fixture-A"]}\n'):
    torch = TorchDouble()
    return HuggingFaceRuntime(TokenizerDouble(response), ModelDouble(torch), torch)
