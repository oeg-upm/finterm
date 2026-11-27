"""Production English Punkt segmentation, independent of notebooks/corpora."""

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
from threading import RLock


_PUNKT_LOCK = RLock()


@dataclass(frozen=True)
class Sentence:
    number: int
    start: int
    end: int
    text: str


class SegmentationError(RuntimeError):
    pass


def sentence_slices(text, spans):
    """Validate full non-whitespace coverage without rewriting original slices."""
    previous, result = 0, []
    for number, (left, right) in enumerate(spans, 1):
        if (type(left) is not int or type(right) is not int
                or not 0 <= previous <= left < right <= len(text)
                or text[previous:left].strip() or not text[left:right].strip()):
            raise SegmentationError("Punkt omitted/overlapped original content or produced an empty unit")
        result.append(Sentence(number, left, right, text[left:right])); previous = right
    if text[previous:].strip():
        raise SegmentationError("Punkt did not cover the complete input")
    return tuple(result)


class PunktSegmenter:
    """Lazy, original English Punkt parameters; never downloads resources."""
    def __init__(self, data_root=None):
        value = data_root or os.environ.get("FINTERM_PUNKT_ROOT")
        self.data_root = Path(value).expanduser().resolve() if value else None
        self.info = {"policy": "english_punkt_original_slices_v1", "language": "english"}

    def segment(self, text):
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if not text.strip():
            return ()
        try:
            import nltk.data
            from nltk.tokenize.punkt import PunktTokenizer
            from experimental_evaluation.resources import PUNKT_FILES, private_path
            from experimental_evaluation.common import file_hash
            root = private_path(self.data_root) if self.data_root is not None else None
            with _PUNKT_LOCK:
                old = list(nltk.data.path)
                try:
                    if root is not None:
                        nltk.data.path[:] = [str(root)]
                    actual = Path(nltk.data.find("tokenizers/punkt_tab/english/")).resolve()
                    private_path(actual)
                    files = {name: file_hash(private_path(actual / name)) for name in PUNKT_FILES}
                    tokenizer = PunktTokenizer("english")
                    spans = list(tokenizer.span_tokenize(text))
                finally:
                    nltk.data.path[:] = old
            self.info = {**self.info, "nltk_version": metadata.version("nltk"),
                         "parameters_root": str(actual), "sha256": files,
                         "boundaries_human_verified": False}
            return sentence_slices(text, spans)
        except SegmentationError:
            raise
        except (ImportError, LookupError, OSError, ValueError) as exc:
            raise SegmentationError(
                "English Punkt is unavailable/untrusted. Install the text extra and provision "
                "punkt_tab in a private NLTK directory; no text was truncated."
            ) from exc
