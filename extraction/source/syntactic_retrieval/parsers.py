"""Lazy constituency-parser integration using locally provisioned Stanford PCFG."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Any

from .models import ParsedText, SyntaxTree, require_text


class ParserUnavailableError(RuntimeError):
    """A local parser/resource is missing; nothing is downloaded automatically."""


class ConstituencyParseError(RuntimeError):
    """An input could not be converted to a nonempty constituency forest."""


@dataclass(frozen=True, slots=True)
class StanfordPCFGConfig:
    parser_jar: Path
    models_jar: Path
    model_path: str = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    sentence_segmentation: str = "single_unit"
    java_options: str = "-mx2g"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parser_jar", Path(self.parser_jar))
        object.__setattr__(self, "models_jar", Path(self.models_jar))
        require_text(self.model_path, "model_path")
        require_text(self.java_options, "java_options")
        if self.sentence_segmentation not in ("single_unit", "punkt"):
            raise ValueError("sentence_segmentation must be 'single_unit' or 'punkt'")


class StanfordPCFGParser:
    """Real NLTK/Stanford adapter; construction/import never loads a model.

    single_unit: caller supplies an already delimited unit (the article's setting).
    punkt: use an English Punkt tokenizer that must already be installed locally.
    Backend initialization and parsing happen ONLY on an explicit parse() call.
    """

    def __init__(self, config: StanfordPCFGConfig) -> None:
        self.config = config
        self._backend: Any = None
        self._tokenizer: Any = None
        identity = repr(config).encode("utf-8")
        self.parser_id = "stanford-pcfg:" + hashlib.sha256(identity).hexdigest()

    def _get_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        for path in (self.config.parser_jar, self.config.models_jar):
            if not path.is_file():
                raise ParserUnavailableError(f"Stanford resource not found: {path}")
        if shutil.which("java") is None:
            raise ParserUnavailableError("Java is required to execute Stanford PCFG")
        try:
            from nltk.parse.stanford import StanfordParser

            self._backend = StanfordParser(
                path_to_jar=str(self.config.parser_jar),
                path_to_models_jar=str(self.config.models_jar),
                model_path=self.config.model_path,
                java_options=self.config.java_options,
                encoding="utf8",
            )
        except (ImportError, LookupError, OSError) as exc:
            raise ParserUnavailableError("Cannot initialize the local Stanford parser") from exc
        return self._backend

    def _sentences(self, text: str) -> list[str]:
        if self.config.sentence_segmentation == "single_unit":
            return [" ".join(text.split())]
        if self._tokenizer is None:
            try:
                import nltk.data

                self._tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            except (ImportError, LookupError) as exc:
                raise ParserUnavailableError(
                    "English Punkt must be provisioned locally; on NLTK 3.9+ "
                    "its punkt_tab resources are also required"
                ) from exc
        return [" ".join(sentence.split()) for sentence in self._tokenizer.tokenize(text)]

    def parse(self, text: str) -> ParsedText:
        require_text(text, "text")
        sentences = self._sentences(text)
        if not sentences or any(not sentence for sentence in sentences):
            raise ConstituencyParseError("Segmentation produced no usable sentence")
        backend = self._get_backend()
        try:
            # Stanford's iterator yields an iterator of analyses per input sentence.
            analyses = list(backend.raw_parse_sents(sentences))
            if len(analyses) != len(sentences):
                raise ValueError("Parser result count differs from sentence count")
            trees = tuple(SyntaxTree.from_nltk(next(iter(items))) for items in analyses)
            return ParsedText(trees=trees, parser_id=self.parser_id)
        except (StopIteration, TypeError, ValueError, OSError, RuntimeError) as exc:
            raise ConstituencyParseError("Constituency parsing failed; no score produced") from exc
