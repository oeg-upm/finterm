"""Operating presets over FinancialExtractor; no separate extraction pipeline."""

from dataclasses import dataclass
import os
from pathlib import Path

from experimental_evaluation.common import file_hash, plain
from experimental_evaluation.resources import Resources
from experimental_evaluation.syntax import ForestConfig

from .financial import FinancialExtractor
from .financial_prompts import PROMPT_VERSION
from .models import MissingRetrievalResourcesError
from .retrieval import load_financial_bank, make_financial_retriever
from .segmentation import PunktSegmenter


SELECTOR_VERSION = "financial_modes_v1"
DEFAULT_BANK = Path(__file__).resolve().parent.parent / "demonstrations/banco_piloto_financiero_10_provisional.jsonl"
DEMONSTRATION_DOCUMENTS = ("CELEX:32013R0549", "CELEX:32021B0417")


@dataclass(frozen=True)
class _Mode:
    model: str
    strategy: str
    fastkassim: bool


_MODES = {"balanced": _Mode("qwen3", "basic", False),
          "high_recall": _Mode("gemma2", "few_shot", True)}
MODE_NAMES = tuple(_MODES)


def validate_mode_options(mode, *, model=None, fastkassim=None, authorized_documents=None,
                          bank=None, java_bin=None, parser_jar=None, models_jar=None, punkt_data=None):
    """Pure option validation shared by Python and CLI, before resource/model I/O.

    Manual selectors are rejected even when redundant with the requested mode.
    Paths, revision and token budget remain explicitly configurable.
    """
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("Select mode 'balanced' or 'high_recall'")
    manual = [name for name, value in (("--model", model), ("--fastkassim", fastkassim),
                                      ("--authorized-document", authorized_documents)) if value is not None]
    if manual:
        raise ValueError("--mode cannot be combined with manual selectors: " + ", ".join(manual))
    spec = _MODES[mode]
    retrieval = {"--bank": bank, "--java-bin": java_bin, "--parser-jar": parser_jar, "--models-jar": models_jar}
    if not spec.fastkassim and any(value is not None for value in retrieval.values()):
        raise ValueError("balanced is basic without FastKASSIM; retrieval resource options require high_recall")
    if spec.fastkassim:
        missing = [name for name, value in retrieval.items() if name != "--bank" and not value]
        if not (punkt_data or os.environ.get("FINTERM_PUNKT_ROOT")):
            missing.append("--punkt-data (or FINTERM_PUNKT_ROOT)")
        if missing:
            raise MissingRetrievalResourcesError("high_recall requires FastKASSIM resources: " + ", ".join(missing))
    return spec


def create_financial_extractor(mode="balanced", *, model_path, punkt_data=None, revision=None,
                               max_new_tokens=None, bank=None, java_bin=None, parser_jar=None, models_jar=None,
                               model=None, fastkassim=None, authorized_documents=None, _engine_factory=None):
    """Select one lazy local model and return the existing FinancialExtractor.

    high_recall uses the current provisional CELEX bank by default; `bank` may
    locate its copy elsewhere. Resource/preflight failures propagate explicitly,
    before the adapter factory. No fallback to basic or to another model.
    `_engine_factory` is the existing CLI dependency-injection test hook.
    """
    spec = validate_mode_options(
        mode, model=model, fastkassim=fastkassim, authorized_documents=authorized_documents,
        bank=bank, java_bin=java_bin, parser_jar=parser_jar, models_jar=models_jar, punkt_data=punkt_data,
    )
    segmenter = PunktSegmenter(punkt_data)
    retriever = None
    retrieval = {"enabled": False, "top_k": 0}
    if spec.fastkassim:
        bank_path = Path(bank if bank is not None else DEFAULT_BANK).expanduser().absolute()
        paths = {"--bank": bank_path, "--java-bin": Path(java_bin).expanduser().absolute(),
                 "--parser-jar": Path(parser_jar).expanduser().absolute(),
                 "--models-jar": Path(models_jar).expanduser().absolute()}
        for name, path in paths.items():
            if not path.is_file() or not path.stat().st_size:
                raise MissingRetrievalResourcesError(f"high_recall needs an existing nonempty {name}: {path}")
        if not segmenter.data_root.is_dir():
            raise MissingRetrievalResourcesError(f"high_recall needs a Punkt data directory: {segmenter.data_root}")
        examples, status = load_financial_bank(bank_path, authorized_documents=DEMONSTRATION_DOCUMENTS, segmenter=segmenter)
        resources = Resources(str(paths["--java-bin"]), str(paths["--parser-jar"]), str(paths["--models-jar"]),
                              str(segmenter.data_root))
        config = ForestConfig()
        retriever = make_financial_retriever(examples, authorized_documents=DEMONSTRATION_DOCUMENTS,
                                            resources=resources, bank_status=status, config=config)
        if retriever is None:
            raise MissingRetrievalResourcesError("high_recall requires the FastKASSIM retriever")
        retrieval = {"enabled": True, "top_k": 3, "bank_path": str(bank_path), "bank_sha256": file_hash(bank_path),
                     "bank_examples": len(examples), "bank_status": status,
                     "authorized_documents": list(DEMONSTRATION_DOCUMENTS), "resources": plain(resources),
                     "forest": plain(config)}
    factory = FinancialExtractor.from_local if _engine_factory is None else _engine_factory
    engine = factory(spec.model, model_path, punkt_data=segmenter.data_root, revision=revision,
                     max_new_tokens=max_new_tokens, retriever=retriever)
    engine.segmenter = segmenter
    engine.mode = mode
    engine.mode_config = {"selector_version": SELECTOR_VERSION, "strategy": spec.strategy,
                          "prompt_version": PROMPT_VERSION, "segmentation": segmenter.info["policy"],
                          "aggregation": "stable_exact_union", "retrieval": retrieval}
    return engine
