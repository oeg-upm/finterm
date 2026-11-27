"""Small installed terminal entry point; no Jupyter or corpus evaluation."""

import argparse
import json
from pathlib import Path
import sys

from experimental_evaluation.common import atomic_json
from experimental_evaluation.resources import Resources

from .financial import FinancialExtractor
from .modes import MODE_NAMES, create_financial_extractor, validate_mode_options
from .retrieval import load_financial_bank, make_financial_retriever
from .segmentation import PunktSegmenter


def parser():
    value = argparse.ArgumentParser(description="FINTERM: financial terms/entities, automatic English Punkt sentences")
    value.add_argument("--version", action="version", version="FINTERM 0.2.0 (finance_entities_v2)")
    value.add_argument("--mode", choices=MODE_NAMES, help="Automatic model/strategy; default balanced unless manual selectors are used")
    value.add_argument("--model", choices=("qwen3", "gemma2", "dister"), help="Legacy manual model selection; incompatible with --mode")
    value.add_argument("--model-path", required=True, type=Path, help="Complete local snapshot; never downloaded automatically")
    value.add_argument("--revision", help="Declared model commit (defaults to the previously pinned approved commit)")
    value.add_argument("--punkt-data", type=Path, help="Private NLTK data root containing tokenizers/punkt_tab/english")
    source = value.add_mutually_exclusive_group()
    source.add_argument("--text", help="Original English financial text")
    source.add_argument("--input", type=Path, help="UTF-8 text file, or '-' for stdin; defaults to stdin")
    value.add_argument("--report", type=Path, help="NEW sealed JSON report with original responses and diagnostics; never overwritten")
    value.add_argument("--max-new-tokens", type=int, help="Explicit override, recorded in the report; defaults 512/512/128")
    value.add_argument("--query-document-id", help="Reliable source identity for excluding same-document demonstrations")
    value.add_argument("--fastkassim", action="store_true", default=None, help="Legacy manual original LTK; incompatible with --mode")
    value.add_argument("--bank", type=Path, help="JSONL bank path; high_recall defaults to the provisional ten-example CELEX bank")
    value.add_argument("--authorized-document", action="append", help="Legacy manual bank authorization; modes use their fixed document set")
    value.add_argument("--java-bin", type=Path)
    value.add_argument("--parser-jar", type=Path)
    value.add_argument("--models-jar", type=Path)
    return value


def require_execution_platform():
    if sys.platform != "linux":
        raise RuntimeError("Real model execution is validated on Linux/CUDA (H100); Windows is for tests and --help")


def read_input(args):
    if args.text is not None:
        return args.text
    if args.input is None or str(args.input) == "-":
        return sys.stdin.buffer.read().decode("utf-8")
    if args.input.suffix.lower() == ".key":
        raise ValueError("Annotations are auxiliary references, not extraction inputs")
    return args.input.read_bytes().decode("utf-8")


def main(argv=None, *, engine_factory=FinancialExtractor.from_local):
    args = parser().parse_args(argv)
    try:
        if args.report is not None and args.report.suffix.lower() != ".json":
            raise ValueError("Detailed reports must be new .json files, never text/annotation files")
        if args.report is not None and args.report.exists():
            raise FileExistsError("The report already exists; choose a new filename")
        retrieval_args = (args.bank, args.authorized_document, args.java_bin, args.parser_jar, args.models_jar)
        mode = args.mode
        if mode is None and args.model is None and not args.fastkassim and not any(retrieval_args):
            mode = "balanced"
        if mode is not None:
            validate_mode_options(mode, model=args.model, fastkassim=args.fastkassim,
                                  authorized_documents=args.authorized_document, bank=args.bank,
                                  java_bin=args.java_bin, parser_jar=args.parser_jar, models_jar=args.models_jar,
                                  punkt_data=args.punkt_data)
        elif args.fastkassim:
            args.model = args.model or "qwen3"
            if args.model == "dister":
                raise ValueError("DiSTER retains its native conversation without additional demonstrations")
            if not all(retrieval_args) or args.punkt_data is None:
                raise ValueError("FastKASSIM needs --bank, --authorized-document, private Java/JARs and --punkt-data")
        elif any(retrieval_args):
            raise ValueError("Retrieval resource arguments require --fastkassim")
        require_execution_platform()
        text = read_input(args)
        retriever = None
        if mode is not None:
            engine = create_financial_extractor(mode, model_path=args.model_path, punkt_data=args.punkt_data,
                                               revision=args.revision, max_new_tokens=args.max_new_tokens,
                                               bank=args.bank, java_bin=args.java_bin, parser_jar=args.parser_jar,
                                               models_jar=args.models_jar, _engine_factory=engine_factory)
        else:
            if args.fastkassim:
                segmenter = PunktSegmenter(args.punkt_data)
                examples, status = load_financial_bank(args.bank, authorized_documents=args.authorized_document, segmenter=segmenter)
                resources = Resources(str(args.java_bin.resolve()), str(args.parser_jar.resolve()),
                                      str(args.models_jar.resolve()), str(args.punkt_data.resolve()))
                retriever = make_financial_retriever(examples, authorized_documents=args.authorized_document,
                                                    resources=resources, bank_status=status)
            engine = engine_factory(args.model or "qwen3", args.model_path, punkt_data=args.punkt_data, revision=args.revision,
                                    max_new_tokens=args.max_new_tokens, retriever=retriever)
        result = engine.extract_detailed(text, query_document_id=args.query_document_id)
        if args.report is not None:
            atomic_json(args.report, result.to_dict())
        if not result.complete:
            print(json.dumps({"error": "incomplete_extraction", "failed_sentences": [s["number"] for s in result.sentences if s["error"]],
                              "report": str(args.report) if args.report else None}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"terms": list(result.terms)}, ensure_ascii=False))
        warnings = [s["number"] for s in result.sentences if s.get("nonliteral_in_sentence")]
        if warnings:
            print(json.dumps({"nonliteral_output_warning": warnings, "terms_were_not_silently_filtered": True}), file=sys.stderr)
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
