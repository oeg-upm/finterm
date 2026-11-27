"""Collaborator CLI; installs/downloads nothing during linking or doctor."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from . import Linker, LinkerConfig, __version__
from ._runtime import frozen_runtime
from .verification import verify_run


def doctor(code_dir=None, *, embedding_resources=None, hub_cache=None):
    with frozen_runtime(code_dir) as runtime:
        from finterm_defterm import load_fewshots, recipe
        from finterm_sense2_qwen import QwenContextualV2, generation_contract
        load_fewshots().validate()
        report = {"version": __version__, "imports_and_contracts_valid": True, "source_export": runtime.integrity,
                  "h100_environment": runtime.native.check_environment(), "models_imported": False,
                  "models": {"translation": generation_contract(), "gate": recipe("gate"),
                             "definition": recipe("definition"), "embedding": runtime.native.contract()}}
        if embedding_resources is not None:
            _, resource = runtime.native.verify_resources(embedding_resources)
            report["embedding_resources"] = {"valid": True, "content_sha256": resource["content_sha256"]}
        if hub_cache is not None:
            snapshot, fingerprints = QwenContextualV2(cache_dir=hub_cache)._snapshot()
            report["qwen14_resources"] = {"valid": True, "snapshot": str(snapshot), "json_sha256": fingerprints}
        return report


def _config(args):
    values = {}
    if args.config:
        path = args.config.resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        valid = {f.name for f in fields(LinkerConfig)}
        if set(values) - valid:
            raise ValueError("Unknown config keys: " + repr(sorted(set(values) - valid)))
        paths = {f.name for f in fields(LinkerConfig) if "Path" in str(f.type)}
        for key in paths & values.keys():
            if values[key] is not None:
                value = Path(values[key]).expanduser()
                values[key] = value if value.is_absolute() else path.parent / value
    for key in ("work_dir", "mode", "offline", "search_policy", "translation_policy", "preselection_policy",
                "lemma_map", "wikidata_seed_cache", "qwen14_hub_cache", "embedding_resource_dir", "gpu_lock",
                "max_translations", "max_gate_calls", "max_definition_calls", "max_new_embeddings", "http_contact"):
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    if args.execution:
        values["similarity_mode"] = args.execution
    return LinkerConfig(**values)


def main(argv=None):
    parser = argparse.ArgumentParser(description="FINTERM original entity candidate ranking pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Imports, hashes and metadata only; no model inference")
    check.add_argument("--require-h100", action="store_true")
    check.add_argument("--embedding-resources", type=Path)
    check.add_argument("--hub-cache", type=Path)
    verify = commands.add_parser("verify", help="Verify a saved run without loading models")
    verify.add_argument("run_dir", type=Path)
    run = commands.add_parser("run", help="Run quick or full; keep all original artifacts")
    run.add_argument("--config", type=Path)
    text = run.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    terms = run.add_mutually_exclusive_group(required=True)
    terms.add_argument("--term", action="append", dest="terms")
    terms.add_argument("--terms-file", type=Path)
    run.add_argument("--mode", choices=("quick", "full"))
    run.add_argument("--execution", choices=("prepare", "h100"))
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--document-id")
    run.add_argument("--work-dir", type=Path)
    network = run.add_mutually_exclusive_group()
    network.add_argument("--offline", dest="offline", action="store_true", default=None)
    network.add_argument("--online", dest="offline", action="store_false")
    run.add_argument("--search-policy", choices=("historical", "coverage"))
    run.add_argument("--translation-policy", choices=("historical", "multilingual"))
    run.add_argument("--preselection-policy", choices=("conservative", "direct_only"))
    for flag in ("lemma-map", "wikidata-seed-cache", "qwen14-hub-cache", "embedding-resource-dir", "gpu-lock"):
        run.add_argument("--" + flag, type=Path)
    for flag in ("max-translations", "max-gate-calls", "max-definition-calls", "max-new-embeddings"):
        run.add_argument("--" + flag, type=int)
    run.add_argument("--contact", dest="http_contact")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor(embedding_resources=args.embedding_resources, hub_cache=args.hub_cache)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2 if args.require_h100 and not report["h100_environment"]["compatible"] else 0
        if args.command == "verify":
            print(json.dumps(verify_run(args.run_dir), ensure_ascii=False, indent=2))
            return 0
        config = _config(args)
        original = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
        supplied = args.terms if args.terms is not None else [line for line in args.terms_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        result = Linker(config).link(original, supplied, document_id=args.document_id, top_k=args.top_k)
        print(json.dumps({"run_dir": str(result.run_dir), "mode": result.mode, "status": result.status,
            "execution_kind": result.execution_kind, "top_candidates": result.top(args.top_k),
            "inferences": result.trace["inferences"], "coverage": result.trace["coverage"],
            "final_link_selected": False}, ensure_ascii=False, indent=2))
        return 0 if result.status in {"COMPLETED", "PREPARED"} else 2
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "ERROR", "type": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
