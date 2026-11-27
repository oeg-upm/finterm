"""Portable code identity and explicit configuration/resource inventories."""

import argparse
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sys

from llm_adapters import AdapterProfile, GenerationOptions, LoadConfig

from .common import atomic_json, file_hash, plain, read_json, utc_now
from .data import EXPLORATORY, AUDIT_EXPOSURES, EXPOSURE_SOURCE, EVALUATION_DOCUMENT, exposure
from .formats import FORMAT_POLICY
from .metrics import NORMALIZATION
from .resources import check_versions, resource_inventory


CONDITIONS = {"qwen_basic": ("qwen3", "basic"), "qwen_fastkassim": ("qwen3", "few_shot"),
              "gemma_basic": ("gemma2", "basic"), "gemma_fastkassim": ("gemma2", "few_shot"),
              "dister_native": ("dister", "native_zero_shot")}
HISTORICAL_BUDGETS = {"qwen3": 512, "gemma2": 512, "dister": 128}
CODE_DIRS = ("extractor", "llm_adapters", "prompts", "syntactic_retrieval",
             "corpus_readers", "experimental_evaluation")


def code_inventory(project):
    project = Path(project).resolve()
    paths = [p for directory in CODE_DIRS for p in (project / directory).rglob("*.py")]
    paths += list((project / "config").glob("*.toml"))
    paths += [project / p for p in ("pyproject.toml", "demonstrations/__init__.py", "demonstrations/matching.py")]
    return {"files": {p.relative_to(project).as_posix(): file_hash(p) for p in sorted(paths)}}


def verify_code(project, expected):
    project = Path(project).resolve()
    for name, module in tuple(sys.modules.items()):
        if (name.split(".")[0] in (*CODE_DIRS, "demonstrations")) and getattr(module, "__file__", None):
            origin = Path(module.__file__).resolve()
            if project not in origin.parents:
                raise ValueError(f"Module from another project/kernel state: {name}: {origin}")
    actual = code_inventory(project)
    if actual != expected:
        differences = sorted(k for k in set(actual["files"]) | set(expected["files"])
                             if actual["files"].get(k) != expected["files"].get(k))
        raise ValueError(f"Code/configuration mismatch; inspect before proceeding: {differences}")
    return actual


def environment():
    names = ("torch", "transformers", "accelerate", "safetensors", "nltk", "fkassim", "numpy", "scipy", "zss", "tokenizers")
    result = {"python": sys.version, "platform": platform.platform(), "executable": sys.executable, "packages": {}}
    for name in names:
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["packages"][name] = None
    return result


def _model_relative_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(c) < 32 for c in name) or PurePosixPath(name).is_absolute()
            or PureWindowsPath(name).drive or any(part in {"", ".", ".."} or part.endswith((" ", "."))
                                                   for part in name.split("/"))):
        raise ValueError(f"Unsafe model asset name: {name!r}")
    return name


def _directory_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _hf_blobs_directory(root, model_id):
    """Authorize only the selected official repository's NORMAL cache layout.

    No global cache search or user-controlled arbitrary external directory.
    A source directory symlink is allowed when its resolved root has this layout.
    """
    if (root.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-fA-F]{40}", root.name)
            or root.parent.parent.name != "models--" + model_id.replace("/", "--")):
        return None
    repo = root.parent.parent
    blobs = repo / "blobs"
    if _directory_link(blobs):
        raise ValueError("The authorized Hugging Face blobs directory cannot itself be a link")
    if not blobs.exists():
        return None  # A fully copied snapshot still works without a blobs store.
    resolved = blobs.resolve(strict=True)
    if not blobs.is_dir() or resolved.parent != repo or resolved.name != "blobs":
        raise ValueError("Unsafe Hugging Face repository blobs directory")
    return resolved


def _model_asset_path(root, name, blobs=None):
    """Validate every file/link hop before reading; fail closed on bad links.

    The INDEX may never contain '..'. A legitimate link target such as
    '../../blobs/<hash>' may use it, but its normalized destination must stay
    inside this snapshot or its explicitly authorized repository blobs store.
    Directory links and an intermediate hop outside both roots are rejected.
    """
    path = root / _model_relative_name(name)
    allowed, seen = (root,) if blobs is None else (root, blobs), set()
    try:
        while True:
            container = next((base for base in allowed if path == base or base in path.parents), None)
            if container is None:
                raise ValueError(f"External model asset link: {name}")
            parent = container
            for part in path.relative_to(container).parts[:-1]:
                parent = parent / part
                if _directory_link(parent) or not parent.is_dir():
                    raise ValueError(f"Unsafe model asset parent directory: {name}")
            if path.is_symlink():
                if path in seen or len(seen) >= 40:
                    raise ValueError(f"Cyclic/excessive model asset link chain: {name}")
                seen.add(path)
                target = path.readlink()
                path = Path(os.path.abspath(target if target.is_absolute() else path.parent / target))
                continue
            resolved = path.resolve(strict=True)
            if not any(base in resolved.parents for base in allowed) or not resolved.is_file() or not resolved.stat().st_size:
                raise ValueError(f"Missing/unsafe model asset: {name}")
            return resolved
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Missing/unsafe model asset: {name}") from exc


def model_inventory(profile, *, hash_weights=False, allow_hf_cache_blobs=False):
    """Snapshot-only by default; optional same-repository HF cache file links.

    Normal copied directories retain the historical inventory shape/values.
    File-name/traversal and broken-link checks apply before config/index reads.
    """
    if type(allow_hf_cache_blobs) is not bool:
        raise TypeError("allow_hf_cache_blobs must be an explicit boolean")
    source = Path(profile.load.source or "")
    if ".." in source.parts:
        raise ValueError("Model source must not contain '..' path components")
    root = source.resolve()
    if not profile.load.source or not root.is_dir() or not profile.load.local_files_only:
        raise ValueError("Supply a complete local model snapshot with local_files_only=True")
    blobs = _hf_blobs_directory(root, profile.model_id) if allow_hf_cache_blobs else None
    config = json.loads(_model_asset_path(root, "config.json", blobs).read_text(encoding="utf-8"))
    expected_type = {"qwen3": "qwen3", "gemma2": "gemma2", "dister": "llama"}[profile.key]
    if config.get("model_type") != expected_type:
        raise ValueError("Wrong model architecture in the local snapshot")
    names = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "tokenizer.model", "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt",
             "model.safetensors.index.json")
    files, resolved_files = {}, {}
    for name in names:
        candidate = root / name
        if candidate.exists() or candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            path = _model_asset_path(root, name, blobs)
            resolved_files[name] = path
            files[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if "tokenizer_config.json" not in files or not ({"tokenizer.json", "tokenizer.model"} & files.keys()):
        raise ValueError("Tokenizer files are incomplete")
    index_path = resolved_files.get("model.safetensors.index.json")
    if index_path is not None:
        mapping = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Model index must declare a nonempty weight_map")
        names = {_model_relative_name(name) for name in mapping.values()}
    else:
        names = {"model.safetensors"}
    shards = {}
    for name in sorted(names):
        path = _model_asset_path(root, name, blobs)
        shards[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path) if hash_weights else None}
    result = {"metadata_files": files, "weight_shards": shards,
              "weight_content_hashes_verified": hash_weights, "revision_declared": profile.load.revision}
    if allow_hf_cache_blobs:
        result["local_file_policy"] = {"name": "snapshot_or_same_hf_repository_blobs_v1",
                                       "snapshot_root": str(root),
                                       "authorized_blobs": str(blobs) if blobs is not None else None}
    return result


def profile_from_dict(data):
    return AdapterProfile(data["key"], data["model_id"], LoadConfig(**data["load"]),
                          GenerationOptions(**data["generation"]))


def make_manifest(*, project, queries, bank, profiles, resources, forest_config,
                  purpose, experiment_label, expected_code, generation_label="historical_512_512_128",
                  hash_model_weights=False):
    """Preparation reads text/bank/configuration/resources, NEVER evaluation keys."""
    check_versions()
    if purpose not in {"technical", "development", "confirmatory"} or not experiment_label.strip():
        raise ValueError("Declare purpose and experiment label")
    if not queries or len({q["identifier"] for q in queries}) != len(queries):
        raise ValueError("Freeze a nonempty, explicit, unique sample")
    if len({q["split"] for q in queries}) != 1:
        raise ValueError("Do not pool development, exploratory and test in one run")
    for query in queries:
        number = query["segment_number"]
        if (type(number) is not int or number < 0 or query["document_id"] != EVALUATION_DOCUMENT
                or query["identifier"] != f"CELEX:32013R0575_seg_{number}"
                or query["prior_exposure"] != exposure(number)
                or query["split"] not in {"exploratory", "development", "test"}
                or (number in EXPLORATORY and query["split"] != "exploratory")
                or (number in AUDIT_EXPOSURES and query["split"] == "test")):
            raise ValueError("Invalid sample identity/split or altered exposure declaration")
    if purpose == "confirmatory" and (not bank.all_human_reviewed or any(q["split"] != "test" for q in queries)):
        raise ValueError("Confirmatory runs require an explicitly reviewed bank and untouched test split")
    if set(profiles) != set(HISTORICAL_BUDGETS):
        raise ValueError("Declare the three approved model profiles for the five configurations")
    for key, profile in profiles.items():
        if profile.key != key or profile.generation.do_sample or profile.generation.num_beams != 1:
            raise ValueError("Reuse deterministic approved profiles")
        if not re.fullmatch(r"[0-9a-f]{40}", profile.load.revision):
            raise ValueError("Pin an explicit model commit, not main")
        if generation_label == "historical_512_512_128" and profile.generation.max_new_tokens != HISTORICAL_BUDGETS[key]:
            raise ValueError("A different token budget requires a different generation_label")
    if not generation_label.strip():
        raise ValueError("Name the generation experiment")
    return {"schema": "finterm-evaluation-v1", "created_at": utc_now(), "purpose": purpose,
            "experiment_label": experiment_label, "generation_label": generation_label,
            "conditions": plain(CONDITIONS), "queries": queries, "bank": bank.snapshot(),
            "code": verify_code(project, expected_code), "environment": environment(),
            "profiles": {key: plain(profile) for key, profile in profiles.items()},
            "model_files": {key: model_inventory(profile, hash_weights=hash_model_weights) for key, profile in profiles.items()},
            "resources": plain(resources), "resource_inventory": resource_inventory(resources),
            "forest": plain(forest_config), "format_policy": FORMAT_POLICY, "normalization": NORMALIZATION,
            "prompt_toml": (Path(project) / "config/prompts.toml").read_text(encoding="utf-8"),
            "exposures": {"historical_exploratory": sorted(EXPLORATORY),
                          "additional_previews": sorted(AUDIT_EXPOSURES), "source": EXPOSURE_SOURCE},
            "reference_kind": "automatic_silver_not_human_quality"}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Export a source-only inventory without models or corpus access")
    cli.add_argument("--project", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    atomic_json(args.output, code_inventory(args.project))
