"""Read-only access to the authoritative flat H100 export.

The lock serializes temporary changes to Python's import search path/bytecode
policy. Foreign modules with the same names are rejected, never substituted.
"""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from threading import RLock
from types import SimpleNamespace

CODE_MANIFEST_SHA256 = "00a9640aa651c69567f4e40e652219ee0e5b741323542b1c549047aadb605812"
_IMPORT_LOCK = RLock()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line]


def write_json(path, value):
    # New run outputs only. A previous artifact can never be overwritten.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def safe_member(root, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe artifact member: " + str(name))
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes its input directory")
    return path


def verify_export(root, *, expected_manifest_sha256=None):
    root = Path(root).resolve()
    path = root / "SHA256_MANIFEST.json"
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and checksum != expected_manifest_sha256:
        raise ValueError("The authorized H100 export manifest differs")
    entries = json.loads(raw)["files"]
    seen = set()
    for entry in entries:
        member = safe_member(root, entry["path"])
        if member in seen or not member.is_file() or member.stat().st_size != entry["size_bytes"]:
            raise ValueError("Export member missing, duplicated or wrong size: " + entry["path"])
        if file_hash(member) != entry["sha256"].lower():
            raise ValueError("Export hash differs: " + entry["path"])
        seen.add(member)
    return {"manifest": str(path), "manifest_sha256": checksum, "file_count": len(entries)}


def verify_outputs(root, manifest_name):
    root = Path(root).resolve()
    manifest = json.loads(safe_member(root, manifest_name).read_text(encoding="utf-8"))
    for name, expected in manifest["output_sha256"].items():
        if file_hash(safe_member(root, name)) != expected.lower():
            raise ValueError("Artifact hash differs: " + name)
    return manifest


def protected_roots(*paths):
    roots = set()
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw).resolve()
        for parent in (path, *path.parents):
            if (parent / "SHA256_MANIFEST.json").is_file():
                roots.add(parent)
                break
    return roots


def guard_destination(path, roots):
    path = Path(path).resolve()
    for root in roots:
        root = Path(root).resolve()
        if path == root or path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError("Writable path overlaps a protected input: " + str(path))


def _numbered(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError("A different H100 runtime is already loaded; use a separate process")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def frozen_runtime(code_dir=None):
    code = Path(code_dir or Path(__file__).parent / "_vendor/terminlink_en").resolve()
    with _IMPORT_LOCK:
        integrity = verify_export(code.parent, expected_manifest_sha256=CODE_MANIFEST_SHA256)
        if code.name != "terminlink_en":
            raise ValueError("Preserve the authorized export layout terminlink_en/")
        names = ("terminlink_inputs", "terminlink_stage02", "finterm_sense2", "finterm_sense2_qwen",
                 "finterm_sense2_store", "finterm_defterm", "finterm_defterm_store",
                 "finterm_similarity", "finterm_similarity_qwen")
        for name in names:
            module = sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve() != code / (name + ".py"):
                raise ValueError("Foreign module already loaded: " + name + "; use a separate process")
        previous_path, previous_bytecode = list(sys.path), sys.dont_write_bytecode
        sys.path.insert(0, str(code))
        sys.dont_write_bytecode = True
        try:
            loaded = {name: importlib.import_module(name) for name in names}
            yield SimpleNamespace(
                stage01=_numbered("_baseline_h100_stage01", code / "01_extraer_contextos.py"),
                api=_numbered("_baseline_h100_stage02_api", code / "02_wikidata_candidatos.py"),
                stage02=loaded["terminlink_stage02"], sense2=loaded["finterm_sense2"],
                sense2_store=loaded["finterm_sense2_store"], defterm_store=loaded["finterm_defterm_store"],
                similarity=loaded["finterm_similarity"], native=loaded["finterm_similarity_qwen"],
                integrity=integrity)
        finally:
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode
            verify_export(code.parent, expected_manifest_sha256=CODE_MANIFEST_SHA256)
