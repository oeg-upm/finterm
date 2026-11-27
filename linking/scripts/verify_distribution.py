"""Verify source closure, syntax, inventory and absence of private/heavy artifacts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from finterm_linking._runtime import frozen_runtime


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect():
    records, issues = [], []
    secret = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    for path in sorted(PROJECT.rglob("*"), key=lambda p: p.relative_to(PROJECT).as_posix()):
        if not path.is_file() or ".git" in path.relative_to(PROJECT).parts:
            continue
        relative = path.relative_to(PROJECT).as_posix()
        if relative == "DISTRIBUTION_MANIFEST.json":
            continue
        if path.suffix.lower() in {".pyc", ".sqlite", ".sqlite3", ".db", ".safetensors", ".bin", ".whl", ".pt", ".pth"} or any(
                part in {"__pycache__", ".venv", "venv", "build", "dist", "runs", "cache"} or part.endswith(".egg-info")
                for part in path.relative_to(PROJECT).parts):
            issues.append({"path": relative, "issue": "FORBIDDEN_DISTRIBUTION_ARTIFACT"})
        if path.stat().st_size > 5 * 1024 ** 2:
            issues.append({"path": relative, "issue": "UNEXPECTED_LARGE_FILE"})
        if path.suffix in {".py", ".md", ".json", ".toml", ".txt", ".csv", ".ipynb", ".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8-sig")
            if secret.search(text):
                issues.append({"path": relative, "issue": "POSSIBLE_SECRET_LITERAL_REDACTED"})
            if path.suffix == ".py":
                ast.parse(text, filename=relative)
            if path.suffix in {".json", ".ipynb"}:
                payload = json.loads(text)
                if path.suffix == ".ipynb":
                    for cell in payload["cells"]:
                        if cell["cell_type"] == "code":
                            ast.parse("".join(cell["source"]))
                            if cell.get("outputs") or cell.get("execution_count") is not None:
                                issues.append({"path": relative, "issue": "NOTEBOOK_HAS_EXECUTION_RESULTS"})
        records.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha(path)})
    with frozen_runtime() as runtime:
        from finterm_defterm import load_fewshots
        load_fewshots().validate()
        source = runtime.integrity
    if issues:
        raise ValueError(json.dumps(issues, ensure_ascii=False))
    return records, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    records, source = inspect()
    path = PROJECT / "DISTRIBUTION_MANIFEST.json"
    value = {"schema_version": "finterm-distribution-manifest-v1", "version": "0.1.0",
             "manifest_excludes_itself": True, "files": records}
    if args.write_manifest:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("schema_version") != value["schema_version"]:
            raise ValueError("Refusing to replace a foreign manifest")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    elif json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Distribution file list/sizes/hashes differ")
    print(json.dumps({"verified_files": len(records), "bytes": sum(r["size_bytes"] for r in records),
                      "original_source_files_verified": source["file_count"], "forbidden_artifacts": 0,
                      "models_executed": False, "full_h100_validated": False}))


if __name__ == "__main__":
    main()
