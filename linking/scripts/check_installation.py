"""Build/install a standalone wheel in an external scratch area, offline/no deps.

Does not install anything globally. A copy is built so build/egg-info/temp files
never pollute this distribution. Requires installed setuptools/wheel build tools.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def command(args, *, cwd, env):
    process = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(process.stdout + "\n" + process.stderr)
    return process.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work == PROJECT or work.is_relative_to(PROJECT) or PROJECT.is_relative_to(work):
        raise ValueError("Choose external scratch storage")
    if work.exists() and any(work.iterdir()):
        raise ValueError("Scratch must be new or empty")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("tmp", "wheels", "outside_checkout"):
        (work / name).mkdir()
    source = work / "source"
    shutil.copytree(PROJECT, source, ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "dist", "*.egg-info"))
    env = dict(os.environ, PIP_NO_INDEX="1", PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTHONDONTWRITEBYTECODE="1", TMP=str(work / "tmp"), TEMP=str(work / "tmp"))
    command([sys.executable, "-B", "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "--no-index",
             "--wheel-dir", str(work / "wheels")], cwd=source, env=env)
    wheels = list((work / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one wheel")
    wheel, installed = wheels[0], work / "installed"
    command([sys.executable, "-B", "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(wheel)],
            cwd=work / "outside_checkout", env=env)
    env["PYTHONPATH"] = str(installed)
    program = (
        "import json,sys; from pathlib import Path; import finterm_linking as package; "
        "from finterm_linking.cli import doctor; "
        "assert Path(package.__file__).is_relative_to(Path(sys.argv[1])); "
        "d=doctor(); assert d['imports_and_contracts_valid']; "
        "c=package.LinkerConfig(work_dir=Path(sys.argv[2])); "
        "r=package.link('An original financial statement.', ['absent supplied term'], config=c); "
        "assert r.rankings==[] and r.terms[0]['retrieval_status']=='TERM_NOT_FOUND_IN_TEXT'; "
        "assert not ({'torch','transformers','sentence_transformers','spacy','external_extractor'} & set(sys.modules)); "
        "print(json.dumps({'standalone_api':True,'original_source_files':d['source_export']['file_count'],'model_imports':False}))"
    )
    smoke = json.loads(command([sys.executable, "-B", "-c", program, str(installed), str(work / "installed_run")],
                               cwd=work / "outside_checkout", env=env))
    cli = json.loads(command([sys.executable, "-B", "-m", "finterm_linking", "doctor"],
                             cwd=work / "outside_checkout", env=env))
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        entries = archive.read(next(n for n in members if n.endswith(".dist-info/entry_points.txt"))).decode()
    if "finterm-link = finterm_linking.cli:main" not in entries:
        raise ValueError("Missing CLI entry point")
    report = {"schema_version": "finterm-installation-check-v0.1", "wheel": wheel.name,
              "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "wheel_size_bytes": wheel.stat().st_size,
              "wheel_entries": len(members), "standalone_api": smoke["standalone_api"],
              "original_source_files_verified": smoke["original_source_files"], "cli_outside_checkout": cli["imports_and_contracts_valid"],
              "cli_entry_point_present": True, "global_install_performed": False,
              "model_dependencies_installed": False, "network_used": False, "models_executed": False,
              "temporary_build_not_in_distribution": True, "full_h100_validated": False}
    with (PROJECT / "docs/INSTALLATION_CHECK.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
