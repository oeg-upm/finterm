"""Explicit offline staging/preflight of trusted resources; never downloads."""

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import zipfile

from .common import file_hash, outside, plain


PUNKT_FILES = ("abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt")
JAVA_RUNTIME_POLICY = "jdk17_standalone_nltk_child_env_v2"


@dataclass(frozen=True)
class Resources:
    java_bin: str
    parser_jar: str
    models_jar: str
    nltk_data: str
    model_path: str = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"


def private_path(path):
    """Check ownership/ancestors, not merely permissions on the final folder."""
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink in trusted path: {parent}")
        if not parent.exists():
            raise FileNotFoundError(parent)
        if os.name == "posix":
            info = parent.stat()
            if info.st_uid not in (0, os.getuid()) or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise ValueError(f"Untrusted owner/writable ancestor: {parent}")
    return path.resolve()


def java_worker_environment(resources, environ=None):
    """Build a fresh Python-worker environment using only the trusted Java.

    The adapter's `shutil.which('java')` check needs PATH, independently of
    NLTK's absolute config_java setting. Never mutate the notebook environment.
    JVM loader-variable sanitization remains exclusively NLTK's responsibility;
    this is the environment of the Python worker, not a replacement JVM env.
    """
    java = private_path(resources.java_bin)
    directory = str(java.parent)
    env = dict(os.environ if environ is None else environ)
    existing = [part for part in env.get("PATH", "").split(os.pathsep)
                if part and os.path.normcase(os.path.normpath(part)) != os.path.normcase(directory)]
    env["PATH"] = os.pathsep.join([directory, *existing])
    env["JAVA_HOME"] = str(java.parent.parent)
    return env


def jdk_layout(java_bin, *, trusted=True):
    """Require the relocatable Linux JDK layout, not a copied Conda prefix.

    Presence of these files is necessary, NOT proof of runnable dependencies;
    the NLTK-sanitized startup check below is mandatory in real preflight.
    """
    java = Path(java_bin).absolute()
    if java.name != "java" or java.parent.name != "bin":
        raise ValueError("Use the complete standalone Linux JDK 17 home: <jdk>/bin/java")
    home = java.parent.parent
    files = {"java": java, "java_release": home / "release",
             "java_libjli": home / "lib/libjli.so", "java_libjvm": home / "lib/server/libjvm.so"}
    for name, path in files.items():
        if trusted:
            files[name] = private_path(path)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Incomplete standalone JDK 17: {path}")
    release = {}
    for line in files["java_release"].read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if not re.fullmatch(r"17(?:[.+_-].*)?", release.get("JAVA_VERSION", "")):
        raise ValueError("The standalone JDK release must declare JAVA_VERSION=17.x")
    return home, files, release


def _jdk_sources(home):
    """Keep the JDK's relative layout, safely materializing internal symlinks.

    Resolve only within the approved JDK. Reject external links, cycles and
    special files BEFORE creating the new runtime. Source checksum keys remain
    canonical absolute paths, as in the existing approved_sources format.
    """
    def visit(path, relative, ancestors):
        resolved = path.resolve(strict=True)
        if resolved != home and home not in resolved.parents:
            raise ValueError(f"JDK symlink escapes the approved home: {path}")
        if path.is_dir():
            if resolved in ancestors:
                raise ValueError(f"JDK directory symlink cycle: {path}")
            for child in sorted(path.iterdir()):
                yield from visit(child, relative / child.name, ancestors | {resolved})
        elif path.is_file():
            yield resolved, Path("java17") / relative
        else:
            raise ValueError(f"Unsupported JDK entry: {path}")
    yield from visit(home, Path(), set())


def resource_inventory(resources, *, check_java=True, hash_files=True):
    paths = plain(resources)
    root = private_path(paths["nltk_data"])
    parser, models = private_path(paths["parser_jar"]), private_path(paths["models_jar"])
    home, java_files, release = jdk_layout(paths["java_bin"])
    if any(root not in path.parents for path in (parser, models)):
        raise ValueError("Stanford JARs must be inside the private NLTK trusted root")
    for path, member in ((parser, "edu/stanford/nlp/parser/lexparser/LexicalizedParser.class"),
                         (models, resources.model_path)):
        with zipfile.ZipFile(path) as archive:
            if archive.getinfo(member).file_size == 0:
                raise ValueError(f"Empty JAR member: {member}")
    files = {**java_files, "parser_jar": parser, "models_jar": models}
    for name in PUNKT_FILES:
        files["punkt/" + name] = private_path(root / "tokenizers" / "punkt_tab" / "english" / name)
    result = {"paths": paths, "files": {name: {"sha256": file_hash(path) if hash_files else None, "bytes": path.stat().st_size}
                                         for name, path in files.items()}}
    result["java_runtime"] = {"policy": JAVA_RUNTIME_POLICY, "home": str(home), "release": release}
    if check_java:
        # Same process-group deadline used by Stanford. The child delegates to
        # NLTK's own environment sanitizer AND trusted process launcher, never
        # to an inherited-env `subprocess.run(java -version)` or a custom bypass.
        from .syntax import worker_call
        probe = worker_call({"operation": "java_preflight", "resources": paths}, 15)
        if probe.get("policy") != JAVA_RUNTIME_POLICY or not probe.get("java_version"):
            raise ValueError("Missing NLTK-sanitized Java startup verification")
        result["java_preflight"] = probe
        result["java_version"] = probe["java_version"]
    return result


def stage_resources(*, java_home, parser_jar, models_jar, nltk_data, destination,
                    expected_sha256, corpus_root):
    """Rehydrate a NEW private runtime from persistent, verified local copies.

    expected_sha256 maps absolute source file names to checksums obtained when
    provisioning/approving artifacts. Every copied file must be inventoried.
    No extraction of archives, installation, download or replacement occurs.
    """
    destination = outside(destination, [corpus_root])
    private_path(destination.parent)
    sources = []
    java_home, nltk_data = Path(java_home).resolve(), Path(nltk_data).resolve()
    jdk_layout(java_home / "bin/java", trusted=False)
    sources.extend(_jdk_sources(java_home))
    for source, name in ((Path(parser_jar).resolve(), "stanford-parser.jar"),
                         (Path(models_jar).resolve(), "stanford-parser-3.5.2-models.jar")):
        sources.append((source, Path("nltk_data/stanford-parser") / name))
    for name in PUNKT_FILES:
        relative = Path("tokenizers/punkt_tab/english") / name
        sources.append((nltk_data / relative, Path("nltk_data") / relative))
    if not sources or not any(relative == Path("java17/bin/java") for _, relative in sources):
        raise ValueError("A complete local Java home with bin/java is required")
    for source, _ in sources:
        if expected_sha256.get(str(source)) != file_hash(source):
            raise ValueError(f"Source not approved or changed: {source}")
    destination.mkdir(mode=0o700, exist_ok=False)
    for source, relative in sources:
        target = destination / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        target.chmod(0o700 if os.access(source, os.X_OK) else 0o600)
        if file_hash(target) != expected_sha256[str(source)]:
            raise ValueError(f"Copy verification failed: {target}")
    return Resources(str(destination / "java17/bin/java"),
                     str(destination / "nltk_data/stanford-parser/stanford-parser.jar"),
                     str(destination / "nltk_data/stanford-parser/stanford-parser-3.5.2-models.jar"),
                     str(destination / "nltk_data"))


def check_versions():
    """Call only in H100 preflight. No torch/NLTK/model imports."""
    from packaging.version import Version
    if sys.version_info < (3, 11) or sys.platform != "linux":
        raise RuntimeError("Real execution requires Linux/Python >=3.11 on H100")
    ranges = {"torch": ("2.6", "3"), "transformers": ("4.57.1", "5"),
              "accelerate": ("1.2", "2"), "nltk": ("3.9.4", "4"),
              "numpy": ("1.26", "3"), "scipy": ("1.11", "2"), "zss": ("1.2", "2")}
    versions = {}
    for name, (lower, upper) in ranges.items():
        versions[name] = metadata.version(name)
        if not Version(lower) <= Version(versions[name]) < Version(upper):
            raise RuntimeError(f"Unsupported {name}={versions[name]}")
    dist = metadata.distribution("fkassim")
    import json
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != "a6842158ae8e958ae665d72c27b7606f93be39a6":
        raise RuntimeError("FastKASSIM Git revision is not the approved revision")
    versions["fkassim"] = dist.version
    versions["python"] = sys.version
    return versions
