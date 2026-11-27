"""Isolated worker. No corpus paths/keys, prompts, examples' terms or LLMs."""

import base64
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

from syntactic_retrieval import FastKassimConfig, FastKassimSimilarity, ParsedText
from syntactic_retrieval import StanfordPCFGConfig, StanfordPCFGParser

from .resources import JAVA_RUNTIME_POLICY, Resources, java_worker_environment, resource_inventory
from .syntax import tree_data, tree_from_data


def configure_java(resources):
    """Common trusted configuration for preflight AND Stanford in this child."""
    if any(os.environ.get(name) == "1" for name in
           ("NLTK_ALLOW_UNSAFE_JARS", "NLTK_ALLOW_UNSAFE_CLASSPATH")):
        raise ValueError("Remove NLTK trust-bypass flags; this runtime never permits unsafe classpaths")
    import nltk.data
    from nltk.internals import config_java
    env = java_worker_environment(resources)
    os.environ["PATH"] = env["PATH"]
    os.environ["JAVA_HOME"] = env["JAVA_HOME"]
    os.environ["NLTK_DATA"] = resources.nltk_data
    nltk.data.path[:] = [resources.nltk_data]
    config_java(bin=resources.java_bin, options=[])
    return {"configured_java": resources.java_bin, "java_home": env["JAVA_HOME"],
            "path_first": env["PATH"].split(os.pathsep)[0],
            "which_java": shutil.which("java")}


def stream_evidence(value):
    """Keep complete text and, for native bytes, their lossless representation."""
    if isinstance(value, bytes):
        return {"text": value.decode("utf-8", errors="replace"),
                "base64": base64.b64encode(value).decode("ascii")}
    return {"text": value if isinstance(value, str) else str(value)}


def exception_evidence(exc):
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        item = {"type": type(current).__name__, "module": type(current).__module__,
                "message": str(current)}
        for name in ("stdout", "stderr", "output"):
            value = getattr(current, name, None)
            if value is not None:
                item[name] = stream_evidence(value)
        code = getattr(current, "returncode", None)
        if isinstance(code, int):
            item["returncode"] = code
        chain.append(item)
        current = current.__cause__ if current.__cause__ is not None else (
            None if current.__suppress_context__ else current.__context__)
    return {"chain": chain, "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


@contextmanager
def observe_stanford_java(diagnostics):
    """Observe the child-local Stanford alias, NEVER replace security checks.

    Delegate all arguments to the original protected NLTK Java launcher and
    return its result unchanged. This also preserves stderr on exit 0, which
    NLTK's parser normally discards before it reports an empty analysis.
    """
    module = importlib.import_module("nltk.parse.stanford")
    original = module.java
    calls = diagnostics.setdefault("java_calls", [])
    def observed(*args, **kwargs):
        item = {"command": list(args[0]) if args and isinstance(args[0], (list, tuple)) else None,
                "classpath": list(kwargs["classpath"]) if isinstance(kwargs.get("classpath"), (tuple, list)) else kwargs.get("classpath"),
                "options": list(kwargs["options"]) if isinstance(kwargs.get("options"), (tuple, list)) else kwargs.get("options")}
        calls.append(item)
        began = time.monotonic()
        try:
            value = original(*args, **kwargs)
            item["status"] = "returned"
            if isinstance(value, tuple) and len(value) == 2:
                item.update(stdout=stream_evidence(value[0]), stderr=stream_evidence(value[1]),
                            returncode=0, returncode_evidence="successful return of NLTK java()")
            return value
        except Exception as exc:
            item.update(status="error", exception=exception_evidence(exc))
            raise
        finally:
            item["seconds"] = time.monotonic() - began
    module.java = observed
    try:
        yield
    finally:
        module.java = original


def java_startup_probe(resources):
    """Use the very same sanitizer/trusted spawn as installed NLTK java().

    NLTK 3.10.4 java(cmd) requires a main class, so `java(['-version'])`
    is deliberately NOT used, and no trusted_raw_options escape hatch is used.
    """
    import nltk
    import nltk.internals as internals
    import nltk.pathsec as pathsec
    sanitizer = getattr(internals, "_java_child_env", None)
    launcher = getattr(pathsec, "spawn_trusted", None)
    if not callable(sanitizer) or not callable(launcher):
        raise RuntimeError("NLTK's protected Java environment/launcher API is unavailable; use the approved NLTK 3.10.4 environment")
    env = sanitizer()
    if any(name.upper() in {"LD_LIBRARY_PATH", "LD_PRELOAD", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS"} for name in env):
        raise RuntimeError("NLTK did not sanitize the Java child environment; no fallback is permitted")
    process = launcher(resources.java_bin, ["-version"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, env=env)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else (stdout or "")
    stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else (stderr or "")
    if process.returncode != 0:
        raise RuntimeError(f"Java cannot start under NLTK's sanitized environment (exit {process.returncode}): {stderr or stdout}")
    version = stderr or stdout
    if not re.search(r'\b(?:openjdk|java) version "17(?:[.+_-][^\"]*)?"', version):
        raise RuntimeError(f"The executed runtime is not the required JDK 17: {version}")
    return {"policy": JAVA_RUNTIME_POLICY, "nltk_version": nltk.__version__,
            "environment_sanitizer": "nltk.internals._java_child_env",
            "trusted_launcher": "nltk.pathsec.spawn_trusted", "loader_overrides_forwarded": False,
            "java_version": version, "stdout": stdout, "stderr": stderr}


def execute(payload, *, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["operation"] = payload["operation"]
    if payload["operation"] == "score":
        parser_id = payload["parser_id"]
        query = ParsedText(tuple(tree_from_data(t) for t in payload["query"]), parser_id)
        example = ParsedText(tuple(tree_from_data(t) for t in payload["example"]), parser_id)
        # The original kernel, normalization and assignment are reused unchanged.
        return FastKassimSimilarity(FastKassimConfig()).score(query, example)
    if payload["operation"] not in {"parse", "java_preflight"}:
        raise ValueError("Unknown worker operation")
    resources = Resources(**payload["resources"])
    # Content hashes are frozen/checked once by the parent preflight. Here keep
    # the trust/JAR checks without rehashing the large models JAR for every unit.
    resource_inventory(resources, check_java=False, hash_files=False)
    diagnostics["java_configuration"] = configure_java(resources)
    if payload["operation"] == "java_preflight":
        return java_startup_probe(resources)
    parser = StanfordPCFGParser(StanfordPCFGConfig(
        parser_jar=Path(resources.parser_jar), models_jar=Path(resources.models_jar),
        model_path=resources.model_path, sentence_segmentation="single_unit",
        java_options=payload["java_options"],
    ))
    with observe_stanford_java(diagnostics):
        parsed = parser.parse(payload["text"])
    if len(parsed.trees) != 1:
        raise ValueError("Stanford did not return exactly one tree for the full unit")
    return {"tree": tree_data(parsed.trees[0])}


class CapturedText(io.StringIO):
    """Keep Python logs out of JSON stdout; mirror them for timeout diagnosis."""
    def __init__(self, mirror):
        super().__init__()
        self.mirror = mirror

    def write(self, text):
        count = super().write(text)
        self.mirror.write(text)
        self.mirror.flush()
        return count


def worker_response(payload):
    diagnostics = {}
    stdout, stderr = CapturedText(sys.stderr), CapturedText(sys.stderr)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            value = execute(payload, diagnostics=diagnostics)
            response = {"ok": True, "value": value}
        except Exception as exc:
            response = {"ok": False, "type": type(exc).__name__, "message": str(exc),
                        "error": exception_evidence(exc)}
    diagnostics.update(python_stdout=stdout.getvalue(), python_stderr=stderr.getvalue())
    response["diagnostics"] = diagnostics
    return response


if __name__ == "__main__":
    print(json.dumps(worker_response(json.load(sys.stdin))), flush=True)
