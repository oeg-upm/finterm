"""Experimental segment_forest_v1, with original parsing/kernel implementations."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from syntactic_retrieval import (
    ParsedText, RetrievalConfig, SyntacticRetriever, SyntaxTree,
)

from .common import digest, plain
from .resources import Resources, java_worker_environment


@dataclass(frozen=True)
class ForestConfig:
    strategy: str = "segment_forest_v1"
    unit_timeout_s: float = 60
    segment_timeout_s: float = 240
    score_timeout_s: float = 60
    java_options: str = "-mx2g"

    def __post_init__(self):
        if self.strategy != "segment_forest_v1":
            raise ValueError("This is a separate, explicitly versioned forest strategy")
        for value in (self.unit_timeout_s, self.segment_timeout_s, self.score_timeout_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be finite and positive")


class SyntaxFailure(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class WorkerFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def tree_data(tree):
    return {"label": tree.label, "children": [tree_data(child) for child in tree.children]}


def tree_from_data(data):
    return SyntaxTree(data["label"], tuple(tree_from_data(child) for child in data["children"]))


def worker_call(payload, timeout):
    """Hard deadline for Java preflight/Stanford/LTK and all Java descendants."""
    if sys.platform != "linux":
        raise RuntimeError("Real Java/Stanford/LTK workers are restricted to Linux/H100")
    deadline = time.monotonic() + timeout
    serialized = json.dumps(payload)
    if time.monotonic() >= deadline:
        raise TimeoutError("Syntax request serialization exceeded the deadline")
    env = None
    if payload.get("resources") is not None:
        env = java_worker_environment(Resources(**payload["resources"]))
    process = subprocess.Popen(
        [sys.executable, "-B", "-m", "experimental_evaluation.syntax_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", start_new_session=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
    try:
        stdout, stderr = process.communicate(serialized, timeout=max(0.001, deadline - time.monotonic()))
    except BaseException as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        # Retain the original exception type (especially TimeoutExpired) while
        # keeping all output available after killing/draining the process group.
        exc.diagnostics = {"status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted",
                           "operation": payload.get("operation"), "timeout_s": timeout,
                           "worker_returncode": process.returncode if isinstance(process.returncode, int) else None,
                           "worker_stdout": stdout, "worker_stderr": stderr,
                           "java_output_complete": False}
        if isinstance(exc, subprocess.TimeoutExpired):
            exc.output, exc.stderr = stdout, stderr
        raise
    diagnostics = {"operation": payload.get("operation"), "worker_returncode": process.returncode,
                   "worker_stdout": stdout, "worker_stderr": stderr}
    if process.returncode:
        raise WorkerFailure(f"Syntax worker exited {process.returncode}: {stderr}", diagnostics)
    try:
        response = json.loads(stdout)
    except (ValueError, TypeError) as exc:
        raise WorkerFailure("Invalid syntax worker JSON protocol", diagnostics) from exc
    if (not isinstance(response, dict) or not isinstance(response.get("ok"), bool)
            or (response["ok"] and "value" not in response)):
        raise WorkerFailure("Invalid syntax worker response schema", diagnostics)
    if not response.get("ok"):
        diagnostics.update(worker=response.get("diagnostics", {}), error=response.get("error"))
        raise WorkerFailure(f"{response.get('type')}: {response.get('message')}; {stderr}", diagnostics)
    value = response["value"]
    if payload.get("operation") == "parse" and isinstance(value, dict):
        value = {**value, "_worker_diagnostics": {**diagnostics, "worker": response.get("diagnostics", {})}}
    return value


class ForestParser:
    def __init__(self, resources, config=ForestConfig(), *, segmenter=None, call=worker_call):
        self.resources, self.config, self.call = resources, config, call
        self.segmenter = segmenter
        self.parser_id = "segment_forest_v1:" + digest({"resources": plain(resources), "config": plain(config)})
        self.reports = {}

    def _spans(self, text):
        if self.segmenter is not None:
            return list(self.segmenter(text))
        import nltk.data
        os.environ["NLTK_DATA"] = self.resources.nltk_data
        nltk.data.path[:] = [self.resources.nltk_data]
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        return list(tokenizer.span_tokenize(text))

    def parse(self, text):
        start = time.monotonic()
        report = {"strategy": self.config.strategy, "text_sha256": digest(text),
                  "units": [], "complete": False}
        self.reports[digest(text)] = report
        try:
            spans = self._spans(text)
            previous = 0
            if not spans:
                raise ValueError("Punkt produced no units")
            trees = []
            for number, (left, right) in enumerate(spans, 1):
                if not 0 <= previous <= left < right <= len(text) or text[previous:left].strip():
                    raise ValueError("Segmentation omitted/overlapped non-whitespace text")
                previous = right
                unit = text[left:right]
                detail = {"number": number, "start": left, "end": right,
                          "word_count": len(unit.split()), "long_unit": len(unit.split()) > 100}
                report["units"].append(detail)
                remaining = self.config.segment_timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Complete segment parsing deadline exceeded")
                began = time.monotonic()
                try:
                    value = self.call({"operation": "parse", "text": unit,
                                       "resources": plain(self.resources),
                                       "java_options": self.config.java_options},
                                      min(remaining, self.config.unit_timeout_s))
                    tree = tree_from_data(value["tree"])
                    detail["status"] = "parsed"
                    if "_worker_diagnostics" in value:
                        detail["worker_diagnostics"] = value["_worker_diagnostics"]
                    trees.append(tree)
                finally:
                    detail["seconds"] = time.monotonic() - began
            if text[previous:].strip():
                raise ValueError("Segmentation omitted the end of the original text")
            if time.monotonic() - start > self.config.segment_timeout_s:
                raise TimeoutError("Complete segment parsing deadline exceeded")
            report.update(complete=True, tree_count=len(trees))
            return ParsedText(tuple(trees), self.parser_id)
        except Exception as exc:
            report.update(error_type=type(exc).__name__, error=str(exc))
            if getattr(exc, "diagnostics", None) is not None:
                report["worker_diagnostics"] = exc.diagnostics
                if report["units"]:
                    report["units"][-1]["worker_diagnostics"] = exc.diagnostics
            if report["units"] and "status" not in report["units"][-1]:
                report["units"][-1]["status"] = "failed"
            raise SyntaxFailure(str(exc), report) from exc
        finally:
            report["seconds"] = time.monotonic() - start


class OriginalTimedSimilarity:
    def __init__(self, config=ForestConfig(), *, call=worker_call):
        self.config, self.call = config, call
        self.reports = []

    def score(self, query, example):
        if query.parser_id != example.parser_id:
            raise ValueError("Parser identity mismatch")
        began = time.monotonic()
        report = {"query_trees": len(query.trees), "example_trees": len(example.trees)}
        self.reports.append(report)
        try:
            value = self.call({"operation": "score", "parser_id": query.parser_id,
                               "query": [tree_data(t) for t in query.trees],
                               "example": [tree_data(t) for t in example.trees]},
                              self.config.score_timeout_s)
            report.update(status="scored", score=value)
            return value
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            if getattr(exc, "diagnostics", None) is not None:
                report["worker_diagnostics"] = exc.diagnostics
            raise SyntaxFailure(str(exc), report) from exc
        finally:
            report["seconds"] = time.monotonic() - began


def make_retriever(examples, parser, similarity):
    return SyntacticRetriever(
        examples, parser, similarity,
        RetrievalConfig(top_k=3, authorized_document_ids=frozenset({"CELEX:32013R0549", "CELEX:32021B0417"}),
                        exclude_query_document=True, exclude_same_text=True),
    )
