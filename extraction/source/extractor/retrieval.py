"""Optional sentence-level original FastKASSIM with the trusted Java workers."""

import json
from pathlib import Path
import time

from syntactic_retrieval import AnnotatedExample, ParsedText, RetrievalConfig, SyntacticRetriever
from experimental_evaluation.common import digest, plain
from experimental_evaluation.resources import resource_inventory
from experimental_evaluation.syntax import ForestConfig, OriginalTimedSimilarity, SyntaxFailure, tree_from_data, worker_call


RESERVED_DOCUMENTS = frozenset({"CELEX:32013R0575"})


def load_financial_bank(path, *, authorized_documents, segmenter):
    """Read an explicit JSONL bank, not .key/silver or predicted user terms."""
    authorized = frozenset(authorized_documents)
    if not authorized or authorized & RESERVED_DOCUMENTS:
        raise ValueError("Authorize demonstration documents explicitly and exclude the reserved evaluation act")
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError("Provide a separately versioned JSONL demonstration bank")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples, seen, reviewed = [], set(), []
    for row in rows:
        if row["document_id"] not in authorized or row["domain"].casefold() != "finance":
            raise ValueError("Unauthorized/nonfinancial demonstration")
        if not isinstance(row["terms"], list):
            raise ValueError("Bank terms must be lists")
        example = AnnotatedExample(row["identifier"], row["text"], tuple(row["terms"]), row["document_id"], row["domain"])
        if example.identifier in seen or any(term not in example.text for term in example.terms):
            raise ValueError("Duplicate identity or nonliteral demonstration")
        if len(segmenter.segment(example.text)) != 1:
            raise ValueError("This functional sentence mode requires sentence-level demonstrations")
        seen.add(example.identifier); examples.append(example)
        reviewed.append(row.get("human_verified") is True and row.get("human_review_status") == "approved")
    if len(examples) < 3:
        raise ValueError("Provide at least three authorized sentence examples")
    status = "declared_human_reviewed_not_independently_certified" if all(reviewed) else "provisional_or_review_not_declared"
    return tuple(examples), status


class SentenceWorkerParser:
    """Analyze an already-delimited complete sentence; no second segmentation."""
    def __init__(self, resources, config):
        self.resources, self.config = resources, config
        self.parser_id = "functional_stanford_single_unit_v1:" + digest({"resources": plain(resources), "config": plain(config)})
        self.reports = {}

    def parse(self, text):
        began = time.monotonic()
        report = {"complete": False, "word_count": len(text.split()), "text_sha256": digest(text)}
        self.reports[digest(text)] = report
        try:
            value = worker_call({"operation": "parse", "text": text, "resources": plain(self.resources),
                                 "java_options": self.config.java_options}, self.config.unit_timeout_s)
            parsed = ParsedText((tree_from_data(value["tree"]),), self.parser_id)
            report.update(complete=True, tree_count=1, worker_diagnostics=value.get("_worker_diagnostics"))
            return parsed
        except Exception as exc:
            report.update(type=type(exc).__name__, message=str(exc), worker_diagnostics=getattr(exc, "diagnostics", None))
            raise SyntaxFailure(str(exc), report) from exc
        finally:
            report["seconds"] = time.monotonic() - began


def make_financial_retriever(examples, *, authorized_documents, resources, bank_status="externally_supplied",
                             config=ForestConfig()):
    authorized = frozenset(authorized_documents)
    if not authorized or authorized & RESERVED_DOCUMENTS:
        raise ValueError("The reserved evaluation document cannot be authorized as demonstrations")
    resource_inventory(resources)
    retriever = SyntacticRetriever(
        examples, SentenceWorkerParser(resources, config), OriginalTimedSimilarity(config),
        RetrievalConfig(top_k=3, authorized_document_ids=authorized, exclude_query_document=True, exclude_same_text=True),
    )
    retriever.bank_status = bank_status
    return retriever
