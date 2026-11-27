"""Explicitly authorized demonstration banks; no automatic experimental split."""

from dataclasses import dataclass
from typing import Iterable

from syntactic_retrieval.models import AnnotatedExample, identifier_set

from .acter import read_acter_document
from .celex import read_celex_segment
from .discovery import CELEX_NAME
from .models import CorpusCatalog, Diagnostic, ReadResult, ReadUnit, ReferenceKind, SourceSnapshot


class BankAuthorizationError(ValueError):
    """Caller-supplied demonstration/evaluation permissions conflict."""


def _example_document(identifier: str) -> str | None:
    if identifier.startswith("ACTER:") and ":unit:" in identifier:
        document_id, number = identifier.rsplit(":unit:", 1)
        if number.isdecimal() and int(number) > 0:
            return document_id
    if identifier.startswith("CELEX:"):
        match = CELEX_NAME.fullmatch(identifier.removeprefix("CELEX:"))
        if match:
            return "CELEX:" + match["celex"]
    return None


@dataclass(frozen=True, slots=True)
class BankAuthorization:
    authorized_document_ids: frozenset[str]
    evaluation_document_ids: frozenset[str] = frozenset()
    authorized_example_ids: frozenset[str] | None = None
    evaluation_example_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("authorized_document_ids", "evaluation_document_ids", "evaluation_example_ids"):
            object.__setattr__(self, name, identifier_set(getattr(self, name), name))
        if self.authorized_example_ids is not None:
            object.__setattr__(self, "authorized_example_ids", identifier_set(self.authorized_example_ids, "authorized_example_ids"))
        if self.authorized_document_ids & self.evaluation_document_ids:
            raise BankAuthorizationError("A document cannot be authorized for demonstrations and reserved for evaluation")
        for identifier in self.evaluation_example_ids:
            if _example_document(identifier) in self.authorized_document_ids:
                raise BankAuthorizationError("An evaluation example belongs to an authorized document; separate whole documents explicitly")
        for identifier in self.authorized_example_ids or ():
            document = _example_document(identifier)
            if document is not None and document not in self.authorized_document_ids:
                raise BankAuthorizationError("An authorized example's document must also be authorized")
            if identifier in self.evaluation_example_ids:
                raise BankAuthorizationError("An example cannot be both demonstration and evaluation")


@dataclass(frozen=True, slots=True)
class ExampleBank:
    examples: tuple[AnnotatedExample, ...]
    records: tuple[ReadUnit, ...]
    diagnostics: tuple[Diagnostic, ...]
    sources: tuple[SourceSnapshot, ...]
    authorization: BankAuthorization

    @property
    def authorized_document_ids(self) -> frozenset[str]:
        return self.authorization.authorized_document_ids

    @property
    def reference_kinds(self) -> frozenset[ReferenceKind]:
        return frozenset(record.trace.reference_kind for record in self.records)


def build_example_bank(readings: Iterable[ReadResult], authorization: BankAuthorization) -> ExampleBank:
    """Select already read records using permissions, without ranking or relabeling."""
    readings = tuple(readings)
    records, issues, sources, seen = [], [], {}, set()
    known_examples = set()
    for reading in readings:
        issues.extend(reading.issues)
        for source in reading.sources:
            sources[source.path] = source
        for record in reading.units:
            trace = record.trace
            known_examples.add(trace.unit_id)
            if trace.unit_id in authorization.evaluation_example_ids and trace.document_id in authorization.authorized_document_ids:
                raise BankAuthorizationError("An evaluation unit shares an authorized demonstration document")
            if trace.document_id not in authorization.authorized_document_ids or trace.document_id in authorization.evaluation_document_ids:
                continue
            if trace.unit_id in authorization.evaluation_example_ids:
                continue
            if authorization.authorized_example_ids is not None and trace.unit_id not in authorization.authorized_example_ids:
                continue
            issues.extend(record.diagnostics)
            if record.example is None or any(issue.severity == "error" for issue in reading.issues + record.diagnostics):
                issues.append(Diagnostic("unit_withheld", "Requested unit is not exported as a complete demonstration; inspect its source diagnostics", trace.annotation_path, trace.unit_id))
                continue
            if record.example.identifier != trace.unit_id or record.example.document_id != trace.document_id:
                raise ValueError("Example identity differs from its source trace")
            if trace.unit_id in seen:
                raise ValueError("Duplicate example identity in the supplied readings")
            seen.add(trace.unit_id)
            records.append(record)
    for identifier in sorted((authorization.authorized_example_ids or frozenset()) - known_examples):
        issues.append(Diagnostic("requested_example_not_found", "Requested source example was not found", unit_id=identifier))
    return ExampleBank(tuple(record.example for record in records), tuple(records), tuple(issues), tuple(sources.values()), authorization)


def read_authorized_bank(catalog: CorpusCatalog, authorization: BankAuthorization) -> ExampleBank:
    """Authorize descriptors BEFORE content I/O. Never choose train/test partitions."""
    readings = []
    known_documents = {item.document_id for item in catalog.acter_documents + catalog.celex_segments}
    for document in catalog.acter_documents:
        if document.document_id not in authorization.authorized_document_ids:
            continue
        numbers = None
        if authorization.authorized_example_ids is not None:
            prefix = document.document_id + ":unit:"
            numbers = {
                int(identifier[len(prefix):]) for identifier in authorization.authorized_example_ids
                if identifier.startswith(prefix) and identifier[len(prefix):].isdecimal()
            }
            if not numbers:
                continue
        readings.append(read_acter_document(document, unit_numbers=numbers))
    for segment in catalog.celex_segments:
        if segment.document_id not in authorization.authorized_document_ids:
            continue
        if authorization.authorized_example_ids is not None and segment.identifier not in authorization.authorized_example_ids:
            continue
        readings.append(read_celex_segment(segment))
    bank = build_example_bank(readings, authorization)
    missing = authorization.authorized_document_ids - known_documents
    issues = catalog.diagnostics + bank.diagnostics + tuple(
        Diagnostic("authorized_document_not_found", f"Authorized document is not in this catalog: {identifier}")
        for identifier in sorted(missing)
    )
    return ExampleBank(bank.examples, bank.records, issues, bank.sources, authorization)
