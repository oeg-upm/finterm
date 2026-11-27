"""Reader provenance and diagnostics, separate from the extractor's data model."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from syntactic_retrieval.models import AnnotatedExample


class ReferenceKind(str, Enum):
    HUMAN = "human_origin"
    SILVER = "automatic_silver"


class Granularity(str, Enum):
    IOB_UNIT = "segmented_iob_unit"
    SEGMENT = "document_segment"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    path: Path | None = None
    unit_id: str | None = None
    line: int | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "message": self.message,
            "path": str(self.path) if self.path is not None else None,
            "unit_id": self.unit_id, "line": self.line, "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ActerDocument:
    source_id: str
    domain: str
    annotation_path: Path
    tokenized_path: Path | None
    original_path: Path | None

    @property
    def document_id(self) -> str:
        return f"ACTER:{self.source_id}"


@dataclass(frozen=True, slots=True)
class CelexSegment:
    source_id: str
    celex_id: str
    segment_id: str
    text_path: Path | None
    key_path: Path | None

    @property
    def document_id(self) -> str:
        return f"CELEX:{self.celex_id}"

    @property
    def identifier(self) -> str:
        return f"CELEX:{self.source_id}"


@dataclass(frozen=True, slots=True)
class CorpusCatalog:
    root: Path
    acter_documents: tuple[ActerDocument, ...]
    celex_segments: tuple[CelexSegment, ...]
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class UnitTrace:
    corpus: str
    reference_kind: ReferenceKind
    granularity: Granularity
    document_id: str
    unit_id: str
    source_id: str
    unit_number: int
    text_path: Path | None
    annotation_path: Path | None
    original_text_path: Path | None = None
    text_line: int | None = None
    annotation_start_line: int | None = None
    annotation_end_line: int | None = None
    segment_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReadUnit:
    trace: UnitTrace
    text: str | None
    example: AnnotatedExample | None
    diagnostics: tuple[Diagnostic, ...] = ()
    # Original annotation payload retained even for quarantined units.
    raw_annotation_lines: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    iob_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadResult:
    units: tuple[ReadUnit, ...] = ()
    issues: tuple[Diagnostic, ...] = ()
    sources: tuple[SourceSnapshot, ...] = ()

    @property
    def examples(self) -> tuple[AnnotatedExample, ...]:
        return tuple(unit.example for unit in self.units if unit.example is not None)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return self.issues + tuple(issue for unit in self.units for issue in unit.diagnostics)


def acter_unit_id(document_id: str, unit_number: int) -> str:
    return f"{document_id}:unit:{unit_number:06d}"
