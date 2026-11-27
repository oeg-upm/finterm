"""External read-only corpus adapters; no model, retrieval or extractor execution."""

from .acter import read_acter_document
from .banks import BankAuthorization, BankAuthorizationError, ExampleBank, build_example_bank, read_authorized_bank
from .celex import read_celex_segment
from .discovery import discover_corpora
from .models import (
    ActerDocument, CelexSegment, CorpusCatalog, Diagnostic, Granularity,
    ReadResult, ReadUnit, ReferenceKind, SourceSnapshot, UnitTrace, acter_unit_id,
)

__all__ = [
    "ActerDocument", "BankAuthorization", "BankAuthorizationError", "CelexSegment",
    "CorpusCatalog", "Diagnostic", "ExampleBank", "Granularity", "ReadResult",
    "ReadUnit", "ReferenceKind", "SourceSnapshot", "UnitTrace", "acter_unit_id",
    "build_example_bank", "discover_corpora", "read_acter_document",
    "read_authorized_bank", "read_celex_segment",
]
