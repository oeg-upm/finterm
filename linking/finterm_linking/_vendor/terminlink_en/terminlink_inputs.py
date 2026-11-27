"""Entity-linker input contracts: user documents and supplied terms. Standard library only."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CONTRACT_VERSION = "finterm-linking-input-v1"


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def decode_text(payload: bytes) -> str:
    with io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline=None) as stream:
        return stream.read()


def validate_document_id(value: str) -> None:
    if not value or value in {".", ".."} or any(char in value for char in '/\\:*?"<>|\x00\n\r'):
        raise ValueError("document_id must be a nonempty portable filename stem")


@dataclass(frozen=True)
class DocumentInput:
    document_id: str
    text: str
    language: str
    source_kind: str
    source_uri: str | None
    source_sha256: str

    def __post_init__(self):
        validate_document_id(self.document_id)
        if not self.language:
            raise ValueError("Language label cannot be empty; use und when unspecified")

    @classmethod
    def from_file(cls, path: Path, document_id: str | None = None, language: str = "und"):
        path = path.resolve()
        payload = path.read_bytes()
        text = decode_text(payload)
        identifier = document_id or path.stem + "_" + digest(text.encode("utf-8"))[:16].lower()
        return cls(identifier, text, language, "file", str(path), digest(payload))

    @classmethod
    def from_text(cls, text: str, document_id: str | None = None, language: str = "und"):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        payload = text.encode("utf-8")
        decoded = decode_text(payload)
        identifier = document_id or "text_" + digest(decoded.encode("utf-8"))[:24].lower()
        # Preserve the supplied str for original offsets. The canonicalized
        # identity formula stays unchanged for compatibility with existing IDs.
        return cls(identifier, text, language, "inline", None, digest(payload))

    def to_dict(self) -> dict:
        return {"contract_version": CONTRACT_VERSION, "document_id": self.document_id, "text": self.text,
                "language": self.language, "source_kind": self.source_kind, "source_uri": self.source_uri,
                "source_sha256": self.source_sha256, "decoded_text_sha256": digest(self.text.encode("utf-8"))}


@dataclass(frozen=True)
class TermRecord:
    document_id: str
    term: str
    language: str
    provenance: tuple[dict, ...]

    def __post_init__(self):
        validate_document_id(self.document_id)
        if not self.term or self.term != self.term.strip() or "\n" in self.term or "\r" in self.term:
            raise ValueError("A term must be a nonempty single-line expression")
        if not self.provenance or any(not item.get("provider") for item in self.provenance):
            raise ValueError("Term provenance requires an explicit provider")

    @property
    def term_id(self) -> str:
        return "term_" + digest(canonical([self.document_id, self.term]).encode("utf-8")).lower()

    def to_dict(self) -> dict:
        return {"contract_version": CONTRACT_VERSION, "document_id": self.document_id, "term_id": self.term_id,
                "term": self.term, "language": self.language, "provenance": list(self.provenance)}


def supplied_terms(
    document: DocumentInput, *, path: Path | None = None, terms: Sequence[str] | None = None,
    origin: str = "user", legacy: bool = False,
) -> tuple[list[TermRecord], dict]:
    if (path is None) == (terms is None):
        raise ValueError("Supply exactly one term file or term list")
    if not origin or "\n" in origin or "\r" in origin:
        raise ValueError("Term origin must be an explicit single-line provenance label")
    if path is not None:
        path = path.resolve()
        payload = path.read_bytes()
        lines = decode_text(payload).splitlines()
        provider = "legacy_key" if legacy else origin + "_file"
        source = {"provider": provider, "source_kind": "file", "source_uri": str(path), "source_sha256": digest(payload)}
    else:
        if isinstance(terms, (str, bytes)):
            raise ValueError("terms must be a list of expressions, not one string")
        lines = list(terms)
        if any(not isinstance(term, str) for term in lines):
            raise ValueError("Every supplied term must be a string")
        if any(not term or term != term.strip() or "\n" in term or "\r" in term for term in lines):
            raise ValueError("Inline terms must be nonempty single-line expressions without surrounding whitespace; terms are not silently trimmed")
        source = {"provider": origin + "_list", "source_kind": "inline", "source_uri": None,
                  "source_sha256": digest(canonical(lines).encode("utf-8"))}
        provider = source["provider"]
    by_term: dict[str, list[dict]] = {}
    counts: dict[str, int] = {}
    for index, raw in enumerate(lines, 1):
        term = raw.strip()
        if not term:
            continue
        if "\n" in term or "\r" in term:
            raise ValueError("Inline terms must each contain a single expression")
        by_term.setdefault(term, []).append({**source, "record_number": index})
        counts[term] = counts.get(term, 0) + 1
    records = [TermRecord(document.document_id, term, document.language, tuple(provenance))
               for term, provenance in by_term.items()]
    return records, {**source, "supplied_terms_count": sum(counts.values()), "unique_terms_count": len(records),
                     "duplicate_terms": {term: count for term, count in counts.items() if count > 1}}
