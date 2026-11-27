"""Read existing financial segment lists literally, explicitly as silver data."""

from syntactic_retrieval.models import AnnotatedExample

from ._io import physical_lines, read_utf8
from .discovery import CELEX_NAME
from .models import CelexSegment, Diagnostic, Granularity, ReadResult, ReadUnit, ReferenceKind, UnitTrace


def read_celex_segment(segment: CelexSegment) -> ReadResult:
    identity = CELEX_NAME.fullmatch(segment.source_id)
    if (
        identity is None or identity["celex"] != segment.celex_id
        or identity["segment"] != segment.segment_id
        or (segment.text_path is not None and segment.text_path.name != segment.source_id + ".txt")
        or (segment.key_path is not None and segment.key_path.name != segment.source_id + ".key")
    ):
        return ReadResult(issues=(Diagnostic("wrong_celex_identity", "Filename, document and segment identities do not agree", segment.key_path or segment.text_path, segment.identifier),))
    if segment.text_path is None or segment.key_path is None:
        return ReadResult(issues=(Diagnostic(
            "missing_celex_pair", "An existing .txt/.key counterpart is missing; no annotation generated",
            segment.text_path or segment.key_path, segment.identifier,
        ),))
    text_source, key_source = read_utf8(segment.text_path), read_utf8(segment.key_path)
    sources = tuple(item.snapshot for item in (text_source, key_source) if item.snapshot is not None)
    issues = list(text_source.issues + key_source.issues)
    if text_source.text is None or key_source.text is None:
        return ReadResult(issues=tuple(issues), sources=sources)
    text = text_source.text
    rows = physical_lines(key_source.text)
    terms = tuple(line for _, line in rows)
    local = []
    if not text.strip():
        local.append(Diagnostic("empty_celex_text", "The stored segment text is empty", segment.text_path, segment.identifier))
    if not rows:
        local.append(Diagnostic("empty_key_reference", "The existing key has zero entries; no missing reference was synthesized", segment.key_path, segment.identifier, severity="warning"))
    seen = set()
    for line_number, term in rows:
        if not term.strip():
            local.append(Diagnostic("empty_key_entry", "An empty/whitespace annotation cannot be represented; segment withheld", segment.key_path, segment.identifier, line_number))
            continue
        if term in seen:
            local.append(Diagnostic("duplicate_key_entry", "Repeated source entry retained, not deduplicated", segment.key_path, segment.identifier, line_number, "warning"))
        seen.add(term)
        if term not in text:
            local.append(Diagnostic("nonliteral_key_entry", "Stored annotation is not a literal substring of this segment; entry retained", segment.key_path, segment.identifier, line_number, "warning"))
    valid = not any(issue.severity == "error" for issue in issues + local)
    example = AnnotatedExample(segment.identifier, text, terms, segment.document_id, "finance") if valid else None
    trace = UnitTrace(
        "CELEX", ReferenceKind.SILVER, Granularity.SEGMENT, segment.document_id,
        segment.identifier, segment.source_id, int(segment.segment_id),
        segment.text_path, segment.key_path, original_text_path=segment.text_path,
        annotation_start_line=1 if rows else None,
        annotation_end_line=rows[-1][0] if rows else None, segment_id=segment.segment_id,
    )
    unit = ReadUnit(trace, text, example, tuple(local), terms)
    return ReadResult((unit,), tuple(issues), sources)
