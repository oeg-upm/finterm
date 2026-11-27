"""Recover existing spans from English IOB-without-NE, without repairing labels."""

from dataclasses import dataclass, field
import re
from typing import Iterable

from syntactic_retrieval.models import AnnotatedExample

from ._io import physical_lines, read_utf8
from .discovery import ACTER_NAME, DOMAINS
from .models import (
    ActerDocument, Diagnostic, Granularity, ReadResult, ReadUnit,
    ReferenceKind, UnitTrace, acter_unit_id,
)


@dataclass
class _Block:
    lines: list[tuple[int, str]]
    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    issues: list[Diagnostic] = field(default_factory=list)


def _blocks(text: str) -> list[_Block]:
    blocks, current = [], []
    for number, line in physical_lines(text):
        if not line or (not line.strip() and "\t" not in line):
            if current:
                blocks.append(_Block(current))
                current = []
        else:
            current.append((number, line))
    if current:
        blocks.append(_Block(current))
    return blocks


def _terms(text: str, labels: list[str], positions) -> tuple[str, ...]:
    # Positions refer to fields in the supplied tokenised line, not a new NLP
    # tokenization or offsets mapped to the untokenised original.
    terms, start = [], None
    for index, label in enumerate(labels):
        if label in ("B", "O") and start is not None:
            terms.append(text[positions[start].start():positions[index - 1].end()])
            start = None
        if label == "B":
            start = index
    if start is not None:
        terms.append(text[positions[start].start():positions[-1].end()])
    return tuple(terms)


def read_acter_document(
    document: ActerDocument, *, unit_numbers: Iterable[int] | None = None,
) -> ReadResult:
    """Read one native document; return diagnostics and quarantine invalid units.

    A selector restricts returned units, never the boundary validation context.
    Both sides of a span cut across a source boundary are withheld as examples.
    """
    selected = None if unit_numbers is None else frozenset(unit_numbers)
    if selected is not None and any(type(value) is not int or value < 1 for value in selected):
        raise ValueError("ACTer unit numbers must be positive integers")
    path = document.annotation_path
    identity = ACTER_NAME.fullmatch(document.source_id)
    if (
        identity is None
        or path.parent.name != "without_named_entities"
        or path.parent.parent.name != "iob_annotations"
        or path.name != document.source_id + "_seq_terms.tsv"
    ):
        return ReadResult(issues=(Diagnostic("wrong_acter_variant", "Only native English IOB without NE is accepted", path),))
    if document.domain != DOMAINS[identity.group(1)]:
        return ReadResult(issues=(Diagnostic("wrong_acter_domain", "Domain metadata differs from the source document identity", path),))
    if document.tokenized_path is None or document.original_path is None:
        return ReadResult(issues=(Diagnostic("missing_acter_text", "Source or tokenised text counterpart is absent/ambiguous", path),))
    if any(counterpart.name != document.source_id + ".txt" for counterpart in (document.tokenized_path, document.original_path)):
        return ReadResult(issues=(Diagnostic("wrong_acter_identity", "Text filenames differ from the annotated source document", path),))
    if not document.original_path.is_file():
        return ReadResult(issues=(Diagnostic("missing_acter_text", "The original text path no longer exists", document.original_path),))
    annotation, tokenized = read_utf8(path), read_utf8(document.tokenized_path)
    sources = tuple(item.snapshot for item in (annotation, tokenized) if item.snapshot is not None)
    issues = list(annotation.issues + tokenized.issues)
    if annotation.text is None or tokenized.text is None:
        return ReadResult(issues=tuple(issues), sources=sources)
    blocks = _blocks(annotation.text)
    text_lines = [(number, line) for number, line in physical_lines(tokenized.text) if line.strip()]
    if not blocks:
        issues.append(Diagnostic("empty_acter_annotation", "The IOB source contains no annotated units", path))
    if len(blocks) != len(text_lines):
        issues.append(Diagnostic("unit_count_mismatch", "IOB block and tokenised text-line counts differ; no alignment assumed", path))

    for ordinal, block in enumerate(blocks, 1):
        uid = acter_unit_id(document.document_id, ordinal)
        previous = "O"
        for line_number, raw in block.lines:
            fields = raw.split("\t")
            if len(fields) != 2 or not fields[0] or any(char.isspace() for char in fields[0]):
                block.issues.append(Diagnostic("malformed_iob_row", "Expected one token, TAB and one label", path, uid, line_number))
                continue
            token, label = fields
            block.tokens.append(token)
            block.labels.append(label)
            if label not in {"B", "I", "O"}:
                block.issues.append(Diagnostic("unknown_iob_label", "Only original B/I/O labels are accepted", path, uid, line_number))
            if label == "I" and previous not in {"B", "I"}:
                block.issues.append(Diagnostic("orphan_iob_I", "I has no B/I predecessor inside this source unit; no repair applied", path, uid, line_number))
            previous = label

    for index in range(1, len(blocks)):
        previous, current = blocks[index - 1], blocks[index]
        if current.labels and current.labels[0] == "I" and previous.labels and previous.labels[-1] in {"B", "I"}:
            line = current.lines[0][0]
            for block_index in (index - 1, index):
                blocks[block_index].issues.append(Diagnostic(
                    "iob_cross_unit_continuation",
                    "A source unit boundary interrupts a positive span; both units withheld, not merged",
                    path, acter_unit_id(document.document_id, block_index + 1), line,
                ))

    units = []
    global_error = any(issue.severity == "error" for issue in issues)
    for index in range(max(len(blocks), len(text_lines))):
        ordinal = index + 1
        if selected is not None and ordinal not in selected:
            continue
        block = blocks[index] if index < len(blocks) else _Block([])
        text_line, text = text_lines[index] if index < len(text_lines) else (None, None)
        uid = acter_unit_id(document.document_id, ordinal)
        local = list(block.issues)
        positions = list(re.finditer(r"\S+", text)) if text is not None else []
        if tuple(match.group() for match in positions) != tuple(block.tokens):
            local.append(Diagnostic("token_text_mismatch", "Source tokens do not align literally with the supplied tokenised line", document.tokenized_path, uid, text_line))
        valid = text is not None and bool(block.lines) and not global_error and not any(issue.severity == "error" for issue in local)
        example = (
            AnnotatedExample(uid, text, _terms(text, block.labels, positions), document.document_id, document.domain)
            if valid else None
        )
        trace = UnitTrace(
            "ACTER", ReferenceKind.HUMAN, Granularity.IOB_UNIT, document.document_id,
            uid, document.source_id, ordinal, document.tokenized_path, path,
            original_text_path=document.original_path, text_line=text_line,
            annotation_start_line=block.lines[0][0] if block.lines else None,
            annotation_end_line=block.lines[-1][0] if block.lines else None,
        )
        units.append(ReadUnit(
            trace, text, example, tuple(local), tuple(raw for _, raw in block.lines),
            tuple(block.tokens), tuple(block.labels),
        ))
    if selected is not None:
        missing = selected - set(range(1, max(len(blocks), len(text_lines)) + 1))
        for ordinal in sorted(missing):
            issues.append(Diagnostic("missing_acter_unit", "Requested unit number is not present", path, acter_unit_id(document.document_id, ordinal)))
    return ReadResult(tuple(units), tuple(issues), sources)
