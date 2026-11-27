"""Strict UTF-8, read-only I/O and physical lines; no linguistic tokenization."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .models import Diagnostic, SourceSnapshot


@dataclass(frozen=True, slots=True)
class LoadedSource:
    text: str | None
    snapshot: SourceSnapshot | None
    issues: tuple[Diagnostic, ...]


def read_utf8(path: Path) -> LoadedSource:
    path = Path(path)
    try:
        with path.open("rb") as stream:
            data = stream.read()
    except OSError as exc:
        return LoadedSource(None, None, (Diagnostic("source_unreadable", str(exc), path),))
    snapshot = SourceSnapshot(path, hashlib.sha256(data).hexdigest(), len(data))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return LoadedSource(None, snapshot, (Diagnostic("invalid_utf8", str(exc), path),))
    # Do not silently remove a BOM or replace malformed characters.
    issues = ()
    if text.startswith("\ufeff"):
        issues = (Diagnostic("utf8_bom", "BOM retained; source requires explicit review", path),)
    return LoadedSource(text, snapshot, issues)


def physical_lines(text: str) -> tuple[tuple[int, str], ...]:
    """Remove CR/LF record delimiters only, preserving all other characters.

    Unlike str.splitlines(), Unicode separators inside an annotation are not
    treated as extra file records. A final EOL does not create an empty record.
    """
    result = []
    for match in re.finditer(r"([^\r\n]*)(\r\n|\r|\n|$)", text):
        if match.start() == match.end():
            continue
        result.append((len(result) + 1, match.group(1)))
    return tuple(result)
