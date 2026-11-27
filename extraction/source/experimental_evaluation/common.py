"""Small, model-free provenance and immutable-file primitives."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    return value


def canonical(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    """Durable, sealed JSON, atomically published WITHOUT replacing a file.

    Hard-link publication is supported by the target Linux/NTFS filesystems.
    An unsupported filesystem fails explicitly; no unsafe overwrite fallback.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical({"sha256": digest(payload), "payload": payload}))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("sha256") != digest(value["payload"]):
        raise ValueError(f"Corrupt sealed artifact: {path}")
    return value["payload"]


def outside(path, forbidden_roots):
    path = Path(path).resolve()
    if any(path == Path(root).resolve() or Path(root).resolve() in path.parents
           for root in forbidden_roots):
        raise ValueError("Output must be outside all original corpus directories")
    return path
