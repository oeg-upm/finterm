"""Immutable generation receipts, explicit locking and resumable analysis."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import uuid

from .common import atomic_json, digest, outside, read_json, utc_now


class RunStore:
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest

    @classmethod
    def create(cls, root, manifest, *, corpus_root):
        root = outside(root, [corpus_root])
        if manifest.get("schema") != "finterm-evaluation-v1":
            raise ValueError("Wrong manifest schema")
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "manifest.json", manifest)
        return cls(root, manifest)

    @classmethod
    def open(cls, root, *, corpus_root):
        root = outside(root, [corpus_root])
        manifest = read_json(root / "manifest.json")
        if manifest.get("schema") != "finterm-evaluation-v1":
            raise ValueError("Not a resumable run created by this evaluator")
        return cls(root, manifest)

    @property
    def fingerprint(self):
        return digest(self.manifest)

    def job_path(self, condition, identifier):
        if condition not in self.manifest["conditions"] or identifier not in {q["identifier"] for q in self.manifest["queries"]}:
            raise ValueError("Job outside the frozen grid")
        return self.root / "jobs" / condition / digest(identifier)

    def generation(self, condition, identifier):
        path = self.job_path(condition, identifier) / "generation.json"
        if not path.exists():
            return None
        value = read_json(path)
        request = value["request"]
        if (request["manifest_sha256"] != self.fingerprint or request["condition"] != condition
                or request["query"]["identifier"] != identifier):
            raise ValueError("Generation receipt belongs to a different frozen job")
        return value

    def failed(self, condition, identifier):
        directory = self.job_path(condition, identifier) / "errors"
        return sorted(directory.glob("*.json")) if directory.exists() else []

    def save_generation(self, condition, identifier, payload):
        atomic_json(self.job_path(condition, identifier) / "generation.json", payload)

    def save_error(self, condition, identifier, payload):
        name = utc_now().replace(":", "-") + "_" + uuid.uuid4().hex + ".json"
        atomic_json(self.job_path(condition, identifier) / "errors" / name, payload)

    @contextmanager
    def writer(self):
        path = self.root / ".writer.lock"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "time": utc_now()}).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            yield
        finally:
            path.unlink()

    def recover_lock(self, *, previous_writer_stopped, reason):
        """Only after verifying the old kernel/process is no longer running."""
        if previous_writer_stopped is not True or not reason.strip():
            raise ValueError("Explicitly confirm that the old writer stopped and record a reason")
        path = self.root / ".writer.lock"
        old = json.loads(path.read_text(encoding="utf-8"))
        atomic_json(self.root / "events" / (uuid.uuid4().hex + ".json"),
                    {"event": "stale_lock_recovery", "old_lock": old, "reason": reason, "time": utc_now()})
        path.unlink()
