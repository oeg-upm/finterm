"""Text-only input preparation and independent, authorized bank loading."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from corpus_readers._io import read_utf8
from syntactic_retrieval import AnnotatedExample

from .common import file_hash, plain


EVALUATION_DOCUMENT = "CELEX:32013R0575"
AUTHORIZED_DOCUMENTS = frozenset({"CELEX:32013R0549", "CELEX:32021B0417"})
EXPLORATORY = frozenset({348, 302, 201, 398, 259})
AUDIT_EXPOSURES = frozenset({0, 10, 12, 103, 107, 108, 109, 111, 121, 127, 444, 445, 446})
EXPOSURE_SOURCE = "06_auditoria_snapshot_H100_20261029.md"


def exposure(number):
    if number in EXPLORATORY:
        return "historical_exploratory_inference"
    if number in AUDIT_EXPOSURES:
        return "text_or_annotation_preview"
    return "no_individual_exposure_recorded_not_a_guarantee_of_blindness"


def freeze_inputs(corpus_root, segment_numbers, *, split):
    """Read only explicitly chosen .txt files. Never open/stat/hash any .key."""
    numbers = tuple(segment_numbers)
    if not numbers or len(numbers) != len(set(numbers)):
        raise ValueError("Provide a nonempty explicit sample without repeated IDs")
    if split not in {"exploratory", "development", "test"}:
        raise ValueError("Declare the split")
    root = Path(corpus_root).resolve()
    queries = []
    for number in numbers:
        if type(number) is not int or number < 0:
            raise ValueError("Invalid CELEX segment number")
        if split != "exploratory" and number in EXPLORATORY:
            raise ValueError("The five historical samples must remain exploratory")
        if split == "test" and number in AUDIT_EXPOSURES:
            raise ValueError("Individually exposed material is not a fresh blind test")
        stem = f"32013R0575_seg_{number}"
        relative = Path("Corpus financiero/Silver Standard/docsutf8") / (stem + ".txt")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("Input outside the declared corpus")
        loaded = read_utf8(path)
        if loaded.issues or loaded.text is None or not loaded.text.strip():
            raise ValueError(f"Unusable original input, not corrected: {path}: {loaded.issues}")
        queries.append({"identifier": "CELEX:" + stem, "document_id": EVALUATION_DOCUMENT,
                        "segment_number": number, "split": split, "text": loaded.text,
                        "text_path": relative.as_posix(), "text_sha256": loaded.snapshot.sha256,
                        "key_path": (Path("Corpus financiero/Silver Standard/gold") / (stem + ".key")).as_posix(),
                        "prior_exposure": exposure(number), "exposure_source": EXPOSURE_SOURCE})
    return queries


def read_query(query, corpus_root):
    """A capability restricted to the frozen text; no reference reader involved."""
    root = Path(corpus_root).resolve()
    path = (root / query["text_path"]).resolve()
    if root not in path.parents or path.suffix != ".txt":
        raise ValueError("Query must be an original text inside the corpus")
    loaded = read_utf8(path)
    if (loaded.issues or loaded.text != query["text"] or loaded.snapshot is None
            or loaded.snapshot.sha256 != query["text_sha256"]):
        raise ValueError("Original query changed since sample freezing")
    return loaded.text


@dataclass(frozen=True)
class Bank:
    examples: tuple[AnnotatedExample, ...]
    records: tuple[dict, ...]
    source_path: str
    sha256: str
    all_human_reviewed: bool

    def snapshot(self):
        return {"examples": [plain(e) for e in self.examples], "records": list(self.records),
                "source_path": self.source_path, "sha256": self.sha256,
                "all_human_reviewed": self.all_human_reviewed}


def reviewed(row):
    review = row.get("human_review", {})
    if not (row.get("human_verified") is True and row.get("human_review_status") == "approved"
            and review.get("reviewer_id") and review.get("guidelines_version") and review.get("reviewed_at")):
        return False
    datetime.fromisoformat(review["reviewed_at"])
    return True  # Structural evidence supplied by reviewers, not proof of expertise.


def load_bank(path, *, purpose):
    """Load existing annotations; NEVER generate/repair/approve a reference."""
    if purpose not in {"technical", "development", "confirmatory"}:
        raise ValueError("Declare the experimental purpose")
    path = Path(path).resolve()
    if path.suffix != ".jsonl":
        raise ValueError("Demonstrations must come from a separate JSONL bank")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples, ids, human = [], set(), []
    for row in rows:
        if row["document_id"] not in AUTHORIZED_DOCUMENTS or row["domain"].casefold() != "finance":
            raise ValueError("A demonstration is not from an authorized financial document")
        source = row.get("source", {}).get("candidate_id", row["identifier"])
        if not source.startswith(row["document_id"] + "_seg_"):
            raise ValueError("Demonstration provenance disagrees with document identity")
        if not isinstance(row["terms"], list):
            raise ValueError("Annotation terms must be a JSON list")
        example = AnnotatedExample(row["identifier"], row["text"], tuple(row["terms"]),
                                   row["document_id"], row["domain"])
        if example.identifier in ids or any(term not in example.text for term in example.terms):
            raise ValueError("Duplicate identity or nonliteral demonstration term")
        is_reviewed = reviewed(row)
        if is_reviewed and example.identifier.startswith("PILOT_UNVERIFIED:"):
            raise ValueError("A reviewed bank must have new identities, not relabel the provisional pilot")
        if not is_reviewed and not (row.get("human_verified") is False
                                    and row.get("human_review_status") == "pending"):
            raise ValueError("Missing/unrecognized provisional or human review provenance")
        if not is_reviewed and purpose == "confirmatory":
            raise ValueError("A provisional bank cannot support a confirmatory experiment")
        examples.append(example); ids.add(example.identifier); human.append(is_reviewed)
    if len(examples) < 3:
        raise ValueError("At least three existing authorized demonstrations are required")
    return Bank(tuple(examples), tuple(rows), str(path), file_hash(path), all(human))
