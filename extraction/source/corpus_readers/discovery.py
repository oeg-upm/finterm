"""Discover native annotation variants and matching filenames, without contents."""

from collections import defaultdict
from pathlib import Path
import re

from .models import ActerDocument, CelexSegment, CorpusCatalog, Diagnostic


ACTER_NAME = re.compile(r"(corp|equi|htfl|wind)_en_\d+")
CELEX_NAME = re.compile(r"(?P<celex>\d{5}[A-Z]\d{4})_seg_(?P<segment>\d+)")
DOMAINS = {"corp": "corruption", "equi": "equitation/dressage", "htfl": "heart failure", "wind": "wind energy"}


def discover_corpora(root: str | Path) -> CorpusCatalog:
    """Inspect filenames/semantic variant folders under an explicitly given root.

    No dependency on the name/depth of an ACTER or financial top-level folder.
    Ambiguous identities are diagnosed rather than paired across corpus copies.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    issues = []
    annotations = defaultdict(list)
    for path in sorted(root.rglob("*_seq_terms.tsv")):
        if not path.is_file():
            continue
        if path.parent.name != "without_named_entities" or path.parent.parent.name != "iob_annotations":
            continue
        source_id = path.name.removesuffix("_seq_terms.tsv")
        match = ACTER_NAME.fullmatch(source_id)
        if match:
            annotations[source_id].append(path)

    acter = []
    text_indexes = {}
    for source_id, candidates in sorted(annotations.items()):
        if len(candidates) != 1:
            issues.append(Diagnostic("duplicate_acter_document", "Ambiguous copies of the same English IOB document", candidates[0]))
            continue
        path = candidates[0]
        domain_code = ACTER_NAME.fullmatch(source_id).group(1)
        domain_root = next((parent for parent in path.parents if parent.name == domain_code and parent.parent.name == "en"), None)
        if domain_root is None:
            issues.append(Diagnostic("unknown_acter_layout", "Cannot establish the English domain's source identity", path))
            continue
        if domain_root not in text_indexes:
            index = defaultdict(list)
            for text_path in domain_root.rglob("*.txt"):
                if text_path.is_file() and text_path.parent.name in ("texts", "texts_tokenised"):
                    index[(text_path.stem, text_path.parent.name)].append(text_path)
            text_indexes[domain_root] = index
        index = text_indexes[domain_root]
        resolved = {}
        for kind in ("texts", "texts_tokenised"):
            matches = index.get((source_id, kind), [])
            resolved[kind] = matches[0] if len(matches) == 1 else None
            if len(matches) != 1:
                issues.append(Diagnostic("acter_text_counterpart", f"Expected one {kind} counterpart; found {len(matches)}", path))
        acter.append(ActerDocument(source_id, DOMAINS[domain_code], path, resolved["texts_tokenised"], resolved["texts"]))

    texts, keys = defaultdict(list), defaultdict(list)
    for suffix, index in (("*.txt", texts), ("*.key", keys)):
        for path in root.rglob(suffix):
            if path.is_file() and CELEX_NAME.fullmatch(path.stem):
                index[path.stem].append(path)
    celex = []
    for source_id in sorted(set(texts) | set(keys)):
        text_paths, key_paths = texts.get(source_id, []), keys.get(source_id, [])
        if len(text_paths) > 1 or len(key_paths) > 1:
            location = (text_paths or key_paths)[0]
            issues.append(Diagnostic("duplicate_celex_segment", "Ambiguous copies of a CELEX segment; no pair chosen", location))
            continue
        text_path = text_paths[0] if text_paths else None
        key_path = key_paths[0] if key_paths else None
        if text_path is None or key_path is None:
            issues.append(Diagnostic("missing_celex_pair", "Existing segment has no text or key counterpart", text_path or key_path))
        match = CELEX_NAME.fullmatch(source_id)
        celex.append(CelexSegment(source_id, match["celex"], match["segment"], text_path, key_path))
    return CorpusCatalog(root, tuple(acter), tuple(celex), tuple(issues))
