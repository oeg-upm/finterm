"""Conservative localisation of EXISTING key entries, never term prediction."""

from dataclasses import dataclass
import re
import unicodedata


DASHES = "-‐‑‒–—−"
APOSTROPHES = "'’‘"
DOUBLE_QUOTES = '\"“”'
JOINERS = set(DASHES + APOSTROPHES + "/.")
SEPARATOR = r"(?:\s+|\s*[" + re.escape(DASHES) + r"]\s*)"


def word_character(character: str) -> bool:
    return bool(character) and (
        character.isalnum() or character == "_" or unicodedata.category(character).startswith("M")
    )


def complete_boundaries(text: str, start: int, end: int) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    if start > 0:
        previous = text[start - 1]
        if word_character(previous):
            reasons.append("inside_left_word")
        elif previous in JOINERS and start > 1 and word_character(text[start - 2]):
            reasons.append("left_compound_or_abbreviation_component")
    if end < len(text):
        following = text[end]
        if word_character(following):
            reasons.append("inside_right_word")
        elif following in JOINERS and end + 1 < len(text) and word_character(text[end + 1]):
            reasons.append("right_compound_or_abbreviation_component")
    return not reasons, tuple(reasons)


def variant_pattern(term: str) -> str:
    """Allow only whitespace/dash separators and typographic quote variants.

    Do not delete arbitrary punctuation/symbols, stem, singularise, translate or
    equate e.g. €-day with day, C.I.F. with CIF, or account with accounts.
    """
    pieces = []
    index = 0
    while index < len(term):
        character = term[index]
        if character.isspace() or character in DASHES:
            while index < len(term) and (term[index].isspace() or term[index] in DASHES):
                index += 1
            pieces.append(SEPARATOR)
            continue
        if character in APOSTROPHES:
            pieces.append("[" + re.escape(APOSTROPHES) + "]")
        elif character in DOUBLE_QUOTES:
            pieces.append("[" + re.escape(DOUBLE_QUOTES) + "]")
        else:
            pieces.append(re.escape(character))
        index += 1
    return "".join(pieces)


def differences(original: str, observed: str) -> list[str]:
    if original == observed:
        return []
    flags = []
    original_words, observed_words = re.findall(r"\w+", original), re.findall(r"\w+", observed)
    if (
        [word.casefold() for word in original_words] == [word.casefold() for word in observed_words]
        and original_words != observed_words
    ):
        flags.append("case")
    if re.findall(r"\s+", original) != re.findall(r"\s+", observed):
        flags.append("whitespace")
    original_punctuation = "".join(char for char in original if not word_character(char) and not char.isspace())
    observed_punctuation = "".join(char for char in observed if not word_character(char) and not char.isspace())
    if original_punctuation != observed_punctuation:
        flags.append("punctuation")
    return flags or ["unclassified_variant"]


@dataclass(frozen=True)
class KeyEntry:
    line: int
    term: str
    literal: re.Pattern
    variant: re.Pattern

    @classmethod
    def compile(cls, line: int, term: str):
        if not isinstance(term, str) or not term.strip():
            raise ValueError("A source key entry must be a nonempty string")
        return cls(line, term, re.compile(re.escape(term)), re.compile(variant_pattern(term), re.IGNORECASE))


def locate_key_entries(
    text: str, entries: tuple[KeyEntry, ...], segment_start: int = 0,
    *, source_text: str | None = None,
) -> dict:
    if source_text is not None and source_text[segment_start:segment_start + len(text)] != text:
        raise ValueError("Sentence offsets do not refer to the original segment")
    exact_terms, variants, matches, rejected = [], [], [], []
    for entry in entries:
        occurrences, blocked, seen = [], [], set()
        for pattern in (entry.literal, entry.variant):
            for match in pattern.finditer(text):
                location = (match.start(), match.end())
                if location in seen:
                    continue
                seen.add(location)
                accepted, reasons = (
                    complete_boundaries(source_text, segment_start + location[0], segment_start + location[1])
                    if source_text is not None else complete_boundaries(text, *location)
                )
                if not accepted:
                    blocked.append({"sentence_start": location[0], "sentence_end": location[1], "reasons": list(reasons)})
                    continue
                surface = text[match.start():match.end()]
                diff = differences(entry.term, surface)
                occurrences.append({
                    "sentence_start": match.start(), "sentence_end": match.end(),
                    "segment_start": segment_start + match.start(),
                    "segment_end": segment_start + match.end(),
                    "surface": surface, "differences": diff,
                    "kind": "literal_complete_expression" if not diff else "variant_review_only",
                    "terminological_validity": "unverified",
                })
        if occurrences:
            if any(not occurrence["differences"] for occurrence in occurrences):
                exact_terms.append(entry.term)
            else:
                variants.append(entry.term)
            matches.append({
                "key_line": entry.line, "key_term": entry.term,
                "reference": "existing_segment_key_silver", "human_verified": False,
                "occurrences": sorted(occurrences, key=lambda item: (item["sentence_start"], item["sentence_end"])),
            })
        if blocked:
            rejected.append({"key_line": entry.line, "key_term": entry.term,
                             "not_associated": True, "rejected_occurrences": blocked})
    return {"silver_terms_exact": exact_terms, "silver_terms_variant_only": variants,
            "term_matches": matches, "boundary_rejections": rejected}
