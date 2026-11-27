"""Exact normalized term TYPES per segment, against automatic silver only."""

import re

from demonstrations.matching import complete_boundaries


NORMALIZATION = "casefold_whitespace_exact_v1"


def normalize(term):
    if not isinstance(term, str) or not term.strip():
        raise ValueError("A term must be a nonempty string")
    return " ".join(term.casefold().split())


def contained(short, long):
    return any(complete_boundaries(long, match.start(), match.end())[0]
               for match in re.finditer(re.escape(short), long))


def nested_pairs(terms):
    return [[short, long] for short in sorted(terms) for long in sorted(terms)
            if len(short) < len(long) and contained(short, long)]


def prf(tp, fp, fn):
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def term_statistics(raw):
    unique = {normalize(term) for term in raw}
    nested = nested_pairs(unique)
    return {"raw_count": len(raw), "normalized_unique_count": len(unique),
            "normalization_collisions_or_duplicates": len(raw) - len(unique),
            "simple_whitespace_units": sum(len(t.split()) == 1 for t in unique),
            "multiword_whitespace_units": sum(len(t.split()) > 1 for t in unique),
            "nested_short_types": len({pair[0] for pair in nested}), "nested_pairs": nested}


def compare(predicted, reference, text):
    p, r = {normalize(t) for t in predicted}, {normalize(t) for t in reference}
    unmatched, missed = p - r, r - p
    normalized_text = " ".join(text.casefold().split())
    result = {"tp": len(p & r), "fp": len(unmatched), "fn": len(missed),
              **prf(len(p & r), len(unmatched), len(missed)),
              "matching_terms": sorted(p & r), "unmatched_prediction_terms": sorted(unmatched),
              "missed_silver_terms": sorted(missed),
              "predictions": term_statistics(predicted), "reference": term_statistics(reference),
              "predicted_not_literal_raw": [t for t in predicted if t not in text],
              "predicted_not_present_normalized_boundaries": sorted(t for t in p if not contained(t, normalized_text)),
              "reference_not_literal_raw": [t for t in reference if t not in text],
              "possible_boundary_relations_not_matches": [[a, b] for a in sorted(unmatched) for b in sorted(r)
                                                          if contained(a, b) or contained(b, a)]}
    result["by_type"] = {}
    for label, predicate in (("simple", lambda t: len(t.split()) == 1),
                             ("multiword", lambda t: len(t.split()) > 1)):
        pp, rr = {t for t in p if predicate(t)}, {t for t in r if predicate(t)}
        result["by_type"][label] = {"tp": len(pp & rr), "fp": len(pp - rr), "fn": len(rr - pp),
                                    **prf(len(pp & rr), len(pp - rr), len(rr - pp))}
    return result


def aggregate(rows):
    totals = {name: sum(row[name] for row in rows) for name in ("tp", "fp", "fn")}
    return {"segments": len(rows), "counts": totals,
            "micro": prf(**totals) if rows else {name: None for name in ("precision", "recall", "f1")},
            "macro": {name: sum(row[name] for row in rows) / len(rows) if rows else None
                      for name in ("precision", "recall", "f1")},
            "zero_denominator_policy": "zero_per_segment; macro_without_segments_is_null"}
