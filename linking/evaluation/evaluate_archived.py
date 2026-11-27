#!/usr/bin/env python3
"""Recalculate FINTERM linking metrics from archived ranks, offline."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(relative):
    with (ARCHIVE / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rank(value):
    return int(value) if value else None


def metrics(ranks):
    n = len(ranks)
    require(n and all(r is None or r > 0 for r in ranks), "Invalid ranks")
    return {"n": n, "hits_at_1": sum(r == 1 for r in ranks) / n,
            "hits_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r is not None) / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT.parents[1] / "data/finterm_corpus_completo.zip")
    args = parser.parse_args()
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, "Hash mismatch: " + relative)
    baseline = read_csv("baseline/resultados_30_rankings.csv")
    paired = read_csv("paired/comparacion_pareada_celex24.csv")
    full = read_csv("paired/ranking_completo_celex24.csv")
    contexts = read_csv("contexts/final_celex24.csv")
    synthetic = [json.loads(line) for line in (ARCHIVE / "contexts/synthetic30.jsonl").read_text(encoding="utf-8").splitlines()]
    base = {r["caso"]: r for r in baseline}
    context = {r["document_id"]: r for r in contexts}
    require(len(baseline) == len(base) == len(synthetic) == 30, "Expected 30 unique baseline cases")
    require(len(paired) == len({r["caso"] for r in paired}) == len(contexts) == len(context) == 24, "Expected 24 unique paired cases")
    require({r["document_id"] for r in synthetic} == set(base), "Synthetic input IDs mismatch")
    require({r["caso"].replace("eurovoc_", "celex_", 1) for r in paired} == set(context), "CELEX input IDs mismatch")
    for row in synthetic + contexts:
        require(row["text"][int(row["start_char"]):int(row["end_char"])].casefold() == row["term"].casefold(), "Term offsets mismatch: " + row["document_id"])
    groups = defaultdict(list)
    for r in full:
        groups[r["caso"]].append(r)
    require(len(full) == 1832 and set(groups) == {r["caso"] for r in paired}, "Full CELEX ranking coverage mismatch")
    for r in baseline:
        pos = rank(r["posicion"])
        require(int(r["hit1"]) == int(pos == 1) and int(r["hit5"]) == int(pos is not None and pos <= 5), "Baseline hit mismatch")
        require(math.isclose(float(r["reciprocal_rank"]), 1 / pos if pos else 0, abs_tol=1e-12), "Baseline RR mismatch")
    for r in paired:
        b = base[r["caso"]]
        ctx = context[r["caso"].replace("eurovoc_", "celex_", 1)]
        require(b["termino"] == r["termino"] == ctx["term"] and r["celex"] == ctx["celex"], "Paired term/context mismatch")
        require(b["qid_referencia_propuesto"] == r["qid_referencia"] and rank(b["posicion"]) == rank(r["posicion_sintetico"]), "Synthetic subset mismatch")
        g = groups[r["caso"]]
        require(len(g) == len({x["qid"] for x in g}) == int(r["candidatos_puntuados"]), "Candidate count mismatch")
        require(sorted(int(x["posicion_celex"]) for x in g) == list(range(1, len(g)+1)), "Non-contiguous ranks")
        ref = [x for x in g if x["qid"] == r["qid_referencia"]]
        top = [x for x in g if int(x["posicion_celex"]) == 1]
        require(len(ref) == 1 and rank(ref[0]["posicion_celex"]) == rank(r["posicion_celex"]), "CELEX reference rank mismatch")
        require(len(top) == 1 and top[0]["qid"] == r["top1_celex"], "CELEX top-1 mismatch")
    results = {"Synthetic 30": metrics([rank(r["posicion"]) for r in baseline]),
               "Synthetic 24": metrics([rank(r["posicion_sintetico"]) for r in paired]),
               "CELEX 24": metrics([rank(r["posicion_celex"]) for r in paired])}
    with (ROOT / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    require(len(expected) == 3 and {r["setting"] for r in expected} == set(results), "Aggregate table coverage mismatch")
    for r in expected:
        for k, v in results[r["setting"]].items():
            require(math.isclose(float(r[k]), v, abs_tol=1e-12), "Aggregate mismatch: " + r["setting"] + " " + k)
    if args.corpus.is_file():
        with ZipFile(args.corpus) as z:
            for row in contexts:
                raw = z.read(row["archivo_origen"])
                require(hashlib.sha256(raw).hexdigest() == row["sha256_archivo_origen"], "Corpus segment hash mismatch")
                s = raw.decode("utf-8-sig")
                require(s[int(row["inicio_oracion_origen"]):int(row["fin_oracion_origen"])] == row["text"], "Source context mismatch")
        print("Verified source hashes and exact contexts: 24/24")
    else:
        print("Corpus archive not found; source-segment verification skipped.")
    print(f'{"Setting":<16} {"N":>3} {"Hits@1 (%)":>12} {"Hits@5 (%)":>12} {"MRR":>8}')
    for setting, m in results.items():
        print(f'{setting:<16} {m["n"]:>3} {100*m["hits_at_1"]:>12.2f} {100*m["hits_at_5"]:>12.2f} {m["mrr"]:>8.4f}')
    print("OK: archived metrics reproduced; no inference calls made.")


if __name__ == "__main__":
    main()
