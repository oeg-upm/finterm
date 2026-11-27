# FINTERM: preliminary entity-linking evaluation

This directory archives the results behind the three entity-linking evaluation rows. It is an offline evaluation of candidate retrieval and semantic ranking against proposed EuroVoc--Wikidata QIDs, not a human-validated gold standard or an evaluation of all enrichment stages.

## Reproduce the reported metrics

From the repository root, using Python 3.9+ (standard library only):

```sh
python linking/evaluation/evaluate_archived.py
```

The script checks file integrity, verifies the paired cases and archived CELEX ranking positions, and prints Hits@1, Hits@5 and MRR. When `data/finterm_corpus_completo.zip` is available, it also verifies the source-segment hashes, context offsets and text for the final 24 CELEX cases. A different corpus archive may be supplied via `--corpus PATH`. No API calls, GPU or model downloads are required. The script does not regenerate candidates, embeddings or rankings.

| Setting | Cases | Hits@1 (%) | Hits@5 (%) | MRR |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 30 | 30 | 33.33 | 73.33 | 0.5005 |
| Synthetic 24 | 24 | 29.17 | 70.83 | 0.4659 |
| CELEX 24 | 24 | 29.17 | 54.17 | 0.4205 |

Hits@k is the fraction of cases in which the proposed reference QID occurs among the first k ranked candidates. MRR averages reciprocal reference ranks over all cases, with zero for a missing reference. Synthetic 24 is the exact subset used for the paired comparison, not an independent experiment. The synthetic-30 retrieval stage recovered 29 of 30 proposed QIDs (96.67%). The unretrieved case contributes zero to all ranking metrics. The historical comparison manifest records Qwen3-Embedding-4B, cosine similarity, maximum candidate aggregation and numeric QID tie-breaking. No new translations or definitions were generated in the baseline evaluation.

## Contents

- `metrics.csv`: recalculated aggregate metrics (fractions, not percentages).
- `archive/baseline/`: synthetic-30 reference ranks and original metrics. The archive retains reference ranks and top-5 candidate details (in `semantic_review/`), not the complete synthetic rankings.
- `archive/paired/`: exact 24-case paired comparison, complete archived CELEX ranked candidate lists, and historical configuration manifest.
- `archive/contexts/final_celex24.csv`: final 24 selected CELEX contexts, source hashes and offsets. `archivo_origen` points to a member of the corpus ZIP, not a local private path.
- `archive/contexts/synthetic30.jsonl`: original synthetic inputs without reference QIDs.
- `archive/contexts/preselection/`: earlier selection records, including 27 preliminary inputs and 509 candidate excerpts. These are historical preparation records, NOT the final test inputs.
- `archive/retrieval/`: historical candidate-retrieval inventory and report, preceding the semantic ranking experiment. The report's statement that ranking had not been run refers to that earlier stage.
- `archive/semantic_review/`: reference-versus-top-1 and top-5 inspection tables. The manual semantic review fields remain pending; these tables do not establish human validation.
- `provenance.json`, `SHA256SUMS.txt`: source-to-publication mapping, anonymisation record and integrity checks.

## Limitations

The proposed QIDs were not manually validated against each occurrence. This is a preliminary evaluation of archived rankings. Candidate selection and the 24-case subset are not a representative random sample. It cannot establish the contribution of translation or definition generation. Historical caches and the exact full synthetic rankings are not included; reproducing the scores from archived ranks is distinct from reproducing the complete inference pipeline. Historical manifest hashes refer to the original experiment artifacts and may differ from the anonymised copies; use `SHA256SUMS.txt` for the published copies. Private paths and project identifiers were removed, without changing terms, context text, QIDs, ranking scores, ranks or model settings.
