# Silver-standard review

This package documents the complete V1 panel review (990 term–segment pairs: 840 existing annotations and 150 absent candidates) and the V2 calibration (120 of the same pairs: 100 existing, 20 absent). It is **not** a complete V2 review of the 990 pairs. The 58 evaluable CELEX segments are a subset of the original evaluation sample.

## Reproduce the results

From the repository root, run:

```bash
python silver_standard/review/evaluate_archived.py
```

Python 3 standard library only. No API calls, GPU or external paths. The script verifies the archived metrics and decisions without changing the original reference or predictions. The snapshot contains the normalized reference and prediction sets actually used for these calculations (58 segments, 3,433 original associations, 11 extraction configurations). The archived decisions can be applied to those sets to recover the revised references. It does not regenerate the LLM judgments or the extracted predictions from scratch.

## Files

- `data/pairs_990.csv`: the input archive used for context identity with the V2 calibration; `pairs_990_original_panel_contexts.csv`: the earlier panel input archive, retained separately because some context strings differ. Cohere's V1 per-call input text is not independently hash-verified.
- `votes/v1/`: 990 judgments per model, with the original labels, rationales and available model/configuration/hash metadata. Raw duplicate response text and timestamps are omitted from this public copy.
- `votes/v2_A/`: 360 archived calibration responses in three JSONL files (120 records per judge). Each record preserves every field of the original public JSON response; only the outer serialization has changed. Full prompt, context, model and configuration checks are performed by the evaluator.
- `prompts/`: V1 prompt recovered from the paper appendix (rendered-prompt hash matches: Mistral 0/990, Nemotron 0/990); V2 exact Python template. The JSON braces are escaped for Python `str.format(term=..., context=...)`. Where hashes do not match, V1 is an appendix transcription, not a verified exact request template.
- `data/human_calibration_120.csv`: the 120 human labels used for calibration, cross-checked against the archived calibration vote table and evaluation code. The empty 200-row annotation template and unused notes are excluded. `data/calibration_votes.csv` records the human and model votes. No annotator name is included.
- `results/`: reference-revision metrics over 58 segments, pair-restricted metrics over 120 cases, per-pair decisions and agreement results. P, R, F1 are fractions (multiply by 100 for percentages).
