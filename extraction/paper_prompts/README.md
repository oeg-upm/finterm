# Paper prompt evidence

The archived JSON files preserve the original message content and use only anonymized source-record basenames. `qwen_few_shot_archived_example.json` retains the six sentence-level prompts and predictions from one Qwen few-shot diagnostic record. These examples are not full-run prompt inventories.

The supplied six-condition summary contains archived examples for Qwen few-shot, Gemma zero-shot and Gemma few-shot; it contains null entries for Qwen zero-shot, DiSTER zero-shot and DiSTER few-shot. The Qwen basic example was reconstructed with the existing builder, not recovered from a historical execution. The Gemma high-recall example illustrates the template using three recorded examples, but does not establish historical FastKASSIM selection. Both source notes are preserved in Spanish. Native DiSTER and finance_entities_v2 builders are present in `../source/prompts/dister.py` and `../source/extractor/financial_prompts.py`; this is not evidence that the paper's DiSTER configurations used precisely these builders. Full historical runner provenance and demonstration selection remain incomplete.

The parent extraction directory contains archived predictions and a separate evaluation script. The evidence here is supplementary, not a substitute for those predictions.
