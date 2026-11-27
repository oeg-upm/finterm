# Archived extraction evaluation

`archived_predictions.zip` contains 635 prediction files for eleven comparative configurations: AttentionRank (BERT, FinBERT, RoBERTa), KeyBERT, PromptRank, and Qwen, Gemma and DiSTER with zero-shot and few-shot prompts.

Run `python extraction/evaluate_archived.py` from the repository root (standard library only). The script recalculates micro-averaged precision, recall and F1 for the 58 common segment identifiers against the core and original enriched references. `metrics_archived.csv` contains the output. Exact matching uses casefold and whitespace normalisation, without lemmatisation. Ranking methods retain the first 15 archived entries; LLM methods use the full saved lists.
