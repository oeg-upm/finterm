# finterm
An LLM-based framework for financial terminology under semantic scarcity: silver standard construction, term extraction and concept linking to Wikidata.

# Filling the Semantic Gap in Financial Terminology

This repository contains the code and data for an LLM-based framework that addresses semantic scarcity in financial terminology at three levels:

1. **Resources.** A silver standard of English financial terminology, built from expert-validated IATE entries and a domain glossary projected onto a corpus of EU financial documents (2,555 segments), and enriched with a panel of LLM judges.
2. **Extraction.** A comparison of ranking-based keyphrase methods (AttentionRank, KeyBERT, PromptRank) and direct extraction with LLMs (Qwen3-14B, Gemma-2-9B-it, DiSTER-Llama-3-8B-Instruct), evaluated against the silver standard.
3. **Concepts.** A linking method that maps extracted terms to Wikidata by comparing their meaning in context with an enriched set of candidate senses: original descriptions, context-aware translations of descriptions in other languages, and definitions generated only when the available evidence is sufficient.

Linking to Wikidata turns the English term list into a multilingual resource intended to support glossary creation for computer-assisted interpreting.

## Repository structure

- `silver_standard/`: construction scripts and the core and enriched reference annotations
- `extraction/`: ranking baselines, LLM extraction prompts and evaluation scripts
- `linking/`: candidate retrieval, sense construction (translation and definition generation) and ranking
- `data/`: corpus segments and the resulting linked resource
