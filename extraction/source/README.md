# FINTERM extraction source

This anonymized source snapshot (development version 0.2.0) is derived from the supplied source distribution. Project labels, CLI name and environment-variable prefixes were anonymized consistently; model IDs, prompt wording, algorithms and inference settings were retained. This is not a claim that the snapshot reproduces every historical paper experiment. See `../paper_prompts/README.md` for the available experimental evidence.

## Install and run

Python 3.11+ is required. From this directory:

```bash
python -m pip install -e '.[text,llm]'
python -m extractor --help
```

Provision local checkpoints, CUDA/PyTorch, and English NLTK Punkt separately. The implementation uses sentence-level extraction with stable exact union of output forms. The balanced mode uses Qwen3-14B and the finance_entities_v2 basic prompt; high_recall uses Gemma-2-9B-it and three syntactically retrieved demonstrations via FastKASSIM. These production modes are not interchangeable with the six paper configurations.

```bash
finterm-finance --mode balanced --model-path /path/to/Qwen3-14B --punkt-data /path/to/nltk_data --text 'The bank assessed credit risk.'
```

The high_recall mode also needs `.[syntax]`, Java, Stanford Parser and model JARs, and an approved demonstration bank supplied using `--bank`. This bank, model weights, parser binaries and tokenizer resources are not included. The code includes native DiSTER support but not the complete historical DiSTER few-shot runner. No fallback from a missing retrieval resource to another method is intended.

The source contains the extractor, prompt builders, model adapters, syntactic retrieval, corpus readers, evaluation utilities, and offline tests. It does not include implementations of the five historical candidate-ranking baselines. Their archived outputs and evaluation are provided separately in the parent directory.

To reproduce archived extraction metrics from the repository root, use `python extraction/evaluate_archived.py` with the archived predictions and reference ZIP files. Regenerating all predictions from scratch is not supported by this snapshot alone.

External code, model checkpoints and resources retain their respective licenses. Redistribution of this source should be authorized by its rights holders; no new license is asserted here.
