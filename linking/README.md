# FINTERM — Financial term linking to Wikidata

Experimental implementation of the English-language linking pipeline accompanying the paper. Given a document and a list of terms, the linker returns ranked Wikidata candidates for each retained occurrence. It is independent of the term extractor.

## Pipeline

1. Extract term contexts and occurrence identifiers (01).
2. Retrieve and lemmatise Wikidata candidates (02).
3. Prepare candidate semantic representations, including contextualised translations where applicable (03).
4. Generate definitions only for eligible candidates with sufficient semantic evidence (05).
5. Rank candidates using Qwen3-Embedding-4B, cosine similarity and MAX aggregation (06).

Translation and definition generation use Qwen3-14B. Candidate enrichment uses candidate evidence, not document context. Model revisions, prompts, few-shot examples and inference parameters are included in the source; see [installation](docs/INSTALL.md).

## Installation and offline tests

From this `linking/` directory, using Python 3.11 or 3.12 in a separate environment:

```bash
python -B scripts/verify_distribution.py
python -m pip install .
python -B -m unittest discover -s tests -v
finterm-link doctor
```

Run the distribution verifier from the clean source tree before installation/build if the build generates local artifacts. The base package does not load model weights. Offline tests exercise contracts and controlled backends, not real GPU inference. An optional external archive test is skipped when its historical fixture is absent.

For online candidate retrieval install `.[wikidata]` and the spaCy English model described in [INSTALL.md](docs/INSTALL.md). Set an authorised Wikimedia contact. For full execution use the CUDA 12.6 environment and local model resources described there. No weights or private caches are distributed.

## Usage

Preparation without model inference (no invented scores):

```bash
finterm-link run --text "The bank adjusted interest rates." --term bank --term "interest rates" --mode quick --execution prepare --online --contact "mailto:research-group@example.org" --work-dir ./runs_demo --top-k 3
```

Replace the example contact with an authorised public contact before making online requests. For full execution, adapt the paths in `configs/full_h100.example.json` and use:

```bash
finterm-link run --config configs/full_h100.example.json --text-file document.txt --terms-file terms.txt --contact "mailto:research-group@example.org" --top-k 5
```

Python API (paths and contact must be adapted):

```python
from pathlib import Path
from finterm_linking import LinkerConfig, link

config = LinkerConfig(
    mode="full", similarity_mode="h100", offline=False,
    work_dir=Path("/WORK/runs"),
    qwen14_hub_cache=Path("/RESOURCES/qwen14_hub"),
    embedding_resource_dir=Path("/RESOURCES/qwen3_embedding4b"),
    http_contact="mailto:research-group@example.org",
    max_translations=128, max_gate_calls=16,
    max_definition_calls=8, max_new_embeddings=512,
)
result = link(text="The bank adjusted interest rates.",
              terms=["bank", "interest rates"], config=config,
              mode="full", top_k=5)
print(result.top(5))
```

Each run stores phase contracts, `result.json`, `rankings.csv`, `top_candidates.json`, and `integration_manifest.json`. Missing scores are not set to zero. Top 1 is a ranking position, not a guaranteed correct link. `finterm-link verify /PATH/TO/RUN` checks persisted outputs.
