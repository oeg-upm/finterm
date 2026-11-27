"""Real full-mode example: configure resources externally, then run on H100."""
import os
from pathlib import Path

from finterm_linking import LinkerConfig, link


def main():
    config = LinkerConfig(
        mode="full", similarity_mode="h100", offline=False,
        work_dir=Path(os.environ["FINTERM_WORK_DIR"]),
        qwen14_hub_cache=Path(os.environ["FINTERM_QWEN14_CACHE"]),
        embedding_resource_dir=Path(os.environ["FINTERM_EMBEDDING_RESOURCES"]),
        http_contact=os.environ["FINTERM_WIKIMEDIA_CONTACT"],
        max_translations=128, max_gate_calls=16, max_definition_calls=8, max_new_embeddings=512,
    )
    result = link(
        text="Before investing in the stock market, you must deposit funds into your brokerage account.",
        terms=["account"], config=config, mode="full", top_k=5,
    )
    print(result.mode, result.status, result.execution_kind)
    print(result.top(3))
    print(result.top(5))
    print(result.trace["inferences"])
    print(result.run_dir)


if __name__ == "__main__":
    main()
