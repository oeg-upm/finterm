"""Explicit H100 resource setup, separate from installation/linking.

May download pinned weights into the user-selected external storage. Reuses
existing cache entries. Never called automatically and forbidden on Windows.
"""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from finterm_linking._runtime import frozen_runtime, guard_destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen14-hub-cache", type=Path, required=True)
    parser.add_argument("--embedding-resources", type=Path, required=True)
    parser.add_argument("--embedding-hub-cache", type=Path)
    args = parser.parse_args()
    if sys.platform == "win32":
        raise ValueError("Resource provisioning is an explicit H100 setup step, not a Windows operation")
    for path in (args.qwen14_hub_cache, args.embedding_resources, args.embedding_hub_cache):
        if path is not None:
            guard_destination(path, {PROJECT})
    with frozen_runtime() as runtime:
        from finterm_sense2_qwen import generation_contract
        if not runtime.native.check_environment()["compatible"]:
            raise ValueError("Install/check the validated H100 environment first")
        from huggingface_hub import snapshot_download
        model = generation_contract()
        args.qwen14_hub_cache.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model["model_id"], revision=model["model_revision"], cache_dir=str(args.qwen14_hub_cache))
        args.embedding_resources.parent.mkdir(parents=True, exist_ok=True)
        runtime.native.provision_resources(args.embedding_resources, hub_cache_dir=args.embedding_hub_cache)
    print("Pinned resources provisioned; no model inference executed")


if __name__ == "__main__":
    main()
