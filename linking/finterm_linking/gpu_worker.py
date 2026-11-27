"""One original GPU stage per process: no simultaneous 14B/4B residency."""
import argparse
import gc
import json
from pathlib import Path
import sys

from ._runtime import frozen_runtime, guard_destination, protected_roots


def run_job(job):
    if sys.platform == "win32":
        raise ValueError("Real inference is disabled on Windows")
    if job.get("schema_version") != "finterm-gpu-stage-job-v0.1" or job.get("stage") not in {"03", "05", "06"}:
        raise ValueError("Unsupported GPU stage job")
    if job["stage"] != "06" and job["mode"] != "full":
        raise ValueError("Enrichment inference requires full mode")
    roots = protected_roots(job["code_dir"], job["input_dir"], job["resource_dir"])
    roots.update(Path(job[k]).resolve() for k in ("code_dir", "input_dir", "stage01_dir", "stage02_dir"))
    for key in ("resource_dir", "hub_cache_dir"):
        if job.get(key):
            roots.add(Path(job[key]).resolve())
    for key in ("output_dir", "cache_path", "gpu_lock"):
        guard_destination(job[key], roots)
    with frozen_runtime(job["code_dir"]) as runtime, runtime.native.exclusive_lock(Path(job["gpu_lock"])):
        environment = runtime.native.check_environment()
        if not environment["compatible"]:
            raise ValueError("GPU environment differs from the original validated profile")
        phase, budget, backend = job["stage"], job["budgets"], None
        try:
            if phase == "03":
                from finterm_sense2_qwen import QwenContextualV2
                backend = QwenContextualV2(cache_dir=job.get("hub_cache_dir"))
                return runtime.sense2_store.run_stage03(job["input_dir"], job["output_dir"], job["cache_path"],
                    policy=job["translation_policy"], max_translations=budget["translations"], adapter=backend)
            if phase == "05":
                from finterm_defterm import QwenDefTerm
                backend = QwenDefTerm(cache_dir=job.get("hub_cache_dir"))
                return runtime.defterm_store.run_defterm(job["input_dir"], job["output_dir"], job["cache_path"],
                    max_gate_calls=budget["gate"], max_definition_calls=budget["definitions"], backend=backend,
                    preselection_policy=job["preselection_policy"])
            backend = runtime.native.QwenEmbedding4B(job["resource_dir"], batch_size=1,
                                                   gpu_reserve_gib=8.0, allow_shared_gpu=False)
            return runtime.similarity.run_similarity(job["input_dir"], job["output_dir"], job["cache_path"],
                backend=backend, max_new_embeddings=budget["embeddings"],
                stage01_dir=job["stage01_dir"], stage02_dir=job["stage02_dir"])
        finally:
            if backend is not None:
                if hasattr(backend, "close"):
                    backend.close()
                else:
                    backend.model = None
                    backend.tokenizer = None
                    gc.collect()
                    if "torch" in sys.modules:
                        sys.modules["torch"].cuda.empty_cache()
            # Process exit is the final release boundary for every stage.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_job(json.loads(args.job.read_text(encoding="utf-8")))
    print(json.dumps({"status": result["status"], "execution": result["execution"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
