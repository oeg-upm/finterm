"""Run portable tests and record a small, non-GPU validation report."""
import json
import os
from pathlib import Path
import platform
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"), top_level_dir=str(PROJECT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    forbidden = sorted({name.partition(".")[0] for name in sys.modules} &
                       {"torch", "transformers", "sentence_transformers", "spacy", "external_extractor"})
    report = {"schema_version": "finterm-offline-validation-v0.1", "python": platform.python_version(),
              "platform": platform.system(), "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
              "external_account_archive_configured": bool(os.environ.get("FINTERM_ACCOUNT_PILOT")),
              "real_model_modules_imported": forbidden, "h100_inference_executed": False,
              "full_gpu_validated": False, "successful": result.wasSuccessful() and not forbidden}
    path = PROJECT / "docs/LOCAL_VALIDATION.json"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
