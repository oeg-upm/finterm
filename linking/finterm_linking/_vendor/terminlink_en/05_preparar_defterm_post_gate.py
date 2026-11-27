#!/usr/bin/env python3
"""Validate the future complete gate and publish a committed two-way partition.

Default execution requires REAL, complete, compatible gate results. It is not
run during local preparation. --self-test uses synthetic temporary files only.
The manifest is published last and binds both CSVs to the inventory, gate file,
scripts and row-level provenance, preventing half-published input reaching DefTerm.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent
FULL_GATE_FILE = "05_defterm_qwen14_sufficiency_gate_fewshot_full_v2.py"


def load_full_gate() -> ModuleType:
    name = "_finterm_post_gate_full_gate_v2"
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / FULL_GATE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Full gate script dependency is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_full_gate()
ENGINE = GATE.ENGINE
ValidationError = GATE.ValidationError
require = GATE.require
canonical_json = GATE.canonical_json
sha256_bytes = GATE.sha256_bytes
SUFFICIENT_FILE = "05_defterm_sufficient_entradas_contextual_v2.csv"
INSUFFICIENT_FILE = "05_defterm_insufficient_contextual_v2.csv"
MANIFEST_FILE = "05_defterm_post_gate_manifest.json"
SCHEMA_VERSION = "defterm-post-gate-contextual-v2-full-v2-leave-self-out-v1"
PROVENANCE_FIELDS = (
    "sufficiency_decision", "sufficiency_gate_model", "sufficiency_gate_model_revision",
    "sufficiency_gate_experiment_type", "sufficiency_gate_prompt_version",
    "sufficiency_gate_result_id", "sufficiency_gate_results_file",
    "sufficiency_gate_results_sha256", "sufficiency_gate_result_row_sha256",
    "sufficiency_gate_script_sha256", "sufficiency_gate_runtime_engine_sha256",
    "sufficiency_gate_input_sha256", "sufficiency_gate_fewshot_sha256",
    *GATE.ROW_PROVENANCE_FIELDS,
    "post_gate_partition_id", "post_gate_preparer_script_sha256",
)
PARTITION_FIELDS = (*GATE.INVENTORY_FIELDS, *PROVENANCE_FIELDS)


@dataclass(frozen=True)
class PartitionPaths:
    sufficient: Path
    insufficient: Path
    manifest: Path
    journal: Path
    lock: Path


PATHS = PartitionPaths(
    BASE_DIR / SUFFICIENT_FILE, BASE_DIR / INSUFFICIENT_FILE,
    BASE_DIR / MANIFEST_FILE, BASE_DIR / ".05_preparar_defterm_post_gate.inflight.json",
    BASE_DIR / ".05_preparar_defterm_post_gate.lock",
)


@dataclass
class PartitionPlan:
    sufficient: list[dict[str, str]]
    insufficient: list[dict[str, str]]
    sufficient_payload: bytes
    insufficient_payload: bytes
    manifest: dict[str, Any]
    manifest_payload: bytes


def serialize_csv(rows: list[dict[str, str]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def build_plan(
    inventory: list[dict[str, str]], gate_rows: list[dict[str, str]],
    gate_results_hash: str, static: dict[str, str], *, allow_self_test: bool = False,
) -> PartitionPlan:
    GATE.validate_gate_rows(inventory, gate_rows, static, allow_self_test=allow_self_test)
    preparer_hash = sha256_bytes(Path(__file__).read_bytes())
    identity = {
        'schema_version': SCHEMA_VERSION, 'inventory_sha256': static['input_sha256'],
        'gate_results_sha256': gate_results_hash, 'gate_script_sha256': static['inference_script_sha256'],
        'preparer_script_sha256': preparer_hash, 'metadata_mode': static['metadata_mode'],
    }
    partition_id = 'post_gate_partition_' + sha256_bytes(canonical_json(identity).encode('utf-8')).lower()
    sufficient: list[dict[str, str]] = []
    insufficient: list[dict[str, str]] = []
    for source, result in zip(inventory, gate_rows):
        row_hash = sha256_bytes(canonical_json(result).encode('utf-8'))
        result_id = 'sufficiency_gate_result_' + sha256_bytes(canonical_json({
            'gate_results_sha256': gate_results_hash, 'row_sha256': row_hash,
            'defterm_input_id': source['defterm_input_id'],
        }).encode('utf-8')).lower()
        row = {
            **{field: source[field] for field in GATE.INVENTORY_FIELDS},
            'sufficiency_decision': result['sufficiency_decision'],
            'sufficiency_gate_model': result['model_id'],
            'sufficiency_gate_model_revision': result['model_revision'],
            'sufficiency_gate_experiment_type': result['experiment_type'],
            'sufficiency_gate_prompt_version': result['prompt_version'],
            'sufficiency_gate_result_id': result_id,
            'sufficiency_gate_results_file': GATE.OUTPUT_FILE,
            'sufficiency_gate_results_sha256': gate_results_hash,
            'sufficiency_gate_result_row_sha256': row_hash,
            'sufficiency_gate_script_sha256': result['inference_script_sha256'],
            'sufficiency_gate_runtime_engine_sha256': result['runtime_engine_sha256'],
            'sufficiency_gate_input_sha256': result['input_sha256'],
            'sufficiency_gate_fewshot_sha256': result['fewshot_sha256'],
            **{field: result[field] for field in GATE.ROW_PROVENANCE_FIELDS},
            'post_gate_partition_id': partition_id,
            'post_gate_preparer_script_sha256': preparer_hash,
        }
        require(tuple(row) == PARTITION_FIELDS, "Partition schema differs")
        require(row['defterm_evidence_text'] == source['defterm_evidence_text'], "Evidence changed during partitioning")
        destination = sufficient if row['sufficiency_decision'] == ENGINE.SUFFICIENT_EVIDENCE else insufficient
        destination.append(row)
    require(len(sufficient) + len(insufficient) == len(inventory), "Partition lost candidates")
    require({row['defterm_input_id'] for row in sufficient}.isdisjoint(row['defterm_input_id'] for row in insufficient), "Partitions overlap")
    sufficient_payload = serialize_csv(sufficient, PARTITION_FIELDS)
    insufficient_payload = serialize_csv(insufficient, PARTITION_FIELDS)
    manifest = {
        **identity, 'runtime_engine_sha256': GATE.PILOT_SHA256,
        'inventory_file': GATE.INPUT_FILE, 'gate_results_file': GATE.OUTPUT_FILE,
        'partition_id': partition_id, 'gate_model_id': GATE.MODEL_ID,
        'gate_model_revision': GATE.MODEL_REVISION, 'gate_prompt_version': GATE.PROMPT_VERSION,
        'sufficient_file': SUFFICIENT_FILE, 'sufficient_sha256': sha256_bytes(sufficient_payload),
        'sufficient_count': len(sufficient), 'insufficient_file': INSUFFICIENT_FILE,
        'insufficient_sha256': sha256_bytes(insufficient_payload), 'insufficient_count': len(insufficient),
        'total_count': len(inventory),
        'self_exclusion_policy': GATE.SELF_EXCLUSION_POLICY,
        'full_10_shot_count': sum(row['gate_inference_mode'] == GATE.FULL_MODE for row in gate_rows),
        'leave_self_out_9_shot_count': sum(row['gate_inference_mode'] == GATE.LOO_MODE for row in gate_rows),
    }
    manifest_payload = (canonical_json(manifest) + '\n').encode('utf-8')
    return PartitionPlan(sufficient, insufficient, sufficient_payload, insufficient_payload, manifest, manifest_payload)


def read_plan(
    inventory_path: Path = BASE_DIR / GATE.INPUT_FILE,
    gate_path: Path = BASE_DIR / GATE.OUTPUT_FILE,
    *, allow_self_test: bool = False,
) -> PartitionPlan:
    require(inventory_path.is_file(), "Inventory is missing")
    inventory_payload = inventory_path.read_bytes()
    inventory_hash = sha256_bytes(inventory_payload)
    if not allow_self_test:
        require(inventory_hash == GATE.INPUT_SHA256, "Official inventory hash differs")
        require(gate_path.name == GATE.OUTPUT_FILE, "Post-gate accepts exclusively the full V2 result filename")
    inventory = ENGINE.parse_csv_payload(inventory_path.name, inventory_payload, GATE.INVENTORY_FIELDS)
    GATE.validate_inventory(inventory, allow_fixture=allow_self_test)
    require(gate_path.is_file(), "Real full-gate results are not available yet")
    gate_payload = gate_path.read_bytes()
    require(bool(gate_payload) and gate_payload.endswith(b'\n'), "Full-gate results are empty/incomplete")
    rows = ENGINE.parse_csv_payload(gate_path.name, gate_payload, GATE.RESULT_FIELDS)
    static = GATE.expected_static_metadata('SELF_TEST' if allow_self_test else 'PRODUCTION', input_hash=inventory_hash)
    return build_plan(inventory, rows, sha256_bytes(gate_payload), static, allow_self_test=allow_self_test)


def verify_committed_plan(plan: PartitionPlan, paths: PartitionPaths = PATHS) -> None:
    require(paths.manifest.is_file(), "Post-gate partition has no commit manifest")
    require(paths.manifest.read_bytes() == plan.manifest_payload, "Partition manifest/provenance is incompatible")
    require(paths.sufficient.is_file() and paths.insufficient.is_file(), "Only one partition was published")
    require(paths.sufficient.read_bytes() == plan.sufficient_payload, "Sufficient partition was changed, mislabeled or is incomplete")
    require(paths.insufficient.read_bytes() == plan.insufficient_payload, "Insufficient partition was changed or is incomplete")


def load_committed_partition(
    paths: PartitionPaths = PATHS,
    inventory_path: Path = BASE_DIR / GATE.INPUT_FILE,
    gate_path: Path = BASE_DIR / GATE.OUTPUT_FILE,
    *, allow_self_test: bool = False,
) -> tuple[PartitionPlan, str]:
    # Revalidate the REAL gate, then reproduce the expected partition in memory.
    # This binds every sufficient row to the actual upstream decision, not merely
    # to a potentially edited sufficiency_decision cell in the input CSV.
    plan = read_plan(inventory_path, gate_path, allow_self_test=allow_self_test)
    verify_committed_plan(plan, paths)
    return plan, sha256_bytes(paths.manifest.read_bytes())


def publish_plan(plan: PartitionPlan, paths: PartitionPaths = PATHS) -> str:
    lock = ENGINE.acquire_run_lock(paths.lock)
    try:
        if paths.manifest.exists():
            verify_committed_plan(plan, paths)
            if paths.journal.exists():
                require(paths.journal.read_bytes() == plan.manifest_payload, "Surviving publication journal differs")
                paths.journal.unlink()
                ENGINE.sync_directory(paths.journal.parent)
            return 'UNCHANGED_IDENTICAL'
        if paths.journal.exists():
            require(paths.journal.read_bytes() == plan.manifest_payload, "Interrupted publication belongs to incompatible inputs")
            for path, payload in ((paths.sufficient, plan.sufficient_payload), (paths.insufficient, plan.insufficient_payload)):
                if path.exists():
                    require(path.read_bytes() == payload, "Existing uncommitted CSV differs; refusing overwrite")
        else:
            require(not paths.sufficient.exists() and not paths.insufficient.exists(), "Uncommitted outputs exist without a publication journal")
            ENGINE.write_bytes_atomically(paths.journal, plan.manifest_payload)
        ENGINE.write_bytes_atomically(paths.sufficient, plan.sufficient_payload)
        ENGINE.write_bytes_atomically(paths.insufficient, plan.insufficient_payload)
        # Commit point: both durable CSVs exist before the durable manifest.
        ENGINE.write_bytes_atomically(paths.manifest, plan.manifest_payload)
        verify_committed_plan(plan, paths)
        paths.journal.unlink()
        ENGINE.sync_directory(paths.journal.parent)
        return 'COMMITTED'
    finally:
        ENGINE.release_run_lock(lock)


def synthetic_fixture(root: Path) -> tuple[PartitionPlan, PartitionPaths, Path, Path]:
    inventory = [GATE.synthetic_inventory_row(index) for index in range(1, 5)]
    inventory_payload = serialize_csv(inventory, GATE.INVENTORY_FIELDS)
    inventory_path = root / 'synthetic_inventory.csv'
    gate_path = root / 'synthetic_gate_results.csv'
    ENGINE.write_bytes_atomically(inventory_path, inventory_payload)
    rows = GATE.synthetic_gate_rows(inventory, input_hash=sha256_bytes(inventory_payload))
    ENGINE.write_bytes_atomically(gate_path, ENGINE.serialize_result_rows(rows))
    paths = PartitionPaths(root / 'synthetic_sufficient.csv', root / 'synthetic_insufficient.csv', root / 'synthetic_manifest.json', root / 'synthetic_publication.json', root / 'synthetic_post.lock')
    plan = read_plan(inventory_path, gate_path, allow_self_test=True)
    return plan, paths, inventory_path, gate_path


def run_self_tests() -> None:
    ENGINE.assert_no_model_runtime()
    tests: dict[str, str] = {}

    def passed(name: str, condition: bool) -> None:
        require(condition, f"{name} failed")
        tests[name] = 'PASS'

    parent = Path(tempfile.gettempdir()) / 'opencode'
    with tempfile.TemporaryDirectory(prefix='post-gate-self-test-', dir=parent if parent.is_dir() else None) as name:
        plan, paths, inventory_path, gate_path = synthetic_fixture(Path(name))
        passed('POST_GATE_PARTITION_TEST', len(plan.sufficient) == len(plan.insufficient) == 2)
        passed('POST_GATE_EVIDENCE_IMMUTABILITY_TEST', all(row['defterm_evidence_text'].startswith('Label: Synthetic candidate') for row in [*plan.sufficient, *plan.insufficient]))
        passed('POST_GATE_PROVENANCE_TEST', len({row['sufficiency_gate_result_id'] for row in [*plan.sufficient, *plan.insufficient]}) == 4)
        require(publish_plan(plan, paths) == 'COMMITTED', 'Synthetic publication failed')
        loaded, _ = load_committed_partition(paths, inventory_path, gate_path, allow_self_test=True)
        passed('ATOMIC_PARTITION_COMMIT_TEST', loaded.manifest == plan.manifest)
        passed('POST_GATE_IDEMPOTENCE_TEST', publish_plan(plan, paths) == 'UNCHANGED_IDENTICAL')
        invalid = [dict(row) for row in plan.sufficient]
        invalid[0]['sufficiency_decision'] = ENGINE.INSUFFICIENT_EVIDENCE
        ENGINE.write_bytes_atomically(paths.sufficient, serialize_csv(invalid, PARTITION_FIELDS))
        ENGINE.expect_validation_error(lambda: load_committed_partition(paths, inventory_path, gate_path, allow_self_test=True))
        passed('TAMPERED_PARTITION_REJECTION_TEST', True)
        ENGINE.write_bytes_atomically(paths.sufficient, plan.sufficient_payload)
        # A CSV with valid-looking labels but from an insufficient source row
        # cannot pass the upstream-replayed content/manifest checks.
        wrong = [*plan.sufficient, {**plan.insufficient[0], 'sufficiency_decision': ENGINE.SUFFICIENT_EVIDENCE}]
        ENGINE.write_bytes_atomically(paths.sufficient, serialize_csv(wrong, PARTITION_FIELDS))
        ENGINE.expect_validation_error(lambda: verify_committed_plan(plan, paths))
        passed('NO_INSUFFICIENT_ROW_CAN_REACH_DEFTERM_TEST', True)
        ENGINE.write_bytes_atomically(paths.sufficient, plan.sufficient_payload)
        saved_manifest = paths.manifest.read_bytes()
        paths.manifest.unlink()
        ENGINE.expect_validation_error(lambda: verify_committed_plan(plan, paths))
        ENGINE.write_bytes_atomically(paths.journal, saved_manifest)
        require(publish_plan(plan, paths) == 'COMMITTED', 'Interrupted pair publication did not recover')
        passed('INTERRUPTED_PARTITION_RECOVERY_TEST', True)
        gate_rows = ENGINE.parse_csv_payload(gate_path.name, gate_path.read_bytes(), GATE.RESULT_FIELDS)
        incomplete = gate_rows[:-1]
        inventory = ENGINE.parse_csv_payload(inventory_path.name, inventory_path.read_bytes(), GATE.INVENTORY_FIELDS)
        static = GATE.expected_static_metadata('SELF_TEST', input_hash=sha256_bytes(inventory_path.read_bytes()))
        ENGINE.expect_validation_error(lambda: GATE.validate_gate_rows(inventory, incomplete, static, allow_self_test=True))
        ENGINE.expect_validation_error(lambda: GATE.validate_gate_rows(inventory, [gate_rows[0], gate_rows[0], *gate_rows[2:]], static, allow_self_test=True))
        bad = [dict(row) for row in gate_rows]
        bad[0]['generation_raw'] += '\n'
        ENGINE.expect_validation_error(lambda: GATE.validate_gate_rows(inventory, bad, static, allow_self_test=True))
        passed('INCOMPLETE_DUPLICATE_NONEXACT_GATE_REJECTION_TEST', True)
        production = GATE.expected_static_metadata('PRODUCTION', input_hash=static['input_sha256'])
        ENGINE.expect_validation_error(lambda: GATE.validate_gate_rows(inventory, gate_rows, production))
        passed('SYNTHETIC_PRODUCTION_REJECTION_TEST', True)
        previous_fields = GATE.BASE.RESULT_FIELDS
        previous_rows = [{field: row[field] for field in previous_fields} for row in gate_rows]
        for row in previous_rows:
            row['run_scope'] = GATE.BASE.RUN_SCOPE
            row['inference_script_sha256'] = GATE.PREVIOUS_FULL_SHA256
        previous_path = Path(name) / 'synthetic_previous_full.csv'
        ENGINE.write_bytes_atomically(previous_path, serialize_csv(previous_rows, previous_fields))
        ENGINE.expect_validation_error(lambda: read_plan(inventory_path, previous_path, allow_self_test=True))
        passed('PREVIOUS_FULL_RESULT_REJECTION_TEST', True)
        official = GATE.read_inventory()
        loo_inventory = [row for row in official if row['defterm_input_id'] in GATE.LOO_BY_ID]
        loo_rows = GATE.synthetic_gate_rows(loo_inventory)
        loo_payload = ENGINE.serialize_result_rows(loo_rows)
        loo_plan = build_plan(loo_inventory, loo_rows, sha256_bytes(loo_payload), GATE.expected_static_metadata('SELF_TEST'), allow_self_test=True)
        by_id = {row['defterm_input_id']: row for row in loo_rows}
        passed('LOO_PROVENANCE_PRESERVATION_TEST', len(loo_inventory) == 10
               and loo_plan.manifest['leave_self_out_9_shot_count'] == 10
               and all(all(row[field] == by_id[row['defterm_input_id']][field] for field in GATE.ROW_PROVENANCE_FIELDS)
                       for row in [*loo_plan.sufficient, *loo_plan.insufficient]))
        invalid_provenance = [dict(row) for row in loo_rows]
        invalid_provenance[0]['effective_fewshot_count'] = '10'
        ENGINE.expect_validation_error(lambda: GATE.validate_gate_rows(loo_inventory, invalid_provenance, GATE.expected_static_metadata('SELF_TEST'), allow_self_test=True))
        invalid_provenance = [dict(row) for row in loo_rows]
        invalid_provenance[0]['effective_fewshot_messages_sha256'] = GATE.ENGINE.EXPECTED_FEWSHOT_MESSAGES_SHA256
        ENGINE.expect_validation_error(lambda: GATE.validate_gate_rows(loo_inventory, invalid_provenance, GATE.expected_static_metadata('SELF_TEST'), allow_self_test=True))
        passed('INCORRECT_EFFECTIVE_PREFIX_REJECTION_TEST', True)
        passed('OUTPUT_SCHEMA_TEST', all(tuple(row) == PARTITION_FIELDS for row in [*plan.sufficient, *plan.insufficient]))
    ENGINE.assert_no_model_runtime()
    print('SELF_TEST_STATUS=SUCCESS')
    for name, status in tests.items():
        print(f'{name}={status}')
    print('SELF_TEST_SCOPE=SYNTHETIC_TEMPORARY_FILES_ONLY')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--validate-prerequisites', action='store_true')
    args = parser.parse_args(argv)
    ENGINE.assert_no_model_runtime()
    if args.self_test:
        run_self_tests()
    elif args.validate_prerequisites:
        inventory = GATE.read_inventory()
        print(f'FULL_INPUT_COUNT={len(inventory)}')
        print('INPUT_VALIDATION=STATIC_PREREQUISITES_OK')
        print(f'REAL_GATE_RESULTS_AVAILABLE={ENGINE.bool_text((BASE_DIR / GATE.OUTPUT_FILE).is_file())}')
    else:
        plan = read_plan()
        print(f'POST_GATE_PUBLICATION={publish_plan(plan)}')
        print(f'SUFFICIENT_COUNT={len(plan.sufficient)}')
        print(f'INSUFFICIENT_COUNT={len(plan.insufficient)}')
    ENGINE.assert_no_model_runtime()
    print('MODEL_LOADED=NO')
    print('INFERENCE_EXECUTED=NO')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f'POST_GATE_STATUS=FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
