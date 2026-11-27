#!/usr/bin/env python3
"""Frozen gate with per-target exclusion of the target's own demonstration.

140 targets retain the frozen 10-shot prefix; 10 targets omit exactly their own
user/assistant pair. This is self-leakage prevention, not a criterion/prompt or
parameter change. No example label is used as a result. Default execution makes
one model call per target. Local modes import no ML runtime and create no real
results. The previous full and pilot are immutable, SHA-pinned utility modules;
neither previous runner's main or inference loop is called.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent
PREVIOUS_FULL_FILE = '05_defterm_qwen14_sufficiency_gate_fewshot_full.py'
PREVIOUS_FULL_SHA256 = '18B87213D5192E072674497F5DEF936C2C1C9DEA738C1E8E5EC53640FD5C16FE'


def load_previous_utilities() -> ModuleType:
    path = BASE_DIR / PREVIOUS_FULL_FILE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest().upper() != PREVIOUS_FULL_SHA256:
        raise RuntimeError('Immutable previous full utility dependency changed')
    name = '_finterm_full_v2_immutable_utilities'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise RuntimeError('Cannot load immutable full utilities')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


BASE = load_previous_utilities()
load_frozen_engine = BASE.load_frozen_engine
ENGINE = load_frozen_engine('full_gate_v2')
ENGINE.ValidationError = BASE.ValidationError
ValidationError = BASE.ValidationError
require = BASE.require
sha256_bytes = BASE.sha256_bytes
canonical_json = BASE.canonical_json
PILOT_FILE = BASE.PILOT_FILE
PILOT_SHA256 = BASE.PILOT_SHA256
INPUT_FILE = BASE.INPUT_FILE
INPUT_SHA256 = BASE.INPUT_SHA256
FEWSHOT_FILE = BASE.FEWSHOT_FILE
FEWSHOT_SHA256 = BASE.FEWSHOT_SHA256
MODEL_ID = ENGINE.MODEL_ID
MODEL_REVISION = ENGINE.MODEL_REVISION
EXPERIMENT_TYPE = ENGINE.EXPERIMENT_TYPE
PROMPT_VERSION = ENGINE.PROMPT_VERSION
SYSTEM_MSG = ENGINE.SYSTEM_MSG
USER_MSG_TEMPLATE = ENGINE.USER_MSG_TEMPLATE
FEWSHOT_USER_MSG_TEMPLATE = ENGINE.FEWSHOT_USER_MSG_TEMPLATE
INVENTORY_FIELDS = BASE.INVENTORY_FIELDS
validate_inventory = BASE.validate_inventory
read_inventory = BASE.read_inventory
synthetic_inventory_row = BASE.synthetic_inventory_row
validate_technical_metadata = BASE.validate_technical_metadata
run_checkpointed_stage = BASE.run_checkpointed_stage  # DefTerm runtime/persistence only.
OUTPUT_FILE = '05_defterm_qwen14_sufficiency_gate_fewshot_full_v2_results.csv'
RUN_SCOPE = 'FULL_CONTEXTUAL_V2_150_LEAVE_SELF_OUT'
SELF_EXCLUSION_POLICY = 'LEAVE_SELF_OUT_BY_DEFTERM_INPUT_ID_V1'
FULL_MODE = 'FULL_10_SHOT'
LOO_MODE = 'LEAVE_SELF_OUT_9_SHOT'
ROW_PROVENANCE_FIELDS = (
    'gate_inference_mode', 'effective_fewshot_count', 'excluded_self_example_id',
    'effective_fewshot_messages_sha256',
)

# No labels in this membership/provenance index. Identity is always the full
# defterm_input_id; term/QID are consistency checks, never alternative joins.
LOO_IDENTITIES = (
    ('defterm_input_96ad541f3cb9c06905bb238eb3d5bdcc6e8bbd7dca17e60350930e3a703ed2b1', 'European Central Bank', 'Q76161279', 'sufficiency_example_01'),
    ('defterm_input_7fc21a5ebb891c4d5572e54b0ed5503435bdf67f7a6b3eea5f8ed7a8bda8da65', 'Timely', 'Q124407807', 'sufficiency_example_02'),
    ('defterm_input_c6ac6ac4ffd521a9204a02e14c13f572f5600eb6abd2a48a955960d25c1ea793', 'regulation', 'Q18845817', 'sufficiency_example_03'),
    ('defterm_input_1c0777b096ca0f5eb7766fc04682305d080cec786d74ceee16ba09ba1d232782', 'solely', 'Q112204255', 'sufficiency_example_04'),
    ('defterm_input_f19b649900c47bdc7d714ae4c5e685225efa1e7d9719908407ff0c1247afc2a6', 'Commission', 'Q2986626', 'sufficiency_example_05'),
    ('defterm_input_20f32550d9554b278bbde134e959bd73e9afb81a8f0e2f911ce844fbdd3f1152', 'transparency', 'Q4380181', 'sufficiency_example_06'),
    ('defterm_input_90410b9ce235267377b4b890af07a1693749b38a94f6ee91a5a0f586d8d3f970', 'ESA', 'Q5358555', 'sufficiency_example_07'),
    ('defterm_input_1479c75eb80483cb26932ac289ad2269f8635b0baa30b759b0d67d63c89df178', 'EEA', 'Q7887783', 'sufficiency_example_08'),
    ('defterm_input_9ec1e1caf76510764eec9989651a3a53c45761fafeeeea4416fa06c3cfff681c', 'structure', 'Q28659665', 'sufficiency_example_09'),
    ('defterm_input_a021b357c51e51576bdb5b6a6328dff90237829b684415ba136222b18b8d648a', 'covering', 'Q134100366', 'sufficiency_example_10'),
)
FROZEN_LOO_HASHES = {
    'sufficiency_example_01': '28B72A05856E383BC814F1F81E98B85D954B002A4FD8E15CBC191A02AB823E00',
    'sufficiency_example_02': '56119C4075DDB5913B61009CAF4ABE83AB5AD0E330DB84ABB87786257E9C80DC',
    'sufficiency_example_03': 'BDEA5F822E91547CB60767C709CBAC79611B72B8302286FDCD2212DA2458E15D',
    'sufficiency_example_04': 'FBFB58137DDE0EB1E843D16DDF3A8ED0AB24D706133F0FF617FABC06084ED02B',
    'sufficiency_example_05': 'D19F8D01C79E3CCFCB4E8B9EFA6FD6412BA1EAEC469039EED30F957121778CC7',
    'sufficiency_example_06': 'BC3555BE520D323166478C46D74B2B48E6EBF8CF534A2352BFE045A20CB36CC4',
    'sufficiency_example_07': 'CA63C0448BE63DAF8FB5309DEF6155F0E32C2437041CFFF1E99D3B8AC127D6B7',
    'sufficiency_example_08': 'AF611963102E99816606F5BAEC0A88192A93BC1BCA9B0070BD999903C4924ACB',
    'sufficiency_example_09': 'D37383F3E9B8CAC090ECDA5D4855E4202B0038C8019BCD04D079A4FA7F24EFE1',
    'sufficiency_example_10': 'AC4FEF571248D8CA95F9DB55D58D948E3D215CD2CA45DFAE96D4A613C49D37CD',
}
LOO_BY_ID = {item[0]: item[1:] for item in LOO_IDENTITIES}

ENGINE.TRACEABILITY_FIELDS = (*INVENTORY_FIELDS, *ROW_PROVENANCE_FIELDS)
ENGINE.STATIC_METADATA_FIELDS = (*ENGINE.STATIC_METADATA_FIELDS, 'run_scope', 'runtime_engine_sha256')
ENGINE.RUNTIME_METADATA_FIELDS = (*ENGINE.STATIC_METADATA_FIELDS, 'generation_parameters', 'runtime_versions', 'device_name')
RESULT_FIELDS = (*ENGINE.TRACEABILITY_FIELDS, *ENGINE.RUNTIME_METADATA_FIELDS, *BASE.GATE_GENERATION_FIELDS)
ENGINE.RESULT_FIELDS = RESULT_FIELDS
PATHS = ENGINE.StatePaths(
    BASE_DIR / OUTPUT_FILE, BASE_DIR / '.05_defterm_qwen14_sufficiency_gate_fewshot_full_v2.lock',
    BASE_DIR / '.05_defterm_qwen14_sufficiency_gate_fewshot_full_v2.inflight.json',
)
CORE_VALIDATE_RESULT_ROW = ENGINE.validate_result_row


@dataclass(frozen=True)
class EffectiveContext:
    examples: tuple[Any, ...]
    retained_example_ids: tuple[str, ...]
    mode: str
    excluded_id: str
    prefix_hash: str

    def provenance(self) -> dict[str, str]:
        return {
            'gate_inference_mode': self.mode,
            'effective_fewshot_count': str(len(self.examples)),
            'excluded_self_example_id': self.excluded_id,
            'effective_fewshot_messages_sha256': self.prefix_hash,
        }


def effective_context(source: dict[str, str], examples: Sequence[dict[str, str]]) -> EffectiveContext:
    require(len(examples) == 10, 'Frozen example set must contain ten rows')
    matches = [row for row in examples if row['defterm_input_id'] == source['defterm_input_id']]
    require(len(matches) <= 1, 'Duplicate self example ID')
    excluded = ''
    if matches:
        own = matches[0]
        require(all(source[field] == own[field] for field in ('qid', 'term', 'defterm_evidence_text')), 'Self example identity/evidence disagrees with inventory')
        excluded = own['sufficiency_example_id']
    retained = [row for row in examples if row['defterm_input_id'] != source['defterm_input_id']]
    require(len(retained) == (9 if excluded else 10), 'Wrong effective example count')
    require(all(row['defterm_evidence_text'] != source['defterm_evidence_text'] for row in retained), 'Retained demonstration contains the target evidence')
    projected = ENGINE.project_examples(retained)
    prefix = ENGINE.build_demo_messages(projected)
    return EffectiveContext(
        projected, tuple(row['sufficiency_example_id'] for row in retained),
        LOO_MODE if excluded else FULL_MODE, excluded,
        ENGINE.sha256_text(canonical_json(prefix)),
    )


def expected_row_provenance(source: dict[str, str]) -> dict[str, str]:
    identity = LOO_BY_ID.get(source['defterm_input_id'])
    if identity is None:
        return {
            'gate_inference_mode': FULL_MODE, 'effective_fewshot_count': '10',
            'excluded_self_example_id': '',
            'effective_fewshot_messages_sha256': ENGINE.EXPECTED_FEWSHOT_MESSAGES_SHA256,
        }
    term, qid, example_id = identity
    require((source['term'], source['qid']) == (term, qid), 'Frozen LOO identity disagrees with inventory')
    prefix_hash = FROZEN_LOO_HASHES[example_id]
    require(len(prefix_hash) == 64 and all(char in '0123456789ABCDEF' for char in prefix_hash), 'LOO prefix hashes have not been frozen')
    return {
        'gate_inference_mode': LOO_MODE, 'effective_fewshot_count': '9',
        'excluded_self_example_id': example_id,
        'effective_fewshot_messages_sha256': prefix_hash,
    }


def prepare_targets(
    inventory: Sequence[dict[str, str]], examples: Sequence[dict[str, str]],
    *, check_frozen_hashes: bool = True,
) -> tuple[list[dict[str, str]], dict[str, EffectiveContext]]:
    targets = []
    contexts = {}
    for source in inventory:
        context = effective_context(source, examples)
        provenance = context.provenance()
        if check_frozen_hashes:
            require(provenance == expected_row_provenance(source), 'Effective prefix differs from frozen per-target specification')
        target = {**{field: source[field] for field in INVENTORY_FIELDS}, **provenance}
        require(tuple(target) == ENGINE.TRACEABILITY_FIELDS, 'Augmented target schema differs')
        targets.append(target)
        contexts[source['defterm_input_id']] = context
    return targets, contexts


def load_inputs(*, check_frozen_hashes: bool = True) -> tuple[Any, list[dict[str, str]], dict[str, EffectiveContext]]:
    ENGINE.validate_static_contract()
    inventory = read_inventory()
    examples, fewshot_hash = ENGINE.read_fixed_csv(BASE_DIR / FEWSHOT_FILE, ENGINE.FEWSHOT_FIELDS, FEWSHOT_SHA256)
    ENGINE.validate_examples(examples)
    require(tuple((row['defterm_input_id'], row['term'], row['qid'], row['sufficiency_example_id']) for row in examples) == LOO_IDENTITIES, 'Frozen demonstration identities/order changed')
    require(ENGINE.sha256_text(canonical_json(ENGINE.build_demo_messages(ENGINE.project_examples(examples)))) == ENGINE.EXPECTED_FEWSHOT_MESSAGES_SHA256, 'Frozen 10-shot prefix changed')
    by_id = {row['defterm_input_id']: row for row in inventory}
    for example in examples:
        require(example['defterm_input_id'] in by_id, 'A frozen example is absent from inventory')
        source = by_id[example['defterm_input_id']]
        require(all(source[field] == example[field] for field in ('qid', 'term', 'defterm_evidence_text')), 'Example and inventory disagree')
    targets, contexts = prepare_targets(inventory, examples, check_frozen_hashes=check_frozen_hashes)
    counts = Counter(context.mode for context in contexts.values())
    require(len(targets) == 150 and counts == {FULL_MODE: 140, LOO_MODE: 10}, 'Expected exactly 140 normal and ten leave-self-out targets')
    return ENGINE.Inputs(inventory, examples, INPUT_SHA256, fewshot_hash), targets, contexts


def expected_static_metadata(mode: str = 'PRODUCTION', *, input_hash: str = INPUT_SHA256) -> dict[str, str]:
    metadata = BASE.expected_static_metadata(mode, input_hash=input_hash)
    metadata['inference_script_sha256'] = sha256_bytes(Path(__file__).read_bytes())
    metadata['run_scope'] = RUN_SCOPE
    require(tuple(metadata) == ENGINE.STATIC_METADATA_FIELDS, 'V2 static metadata schema differs')
    return metadata


def validate_result_row(
    row: dict[str, str], source: dict[str, str], static: dict[str, str],
    runtime: dict[str, str] | None = None, *, allow_self_test: bool = False,
) -> None:
    expected = expected_row_provenance(source)
    require({field: row[field] for field in ROW_PROVENANCE_FIELDS} == expected, 'Result effective demonstration provenance is invalid')
    augmented = {**{field: source[field] for field in INVENTORY_FIELDS}, **expected}
    CORE_VALIDATE_RESULT_ROW(row, augmented, static, runtime, allow_self_test=allow_self_test)


ENGINE.validate_result_row = validate_result_row


def validate_gate_rows(
    inventory: list[dict[str, str]], rows: list[dict[str, str]], static: dict[str, str],
    *, allow_self_test: bool = False,
) -> None:
    require(len(rows) == len(inventory), 'Incomplete V2 gate results')
    require([row['defterm_input_id'] for row in rows] == [row['defterm_input_id'] for row in inventory], 'V2 results are not the exact inventory sequence')
    common = None
    for row, source in zip(rows, inventory):
        validate_result_row(row, source, static, common, allow_self_test=allow_self_test)
        require(row['generation_status'] == 'OK' and row['generation_call_status'] == 'RETURNED', 'An ERROR/uncertain V2 call cannot enter post-gate')
        common = {field: row[field] for field in ENGINE.RUNTIME_METADATA_FIELDS}


def write_inflight_marker(source: dict[str, str], metadata: dict[str, str], paths: Any = PATHS) -> None:
    require(not paths.inflight.exists(), 'Unresolved V2 attempt marker exists')
    require(tuple(metadata) == ENGINE.RUNTIME_METADATA_FIELDS, 'Marker runtime schema differs')
    prompt_provenance = {field: source[field] for field in ROW_PROVENANCE_FIELDS}
    require(prompt_provenance == expected_row_provenance(source), 'Marker effective prompt provenance differs')
    marker = {'defterm_input_id': source['defterm_input_id'], 'runtime_metadata': metadata, 'effective_prompt_provenance': prompt_provenance}
    ENGINE.write_bytes_atomically(paths.inflight, (canonical_json(marker) + '\n').encode('utf-8'))


def read_inflight_marker(paths: Any = PATHS) -> dict[str, Any] | None:
    if not paths.inflight.exists():
        return None
    payload = paths.inflight.read_bytes()
    require(payload.endswith(b'\n'), 'Incomplete V2 attempt marker')
    marker = json.loads(payload.decode('utf-8'))
    require(isinstance(marker, dict) and set(marker) == {'defterm_input_id', 'runtime_metadata', 'effective_prompt_provenance'}, 'Invalid V2 marker schema')
    require(type(marker['defterm_input_id']) is str and isinstance(marker['runtime_metadata'], dict), 'Invalid V2 marker identity/runtime')
    require(set(marker['runtime_metadata']) == set(ENGINE.RUNTIME_METADATA_FIELDS) and all(type(value) is str for value in marker['runtime_metadata'].values()), 'Invalid V2 marker runtime fields')
    require(isinstance(marker['effective_prompt_provenance'], dict) and set(marker['effective_prompt_provenance']) == set(ROW_PROVENANCE_FIELDS) and all(type(value) is str for value in marker['effective_prompt_provenance'].values()), 'Invalid V2 marker prompt fields')
    marker['runtime_metadata'] = {field: marker['runtime_metadata'][field] for field in ENGINE.RUNTIME_METADATA_FIELDS}
    return marker


def clear_inflight_marker(candidate_id: str, paths: Any = PATHS) -> None:
    marker = read_inflight_marker(paths)
    require(marker is not None and marker['defterm_input_id'] == candidate_id, 'V2 attempt marker changed')
    paths.inflight.unlink()
    ENGINE.sync_directory(paths.inflight.parent)


def recover_interrupted_attempt(
    targets: list[dict[str, str]], rows: list[dict[str, str]], static: dict[str, str],
    common: dict[str, str] | None, paths: Any = PATHS, *, allow_self_test: bool = False,
) -> tuple[list[dict[str, str]], dict[str, str] | None, bool]:
    marker = read_inflight_marker(paths)
    if marker is None:
        return rows, common, False
    candidate_id = marker['defterm_input_id']
    metadata = marker['runtime_metadata']
    if rows and rows[-1]['defterm_input_id'] == candidate_id:
        require({field: rows[-1][field] for field in ROW_PROVENANCE_FIELDS} == marker['effective_prompt_provenance'], 'Committed result and marker use different effective prefixes')
        require({field: rows[-1][field] for field in ENGINE.RUNTIME_METADATA_FIELDS} == metadata, 'Committed result and marker runtimes disagree')
        clear_inflight_marker(candidate_id, paths)
        return rows, common, False
    require(len(rows) < len(targets), 'Marker remains after all targets')
    source = targets[len(rows)]
    require(source['defterm_input_id'] == candidate_id, 'Marker is not for the next target')
    require({field: source[field] for field in ROW_PROVENANCE_FIELDS} == marker['effective_prompt_provenance'], 'Interrupted effective prefix is incompatible')
    if common is not None:
        require(common == metadata, 'Interrupted runtime is incompatible')
    result = ENGINE.interrupted_result(source, metadata)
    validate_result_row(result, source, static, metadata, allow_self_test=allow_self_test)
    recovered = [*rows, result]
    ENGINE.write_bytes_atomically(paths.output, ENGINE.serialize_result_rows(recovered))
    clear_inflight_marker(candidate_id, paths)
    return recovered, metadata, True


def synthetic_gate_rows(inventory: list[dict[str, str]], *, input_hash: str = INPUT_SHA256) -> list[dict[str, str]]:
    # Downstream self-test fixtures only, never invoked by default inference.
    runtime = ENGINE.synthetic_runtime_metadata(expected_static_metadata('SELF_TEST', input_hash=input_hash))
    targets = [{**source, **expected_row_provenance(source)} for source in inventory]
    return [ENGINE.process_candidate(source, None, None, runtime, lambda *_: ENGINE.SUFFICIENT_EVIDENCE if index % 2 == 0 else ENGINE.INSUFFICIENT_EVIDENCE) for index, source in enumerate(targets)]


def run_inference(targets: list[dict[str, str]], contexts: dict[str, EffectiveContext]) -> None:
    static = expected_static_metadata()
    lock = ENGINE.acquire_run_lock(PATHS.lock)
    try:
        rows, common = ENGINE.load_existing_results(targets, static, PATHS)
        rows, common, recovered = recover_interrupted_attempt(targets, rows, static, common, PATHS)
        ENGINE.require_no_recorded_errors(rows)
        print(f'ALREADY_PROCESSED={len(rows)}')
        print(f'PENDING={len(targets) - len(rows)}')
        print(f'INTERRUPTED_ATTEMPT_RECOVERED={ENGINE.bool_text(recovered)}')
        if len(rows) < len(targets):
            tokenizer, model, torch_module = ENGINE.load_model_and_tokenizer()
            runtime = ENGINE.collect_runtime_metadata(static, tokenizer, model, torch_module)
            if common is not None:
                require(common == runtime, 'Active runtime differs from partial V2 results')
            print(f"GENERATION_PARAMETERS={runtime['generation_parameters']}")
            print(f"RUNTIME_VERSIONS={runtime['runtime_versions']}")
            print(f"DEVICE_NAME={runtime['device_name']}")
            pending = targets[len(rows):]
            for source in pending:
                context = contexts[source['defterm_input_id']]
                require(context.provenance() == {field: source[field] for field in ROW_PROVENANCE_FIELDS}, 'Effective context changed before inference')
                write_inflight_marker(source, runtime, PATHS)

                def generator(tok: Any, mdl: Any, evidence: str) -> str:
                    # Exactly one call. There is no branch returning a CSV label.
                    return ENGINE.generate_once(tok, mdl, evidence, context.examples, torch_module)

                result = ENGINE.process_candidate(source, tokenizer, model, runtime, generator)
                validate_result_row(result, source, static, runtime)
                rows = [*rows, result]
                ENGINE.write_bytes_atomically(PATHS.output, ENGINE.serialize_result_rows(rows))
                clear_inflight_marker(source['defterm_input_id'], PATHS)
                require(result['generation_status'] == 'OK', 'ERROR committed; candidate will not be retried')
        verified, _ = ENGINE.load_existing_results(targets, static, PATHS)
        require(len(verified) == 150, 'Not all V2 targets have durable decisions')
        ENGINE.require_no_recorded_errors(verified)
        print(f'TOTAL_RESULTS={len(verified)}')
    finally:
        ENGINE.release_run_lock(lock)


def print_input_audit(inputs: Any, targets: Sequence[dict[str, str]]) -> None:
    counts = Counter(row['gate_inference_mode'] for row in targets)
    print('INPUT_VALIDATION=OK')
    print(f'FULL_INPUT_COUNT={len(inputs.sample)}')
    print(f'NORMAL_FULL_10_SHOT_COUNT={counts[FULL_MODE]}')
    print(f'LEAVE_SELF_OUT_9_SHOT_COUNT={counts[LOO_MODE]}')
    print(f'TOTAL_INFERENCE_TARGETS={len(targets)}')
    print('SELF_EXAMPLE_PRESENT_IN_OWN_PROMPT=0')
    print('FEWSHOT_LABEL_COPIED_TO_RESULT=0')
    print(f'SELF_EXCLUSION_POLICY={SELF_EXCLUSION_POLICY}')
    for key, value in expected_static_metadata().items():
        print(f'{key.upper()}={value}')
    print('LOO_AUDIT_TABLE_BEGIN')
    for row in targets:
        if row['gate_inference_mode'] == LOO_MODE:
            print(canonical_json({field: row[field] for field in ('defterm_input_id', 'term', 'qid', 'excluded_self_example_id', 'effective_fewshot_count', 'effective_fewshot_messages_sha256')}))
    print('LOO_AUDIT_TABLE_END')


def run_self_tests(inputs: Any, targets: list[dict[str, str]], contexts: dict[str, EffectiveContext]) -> None:
    ENGINE.assert_no_model_runtime()
    tests: dict[str, str] = {}

    def passed(name: str, condition: bool) -> None:
        require(condition, f'{name} failed')
        tests[name] = 'PASS'

    counts = Counter(row['gate_inference_mode'] for row in targets)
    passed('FULL_INPUT_COUNT_TEST', len(targets) == 150)
    passed('FEWSHOT_FULL_OVERLAP_COUNT_TEST', len({row['defterm_input_id'] for row in inputs.examples} & {row['defterm_input_id'] for row in inputs.sample}) == 10)
    passed('NORMAL_10SHOT_COUNT_TEST', counts[FULL_MODE] == 140)
    passed('LEAVE_SELF_OUT_COUNT_TEST', counts[LOO_MODE] == 10)
    full_prefix = ENGINE.build_demo_messages(ENGINE.project_examples(inputs.examples))
    normal = next(row for row in targets if row['gate_inference_mode'] == FULL_MODE)
    loo_targets = [row for row in targets if row['gate_inference_mode'] == LOO_MODE]
    order = tuple(row['sufficiency_example_id'] for row in inputs.examples)
    exclusions_ok = other_nine_ok = order_ok = own_label_hidden = True
    for source in loo_targets:
        context = contexts[source['defterm_input_id']]
        own_index = order.index(context.excluded_id)
        expected_prefix = full_prefix[:1 + 2 * own_index] + full_prefix[3 + 2 * own_index:]
        actual_prefix = ENGINE.build_demo_messages(context.examples)
        exclusions_ok &= context.excluded_id not in context.retained_example_ids and all(example.evidence != source['defterm_evidence_text'] for example in context.examples)
        other_nine_ok &= len(context.examples) == 9 and actual_prefix == expected_prefix
        order_ok &= context.retained_example_ids == tuple(example_id for example_id in order if example_id != context.excluded_id)
        poisoned = [{**row, 'sufficiency_label': 'FORBIDDEN_OWN_LABEL_SENTINEL'} if row['defterm_input_id'] == source['defterm_input_id'] else dict(row) for row in inputs.examples]
        poisoned_context = effective_context(source, poisoned)
        own_label_hidden &= poisoned_context == context and 'FORBIDDEN_OWN_LABEL_SENTINEL' not in canonical_json(ENGINE.build_messages(source['defterm_evidence_text'], poisoned_context.examples))
    passed('SELF_EXCLUSION_TEST', exclusions_ok)
    passed('OTHER_NINE_EXAMPLES_PRESERVED_TEST', other_nine_ok)
    passed('FEWSHOT_ORDER_PRESERVATION_TEST', order_ok)
    passed('OWN_LABEL_NOT_VISIBLE_TEST', own_label_hidden)
    passed('EFFECTIVE_FEWSHOT_HASH_TEST', all(context.prefix_hash == ENGINE.sha256_text(canonical_json(ENGINE.build_demo_messages(context.examples))) and context.provenance() == expected_row_provenance(source) for source in targets for context in [contexts[source['defterm_input_id']]]))
    pilot = load_frozen_engine('full_v2_identity_test')
    normal_context = contexts[normal['defterm_input_id']]
    passed('PROMPT_IDENTITY_TEST', ENGINE.build_messages(normal['defterm_evidence_text'], normal_context.examples) == pilot.build_messages(normal['defterm_evidence_text'], pilot.project_examples(inputs.examples)) and (SYSTEM_MSG, USER_MSG_TEMPLATE, FEWSHOT_USER_MSG_TEMPLATE) == (pilot.SYSTEM_MSG, pilot.USER_MSG_TEMPLATE, pilot.FEWSHOT_USER_MSG_TEMPLATE))
    passed('MODEL_PARAMETER_IDENTITY_TEST', MODEL_ID == pilot.MODEL_ID and MODEL_REVISION == pilot.MODEL_REVISION and ENGINE.generation_parameters_json((90, 91), 98) == pilot.generation_parameters_json((90, 91), 98) and ENGINE.quantization_config_json() == pilot.quantization_config_json())
    runtime = ENGINE.synthetic_runtime_metadata(expected_static_metadata('SELF_TEST'))
    fake_torch = SimpleNamespace(inference_mode=nullcontext)
    reference_by_id = {row['defterm_input_id']: row['sufficiency_label'] for row in inputs.examples}
    no_copy = single_call = True
    fake_results = []
    for source in [normal, *loo_targets]:
        context = contexts[source['defterm_input_id']]
        original_label = reference_by_id.get(source['defterm_input_id'], ENGINE.INSUFFICIENT_EVIDENCE)
        opposite = ENGINE.INSUFFICIENT_EVIDENCE if original_label == ENGINE.SUFFICIENT_EVIDENCE else ENGINE.SUFFICIENT_EVIDENCE
        model = ENGINE.FakeModel((60 if opposite == ENGINE.INSUFFICIENT_EVIDENCE else 50, 90))
        tokenizer = ENGINE.FakeTokenizer()

        def generator(tok: Any, mdl: Any, evidence: str) -> str:
            return ENGINE.generate_once(tok, mdl, evidence, context.examples, fake_torch, real_inference=False)

        result = ENGINE.process_candidate(source, tokenizer, model, runtime, generator)
        validate_result_row(result, source, expected_static_metadata('SELF_TEST'), runtime, allow_self_test=True)
        no_copy &= result['sufficiency_decision'] == opposite and result['sufficiency_decision'] != original_label
        single_call &= model.calls == 1 and result['generation_attempts'] == '1' and tokenizer.messages == ENGINE.build_messages(source['defterm_evidence_text'], context.examples)
        fake_results.append(result)
    passed('NO_DIRECT_LABEL_COPY_TEST', no_copy)
    passed('SINGLE_GENERATION_POLICY_TEST', single_call)
    for field in ('qid', 'term', 'defterm_evidence_text'):
        mismatched = {**loo_targets[0], field: 'INCONSISTENT_VALUE'}
        ENGINE.expect_validation_error(lambda source=mismatched: effective_context(source, inputs.examples))
    passed('SELF_IDENTITY_MISMATCH_REJECTION_TEST', True)
    parent = Path(tempfile.gettempdir()) / 'opencode'
    with tempfile.TemporaryDirectory(prefix='gate-loo-self-test-', dir=parent if parent.is_dir() else None) as name:
        source = loo_targets[0]
        paths = ENGINE.StatePaths(Path(name) / 'synthetic.csv', Path(name) / 'synthetic.lock', Path(name) / 'synthetic.inflight.json')
        static = expected_static_metadata('SELF_TEST')
        write_inflight_marker(source, runtime, paths)
        recovered, common, happened = recover_interrupted_attempt([source], [], static, None, paths, allow_self_test=True)
        ENGINE.expect_validation_error(lambda: ENGINE.require_no_recorded_errors(recovered))
        write_inflight_marker(source, runtime, paths)
        same, _, repeated = recover_interrupted_attempt([source], recovered, static, common, paths, allow_self_test=True)
        passed('NO_RETRY_POLICY_TEST', happened and not repeated and same == recovered and recovered[0]['sufficiency_decision'] == '' and recovered[0]['effective_fewshot_messages_sha256'] == source['effective_fewshot_messages_sha256'])
        write_inflight_marker(source, runtime, paths)
        marker = read_inflight_marker(paths)
        marker['effective_prompt_provenance']['effective_fewshot_count'] = '10'
        ENGINE.write_bytes_atomically(paths.inflight, (canonical_json(marker) + '\n').encode('utf-8'))
        ENGINE.expect_validation_error(lambda: recover_interrupted_attempt([source], [], static, None, paths, allow_self_test=True))
        passed('INTERRUPTED_PREFIX_PROVENANCE_REJECTION_TEST', True)
    passed('OUTPUT_SCHEMA_TEST', all(tuple(row) == RESULT_FIELDS for row in fake_results) and ENGINE.parse_csv_payload('synthetic.csv', ENGINE.serialize_result_rows(fake_results), RESULT_FIELDS) == fake_results)
    repeat_targets, repeat_contexts = prepare_targets(inputs.sample, inputs.examples)
    passed('DETERMINISM_TEST', repeat_targets == targets and repeat_contexts == contexts and all(effective_context(source, inputs.examples) == contexts[source['defterm_input_id']] for source in targets))
    ENGINE.assert_no_model_runtime()
    print('SELF_TEST_STATUS=SUCCESS')
    for name, status in tests.items():
        print(f'{name}={status}')
    print('SELF_TEST_SCOPE=LOCAL_CONTRACTS_AND_SYNTHETIC_MODEL_ONLY')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--validate-inputs', action='store_true')
    modes.add_argument('--audit-loo', action='store_true', help='local-only prefix audit; never runs inference')
    args = parser.parse_args(argv)
    inputs, targets, contexts = load_inputs(check_frozen_hashes=not args.audit_loo)
    print_input_audit(inputs, targets)
    if args.self_test or args.validate_inputs or args.audit_loo:
        ENGINE.assert_no_model_runtime()
        if args.self_test:
            run_self_tests(inputs, targets, contexts)
    else:
        run_inference(targets, contexts)
    print(f"MODEL_LOADED={'YES' if ENGINE.MODEL_LOADED else 'NO'}")
    print(f"INFERENCE_EXECUTED={'YES' if ENGINE.INFERENCE_EXECUTED else 'NO'}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f'FULL_GATE_V2_STATUS=FAILED: {exc}', file=sys.stderr)
        print(f"MODEL_LOADED={'YES' if ENGINE.MODEL_LOADED else 'NO'}", file=sys.stderr)
        print(f"INFERENCE_EXECUTED={'YES' if ENGINE.INFERENCE_EXECUTED else 'NO'}", file=sys.stderr)
        raise SystemExit(1) from exc
