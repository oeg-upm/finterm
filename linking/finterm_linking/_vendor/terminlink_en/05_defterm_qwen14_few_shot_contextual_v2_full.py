#!/usr/bin/env python3
"""Generate DefTerm only for a committed, upstream-verified sufficient partition.

Classification and definition remain separate tasks. The model receives only
the candidate's unchanged evidence and the six historical approved definition
examples. The frozen runtime provides locks, durable journals and checkpoints.
The approved DefTerm prompts are unchanged; there is one call and NO retry for
circularity. definition_clean normalizes whitespace only, never wording/content.
--self-test and --validate-prerequisites run no ML model or real pipeline stage.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import tempfile
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Sequence


BASE_DIR = Path(__file__).resolve().parent
POST_SCRIPT_FILE = '05_preparar_defterm_post_gate.py'


def load_post_gate() -> ModuleType:
    name = '_finterm_defterm_post_gate'
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / POST_SCRIPT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError('Post-gate script dependency is missing')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POST = load_post_gate()
GATE = POST.GATE
ENGINE = GATE.load_frozen_engine('defterm_contextual_v2')
ENGINE.ValidationError = GATE.ValidationError
ValidationError = GATE.ValidationError
require = GATE.require
sha256_bytes = GATE.sha256_bytes
canonical_json = GATE.canonical_json
INPUT_FILE = POST.SUFFICIENT_FILE
FEWSHOT_FILE = '05_fewshot_ejemplos_aprobados.csv'
FEWSHOT_SHA256 = 'AC83C1996824E1D2645FCBCC56FBA14282572498DA8000238C3A77D50B088026'
OUTPUT_FILE = '05_defterm_qwen14_few_shot_contextual_v2_full_results.csv'
HISTORICAL_SCRIPT_FILE = '05_defterm_qwen14_few_shot.py'
HISTORICAL_SCRIPT_SHA256 = 'C7A18C2B2836276B008E974A6E4C9A89C83F5D3A7DE346C186020ADC3119B3BC'
MODEL_ID = ENGINE.MODEL_ID
MODEL_REVISION = ENGINE.MODEL_REVISION
EXPERIMENT_TYPE = 'FEW_SHOT'
PROMPT_VERSION = 'defterm-en-few-v1'
RUN_SCOPE = 'DEFTERM_CONTEXTUAL_V2_FULL_AFTER_GATE'
FEWSHOT_QIDS = ('Q76161279', 'Q18845817', 'Q2986626', 'Q5358555', 'Q28659665', 'Q134100366')
FEWSHOT_FIELDS = (
    'fewshot_example_id', 'defterm_input_id', 'term', 'qid', 'label_en',
    'evidence_profile', 'evidence_sparsity', 'defterm_evidence_text',
    'manual_definition', 'manual_status', 'manual_notes',
)

SYSTEM_MSG = """You are an expert terminologist in the financial and economic domain.
Write ONE precise and concise intensional definition in English.
An intensional definition should begin with the immediate hypernym of the concept and continue with only the distinguishing characteristics needed to differentiate it from other concepts of the same class.
Use only the semantic evidence provided.
Do not add information that is not supported by the evidence.
Do not repeat the concept label as the subject of the definition.
Do not include examples, citations, references, URLs, explanations or commentary.
Write a single definition and do not reformulate or repeat it."""

USER_MSG_TEMPLATE = """Semantic evidence about the Wikidata candidate:

{defterm_evidence_text}

Write ONE terminological definition in English and nothing else.

Definition:"""

SYSTEM_SHA256 = '8045479E27C965AA1F17CB7F7BED109574B6211D407EE625CBCE4EAF31C3D1E8'
USER_SHA256 = '4DBBCCDA2ECC721CC551B8BDA65C029EED161492C5E66FD3FCCF5BF6A77F5AB0'
FEWSHOT_MESSAGES_SHA256 = 'FE3B348AEA04465821C0BA4AC6FF5A829783298A63B1D7E776DFAB3DD3F43539'


@dataclass(frozen=True)
class DefinitionExample:
    evidence: str
    definition: str


def project_examples(rows: Sequence[dict[str, str]]) -> tuple[DefinitionExample, ...]:
    return tuple(DefinitionExample(row['defterm_evidence_text'], row['manual_definition']) for row in rows)


def build_demo_messages(examples: Sequence[DefinitionExample]) -> list[dict[str, str]]:
    messages = [{'role': 'system', 'content': SYSTEM_MSG}]
    for example in examples:
        messages.extend((
            {'role': 'user', 'content': USER_MSG_TEMPLATE.format(defterm_evidence_text=example.evidence)},
            {'role': 'assistant', 'content': example.definition},
        ))
    return messages


def build_messages(evidence: str, examples: Sequence[DefinitionExample]) -> list[dict[str, str]]:
    return [*build_demo_messages(examples), {'role': 'user', 'content': USER_MSG_TEMPLATE.format(defterm_evidence_text=evidence)}]


ENGINE.SYSTEM_MSG = SYSTEM_MSG
ENGINE.USER_MSG_TEMPLATE = USER_MSG_TEMPLATE
ENGINE.FEWSHOT_USER_MSG_TEMPLATE = USER_MSG_TEMPLATE
ENGINE.EXPECTED_SYSTEM_PROMPT_SHA256 = SYSTEM_SHA256
ENGINE.EXPECTED_USER_PROMPT_TEMPLATE_SHA256 = USER_SHA256
ENGINE.EXPECTED_FEWSHOT_USER_TEMPLATE_SHA256 = USER_SHA256
ENGINE.EXPERIMENT_TYPE = EXPERIMENT_TYPE
ENGINE.PROMPT_VERSION = PROMPT_VERSION
ENGINE.MAX_NEW_TOKENS = 150
ENGINE.build_demo_messages = build_demo_messages
ENGINE.build_messages = build_messages
ENGINE.TRACEABILITY_FIELDS = POST.PARTITION_FIELDS
ENGINE.STATIC_METADATA_FIELDS = (
    *ENGINE.STATIC_METADATA_FIELDS, 'run_scope', 'runtime_engine_sha256',
    'historical_prompt_script_sha256', 'post_gate_manifest_sha256',
)
ENGINE.RUNTIME_METADATA_FIELDS = (*ENGINE.STATIC_METADATA_FIELDS, 'generation_parameters', 'runtime_versions', 'device_name')
DEFINITION_FIELDS = (
    'generation_attempts', 'generation_call_status', 'definition_raw', 'definition_clean',
    'circularity_flag_final', 'generation_status', 'error_message',
)
RESULT_FIELDS = (*POST.PARTITION_FIELDS, *ENGINE.RUNTIME_METADATA_FIELDS, *DEFINITION_FIELDS)
ENGINE.RESULT_FIELDS = RESULT_FIELDS
PATHS = ENGINE.StatePaths(
    BASE_DIR / OUTPUT_FILE, BASE_DIR / '.05_defterm_qwen14_few_shot_contextual_v2_full.lock',
    BASE_DIR / '.05_defterm_qwen14_few_shot_contextual_v2_full.inflight.json',
)


def load_examples() -> list[dict[str, str]]:
    ENGINE.validate_static_contract()
    rows, _ = ENGINE.read_fixed_csv(BASE_DIR / FEWSHOT_FILE, FEWSHOT_FIELDS, FEWSHOT_SHA256)
    require(len(rows) == 6 and tuple(row['qid'] for row in rows) == FEWSHOT_QIDS, 'Historical six definition examples/order changed')
    ENGINE.validate_unique_candidates(rows, FEWSHOT_FILE)
    require(len({row['fewshot_example_id'] for row in rows}) == 6, 'Duplicate definition example ID')
    for row in rows:
        require(re.fullmatch(r'fewshot_example_[0-9a-f]{64}', row['fewshot_example_id']) is not None, 'Invalid definition example ID')
        require(row['manual_status'] == 'APPROVED_MANUAL' and bool(row['manual_definition'].strip()), 'Definition example is not approved')
    require(ENGINE.sha256_text(canonical_json(build_demo_messages(project_examples(rows)))) == FEWSHOT_MESSAGES_SHA256, 'Historical DefTerm message prefix changed')
    return rows


def ensure_sufficient(source: dict[str, str]) -> None:
    # Repeat the last-mile guard immediately before journaling/generation too.
    require(source.get('sufficiency_decision') == GATE.ENGINE.SUFFICIENT_EVIDENCE, 'INSUFFICIENT or unclassified row cannot reach DefTerm')
    require(bool(source.get('sufficiency_gate_result_id')) and bool(source.get('post_gate_partition_id')), 'Missing upstream gate/partition provenance')


def normalize_definition(raw: str) -> str:
    require(isinstance(raw, str), 'Definition output is not text')
    # No first-sentence truncation, removed commentary, substitutions or repair.
    return re.sub(r'\s+', ' ', raw).strip()


def circularity_flag(definition: str, label: str) -> bool:
    def tokens(text: str) -> list[str]:
        values = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", unicodedata.normalize('NFKC', text).casefold())
        return values[1:] if values and values[0] in {'a', 'an', 'the'} else values
    words, label_words = tokens(definition), tokens(label)
    return bool(label_words and words[:len(label_words)] == label_words)


def process_candidate(
    source: dict[str, str], tokenizer: Any, model: Any, metadata: dict[str, str],
    generator: Callable[[Any, Any, str], str],
) -> dict[str, str]:
    ensure_sufficient(source)
    require(tuple(metadata) == ENGINE.RUNTIME_METADATA_FIELDS, 'Definition runtime metadata schema differs')
    raw = clean = error = ''
    status = 'ERROR'
    call_status = 'FAILED_OR_UNKNOWN'
    circular = False
    try:
        raw = generator(tokenizer, model, source['defterm_evidence_text'])
        call_status = 'RETURNED'
        clean = normalize_definition(raw)
        require(bool(clean), 'Empty definition')
        require(clean not in GATE.ENGINE.VALID_DECISIONS, 'Classifier label is not a definition')
        require('<think' not in clean.casefold() and '<|' not in clean, 'Reasoning/special-token residue in definition')
        circular = circularity_flag(clean, source['label_en'])
        # Circularity is recorded only: no second call and no rewritten wording.
        status = 'OK'
    except Exception as exc:
        detail = re.sub(r'\s+', ' ', str(exc)).strip()
        error = f"{type(exc).__name__}: {detail}"
    return {
        **{field: source[field] for field in POST.PARTITION_FIELDS}, **metadata,
        'generation_attempts': '1', 'generation_call_status': call_status,
        'definition_raw': raw, 'definition_clean': clean,
        'circularity_flag_final': ENGINE.bool_text(circular), 'generation_status': status,
        'error_message': error,
    }


def interrupted_result(source: dict[str, str], metadata: dict[str, str]) -> dict[str, str]:
    ensure_sufficient(source)
    return {
        **{field: source[field] for field in POST.PARTITION_FIELDS}, **metadata,
        'generation_attempts': '1', 'generation_call_status': 'INTERRUPTED_OR_UNKNOWN',
        'definition_raw': '', 'definition_clean': '', 'circularity_flag_final': 'false',
        'generation_status': 'ERROR',
        'error_message': 'InterruptedGeneration: durable marker without committed definition; the candidate was not retried',
    }


def validate_result_row(
    row: dict[str, str], source: dict[str, str], static: dict[str, str],
    runtime: dict[str, str] | None = None, *, allow_self_test: bool = False,
) -> None:
    ensure_sufficient(source)
    require(tuple(row) == RESULT_FIELDS, 'Definition result schema differs')
    require(all(row[field] == source[field] for field in POST.PARTITION_FIELDS), 'Definition evidence/gate provenance changed')
    GATE.validate_technical_metadata(ENGINE, row, static, runtime, allow_self_test=allow_self_test)
    require(row['generation_attempts'] == '1', 'Definition attempt budget differs')
    require(row['generation_call_status'] in {'RETURNED', 'FAILED_OR_UNKNOWN', 'INTERRUPTED_OR_UNKNOWN'}, 'Invalid definition call status')
    require(row['generation_status'] in {'OK', 'ERROR'}, 'Invalid definition status')
    require(row['circularity_flag_final'] in {'true', 'false'}, 'Invalid circularity flag')
    require(row['definition_clean'] == normalize_definition(row['definition_raw']), 'Definition was semantically modified')
    if row['generation_status'] == 'OK':
        require(row['generation_call_status'] == 'RETURNED' and bool(row['definition_clean']) and row['error_message'] == '', 'Inconsistent successful definition')
        require(row['definition_clean'] not in GATE.ENGINE.VALID_DECISIONS, 'Classification output reached definition results')
        require('<think' not in row['definition_clean'].casefold() and '<|' not in row['definition_clean'], 'Invalid token/thinking residue')
        require(row['circularity_flag_final'] == ENGINE.bool_text(circularity_flag(row['definition_clean'], source['label_en'])), 'Circularity audit differs')
    else:
        require(bool(row['error_message'].strip()), 'Failed definition lacks error details')


ENGINE.validate_result_row = validate_result_row
ENGINE.interrupted_result = interrupted_result


def static_metadata(
    examples: list[dict[str, str]], input_hash: str, manifest_hash: str,
    mode: str = 'PRODUCTION',
) -> dict[str, str]:
    metadata = {
        'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'experiment_type': EXPERIMENT_TYPE, 'metadata_mode': mode,
        'prompt_version': PROMPT_VERSION, 'input_sha256': input_hash,
        'fewshot_sha256': FEWSHOT_SHA256, 'fewshot_count': '6',
        'fewshot_class_balance': canonical_json({'APPROVED_MANUAL_DEFINITION': 6}),
        'fewshot_example_order': canonical_json([row['fewshot_example_id'] for row in examples]),
        'system_prompt_sha256': SYSTEM_SHA256, 'user_prompt_template_sha256': USER_SHA256,
        'fewshot_user_template_sha256': USER_SHA256,
        'fewshot_messages_sha256': ENGINE.sha256_text(canonical_json(build_demo_messages(project_examples(examples)))),
        'inference_script_sha256': sha256_bytes(Path(__file__).read_bytes()),
        'quantization_config': ENGINE.quantization_config_json(), 'run_scope': RUN_SCOPE,
        'runtime_engine_sha256': GATE.PILOT_SHA256,
        'historical_prompt_script_sha256': HISTORICAL_SCRIPT_SHA256,
        'post_gate_manifest_sha256': manifest_hash,
    }
    require(tuple(metadata) == ENGINE.STATIC_METADATA_FIELDS, 'Definition static schema differs')
    return metadata


def load_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    examples = load_examples()
    plan, manifest_hash = POST.load_committed_partition()
    require(plan.manifest['metadata_mode'] == 'PRODUCTION', 'Synthetic partition cannot be used for real inference')
    for source in plan.sufficient:
        ensure_sufficient(source)
    return plan.sufficient, examples, static_metadata(examples, plan.manifest['sufficient_sha256'], manifest_hash)


def validate_historical_prompt_identity() -> None:
    # Parse literals only. Importing the old script would import Torch eagerly.
    payload = (BASE_DIR / HISTORICAL_SCRIPT_FILE).read_bytes()
    require(sha256_bytes(payload) == HISTORICAL_SCRIPT_SHA256, 'Historical DefTerm source changed')
    values = {}
    for node in ast.parse(payload).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'SYSTEM_MSG', 'USER_MSG_TEMPLATE', 'PROMPT_VERSION', 'MODEL_ID'}:
                    values[target.id] = ast.literal_eval(node.value)
    require(values == {'SYSTEM_MSG': SYSTEM_MSG, 'USER_MSG_TEMPLATE': USER_MSG_TEMPLATE, 'PROMPT_VERSION': PROMPT_VERSION, 'MODEL_ID': MODEL_ID}, 'Approved DefTerm prompt identity differs')


class DefinitionFakeTokenizer(ENGINE.FakeTokenizer):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = output

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        require(list(tokens) == [50] and kwargs == {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}, 'Definition decoding changed')
        return self.output


def run_self_tests(examples: list[dict[str, str]]) -> None:
    ENGINE.assert_no_model_runtime()
    tests: dict[str, str] = {}

    def passed(name: str, condition: bool) -> None:
        require(condition, f'{name} failed')
        tests[name] = 'PASS'

    validate_historical_prompt_identity()
    passed('DEFTERM_PROMPT_IDENTITY_TEST', True)
    passed('DEFTERM_FEWSHOT_COUNT_TEST', len(examples) == 6)
    projected = project_examples(examples)
    serialized = canonical_json(build_demo_messages(projected))
    passed('DEFTERM_GATE_FEWSHOT_SEPARATION_TEST', FEWSHOT_FILE != GATE.FEWSHOT_FILE and FEWSHOT_SHA256 != GATE.FEWSHOT_SHA256 and not any(label in serialized for label in GATE.ENGINE.VALID_DECISIONS))
    parent = Path(tempfile.gettempdir()) / 'opencode'
    with tempfile.TemporaryDirectory(prefix='defterm-full-self-test-', dir=parent if parent.is_dir() else None) as name:
        plan, partition_paths, inventory_path, gate_path = POST.synthetic_fixture(Path(name))
        POST.publish_plan(plan, partition_paths)
        committed, manifest_hash = POST.load_committed_partition(partition_paths, inventory_path, gate_path, allow_self_test=True)
        source = committed.sufficient[0]
        static = static_metadata(examples, committed.manifest['sufficient_sha256'], manifest_hash, 'SELF_TEST')
        runtime = ENGINE.synthetic_runtime_metadata(static)
        model = ENGINE.FakeModel()
        output = f"{source['label_en']} is a synthetic class instance."
        tokenizer = DefinitionFakeTokenizer(output)
        fake_torch = SimpleNamespace(inference_mode=nullcontext)

        def generator(tok: Any, mdl: Any, evidence: str) -> str:
            return ENGINE.generate_once(tok, mdl, evidence, projected, fake_torch, real_inference=False)

        result = process_candidate(source, tokenizer, model, runtime, generator)
        validate_result_row(result, source, static, runtime, allow_self_test=True)
        passed('SINGLE_GENERATION_POLICY_TEST', model.calls == 1 and result['generation_attempts'] == '1' and result['circularity_flag_final'] == 'true' and result['definition_raw'] == output)
        bad = committed.insufficient[0]
        untouched = ENGINE.FakeModel()
        ENGINE.expect_validation_error(lambda: process_candidate(bad, tokenizer, untouched, runtime, generator))
        wrong = [*committed.sufficient, {**bad, 'sufficiency_decision': GATE.ENGINE.SUFFICIENT_EVIDENCE}]
        ENGINE.write_bytes_atomically(partition_paths.sufficient, POST.serialize_csv(wrong, POST.PARTITION_FIELDS))
        ENGINE.expect_validation_error(lambda: POST.load_committed_partition(partition_paths, inventory_path, gate_path, allow_self_test=True))
        ENGINE.write_bytes_atomically(partition_paths.sufficient, committed.sufficient_payload)
        passed('NO_INSUFFICIENT_ROW_CAN_REACH_DEFTERM_TEST', untouched.calls == 0)
        passed('DEFTERM_EVIDENCE_IMMUTABILITY_TEST', result['defterm_evidence_text'] == source['defterm_evidence_text'] and all(result[field] == source[field] for field in POST.PARTITION_FIELDS))
        rich = {**source, 'context': 'FORBIDDEN_CONTEXT_SENTINEL', 'sense1': 'FORBIDDEN_SENSE1_SENTINEL', 'embeddings': 'FORBIDDEN_EMBEDDINGS_SENTINEL', 'TermiLink': 'FORBIDDEN_TERMILINK_SENTINEL'}
        expected_messages = build_messages(source['defterm_evidence_text'], projected)
        text = canonical_json(expected_messages)
        passed('NO_SENSE1_CONTAMINATION_TEST', tokenizer.messages == expected_messages and build_messages(rich['defterm_evidence_text'], projected) == expected_messages and all(rich[field] not in text for field in ('context', 'sense1', 'embeddings', 'TermiLink')) and len(expected_messages) == 14)
        failed_model = ENGINE.FakeModel(fail=True)
        failed = process_candidate(source, DefinitionFakeTokenizer(output), failed_model, runtime, generator)
        paths = ENGINE.StatePaths(Path(name) / 'synthetic_definitions.csv', Path(name) / 'synthetic_def.lock', Path(name) / 'synthetic_def.inflight.json')
        ENGINE.write_inflight_marker(source, runtime, paths)
        recovered, common, happened = ENGINE.recover_interrupted_attempt([source], [], static, None, paths, allow_self_test=True)
        ENGINE.expect_validation_error(lambda: ENGINE.require_no_recorded_errors(recovered))
        ENGINE.write_inflight_marker(source, runtime, paths)
        same, _, repeated = ENGINE.recover_interrupted_attempt([source], recovered, static, common, paths, allow_self_test=True)
        passed('NO_RETRY_POLICY_TEST', failed_model.calls == 1 and failed['generation_status'] == 'ERROR' and happened and not repeated and same == recovered and recovered[0]['definition_raw'] == '')
        for raw in ('  A\n synthetic\tdefinition.  ', 'One sentence. A second sentence.', 'Definition: Original wording.'):
            clean = normalize_definition(raw)
            require(raw.split() == clean.split(), 'Definition words were added, removed or changed')
        passed('NO_SEMANTIC_REPAIR_TEST', True)
        passed('OUTPUT_SCHEMA_TEST', tuple(result) == RESULT_FIELDS and ENGINE.parse_csv_payload('synthetic.csv', ENGINE.serialize_result_rows([result]), RESULT_FIELDS) == [result])
    ENGINE.assert_no_model_runtime()
    print('SELF_TEST_STATUS=SUCCESS')
    for name, status in tests.items():
        print(f'{name}={status}')
    print('SELF_TEST_SCOPE=SYNTHETIC_TEMPORARY_FILES_AND_FROZEN_PROMPTS_ONLY')


def print_prerequisites(examples: list[dict[str, str]]) -> None:
    print('INPUT_VALIDATION=STATIC_PREREQUISITES_OK')
    print(f'DEFTERM_FEWSHOT_COUNT={len(examples)}')
    print(f'DEFTERM_FEWSHOT_FILE={FEWSHOT_FILE}')
    print(f'DEFTERM_FEWSHOT_SHA256={FEWSHOT_SHA256}')
    print(f'POST_GATE_INPUT_AVAILABLE={ENGINE.bool_text((BASE_DIR / INPUT_FILE).is_file())}')
    print(f'SYSTEM_PROMPT_SHA256={SYSTEM_SHA256}')
    print(f'USER_PROMPT_TEMPLATE_SHA256={USER_SHA256}')
    print(f'FEWSHOT_MESSAGES_SHA256={ENGINE.sha256_text(canonical_json(build_demo_messages(project_examples(examples))))}')
    parameters = json.loads(ENGINE.generation_parameters_json((90, 91), 98))
    parameters['eos_token_ids'] = 'RESOLVED_FROM_PINNED_MODEL_AT_INFERENCE'
    parameters['pad_token_id'] = 'RESOLVED_FROM_PINNED_MODEL_AT_INFERENCE'
    print(f'GENERATION_PARAMETER_TEMPLATE={canonical_json(parameters)}')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--validate-prerequisites', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test or args.validate_prerequisites:
        ENGINE.assert_no_model_runtime()
        examples = load_examples()
        if args.self_test:
            run_self_tests(examples)
        print_prerequisites(examples)
        ENGINE.assert_no_model_runtime()
        print('MODEL_LOADED=NO')
        print('INFERENCE_EXECUTED=NO')
    else:
        rows, examples, static = load_inputs()
        print(f'SUFFICIENT_INPUT_COUNT={len(rows)}')
        print('INPUT_VALIDATION=OK')
        for key, value in static.items():
            print(f'{key.upper()}={value}')
        GATE.run_checkpointed_stage(ENGINE, rows, project_examples(examples), static, PATHS, process_candidate, ensure_sufficient)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f'DEFTERM_FULL_STATUS=FAILED: {exc}', file=sys.stderr)
        print(f"MODEL_LOADED={'YES' if ENGINE.MODEL_LOADED else 'NO'}", file=sys.stderr)
        print(f"INFERENCE_EXECUTED={'YES' if ENGINE.INFERENCE_EXECUTED else 'NO'}", file=sys.stderr)
        raise SystemExit(1) from exc
