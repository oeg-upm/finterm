#!/usr/bin/env python3
"""Consolidate the frozen contextual V2 Sense 2 artifacts without any model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = '03_sentido2_estado_candidatos_contextual_v2.csv'
TEXT_FILE = '03_sentidos2_textuales_contextual_v2.csv'
NO_FILE = '03_sin_sentido2_contextual_v2.csv'
DEFTERM_FILE = '05_defterm_qwen14_few_shot_contextual_v2_full_results.csv'
INSUFFICIENT_FILE = '05_defterm_insufficient_contextual_v2.csv'
POST_MANIFEST_FILE = '05_defterm_post_gate_manifest.json'
EXPECTED_INPUT_HASHES = {
    STATE_FILE: '242DB39D193BB6B8E890F539FA96CA9859518D4A4E4391C9824D9E39CA55ADB2',
    TEXT_FILE: '008C061DAF4659AD29A32163E51723312A256AFFFFED32040BEEB29229EE2ACF',
    NO_FILE: 'DA4BCA8B69200AC82037101262E2954F132A79ABFFAB9BF88765AACC4AF39D5B',
    DEFTERM_FILE: '998B671076AA4F97AC88E422AEB448CFD3ADDB1F038582CDC088BECA3B6F85F1',
    INSUFFICIENT_FILE: '43881429BE56893A0F7CB657B2DC4F9439DB809CE5A6F91F3D20866A20763293',
    POST_MANIFEST_FILE: 'E82AEC51E5B4A12D461E1A2AC97117450709143C8C662347B2F37CC3747E3083',
}
SCHEMA_VERSION = 'sentido2-final-contextual-v2-operational-v1'
DEFTERM_ROUTE = 'DEFTERM_QWEN3_14B_FEW_SHOT'
STATE_OUTPUT = '05_sentido2_estado_final.csv'
REP_OUTPUT = '05_sentidos2_representaciones_final.csv'
NO_OUTPUT = '05_sin_sentido2_final.csv'
MANIFEST_OUTPUT = '05_sentido2_final_manifest.json'
OUTPUT_FILES = (STATE_OUTPUT, REP_OUTPUT, NO_OUTPUT, MANIFEST_OUTPUT)
LOCK_PATH = BASE_DIR / '.05_consolidar_sentidos2_final.lock'
JOURNAL_PATH = BASE_DIR / '.05_consolidar_sentidos2_final.inflight.json'
INITIAL_STATES = {
    'TEXTUAL_EN_READY': 2987, 'TEXTUAL_MULTILINGUAL_READY': 299,
    'DEFTERM_PENDING': 150, 'NO_REPRESENTATION_EVIDENCE': 38,
}
FINAL_STATES = {
    'TEXTUAL_EN_READY': 2987, 'TEXTUAL_MULTILINGUAL_READY': 299, 'DEFTERM_READY': 63,
    'NO_SENSE2_GATE_INSUFFICIENT': 87, 'NO_REPRESENTATION_EVIDENCE': 38,
}
USABLE_STATES = frozenset({'TEXTUAL_EN_READY', 'TEXTUAL_MULTILINGUAL_READY', 'DEFTERM_READY'})
STATE_SOURCE_FIELDS = (
    'term', 'qid', 'label_en', 'representation_route', 'total_textual_representation_count',
    'usable_textual_representation_count', 'contextual_translation_count',
    'english_variant_count', 'translation_error_count', 'sense2_status',
)
TEXT_FIELDS = (
    'sense2_representation_id', 'term', 'qid', 'label_en', 'sense2_text_en',
    'representation_source', 'source_language', 'source_description', 'representation_id',
    'translated_description_en', 'translation_model', 'translation_model_revision',
    'translation_status', 'semantic_translation_validation', 'english_word_count',
    'textual_representation_usable',
)
NO_SOURCE_FIELDS = ('term', 'qid', 'label_en', 'representation_route', 'translation_error_count', 'sense2_status')
INVENTORY_FIELDS = (
    'defterm_input_id', 'term', 'qid', 'label_en', 'aliases_en', 'p31', 'p279', 'p1343',
    'representation_route', 'has_aliases', 'has_p31', 'has_p279', 'has_p1343',
    'aliases_count', 'p31_count', 'p279_count', 'p1343_count', 'evidence_profile',
    'evidence_sparsity', 'defterm_evidence_text',
)
GATE_PROVENANCE_FIELDS = (
    'sufficiency_decision', 'sufficiency_gate_model', 'sufficiency_gate_model_revision',
    'sufficiency_gate_experiment_type', 'sufficiency_gate_prompt_version',
    'sufficiency_gate_result_id', 'sufficiency_gate_results_file',
    'sufficiency_gate_results_sha256', 'sufficiency_gate_result_row_sha256',
    'sufficiency_gate_script_sha256', 'sufficiency_gate_runtime_engine_sha256',
    'sufficiency_gate_input_sha256', 'sufficiency_gate_fewshot_sha256',
    'gate_inference_mode', 'effective_fewshot_count', 'excluded_self_example_id',
    'effective_fewshot_messages_sha256', 'post_gate_partition_id',
    'post_gate_preparer_script_sha256',
)
PARTITION_FIELDS = (*INVENTORY_FIELDS, *GATE_PROVENANCE_FIELDS)
DEFTERM_FIELDS = (
    *PARTITION_FIELDS, 'model_id', 'model_revision', 'experiment_type', 'metadata_mode',
    'prompt_version', 'input_sha256', 'fewshot_sha256', 'fewshot_count',
    'fewshot_class_balance', 'fewshot_example_order', 'system_prompt_sha256',
    'user_prompt_template_sha256', 'fewshot_user_template_sha256', 'fewshot_messages_sha256',
    'inference_script_sha256', 'quantization_config', 'run_scope', 'runtime_engine_sha256',
    'historical_prompt_script_sha256', 'post_gate_manifest_sha256', 'generation_parameters',
    'runtime_versions', 'device_name', 'generation_attempts', 'generation_call_status',
    'definition_raw', 'definition_clean', 'circularity_flag_final', 'generation_status', 'error_message',
)
LINEAGE_FIELDS = (
    'source_input_file', 'source_input_sha256', 'source_input_row_number',
    'source_input_row_sha256', 'consolidation_version',
)
STATE_FIELDS = (
    *STATE_SOURCE_FIELDS[:-1], 'source_candidate_state',
    'candidate_state_final', 'has_usable_sense2', 'sense2_representation_count',
    'primary_representation_route', *LINEAGE_FIELDS,
    *tuple(field for field in DEFTERM_FIELDS if field not in STATE_SOURCE_FIELDS),
    'source_pipeline_input_file', 'source_pipeline_input_sha256', 'source_pipeline_row_sha256',
    'source_no_sense2_row_sha256',
)
REP_FIELDS = (
    *TEXT_FIELDS, 'sense2_text', 'sense2_route', 'candidate_state_final',
    'representation_usable', *LINEAGE_FIELDS,
    *tuple(field for field in DEFTERM_FIELDS if field not in TEXT_FIELDS),
)
NO_FIELDS = (*STATE_FIELDS, 'no_sense2_reason')
FORBIDDEN_FIELDS = frozenset({
    'sense1', 'sense_1', 'sentido1', 'sentido_1', 'context', 'corpus_context', 'context_text',
    'occurrence', 'occurrences', 'embedding', 'embeddings', 'similarity', 'similarities',
    'cosine', 'cosine_similarity', 'termilink', 'terminlink', 'gold', 'gold_qid',
})
MODEL_MODULES = frozenset({
    'torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface_hub',
    'sentence_transformers', 'ctranslate2', 'llama_cpp', 'onnxruntime',
})
INITIAL_MODEL_MODULES = {name.partition('.')[0] for name in sys.modules} & MODEL_MODULES


@dataclass
class Consolidation:
    states: list[dict[str, str]]
    representations: list[dict[str, str]]
    without_sense2: list[dict[str, str]]
    statistics: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def key(row: dict[str, str]) -> tuple[str, str]:
    return row['term'], row['qid']


def row_hash(row: dict[str, str]) -> str:
    return sha256_bytes(canonical_json(row).encode('utf-8'))


def serialize_csv(rows: Sequence[dict[str, str]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def assert_no_model_runtime() -> None:
    loaded = {name.partition('.')[0] for name in sys.modules} & MODEL_MODULES
    require(not (loaded - INITIAL_MODEL_MODULES), 'Consolidation imported an ML runtime')


def validate_no_sense1_fields(fields: Sequence[str]) -> None:
    require(not FORBIDDEN_FIELDS.intersection(field.casefold() for field in fields), 'Sense 1/corpus/model-selection field is forbidden')


class ValidationError(RuntimeError):
    """A frozen input or consolidation contract is inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def resolve_input(filename: str) -> Path:
    path = BASE_DIR / filename
    if path.is_file():
        return path
    # The real downloaded DefTerm artifact is read directly and hash-verified;
    # never regenerate, edit, replace, or infer definitions to fill a missing file.
    if filename == DEFTERM_FILE:
        downloaded = Path.home() / 'Downloads' / filename
        if downloaded.is_file():
            return downloaded
    raise ValidationError(f'Required input is missing: {filename}')


def parse_csv(filename: str, payload: bytes) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    text = payload.decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    fields = tuple(reader.fieldnames or ())
    require(bool(fields) and len(fields) == len(set(fields)), f'{filename}: invalid header')
    rows = []
    for row in reader:
        require(None not in row and all(value is not None for value in row.values()), f'{filename}: malformed record near line {reader.line_num}')
        rows.append({field: row[field] for field in fields})
    return fields, rows


def read_inputs() -> tuple[dict[str, Any], dict[str, tuple[str, ...]], dict[str, Path]]:
    datasets = {}
    schemas = {}
    locations = {}
    for filename, expected in EXPECTED_INPUT_HASHES.items():
        path = resolve_input(filename)
        payload = path.read_bytes()
        require(sha256_bytes(payload) == expected, f'{filename}: input SHA-256 differs')
        locations[filename] = path
        if filename.endswith('.json'):
            datasets[filename] = json.loads(payload.decode('utf-8'))
        else:
            schemas[filename], datasets[filename] = parse_csv(filename, payload)
    return datasets, schemas, locations


def inspect_inputs(datasets: dict[str, Any], locations: dict[str, Path]) -> None:
    states = datasets[STATE_FILE]
    representations = datasets[TEXT_FILE]
    state_by_key = {(row['term'], row['qid']): row for row in states}
    usable = [row for row in representations if row['textual_representation_usable'] == 'true']
    unusable = [row for row in representations if row['textual_representation_usable'] == 'false']
    require(len(usable) + len(unusable) == len(representations), 'Invalid textual usability flag')
    diagnostics = {
        'candidate_count': len(states),
        'candidate_states': dict(Counter(row['sense2_status'] for row in states)),
        'textual_rows_total': len(representations),
        'textual_rows_usable_true': len(usable),
        'textual_rows_usable_false': len(unusable),
        'all_textual_routes': dict(Counter(row['representation_source'] for row in representations)),
        'usable_textual_routes': dict(Counter(row['representation_source'] for row in usable)),
        'textual_rows_by_candidate_state': dict(Counter(state_by_key[(row['term'], row['qid'])]['sense2_status'] for row in representations)),
        'unusable_textual_rows_by_candidate_state': dict(Counter(state_by_key[(row['term'], row['qid'])]['sense2_status'] for row in unusable)),
        'candidates_with_any_textual_row': len({(row['term'], row['qid']) for row in representations}),
        'candidates_with_usable_textual_row': len({(row['term'], row['qid']) for row in usable}),
        'defterm_rows': len(datasets[DEFTERM_FILE]),
        'insufficient_rows': len(datasets[INSUFFICIENT_FILE]),
        'defterm_statuses': dict(Counter(row['generation_status'] for row in datasets[DEFTERM_FILE])),
        'defterm_results_location': str(locations[DEFTERM_FILE]),
        'retained_total_with_defterm': len(representations) + len(datasets[DEFTERM_FILE]),
        'usable_total_with_defterm': len(usable) + len(datasets[DEFTERM_FILE]),
    }
    print(json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, indent=2))
    print('MODEL_LOADED=NO')
    print('INFERENCE_EXECUTED=NO')


def validate_inputs(datasets: dict[str, Any], schemas: dict[str, tuple[str, ...]]) -> None:
    expected_schemas = {
        STATE_FILE: STATE_SOURCE_FIELDS, TEXT_FILE: TEXT_FIELDS, NO_FILE: NO_SOURCE_FIELDS,
        DEFTERM_FILE: DEFTERM_FIELDS, INSUFFICIENT_FILE: PARTITION_FIELDS,
    }
    require(schemas == expected_schemas, 'An input schema/order differs from the frozen pipeline')
    for fields in schemas.values():
        validate_no_sense1_fields(fields)
    states = datasets[STATE_FILE]
    texts = datasets[TEXT_FILE]
    definitions = datasets[DEFTERM_FILE]
    insufficient = datasets[INSUFFICIENT_FILE]
    absent = datasets[NO_FILE]
    manifest = datasets[POST_MANIFEST_FILE]
    require(len(states) == 3474 and len({key(row) for row in states}) == 3474, 'Candidate inventory count or uniqueness differs')
    require(Counter(row['sense2_status'] for row in states) == INITIAL_STATES, 'Initial candidate-state counts differ')
    by_key = {key(row): row for row in states}
    require(len(texts) == 5032 and len({row['sense2_representation_id'] for row in texts}) == 5032, 'Source textual IDs/count differ')
    require(Counter(row['textual_representation_usable'] for row in texts) == {'true': 4591, 'false': 441}, 'Corrected source usability counts differ')
    total_by_key = Counter(key(row) for row in texts)
    usable_by_key = Counter(key(row) for row in texts if row['textual_representation_usable'] == 'true')
    for row in texts:
        require(key(row) in by_key, 'Text representation has an unknown candidate')
        require(bool(row['sense2_text_en'].strip()), 'Source textual representation is empty')
        require(re.fullmatch(r's2rep_[0-9a-f]{64}', row['sense2_representation_id']) is not None, 'Invalid existing representation ID')
        require(row['representation_source'] in {
            'WIKIDATA_DESCRIPTION_EN', 'WIKIDATA_CONTEXTUAL_TRANSLATED_DESCRIPTION_V2',
            'WIKIDATA_ENGLISH_VARIANT',
        }, 'Unexpected/historical textual representation route')
        if row['textual_representation_usable'] == 'true':
            require(by_key[key(row)]['sense2_status'] in {'TEXTUAL_EN_READY', 'TEXTUAL_MULTILINGUAL_READY'}, 'A non-ready candidate has a usable textual representation')
    for state in states:
        require(int(state['total_textual_representation_count']) == total_by_key[key(state)], 'Original textual count differs from source rows')
        require(int(state['usable_textual_representation_count']) == usable_by_key[key(state)], 'Original usable textual count differs from flags')
    pending_keys = {key(row) for row in states if row['sense2_status'] == 'DEFTERM_PENDING'}
    definition_keys = {key(row) for row in definitions}
    insufficient_keys = {key(row) for row in insufficient}
    require(len(definitions) == len(definition_keys) == 63, 'Expected exactly 63 unique DefTerm rows')
    require(len(insufficient) == len(insufficient_keys) == 87, 'Expected exactly 87 unique insufficient rows')
    require(definition_keys.isdisjoint(insufficient_keys) and definition_keys | insufficient_keys == pending_keys, 'Gate partition does not cover precisely the 150 pending candidates')
    require(len({row['defterm_input_id'] for row in [*definitions, *insufficient]}) == 150, 'Duplicate DefTerm input ID')
    require(len(absent) == 38 and {key(row) for row in absent} == {key(row) for row in states if row['sense2_status'] == 'NO_REPRESENTATION_EVIDENCE'}, 'No-representation input does not match candidate states')
    for row in absent:
        require(all(row[field] == by_key[key(row)][field] for field in NO_SOURCE_FIELDS), 'No-representation source metadata differs')
    require(manifest['metadata_mode'] == 'PRODUCTION' and manifest['total_count'] == 150
            and manifest['sufficient_count'] == 63 and manifest['insufficient_count'] == 87,
            'Post-gate manifest counts/mode differ')
    require(manifest['insufficient_file'] == INSUFFICIENT_FILE and manifest['insufficient_sha256'] == EXPECTED_INPUT_HASHES[INSUFFICIENT_FILE], 'Insufficient input is not the committed partition')
    require(manifest['gate_results_file'] == '05_defterm_qwen14_sufficiency_gate_fewshot_full_v2_results.csv', 'Manifest is not from the accepted full V2')
    require(manifest['full_10_shot_count'] == 140 and manifest['leave_self_out_9_shot_count'] == 10, 'Gate inference mode counts differ')
    require(manifest['self_exclusion_policy'] == 'LEAVE_SELF_OUT_BY_DEFTERM_INPUT_ID_V1', 'Unexpected gate self-exclusion policy')
    require(manifest['gate_model_id'] == 'Qwen/Qwen3-14B'
            and manifest['gate_model_revision'] == '40c069824f4251a91eefaf281ebe4c544efd3e18', 'Gate model/revision differs')
    reconstructed_sufficient = [{field: row[field] for field in PARTITION_FIELDS} for row in definitions]
    require(sha256_bytes(serialize_csv(reconstructed_sufficient, PARTITION_FIELDS)) == manifest['sufficient_sha256'], 'Definition rows do not reconstruct the committed sufficient partition')
    for row in [*definitions, *insufficient]:
        require(all(row[field] == by_key[key(row)][field] for field in ('term', 'qid', 'label_en', 'representation_route')), 'Pipeline candidate identity differs from source state')
        require(row['post_gate_partition_id'] == manifest['partition_id'], 'Partition provenance differs')
        require(row['post_gate_preparer_script_sha256'] == manifest['preparer_script_sha256'], 'Preparer provenance differs')
        require(row['sufficiency_gate_results_sha256'] == manifest['gate_results_sha256']
                and row['sufficiency_gate_script_sha256'] == manifest['gate_script_sha256']
                and row['sufficiency_gate_input_sha256'] == manifest['inventory_sha256'], 'Gate/input hashes disagree with manifest')
        require(row['sufficiency_gate_model'] == manifest['gate_model_id']
                and row['sufficiency_gate_model_revision'] == manifest['gate_model_revision']
                and row['sufficiency_gate_prompt_version'] == manifest['gate_prompt_version'], 'Gate technical provenance differs')
        values = {field: json.loads(row[field]) for field in ('aliases_en', 'p31', 'p279', 'p1343')}
        identity = {**values, 'term': row['term'], 'qid': row['qid'], 'label_en': row['label_en'], 'version': 'defterm-input-v1'}
        expected_id = 'defterm_input_' + hashlib.sha256(canonical_json(identity).encode('utf-8')).hexdigest()
        require(row['defterm_input_id'] == expected_id, 'Content-based DefTerm identity differs')
        sections = [f"Label: {row['label_en']}"] if row['label_en'] else []
        if values['aliases_en']:
            sections.append(f"Aliases: {'; '.join(values['aliases_en'])}")
        for heading, field in (('Instance of', 'p31'), ('Subclass of', 'p279'), ('Described by source', 'p1343')):
            labels = [relation['label_en'] for relation in values[field] if relation['label_en']]
            if labels:
                sections.append(f"{heading}: {'; '.join(labels)}")
        require(row['defterm_evidence_text'] == '\n'.join(sections), 'DefTerm evidence differs from its immutable structured content')
    for row in definitions:
        require(row['sufficiency_decision'] == 'SUFFICIENT_EVIDENCE', 'A definition did not pass the gate')
        require(row['generation_status'] == 'OK' and row['generation_attempts'] == '1'
                and row['generation_call_status'] == 'RETURNED' and row['error_message'] == '', 'DefTerm technical completion differs')
        require(bool(row['definition_clean'].strip()), 'Empty definition_clean')
        require(row['circularity_flag_final'] == 'false', 'Circularity audit differs from the frozen result')
        require(row['metadata_mode'] == 'PRODUCTION' and row['experiment_type'] == 'FEW_SHOT', 'Unexpected DefTerm run type')
        require(row['model_id'] == 'Qwen/Qwen3-14B' and row['model_revision'] == manifest['gate_model_revision'], 'Unexpected DefTerm model/revision')
        require(row['prompt_version'] == 'defterm-en-few-v1'
                and row['fewshot_sha256'] == 'AC83C1996824E1D2645FCBCC56FBA14282572498DA8000238C3A77D50B088026', 'DefTerm prompt/example provenance differs')
        require(row['input_sha256'] == manifest['sufficient_sha256']
                and row['post_gate_manifest_sha256'] == EXPECTED_INPUT_HASHES[POST_MANIFEST_FILE], 'DefTerm input/manifest provenance differs')
    require(all(row['sufficiency_decision'] == 'INSUFFICIENT_EVIDENCE' for row in insufficient), 'Insufficient partition contains a different decision')


def lineage(filename: str, number: int, row: dict[str, str]) -> dict[str, str]:
    return {
        'source_input_file': filename, 'source_input_sha256': EXPECTED_INPUT_HASHES[filename],
        'source_input_row_number': str(number), 'source_input_row_sha256': row_hash(row),
        'consolidation_version': SCHEMA_VERSION,
    }


def defterm_representation_id(row: dict[str, str]) -> str:
    identity = {
        'version': SCHEMA_VERSION, 'route': DEFTERM_ROUTE,
        'defterm_input_id': row['defterm_input_id'], 'term': row['term'], 'qid': row['qid'],
        'sense2_text': row['definition_clean'], 'source_file': DEFTERM_FILE,
        'source_sha256': EXPECTED_INPUT_HASHES[DEFTERM_FILE], 'source_row_sha256': row_hash(row),
        'post_gate_partition_id': row['post_gate_partition_id'],
        'post_gate_manifest_sha256': row['post_gate_manifest_sha256'],
    }
    return 's2rep_' + hashlib.sha256(canonical_json(identity).encode('utf-8')).hexdigest()


def consolidate(datasets: dict[str, Any]) -> Consolidation:
    initial_states = datasets[STATE_FILE]
    definitions = datasets[DEFTERM_FILE]
    insufficient = datasets[INSUFFICIENT_FILE]
    definition_by_key = {key(row): row for row in definitions}
    insufficient_by_key = {key(row): row for row in insufficient}
    absent_by_key = {key(row): row for row in datasets[NO_FILE]}
    final_state_by_key = {}
    for source in initial_states:
        state = source['sense2_status']
        if state == 'DEFTERM_PENDING':
            state = 'DEFTERM_READY' if key(source) in definition_by_key else 'NO_SENSE2_GATE_INSUFFICIENT'
        final_state_by_key[key(source)] = state
    representations = []
    for number, source in enumerate(datasets[TEXT_FILE], 1):
        if source['textual_representation_usable'] != 'true':
            continue
        row = {field: '' for field in REP_FIELDS}
        row.update(source)  # All 16 original textual fields remain identical.
        row.update({
            'sense2_text': source['sense2_text_en'],
            'sense2_route': source['representation_source'],
            'candidate_state_final': final_state_by_key[key(source)], 'representation_usable': 'true',
            **lineage(TEXT_FILE, number, source),
        })
        representations.append(row)
    for number, source in enumerate(definitions, 1):
        row = {field: '' for field in REP_FIELDS}
        row.update(source)  # Includes every original gate and model provenance field.
        row.update({
            'sense2_representation_id': defterm_representation_id(source),
            'sense2_text': source['definition_clean'], 'sense2_text_en': source['definition_clean'],
            'sense2_route': DEFTERM_ROUTE, 'representation_source': DEFTERM_ROUTE,
            'source_language': 'en', 'candidate_state_final': 'DEFTERM_READY',
            'representation_usable': 'true', **lineage(DEFTERM_FILE, number, source),
        })
        representations.append(row)
    representation_counts = Counter(key(row) for row in representations)
    states = []
    without_sense2 = []
    for number, source in enumerate(initial_states, 1):
        final_state = final_state_by_key[key(source)]
        row = {field: '' for field in STATE_FIELDS}
        row.update({field: source[field] for field in STATE_SOURCE_FIELDS[:-1]})
        row.update({
            'source_candidate_state': source['sense2_status'],
            'candidate_state_final': final_state, 'has_usable_sense2': 'true' if final_state in USABLE_STATES else 'false',
            'sense2_representation_count': str(representation_counts[key(source)]),
            # This preserves the previously assigned route category; it does not
            # select a preferred representation among the retained alternatives.
            'primary_representation_route': (DEFTERM_ROUTE if final_state == 'DEFTERM_READY' else source['representation_route']) if final_state in USABLE_STATES else '',
            **lineage(STATE_FILE, number, source),
        })
        pipeline_source = definition_by_key.get(key(source)) or insufficient_by_key.get(key(source))
        if pipeline_source is not None:
            for field in DEFTERM_FIELDS:
                if field not in STATE_SOURCE_FIELDS and field in pipeline_source:
                    row[field] = pipeline_source[field]
            filename = DEFTERM_FILE if final_state == 'DEFTERM_READY' else INSUFFICIENT_FILE
            row.update({
                'source_pipeline_input_file': filename,
                'source_pipeline_input_sha256': EXPECTED_INPUT_HASHES[filename],
                'source_pipeline_row_sha256': row_hash(pipeline_source),
            })
        if key(source) in absent_by_key:
            row['source_no_sense2_row_sha256'] = row_hash(absent_by_key[key(source)])
        states.append(row)
        if final_state not in USABLE_STATES:
            without_sense2.append({**row, 'no_sense2_reason': 'GATE_INSUFFICIENT_EVIDENCE' if final_state == 'NO_SENSE2_GATE_INSUFFICIENT' else 'NO_REPRESENTATION_EVIDENCE'})
    statistics = {
        'candidate_states': dict(Counter(row['candidate_state_final'] for row in states)),
        'total_candidates': len(states), 'candidates_with_usable_sense2': sum(row['has_usable_sense2'] == 'true' for row in states),
        'candidates_without_usable_sense2': len(without_sense2),
        'source_textual_total': len(datasets[TEXT_FILE]), 'source_textual_usable': 4591,
        'source_textual_unusable': 441, 'usable_textual_representations': 4591,
        'defterm_representations': len(definitions), 'total_final_operational_representations': len(representations),
        'representation_routes': dict(Counter(row['sense2_route'] for row in representations)),
        'final_representation_candidate_count': len(representation_counts),
    }
    return Consolidation(states, representations, without_sense2, statistics)


def validate_outputs(datasets: dict[str, Any], result: Consolidation) -> None:
    states, representations, absent = result.states, result.representations, result.without_sense2
    for fields in (STATE_FIELDS, REP_FIELDS, NO_FIELDS):
        require(len(fields) == len(set(fields)), 'Duplicate output field names')
        validate_no_sense1_fields(fields)
    require(len(states) == 3474 and len({key(row) for row in states}) == 3474, 'Final candidate count/uniqueness differs')
    require(Counter(row['candidate_state_final'] for row in states) == FINAL_STATES, 'Final candidate-state distribution differs')
    require(all(row['candidate_state_final'] != 'DEFTERM_PENDING' for row in states), 'Active DEFTERM_PENDING remains')
    require(sum(row['has_usable_sense2'] == 'true' for row in states) == 3349 and len(absent) == 125, 'Final usable/unusable candidate counts differ')
    require(len(representations) == 4654 and len({row['sense2_representation_id'] for row in representations}) == 4654, 'Operational representation count/uniqueness differs')
    require(len({key(row) for row in representations}) == 3349, 'Operational representation candidate count differs')
    state_by_key = {key(row): row for row in states}
    representation_counts = Counter(key(row) for row in representations)
    for row in representations:
        require(tuple(row) == REP_FIELDS, 'Representation schema/order differs')
        require(bool(row['sense2_text'].strip()), 'Empty operative Sense 2 text')
        require(row['representation_usable'] == 'true', 'Unusable representation reached the operational output')
        require(key(row) in state_by_key and state_by_key[key(row)]['has_usable_sense2'] == 'true', 'Representation belongs to a no-Sense-2 candidate')
        require(row['candidate_state_final'] == state_by_key[key(row)]['candidate_state_final'], 'Representation state differs from candidate state')
    usable_texts = [row for row in datasets[TEXT_FILE] if row['textual_representation_usable'] == 'true']
    require(len(usable_texts) == 4591, 'Source usable textual count differs')
    for actual, source in zip(representations[:4591], usable_texts):
        require(all(actual[field] == source[field] for field in TEXT_FIELDS), 'Existing textual ID, content or provenance was modified')
        require(actual['sense2_text'] == source['sense2_text_en'] and actual['sense2_route'] == source['representation_source'], 'Textual operational projection was modified')
        require(actual['source_input_row_sha256'] == row_hash(source), 'Textual source lineage differs')
    false_ids = {row['sense2_representation_id'] for row in datasets[TEXT_FILE] if row['textual_representation_usable'] == 'false'}
    require(false_ids.isdisjoint(row['sense2_representation_id'] for row in representations), 'Historical false representation reached the operational output')
    new_definitions = representations[4591:]
    require(len(new_definitions) == 63, 'Expected exactly 63 new definitions')
    for actual, source in zip(new_definitions, datasets[DEFTERM_FILE]):
        require(all(actual[field] == source[field] for field in DEFTERM_FIELDS), 'DefTerm input content/evidence/provenance was changed')
        require(actual['sense2_text'] == actual['sense2_text_en'] == source['definition_clean'], 'definition_clean was modified or replaced')
        require(actual['sense2_representation_id'] == defterm_representation_id(source), 'Deterministic DefTerm ID differs')
        require(actual['sense2_route'] == DEFTERM_ROUTE and actual['source_language'] == 'en', 'DefTerm route/language differs')
    for actual, source in zip(states, datasets[STATE_FILE]):
        require(tuple(actual) == STATE_FIELDS, 'Final state schema/order differs')
        require(all(actual[field] == source[field] for field in STATE_SOURCE_FIELDS[:-1]), 'Original state metadata was modified')
        require(actual['source_candidate_state'] == source['sense2_status'], 'Historical state provenance differs')
        count = representation_counts[key(actual)]
        require(actual['sense2_representation_count'] == str(count), 'Candidate representation count differs')
        require((actual['has_usable_sense2'] == 'true') == (count > 0), 'Candidate usability differs from operational representations')
    expected_absent = [row for row in states if row['has_usable_sense2'] == 'false']
    require(len(absent) == len({key(row) for row in absent}) == 125, 'Duplicate/missing no-Sense-2 candidates')
    for actual, expected in zip(absent, expected_absent):
        require(tuple(actual) == NO_FIELDS and all(actual[field] == expected[field] for field in STATE_FIELDS), 'No-Sense-2 projection differs')
        expected_reason = 'GATE_INSUFFICIENT_EVIDENCE' if expected['candidate_state_final'] == 'NO_SENSE2_GATE_INSUFFICIENT' else 'NO_REPRESENTATION_EVIDENCE'
        require(actual['no_sense2_reason'] == expected_reason, 'No-Sense-2 reason differs')
    require(Counter(row['no_sense2_reason'] for row in absent) == {'GATE_INSUFFICIENT_EVIDENCE': 87, 'NO_REPRESENTATION_EVIDENCE': 38}, 'No-Sense-2 reason distribution differs')


def build_payloads(
    datasets: dict[str, Any], schemas: dict[str, tuple[str, ...]], locations: dict[str, Path],
    result: Consolidation,
) -> dict[str, bytes]:
    payloads = {
        STATE_OUTPUT: serialize_csv(result.states, STATE_FIELDS),
        REP_OUTPUT: serialize_csv(result.representations, REP_FIELDS),
        NO_OUTPUT: serialize_csv(result.without_sense2, NO_FIELDS),
    }
    row_counts = {STATE_OUTPUT: len(result.states), REP_OUTPUT: len(result.representations), NO_OUTPUT: len(result.without_sense2)}
    output_schemas = {STATE_OUTPUT: STATE_FIELDS, REP_OUTPUT: REP_FIELDS, NO_OUTPUT: NO_FIELDS}
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'script_file': Path(__file__).name, 'script_sha256': sha256_bytes(Path(__file__).read_bytes()),
        'inputs': {
            filename: {
                'sha256': expected, 'resolved_path': str(locations[filename].resolve()),
                'record_count': len(datasets[filename]) if filename in schemas else None,
                'fields': list(schemas[filename]) if filename in schemas else None,
            }
            for filename, expected in EXPECTED_INPUT_HASHES.items()
        },
        'outputs': {
            filename: {'sha256': sha256_bytes(payload), 'sha256_scope': 'EXACT_FILE_BYTES',
                       'record_count': row_counts[filename], 'fields': list(output_schemas[filename])}
            for filename, payload in payloads.items()
        },
        'counts': result.statistics,
        'validation': {
            'remaining_defterm_pending': 0, 'duplicate_term_qid_state_rows': 0,
            'empty_usable_sense2_text': 0, 'final_unusable_textual_representations': 0,
            'final_representations_from_no_sense2_candidates': 0,
            'defterm_evidence_immutability': 'PASS', 'defterm_definition_immutability': 'PASS',
            'existing_usable_textual_content_and_id_immutability': 'PASS',
            'no_sense1_fields': 'PASS', 'historical_quality_evaluations_used': False,
        },
        'policies': {
            'textual_selection': 'PRESERVE_ONLY_EXISTING_TEXTUAL_REPRESENTATION_USABLE_TRUE',
            'unusable_source_rows': '441_ROWS_REMAIN_IN_EXISTING_AUDIT_ARTIFACT_ONLY',
            'defterm_text': 'EXACT_DEFINITION_CLEAN_NO_NORMALIZATION_OR_REPAIR',
            'primary_route': 'PREEXISTING_ROUTE_CATEGORY_OR_DEFTERM_ROUTE_NOT_REPRESENTATION_SELECTION',
            'representation_aggregation': 'NONE', 'corpus_context_used': False,
        },
        'post_gate_partition_id': datasets[POST_MANIFEST_FILE]['partition_id'],
        'publication_protocol': 'KERNEL_LOCK_JOURNAL_DURABLE_CSVS_MANIFEST_LAST',
        'model_loaded': False, 'inference_executed': False,
    }
    # A file cannot contain its own exact-file SHA-256 without circularity.
    # Its scoped checksum covers canonical UTF-8 JSON (with LF) excluding this
    # output entry. The exact manifest-file SHA is printed after publication.
    checksum_payload = (canonical_json(manifest) + '\n').encode('utf-8')
    manifest['outputs'][MANIFEST_OUTPUT] = {
        'sha256': sha256_bytes(checksum_payload),
        'sha256_scope': 'CANONICAL_UTF8_JSON_WITH_LF_EXCLUDING_THIS_OUTPUT_ENTRY',
        'record_count': None,
    }
    payloads[MANIFEST_OUTPUT] = (canonical_json(manifest) + '\n').encode('utf-8')
    return payloads


def validate_payloads(datasets: dict[str, Any], result: Consolidation, payloads: dict[str, bytes]) -> None:
    reconstructed = []
    for filename, fields, expected in (
        (STATE_OUTPUT, STATE_FIELDS, result.states),
        (REP_OUTPUT, REP_FIELDS, result.representations),
        (NO_OUTPUT, NO_FIELDS, result.without_sense2),
    ):
        parsed_fields, rows = parse_csv(filename, payloads[filename])
        require(parsed_fields == fields and rows == expected, f'{filename}: loss in CSV serialization')
        reconstructed.append(rows)
    validate_outputs(datasets, Consolidation(*reconstructed, result.statistics))
    manifest = json.loads(payloads[MANIFEST_OUTPUT].decode('utf-8'))
    for filename in (STATE_OUTPUT, REP_OUTPUT, NO_OUTPUT):
        require(manifest['outputs'][filename]['sha256'] == sha256_bytes(payloads[filename]), 'Manifest CSV hash differs')
    self_entry = manifest['outputs'].pop(MANIFEST_OUTPUT)
    require(self_entry['sha256'] == sha256_bytes((canonical_json(manifest) + '\n').encode('utf-8')), 'Manifest canonical self-checksum differs')
    require(manifest['counts'] == result.statistics, 'Manifest counts differ')


def assert_inputs_unchanged(locations: dict[str, Path]) -> None:
    for filename, path in locations.items():
        require(sha256_bytes(path.read_bytes()) == EXPECTED_INPUT_HASHES[filename], f'Input changed during consolidation: {filename}')


def sync_directory(path: Path) -> None:
    if os.name == 'nt':
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomically(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes

            move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
            move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
            move.restype = wintypes.BOOL
            if not move(str(temporary), str(path), 0x1 | 0x8):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.replace(temporary, path)
        sync_directory(path.parent)
        require(path.read_bytes() == payload, f'{path.name}: published bytes differ')
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_lock(path: Path) -> tuple[int, str]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        if os.name == 'nt':
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b'\0')
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return descriptor, 'nt'
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor, 'posix'
    except OSError as exc:
        os.close(descriptor)
        raise ValidationError('Another consolidation holds the output lock') from exc


def release_lock(lock: tuple[int, str]) -> None:
    descriptor, kind = lock
    try:
        if kind == 'nt':
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def publish(payloads: dict[str, bytes], directory: Path = BASE_DIR) -> str:
    lock_path = directory / LOCK_PATH.name
    journal_path = directory / JOURNAL_PATH.name
    lock = acquire_lock(lock_path)
    try:
        expected_hashes = {filename: sha256_bytes(payloads[filename]) for filename in OUTPUT_FILES}
        journal_payload = (canonical_json({'schema_version': SCHEMA_VERSION, 'outputs_sha256': expected_hashes}) + '\n').encode('utf-8')
        manifest_path = directory / MANIFEST_OUTPUT
        if manifest_path.exists():
            require(all((directory / filename).is_file() and (directory / filename).read_bytes() == payloads[filename] for filename in OUTPUT_FILES), 'Existing committed outputs differ; refusing overwrite')
            if journal_path.exists():
                require(journal_path.read_bytes() == journal_payload, 'Surviving publication journal differs')
                journal_path.unlink()
                sync_directory(directory)
            return 'UNCHANGED_IDENTICAL'
        if journal_path.exists():
            require(journal_path.read_bytes() == journal_payload, 'Interrupted publication belongs to different inputs/version')
            for filename in OUTPUT_FILES:
                path = directory / filename
                if path.exists():
                    require(path.read_bytes() == payloads[filename], 'Uncommitted file differs; refusing overwrite')
        else:
            require(not any((directory / filename).exists() for filename in OUTPUT_FILES), 'Outputs exist without a valid commit/journal')
            write_atomically(journal_path, journal_payload)
        for filename in OUTPUT_FILES:  # Manifest is the final commit point.
            write_atomically(directory / filename, payloads[filename])
        require(all((directory / filename).read_bytes() == payloads[filename] for filename in OUTPUT_FILES), 'Post-publication verification failed')
        journal_path.unlink()
        sync_directory(directory)
        return 'COMMITTED'
    finally:
        release_lock(lock)


def expect_rejection(operation: Any) -> None:
    try:
        operation()
    except ValidationError:
        return
    raise ValidationError('A deliberately invalid consolidation was accepted')


def run_self_tests(
    datasets: dict[str, Any], schemas: dict[str, tuple[str, ...]], locations: dict[str, Path],
    result: Consolidation, payloads: dict[str, bytes],
) -> None:
    tests = {}

    def passed(name: str, condition: bool) -> None:
        require(condition, f'{name} failed')
        tests[name] = 'PASS'

    passed('INPUT_HASH_TEST', all(sha256_bytes(path.read_bytes()) == EXPECTED_INPUT_HASHES[filename] for filename, path in locations.items()))
    passed('FINAL_CANDIDATE_STATE_COUNTS_TEST', Counter(row['candidate_state_final'] for row in result.states) == FINAL_STATES)
    passed('CANDIDATE_LEVEL_COUNT_TEST', len(result.states) == 3474 and len(result.without_sense2) == 125 and sum(row['has_usable_sense2'] == 'true' for row in result.states) == 3349)
    passed('CORRECTED_TEXTUAL_USABILITY_COUNTS_TEST', Counter(row['textual_representation_usable'] for row in datasets[TEXT_FILE]) == {'true': 4591, 'false': 441})
    source_true = [row for row in datasets[TEXT_FILE] if row['textual_representation_usable'] == 'true']
    passed('EXISTING_TEXTUAL_CONTENT_AND_ID_IMMUTABILITY_TEST', all(all(actual[field] == source[field] for field in TEXT_FIELDS) for actual, source in zip(result.representations[:4591], source_true)))
    false_ids = {row['sense2_representation_id'] for row in datasets[TEXT_FILE] if row['textual_representation_usable'] == 'false'}
    passed('NO_UNUSABLE_TEXTUAL_REPRESENTATION_TEST', false_ids.isdisjoint(row['sense2_representation_id'] for row in result.representations))
    passed('OPERATIONAL_REPRESENTATION_COUNT_TEST', len(result.representations) == 4654 and len({key(row) for row in result.representations}) == 3349)
    definitions = result.representations[4591:]
    passed('DEFTERM_EVIDENCE_IMMUTABILITY_TEST', all(actual['defterm_evidence_text'] == source['defterm_evidence_text'] for actual, source in zip(definitions, datasets[DEFTERM_FILE])))
    passed('DEFTERM_DEFINITION_IMMUTABILITY_TEST', all(actual['sense2_text'] == source['definition_clean'] for actual, source in zip(definitions, datasets[DEFTERM_FILE])))
    passed('DEFTERM_COMPLETE_PROVENANCE_TEST', all(all(actual[field] == source[field] for field in DEFTERM_FIELDS) for actual, source in zip(definitions, datasets[DEFTERM_FILE])))
    passed('DEFTERM_ID_DETERMINISM_TEST', len({row['sense2_representation_id'] for row in definitions}) == 63 and all(row['sense2_representation_id'] == defterm_representation_id(source) for row, source in zip(definitions, datasets[DEFTERM_FILE])))
    passed('NO_REMAINING_DEFTERM_PENDING_TEST', all(row['candidate_state_final'] != 'DEFTERM_PENDING' for row in result.states))
    no_keys = {key(row) for row in result.without_sense2}
    passed('NO_REPRESENTATION_FROM_NO_SENSE2_CANDIDATE_TEST', no_keys.isdisjoint(key(row) for row in result.representations))
    passed('NO_SENSE2_REASON_TEST', Counter(row['no_sense2_reason'] for row in result.without_sense2) == {'GATE_INSUFFICIENT_EVIDENCE': 87, 'NO_REPRESENTATION_EVIDENCE': 38})
    validate_no_sense1_fields(STATE_FIELDS)
    validate_no_sense1_fields(REP_FIELDS)
    expect_rejection(lambda: validate_no_sense1_fields((*REP_FIELDS, 'corpus_context')))
    passed('NO_SENSE1_FIELDS_TEST', True)
    passed('NO_HISTORICAL_EVALUATION_ACCESS_TEST', set(locations) == set(EXPECTED_INPUT_HASHES) and len(locations) == 6)
    bad = deepcopy(result)
    bad.representations[4591]['sense2_text'] += ' '
    expect_rejection(lambda: validate_outputs(datasets, bad))
    passed('DEFINITION_REPAIR_REJECTION_TEST', True)
    bad = deepcopy(result)
    bad.representations[4591]['defterm_evidence_text'] += '\nUnsupported change'
    expect_rejection(lambda: validate_outputs(datasets, bad))
    passed('EVIDENCE_REPAIR_REJECTION_TEST', True)
    bad = deepcopy(result)
    rejected_source = next(row for row in datasets[TEXT_FILE] if row['textual_representation_usable'] == 'false')
    bad.representations[0]['sense2_representation_id'] = rejected_source['sense2_representation_id']
    expect_rejection(lambda: validate_outputs(datasets, bad))
    passed('UNUSABLE_TEXTUAL_PROMOTION_REJECTION_TEST', True)
    snapshot = canonical_json(datasets)
    repeated = consolidate(datasets)
    validate_outputs(datasets, repeated)
    repeated_payloads = build_payloads(datasets, schemas, locations, repeated)
    passed('DETERMINISM_TEST', result == repeated and payloads == repeated_payloads)
    passed('IN_MEMORY_INPUT_IMMUTABILITY_TEST', canonical_json(datasets) == snapshot)
    validate_payloads(datasets, result, payloads)
    passed('OUTPUT_SCHEMA_CSV_ROUNDTRIP_AND_MANIFEST_HASH_TEST', True)
    parent = Path(tempfile.gettempdir()) / 'opencode'
    with tempfile.TemporaryDirectory(prefix='sense2-final-self-test-', dir=parent if parent.is_dir() else None) as name:
        directory = Path(name)
        require(publish(payloads, directory) == 'COMMITTED', 'Synthetic publication failed')
        passed('ATOMIC_PUBLICATION_IDEMPOTENCE_TEST', publish(payloads, directory) == 'UNCHANGED_IDENTICAL')
        (directory / MANIFEST_OUTPUT).unlink()
        journal = {'schema_version': SCHEMA_VERSION, 'outputs_sha256': {filename: sha256_bytes(payloads[filename]) for filename in OUTPUT_FILES}}
        write_atomically(directory / JOURNAL_PATH.name, (canonical_json(journal) + '\n').encode('utf-8'))
        passed('INTERRUPTED_PUBLICATION_RECOVERY_TEST', publish(payloads, directory) == 'COMMITTED')
        write_atomically(directory / REP_OUTPUT, b'changed\n')
        expect_rejection(lambda: publish(payloads, directory))
        passed('EXISTING_DIFFERENT_OUTPUT_REJECTION_TEST', True)
    assert_inputs_unchanged(locations)
    assert_no_model_runtime()
    passed('NO_MODEL_OR_INFERENCE_TEST', True)
    print('SELF_TEST_STATUS=SUCCESS')
    print(f'SELF_TEST_COUNT={len(tests)}')
    for name, status in tests.items():
        print(f'{name}={status}')


def print_final_audit(result: Consolidation, locations: dict[str, Path]) -> None:
    print('INPUT_VALIDATION=OK')
    print(f'DEFTERM_RESULTS_LOCATION={locations[DEFTERM_FILE]}')
    print('TOTAL_CANDIDATES=3474')
    for state, count in sorted(result.statistics['candidate_states'].items()):
        print(f'{state}={count}')
    print('CANDIDATES_WITH_USABLE_SENSE2=3349')
    print('CANDIDATES_WITHOUT_USABLE_SENSE2=125')
    print('SOURCE_TEXTUAL_REPRESENTATIONS=5032')
    print('SOURCE_TEXTUAL_USABLE_TRUE=4591')
    print('SOURCE_TEXTUAL_USABLE_FALSE=441')
    print('DEFTERM_REPRESENTATIONS=63')
    print('TOTAL_FINAL_OPERATIONAL_REPRESENTATIONS=4654')
    print(f"REPRESENTATIONS_BY_ROUTE={canonical_json(result.statistics['representation_routes'])}")
    print('FINAL_UNUSABLE_TEXTUAL_REPRESENTATIONS=0')
    print('FINAL_REPRESENTATIONS_FROM_NO_SENSE2_CANDIDATES=0')
    print('FINAL_REPRESENTATION_CANDIDATE_COUNT=3349')
    print('REMAINING_DEFTERM_PENDING=0')
    print('DUPLICATE_TERM_QID_STATE_ROWS=0')
    print('EMPTY_USABLE_SENSE2_TEXT=0')
    print('DEFTERM_EVIDENCE_IMMUTABILITY=PASS')
    print('DEFTERM_DEFINITION_IMMUTABILITY=PASS')
    print('NO_SENSE1_FIELDS=PASS')
    print('INPUTS_UNCHANGED=YES')
    print('MODEL_LOADED=NO')
    print('INFERENCE_EXECUTED=NO')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect-inputs', action='store_true')
    modes.add_argument('--validate-inputs', action='store_true')
    modes.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    assert_no_model_runtime()
    datasets, schemas, locations = read_inputs()
    if args.inspect_inputs:
        inspect_inputs(datasets, locations)
        return 0
    validate_inputs(datasets, schemas)
    result = consolidate(datasets)
    validate_outputs(datasets, result)
    payloads = build_payloads(datasets, schemas, locations, result)
    validate_payloads(datasets, result, payloads)
    if args.self_test:
        run_self_tests(datasets, schemas, locations, result, payloads)
    elif not args.validate_inputs:
        assert_inputs_unchanged(locations)
        print(f'OUTPUT_STATUS={publish(payloads)}')
        # Validate the committed artifacts read back from disk, not just memory.
        actual = {filename: (BASE_DIR / filename).read_bytes() for filename in OUTPUT_FILES}
        require(actual == payloads, 'Committed output bytes differ')
        validate_payloads(datasets, result, actual)
        for filename in OUTPUT_FILES:
            print(f'OUTPUT_SHA256[{filename}]={sha256_bytes(actual[filename])}')
    assert_inputs_unchanged(locations)
    assert_no_model_runtime()
    print_final_audit(result, locations)
    print(f'SCRIPT_SHA256={sha256_bytes(Path(__file__).read_bytes())}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError) as exc:
        print(f'SENSE2_CONSOLIDATION_STATUS=FAILED: {exc}')
        raise SystemExit(1) from exc
