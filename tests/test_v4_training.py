import copy
import json

import pytest

from dona_search.adapters import file_digest
from dona_search.interfaces import OrganizationDecision, QueryPlan
from dona_search.organizer import parse_organization, selected_evidence
from dona_search.runtime import ModelRuntime
from scripts.eval_query import evaluate_queries
from scripts.eval_v4 import evaluate_organizing, summarize_v4
from scripts.train_models import probe_messages
from scripts.training_data import read_rows, verify_dataset, write_rows
from scripts.v4_data import (
    build_v4_dataset,
    candidate_hits,
    v4_messages,
    validate_v4_record,
)
from tests.test_training_data import Tokenizer


def record(split='test', task='organizing'):
    row = {'id': f'{task}:{split}', 'group_id': f'{task}:{split}', 'task': task, 'split': split,
           'language': 'en', 'query': f'SQLite {task} {split}', 'review_status': 'confirmed', 'reviewer': 'fixture_reviewer',
           'source': {'name': 't2ranking', 'revision': 'fixture', 'license': 'fixture', 'url': 'https://fixture.test/'}}
    if task == 'query':
        return row | {'queries': [row['query']]}
    return row | {'candidates': [{'title': 'wrong', 'url': 'https://wrong.test/', 'snippet': 'Irrelevant.'},
                                {'title': 'right', 'url': 'https://right.test/', 'snippet': 'First. Relevant.'}],
                  'judgments': [0, 3], 'judgment_source': 'human_confirmed', 'selections': [None, 2]}


def test_v4_targets_use_production_parser_and_keep_shuffled_source_binding():
    row = record()
    messages = v4_messages(row)
    payload = json.loads(messages[1]['content'])['results']
    shuffled = [next(hit for hit in candidate_hits(row) if hit.url == item['url']) for item in payload]
    decision = parse_organization(messages[-1]['content'], shuffled)
    for index, hit in enumerate(shuffled):
        evidence = selected_evidence(hit, decision.selections[index])
        assert (evidence.text if evidence else None) == ('Relevant.' if hit.title == 'right' else None)
    assert shuffled[decision.order[0]].title == 'right'
    probe = probe_messages('organizing', Tokenizer())
    assert json.loads(probe[-1]['content']) == {'order': [0, 1], 'selections': [0, None]}


@pytest.mark.parametrize('changes', [{'review_status': 'pending'}, {'reviewer': ''}, {'selections': None},
                                     {'selections': [0, 999]}, {'selections': [False, 2]}])
def test_evidence_labels_require_explicit_human_review_and_valid_indices(changes):
    with pytest.raises((ValueError, TypeError)):
        validate_v4_record(record() | changes)


def test_v4_query_targets_match_two_generated_queries():
    row = record(task='query')
    assert 'one and 2' in v4_messages(row)[0]['content']
    with pytest.raises(ValueError, match='v4_query_limit_exceeded'):
        validate_v4_record(row | {'queries': ['a', 'b', 'c']})
    class Planner:
        def plan(self, query): return QueryPlan(query, ['a', 'b', 'c'])
    assert evaluate_queries([row], Planner(), max_queries=2)['format_failures'] == 1


def test_v4_dataset_roundtrip_and_target_tampering_cannot_be_hidden_by_new_hash(tmp_path):
    rows = [record(split, task) for task in ('query', 'organizing') for split in ('train', 'valid', 'test')]
    output = tmp_path / 'dataset'
    manifest = build_v4_dataset(rows, output, Tokenizer())
    assert verify_dataset(output) == manifest
    assert manifest['framework_schema_version'] == 4 and manifest['tasks'] == ['query', 'organizing']
    target = output / 'organizing/train.jsonl'
    modified = read_rows(target)
    modified[0]['messages'][-1]['content'] = '{"order":[0,1],"selections":[0,0]}'
    write_rows(target, modified)
    manifest['files']['organizing/train.jsonl'] = file_digest(target)
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='training_target_mismatch'):
        verify_dataset(output)


def test_organizer_evaluation_separates_verbatim_copy_from_semantic_selection():
    class Organizer:
        def __init__(self, selections): self.selections = selections
        def organize(self, query, hits): return OrganizationDecision([1, 0], self.selections)
    rows = [record()]
    good = evaluate_organizing(rows, Organizer([None, 2]))[0]
    assert good['evidence_match'] and good['source_match'] and good['top1_relevant']
    wrong = evaluate_organizing(rows, Organizer([None, 0]))[0]
    assert not wrong['evidence_match'] and wrong['source_match']
    invalid = evaluate_organizing(rows, Organizer([None, 999]))[0]
    assert invalid['ranking_status'] == 'failed' and not invalid['evidence_match']


def test_v4_acceptance_requires_all_evidence_cases_and_bound_query_reviews(tmp_path):
    rows = {split: [{'id': f'{language}:{i}', 'language': language, 'quality_status': 'judged',
                    'ranking_status': 'ok', 'has_relevant_candidate': True, 'top1_relevant': True,
                    'ranked': {'ndcg_at_10': 1.0}, 'evidence_match': True, 'source_match': True}
                   for language in ('zh', 'en') for i in range(count)]
            for split, count in [('public', 100), ('web', 10)]}
    for variant in ('bf16', 'quantized', 'adapted'):
        (tmp_path / f'{variant}.organizing.json').write_text(json.dumps(rows | {
            'framework_schema_version': 4, 'dataset_sha256': 'fixture'}))

    class Planner:
        def plan(self, query): return QueryPlan(query, [query])
    query = evaluate_queries([{'id': str(i), 'query': f'q{i}', 'language': 'zh' if i < 50 else 'en'}
                              for i in range(100)], Planner())
    (tmp_path / 'adapted.query.json').write_text(json.dumps(query | {
        'framework_schema_version': 4, 'dataset_sha256': 'fixture'}))
    assert not summarize_v4(tmp_path, 'fixture')['accepted']
    reviews = [{'id': row['id'], 'prediction_sha256': row['prediction_sha256'],
                'intent_preserved': True, 'reviewer': 'fixture_reviewer'} for row in query['records']]
    path = tmp_path / 'reviews.jsonl'
    write_rows(path, reviews)
    assert summarize_v4(tmp_path, 'fixture', path)['accepted']
    assert summarize_v4(tmp_path, 'fixture')['accepted']
    bad = copy.deepcopy(rows)
    bad['web'][0]['evidence_match'] = False
    (tmp_path / 'adapted.organizing.json').write_text(json.dumps(bad | {
        'framework_schema_version': 4, 'dataset_sha256': 'fixture'}))
    assert not summarize_v4(tmp_path, 'fixture')['accepted']


def test_component_evaluation_loads_only_organizer():
    events = []
    class Organizer:
        def __init__(self): events.append('load')
        def warmup(self): events.append('warm')
        def cleanup(self): pass
        def close(self): events.append('close')
        def organize(self, query, hits): return OrganizationDecision([], [])
    with pytest.raises(ValueError, match='invalid_organizer_runtime'):
        ModelRuntime(build_organizer=Organizer)
    runtime = ModelRuntime(build_organizer=Organizer, organizer_only=True)
    try:
        assert runtime.query_processor is None
        assert runtime.call(runtime.organizer.organize, 'query', []) == OrganizationDecision([], [])
    finally:
        runtime.close()
    assert events == ['load', 'warm', 'close']


def test_v4_review_exports_pending_evidence_without_inventing_annotations(tmp_path, monkeypatch, capsys):
    from scripts.prepare_training_data import main
    ranking = record('train') | {'task': 'ranking'}
    query = record('train', 'query') | {'queries': ['one', 'two', 'three']}
    source = tmp_path / 'source.jsonl'
    write_rows(source, [ranking, query])
    original = source.read_bytes()
    output = tmp_path / 'review'
    monkeypatch.setattr('sys.argv', ['prepare_training_data', 'review', '--schema-version', '4',
                                    '--records', str(source), '--output', str(output)])
    main()
    rows = read_rows(output / '0000.jsonl')
    assert source.read_bytes() == original
    assert rows[0]['task'] == 'organizing' and rows[0]['selections'] is None
    assert rows[0]['review_status'] == 'pending' and rows[0]['reviewer'] is None
    assert rows[0]['candidate_segments'][1][2]['text'] == 'Relevant.'
    assert rows[1]['queries'] == query['queries'] and rows[1]['review_status'] == 'pending'
    assert json.loads(capsys.readouterr().out)['records'] == 2
    for row in rows:
        with pytest.raises(ValueError, match='human_review_required'):
            validate_v4_record(row)


def test_training_reload_accepts_a_bound_organizing_adapter_with_legacy_config(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file

    from dona_search.adapters import model_fingerprint
    from dona_search.config import RankingConfig
    from dona_search.organizer import MLXOrganizer
    from tests.test_model_setup import bundle

    base = bundle(tmp_path / 'base', quantized=True)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    save_file({'layer.lora_a': np.zeros((4, 1), np.float32)}, str(adapter / 'adapters.safetensors'))
    (adapter / 'dona_adapter_manifest.json').write_text(json.dumps({
        'schema_version': 1, 'task': 'organizing', 'base_fingerprint': model_fingerprint(base),
        'files': {name: file_digest(adapter / name) for name in ('adapter_config.json', 'adapters.safetensors')}}))
    loaded = []
    monkeypatch.setattr('dona_search.organizer.MLXQueryProcessor', lambda path, **kwargs: loaded.append((path, kwargs)))
    MLXOrganizer(RankingConfig(model_path=base, adapter_path=adapter))
    assert loaded == [(str(base), {'adapter_path': adapter})]
