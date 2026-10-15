import json

import pytest

from scripts.fetch_training_sources import read_qrels, select_groups, source_records
from scripts.training_data import (
    audit_splits,
    messages_for,
    tokenize_example,
    validate_record,
)


def ranking(**changes):
    return {'id': 'r', 'group_id': 'g', 'task': 'ranking', 'split': 'train', 'language': 'zh',
            'query': 'SQLite', 'candidates': [{'title': '', 'url': '', 'snippet': t} for t in ['wrong', 'right', 'also right']],
            'judgments': [0, 3, 3], 'judgment_source': 'published_human',
            'source': {'name': 't2ranking', 'revision': 'abc', 'license': 'Apache-2.0', 'url': 'https://example.test'}} | changes


def query(**changes):
    return ranking(task='query', id='q', query='SQLite 3.46 -Django', group_id='q',
                   queries=['SQLite 3.46 -Django'], review_status='confirmed', reviewer='human') | changes


def test_same_query_contrast_and_unknown_labels():
    validate_record(ranking())
    with pytest.raises(ValueError, match='contrast'):
        validate_record(ranking(judgments=[3, 3, 3]))
    with pytest.raises(ValueError, match='relevance'):
        validate_record(ranking(judgments=[0, None, 3]))
    with pytest.raises(ValueError, match='human_judgments'):
        validate_record(ranking(judgment_source='model'))


def test_query_requires_actual_review():
    validate_record(query())
    for values in [{'review_status': 'pending'}, {'reviewer': None}, {'queries': []}]:
        with pytest.raises(ValueError):
            validate_record(query(**values))


def test_cross_task_leakage_includes_rewritten_query():
    with pytest.raises(ValueError, match='cross_split'):
        audit_splits([ranking(), query(split='test', queries=[' SQLite '])])
    with pytest.raises(ValueError, match='duplicate'):
        audit_splits([ranking(), ranking()])
    audit_splits([ranking(), query()])


def test_target_uses_shuffled_ids_and_stable_ties_without_rewriting():
    row = ranking()
    first = messages_for(row)
    assert first == messages_for(row)
    payload = json.loads(first[1]['content'])['results']
    grades = {'wrong': 0, 'right': 3, 'also right': 3}
    expected = sorted(range(3), key=lambda i: -grades[payload[i]['snippet']])
    assert json.loads(first[-1]['content']) == expected
    assert sorted(x['snippet'] for x in payload) == ['also right', 'right', 'wrong']
    assert all(x['url'] == '' for x in payload)


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=True):
        assert not enable_thinking
        text = ''.join(m['role'] + ':' + m['content'] + '|' for m in messages)
        if add_generation_prompt:
            text += 'assistant:'
        return list(text.encode())


def test_training_prefix_matches_inference_and_masks_input():
    tokens, offset = tokenize_example(Tokenizer(), messages_for(query()), 'query')
    assert bytes(tokens[offset:]).startswith(b'{"queries"')
    with pytest.raises(ValueError, match='context_limit'):
        tokenize_example(Tokenizer(), messages_for(query(query='a' * 2048)), 'query')


def test_qrels_header_conflicts_and_selection(tmp_path):
    p = tmp_path / 'qrels.tsv'
    p.write_text('qid iter pid rel\nq 0 p1 3\nq 0 p2 0\n')
    qrels = read_qrels(p)
    selected = select_groups({'q': 'question'}, qrels, 1, 2, '42')
    assert set(selected['q'].values()) == {0, 3}
    with pytest.raises(ValueError, match='passage_missing'):
        list(source_records('t2ranking', 'zh', 'train', {'q': 'question'}, selected, {}, 'rev'))
    p.write_text('q 0 p1 3\nq 0 p1 0\n')
    with pytest.raises(ValueError, match='annotation_conflict'):
        read_qrels(p)


def test_dataset_export_rejects_modified_target_even_with_new_file_hash(tmp_path):
    from dona_search.adapters import file_digest
    from scripts.training_data import (
        COUNTS,
        build_dataset,
        read_rows,
        verify_dataset,
        write_rows,
    )

    rows = []
    for task, splits in COUNTS.items():
        for split, count in splits.items():
            for language in ('zh', 'en'):
                for index in range(count // 2):
                    key = f'{task}:{split}:{language}:{index}'
                    if task == 'query':
                        record = query(id=key, group_id=key, query=key, split=split, language=language,
                                       queries=[key + f':variant-{i}' for i in range(index % 3 + 1)])
                    else:
                        source = 'miracl' if language == 'en' or (split == 'train' and index >= 800) else 't2ranking'
                        record = ranking(id=key, group_id=key, query=key, split=split, language=language,
                                         judgments=[0, 1, 1] if source == 'miracl' else [0, 3, 3])
                        record['source'] = record['source'] | {'name': source}
                    rows.append(record)
    for language in ('zh', 'en'):
        for index in range(10):
            key = f'web:{language}:{index}'
            rows.append(ranking(id=key, group_id=key, query=key, split='web_test', language=language,
                                judgment_source='human_confirmed', reviewer='human'))
    output = tmp_path / 'dataset'
    build_dataset(rows, output, Tokenizer())
    verify_dataset(output)
    path = output / 'ranking/train.jsonl'
    changed = read_rows(path)
    changed[0]['messages'][-1]['content'] = '[0,1,2]'
    write_rows(path, changed)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['ranking/train.jsonl'] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='training_target_mismatch'):
        verify_dataset(output)
    rows[0]['query'] = 'a' * 2048
    rejected_output = tmp_path / 'rejected-dataset'
    with pytest.raises(ValueError, match='training_records_rejected'):
        build_dataset(rows, rejected_output, Tokenizer())
    rejections = json.loads((tmp_path / 'rejected-dataset.rejected.json').read_text())
    assert rejections == [{'id': rows[0]['id'], 'task': 'query', 'error': 'training_context_limit'}]
    assert not rejected_output.exists()
