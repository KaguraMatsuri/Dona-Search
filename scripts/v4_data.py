"""Human-reviewed v4 targets using the production planner and organizer protocol."""
import json
import random
from collections import Counter
from pathlib import Path

from dona_search.adapters import file_digest
from dona_search.interfaces import OrganizationDecision, SearchHit
from dona_search.organizer import organization_messages, validate_decision
from dona_search.query.planner import query_messages
from scripts.training_data import (
    audit_splits,
    read_rows,
    tokenize_example,
    validate_record,
    write_rows,
)

TASKS = ('query', 'organizing')
SPLITS = ('train', 'valid', 'test')


def candidate_hits(row):
    return [SearchHit(i + 1, item['title'], item['url'], item['snippet'], 'fixture')
            for i, item in enumerate(row['candidates'])]


def validate_v4_record(row):
    task = row.get('task')
    if task not in TASKS:
        raise ValueError('invalid_v4_task')
    validate_record(row if task == 'query' else row | {'task': 'ranking'})
    if row.get('review_status') != 'confirmed' or not isinstance(row.get('reviewer'), str) or not row['reviewer'].strip():
        raise ValueError('human_review_required')
    if task == 'query':
        if len(row['queries']) > 2:
            raise ValueError('v4_query_limit_exceeded')
    else:
        hits = candidate_hits(row)
        validate_decision(OrganizationDecision(list(range(len(hits))), row.get('selections')), hits)
    return row


def v4_messages(row):
    validate_v4_record(row)
    if row['task'] == 'query':
        messages, answer = query_messages(row['query'], 2), {'queries': row['queries']}
    else:
        indices = list(range(len(row['candidates'])))
        random.Random('42:' + row['id']).shuffle(indices)
        hits = candidate_hits(row)
        messages = organization_messages(row['query'], [hits[i] for i in indices])
        answer = {'order': sorted(range(len(indices)), key=lambda i: -row['judgments'][indices[i]]),
                  'selections': [row['selections'][i] for i in indices]}
    return messages + [{'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)}]


def validate_v4_rows(rows):
    if not rows:
        raise ValueError('empty_v4_dataset')
    for row in rows:
        validate_v4_record(row)
    audit_splits([row if row['task'] == 'query' else row | {'task': 'ranking'} for row in rows])
    tasks = [task for task in TASKS if any(row['task'] == task for row in rows)]
    if 'organizing' not in tasks:
        raise ValueError('organizing_data_required')
    for task in tasks:
        for split in SPLITS:
            if not any(row['task'] == task and row['split'] == split for row in rows):
                raise ValueError('v4_split_missing')
    return tasks


def build_v4_dataset(rows, output, tokenizer):
    tasks = validate_v4_rows(rows)
    output = Path(output)
    if output.exists():
        raise ValueError('output_already_exists')
    prepared = [(row, v4_messages(row)) for row in rows]
    for row, messages in prepared:
        tokenize_example(tokenizer, messages, row['task'])
    output.mkdir(parents=True)
    for task in tasks:
        (output / task).mkdir()
        for split in SPLITS:
            selected = [(row, messages) for row, messages in prepared if row['task'] == task and row['split'] == split]
            write_rows(output / task / f'{split}.records.jsonl', [row for row, _ in selected])
            write_rows(output / task / f'{split}.jsonl', [{'messages': messages} for _, messages in selected])
    write_rows(output / 'web_test.records.jsonl', [row for row in rows if row['split'] == 'web_test'])
    manifest = {'schema_version': 2, 'framework_schema_version': 4, 'tasks': tasks,
                'counts': dict(Counter(f'{row["task"]}:{row["split"]}:{row["language"]}' for row in rows)),
                'files': {str(path.relative_to(output)): file_digest(path) for path in sorted(output.rglob('*.jsonl'))}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_v4_dataset(path, manifest):
    path = Path(path)
    tasks = manifest.get('tasks', [])
    if (manifest.get('schema_version') != 2 or manifest.get('framework_schema_version') != 4
            or not isinstance(tasks, list) or not tasks
            or any(not isinstance(task, str) or task not in TASKS for task in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError('invalid_v4_dataset_manifest')
    required = {f'{task}/{split}{suffix}.jsonl' for task in tasks for split in SPLITS
                for suffix in ('', '.records')} | {'web_test.records.jsonl'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('dataset_files_mismatch')
    for name in required:
        if file_digest(path / name) != manifest['files'][name]:
            raise ValueError('dataset_hash_mismatch')
    rows = []
    for task in tasks:
        for split in SPLITS:
            records = read_rows(path / task / f'{split}.records.jsonl')
            if any(row['task'] != task or row['split'] != split for row in records):
                raise ValueError('record_file_split_mismatch')
            expected = [{'messages': v4_messages(row)} for row in records]
            if read_rows(path / task / f'{split}.jsonl') != expected:
                raise ValueError('training_target_mismatch')
            rows.extend(records)
    web = read_rows(path / 'web_test.records.jsonl')
    if any(row['task'] != 'organizing' or row['split'] != 'web_test' for row in web):
        raise ValueError('record_file_split_mismatch')
    rows.extend(web)
    if set(validate_v4_rows(rows)) != set(tasks):
        raise ValueError('dataset_task_mismatch')
    if manifest.get('counts') != dict(Counter(f'{row["task"]}:{row["split"]}:{row["language"]}' for row in rows)):
        raise ValueError('dataset_count_mismatch')
    return manifest
