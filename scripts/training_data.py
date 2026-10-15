"""Human-labelled task records, split auditing, and inference-matched targets."""
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from dona_search.adapters import file_digest
from dona_search.interfaces import QueryPlan
from dona_search.query.planner import query_messages, validate_plan
from dona_search.rerank.gemma import ranking_messages

COUNTS = {'query': {'train': 400, 'valid': 100, 'test': 100},
          'ranking': {'train': 2000, 'valid': 200, 'test': 200}}
PROMPT_LIMIT = {'query': 2048, 'ranking': 8192, 'organizing': 8192}


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)


def identity(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).casefold().split()).encode()).hexdigest()


def validate_record(row):
    for name in ('id', 'group_id', 'query'):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError('invalid_record_' + name)
    if row.get('task') not in COUNTS or row.get('split') not in ('train', 'valid', 'test', 'web_test'):
        raise ValueError('invalid_record_task_or_split')
    if row.get('language') not in ('zh', 'en'):
        raise ValueError('invalid_record_language')
    source = row.get('source', {})
    if any(not isinstance(source.get(k), str) or not source[k].strip()
           for k in ('name', 'revision', 'license', 'url')):
        raise ValueError('source_provenance_missing')
    if row['task'] == 'query':
        if (row.get('review_status') != 'confirmed' or not isinstance(row.get('reviewer'), str)
                or not row['reviewer'].strip()):
            raise ValueError('human_review_required')
        validate_plan(QueryPlan(row['query'], row.get('queries')), row['query'])
        if row['split'] == 'web_test':
            raise ValueError('invalid_query_split')
    else:
        candidates, grades = row.get('candidates'), row.get('judgments')
        maximum = 16 if row['split'] == 'train' else 40
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= maximum:
            raise ValueError('invalid_candidate_count')
        if not isinstance(grades, list) or len(grades) != len(candidates):
            raise ValueError('judgments_length_mismatch')
        limit = 1 if source['name'] == 'miracl' else 3
        if any(type(g) is not int or not 0 <= g <= limit for g in grades):
            raise ValueError('invalid_relevance_grade')
        if row.get('judgment_source') not in ('human_confirmed', 'published_human'):
            raise ValueError('human_judgments_required')
        if row['judgment_source'] == 'published_human' and source['name'] not in ('miracl', 't2ranking'):
            raise ValueError('unknown_annotation_source')
        if row['split'] == 'web_test' and (row['judgment_source'] != 'human_confirmed'
                or not isinstance(row.get('reviewer'), str) or not row['reviewer'].strip()):
            raise ValueError('human_review_required')
        for candidate in candidates:
            if any(not isinstance(candidate.get(k), str) for k in ('title', 'url', 'snippet')):
                raise ValueError('invalid_candidate')
        if row['split'] == 'train' and (0 not in grades or max(grades) < positive_grade(row)):
            raise ValueError('same_query_contrast_required')
    return row


def positive_grade(row):
    return 1 if row['source']['name'] == 'miracl' else 2


def audit_splits(rows):
    seen_ids, owners = set(), {}
    for row in rows:
        validate_record(row)
        key = (row['task'], row['id'])
        if key in seen_ids:
            raise ValueError('duplicate_record_id')
        seen_ids.add(key)
        keys = [('group', row['group_id']), ('query', identity(row['query']))]
        keys += [('query', identity(q)) for q in row.get('queries', [])]
        # Includes cross-task leakage, not merely within-task duplicate IDs.
        for key in keys:
            if key in owners and owners[key] != row['split']:
                raise ValueError('cross_split_overlap')
            owners[key] = row['split']


def messages_for(row, seed=42):
    if row['task'] == 'query':
        messages = query_messages(row['query'])
        answer = {'queries': validate_plan(QueryPlan(row['query'], row['queries']), row['query']).queries}
    else:
        order = list(range(len(row['candidates'])))
        random.Random(f'{seed}:{row["id"]}').shuffle(order)
        candidates = [row['candidates'][i] for i in order]
        grades = [row['judgments'][i] for i in order]
        messages = ranking_messages(row['query'], candidates)
        answer = sorted(range(len(order)), key=lambda i: -grades[i])
    return messages + [{'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)}]


def tokenize_example(tokenizer, messages, task):
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError('training_inference_prefix_mismatch')
    if len(prefix) > PROMPT_LIMIT[task] or len(tokens) - len(prefix) > (1024 if task == 'organizing' else 256):
        raise ValueError('training_context_limit')
    if len(tokens) <= len(prefix):
        raise ValueError('missing_training_completion')
    return tokens, len(prefix)


def build_dataset(rows, output, tokenizer):
    output = Path(output)
    if output.exists():
        raise ValueError('output_already_exists')
    audit_splits(rows)
    counts = validate_counts(rows)
    prepared = [(r, messages_for(r)) for r in rows]
    rejected = []
    for row, messages in prepared:
        try:
            tokenize_example(tokenizer, messages, row['task'])
        except ValueError as error:
            rejected.append({'id': row['id'], 'task': row['task'], 'error': str(error)})
    if rejected:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + '.rejected.json').write_text(json.dumps(rejected, indent=2))
        raise ValueError('training_records_rejected')
    output.mkdir(parents=True, exist_ok=False)
    for task, splits in COUNTS.items():
        folder = output / task
        folder.mkdir()
        for split in splits:
            selected = [(r, m) for r, m in prepared if r['task'] == task and r['split'] == split]
            write_rows(folder / f'{split}.jsonl', [{'messages': m} for _, m in selected])
            write_rows(folder / f'{split}.records.jsonl', [r for r, _ in selected])
    write_rows(output / 'web_test.records.jsonl', [r for r in rows if r['split'] == 'web_test'])
    manifest = {'schema_version': 1, 'seed': 42, 'counts': [
        {'task': t, 'split': s, 'language': l, 'count': n} for (t, s, l), n in sorted(counts.items())],
        'files': {str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob('*.jsonl'))}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_counts(rows):
    counts = Counter((r['task'], r['split'], r['language']) for r in rows)
    for task, splits in COUNTS.items():
        for split, total in splits.items():
            for language in ('zh', 'en'):
                if counts[task, split, language] != total // 2:
                    raise ValueError(f'dataset_count_mismatch:{task}:{split}:{language}')
    for language in ('zh', 'en'):
        if counts['ranking', 'web_test', language] != 10:
            raise ValueError('web_test_review_required')
        query_sizes = {len(r['queries']) for r in rows
                       if r['task'] == 'query' and r['split'] == 'train' and r['language'] == language}
        if query_sizes != {1, 2, 3}:
            raise ValueError('query_case_coverage_missing')
    mix = Counter((r['language'], r['source']['name']) for r in rows
                  if r['task'] == 'ranking' and r['split'] == 'train')
    if mix != {('zh', 't2ranking'): 800, ('zh', 'miracl'): 200, ('en', 'miracl'): 1000}:
        raise ValueError('ranking_source_mix_mismatch')
    return counts


def verify_dataset(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema_version') == 2:
        from scripts.v4_data import verify_v4_dataset
        return verify_v4_dataset(path, manifest)
    if manifest.get('schema_version') != 1:
        raise ValueError('invalid_dataset_manifest')
    required = {f'{t}/{s}{suffix}.jsonl' for t in COUNTS for s in COUNTS[t]
                for suffix in ('', '.records')} | {'web_test.records.jsonl'}
    if set(manifest['files']) != required:
        raise ValueError('dataset_files_mismatch')
    for name, digest in manifest['files'].items():
        if file_digest(path / name) != digest:
            raise ValueError('dataset_hash_mismatch')
    rows = [r for name in sorted(required) if '.records.' in name for r in read_rows(path / name)]
    audit_splits(rows)
    validate_counts(rows)
    for task, splits in COUNTS.items():
        for split in splits:
            originals = read_rows(path / task / f'{split}.records.jsonl')
            if any(r['task'] != task or r['split'] != split for r in originals):
                raise ValueError('record_file_split_mismatch')
            if read_rows(path / task / f'{split}.jsonl') != [{'messages': messages_for(r)} for r in originals]:
                raise ValueError('training_target_mismatch')
    return manifest
