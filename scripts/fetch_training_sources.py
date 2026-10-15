"""Pinned official sources. Only selected, human-judged passages are retained."""
import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dona_search.adapters import file_digest
from scripts.training_data import write_rows

T2 = ('THUIR/T2Ranking', '2a369a430a70979223f1b9a41b1919774d46b432')
MIRACL = ('miracl/miracl', '5be20db9509754dadad47689368639fcec739c00')
CORPUS = ('miracl/miracl-corpus', 'd921ec7e349ce0d28daf30b2da9da5ee698bef0d')


def fetch_mqr(output):
    import httpx

    from scripts.training_data import identity
    # Author-linked, Stack Exchange-only files; deliberately exclude Quora/Paralex mixtures.
    files = [('train', '1n425bntfk6G5Cf-Oiq_Y5-w2I5Hmd6Jl', 200),
             ('valid', '1_Pw8lOVMQwgggwk_qrpkPpSSUherd0wV', 50),
             ('test', '1maZTFL2D-7OSTnTvC77_tWLQhSnUiReC', 50)]
    drafts, receipts = [], []
    seen = set()
    for split, file_id, count in files:
        url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download'
        path = output / f'mqr.{split}.tsv'
        with httpx.stream('GET', url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('content-type', ''):
                raise ValueError('source_download_not_data')
            with path.open('wb') as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        digest = file_digest(path)
        receipts.append({'url': url, 'file_id': file_id, 'sha256': digest})
        with path.open(encoding='utf-8') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            if not {'ill formed', 'well formed', 'category'} <= set(reader.fieldnames or []):
                raise ValueError('mqr_schema_mismatch')
            rows = list(reader)
        random.Random('42:mqr:' + split).shuffle(rows)
        selected = 0
        for record in rows:
            original, target = record['ill formed'], record['well formed']
            if not original or not target:
                continue
            group = 'mqr:' + identity(record['category'] + ':' + target)
            if group in seen:
                continue
            seen.add(group)
            drafts.append({'id': group, 'group_id': group, 'task': 'query', 'split': split,
                'language': 'en', 'query': original, 'queries': [target],
                'review_status': 'pending', 'reviewer': None,
                'source': {'name': 'mqr_stackexchange', 'revision': digest,
                           'license': 'CC-BY-SA-4.0', 'url': 'https://github.com/ZeweiChu/MQR',
                           'download_url': url, 'category': record['category']}})
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError('insufficient_mqr_records')
    write_rows(output / 'query.en.pending.jsonl', drafts)
    (output / 'sources.json').write_text(json.dumps(receipts, indent=2))
    print(json.dumps({'status': 'awaiting_human_review', 'records': len(drafts)}))


def fetch(repo, revision, name, receipts):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision))
    receipts.append({'repo': repo, 'revision': revision, 'file': name, 'sha256': file_digest(path)})
    return path


def read_topics(path):
    result = {}
    for line in path.open(encoding='utf-8'):
        key, value = line.rstrip('\n').split('\t', 1)
        if key.lower() in ('qid', 'query_id'):
            continue
        result[key] = value
    return result


def read_qrels(path):
    result = defaultdict(dict)
    for line in path.open(encoding='utf-8'):
        qid, _, pid, grade = line.split()
        if qid.lower() == 'qid' and grade.lower() == 'rel':
            continue
        grade = int(grade)
        if pid in result[qid] and result[qid][pid] != grade:
            raise ValueError('annotation_conflict')
        result[qid][pid] = grade
    return result


def select_groups(topics, qrels, count, threshold, seed):
    keys = sorted(q for q in qrels if q in topics and 0 in qrels[q].values()
                  and max(qrels[q].values()) >= threshold)
    random.Random(seed).shuffle(keys)
    if len(keys) < count:
        raise ValueError('insufficient_labelled_queries')
    selected = {}
    for q in keys[:count]:
        pairs = sorted(qrels[q].items())
        random.Random(seed + ':' + q).shuffle(pairs)
        positive = next(p for p in pairs if p[1] >= threshold)
        negative = next(p for p in pairs if p[1] == 0)
        chosen = [positive, negative] + [p for p in pairs if p[0] not in (positive[0], negative[0])]
        selected[q] = dict(chosen[:16])
    return selected


def source_records(name, language, split, topics, groups, passages, revision):
    repo = T2[0] if name == 't2ranking' else MIRACL[0]
    for qid, judgments in groups.items():
        ids = list(judgments)
        if any(pid not in passages for pid in ids):
            raise ValueError('annotated_passage_missing')
        yield {'id': f'{name}:{language}:{qid}', 'group_id': f'{name}:{language}:{qid}',
               'task': 'ranking', 'language': language, 'split': split, 'query': topics[qid],
               'candidates': [passages[pid] for pid in ids], 'judgments': [judgments[pid] for pid in ids],
               'judgment_source': 'published_human',
               'source': {'name': name, 'revision': revision, 'license': 'Apache-2.0' if name == 't2ranking' else 'Apache-2.0; Wikipedia source terms',
                          'url': f'https://huggingface.co/datasets/{repo}', 'passage_ids': ids,
                          'corpus_revision': CORPUS[1] if name == 'miracl' else revision}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--mqr-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mqr_only:
        fetch_mqr(args.output)
        return
    receipts, collections, drafts = [], [], []
    # Official dev sets are divided deterministically into our valid/test sets.
    for name, language, repo, revision in [('t2ranking', 'zh', *T2), ('miracl', 'zh', *MIRACL), ('miracl', 'en', *MIRACL)]:
        for upstream in ('train', 'dev'):
            if name == 't2ranking':
                topics_file, qrels_file = f'data/queries.{upstream}.tsv', f'data/qrels.{upstream}.tsv'
            else:
                prefix = f'miracl-v1.0-{language}'
                topics_file = f'{prefix}/topics/topics.{prefix}-{upstream}.tsv'
                qrels_file = f'{prefix}/qrels/qrels.{prefix}-{upstream}.tsv'
            topics = read_topics(fetch(repo, revision, topics_file, receipts))
            qrels = read_qrels(fetch(repo, revision, qrels_file, receipts))
            quota = (800 if name == 't2ranking' else 200 if language == 'zh' else 1000) if upstream == 'train' else (100 if language == 'zh' else 200)
            selected = select_groups(topics, qrels, quota, 2 if name == 't2ranking' else 1,
                                     f'42:{name}:{language}:{upstream}')
            if upstream == 'train':
                collections.append((name, language, 'train', topics, selected, revision))
            else:
                entries = list(selected.items())
                for split, items in [('valid', entries[:quota // 2]), ('test', entries[quota // 2:])]:
                    collections.append((name, language, split, topics, dict(items), revision))
            if name == 't2ranking':
                # Query-task seeds are disjoint from selected ranking queries.
                unused = sorted(set(topics) - set(selected))
                random.Random('query:' + upstream).shuffle(unused)
                splits = ['train'] * 200 if upstream == 'train' else ['valid'] * 50 + ['test'] * 50
                for qid, split in zip(unused, splits):
                    drafts.append({'id': f'query:t2:{qid}', 'group_id': f't2ranking:zh:{qid}',
                        'task': 'query', 'language': 'zh', 'split': split, 'query': topics[qid],
                        'queries': None, 'review_status': 'pending', 'reviewer': None,
                        'source': {'name': 't2ranking', 'revision': revision, 'license': 'Apache-2.0',
                                   'url': f'https://huggingface.co/datasets/{repo}'}})
    write_rows(args.output / 'query.zh.pending.jsonl', drafts)
    (args.output / 'selection.json').write_text(json.dumps([
        {'source': n, 'language': l, 'split': s, 'queries': g} for n, l, s, _, g, _ in collections]))
    if not args.metadata_only:
        from huggingface_hub import list_repo_files
        passages = {}
        for name, language in [('t2ranking', 'zh'), ('miracl', 'zh'), ('miracl', 'en')]:
            wanted = {pid for n, l, _, _, groups, _ in collections if (n, l) == (name, language)
                      for group in groups.values() for pid in group}
            found = {}
            if name == 't2ranking':
                path = fetch(*T2, 'data/collection.tsv', receipts)
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        pid, text = line.rstrip('\n').split('\t', 1)
                        if pid in wanted:
                            found[pid] = {'title': '', 'url': '', 'snippet': text}
            else:
                names = sorted(n for n in list_repo_files(CORPUS[0], repo_type='dataset', revision=CORPUS[1])
                               if n.startswith(f'miracl-corpus-v1.0-{language}/') and n.endswith('.jsonl.gz'))
                for filename in names:
                    path = fetch(*CORPUS, filename, receipts)
                    with gzip.open(path, 'rt', encoding='utf-8') as stream:
                        for line in stream:
                            record = json.loads(line)
                            if record['docid'] in wanted:
                                found[record['docid']] = {'title': record['title'], 'url': '', 'snippet': record['text']}
                    print(json.dumps({'status': 'corpus_scan', 'source': name, 'language': language,
                                      'found': len(found), 'required': len(wanted)}), flush=True)
                    if len(found) == len(wanted):
                        break
            passages[name, language] = found
        rows = [r for n, l, s, topics, groups, rev in collections
                for r in source_records(n, l, s, topics, groups, passages[n, l], rev)]
        write_rows(args.output / 'ranking.records.jsonl', rows)
    (args.output / 'sources.json').write_text(json.dumps(receipts, indent=2))
    print(json.dumps({'status': 'source_metadata_ready' if args.metadata_only else 'ranking_sources_ready',
                      'pending_query_reviews': len(drafts)}))


if __name__ == '__main__':
    main()
