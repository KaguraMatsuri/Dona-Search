"""Three-way, fixed-candidate comparisons; never promote unjudged outputs."""
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dona_search.adapters import file_digest, model_fingerprint
from dona_search.config import RankingConfig
from dona_search.query.planner import MLXQueryPlanner
from dona_search.rerank.gemma import GemmaReranker
from scripts.eval_query import evaluate_queries, evaluation_runner, judge_report
from scripts.eval_search import MemorySampler, evaluate_rows
from scripts.train_models import write_status
from scripts.training_data import read_rows, verify_dataset


def ranking_acceptance(reports):
    adapted = reports['adapted']
    accepted, groups = True, {}
    for subset, expected in [('public', 100), ('web', 10)]:
        for language in ('zh', 'en'):
            key = f'{subset}:{language}'
            rows = [r for r in adapted[subset] if r['language'] == language]
            eligible = [r for r in rows if r.get('has_relevant_candidate')]
            valid = len(rows) == expected and bool(eligible) and all(
                r['quality_status'] == 'judged' and r['ranking_status'] == 'ok' for r in rows)
            valid = valid and all(r['top1_relevant'] for r in eligible)
            scores = {}
            for variant in ('bf16', 'quantized', 'adapted'):
                selected = [r for r in reports[variant][subset] if r['language'] == language]
                if (Counter(r['id'] for r in selected) != Counter(r['id'] for r in rows)
                        or any(r.get('ranked') is None for r in selected)):
                    valid = False
                    scores[variant] = None
                else:
                    scores[variant] = sum(r['ranked']['ndcg_at_10'] for r in selected) / len(selected) if selected else None
            valid = bool(valid and scores['adapted'] is not None and scores['bf16'] is not None
                         and scores['adapted'] >= scores['bf16'])
            groups[key] = {'count': len(rows), 'eligible': len(eligible),
                           'top1_hits': sum(r.get('top1_relevant', False) for r in eligible),
                           'ndcg_at_10': scores, 'accepted': valid}
            accepted = accepted and valid
    return {'accepted': accepted, 'groups': groups}


def summarize(output, dataset_hash, reviews=None):
    reports = {}
    for variant in ('bf16', 'quantized', 'adapted'):
        report = json.loads((output / f'{variant}.ranking.json').read_text())
        if report.get('dataset_sha256') != dataset_hash:
            raise ValueError('evaluation_dataset_mismatch')
        reports[variant] = report
    query = json.loads((output / 'adapted.query.json').read_text())
    if query.get('dataset_sha256') != dataset_hash:
        raise ValueError('evaluation_dataset_mismatch')
    if reviews is not None:
        query = judge_report(query, read_rows(reviews))
        write_status(output / 'adapted.query.json', **query)
    else:
        recorded_reviews = [{'id': r['id'], 'prediction_sha256': r.get('prediction_sha256'),
                             'reviewer': r.get('reviewer'),
                             'intent_preserved': r['intent_status'] == 'confirmed'}
                            for r in query['records'] if r.get('intent_status') in ('confirmed', 'rejected')]
        query = judge_report(query, recorded_reviews)
    ranking = ranking_acceptance(reports)
    result = {'framework_schema_version': 3, 'ranking': ranking, 'query': {'accepted': query['accepted'],
              'status': 'judged' if query['accepted'] else 'unaccepted'},
              'accepted': ranking['accepted'] and query['accepted'], 'dataset_sha256': dataset_hash}
    write_status(output / 'acceptance.json', **result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--bf16', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-reviews', type=Path)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    manifest = verify_dataset(args.data)
    dataset_hash = file_digest(args.data / 'manifest.json')
    if manifest.get('framework_schema_version') == 4:
        from scripts.eval_v4 import evaluate_v4
        evaluate_v4(args, manifest, dataset_hash)
        return
    if args.summarize_only:
        summarize(args.output, dataset_hash, args.query_reviews)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    query_rows = read_rows(args.data / 'query/test.records.jsonl')
    public = read_rows(args.data / 'ranking/test.records.jsonl')
    web = read_rows(args.data / 'web_test.records.jsonl')
    for variant, path in [('bf16', args.bf16), ('quantized', args.base), ('adapted', args.base)]:
        for task in ('query', 'ranking'):
            adapter = args.adapters / task if variant == 'adapted' else None
            cfg = RankingConfig(model_path=path, adapter_path=adapter,
                                max_prompt_tokens=2048 if task == 'query' else 8192)
            kwargs = ({'build_query': lambda cfg=cfg: MLXQueryPlanner(cfg), 'query_only': True} if task == 'query'
                      else {'build_ranker': lambda cfg=cfg: GemmaReranker(cfg)})
            with MemorySampler() as memory, evaluation_runner(**kwargs) as runner:
                if task == 'query':
                    report = evaluate_queries(query_rows, runner)
                else:
                    report = {'public': evaluate_rows(public, runner), 'web': evaluate_rows(web, runner)}
                report['startup_error'] = runner.startup_error
            report.update(model_fingerprint=model_fingerprint(path),
                          adapter_manifest_sha256=file_digest(adapter / 'dona_adapter_manifest.json') if adapter else None,
                          dataset_sha256=dataset_hash, peak_physical_bytes=memory.peak)
            write_status(args.output / f'{variant}.{task}.json', **report)
            gc.collect()
    summarize(args.output, dataset_hash, args.query_reviews)


if __name__ == '__main__':
    main()
