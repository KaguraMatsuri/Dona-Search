import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dona_search.config import read_config, validate_config
from dona_search.query.planner import MLXQueryPlanner, validate_plan
from dona_search.runtime import ModelRuntime
from scripts.training_data import read_rows


@contextmanager
def evaluation_runner(**builders):
    runtime, startup_error = None, None
    try:
        runtime = ModelRuntime(**builders)
    except Exception as error:  # noqa: BLE001 - failed startups remain in the evaluation denominator
        startup_error = type(error).__name__

    class Runner:
        def plan(self, query):
            if runtime is None:
                raise RuntimeError(startup_error)
            return runtime.call(runtime.query_processor.plan, query)

        def rank(self, query, hits):
            if runtime is None:
                raise RuntimeError(startup_error)
            return runtime.call(runtime.ranker.rank, query, hits)

        def organize(self, query, hits):
            if runtime is None:
                raise RuntimeError(startup_error)
            return runtime.call(runtime.organizer.organize, query, hits)

    runner = Runner()
    runner.startup_error = startup_error
    try:
        yield runner
    finally:
        if runtime is not None:
            runtime.close()


def prediction_digest(query, queries):
    return hashlib.sha256(json.dumps({'query': query, 'queries': queries},
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate_queries(rows, planner, max_queries=3):
    records = []
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate_fixture_id')
    for row in rows:
        record = {'id': row['id'], 'query': row['query'], 'language': row['language'],
                  'status': 'failed', 'intent_status': 'unjudged'}
        try:
            plan = validate_plan(planner.plan(row['query']), row['query'], max_queries)
            record.update(status='ok', queries=plan.queries,
                          prediction_sha256=prediction_digest(row['query'], plan.queries))
        except Exception as error:  # noqa: BLE001 - preserve failed cases in evaluation denominator
            record['error'] = type(error).__name__
        records.append(record)
    return {'records': records, 'accepted': False, 'search_requests': 0,
            'format_failures': sum(r['status'] != 'ok' for r in records)}


def judge_report(report, reviews):
    by_id = {r['id']: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) - {r['id'] for r in report['records']}:
        raise ValueError('invalid_review_ids')
    records = []
    for original in report['records']:
        record = dict(original)
        review = by_id.get(record['id'])
        record['intent_status'] = 'unjudged'
        if review is not None:
            if (record['status'] != 'ok' or not isinstance(review.get('reviewer'), str)
                    or not review['reviewer'].strip()
                    or type(review.get('intent_preserved')) is not bool
                    or review.get('prediction_sha256') != record.get('prediction_sha256')
                    or record.get('prediction_sha256') != prediction_digest(record['query'], record['queries'])):
                raise ValueError('review_prediction_mismatch')
            record['intent_status'] = 'confirmed' if review['intent_preserved'] else 'rejected'
            record['reviewer'] = review['reviewer']
        records.append(record)
    counts = Counter(r['language'] for r in records)
    accepted = counts == {'zh': 50, 'en': 50} and all(
        r['status'] == 'ok' and r['intent_status'] == 'confirmed' for r in records)
    return report | {'records': records, 'accepted': accepted}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'judge'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reviews', type=Path)
    args = parser.parse_args()
    if args.action == 'judge':
        if args.reviews is None: parser.error('--reviews')
        report = judge_report(json.loads(args.input.read_text()), read_rows(args.reviews))
    else:
        if args.config is None: parser.error('--config')
        raw, base = read_config(args.config)
        cfg = validate_config(raw, base, required_models=('query_model',))
        with evaluation_runner(build_query=lambda: MLXQueryPlanner(cfg.query_model), query_only=True) as runner:
            report = evaluate_queries(read_rows(args.input), runner)
            report['startup_error'] = runner.startup_error
    report['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'accepted': report['accepted'], 'records': len(report['records'])}))


if __name__ == '__main__':
    main()
