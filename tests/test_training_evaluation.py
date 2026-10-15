import copy

import pytest

from dona_search.interfaces import QueryPlan
from scripts.eval_models import ranking_acceptance
from scripts.eval_query import evaluate_queries, judge_report


def test_query_predictions_need_review_bound_to_exact_output():
    class Planner:
        def plan(self, query): return QueryPlan(query, [query])
    rows = [{'id': str(i), 'query': f'q{i}', 'language': 'zh' if i < 50 else 'en'} for i in range(100)]
    report = evaluate_queries(rows, Planner())
    assert not report['accepted']
    reviews = [{'id': r['id'], 'prediction_sha256': r['prediction_sha256'],
                'intent_preserved': True, 'reviewer': 'human'} for r in report['records']]
    assert judge_report(report, reviews)['accepted']
    reviews[0]['intent_preserved'] = False
    assert not judge_report(report, reviews)['accepted']
    reviews[0]['prediction_sha256'] = 'stale'
    with pytest.raises(ValueError, match='review_prediction_mismatch'):
        judge_report(report, reviews)


def test_ranking_requires_all_groups_and_no_failed_cases():
    rows = {split: [{'id': str(i), 'language': language, 'quality_status': 'judged',
                    'ranking_status': 'ok', 'has_relevant_candidate': True, 'top1_relevant': True,
                    'ranked': {'ndcg_at_10': 1.0}} for language in ['zh', 'en'] for i in range(count)]
            for split, count in [('public', 100), ('web', 10)]}
    reports = {variant: copy.deepcopy(rows) for variant in ['bf16', 'quantized', 'adapted']}
    assert ranking_acceptance(reports)['accepted']
    reports['adapted']['web'][0]['top1_relevant'] = False
    assert not ranking_acceptance(reports)['accepted']
    reports['adapted']['web'][0]['top1_relevant'] = True
    reports['adapted']['public'][0]['ranking_status'] = 'failed'
    assert not ranking_acceptance(reports)['accepted']
    reports['adapted']['public'] = []
    assert not ranking_acceptance(reports)['accepted']


def test_memory_guard_terminates_probe_and_marks_gate_failed(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import scripts.eval_search as evaluation
    import scripts.train_models as training

    class Process:
        pid = 123
        code = None
        def poll(self): return self.code
        def terminate(self): self.code = -15
        def wait(self, timeout=None): return self.code

    class Sampler:
        def __init__(self, pid): assert pid == 123
        def current(self): return training.MEMORY_LIMIT + 1

    monkeypatch.setattr(evaluation, 'MemorySampler', Sampler)
    monkeypatch.setattr(training.subprocess, 'Popen', lambda *a, **k: Process())
    monkeypatch.setattr(training.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(training.platform, 'machine', lambda: 'arm64')
    args = SimpleNamespace(action='preflight', task='query', base=tmp_path / 'base',
                           output=tmp_path / 'output', data=None, preflight_root=None)
    with pytest.raises(SystemExit):
        training.supervise(args)
    report = json.loads((tmp_path / 'output/query/preflight.json').read_text())
    assert report['passed'] is False and report['process_error'] == 'memory_budget_exceeded'


def test_failed_startup_keeps_all_query_cases(monkeypatch):
    from scripts import eval_query
    def fail(**kwargs):
        raise ValueError('fixture_startup_failed')
    monkeypatch.setattr(eval_query, 'ModelRuntime', fail)
    with eval_query.evaluation_runner(query_only=True) as runner:
        report = evaluate_queries([{'id': 'q', 'query': 'SQLite', 'language': 'en'}], runner)
        assert runner.startup_error == 'ValueError'
    assert report['format_failures'] == 1 and len(report['records']) == 1
    assert not report['accepted']


def test_summary_requires_bound_human_review_without_loading_models(tmp_path):
    import json

    from scripts.eval_models import summarize
    from scripts.training_data import write_rows

    ranked = {split: [{'id': f'{language}:{i}', 'language': language, 'quality_status': 'judged',
                      'ranking_status': 'ok', 'has_relevant_candidate': True, 'top1_relevant': True,
                      'ranked': {'ndcg_at_10': 1.0}} for language in ['zh', 'en'] for i in range(count)]
              for split, count in [('public', 100), ('web', 10)]}
    for variant in ('bf16', 'quantized', 'adapted'):
        (tmp_path / f'{variant}.ranking.json').write_text(json.dumps(ranked | {'dataset_sha256': 'frozen'}))
    class Planner:
        def plan(self, query): return QueryPlan(query, [query])
    rows = [{'id': str(i), 'query': f'q{i}', 'language': 'zh' if i < 50 else 'en'} for i in range(100)]
    report = evaluate_queries(rows, Planner()) | {'dataset_sha256': 'frozen', 'accepted': True}
    (tmp_path / 'adapted.query.json').write_text(json.dumps(report))
    assert not summarize(tmp_path, 'frozen')['accepted']
    reviews = [{'id': r['id'], 'prediction_sha256': r['prediction_sha256'],
                'intent_preserved': True, 'reviewer': 'human'} for r in report['records']]
    review_file = tmp_path / 'reviews.jsonl'
    write_rows(review_file, reviews)
    assert summarize(tmp_path, 'frozen', review_file)['accepted']
    assert summarize(tmp_path, 'frozen')['accepted']
    with pytest.raises(ValueError, match='dataset_mismatch'):
        summarize(tmp_path, 'changed')
