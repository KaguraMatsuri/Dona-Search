import pytest

from scripts.eval_search import evaluate_rows, quality_metrics


def row():
    return {'id': 'fixture', 'query': 'q', 'candidates': [
        {'title': 'a', 'url': 'https://a.test/', 'snippet': 'A'},
        {'title': 'b', 'url': 'https://b.test/', 'snippet': 'B'},
    ]}


class Ranker:
    def rank(self, query, hits):
        return [1, 0]


def test_unjudged_does_not_claim_quality():
    record = evaluate_rows([row()], Ranker())[0]
    assert record['quality_status'] == 'unjudged'
    assert 'ranked' not in record and record['order'] == [1, 0]


def test_confirmed_judgments_compare_same_candidates():
    example = row() | {'judgments': [0, 3], 'judgment_source': 'human_confirmed'}
    record = evaluate_rows([example], Ranker())[0]
    assert record['ranked']['ndcg_at_10'] == 1
    assert record['baseline']['ndcg_at_10'] < 1
    assert record['ranked']['mrr_at_10'] == 1
    assert record['baseline']['mrr_at_10'] == .5
    example['judgment_source'] = 'draft'
    assert evaluate_rows([example], Ranker())[0]['quality_status'] == 'unconfirmed_judgments'


def test_invalid_or_unanswerable_grades():
    assert quality_metrics([0, 0], [1, 0]) == {'ndcg_at_10': 0.0, 'mrr_at_10': 0.0}
    with pytest.raises(ValueError): quality_metrics([True, 3], [0, 1])
    with pytest.raises(ValueError): quality_metrics([0, 3], [0, 0])


def test_rank_failure_is_recorded_separately_from_baseline():
    class Invalid:
        def rank(self, query, hits): return [0, 0]
    record = evaluate_rows([row()], Invalid())[0]
    assert record['ranking_status'] == 'failed'
    assert record['order'] == [0, 1]
    judged = row() | {'judgments': [0, 3], 'judgment_source': 'human_confirmed'}
    record = evaluate_rows([judged], Invalid())[0]
    assert record['ranked'] is None
    assert record['fallback'] == record['baseline']


def test_rank_only_config_ignores_unused_query_path_but_checks_rank_path(tmp_path):
    from dona_search.config import validate_config
    (tmp_path / 'rank').mkdir()
    raw = {'schema_version':3, 'query_model':{'model_path':'missing-query'}, 'ranking':{'model_path':'rank'}}
    config = validate_config(raw, tmp_path, required_models=('ranking',))
    assert config.query_model.model_path == tmp_path/'missing-query'
    assert config.ranking.model_path == tmp_path/'rank'
    with pytest.raises(ValueError, match='model_path_missing'):
        validate_config(raw, tmp_path)
    raw['ranking']['model_path'] = 'missing-rank'
    with pytest.raises(ValueError, match='model_path_missing'):
        validate_config(raw, tmp_path, required_models=('ranking',))


def test_rank_only_cli_runs_without_query_weights(tmp_path, monkeypatch, capsys):
    import json
    import sys
    import yaml
    from scripts import eval_search

    rank_dir = tmp_path / 'rank'
    rank_dir.mkdir()
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'schema_version': 3, 'query_model': {'model_path': 'missing-query'},
                                    'ranking': {'model_path': 'rank'}}))
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.json'
    source.write_text(json.dumps(row()) + '\n')
    events = []
    class FakeRanker(Ranker):
        def __init__(self, cfg):
            assert cfg.model_path == rank_dir
            events.append('load_rank')
    class FakeRuntime:
        def __init__(self, build_query=None, *, build_ranker):
            assert build_query is None
            self.ranker = build_ranker()
        def call(self, fn, *args): return fn(*args)
        def close(self): events.append('close_rank')
    monkeypatch.setattr(eval_search, 'GemmaReranker', FakeRanker)
    monkeypatch.setattr(eval_search, 'ModelRuntime', FakeRuntime)
    monkeypatch.setattr(sys, 'argv', ['eval_search', '--input', str(source), '--output', str(output), '--config', str(config)])
    eval_search.main()
    report = json.loads(output.read_text())
    assert report['ranking_failures'] == report['search_requests'] == 0
    assert report['records'][0]['order'] == [1, 0]
    assert events == ['load_rank', 'close_rank']
    capsys.readouterr()
