import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest
from pydantic import ValidationError

from dona_search.config import PipelineConfig
from dona_search.interfaces import ProviderResult, QueryPlan, SearchHit
from dona_search.pipeline import SearchPipeline
from dona_search.query.planner import parse_plan
from dona_search.search.scheduler import SearchScheduler


@pytest.mark.parametrize('response', [
    'bad', '[]', '{"queries": []}', '{"queries": [true]}', '{"queries": [" "]}',
    '{"queries": ["a", "b", "c", "d"]}', '{"queries": ["a"], "answer":"invented"}',
])
def test_invalid_plans_fail_without_raw_query_fallback(response):
    with pytest.raises((ValueError, TypeError)):
        parse_plan(response, '原始查询')


def test_plan_preserves_complete_queries_and_deduplicates():
    plan = parse_plan(json.dumps({'queries': [' "SQLite" 3.46 -Django ', '"SQLite" 3.46 -Django', 'SQLite 3.46 文档']}), 'raw')
    assert plan.raw == 'raw'
    assert plan.queries == ['"SQLite" 3.46 -Django', 'SQLite 3.46 文档']


@pytest.mark.parametrize('config', [
    {'max_queries': 4}, {'providers': []},
    {'providers': [{'type': 'search_google', 'name': 'g', 'params': {'q': 'override'}}]},
    {'providers': [{'type': 'search_xpath', 'name': 'custom', 'search_url': 'https://x.test'}]},
    {'providers': [{'type': 'search_xpath', 'name': 'custom', 'search_url': 'https://x.test',
                    'rules': {'item_xpath': '//*', 'title_xpath': '///', 'url_xpath': './@href', 'snippet_xpath': '.'}}]},
])
def test_v3_invalid_config(config):
    with pytest.raises((ValueError, ValidationError)):
        PipelineConfig.model_validate(config)


def fixtures(events, mode='ok', hit_count=25):
    class Planner:
        def __init__(self): events.append(('load_query', get_ident()))
        def warmup(self):
            events.append(('warm_query', get_ident()))
            if mode == 'warm_failure': raise ValueError('warm_failure')
        def cleanup(self): events.append(('cleanup_query', get_ident()))
        def close(self): events.append(('close_query', get_ident()))
        def plan(self, raw):
            events.append(('plan', get_ident()))
            if mode == 'query_failure': raise ValueError('query_failure')
            return QueryPlan(raw, ['SQLite 3.46 文档', 'SQLite 3.46 -Django', '"SQLite" 3.46'])
    class Ranker:
        def __init__(self):
            events.append(('load_rank', get_ident()))
            if mode == 'load_failure': raise ValueError('load_failure')
        def warmup(self): events.append(('warm_rank', get_ident()))
        def cleanup(self): events.append(('cleanup_rank', get_ident()))
        def close(self): events.append(('close_rank', get_ident()))
        def rank(self, raw, hits):
            assert raw == '原始查询'
            events.append(('rank', get_ident()))
            return [0] if mode == 'rank_failure' else list(reversed(range(len(hits))))
    class Browser:
        def __init__(self, config, timeout): self.seen = []
        def start(self): events.append(('start_browser', get_ident()))
        def close(self): events.append(('close_browser', get_ident()))
        def search(self, provider, query, url, k):
            self.seen.append((provider.name, query, url))
            events.append(('search', get_ident()))
            assert query != '原始查询'
            if mode == 'blocked' and provider.name == 'google':
                return ProviderResult(provider.name, 'blocked', error_code='fixture_captcha')
            hits = [SearchHit(i+1, f'原始标题 {i}', f'https://{provider.name}.test/{query}/{i}',
                              '原始摘要', provider.name) for i in range(min(k, hit_count))]
            if hits:
                hits[0].url = 'https://duplicate.test/'
            return ProviderResult(provider.name, 'ok' if hits else 'empty', hits)
    return Planner, Ranker, Browser


def test_full_pipeline_loads_once_fifo_fair_merge_and_verbatim_output():
    events = []
    planner, ranker, browser = fixtures(events)
    with SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=ranker, browser_factory=browser) as pipeline:
        ids = (id(pipeline.runtime.query_processor), id(pipeline.runtime.ranker))
        first = pipeline.search_report('原始查询')
        second = pipeline.search_report('原始查询')
        assert ids == (id(pipeline.runtime.query_processor), id(pipeline.runtime.ranker))
        assert len(first.tasks) == 9 and first.candidate_count == 40 and len(first.results) == 10
        assert all(len(t.outcome.hits) == 20 for t in first.tasks)
        assert [t.task.provider for t in first.tasks] == ['google','bing','yandex'] * 3
        assert first.query_status == first.ranking_status == 'ok'
        assert len(first.candidates[0].origins) == 9
        assert all(r.preview == '原始摘要' and r.summary is None for r in first.results)
        assert [r.url for r in first.results] == [r.url for r in second.results]
        assert {h.provider for h in first.candidates} == {'google','bing','yandex'}
        assert {t.task.query for t in first.tasks} == set(first.search_queries)
    names = [e[0] for e in events]
    assert names.count('load_query') == names.count('load_rank') == 1
    assert names.index('load_rank') < names.index('warm_query') < names.index('warm_rank') < names.index('plan')
    assert names.count('close_query') == names.count('close_rank') == names.count('close_browser') == 1
    gpu_threads = {thread for name, thread in events if name not in ('search','start_browser','close_browser')}
    search_threads = {thread for name, thread in events if name in ('search','start_browser','close_browser')}
    assert len(gpu_threads) == len(search_threads) == 1 and gpu_threads != search_threads


@pytest.mark.parametrize('mode', ['load_failure', 'warm_failure'])
def test_strict_startup_releases_all_resources(mode):
    events = []
    planner, ranker, browser = fixtures(events, mode)
    with pytest.raises(ValueError, match=mode):
        SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=ranker, browser_factory=browser)
    names = [e[0] for e in events]
    assert 'plan' not in names and 'search' not in names
    assert names.count('close_query') == names.count('close_browser') == 1
    assert names.count('close_rank') == (mode == 'warm_failure')


def test_query_failure_never_searches_and_does_not_unload_models():
    events = []
    planner, ranker, browser = fixtures(events, 'query_failure')
    with SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=ranker, browser_factory=browser) as pipeline:
        report = pipeline.search_report('原始查询')
        assert report.query_status == 'failed' and report.tasks == [] and report.results == []
        assert pipeline.browser.seen == []
        assert not any(name.startswith('close') for name, _ in events)


def test_rank_failure_preserves_merged_order():
    events = []
    planner, ranker, browser = fixtures(events, 'rank_failure')
    with SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=ranker, browser_factory=browser) as pipeline:
        report = pipeline.search_report('原始查询')
        assert report.ranking_status == 'failed'
        assert [r.url for r in report.results] == [h.url for h in report.candidates[:10]]


def test_captcha_skips_engine_for_this_request_only():
    events = []
    planner, ranker, browser = fixtures(events, 'blocked')
    with SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=ranker, browser_factory=browser) as pipeline:
        for _ in range(2):
            report = pipeline.search_report('原始查询')
            assert [t.outcome.status for t in report.tasks if t.task.provider == 'google'] == ['blocked','skipped','skipped']
            assert len([t for t in report.tasks if t.outcome.status == 'ok']) == 6
        assert len(pipeline.browser.seen) == 14


def test_empty_pool_never_invokes_ranker():
    events = []
    planner, ranker, browser = fixtures(events, hit_count=0)
    with SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=ranker, browser_factory=browser) as pipeline:
        report = pipeline.search_report('原始查询')
        assert report.ranking_status == 'no_results' and len(report.tasks) == 9
        assert 'rank' not in [name for name, _ in events]


def test_scheduler_shutdown_stops_remaining_tasks_and_releases_on_worker():
    entered, finish = Event(), Event()
    calls, releases = [], []
    class Provider:
        name = 'test'
        class config:
            top_k = 20
        def search(self, query, k):
            calls.append(query); entered.set(); finish.wait(2)
            return ProviderResult(self.name, 'empty')
    scheduler = SearchScheduler([Provider()])
    with ThreadPoolExecutor(max_workers=2) as callers:
        pending = callers.submit(scheduler.collect, ['one','two','three'])
        assert entered.wait(2)
        closing = callers.submit(scheduler.close, lambda: releases.append(get_ident()))
        assert scheduler._cancel.wait(2)
        finish.set()
        closing.result(2)
        assert len(pending.result(2)) == 1
    assert calls == ['one'] and len(releases) == 1
    scheduler.close()


def test_runtime_close_failure_still_closes_second_model_and_browser():
    events = []
    planner, ranker, browser = fixtures(events)
    class BrokenRanker(ranker):
        def close(self):
            super().close()
            raise ValueError('fixture_close_failure')
    pipeline = SearchPipeline(PipelineConfig(), build_query=planner, build_ranker=BrokenRanker, browser_factory=browser)
    with pytest.raises(ValueError): pipeline.close()
    assert [n for n, _ in events].count('close_query') == 1
    assert [n for n, _ in events].count('close_browser') == 1
    assert pipeline.client.is_closed


def test_cli_runs_complete_pipeline_and_reports_fixture_tasks(tmp_path, monkeypatch, capsys):
    from dona_search import __main__ as cli
    events = []
    planner, ranker, browser = fixtures(events)
    def factory(config, **kwargs):
        return SearchPipeline(config, build_query=planner, build_ranker=ranker, browser_factory=browser, **kwargs)
    monkeypatch.setattr(cli, 'SearchPipeline', factory)
    config = tmp_path / 'v3.yaml'
    config.write_text('schema_version: 3\n')
    assert cli.main(['--config', str(config), '原始查询']) == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[1])['mode'] == 'normal'
    report = json.loads(lines[2])
    assert report['query_status'] == 'ok' and len(report['tasks']) == 9
    assert report['search_queries'] == ['SQLite 3.46 文档', 'SQLite 3.46 -Django', '"SQLite" 3.46']
    assert [n for n, _ in events].count('close_browser') == 1


def test_cli_query_failure_returns_failure_exit_code(tmp_path, monkeypatch, capsys):
    from dona_search import __main__ as cli
    events = []
    planner, ranker, browser = fixtures(events, 'query_failure')
    monkeypatch.setattr(cli, 'SearchPipeline', lambda config, **kwargs: SearchPipeline(
        config, build_query=planner, build_ranker=ranker, browser_factory=browser, **kwargs))
    config = tmp_path / 'v3.yaml'
    config.write_text('schema_version: 3\n')
    assert cli.main(['--config', str(config), '原始查询']) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[2])['query_status'] == 'failed'


def test_search_only_does_not_require_future_weight_paths(tmp_path):
    import yaml
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'schema_version': 3, 'query_model': {'model_path': 'missing-query'},
                                    'ranking': {'model_path': 'missing-rank'}}))
    events = []
    _, _, browser = fixtures(events)
    with SearchPipeline(path, load_model=False, browser_factory=browser) as pipeline:
        assert pipeline.runtime is None
    assert [n for n, _ in events] == ['start_browser', 'close_browser']


def test_incompatible_query_adapter_fails_at_startup():
    events = []
    planner, ranker, browser = fixtures(events)
    class InvalidPlanner(planner):
        plan = None
    with pytest.raises(ValueError, match='query_model_interface_invalid'):
        SearchPipeline(PipelineConfig(), build_query=InvalidPlanner, build_ranker=ranker, browser_factory=browser)
    assert [n for n, _ in events].count('close_rank') == 1
    assert [n for n, _ in events].count('close_query') == 1
