from threading import get_ident

import pytest

from dona_search.config import PipelineConfigV4
from dona_search.interfaces import (
    OrganizationDecision,
    ProviderResult,
    QueryPlan,
    SearchHit,
)
from dona_search.pipeline import SearchPipeline


def components(events, mode='ok'):
    class Planner:
        def __init__(self): events.append(('load_query', get_ident()))
        def warmup(self): events.append(('warm_query', get_ident()))
        def cleanup(self): events.append(('cleanup_query', get_ident()))
        def close(self): events.append(('close_query', get_ident()))
        def plan(self, raw):
            events.append(('plan', get_ident()))
            if mode == 'query_failure': raise ValueError('query_failure')
            return QueryPlan(raw, [raw, raw + ' documentation'])
    class Organizer:
        def __init__(self): events.append(('load_organizer', get_ident()))
        def warmup(self):
            events.append(('warm_organizer', get_ident()))
            if mode == 'warm_failure': raise ValueError('warm_failure')
        def cleanup(self): events.append(('cleanup_organizer', get_ident()))
        def close(self): events.append(('close_organizer', get_ident()))
        def organize(self, query, hits):
            events.append(('organize', get_ident()))
            if mode == 'invalid_selection': return OrganizationDecision(list(range(len(hits))), [999] * len(hits))
            return OrganizationDecision(list(reversed(range(len(hits)))), [2] + [None] * (len(hits) - 1))
    class Browser:
        def __init__(self, config, timeout): pass
        def start(self): events.append(('start_browser', get_ident()))
        def close(self): events.append(('close_browser', get_ident()))
        def search(self, provider, query, url, k):
            events.append(('search', get_ident()))
            if mode == 'empty': return ProviderResult(provider.name, 'empty')
            if mode == 'blocked' and provider.name == 'google': return ProviderResult(provider.name, 'blocked')
            return ProviderResult(provider.name, 'ok', [
                SearchHit(1, '原始标题', 'https://shared.test/', '第一句。SQLite 原文。', provider.name),
                SearchHit(2, provider.name, f'https://{provider.name}.test/', '另一个网站原文。', provider.name)])
    return Planner, Organizer, Browser


def make_pipeline(events, mode='ok', **kwargs):
    planner, organizer, browser = components(events, mode)
    return SearchPipeline(PipelineConfigV4(), build_query=planner, build_organizer=organizer,
                          browser_factory=browser, **kwargs)


def test_v4_complete_flow_is_resident_and_summary_stays_with_original_url():
    events = []
    with make_pipeline(events) as pipeline:
        original = (id(pipeline.runtime.query_processor), id(pipeline.runtime.organizer))
        for _ in range(2):
            report = pipeline.search_report('SQLite')
            assert report.query_status == report.organizing_status == 'ok'
            assert len(report.tasks) == 6 and report.candidate_count == 4
            assert [task.task.provider for task in report.tasks] == ['google', 'bing', 'yandex'] * 2
            selected = next(item for item in report.results if item.url == 'https://shared.test/')
            assert selected.title == '原始标题' and selected.snippet_evidence.text == 'SQLite 原文。'
            assert len(selected.origins) == 6
            assert all(item.summary is None for item in report.results)
            assert all(item.snippet_evidence is None for item in report.results if item != selected)
            assert original == (id(pipeline.runtime.query_processor), id(pipeline.runtime.organizer))
        assert pipeline.runtime.core is pipeline.runtime.ranker is None
        assert not any(name.startswith('close') for name, _ in events)
    names = [name for name, _ in events]
    assert names.count('load_query') == names.count('load_organizer') == 1
    assert names.index('load_organizer') < names.index('warm_query') < names.index('warm_organizer')
    assert names.count('close_query') == names.count('close_organizer') == names.count('close_browser') == 1
    gpu_threads = {thread for name, thread in events if 'browser' not in name and name != 'search'}
    assert len(gpu_threads) == 1


def test_invalid_selection_falls_back_atomically_without_fake_summary():
    with make_pipeline([], 'invalid_selection') as pipeline:
        report = pipeline.search_report('SQLite')
        assert report.organizing_status == 'failed'
        assert [result.url for result in report.results] == [hit.url for hit in report.candidates]
        assert all(result.snippet_evidence is None for result in report.results)


def test_query_failure_does_not_search_or_unload_models():
    events = []
    with make_pipeline(events, 'query_failure') as pipeline:
        assert pipeline.search_report('SQLite').query_status == 'failed'
        assert 'search' not in [name for name, _ in events]
        assert 'organize' not in [name for name, _ in events]
        assert pipeline.runtime.organizer is not None


def test_warmup_failure_releases_both_models_and_browser():
    events = []
    with pytest.raises(ValueError, match='warm_failure'):
        make_pipeline(events, 'warm_failure')
    names = [name for name, _ in events]
    assert names.count('close_query') == names.count('close_organizer') == names.count('close_browser') == 1


@pytest.mark.parametrize('mode', ['empty', 'blocked'])
def test_empty_results_and_partial_engine_failure(mode):
    events = []
    with make_pipeline(events, mode) as pipeline:
        report = pipeline.search_report('SQLite')
        if mode == 'empty':
            assert report.organizing_status == 'no_results'
            assert 'organize' not in [name for name, _ in events]
        else:
            assert report.organizing_status == 'ok'
            assert [t.outcome.status for t in report.tasks if t.task.provider == 'google'] == ['blocked', 'skipped']


def test_v4_search_only_never_constructs_any_model():
    events = []
    with make_pipeline(events, load_model=False) as pipeline:
        report = pipeline.search_report('SQLite')
        assert pipeline.runtime is None and report.search_queries == ['SQLite']
        assert report.organizing_status == 'not_requested'
    assert not any(name.startswith(('load', 'warm')) for name, _ in events)


def test_v4_batch_cli_reports_organizing_and_prints_source_summary(tmp_path, monkeypatch, capsys):
    from dona_search import __main__ as cli
    from dona_search.console import save_selection
    path = tmp_path / 'config.yaml'
    save_selection(PipelineConfigV4(), path)
    planner, organizer, browser = components([])
    monkeypatch.setattr(cli, 'SearchPipeline', lambda config, **kwargs: SearchPipeline(
        config, build_query=planner, build_organizer=organizer, browser_factory=browser, **kwargs))
    assert cli.main(['--config', str(path), 'SQLite']) == 0
    text = capsys.readouterr().out
    assert '"organizing_status": "ok"' in text
    assert '[ 1 ] ' in text and '\nhttps://shared.test/\n摘要: SQLite 原文。' in text


def test_sdk_default_uses_v4_organizer_and_matches_first_console_selection(tmp_path):
    from dona_search.console import load_selection
    planner, organizer, browser = components([])
    with SearchPipeline(build_query=planner, build_organizer=organizer, browser_factory=browser) as pipeline:
        assert pipeline.cfg == load_selection(tmp_path / 'new.yaml')
        report = pipeline.search_report('SQLite')
        assert pipeline.cfg.schema_version == 4 and report.organizing_status == 'ok'
        assert any(result.snippet_evidence is not None for result in report.results)


def test_batch_cli_default_uses_v4_organizer(monkeypatch, capsys):
    from dona_search import __main__ as cli
    planner, organizer, browser = components([])
    monkeypatch.setattr(cli, 'SearchPipeline', lambda config, **kwargs: SearchPipeline(
        config, build_query=planner, build_organizer=organizer, browser_factory=browser, **kwargs))
    assert cli.main(['SQLite']) == 0
    output = capsys.readouterr().out
    assert '"organizing_status": "ok"' in output
    assert '摘要: SQLite 原文。' in output
