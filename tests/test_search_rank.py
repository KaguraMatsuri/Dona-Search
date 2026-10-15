import json
import subprocess
from pathlib import Path
import sys

import httpx
import pytest
from pydantic import ValidationError

from dona_search.config import SearchConfig, validate_config
from dona_search.pipeline import SearchPipeline
from dona_search.rerank.gemma import validate_order

V2_CONFIG = Path(__file__).parent / 'fixtures' / 'search_v2.yaml'


def page(urls):
    return '<ol>' + ''.join(
        f'<li class="res-list"><h3 class="res-title"><a href="{url}">T{i}</a></h3><p class="res-desc">S{i}</p></li>'
        for i, url in enumerate(urls)
    ) + '</ol>'


def item(url, title='M'):
    return {'url': url, 'title': [{'value': title}], 'extract': [{'value': '原文\n片段'}]}


@pytest.mark.parametrize('raw', [
    {'schema_version': 3}, {'unknown': True}, {'providers': []},
    {'providers': [{'type': 'search_unknown', 'name': 'x'}]},
    {'providers': [{'type': 'search_so360', 'name': 'x', 'top_k': True}]},
    {'providers': [{'type': 'search_so360', 'name': 'x', 'params': {'q': 'rewritten'}}]},
    {'providers': [{'type': 'search_so360', 'name': 'x', 'search_url': 'file:///tmp/data'}]},
    {'providers': [{'type': 'search_mwmbl', 'name': 'x'}] * 2},
    {'providers': [{'type': 'search_mwmbl', 'name': str(i)} for i in range(3)]},
])
def test_config_rejects_unsupported_or_ambiguous_values(raw):
    with pytest.raises(ValidationError):
        SearchConfig.model_validate(raw)


def test_relative_model_path(tmp_path):
    (tmp_path / 'weights').mkdir()
    cfg = validate_config({'ranking': {'model_path': 'weights'}}, tmp_path)
    assert cfg.ranking.model_path == tmp_path / 'weights'


def test_legacy_search_only_does_not_accidentally_load_models(tmp_path):
    path = tmp_path / 'old.yaml'
    path.write_text('query_processor: {type: query_mlx}\n')
    with pytest.raises(ValueError, match='search_only_requires_schema_v2'):
        SearchPipeline(path, load_model=False)


def test_import_does_not_load_model_frameworks():
    subprocess.run([sys.executable, '-c',
                    'import sys,dona_search; assert "torch" not in sys.modules; assert "mlx.core" not in sys.modules'], check=True)


def test_raw_query_filter_caps_dedup_and_provenance():
    query = ' 原话 + A&B、 '
    requests = []
    def transport(request):
        requests.append(request)
        if request.url.host == 'www.so.com':
            assert request.url.params['q'] == query
            return httpx.Response(200, text=page(['https://ai.so.com/x', 'https://same.test/'] + [f'https://so.test/{i}' for i in range(30)]))
        assert request.url.params['s'] == query
        return httpx.Response(200, json=[item('https://same.test/', 'duplicate')] + [item(f'https://m.test/{i}') for i in range(30)])
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with SearchPipeline(V2_CONFIG, http_client=client, load_model=False) as pipeline:
            report = pipeline.search_report(query)
            assert report.candidate_count == 39
            assert len(report.candidates) == 39
            assert len(report.results) == 10
            assert report.results[0].title == 'T1'
            assert [o['provider'] for o in report.results[0].origins] == ['360', 'mwmbl']
            assert [len(p.hits) for p in report.providers] == [20, 20]
            assert report.ranking_status == 'not_requested'
            assert [r.url.host for r in requests] == ['www.so.com', 'api.mwmbl.org']
        assert not client.is_closed
        with pytest.raises(RuntimeError, match='pipeline_closed'):
            pipeline.search_report(query)


@pytest.mark.parametrize('failed', ['www.so.com', 'api.mwmbl.org'])
def test_provider_failure_preserves_other_results(failed):
    def transport(request):
        if request.url.host == failed:
            return httpx.Response(302, headers={'location': 'https://qcaptcha.so.com/'})
        return httpx.Response(200, text=page(['https://so.test/'])) if request.url.host == 'www.so.com' else httpx.Response(200, json=[item('https://m.test/')])
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with SearchPipeline(V2_CONFIG, http_client=client, load_model=False) as pipeline:
            report = pipeline.search_report('query')
    assert len(report.results) == 1
    assert sorted(p.status for p in report.providers) == ['blocked', 'ok']


def test_empty_and_malformed_are_distinct():
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text='<html></html>') if r.url.host == 'www.so.com' else httpx.Response(200, json={'wrong': 'shape'}))) as client:
        with SearchPipeline(V2_CONFIG, http_client=client, load_model=False) as pipeline:
            report = pipeline.search_report('query')
    assert [p.status for p in report.providers] == ['empty', 'parse_error']
    assert report.ranking_status == 'no_results'


@pytest.mark.parametrize('mode', ['valid', 'invalid', 'exception', 'load_failure'])
def test_ranker_lifecycle_validation_and_fallback(monkeypatch, mode):
    events = []
    class Ranker:
        def __init__(self, config):
            events.append('load')
            if mode == 'load_failure':
                raise ValueError('fixture_load_failure')
        def warmup(self): events.append('warmup')
        def cleanup(self): events.append('cleanup')
        def close(self): events.append('close')
        def rank(self, query, hits):
            events.append('rank')
            if mode == 'exception': raise ValueError('fixture_failure')
            return [0, 0] if mode == 'invalid' else [1, 0]
    monkeypatch.setattr('dona_search.pipeline.GemmaReranker', Ranker)
    def transport(request):
        return httpx.Response(200, text=page(['https://so.test/'])) if request.url.host == 'www.so.com' else httpx.Response(200, json=[item('https://m.test/')])
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with SearchPipeline(V2_CONFIG, http_client=client) as pipeline:
            first = pipeline.search_report('query')
            second = pipeline.search_report('query')
            assert first.results == second.results or [r.url for r in first.results] == [r.url for r in second.results]
            assert len(first.results) == 2
            assert first.results[0].url == ('https://m.test/' if mode == 'valid' else 'https://so.test/')
            assert first.ranking_status == ('ok' if mode == 'valid' else 'unavailable' if mode == 'load_failure' else 'failed')
    assert events.count('load') == 1
    if mode != 'load_failure':
        assert events.count('warmup') == events.count('close') == 1
        assert events.count('rank') == 2


def test_provider_configuration_and_disabled_source(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('schema_version: 2\nproviders:\n  - type: search_so360\n    name: custom\n    search_url: https://configured.test/search\n    params: {pn: 2}\n    top_k: 1\n  - type: search_mwmbl\n    name: disabled\n    enabled: false\n')
    def transport(request):
        assert request.url.host == 'configured.test'
        assert request.url.params['pn'] == '2'
        return httpx.Response(200, text=page(['https://one.test/', 'https://two.test/']))
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with SearchPipeline(path, http_client=client, load_model=False) as pipeline:
            report = pipeline.search_report('query')
    assert report.candidate_count == 1
    assert len(report.providers) == 1


def test_permutation_rejects_invalid_ids():
    for order in ([0, 0], [0], [0, 2], [False, 1], '0,1'):
        with pytest.raises(ValueError): validate_order(order, 2)
    assert validate_order([1, 0], 2) == [1, 0]


def test_cli_reports_ranking_failure_without_generated_content(monkeypatch, capsys):
    from dona_search.__main__ import print_report
    from dona_search.interfaces import SearchReport, SearchResult
    print_report(SearchReport('q', [SearchResult(1, '原始片段', 'https://example.test/', None, '标题')], [], 1, 'failed', 'fixture'))
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])['ranking_status'] == 'failed'
    assert lines[1:] == ['[1] 标题', '    https://example.test/', '    原始片段', '']


def test_snapshot_preserves_full_pool_and_never_overwrites_judgments(tmp_path):
    from dona_search.__main__ import save_snapshot
    from dona_search.interfaces import SearchHit, SearchReport
    from scripts.eval_search import validate_fixture
    hits = [SearchHit(i + 1, str(i), f'https://x.test/{i}', '原始片段', 'fixture') for i in range(20)]
    report = SearchReport('q', [], [], 20, 'not_requested', candidates=hits)
    path = tmp_path / 'snapshot.jsonl'
    save_snapshot(report, path)
    row = json.loads(path.read_text())
    assert len(validate_fixture(row)) == 20
    assert row['judgments'] is None
    original = path.read_bytes()
    with pytest.raises(FileExistsError): save_snapshot(report, path)
    assert path.read_bytes() == original
