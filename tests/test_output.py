import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from dona_search import __main__ as cli
from dona_search import format_result as sdk_format_result
from dona_search import format_results as sdk_format_results
from dona_search.config import BrowserProviderConfig, ExtractionRules
from dona_search.console import format_results
from dona_search.interfaces import (
    ProviderResult,
    SearchHit,
    SearchReport,
    SearchResult,
    SnippetEvidence,
)
from dona_search.output import terminal_text
from dona_search.search.browser import BrowserSearchProvider
from dona_search.search.scheduler import SearchScheduler


def test_terminal_controls_are_escaped_without_changing_printable_text():
    safe = '中文 😀 café [bold] \\literal\n\t'
    assert terminal_text(safe) == safe
    for code in (*range(32), *range(127, 160)):
        if code in (9, 10):
            continue
        result = terminal_text(safe + chr(code) + safe)
        assert chr(code) not in result
        assert result.startswith(safe) and result.endswith(safe)
        assert terminal_text(result) == result


def test_remote_snippet_controls_never_reach_terminal_and_source_is_preserved():
    provider = BrowserSearchProvider(BrowserProviderConfig(
        type='search_xpath', name='test', search_url='https://search.test/',
        rules=ExtractionRules(item_xpath='//li', title_xpath='.//h2', url_xpath='.//a/@href', snippet_xpath='.//p')), None)
    outcome = provider.inspect('<ul><li><h2>title</h2><a href="https://example.test/">link</a>'
        '<p>&#27;[2J&#27;[HFORGED</p></li></ul>', 'https://search.test/', 'query', 1)
    hit = outcome.hits[0]
    report = SearchReport('query', [SearchResult(1, hit.snippet, hit.url, None, hit.title,
        snippet_evidence=SnippetEvidence(hit.snippet, 0, len(hit.snippet)))], [outcome], 1, 'ok')
    stream = io.StringIO()
    Console(file=stream, force_terminal=True).print(Text(format_results(report)))
    assert '\x1b' not in stream.getvalue()
    assert '\\u001b[2J' in stream.getvalue()
    assert report.results[0].snippet_evidence.text == '\x1b[2J\x1b[HFORGED'
    assert hit.snippet == report.results[0].snippet_evidence.text


@pytest.mark.parametrize('evidence', [None, SnippetEvidence('SQLite 原文。', 4, 14)])
def test_sdk_and_cli_use_the_same_v4_formatter_without_breaking_legacy_format(evidence):
    result = SearchResult(1, '第一句。SQLite 原文。', 'https://sqlite.org/', None,
                          'SQLite 文档', snippet_evidence=evidence)
    report = SearchReport('query', [result], [], 1, 'ok')
    expected = '[ 1 ] SQLite 文档\nhttps://sqlite.org/\n摘要: ' + (evidence.text if evidence else '')
    assert sdk_format_result(result) == sdk_format_results(report) == format_results(report) == expected
    assert result.format(organized=True) == expected
    assert result.format() == '[1] 第一句。SQLite 原文。\n    https://sqlite.org/\n'


@pytest.mark.parametrize('organized', [False, True])
def test_batch_json_remains_valid_and_all_result_fields_are_terminal_safe(capsys, organized):
    value = '中文\x9b2J\x1b]8;;https://other.test/\x1b\\'
    report = SearchReport(value, [SearchResult(1, value, 'https://example.test/', None, value,
        snippet_evidence=SnippetEvidence(value, 0, len(value)))], [], 1, 'ok')
    cli.print_report(report, organized=organized)
    output = capsys.readouterr().out
    assert '\x1b' not in output and '\x9b' not in output
    assert json.loads(output.splitlines()[0])['query'] == value


def test_interactive_status_data_is_safe_and_still_roundtrips_as_json():
    from dona_search.console import print_data
    stream = io.StringIO()
    value = {'error_code': 'source\x9b2J\x1b[H'}
    print_data(Console(file=stream, width=200), value)
    assert '\x9b' not in stream.getvalue() and '\x1b' not in stream.getvalue()
    assert json.loads(stream.getvalue()) == value


@pytest.mark.parametrize('query_status,ranking_status,organizing_status,provider_statuses,expected', [
    ('failed', 'not_requested', 'not_requested', [], 1),
    ('ok', 'failed', 'failed', ['ok'], 1),
    ('not_requested', 'unavailable', 'not_requested', ['ok'], 1),
    ('ok', 'no_results', 'no_results', ['blocked', 'timeout'], 1),
    ('ok', 'no_results', 'no_results', ['empty', 'timeout'], 0),
    ('ok', 'ok', 'ok', ['ok', 'blocked'], 0),
    ('ok', 'no_results', 'no_results', ['empty', 'empty'], 0),
])
def test_batch_exit_code_covers_every_pipeline_stage(monkeypatch, capsys, query_status,
                                                    ranking_status, organizing_status, provider_statuses, expected):
    report = SearchReport('query', [], [ProviderResult(str(i), status) for i, status in enumerate(provider_statuses)],
                          0, ranking_status, query_status=query_status, organizing_status=organizing_status)

    class Pipeline:
        _organizing = True
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def search_report(self, query): return report

    monkeypatch.setattr(cli, 'SearchPipeline', lambda *args, **kwargs: Pipeline())
    assert cli.main(['query']) == expected
    output = capsys.readouterr().out
    assert json.loads(output.splitlines()[2])['organizing_status'] == organizing_status


@pytest.mark.parametrize('bad_result', [[], ProviderResult('wrong', 'empty'),
    ProviderResult('bad', 'ok', [None]), ProviderResult('bad', 'ok', [SearchHit(1, 't', [], 's', 'bad')])])
def test_invalid_provider_result_cannot_break_other_engines(bad_result):
    bad = SimpleNamespace(name='bad', config=SimpleNamespace(top_k=20), search=lambda *args: bad_result)
    good = SimpleNamespace(name='good', config=SimpleNamespace(top_k=20),
                           search=lambda *args: ProviderResult('good', 'empty'))
    scheduler = SearchScheduler([bad, good])
    try:
        tasks = scheduler.collect(['query'])
        assert [task.outcome.status for task in tasks] == ['parse_error', 'empty']
        assert tasks[0].outcome.error_code == 'TypeError'
    finally:
        scheduler.close()
