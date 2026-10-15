import io
import json

import pytest
from rich.console import Console

from dona_search import __main__ as cli
from dona_search.console import print_search_report
from dona_search.interfaces import (
    ProviderResult,
    SearchHit,
    SearchReport,
    SearchResult,
    SnippetEvidence,
)


@pytest.mark.parametrize('mode', ['empty', 'blocked', 'partial', 'cleanup', 'query_failed', 'organizing_failed'])
def test_interactive_feedback_preserves_failure_meaning_and_original_results(mode):
    hit = SearchHit(1, '原始标题', 'https://source.test/', 'SQLite 原文。', 'working')
    result = SearchResult(1, hit.snippet, hit.url, None, hit.title,
                          snippet_evidence=SnippetEvidence(hit.snippet, 0, len(hit.snippet)))
    report = SearchReport('query', [result], [ProviderResult('working', 'ok', [hit])], 1, 'ok',
                          query_status='ok', organizing_status='ok')
    if mode == 'empty':
        report.results, report.providers = [], [ProviderResult('working', 'empty')]
        report.ranking_status = report.organizing_status = 'no_results'
    elif mode == 'blocked':
        report.results = []
        report.providers = [ProviderResult('blocked', 'blocked', error_code='captcha_required'),
                            ProviderResult('slow', 'timeout', error_code='request_timeout')]
        report.ranking_status = report.organizing_status = 'no_results'
    elif mode == 'partial':
        report.providers.append(ProviderResult('blocked\x1b[2J', 'blocked', error_code='captcha_required'))
    elif mode == 'cleanup':
        report.providers[0].cleanup_error = 'WebDriverException'
    elif mode == 'query_failed':
        report.results, report.providers = [], []
        report.query_status, report.query_error = 'failed', 'query_context_limit'
        report.ranking_status = report.organizing_status = 'not_requested'
    else:
        report.ranking_status = report.organizing_status = 'failed'
        report.organizing_error = 'organizing_merge_context_limit'
        result.snippet_evidence = None
    stream = io.StringIO()
    print_search_report(Console(file=stream, width=200), report)
    output = stream.getvalue()
    assert '\x1b' not in output
    assert ('没有找到结果。' in output) == (mode == 'empty')
    assert ('搜索未完成，请检查搜索引擎状态。' in output) == (mode == 'blocked')
    if mode in ('blocked', 'partial'):
        assert 'captcha_required' in output and '搜索请求未完成：' in output
    if mode == 'cleanup':
        assert '浏览器页面清理失败，已保留搜索结果。' in output
        assert 'WebDriverException' in output
    if mode == 'query_failed':
        assert '拆词失败，请检查拆词模型。' in output and 'query_context_limit' in output
    if mode == 'organizing_failed':
        assert '摘要整理失败，以下为未整理的搜索结果。' in output
        assert '所选证据过长，当前上下文预算无法完成合并排序。' in output
    if report.results:
        assert '[ 1 ] 原始标题\nhttps://source.test/\n摘要: ' in output
        assert output.rstrip().endswith('摘要:' if mode == 'organizing_failed' else '摘要: SQLite 原文。')


@pytest.mark.parametrize('error,expected', [
    ('memory_budget_exceeded', '所选模型预计超过内存预算，请选择更小的模型或更低精度的完整权重。'),
    ('organizing_item_context_limit', '单条结果或查询过长，当前上下文预算无法完成整理。'),
    ('organizing_merge_context_limit', '所选证据过长，当前上下文预算无法完成合并排序。'),
    ('startup_failure', '模型准备失败，上次成功配置已保留。'),
])
def test_interactive_startup_messages_do_not_change_batch_json(monkeypatch, capsys, error, expected):
    def fail(*args, **kwargs): raise ValueError(error)
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr('dona_search.console.run_console', fail)
    assert cli.main([]) == 1
    output = capsys.readouterr().out
    assert ''.join(expected.split()) in ''.join(output.split())
    assert error in output
    monkeypatch.setattr(cli, 'SearchPipeline', fail)
    assert cli.main(['query']) == 1
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])['status'] == 'loading_models'
    assert json.loads(lines[1])['error_code'] == error
    assert expected not in '\n'.join(lines)


def test_explicit_setup_cancel_is_not_described_as_model_preparation_failure(monkeypatch, capsys):
    def cancel(*args, **kwargs): raise ValueError('setup_cancelled')
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr('dona_search.console.run_console', cancel)
    assert cli.main([]) == 1
    output = capsys.readouterr().out
    assert json.loads(output)['error_code'] == 'setup_cancelled'
    assert '模型准备失败' not in output
