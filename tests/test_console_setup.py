import io
from contextlib import contextmanager

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from rich.console import Console

from dona_search import console as console_module
from dona_search.config import ArtifactSource, ModelSelection, PipelineConfigV4
from dona_search.console import (
    choice,
    choose_models,
    format_results,
    load_selection,
    run_console,
    save_selection,
    selection_panel,
)
from dona_search.interfaces import SearchReport, SearchResult, SnippetEvidence


@pytest.mark.parametrize('width', [30, 80, 120])
def test_both_roles_are_rendered_without_interpreting_custom_markup(width):
    stream = io.StringIO()
    console = Console(file=stream, width=width, force_terminal=False)
    config = PipelineConfigV4(query_model=ModelSelection(model='custom/[bold]model'))
    console.print(selection_panel(config))
    text = stream.getvalue()
    assert '拆词模型' in text and '摘要模型' in text
    assert '自定义' in text and '模型配置' in text
    assert '[bold]' in text


def test_default_selection_fits_an_80_by_24_terminal():
    stream = io.StringIO()
    Console(file=stream, width=80, height=24).print(selection_panel(PipelineConfigV4()))
    assert len(stream.getvalue().splitlines()) <= 24
    assert '拆词模型' in stream.getvalue() and '摘要模型' in stream.getvalue()


@pytest.mark.parametrize('model,name', [
    ('google/gemma-4-E4B-it', 'gemma-4-E4B-it'),
    ('/Users/example/models/My Imported Model', 'My Imported Model'),
])
def test_selection_shows_model_name_without_exposing_config_or_weights(model, name):
    stream = io.StringIO()
    config = PipelineConfigV4(query_model=ModelSelection(
        model=model, weights=ArtifactSource(path='/private/weights/Another Model')))
    Console(file=stream, width=120).print(selection_panel(config))
    text = stream.getvalue()
    assert name in text
    assert '/Users/example/' not in text
    assert '/private/weights/' not in text and 'Another Model' not in text
    assert '"model"' not in text and '"weights"' not in text and '"path"' not in text
    assert '{' not in text and '}' not in text


def input_console(monkeypatch, values):
    console = Console(file=io.StringIO(), width=80)
    values = iter(values)

    def read(*args):
        try:
            return next(values)
        except StopIteration:
            raise EOFError from None

    def select(console, options, default):
        return int(read() or default)

    monkeypatch.setattr(console, 'input', read)
    monkeypatch.setattr(console_module, 'choice', select)
    return console


@pytest.mark.parametrize('keys,expected', [
    ('\r', 2),
    ('\x1b[A\r', 1),
    ('\x1b[B\r', 3),
    ('1\r', 1),
])
def test_menu_handles_arrow_keys_enter_default_and_number_shortcuts(keys, expected):
    console = Console(file=io.StringIO(), width=80)
    with create_pipe_input() as pipe, create_app_session(input=pipe):
        pipe.send_text(keys)
        assert choice(console, ['Qwen/Qwen3-4B', 'Qwen/Qwen3-8B', 'custom'], 2) == expected
    assert 'Use shortcuts' not in console.file.getvalue()
    assert 'Use arrow keys' not in console.file.getvalue()


@contextmanager
def keyboard_menus(monkeypatch, keys):
    unsafe_ask = console_module.questionary.Question.unsafe_ask
    keys = iter(keys)

    with create_pipe_input() as pipe, create_app_session(input=pipe):
        def ask(question):
            pipe.send_text(next(keys))
            return unsafe_ask(question)

        monkeypatch.setattr(console_module.questionary.Question, 'unsafe_ask', ask)
        yield


def test_keyboard_selection_reaches_custom_weights_and_adapter(monkeypatch, tmp_path):
    weights = tmp_path / 'weights'
    adapter = tmp_path / 'adapter'
    weights.mkdir()
    adapter.mkdir()
    fingerprint = 'a' * 64
    console = Console(file=io.StringIO(), width=80)
    inputs = iter([str(weights), str(adapter), fingerprint, ''])
    monkeypatch.setattr(console, 'input', lambda *args: next(inputs))
    original = PipelineConfigV4()
    with keyboard_menus(monkeypatch, ['\x1b[A\r', '\x1b[B\r', '\x1b[B\r', '\r', '\r', '\r']):
        selected = choose_models(original, console)
    assert selected.query_model.model == 'google/gemma-4-E2B-it'
    assert selected.query_model.weights.path == weights
    assert selected.query_model.adapter.path == adapter
    assert selected.query_model.adapter_base_fingerprint == fingerprint
    assert selected.ranking == original.ranking
    assert original.query_model.weights is None and original.query_model.adapter is None


def test_keyboard_enter_keeps_custom_model_weights_and_adapter(monkeypatch, tmp_path):
    model = tmp_path / 'My Imported Model'
    model.mkdir()
    config = PipelineConfigV4(query_model=ModelSelection(
        model=str(model), weights=ArtifactSource(path=tmp_path / 'weights'),
        adapter=ArtifactSource(path=tmp_path / 'adapter'), adapter_base_fingerprint='b' * 64))
    console = Console(file=io.StringIO(), width=80)
    monkeypatch.setattr(console, 'input', lambda *args: '')
    with keyboard_menus(monkeypatch, ['\r'] * 6):
        assert choose_models(config, console) == config


@pytest.mark.parametrize('keys,error', [
    ('\x03', KeyboardInterrupt),
    ('\x04\x03', EOFError),  # Ctrl+C prevents a hang if Ctrl+D stops being handled.
])
def test_keyboard_exit_does_not_download_or_save(monkeypatch, tmp_path, keys, error):
    path = tmp_path / 'config.yaml'
    save_selection(PipelineConfigV4(), path)
    original = path.read_bytes()
    with keyboard_menus(monkeypatch, [keys]), pytest.raises(error):
        run_console(path, console=Console(file=io.StringIO(), width=80),
                    preparer_factory=lambda **kwargs: pytest.fail('download_started'))
    assert path.read_bytes() == original


def test_model_change_does_not_carry_previous_revision_weights_or_adapter(monkeypatch):
    config = PipelineConfigV4(query_model=ModelSelection(weights=ArtifactSource(path='/old'), adapter=ArtifactSource(path='/adapter')))
    selected = choose_models(config, input_console(monkeypatch, ['1', '', '', '', '', '', '']))
    assert selected.query_model.model == 'Qwen/Qwen3-4B'
    assert selected.query_model.revision != config.query_model.revision
    assert selected.query_model.weights is selected.query_model.adapter is None
    assert selected.ranking.model == config.ranking.model


def test_saved_models_prefill_without_losing_engine_settings(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    config = PipelineConfigV4()
    config.providers[0].enabled = False
    config.browser.kind = 'firefox'
    save_selection(config, path)
    selected = choose_models(load_selection(path), input_console(monkeypatch, [''] * 7))
    assert not selected.providers[0].enabled and selected.browser.kind == 'firefox'
    assert selected.query_model.model == config.query_model.model
    assert not list(tmp_path.glob('.config-*'))


def test_final_confirmation_keeps_model_names_without_dumping_internal_config(monkeypatch):
    config = PipelineConfigV4(browser={'profile_dir': '/private/browser-profile'})
    console = input_console(monkeypatch, [''] * 7)
    selected = choose_models(config, console)
    text = console.file.getvalue()
    assert selected.browser.profile_dir == config.browser.profile_dir
    assert '拆词模型' in text and '摘要模型' in text and 'gemma-4-E4B-it' in text
    assert 'schema_version' not in text and 'max_prompt_tokens' not in text
    assert '/private/browser-profile' not in text and '"config"' not in text


def test_cancel_does_not_download_or_replace_existing_config(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    save_selection(PipelineConfigV4(), path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='setup_cancelled'):
        run_console(path, console=input_console(monkeypatch, [''] * 6 + ['cancel']),
                    preparer_factory=lambda **kwargs: pytest.fail('download_started'))
    assert path.read_bytes() == original


def test_startup_failure_keeps_previous_config(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    save_selection(PipelineConfigV4(), path)
    original = path.read_bytes()
    class Preparer:
        def __init__(self, **kwargs): pass
        def prepare(self, selected): return selected, selected
    def broken(config): raise ValueError('startup_failure')
    monkeypatch.setattr('dona_search.search.browser.BrowserSession', lambda *args: None)
    with pytest.raises(ValueError, match='startup_failure'):
        run_console(path, console=input_console(monkeypatch, [''] * 7), preparer_factory=Preparer, pipeline_factory=broken)
    assert path.read_bytes() == original


def test_requested_three_line_format_and_empty_summary():
    report = SearchReport('query', [
        SearchResult(1, '完整片段', 'https://a.test/', None, '原始标题', snippet_evidence=SnippetEvidence('原文。', 0, 3)),
        SearchResult(2, '原文不可用', 'http://b.test/', None, '第二标题'),
    ], [], 2, 'ok')
    assert format_results(report) == '[ 1 ] 原始标题\nhttps://a.test/\n摘要: 原文。\n\n[ 2 ] 第二标题\nhttp://b.test/\n摘要: '


def test_missing_browser_fails_before_any_large_download(tmp_path, monkeypatch):
    def missing(*args): raise ValueError('browser_binary_missing')
    monkeypatch.setattr('dona_search.search.browser.BrowserSession', missing)
    console = input_console(monkeypatch, [''] * 7)
    with pytest.raises(ValueError, match='browser_binary_missing'):
        run_console(tmp_path / 'config.yaml', console=console,
                    preparer_factory=lambda **kwargs: pytest.fail('download_started'))
    assert not (tmp_path / 'config.yaml').exists()
    assert 'download_url' in console.file.getvalue()


def test_console_ready_search_and_exit_save_config_and_close_resources(tmp_path, monkeypatch):
    events = []
    class Preparer:
        def __init__(self, progress, **kwargs): self.progress = progress
        def prepare(self, selected):
            for role in ['query_model', 'ranking']:
                self.progress(role, 'download', 1, 2)
                self.progress(role, 'verify', 0, None)
                self.progress(role, 'prepared', 1, 1)
            return selected, selected
    class Pipeline:
        def __init__(self, config): events.append('load')
        def __enter__(self): return self
        def __exit__(self, *args): events.append('close')
        def search_report(self, query):
            events.append(query)
            if query == 'failure': raise LookupError('request_failure')
            return SearchReport(query, [SearchResult(1, '原文', 'https://a.test/', None, '标题',
                snippet_evidence=SnippetEvidence('原文', 0, 2))], [], 1, 'ok', organizing_status='ok')
    monkeypatch.setattr('dona_search.search.browser.BrowserSession', lambda *args: None)
    console = input_console(monkeypatch, [''] * 7 + ['failure', 'SQLite'])
    path = tmp_path / 'config.yaml'
    with pytest.raises(EOFError):
        run_console(path, console=console, preparer_factory=Preparer, pipeline_factory=Pipeline)
    assert load_selection(path).schema_version == 4
    assert events == ['load', 'failure', 'SQLite', 'close']
    assert '摘要: 原文' in console.file.getvalue()
    output = console.file.getvalue()
    assert 'downloading:' not in output
    assert '拆词模型 [就绪]' in output and '摘要模型 [就绪]' in output
    assert '准备完成，请输入搜索内容。' in output


def test_cli_noninteractive_does_not_start_setup_or_download(monkeypatch, capsys):
    from dona_search import __main__ as cli
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(cli, 'SearchPipeline', lambda *args, **kwargs: pytest.fail('models_loaded'))
    assert cli.main([]) == 1
    assert 'config_required' in capsys.readouterr().out
