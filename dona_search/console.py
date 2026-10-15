"""The two-screen interactive terminal controller; SDK calls stay noninteractive."""
import json
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path

import questionary
import yaml
from prompt_toolkit.output.defaults import create_output
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text

from .config import (
    MODEL_PRESETS,
    ArtifactSource,
    ModelSelection,
    PipelineConfigV4,
    default_config,
    read_config,
    selection_config,
    user_config_path,
    validate_config,
)
from .models import ModelPreparer
from .output import all_engines_failed, format_results, terminal_text
from .pipeline import SearchPipeline

ROLES = (('query_model', '拆词模型'), ('ranking', '摘要模型'))
PHASE_LABELS = {
    'cached': '已缓存', 'download': '下载中', 'verify': '校验中', 'convert': '转换中',
    'prepared': '文件已准备', 'loading_models': '加载中', 'ready': '就绪',
}
ERROR_MESSAGES = {
    'memory_budget_exceeded': '所选模型预计超过内存预算，请选择更小的模型或更低精度的完整权重。',
    'query_context_limit': '单条结果或查询过长，当前上下文预算无法完成整理。',
    'organizing_context_limit': '单条结果或查询过长，当前上下文预算无法完成整理。',
    'organizing_item_context_limit': '单条结果或查询过长，当前上下文预算无法完成整理。',
    'organizing_merge_context_limit': '所选证据过长，当前上下文预算无法完成合并排序。',
}
logger = logging.getLogger(__name__)


def print_data(console, value):
    console.print(Text(terminal_text(json.dumps(value, ensure_ascii=False))))


def print_startup_error(error):
    console = Console(file=sys.stdout)
    code = str(error) or type(error).__name__
    if code == 'setup_cancelled':
        print_data(console, {'error_code': code})
        return
    console.print(Text(ERROR_MESSAGES.get(code, '模型准备失败，上次成功配置已保留。')))
    print_data(console, {'error': type(error).__name__, 'error_code': code})


def print_search_report(console, report):
    for provider in report.providers:
        if provider.status not in ('ok', 'empty') or provider.cleanup_error:
            code = ' / '.join(str(value) for value in (provider.status, provider.error_code, provider.cleanup_error) if value)
            console.print(Text(terminal_text(f'搜索请求未完成：{provider.provider}（{code}）')))
    if any(provider.cleanup_error and provider.hits for provider in report.providers):
        console.print(Text('浏览器页面清理失败，已保留搜索结果。'))
    if report.query_status == 'failed':
        console.print(Text('拆词失败，请检查拆词模型。'))
        if report.query_error in ERROR_MESSAGES:
            console.print(Text(ERROR_MESSAGES[report.query_error]))
        print_data(console, {'error_code': report.query_error})
    if report.organizing_status == 'failed' or report.ranking_status in ('failed', 'unavailable'):
        error = report.organizing_error or report.ranking_error
        console.print(Text('摘要整理失败，以下为未整理的搜索结果。'))
        if error in ERROR_MESSAGES:
            console.print(Text(ERROR_MESSAGES[error]))
        print_data(console, {'error_code': error})
    failed = all_engines_failed(report)
    if failed:
        console.print(Text('搜索未完成，请检查搜索引擎状态。'))
    output = format_results(report)
    if output:
        console.print(Text(output))
    elif (not failed
          and (report.organizing_status == 'no_results' or report.ranking_status == 'no_results')):
        console.print(Text('没有找到结果。'))


def load_selection(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return validate_config(default_config(), path.parent, require_model_paths=False)
    raw, base = read_config(path)
    config = validate_config(raw, base, require_model_paths=False)
    return selection_config(config)


def save_selection(config, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.config-', suffix='.yaml', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            yaml.safe_dump(config.model_dump(mode='json', exclude_none=True), stream,
                           allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def model_display_name(selected):
    return Path(selected.model.rstrip('/')).name


def selection_panel(config):
    panels = []
    for role, label in ROLES:
        selected = getattr(config, role)
        choices = '\n'.join(f'{i}. {name}' for i, (name, _) in enumerate(MODEL_PRESETS, 1)) + '\n5. 自定义'
        panels.append(Panel(Group(Text(choices), Text(terminal_text(model_display_name(selected)))),
                            title=Text(label), box=box.ROUNDED))
    return Panel(Group(Text('模型配置'), Columns(panels, equal=True, expand=True)),
                 title=Text('Dona Search:'), box=box.ROUNDED)


def choice(console, options, default):
    question = questionary.select(
        '', choices=[questionary.Choice(label, value=i, shortcut_key=str(i))
                     for i, label in enumerate(options, 1)],
        default=default, qmark='>', instruction=' ',
        use_arrow_keys=True, use_shortcuts=True, use_jk_keys=False, use_emacs_keys=False,
        output=create_output(stdout=console.file),
    )

    @question.application.key_bindings.add('c-d', eager=True)
    def end_input(event):
        event.app.exit(exception=EOFError)

    return question.unsafe_ask()


def artifact_input(console, existing=None):
    value = console.input('> ').strip()
    if not value:
        if existing is not None:
            return existing
        raise ValueError('artifact_source_required')
    path = Path(value).expanduser()
    if path.is_dir() or value.startswith(('/', '.', '~')):
        if not path.is_dir():
            raise ValueError('artifact_path_missing')
        return ArtifactSource(path=path.resolve())
    console.print(Text('模型版本（留空使用默认版本）'))
    revision = console.input('> ').strip() or None
    return ArtifactSource(repo=value, revision=revision)


def retry_input(console, reader, *args):
    while True:
        try:
            return reader(*args)
        except (ValueError, TypeError, OSError):
            console.print(Text('输入无效，请重新输入当前项。'))


def custom_model_input(console, previous):
    model = console.input('> ').strip() or previous.model
    path = Path(model).expanduser()
    if model.startswith(('/', '.', '~')) and not path.is_dir():
        raise ValueError('model_path_missing')
    return str(path.resolve()) if path.is_dir() else model


def fingerprint_input(console, current):
    value = console.input('> ').strip() or current.adapter_base_fingerprint
    return ModelSelection.model_validate(current.model_dump() | {'adapter_base_fingerprint': value}).adapter_base_fingerprint


def choose_models(config, console):
    selected = config.model_copy(deep=True)
    console.print(selection_panel(selected))
    for role, label in ROLES:
        previous = getattr(selected, role)
        default = next((i for i, (name, _) in enumerate(MODEL_PRESETS, 1) if name == previous.model), 5)
        console.print(Text(label))
        number = choice(console, [name for name, _ in MODEL_PRESETS] + ['自定义'], default)
        if number == 5:
            console.print(Text('自定义'))
            model = retry_input(console, custom_model_input, console, previous)
            revision = previous.revision if model == previous.model else None
            if model != previous.model and not Path(model).expanduser().is_dir():
                console.print(Text('模型版本（留空使用默认版本）'))
                revision = console.input('> ').strip() or None
        else:
            model, revision = MODEL_PRESETS[number - 1]
        current = previous.model_copy(deep=True) if model == previous.model else ModelSelection(
            model=model, revision=revision, max_prompt_tokens=previous.max_prompt_tokens)
        console.print(Text('模型权重'))
        weight_choice = choice(console, ['默认权重', '自定义'], 2 if current.weights is not None else 1)
        if weight_choice == 1:
            current.weights = None
        else:
            # A blank first input retains an existing custom source.
            print_data(console, {'weights': current.weights.model_dump(mode='json') if current.weights else None})
            current.weights = retry_input(console, artifact_input, console, current.weights)
        console.print(Text('微调权重（可选）'))
        adapter_choice = choice(console, ['不使用', '自定义'], 2 if current.adapter is not None else 1)
        if adapter_choice == 1:
            current.adapter, current.adapter_base_fingerprint = None, None
        else:
            print_data(console, {'adapter': current.adapter.model_dump(mode='json') if current.adapter else None})
            adapter = retry_input(console, artifact_input, console, current.adapter)
            if adapter != current.adapter:
                current.adapter_base_fingerprint = None
            current.adapter = adapter
            console.print(Text('基座模型校验值（适配器未声明绑定时必填）'))
            current.adapter_base_fingerprint = retry_input(console, fingerprint_input, console, current)
        setattr(selected, role, current)
    # Revalidate edited fields before performing downloads or filesystem writes.
    selected = PipelineConfigV4.model_validate(selected.model_dump())
    console.print(selection_panel(selected))
    console.print(Text('按 Enter 确认；输入任意文字取消本次启动。'))
    if console.input('> ').strip():
        raise ValueError('setup_cancelled')
    return selected


def run_console(config_path=None, *, browser_kind=None, show_browser=False,
                console=None, preparer_factory=ModelPreparer, pipeline_factory=SearchPipeline):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('apple_silicon_required')
    console = console or Console(file=sys.stdout)
    path = Path(config_path) if config_path is not None else user_config_path()
    initial = load_selection(path)
    if browser_kind is not None:
        initial.browser.kind = browser_kind
    if show_browser:
        initial.browser.headless = False
    selected = choose_models(initial, console)
    if any(provider.enabled and provider.type not in ('search_so360', 'search_mwmbl') for provider in selected.providers):
        from .search.browser import BrowserSession
        try:
            BrowserSession(selected.browser, selected.timeout)
        except ValueError as error:
            if str(error) == 'browser_binary_missing':
                links = {'ungoogled-chromium': 'https://github.com/ungoogled-software/ungoogled-chromium',
                         'firefox': 'https://www.mozilla.org/firefox/new/',
                         'chrome': 'https://www.google.com/chrome/'}
                print_data(console, {'error_code': 'browser_binary_missing',
                    'browser': selected.browser.kind, 'download_url': links[selected.browser.kind]})
            raise
    progress = Progress(TextColumn('{task.description}', markup=False), BarColumn(),
                        TextColumn('{task.fields[count]}', markup=False), console=console, refresh_per_second=2)
    tasks = {role: progress.add_task(label, total=None, count='') for role, label in ROLES}

    def update(role, phase, completed, total):
        progress.reset(tasks[role], description=f'{dict(ROLES)[role]} [{PHASE_LABELS.get(phase, phase)}]',
                       completed=completed, total=total, count=f'{completed:.0f}/{total}' if total is not None else '')

    with progress:
        resolved, pinned = preparer_factory(download=True, progress=update).prepare(selected)
        for role, _ in ROLES:
            update(role, 'loading_models', 0, None)
        pipeline = pipeline_factory(resolved)
        for role, _ in ROLES:
            update(role, 'ready', 1, 1)
    with pipeline:
        save_selection(pinned, path)
        console.print(Text('准备完成，请输入搜索内容。'))
        while True:
            query = console.input('> ')
            if not query.strip():
                continue
            try:
                with console.status('正在搜索…'):
                    report = pipeline.search_report(query)
            except Exception as error:
                logger.debug('request_failed', exc_info=True)
                print_data(console, {'error': type(error).__name__, 'error_code': str(error)})
                continue
            print_search_report(console, report)
