"""Escape terminal instructions at presentation boundaries, preserving source data."""

_CONTROL_ESCAPES = {
    code: f'\\u{code:04x}'
    for code in (*range(32), *range(127, 160))
    if code not in (9, 10)
}


def terminal_text(value: str) -> str:
    return value.translate(_CONTROL_ESCAPES)


def format_result(result) -> str:
    excerpt = result.snippet_evidence.text if result.snippet_evidence is not None else ''
    return terminal_text(f'[ {result.rank} ] {result.title}\n{result.url}\n摘要: {excerpt}')


def format_results(report) -> str:
    return '\n\n'.join(format_result(result) for result in report.results)


def all_engines_failed(report) -> bool:
    outcomes = [task.outcome for task in report.tasks] if report.tasks else report.providers
    return bool(outcomes) and not any(outcome.status in ('ok', 'empty')
                                     or (outcome.status == 'partial' and outcome.hits) for outcome in outcomes)
