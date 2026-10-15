import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .output import all_engines_failed, format_results, terminal_text
from .pipeline import SearchPipeline

logger = logging.getLogger(__name__)


def print_terminal(value):
    print(terminal_text(value), flush=True)


def report_exit_code(report):
    if (report.query_status == 'failed' or report.ranking_status in ('failed', 'unavailable')
            or report.organizing_status == 'failed'):
        return 1
    if all_engines_failed(report):
        return 1
    return 0


def save_snapshot(report, path):
    if report.ranking_status == 'research':
        raise ValueError('snapshot_requires_schema_v2')
    row = {'query': report.query, 'candidates': [asdict(h) for h in report.candidates],
           'judgments': None, 'judgment_source': None,
           'search_queries': report.search_queries, 'query_status': report.query_status,
           'tasks': [asdict(t) for t in report.tasks]}
    row['id'] = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def print_report(report, *, organized=False):
    status = {
        'query': report.query,
        'providers': {p.provider: {'status': p.status, 'count': len(p.hits), 'http_status': p.http_status,
                                   'error_code': p.error_code, 'cleanup_error': p.cleanup_error} for p in report.providers},
        'candidate_count': report.candidate_count,
        'ranking_status': report.ranking_status,
        'ranking_error': report.ranking_error,
        'query_status': report.query_status,
        'query_error': report.query_error,
        'search_queries': report.search_queries,
        'tasks': [{'id': t.task.id, 'provider': t.task.provider, 'query': t.task.query,
                   'status': t.outcome.status, 'count': len(t.outcome.hits),
                   'error_code': t.outcome.error_code, 'cleanup_error': t.outcome.cleanup_error,
                   'seconds': t.outcome.elapsed_seconds} for t in report.tasks],
        'timings': report.timings,
    }
    if organized:
        status.update(organizing_status=report.organizing_status, organizing_error=report.organizing_error)
    print_terminal(json.dumps(status, ensure_ascii=False))
    if organized:
        output = format_results(report)
        if output:
            print_terminal(output)
        return
    for result in report.results:
        if report.ranking_status == 'research':
            print_terminal(result.format())
            continue
        print_terminal(f'[{result.rank}] {result.title}\n    {result.url}\n    {result.preview}\n')
        if result.summary is not None:
            print_terminal(result.summary.text)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('query', nargs='*')
    parser.add_argument('--config')
    parser.add_argument('--search-only', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--browser', choices=['ungoogled-chromium', 'firefox', 'chrome'])
    parser.add_argument('--show-browser', action='store_true')
    args = parser.parse_args(argv)
    if args.snapshot and (not args.query or args.snapshot.exists()):
        parser.error('snapshot_requires_query_and_new_path')
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive and not args.query and not args.search_only:
        from .config import read_config
        try:
            raw = read_config(args.config)[0] if args.config and Path(args.config).expanduser().is_file() else {'schema_version': 4}
            if raw.get('schema_version') in (3, 4):
                from .console import run_console
                run_console(args.config, browser_kind=args.browser, show_browser=args.show_browser,
                            pipeline_factory=SearchPipeline)
                return 0
        except (EOFError, KeyboardInterrupt):
            return 0
        except (ValueError, TypeError, RuntimeError, OSError, ImportError) as error:
            from .console import print_startup_error
            print_startup_error(error)
            return 1
        except Exception as error:
            from .console import print_startup_error
            logger.debug('startup_failed', exc_info=True)
            print_startup_error(error)
            return 1
    if not interactive and not args.query and not args.config and not args.search_only:
        print_terminal(json.dumps({'status': 'failed', 'error_code': 'config_required'}))
        return 1
    if not args.search_only:
        print_terminal(json.dumps({'status': 'loading_models'}))
    try:
        with SearchPipeline(args.config, load_model=not args.search_only, browser_kind=args.browser,
                            show_browser=args.show_browser) as pipeline:
            print_terminal(json.dumps({'status': 'ready', 'startup_seconds': getattr(pipeline, 'startup_seconds', None),
                              'mode': 'search_only' if args.search_only else 'normal',
                              'ranking_error': getattr(pipeline, 'ranking_error', None)}))
            if args.query:
                report = pipeline.search_report(' '.join(args.query))
                if args.snapshot:
                    save_snapshot(report, args.snapshot)
                print_report(report, organized=getattr(pipeline, '_organizing', False))
                return report_exit_code(report)
            while True:
                query = input('> ')
                if not query.strip():
                    continue
                try:
                    print_report(pipeline.search_report(query), organized=getattr(pipeline, '_organizing', False))
                except Exception as error:
                    logger.debug('request_failed', exc_info=True)
                    print_terminal(json.dumps({'error': type(error).__name__}))
    except (EOFError, KeyboardInterrupt):
        return 0
    except Exception as error:
        logger.debug('pipeline_failed', exc_info=True)
        print_terminal(json.dumps({'status': 'failed', 'error': type(error).__name__, 'error_code': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
